# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tutorial 5: Iterable datasets for online simulation.

Most datasets are *map-style*: a fixed number of samples addressed by
index, read from storage. Some workloads instead *generate* data on the
fly -- an online physics simulation, a procedural sampler, a streaming
source with no fixed length. These are *iterable* datasets.

PhysicsNeMo models this with :class:`IterableDatasetBase`. Unlike a
map-style dataset, an iterable dataset:

- has no length and no indexing -- it only supports iteration;
- is driven entirely on the **main thread** (no worker pool), so it may
  freely launch Warp kernels and use CUDA streams. This is exactly the
  property that makes an online GPU simulation safe here: Warp's
  constraint is a single launching thread, which the main thread
  satisfies.

This tutorial wraps the built-in Warp ``Darcy2D`` flow generator -- which
solves the 2D Darcy equation with a multigrid Jacobi solver and yields a
ready-made batch each step -- as an iterable dataset and drives it through
the PhysicsNeMo :class:`DataLoader`.

Run with::

    python tutorial_5_iterable_online_simulation.py

Requires a CUDA device (the Darcy solver runs Warp kernels on the GPU).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from physicsnemo.datapipes import DataLoader, IterableDatasetBase
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D


class DarcyOnlineDataset(IterableDatasetBase):
    """Online 2D Darcy-flow simulation as an iterable dataset.

    Wraps :class:`~physicsnemo.datapipes.benchmarks.darcy.Darcy2D`, whose
    iterator runs the solver and yields a full ``{"permeability", "darcy"}``
    batch per step. The underlying generator is infinite, so this wrapper
    caps it at ``num_batches`` per epoch to give the loader a finite stream.

    Because ``Darcy2D`` already produces a complete batch, this is a
    *self-batching* dataset: we set :attr:`yields_batches` so the loader
    passes each batch through unchanged instead of re-collating.

    Parameters
    ----------
    num_batches : int
        Number of batches to emit per epoch.
    resolution : int, default=64
        Simulation grid resolution.
    batch_size : int, default=8
        Number of simulations per batch.
    device : str, default="cuda"
        Device the Warp solver runs on.
    base_seed : int, default=0
        Base seed for reproducible permeability sampling.
    """

    # Darcy2D emits a full batch per step; do not re-collate.
    yields_batches = True

    def __init__(
        self,
        num_batches: int,
        *,
        resolution: int = 64,
        batch_size: int = 8,
        device: str = "cuda",
        base_seed: int = 0,
    ) -> None:
        self._sim = Darcy2D(
            resolution=resolution,
            batch_size=batch_size,
            device=device,
            normaliser={"permeability": (1.25, 0.75), "darcy": (4.52e-2, 2.79e-2)},
        )
        self._num_batches = num_batches
        self._base_seed = base_seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch so each epoch draws a distinct, reproducible stream."""
        self._epoch = epoch

    def __iter__(self):
        # One solver iterator drives the simulation; we pull a bounded
        # number of steps from the otherwise-infinite generator.
        sim_iter = iter(self._sim)
        for position in range(self._num_batches):
            # Per-(epoch, position) seeding: the stream is reproducible
            # across runs and distinct across epochs and positions. There is
            # no stable sample index for a generator, so we key on the
            # monotonic emission position instead.
            seed = np.random.SeedSequence(
                [self._base_seed, self._epoch, position]
            ).generate_state(1)[0]
            np.random.seed(int(seed))
            yield next(sim_iter)


def main() -> None:
    """Run the online Darcy simulation over several epochs and report timings.

    Builds an iterable :class:`DarcyOnlineDataset`, wraps it in a stream-overlapped
    ``DataLoader``, and iterates for ``num_epochs``. Each epoch is reseeded via
    ``set_epoch`` for a distinct, reproducible batch stream. For every batch we
    record host wall-clock time and per-step CUDA event timings, then print a
    per-epoch summary. Requires a CUDA device for the Warp Darcy solver; the
    function returns early with a message if none is available.
    """
    if not torch.cuda.is_available():
        print("This tutorial requires a CUDA device (Warp Darcy solver). Skipping.")
        return

    num_epochs = 5
    num_batches = 16
    dataset = DarcyOnlineDataset(num_batches=num_batches, resolution=64, batch_size=8)

    # use_streams=True runs each simulation step on a preprocessing stream
    # and hands the result to the compute stream via a CUDA event, so
    # generation of the next batch can overlap training on the current one.
    loader = DataLoader(dataset, use_streams=True, seed=0)

    # Iterable datasets have no length: this will take the exception path.
    try:
        len(loader)
    except TypeError as exc:
        print(f"len(loader) is undefined for iterable datasets: {exc}")

    for epoch in range(num_epochs):
        loader.set_epoch(epoch)
        print(f"\nEpoch {epoch}")
        host_times = []
        cuda_events = []
        epoch_start = time.perf_counter()
        prev_host = epoch_start
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_start.record(torch.cuda.current_stream())
        for i, batch in enumerate(loader):
            host_now = time.perf_counter()
            permeability = batch["permeability"]
            darcy = batch["darcy"]
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_end.record(torch.cuda.current_stream())
            cuda_events.append((cuda_start, cuda_end))
            cuda_start = cuda_end

            host_times.append(host_now - prev_host)
            prev_host = host_now
            print(
                f"  batch {i}: permeability {tuple(permeability.shape)} "
                f"on {permeability.device}, darcy {tuple(darcy.shape)}, "
                f"host_dt={host_times[-1]:.4f}s"
            )

        torch.cuda.synchronize()
        cuda_times_ms = [start.elapsed_time(end) for start, end in cuda_events]
        epoch_wall = time.perf_counter() - epoch_start
        mean_host = sum(host_times) / len(host_times)
        mean_cuda = sum(cuda_times_ms) / len(cuda_times_ms)
        print(
            f"  epoch summary: batches={len(host_times)}, wall={epoch_wall:.3f}s, "
            f"host_mean={mean_host:.4f}s, cuda_mean={mean_cuda:.2f}ms, "
            f"cuda_min={min(cuda_times_ms):.2f}ms, cuda_max={max(cuda_times_ms):.2f}ms"
        )

    # Train as usual; the batches are ordinary device tensors.


if __name__ == "__main__":
    main()
