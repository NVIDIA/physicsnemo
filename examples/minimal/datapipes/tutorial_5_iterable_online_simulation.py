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
source with no fixed length. These are *iterable* datasets, modeled by
:class:`IterableDatasetBase`. Unlike a map-style dataset, an iterable
dataset has no length and no indexing -- it only supports iteration --
and it is driven entirely on the **main thread** (no worker pool), so its
generator may freely launch device kernels.

An iterable dataset emits in one of two modes:

- **Per-sample** (default, this tutorial): ``__iter__`` yields
  ``(TensorDict, metadata_dict)`` one sample at a time, and the loader
  collates ``batch_size`` of them into a batched ``TensorDict`` (a
  trailing partial batch is kept unless ``drop_last=True``).
- **Self-batching** (``yields_batches = True``): ``__iter__`` yields
  ready-made batches and the loader passes them through unchanged.

With ``use_streams=True``, the loader does four things with each
per-sample item so that generating the *next* sample can overlap training
on the *current* batch:

1. runs your generator step inside a *preprocessing-stream* binding, so
   every torch op you launch lands on a side stream;
2. records a CUDA event on that stream and makes the consumer's stream
   wait on it -- a kernel-level handoff, never a host sync;
3. calls ``record_stream`` on the yielded tensors so the caching
   allocator cannot recycle their memory while consumer reads are pending;
4. stacks ``batch_size`` samples into the batched ``TensorDict`` on the
   consumer's stream and yields it (with ``collate_metadata=True``, as a
   ``(batch, list_of_metadata)`` pair).

(All of this is handled for you, by the loader!)


The simulation here is 2D electrostatics: scatter a few random point
charges into a grounded box and solve the Poisson equation
``lap(phi) = -rho`` for the electric potential with Jacobi relaxation --
an embarrassingly GPU-friendly stencil iteration. Each sample is a
``{charge_density, potential}`` pair, the classic operator-learning
setup. The script ends with a timing comparison: the same epochs run
with streams + prefetch enabled and then fully synchronously, with
matching checksums showing the overlapped run produced identical data.

Note: you wouldn't really solve laplaces equation like this.  Don't
go off and use this solver in production please, this is just to demonstrate
the datapipe!

Run with::

    python tutorial_5_iterable_online_simulation.py

Requires a CUDA device (the overlap being demonstrated is CUDA stream
overlap).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from physicsnemo.datapipes import DataLoader, IterableDatasetBase

# GPU time the proxy training step burns per batch. Overlap is visible
# when this is comparable to the cost of generating one batch of samples.
TRAIN_STEP_SECONDS = 0.3


class ElectrostaticsDataset(IterableDatasetBase):
    """Online 2D electrostatics simulation as a per-sample iterable dataset.

    Each sample scatters ``num_charges`` random point charges (Gaussian
    splats) into a grounded 2D box and solves the Poisson equation
    ``lap(phi) = -rho`` for the potential ``phi`` with a fixed budget of
    Jacobi relaxation sweeps. The zero-padded stencil update enforces the
    Dirichlet ``phi = 0`` boundary.

    Parameters
    ----------
    samples_per_epoch : int
        Number of samples to emit per epoch.
    resolution : int, default=128
        Grid resolution (the domain is the unit square).
    num_charges : int, default=4
        Point charges scattered per sample.
    jacobi_iters : int, default=512
        Fixed number of Jacobi sweeps per sample.
    device : str, default="cuda"
        Device the solver runs on.
    base_seed : int, default=0
        Base seed for reproducible charge placement.
    """

    def __init__(
        self,
        samples_per_epoch: int,
        *,
        resolution: int = 128,
        num_charges: int = 4,
        jacobi_iters: int = 512,
        device: str = "cuda",
        base_seed: int = 0,
    ) -> None:
        self._samples_per_epoch = samples_per_epoch
        self.resolution = resolution
        self.num_charges = num_charges
        self.jacobi_iters = jacobi_iters
        self.device = torch.device(device)
        self._base_seed = base_seed
        self._epoch = 0

        # Grid geometry, built once: cell coordinates in the unit square
        # and the h^2/4 factor of the Jacobi update.
        h = 1.0 / (resolution + 1)
        coords = torch.linspace(h, 1.0 - h, resolution, device=self.device)
        self._yy, self._xx = torch.meshgrid(coords, coords, indexing="ij")
        self._quarter_h2 = 0.25 * h * h
        # Splat width: a couple of grid cells.
        self._sigma2 = (2.5 * h) ** 2
        # Jacobi update as a 5-point stencil convolution (neighbor average).
        self._stencil = torch.tensor(
            [[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]],
            device=self.device,
        ).reshape(1, 1, 3, 3)

        # Capture the solve loop into a CUDA graph so each sample costs
        # one launch instead of jacobi_iters of them.
        if self.device.type == "cuda":
            self._build_solver_graph()
        else:
            self._graph = None

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch so each epoch draws a distinct, reproducible stream."""
        self._epoch = epoch

    def _solve(self, source: torch.Tensor) -> torch.Tensor:
        """Jacobi-relax ``lap(phi) = -rho`` given ``source = h^2/4 * rho``.

        This eager loop is the reference implementation (and the CPU
        path): one stencil convolution per sweep, whose zero padding IS
        the grounded ``phi = 0`` boundary.
        """
        phi = torch.zeros_like(source)
        for _ in range(self.jacobi_iters):
            phi = F.conv2d(phi, self._stencil, padding=1) + source
        return phi

    def _build_solver_graph(self) -> None:
        """Capture the fixed-budget solve into a replayable CUDA graph.

        Follows the standard capture recipe: warm the ops up on a side
        stream (cuDNN plan selection must happen outside capture), then
        record the sweep loop once. ``replay()`` then launches the whole
        solve on the current stream as a single kernel launch, so
        generation is no longer bounded by the host's launch rate.
        ``_static_source`` and ``_static_phi`` are the graph's fixed
        input/output addresses: each sample copies its source in,
        replays, and clones the result out.
        """
        self._static_source = torch.zeros(
            1, 1, self.resolution, self.resolution, device=self.device
        )
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            phi = torch.zeros_like(self._static_source)
            for _ in range(8):
                phi = F.conv2d(phi, self._stencil, padding=1) + self._static_source
        torch.cuda.current_stream().wait_stream(warmup_stream)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_phi = self._solve(self._static_source)

    def _generate_sample(
        self, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one charge configuration and solve for its potential."""
        # Random charges: positions in the interior, signed magnitudes.
        pos = 0.1 + 0.8 * torch.rand(
            self.num_charges, 2, generator=generator, device=self.device
        )
        magnitude = torch.randn(
            self.num_charges, generator=generator, device=self.device
        )

        # Charge density: sum of Gaussian splats on the grid.
        d2 = (self._xx - pos[:, 0, None, None]) ** 2 + (
            self._yy - pos[:, 1, None, None]
        ) ** 2
        rho = (magnitude[:, None, None] * torch.exp(-d2 / (2 * self._sigma2))).sum(0)

        # Solve lap(phi) = -rho. The fixed sweep budget keeps the loop
        # free of host syncs (rule 3; a convergence check would read the
        # residual back every test) -- and makes it graph-replayable. The
        # graph's output buffer is static (the next replay overwrites
        # it), so clone the result out: the clone is the freshly
        # allocated tensor rule 2 requires.
        source = (self._quarter_h2 * rho)[None, None]
        if self._graph is not None:
            self._static_source.copy_(source)
            self._graph.replay()
            phi = self._static_phi.clone()
        else:
            phi = self._solve(source)
        return rho[None], phi[0]

    def __iter__(self):
        for position in range(self._samples_per_epoch):
            # Per-(epoch, position) seeding: reproducible across runs,
            # distinct across epochs and positions. A generator has no
            # stable sample index, so the monotonic emission position is
            # the key.
            seed = int(
                np.random.SeedSequence(
                    [self._base_seed, self._epoch, position]
                ).generate_state(1)[0]
            )
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

            rho, phi = self._generate_sample(generator)

            # One sample: a TensorDict of fresh [1, res, res] tensors (the
            # collator stacks these into [batch, 1, res, res]) plus a
            # metadata dict the loader collects into a list.
            data = TensorDict(
                {"charge_density": rho, "potential": phi},
                batch_size=[],
            )
            yield data, {"position": position, "seed": seed}


def train_step(
    batch: TensorDict, weight: torch.Tensor, spin_cycles: int
) -> torch.Tensor:
    """GPU-bound proxy for a model training step.

    One small matmul ties the loss to the batch data; a low-occupancy
    spin kernel (``torch.cuda._sleep``, a single-block busy-wait) stands
    in for the bulk of a model's GPU time. Both are cheap for the host to
    enqueue, and the loss stays on device (no ``.item()``), so the host
    returns to the loader immediately and the compute queue stays deep --
    the conditions under which next-batch generation overlaps this work.

    A spin kernel rather than a stack of large matmuls keeps the demo
    about *pipeline* overlap: SM-saturating kernels would compete with
    the solver's kernels for GPU resources, which is a device-specific
    story. A real model sits between these extremes.
    """
    x = batch["potential"].flatten(1)
    loss = torch.tanh(x @ weight).square().mean()
    torch.cuda._sleep(spin_cycles)
    return loss


def run_epochs(
    loader: DataLoader, weight: torch.Tensor, spin_cycles: int, epochs, label: str
) -> float:
    """Run the given epochs through ``loader`` and report wall time.

    Accumulates a device-side data checksum and mean proxy loss so the
    batches are visibly consumed. Identical seeding across two calls must
    produce identical checksums -- the correctness check for the
    overlapped path.
    """
    checksum = torch.zeros((), device=weight.device)
    total_loss = torch.zeros((), device=weight.device)
    num_batches = 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in epochs:
        loader.set_epoch(epoch)
        for batch, _metadata in loader:
            checksum += batch["potential"].sum()
            total_loss += train_step(batch, weight, spin_cycles)
            num_batches += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    print(
        f"  {label}: {wall:.3f}s over {num_batches} batches "
        f"(checksum {checksum.item():+.6f}, "
        f"mean loss {(total_loss / num_batches).item():.6f})"
    )
    return wall


def main() -> None:
    """Drive the online electrostatics dataset and demonstrate overlap.

    Builds a per-sample :class:`ElectrostaticsDataset`, wraps it in
    a stream-overlapped ``DataLoader`` that collates 8 samples per batch,
    and runs the same epochs twice: with streams + prefetch enabled, then
    fully synchronously (``disable_prefetch``). Identical seeding makes
    the two runs generate identical data -- the printed checksums must
    match -- while the wall times show the overlap win.
    """
    if not torch.cuda.is_available():
        print("This tutorial requires a CUDA device. Skipping.")
        return

    resolution = 128
    dataset = ElectrostaticsDataset(
        samples_per_epoch=20, resolution=resolution, jacobi_iters=4096
    )

    # Per-sample mode: the loader collates batch_size samples per batch.
    # 20 samples with batch_size=8 leaves a trailing partial batch of 4
    # (kept because drop_last defaults to False). collate_metadata=True
    # yields (batch, list_of_metadata) pairs.
    loader = DataLoader(
        dataset,
        batch_size=8,
        use_streams=True,
        seed=0,
        num_streams=1,
        collate_metadata=True,
    )

    # Iterable datasets have no length: this takes the exception path.
    try:
        len(loader)
    except TypeError as exc:
        print(f"len(loader) is undefined for iterable datasets: {exc}")

    torch.manual_seed(0)
    weight = torch.randn(resolution * resolution, 256, device="cuda")

    # Calibrate the proxy step's spin kernel to TRAIN_STEP_SECONDS of GPU
    # time on this device.
    probe = 10_000_000
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda._sleep(probe)
    torch.cuda.synchronize()
    spin_cycles = int(TRAIN_STEP_SECONDS * probe / (time.perf_counter() - start))

    # Warmup epoch: pays allocator warmup and kernel autotuning so the
    # timed comparison is fair -- and shows the collated batch structure,
    # including the trailing partial batch.
    print("\nWarmup epoch (excluded from timing):")
    loader.set_epoch(999)
    for i, (batch, metadata) in enumerate(loader):
        positions = [m["position"] for m in metadata]
        print(
            f"  batch {i}: potential {tuple(batch['potential'].shape)}, "
            f"sample positions {positions}"
        )

    measured_epochs = range(2)
    print("\nOverlapped run (streams + prefetch ON):")
    wall_overlap = run_epochs(
        loader, weight, spin_cycles, epochs=measured_epochs, label="ON "
    )

    loader.disable_prefetch()
    print("Synchronous run (disable_prefetch):")
    wall_sync = run_epochs(
        loader, weight, spin_cycles, epochs=measured_epochs, label="OFF"
    )

    # Same epochs, same seeds: the checksums above must match (the
    # overlapped run produced identical data) while the wall times differ
    # (generation of the next samples overlapped training on the current
    # batch).
    print(f"\noverlap speedup: {wall_sync / wall_overlap:.2f}x")


if __name__ == "__main__":
    main()
