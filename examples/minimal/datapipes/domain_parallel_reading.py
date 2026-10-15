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

r"""Minimal domain-parallel (rank-local) reading example.

Each rank reads only its chunk of the large arrays straight from disk; the
sample arrives as ``Shard(0)`` ShardTensors over a device mesh you construct
in Python and inject into the reader. Small arrays replicate automatically.

Run with::

    torchrun --nproc-per-node 2 examples/minimal/datapipes/domain_parallel_reading.py

The pattern to take away for recipes:

1. The ``domain_parallel`` dict is plain, Hydra-friendly configuration.
2. The ``DeviceMesh`` is NOT configuration -- construct it at runtime from
   ``DistributedManager`` and pass it to the reader alongside the dict.
3. The dataset needs no domain-parallel arguments at all; readers own the
   rank-local read, datasets assemble the ShardTensors on the GPU.
4. Every rank of the domain mesh must ask for the same sample index (here
   all ranks read ``dataset[0]``); with a data-parallel axis, shard the
   sampler over that axis only. Subsampling needs a seed so every rank
   draws the same window.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import zarr

from physicsnemo.datapipes import Dataset
from physicsnemo.datapipes.readers.zarr import ZarrReader
from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor


def generate_sample_data(root: Path, n_samples: int = 4, n_points: int = 100_000):
    r"""Write example zarr groups: two large point-wise arrays, one small one.

    The size split is deliberate: ``coords`` and ``fields`` share a batch
    axis long enough to pass the example's ``auto_shard_size`` gate and
    will be sharded across the domain mesh, while ``params`` is short and
    will replicate.

    Parameters
    ----------
    root : Path
        Directory to write the ``sample_<i>.zarr`` groups into.
    n_samples : int, default=4
        Number of zarr groups (samples) to create.
    n_points : int, default=100_000
        Number of rows in the large point-wise arrays.
    """
    rng = np.random.default_rng(0)
    for i in range(n_samples):
        group = zarr.open_group(str(root / f"sample_{i}.zarr"), mode="w")
        group["coords"] = rng.standard_normal((n_points, 3), dtype=np.float32)
        group["fields"] = rng.standard_normal((n_points, 4), dtype=np.float32)
        group["params"] = rng.standard_normal((8,), dtype=np.float32)


def main():
    r"""Run the domain-parallel reading example end to end.

    Initializes distributed, builds a 1-D device mesh, generates example
    data on rank 0, constructs a ``ZarrReader`` with a declarative
    ``domain_parallel`` policy plus the runtime-injected mesh, and prints
    each key's global shape, rank-local shape, and placement to show which
    arrays were sharded versus replicated.
    """
    DistributedManager.initialize()
    dm = DistributedManager()

    # The device mesh is a runtime object: build it here, in Python, and
    # inject it into the reader. It never appears in yaml/Hydra config.
    device_mesh = dm.initialize_mesh([-1], ["domain"])

    # Rank 0 generates example data in a shared location.
    if dm.rank == 0:
        root = Path(tempfile.mkdtemp(prefix="dp_datapipe_example_"))
        generate_sample_data(root)
        holder = [str(root)]
    else:
        holder = [None]
    dist.broadcast_object_list(holder, src=0)
    data_root = holder[0]

    reader = ZarrReader(
        data_root,
        # Optional: coordinated subsampling composes with domain-parallel
        # reading -- each rank reads its chunk OF the subsampled window.
        coordinated_subsampling={
            "n_points": 50_000,
            "target_keys": ["coords", "fields"],
        },
        # Declarative policy (Hydra-friendly): a batch axis shards when its
        # length (tensor dim 0) is at least this many entries, decided from
        # store metadata before any data is read. ``placements`` could pin
        # axes explicitly.
        domain_parallel={"auto_shard_size": 1024},
        # Runtime object, injected in Python.
        device_mesh=device_mesh,
    )

    dataset = Dataset(reader, device=dm.device)
    # Subsampling draws a window per sample; the seed makes it identical on
    # every rank (the DataLoader does this for you when given a seed).
    generator = torch.Generator()
    generator.manual_seed(0)
    dataset.set_generator(generator)

    sample, metadata = dataset[0]
    lines = []
    for key, value in sample.items():
        if isinstance(value, ShardTensor):
            local = value.to_local()
            placement = value.placements
        else:
            local, placement = value, "(replicated)"
        lines.append(
            f"  rank {dm.rank}: {key}: global {tuple(value.shape)}, "
            f"local {tuple(local.shape)}, {placement}"
        )
    # Print rank by rank so the report reads cleanly.
    for rank in range(dm.world_size):
        if rank == dm.rank:
            if rank == 0:
                print(f"sample from {metadata['source_filename']}:")
            print("\n".join(lines), flush=True)
        dist.barrier()

    dataset.close()
    dist.barrier()
    if dm.rank == 0:
        shutil.rmtree(data_root, ignore_errors=True)
    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
