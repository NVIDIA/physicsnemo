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

r"""Per-sample loader cost of domain-parallel ``MeshReader`` layouts.

Not an ASV benchmark: it needs a multi-process launch. For each layout
(``global``: points and cells sharded independently, transforms on
ShardTensors; ``local``: each rank reads its cells with their vertices and
transforms plain tensors) it times, per sample and rank,

- ``reader``: ``MeshDataset._load_host`` -- the disk read, which the
  ``DataLoader`` runs on its worker threads;
- ``main``: ``MeshDataset._consume`` -- host-to-device copy, transforms and
  ShardTensor assembly on the consuming (training) thread, synchronized.

and reports the median and max over samples of the slowest rank. The chain
is the unified external-aero recipe's surface chain (centering, field
normalization, cell normals, a no-op subsample, centroid interior). Each
layout reads its own samples, so both see cold page caches on a fresh node.

Usage (on a GPU node)::

    torchrun --nproc-per-node 2 benchmarks/physicsnemo/datapipes/domain_parallel_mesh_reader.py \
        --data /path/to/PhysicsNeMo-DrivaerML --n-cells 200000 --samples 20

    # without data: a synthetic triangulated surface
    torchrun --nproc-per-node 2 benchmarks/physicsnemo/datapipes/domain_parallel_mesh_reader.py \
        --synthetic 2000000 --n-cells 200000

``--read-mode memmap`` disables the sequential reads of the local layout
(memmap indexing instead), to isolate their effect.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist

DRIVAERML_PATTERN = "run_*/*.pdmsh/_tensordict/boundaries/vehicle"


def _synthetic_surface(n_cells: int, root: Path, n_samples: int) -> str:
    """Write *n_samples* triangulated, jittered grids of about *n_cells* cells."""
    from physicsnemo.mesh import Mesh

    side = int((n_cells / 2) ** 0.5) + 1
    for s in range(n_samples):
        g = torch.Generator().manual_seed(s)
        x, y = torch.meshgrid(
            torch.arange(side, dtype=torch.float32),
            torch.arange(side, dtype=torch.float32),
            indexing="ij",
        )
        z = 0.1 * torch.randn(side, side, generator=g)
        points = torch.stack([x, y, z], -1).reshape(-1, 3)
        vid = torch.arange(side * side).reshape(side, side)
        a, b = vid[:-1, :-1].reshape(-1), vid[1:, :-1].reshape(-1)
        c, d = vid[1:, 1:].reshape(-1), vid[:-1, 1:].reshape(-1)
        cells = torch.stack(
            [torch.stack([a, b, c], 1), torch.stack([a, c, d], 1)], 1
        ).reshape(-1, 3)
        mesh = Mesh(points=points, cells=cells)
        mesh.cell_data["pMeanTrim"] = torch.randn(mesh.n_cells, generator=g)
        mesh.cell_data["wallShearStressMeanTrim"] = torch.randn(
            mesh.n_cells, 3, generator=g
        )
        mesh.global_data["TimeValue"] = torch.tensor(0.0)
        mesh.save(root / f"sample_{s}.pmsh")
    return "*.pmsh"


def _chain(n_cells: int):
    from physicsnemo.datapipes.transforms.mesh import (
        CenterMesh,
        ComputeSurfaceNormals,
        DropMeshFields,
        MeshToDomainMesh,
        NormalizeMeshFields,
        SubsampleMesh,
    )

    return [
        DropMeshFields(global_data=["TimeValue"]),
        CenterMesh(use_area_weighting=False),
        NormalizeMeshFields(
            association="cell_data",
            fields={
                "wallShearStressMeanTrim": {
                    "type": "vector",
                    "mean": [0.0, 0.0, 0.0],
                    "std": 0.00313,
                }
            },
        ),
        ComputeSurfaceNormals(store_as="cell_data", field_name="normals"),
        SubsampleMesh(n_cells=n_cells),
        MeshToDomainMesh(
            cell_data_targets=["pMeanTrim", "wallShearStressMeanTrim"],
            interior_points="cell_centroids",
        ),
    ]


def _time_layout(args, root, pattern, layout, indices, device_mesh, device):
    from physicsnemo.datapipes import MeshDataset
    from physicsnemo.datapipes.readers.mesh import MeshReader

    reader = MeshReader(
        root,
        pattern=pattern,
        subsample_n_cells=args.n_cells,
        domain_parallel={"auto_shard_size": 1024, "mesh_layout": layout},
        device_mesh=device_mesh,
        pin_memory=True,
    )
    dataset = MeshDataset(reader, transforms=_chain(args.n_cells), device=device)
    generator = torch.Generator()
    generator.manual_seed(0)
    dataset.set_generator(generator)
    times = {"reader": [], "main": []}
    for k, index in enumerate(indices):
        dist.barrier()
        t0 = time.perf_counter()
        payload = dataset._load_host(index)
        t1 = time.perf_counter()
        if payload.error is not None:
            raise payload.error
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        sample, _ = dataset._consume(payload)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        del sample
        if k >= args.warmup:
            times["reader"].append(1e3 * (t1 - t0))
            times["main"].append(1e3 * (t3 - t2))
    dataset.close()
    # Slowest rank per sample.
    out = {}
    for key, values in times.items():
        t = torch.tensor(values, device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        values = t.tolist()
        out[key] = (statistics.median(values), max(values))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="directory of DrivAerML runs")
    source.add_argument("--synthetic", type=int, help="cells of a synthetic surface")
    parser.add_argument("--n-cells", type=int, default=200_000, help="cell window")
    parser.add_argument("--samples", type=int, default=20, help="timed samples")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--layouts", default="global,local")
    parser.add_argument(
        "--read-mode", choices=["sequential", "memmap"], default="sequential"
    )
    args = parser.parse_args()

    from physicsnemo.distributed import DistributedManager

    DistributedManager.initialize()
    dm = DistributedManager()
    device_mesh = dm.initialize_mesh([-1], ["domain"])
    torch.cuda.set_device(dm.device)

    if args.read_mode == "memmap":
        import physicsnemo.datapipes._local_shard as local_shard

        local_shard._backing_file = lambda t: None

    tmp = None
    if args.synthetic:
        n_samples = (args.samples + args.warmup) * len(args.layouts.split(","))
        holder = [None]
        if dm.rank == 0:
            tmp = tempfile.TemporaryDirectory()
            _synthetic_surface(args.synthetic, Path(tmp.name), n_samples)
            holder = [tmp.name]
        dist.broadcast_object_list(holder, src=0)
        root, pattern = holder[0], "*.pmsh"
    else:
        root, pattern = args.data, DRIVAERML_PATTERN

    per_layout = args.samples + args.warmup
    results = {}
    for k, layout in enumerate(args.layouts.split(",")):
        # Distinct samples per layout: each sees cold page caches.
        indices = range(k * per_layout, (k + 1) * per_layout)
        results[layout] = _time_layout(
            args, root, pattern, layout, indices, device_mesh, dm.device
        )

    if dm.rank == 0:
        print(
            f"world={dm.world_size} n_cells={args.n_cells} samples={args.samples} "
            f"source={'synthetic ' + str(args.synthetic) if args.synthetic else args.data} "
            f"read_mode={args.read_mode}"
        )
        print(
            f"{'layout':8s} {'main ms (median/max)':>22s} {'reader ms (median/max)':>24s}"
        )
        for layout, r in results.items():
            print(
                f"{layout:8s} {r['main'][0]:10.2f} / {r['main'][1]:8.2f} "
                f"{r['reader'][0]:12.2f} / {r['reader'][1]:8.2f}"
            )
    dist.barrier()
    if tmp is not None:
        tmp.cleanup()
    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
