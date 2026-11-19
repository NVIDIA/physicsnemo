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

r"""The ``"local"`` domain-parallel mesh layout through ``MeshDataset``.

Every rank reads its share of a triangulated grid (cells share vertices
across the shard seams) and runs a recipe-style chain -- centering,
augmentations, normalization, cell normals, a no-op subsample, a centroid
interior -- on its plain local mesh; the wrapped ShardTensor sample must
equal the unsharded pipeline: bit for bit when the transforms run on the
host, and to float32 reduction-order rounding on the GPU (the reader
computes the center on the host). The loader must issue no collectives:
the profiler sees none, and rank 0 iterates alone.

Run with ``torchrun --nproc-per-node {2,4} -m pytest --multigpu-static``.
"""

import re
import threading

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.datapipes import MeshDataset
from physicsnemo.datapipes.readers.mesh import MeshReader
from physicsnemo.datapipes.transforms.mesh import (
    CenterMesh,
    ComputeSurfaceNormals,
    DropMeshFields,
    MeshToDomainMesh,
    NormalizeMeshFields,
    RandomRotateMesh,
    RandomTranslateMesh,
    SubsampleMesh,
)
from physicsnemo.distributed import DistributedManager
from physicsnemo.distributed.utils import compute_split_shapes
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.io import to_zarr

pytestmark = [pytest.mark.multigpu_static, pytest.mark.timeout(300)]

_NX, _NY = 24, 12  # 506 cells: uneven over 4 ranks
_WINDOW = 331  # odd: uneven over 2 and 4 ranks
_N_SAMPLES = 3
_SEED = 2024


def _grid_mesh(sample: int) -> Mesh:
    r"""Jittered triangulated grid, cell blocks shuffled; identical on all ranks."""
    g = torch.Generator().manual_seed(31 + sample)
    x, y = torch.meshgrid(
        torch.arange(_NX, dtype=torch.float32),
        torch.arange(_NY, dtype=torch.float32),
        indexing="ij",
    )
    z = 0.1 * torch.randn(_NX, _NY, generator=g)
    points = torch.stack([x, y, z], dim=-1).reshape(-1, 3) + 50.0
    vid = torch.arange(_NX * _NY).reshape(_NX, _NY)
    a, b = vid[:-1, :-1].reshape(-1), vid[1:, :-1].reshape(-1)
    c, d = vid[1:, 1:].reshape(-1), vid[:-1, 1:].reshape(-1)
    cells = torch.cat([torch.stack([a, b, c], 1), torch.stack([a, c, d], 1)])
    cells = cells.reshape(2, -1, 3).transpose(0, 1).reshape(-1, 2, 3)
    cells = cells[torch.randperm(cells.shape[0], generator=g)].reshape(-1, 3)
    mesh = Mesh(points=points, cells=cells)
    mesh.point_data["velocity"] = torch.randn(mesh.n_points, 3, generator=g)
    mesh.cell_data["pressure"] = torch.randn(mesh.n_cells, generator=g)
    mesh.cell_data["wss"] = torch.randn(mesh.n_cells, 3, generator=g)
    mesh.global_data["Re"] = torch.tensor(1.0e6)
    mesh.global_data["U_inf"] = torch.tensor([1.0, 0.0, 0.0])
    return mesh


@pytest.fixture(scope="module", params=["pmsh", "zarr"])
def root_and_fmt(request, tmp_path_factory, distributed_mesh):
    dm = DistributedManager()
    if dm.rank == 0:
        root = tmp_path_factory.mktemp(f"local_{request.param}")
        for i in range(_N_SAMPLES):
            if request.param == "pmsh":
                _grid_mesh(i).save(root / f"sample_{i}.pmsh")
            else:
                to_zarr(_grid_mesh(i), root / f"sample_{i}.zarr")
        holder = [str(root)]
    else:
        holder = [None]
    dist.broadcast_object_list(holder, src=0)
    return holder[0], request.param


def _chain(augment: bool, area_weighted: bool = False):
    fields = {
        "pressure": {"type": "scalar", "mean": 0.5, "std": 2.0},
        "wss": {"type": "vector", "mean": [1.0, 0.0, 0.0], "std": 0.5},
    }
    chain = [
        DropMeshFields(global_data=["Re"]),
        CenterMesh(use_area_weighting=area_weighted),
    ]
    if augment:
        chain += [
            RandomRotateMesh(
                axes=["z"],
                mode="axis_aligned",
                transform_cell_data=True,
                transform_global_data=True,
            ),
            RandomTranslateMesh(
                distribution=torch.distributions.Uniform(
                    torch.tensor([-1.0, -1.0, 0.0]),
                    torch.tensor([1.0, 1.0, 1e-6]),
                    validate_args=False,
                )
            ),
        ]
    chain += [
        NormalizeMeshFields(association="cell_data", fields=fields),
        ComputeSurfaceNormals(store_as="cell_data", field_name="normals"),
        SubsampleMesh(n_cells=_WINDOW),
        MeshToDomainMesh(
            cell_data_targets=["pressure", "wss"], interior_points="cell_centroids"
        ),
    ]
    return chain


def _dataset(root_and_fmt, device, *, chain, device_mesh=None, layout="local"):
    root, fmt = root_and_fmt
    kwargs = {}
    if device_mesh is not None:
        kwargs = dict(
            domain_parallel={"auto_shard_size": 2, "mesh_layout": layout},
            device_mesh=device_mesh,
        )
    reader = MeshReader(
        root, pattern=f"**/*.{fmt}", subsample_n_cells=_WINDOW, **kwargs
    )
    dataset = MeshDataset(reader, transforms=chain, device=device)
    generator = torch.Generator()
    generator.manual_seed(_SEED)
    dataset.set_generator(generator)
    dataset.set_epoch(3)
    return dataset


def _assert_matches(sample: DomainMesh, reference: DomainMesh, world: int, exact: bool):
    device = sample.interior.points._local_tensor.device
    reference = reference.to(device)

    def check(actual, expected):
        if exact:
            assert torch.equal(actual, expected)
        else:
            # The reader computes the center on the host; the unsharded
            # pipeline on the GPU: float32 reduction-order rounding.
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)

    # Cell axis: chunk-sharded, the unsharded rows.
    interior = sample.interior
    assert isinstance(interior.points, ShardTensor)
    assert interior.points._spec.placements == (Shard(0),)
    sizes = [s[0] for s in interior.points._spec.sharding_shapes(0)]
    assert sizes == compute_split_shapes(_WINDOW, world)
    check(interior.points.full_tensor(), reference.interior.points)
    for key in reference.interior.point_data.keys():
        check(
            interior.point_data[key].full_tensor(), reference.interior.point_data[key]
        )

    # The surface: its cells index this rank's copy of their vertices on the
    # concatenated point axis, so the routed gather resolves the unsharded
    # per-cell geometry.
    bnd, ref_bnd = sample.boundaries["vehicle"], reference.boundaries["vehicle"]
    assert isinstance(bnd.cells, ShardTensor)
    assert bnd.n_cells == ref_bnd.n_cells
    check(bnd.cell_data["normals"].full_tensor(), ref_bnd.cell_data["normals"])
    check(
        bnd.points.full_tensor()[bnd.cells.full_tensor()],
        ref_bnd.points[ref_bnd.cells],
    )
    # Recomputed on the device on both sides (the reference's cached
    # centroids were computed on the host).
    check(
        bnd.cell_centroids.full_tensor(),
        ref_bnd.points[ref_bnd.cells].mean(dim=1),
    )
    assert "Re" not in sample.global_data.keys()
    check(sample.global_data["U_inf"], reference.global_data["U_inf"])


@pytest.mark.parametrize("augment", [False, True])
@pytest.mark.parametrize("on_device", [False, True])
def test_local_layout_matches_unsharded(
    root_and_fmt, distributed_mesh, augment, on_device
):
    r"""Sharded local-layout samples equal the unsharded pipeline's."""
    dm = DistributedManager()
    device = dm.device if on_device else None
    sharded = _dataset(
        root_and_fmt, device, chain=_chain(augment), device_mesh=distributed_mesh
    )
    reference = _dataset(root_and_fmt, device, chain=_chain(augment))
    try:
        for i in range(_N_SAMPLES):
            payload = sharded._load_host(i)
            assert payload.error is None
            sample, _ = sharded._consume(payload)
            expected, _ = reference[i]
            _assert_matches(
                sample, expected, distributed_mesh.size(0), exact=not on_device
            )
    finally:
        sharded.close()
        reference.close()


def test_local_layout_area_weighted_center(root_and_fmt, distributed_mesh):
    sharded = _dataset(
        root_and_fmt, None, chain=_chain(False, True), device_mesh=distributed_mesh
    )
    reference = _dataset(root_and_fmt, None, chain=_chain(False, True))
    try:
        sample, _ = sharded[1]
        expected, _ = reference[1]
        _assert_matches(sample, expected, distributed_mesh.size(0), exact=True)
    finally:
        sharded.close()
        reference.close()


# Communication as the profiler records it: c10d / NCCL ops, functional
# collectives and ShardTensor's routed gather.
_COLLECTIVE = re.compile(
    r"nccl|gloo|c10d::|_c10d_functional|record_param_comms|routed_gather|"
    r"exchange_requests|all_?reduce|all_?gather|all_?to_?all|reduce_scatter",
    re.IGNORECASE,
)


def _collective_events(dataset, indices) -> list[str]:
    r"""Profiler event names of the loader that look like communication."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        for i in indices:
            dataset._consume(dataset._load_host(i))
    return sorted({e.name for e in prof.events() if _COLLECTIVE.search(e.name)})


def test_local_layout_issues_no_collectives(root_and_fmt, distributed_mesh):
    r"""Read, transforms and wrap run without communication; the global
    layout on the same chain is the positive control of the detector."""
    dm = DistributedManager()
    local = _dataset(
        root_and_fmt, dm.device, chain=_chain(True), device_mesh=distributed_mesh
    )
    global_ = _dataset(
        root_and_fmt,
        dm.device,
        chain=_chain(True),
        device_mesh=distributed_mesh,
        layout="global",
    )
    try:
        assert _collective_events(local, range(_N_SAMPLES)) == []
        assert _collective_events(global_, [0]) != []
    finally:
        local.close()
        global_.close()


def test_rank_zero_iterates_alone(root_and_fmt, distributed_mesh):
    r"""Rank 0 loads every sample while the other ranks wait at a barrier: a
    collective inside the loader would never complete."""
    dm = DistributedManager()
    dist.barrier()
    if dm.rank == 0:
        dataset = _dataset(
            root_and_fmt, dm.device, chain=_chain(True), device_mesh=distributed_mesh
        )
        result: list = []

        def work():
            try:
                for i in range(_N_SAMPLES):
                    sample, _ = dataset._consume(dataset._load_host(i))
                    sample.interior.points._local_tensor.sum().item()
                result.append("done")
            except Exception as e:  # noqa: BLE001
                result.append(e)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        thread.join(timeout=120)
        dataset.close()
        assert result == ["done"], result or "rank 0 blocked alone (a collective?)"
    dist.barrier()


def test_local_layout_rejects_unsupported_chains(root_and_fmt, distributed_mesh):
    dm = DistributedManager()
    for chain, error, match in [
        ([SubsampleMesh(n_cells=_WINDOW - 7)], NotImplementedError, "subsample"),
        ([ComputeSurfaceNormals(store_as="point_data")], NotImplementedError, "seam"),
        (
            [RandomTranslateMesh(), CenterMesh(use_area_weighting=False)],
            ValueError,
            "RandomTranslateMesh ran before it",
        ),
    ]:
        dataset = _dataset(
            root_and_fmt, dm.device, chain=chain, device_mesh=distributed_mesh
        )
        try:
            with pytest.raises(error, match=match):
                dataset[0]
        finally:
            dataset.close()
