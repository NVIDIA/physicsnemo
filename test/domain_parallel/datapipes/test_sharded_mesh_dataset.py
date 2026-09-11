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

r"""Rank-local sharded reading through ``MeshReader(device_mesh=...)``.

Rank 0 writes seeded ``.pmsh`` samples to a shared tmp dir; every rank then
reads through the datapipe with ``device_mesh`` set and must see a ``Mesh``
of ``Shard(0)`` ShardTensors whose gathered values match the full on-disk
sample -- construction, both load paths (sync and producer/consumer), and
the recipe-style transform chain against an unsharded reference.
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.datapipes import MeshDataset
from physicsnemo.datapipes.readers.mesh import MeshReader
from physicsnemo.datapipes.transforms.mesh.transforms import (
    CenterMesh,
    NormalizeMeshFields,
)
from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import to_zarr

pytestmark = [pytest.mark.multigpu_static, pytest.mark.timeout(300)]

# Uneven on 2/4/8 ranks for both batch dims (n_points = 3 * _N_CELLS).
_N_CELLS = 431
_N_SAMPLES = 2


def _build_full_mesh(sample: int) -> Mesh:
    r"""Seeded triangle soup, distinct per sample, identical on all ranks."""
    torch.manual_seed(101 + sample)
    base = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    offsets = torch.zeros(_N_CELLS, 1, 3)
    offsets[:, 0, 0] = 2.0 * torch.arange(_N_CELLS)
    points = (base.unsqueeze(0) + offsets).reshape(-1, 3)
    points = points + 0.01 * torch.randn_like(points)
    cells = torch.arange(3 * _N_CELLS, dtype=torch.int64).reshape(-1, 3)
    mesh = Mesh(points=points, cells=cells)
    mesh.point_data["velocity"] = torch.randn(mesh.n_points, 3)
    mesh.cell_data["pressure"] = torch.randn(mesh.n_cells)
    mesh.cell_data["wss"] = torch.randn(mesh.n_cells, 3)
    mesh.global_data["Re"] = torch.tensor(1.0e6)
    return mesh


@pytest.fixture(scope="module", params=["pmsh", "zarr"])
def pmsh_root(request, tmp_path_factory, distributed_mesh):
    r"""Shared directory of samples in one on-disk format (memmap ``.pmsh`` or
    zarr); rank 0 writes, path broadcast. Both formats go through the same
    rank-local read plan but different row sources."""
    fmt = request.param
    dm = DistributedManager()
    if dm.rank == 0:
        root = tmp_path_factory.mktemp(f"sharded_{fmt}")
        for i in range(_N_SAMPLES):
            mesh = _build_full_mesh(i)
            if fmt == "pmsh":
                mesh.save(root / f"sample_{i}.pmsh")
            else:
                to_zarr(mesh, root / f"sample_{i}.zarr")
        holder = [str(root), fmt]
    else:
        holder = [None, None]
    dist.broadcast_object_list(holder, src=0)
    return holder[0], holder[1]


def _make_reader(root_and_fmt, distributed_mesh, **kwargs):
    root, fmt = root_and_fmt
    kwargs.setdefault("domain_parallel", {"auto_shard_size": 2})
    kwargs.setdefault("device_mesh", distributed_mesh)
    return MeshReader(root, pattern=f"**/*.{fmt}", **kwargs)


def _make_dataset(pmsh_root, distributed_mesh, transforms=None, **reader_kwargs):
    dm = DistributedManager()
    return MeshDataset(
        _make_reader(pmsh_root, distributed_mesh, **reader_kwargs),
        transforms=transforms,
        device=dm.device,
    )


def _assert_sharded_matches_full(mesh, sample: int, distributed_mesh):
    """Plain layout: points/point_data chunk over points, cells/cell_data chunk
    over cells (cells keep global vertex ids); every gathered leaf equals the
    unsharded sample and per-cell quantities go through the routed gather."""
    full = _build_full_mesh(sample)
    world_size = distributed_mesh.size(0)

    assert mesh.n_points == full.n_points
    assert mesh.n_cells == full.n_cells
    for leaf in (mesh.points, mesh.cells):
        assert isinstance(leaf, ShardTensor)
        assert leaf._spec.placements == (Shard(0),)
    assert mesh.points._local_tensor.shape[0] <= -(-full.n_points // world_size)
    assert mesh.cells._local_tensor.shape[0] <= -(-full.n_cells // world_size)

    device = mesh.points._local_tensor.device
    torch.testing.assert_close(mesh.points.full_tensor(), full.points.to(device))
    torch.testing.assert_close(mesh.cells.full_tensor(), full.cells.to(device))
    torch.testing.assert_close(
        mesh.point_data["velocity"].full_tensor(),
        full.point_data["velocity"].to(device),
    )
    for key in ("pressure", "wss"):
        assert isinstance(mesh.cell_data[key], ShardTensor)
        torch.testing.assert_close(
            mesh.cell_data[key].full_tensor(), full.cell_data[key].to(device)
        )
    torch.testing.assert_close(
        mesh.global_data["Re"], full.global_data["Re"].to(device)
    )
    # Cell quantities: points[cells] is the routed gather across ranks.
    centroids = mesh.cell_centroids
    assert isinstance(centroids, ShardTensor)
    torch.testing.assert_close(centroids.full_tensor(), full.cell_centroids.to(device))


def test_sharded_read_sync_path(pmsh_root, distributed_mesh):
    r"""dataset[i] (synchronous _load): sharded Mesh matches the full sample."""
    dataset = _make_dataset(pmsh_root, distributed_mesh)
    try:
        for i in range(_N_SAMPLES):
            mesh, metadata = dataset[i]
            _assert_sharded_matches_full(mesh, i, distributed_mesh)
            assert metadata["index"] == i
    finally:
        dataset.close()


def test_sharded_read_producer_consumer_path(pmsh_root, distributed_mesh):
    r"""_load_host -> _consume (the prefetch stages, no stream): the slice
    happens host-side, the ShardTensor wrap after device transfer."""
    dataset = _make_dataset(pmsh_root, distributed_mesh)
    try:
        payload = dataset._load_host(0)
        assert payload.error is None
        # Host payload carries this rank's chunks only.
        local_cells = payload.data.tensors["cells"].shape[0]
        local_points = payload.data.tensors["points"].shape[0]
        assert local_cells < _N_CELLS or distributed_mesh.size(0) == 1
        assert local_points < 3 * _N_CELLS or distributed_mesh.size(0) == 1
        assert payload.data.tensors["points"].device.type == "cpu"
        assert payload.data.sharded[("points",)] == (3 * _N_CELLS, 3)
        assert payload.data.sharded[("cells",)] == (_N_CELLS, 3)

        mesh, _ = dataset._consume(payload)
        _assert_sharded_matches_full(mesh, 0, distributed_mesh)
    finally:
        dataset.close()


def test_sharded_read_with_transforms(pmsh_root, distributed_mesh):
    r"""Recipe-style transform chain on the sharded pipe matches the same
    chain applied to the full mesh: CenterMesh is the global reduction,
    NormalizeMeshFields the elementwise cell_data op."""
    fields = {
        "pressure": {"type": "scalar", "mean": 101325.0, "std": 250.0},
        "wss": {"type": "vector", "mean": [1.0, 0.0, 0.0], "std": 0.5},
    }

    def make_transforms():
        return [
            CenterMesh(use_area_weighting=False),
            NormalizeMeshFields(association="cell_data", fields=fields),
        ]

    dm = DistributedManager()
    dataset = _make_dataset(pmsh_root, distributed_mesh, transforms=make_transforms())
    try:
        mesh, _ = dataset[0]
    finally:
        dataset.close()

    reference = _build_full_mesh(0).to(dm.device)
    for t in make_transforms():
        if hasattr(t, "to"):
            t.to(dm.device)
        reference = t(reference)

    assert isinstance(mesh.points, ShardTensor)
    # 1e-4: CenterMesh's COM is a per-rank partial sum resolved by an
    # all-reduce; fp32 summation-order jitter vs the single-device
    # reference is a few 1e-5 on coordinates spanning O(1e3) units.
    torch.testing.assert_close(
        mesh.points.full_tensor(), reference.points, atol=1e-4, rtol=1e-4
    )
    for key in ("pressure", "wss"):
        torch.testing.assert_close(
            mesh.cell_data[key].full_tensor(),
            reference.cell_data[key],
            atol=1e-5,
            rtol=1e-5,
        )


def test_cell_subsample_matches_eager(pmsh_root, distributed_mesh):
    r"""``subsample_n_cells`` under domain parallelism: the whole window of
    cells is compacted identically on every rank and the gathered sharded
    mesh equals the eager (unsharded) subsample with the same seed/epoch."""
    dm = DistributedManager()
    root, fmt = pmsh_root
    n_cells = 97
    seed, epoch = 4321, 2

    def seeded(dataset):
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataset.set_generator(generator)
        dataset.set_epoch(epoch)
        return dataset

    reference_ds = seeded(
        MeshDataset(
            MeshReader(root, pattern=f"**/*.{fmt}", subsample_n_cells=n_cells),
            device=dm.device,
        )
    )
    sharded_ds = seeded(
        _make_dataset(pmsh_root, distributed_mesh, subsample_n_cells=n_cells)
    )
    try:
        reference, _ = reference_ds[1]
        mesh, _ = sharded_ds[1]
        assert mesh.n_cells == n_cells == reference.n_cells
        assert mesh.n_points == reference.n_points
        torch.testing.assert_close(mesh.cells.full_tensor(), reference.cells)
        torch.testing.assert_close(mesh.points.full_tensor(), reference.points)
        torch.testing.assert_close(
            mesh.cell_data["wss"].full_tensor(), reference.cell_data["wss"]
        )
        torch.testing.assert_close(
            mesh.point_data["velocity"].full_tensor(),
            reference.point_data["velocity"],
        )
    finally:
        reference_ds.close()
        sharded_ds.close()


def test_point_subsample_on_cells_is_rejected(pmsh_root, distributed_mesh):
    r"""Point subsampling on a mesh with cells is not supported under domain
    parallelism (it would remap connectivity globally); the reader raises
    before any collective, identically on every rank."""
    dataset = _make_dataset(pmsh_root, distributed_mesh, subsample_n_points=50)
    generator = torch.Generator()
    generator.manual_seed(0)
    dataset.set_generator(generator)
    try:
        with pytest.raises(NotImplementedError, match="subsample_n_points"):
            dataset[0]
    finally:
        dataset.close()


def test_subsample_without_seed_is_rejected(pmsh_root, distributed_mesh):
    r"""An unseeded window would differ per rank; the reader refuses up front."""
    dataset = _make_dataset(pmsh_root, distributed_mesh, subsample_n_cells=97)
    try:
        with pytest.raises(ValueError, match="requires a seed"):
            dataset[0]
    finally:
        dataset.close()
