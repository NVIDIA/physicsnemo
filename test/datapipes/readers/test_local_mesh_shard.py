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

r"""Single-process tests of the ``"local"`` domain-parallel mesh layout.

Every rank's share is read and transformed here in one process, with no
process group at all: a rank processes its share alone, so neither the read
nor the transforms can communicate. Concatenated over the ranks, the shares
must reproduce the unsharded read and transform chain bit for bit. The
ShardTensor wrap and the multi-rank pipeline are covered in
``test/domain_parallel/datapipes/test_local_mesh_shard_dataset.py``.
"""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import physicsnemo.datapipes.mesh_dataset as mesh_dataset_module
from physicsnemo.datapipes._indexing import _cyclic_block_indices
from physicsnemo.datapipes._local_shard import (
    AREA_WEIGHTED_CENTER,
    POINT_MEAN,
    LocalMeshShard,
    LocalShardInfo,
    read_rows,
)
from physicsnemo.datapipes._rng import spawn_generator
from physicsnemo.datapipes.mesh_dataset import MeshDataset
from physicsnemo.datapipes.readers.mesh import (
    _subsample_mesh_cells,
    read_local_cell_shard,
)
from physicsnemo.datapipes.transforms.mesh import (
    CenterMesh,
    ComputeSurfaceNormals,
    DropMeshFields,
    MeshToDomainMesh,
    NormalizeMeshFields,
    ScaleMesh,
    SubsampleMesh,
)
from physicsnemo.distributed.utils import compute_split_shapes
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import io_zarr, to_zarr

# 2 * 23 * 11 = 506 cells: uneven over 3 and 4 ranks; 288 shared vertices.
_NX, _NY = 24, 12
_WINDOW = 331  # odd: uneven over every world size > 1
_SEED = 1234


def _grid_mesh() -> Mesh:
    r"""Triangulated, jittered grid: cells share vertices across every chunk seam."""
    g = torch.Generator().manual_seed(7)
    x, y = torch.meshgrid(
        torch.arange(_NX, dtype=torch.float32),
        torch.arange(_NY, dtype=torch.float32),
        indexing="ij",
    )
    z = 0.1 * torch.randn(_NX, _NY, generator=g)
    points = torch.stack([x, y, z], dim=-1).reshape(-1, 3) + 100.0
    vid = torch.arange(_NX * _NY).reshape(_NX, _NY)
    a, b = vid[:-1, :-1].reshape(-1), vid[1:, :-1].reshape(-1)
    c, d = vid[1:, 1:].reshape(-1), vid[:-1, 1:].reshape(-1)
    cells = torch.cat([torch.stack([a, b, c], 1), torch.stack([a, c, d], 1)])
    # Interleave the two triangles of each quad, then shuffle blocks so a
    # chunk's vertices are not one contiguous id range.
    cells = cells.reshape(2, -1, 3).transpose(0, 1).reshape(-1, 3)
    perm = torch.randperm(cells.shape[0] // 2, generator=g)
    cells = cells.reshape(-1, 2, 3)[perm].reshape(-1, 3)
    mesh = Mesh(points=points, cells=cells)
    mesh.point_data["velocity"] = torch.randn(mesh.n_points, 3, generator=g)
    mesh.cell_data["pressure"] = torch.randn(mesh.n_cells, generator=g)
    mesh.cell_data["wss"] = torch.randn(mesh.n_cells, 3, generator=g)
    mesh.global_data["Re"] = torch.tensor(1.0e6)
    return mesh


@pytest.fixture(scope="module", params=["pmsh", "zarr"])
def stored(request, tmp_path_factory):
    r"""``(open_source, full)``: a callable opening the stored sample lazily
    (memmap ``Mesh`` or zarr group) and the in-memory sample."""
    full = _grid_mesh()
    root = tmp_path_factory.mktemp(f"local_shard_{request.param}")
    if request.param == "pmsh":
        full.save(root / "s.pmsh")
        opener = lambda: Mesh.load(root / "s.pmsh", mode="r")  # noqa: E731
    else:
        pytest.importorskip("zarr")
        to_zarr(full, root / "s.zarr")
        opener = lambda: io_zarr._open_group(root / "s.zarr")  # noqa: E731
    return opener, full


@pytest.fixture
def source(stored):
    r"""A fresh in-memory sample per test: transforms cache geometry on it
    (``cell_centroids`` ...), and a warm cache changes the rounding."""
    return stored[0], _grid_mesh()


def _window_and_reference(full: Mesh, windowed: bool):
    if not windowed:
        return None, full
    window = _cyclic_block_indices(
        full.n_cells, _WINDOW, generator=spawn_generator(_SEED, 0, 0)
    )
    reference = _subsample_mesh_cells(
        full, _WINDOW, generator=spawn_generator(_SEED, 0, 0)
    )
    return window, reference


def _shares(open_source, world, window, stats=frozenset()):
    return [
        read_local_cell_shard(
            open_source(),
            rank=r,
            world_size=world,
            cell_window=window,
            stats=stats,
        )
        for r in range(world)
    ]


def _as_payload(share, rank) -> LocalMeshShard:
    info = LocalShardInfo(
        device_mesh=None,  # never touched before the wrap
        rank=rank,
        cell_sizes=share.cell_sizes,
        point_sizes=share.point_sizes,
        n_points=share.n_points,
    )
    return LocalMeshShard(mesh=share.mesh, info=info, stats=share.stats)


@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_shares_reproduce_unsharded_read(source, world, windowed):
    open_source, full = source
    window, reference = _window_and_reference(full, windowed)
    shares = _shares(open_source, world, window, {POINT_MEAN, AREA_WEIGHTED_CENTER})

    n_cells = reference.n_cells
    assert all(
        s.cell_sizes == tuple(compute_split_shapes(n_cells, world)) for s in shares
    )
    point_sizes = shares[0].point_sizes
    for rank, share in enumerate(shares):
        mesh = share.mesh
        assert share.point_sizes == point_sizes
        assert mesh.n_cells == share.cell_sizes[rank]
        assert mesh.n_points == point_sizes[rank]
        # Cells index exactly this rank's points.
        assert int(mesh.cells.max()) == mesh.n_points - 1
        assert torch.unique(mesh.cells).numel() == mesh.n_points
        assert share.n_points == reference.n_points
    if world > 1:  # seam vertices are held by both neighbours
        assert sum(point_sizes) > reference.n_points

    # Per-cell geometry, point data seen through the cells, cell data and
    # the window's measure correction: exactly the unsharded rows.
    cat = torch.cat
    assert torch.equal(
        cat([s.mesh.points[s.mesh.cells] for s in shares]),
        reference.points[reference.cells],
    )
    assert torch.equal(
        cat([s.mesh.point_data["velocity"][s.mesh.cells] for s in shares]),
        reference.point_data["velocity"][reference.cells],
    )
    assert set(shares[0].mesh.cell_data.keys()) == set(reference.cell_data.keys())
    for key in reference.cell_data.keys():
        assert torch.equal(
            cat([s.mesh.cell_data[key] for s in shares]), reference.cell_data[key]
        )
    assert torch.equal(shares[0].mesh.global_data["Re"], reference.global_data["Re"])

    # Whole-sample statistics: CenterMesh's own computation on the unsharded
    # sample, identical on every rank.
    for share in shares:
        assert torch.equal(
            share.stats[POINT_MEAN],
            CenterMesh(use_area_weighting=False)._compute_com(reference),
        )
        assert torch.equal(
            share.stats[AREA_WEIGHTED_CENTER],
            CenterMesh(use_area_weighting=True)._compute_com(reference),
        )


def test_statistics_only_on_request(source):
    open_source, full = source
    window, _ = _window_and_reference(full, True)
    (share,) = _shares(open_source, 1, window)
    assert len(share.stats.keys()) == 0
    with pytest.raises(ValueError, match="unknown shard statistics"):
        _shares(open_source, 1, window, {"median"})


def test_cached_point_sizes_read_only_own_cells(source):
    open_source, full = source
    first = _shares(open_source, 3, None)
    again = [
        read_local_cell_shard(
            open_source(), rank=r, world_size=3, point_sizes=first[r].point_sizes
        )
        for r in range(3)
    ]
    for a, b in zip(first, again):
        assert a.point_sizes == b.point_sizes
        assert torch.equal(a.mesh.points, b.mesh.points)
        assert torch.equal(a.mesh.cells, b.mesh.cells)


def test_too_few_cells_for_the_ranks_raises(source):
    open_source, _ = source
    window = torch.arange(3)
    with pytest.raises(ValueError, match="cannot split 3 cells over 4 ranks"):
        read_local_cell_shard(open_source(), rank=0, world_size=4, cell_window=window)


def _chain(area_weighted: bool, window: int | None):
    fields = {
        "pressure": {"type": "scalar", "mean": 0.5, "std": 2.0},
        "wss": {"type": "vector", "mean": [1.0, 0.0, 0.0], "std": 0.5},
    }
    return [
        DropMeshFields(global_data=["Re"]),
        CenterMesh(use_area_weighting=area_weighted, store_center_as="center"),
        NormalizeMeshFields(association="cell_data", fields=fields),
        ComputeSurfaceNormals(store_as="cell_data", field_name="normals"),
        # No-op on the sample: the reader already drew the window.
        SubsampleMesh(n_cells=window or 10**6),
        MeshToDomainMesh(
            cell_data_targets=["pressure", "wss"], interior_points="cell_centroids"
        ),
    ]


def _apply_local(chain, payload):
    data = payload.mesh
    for t in chain:
        data = t.apply_local_shard(data, payload)
    return data


@pytest.mark.parametrize("area_weighted", [False, True])
@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("world", [2, 4])
def test_local_transform_chain_matches_unsharded(
    source, world, windowed, area_weighted
):
    r"""The recipe-style chain (centering, normalization, cell normals, a
    no-op subsample, centroid interior) on each share alone gives exactly
    the unsharded chain's rows."""
    open_source, full = source
    window, reference = _window_and_reference(full, windowed)
    chain = _chain(area_weighted, _WINDOW if windowed else None)
    stats = set()
    for t in chain:
        stats |= t.required_shard_stats()
    shares = _shares(open_source, world, window, stats)
    outputs = [_apply_local(chain, _as_payload(s, r)) for r, s in enumerate(shares)]
    for t in chain:
        reference = t(reference)

    cat = torch.cat
    assert torch.equal(
        cat([o.interior.points for o in outputs]), reference.interior.points
    )
    for key in reference.interior.point_data.keys():
        assert torch.equal(
            cat([o.interior.point_data[key] for o in outputs]),
            reference.interior.point_data[key],
        )
    bnd = [o.boundaries["vehicle"] for o in outputs]
    ref_bnd = reference.boundaries["vehicle"]
    assert torch.equal(
        cat([b.cell_data["normals"] for b in bnd]), ref_bnd.cell_data["normals"]
    )
    assert torch.equal(
        cat([b.points[b.cells] for b in bnd]), ref_bnd.points[ref_bnd.cells]
    )
    for o in outputs:
        assert torch.equal(o.global_data["center"], reference.global_data["center"])
        assert "Re" not in o.global_data.keys()


def test_unsupported_transforms_raise(source):
    open_source, full = source
    window, _ = _window_and_reference(full, True)
    payload = _as_payload(_shares(open_source, 2, window)[0], 0)
    with pytest.raises(NotImplementedError, match="subsample"):
        SubsampleMesh(n_cells=_WINDOW - 1).apply_local_shard(payload.mesh, payload)
    with pytest.raises(NotImplementedError, match="subsample"):
        SubsampleMesh(n_points=3).apply_local_shard(payload.mesh, payload)
    with pytest.raises(NotImplementedError, match="seam"):
        ComputeSurfaceNormals(store_as="point_data").apply_local_shard(
            payload.mesh, payload
        )


class _StubReader:
    r"""Serves fixed ``LocalMeshShard`` payloads; records requested statistics."""

    def __init__(self, payloads):
        self.payloads = payloads
        self.requested = None

    def request_shard_stats(self, stats):
        self.requested = set(stats)

    def __len__(self):
        return len(self.payloads)

    def __getitem__(self, index):
        return self.payloads[index], {"index": index}

    def close(self):
        pass


def test_dataset_runs_chain_on_local_share(source, monkeypatch):
    r"""``MeshDataset`` requests the chain's statistics, runs each transform
    through ``apply_local_shard`` and refuses statistics once a transform
    moved the geometry."""
    open_source, full = source
    window, _ = _window_and_reference(full, True)
    share = read_local_cell_shard(
        open_source(), rank=1, world_size=2, cell_window=window, stats={POINT_MEAN}
    )
    payload = _as_payload(share, 1)
    # The wrap needs a device mesh; this test stops before it.
    monkeypatch.setattr(mesh_dataset_module, "wrap_local_sample", lambda d, info: d)

    reader = _StubReader([payload])
    dataset = MeshDataset(
        reader,
        transforms=[DropMeshFields(global_data=["Re"]), CenterMesh(False)],
    )
    assert reader.requested == {POINT_MEAN}
    mesh, _ = dataset[0]
    torch.testing.assert_close(
        mesh.points, payload.mesh.points - share.stats[POINT_MEAN], rtol=0, atol=0
    )
    # Producer/consumer path.
    mesh2, _ = dataset._consume(dataset._load_host(0))
    assert torch.equal(mesh.points, mesh2.points)

    moved = MeshDataset(
        _StubReader([payload]), transforms=[ScaleMesh(2.0), CenterMesh(False)]
    )
    with pytest.raises(ValueError, match="ScaleMesh ran before it"):
        moved[0]


def test_read_rows_sequential_and_fallbacks(tmp_path, monkeypatch):
    r"""``read_rows`` equals indexing for every selection kind, using
    ``np.fromfile`` on memmap files and plain indexing otherwise."""
    values = torch.randn(5000, 3)
    TensorDict({"v": values}, batch_size=[]).memmap(tmp_path / "td")
    lazy = TensorDict.load_memmap(tmp_path / "td")["v"]
    assert getattr(lazy, "filename", None) is not None

    calls = []
    real = np.fromfile
    monkeypatch.setattr(
        np, "fromfile", lambda *a, **k: calls.append(k["count"]) or real(*a, **k)
    )
    window = _cyclic_block_indices(5000, 700, start=4600)  # wraps: 2 runs
    dense = torch.unique(torch.randint(0, 5000, (2000,)))  # sorted, dense
    sparse = torch.tensor([0, 1000, 2000, 3000, 4000, 4999])  # sorted, sparse
    unsorted = torch.tensor([4000, 2, 4999, 7, 1, 3000])  # sparse, unsorted
    cases = [
        (None, values, 1),
        (slice(10, 900), values[10:900], 1),
        (window, values[window], 2),
        (dense, values[dense], 1),
        (sparse, values[sparse], 0),
        (unsorted, values[unsorted], 0),
    ]
    for selection, expected, n_reads in cases:
        calls.clear()
        out = read_rows(lazy, selection)
        assert torch.equal(out, expected)
        assert type(out) is torch.Tensor
        assert len(calls) == n_reads, selection
    # Not file-backed: plain indexing.
    calls.clear()
    assert torch.equal(read_rows(values, dense), values[dense])
    assert calls == []
