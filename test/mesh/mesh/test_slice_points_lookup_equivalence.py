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

"""``Mesh.slice_points`` locates kept vertices by binary search instead of a
full-mesh lookup table. These tests pin the result to the lookup-table
algorithm for every accepted index form, including unsorted indices,
duplicates, negatives and empty selections, and for a memory-mapped mesh
loaded from disk."""

import pytest
import torch

import physicsnemo.mesh.mesh as mesh_module
from physicsnemo.mesh import Mesh


@pytest.fixture(autouse=True, params=["lookup_table", "binary_search"])
def _force_remap_algorithm(request, monkeypatch):
    """Run every test on both cell-remapping algorithms of ``slice_points``:
    the ratio threshold is pushed to force the lookup table or the search."""
    ratio = 10**12 if request.param == "lookup_table" else 0
    monkeypatch.setattr(mesh_module, "_SEARCH_REMAP_RATIO", ratio)


def _reference_slice_points(mesh: Mesh, indices) -> Mesh:
    """Lookup-table implementation of ``slice_points`` (the previous algorithm)."""
    all_indices = torch.arange(mesh.n_points, device=mesh.points.device)
    kept = (
        torch.tensor([indices], device=mesh.points.device)
        if isinstance(indices, int)
        else all_indices[indices]
    )
    old_to_new = torch.full((mesh.n_points,), -1, dtype=torch.long)
    old_to_new[kept] = torch.arange(len(kept), dtype=torch.long)
    remapped = old_to_new[mesh.cells]
    valid = (remapped >= 0).all(dim=-1)
    return Mesh(
        points=mesh.points[kept],
        cells=remapped[valid],
        point_data=mesh.point_data[kept],
        cell_data=mesh.cell_data[valid],
        global_data=mesh.global_data,
    )


def _assert_same_mesh(a: Mesh, b: Mesh) -> None:
    """Exact equality of points, cells and every point/cell data leaf."""
    assert torch.equal(a.points, b.points)
    assert torch.equal(a.cells, b.cells)
    assert set(a.point_data.keys(True, True)) == set(b.point_data.keys(True, True))
    for key in a.point_data.keys(True, True):
        assert torch.equal(a.point_data[key], b.point_data[key]), key
    assert set(a.cell_data.keys(True, True)) == set(b.cell_data.keys(True, True))
    for key in a.cell_data.keys(True, True):
        assert torch.equal(a.cell_data[key], b.cell_data[key]), key


def _random_mesh(n_points: int = 60, n_cells: int = 80) -> Mesh:
    """Triangles over random points, with flat and nested point data."""
    torch.manual_seed(0)
    return Mesh(
        points=torch.randn(n_points, 3),
        cells=torch.randint(0, n_points, (n_cells, 3)),
        point_data={
            "f": torch.randn(n_points),
            "nested": {"v": torch.randn(n_points, 3)},
        },
        cell_data={"g": torch.arange(n_cells, dtype=torch.float32)},
    )


@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor([5, 3, 40, 12, 0]),  # unsorted
        torch.tensor([7, 7, 20, 20, 1]),  # duplicates: last position wins
        torch.tensor([-1, -2, 4]),  # negative indices
        torch.tensor([], dtype=torch.long),  # nothing kept
        [],
        slice(10, 50, 3),
        slice(50, 10),  # empty forward slice with reversed bounds
        slice(-1, 0),
        slice(100, 200),  # clipped to an empty range
        slice(-100, -80),
        slice(5, 5),
        slice(-100, 100, 7),
        torch.tensor([5, 3, 40], dtype=torch.int32),
        [2, 9, 4],
        4,
    ],
)
def test_matches_lookup_table_for_index_forms(indices):
    """Every accepted index form gives the lookup-table result."""
    mesh = _random_mesh()
    _assert_same_mesh(
        mesh.slice_points(indices), _reference_slice_points(mesh, indices)
    )


def test_matches_lookup_table_for_boolean_mask():
    """Boolean masks, as tensors and as lists of bools."""
    mesh = _random_mesh()
    torch.manual_seed(1)
    mask = torch.rand(mesh.n_points) < 0.6
    _assert_same_mesh(mesh.slice_points(mask), _reference_slice_points(mesh, mask))
    _assert_same_mesh(
        mesh.slice_points(mask.tolist()), _reference_slice_points(mesh, mask)
    )


def test_byte_mask_keeps_mask_semantics():
    """Legacy uint8 masks remain masks rather than becoming integer IDs."""
    mesh = _random_mesh()
    mask = torch.zeros(mesh.n_points, dtype=torch.uint8)
    mask[[0, 2, 4]] = 1
    with pytest.warns(UserWarning, match="uint8"):
        expected = _reference_slice_points(mesh, mask)
    _assert_same_mesh(mesh.slice_points(mask), expected)


@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor([1.5, 2.5]),
        torch.tensor([], dtype=torch.float32),
        torch.tensor([1, 2], dtype=torch.int8),
        torch.tensor([1, 2], dtype=torch.int16),
        torch.tensor([1 + 0j, 2 + 0j]),
        torch.tensor([True, False]),
        torch.ones(61, dtype=torch.bool),
        torch.tensor([1, 0], dtype=torch.uint8),
    ],
)
def test_rejects_invalid_tensor_indices_like_native_indexing(indices):
    """Invalid tensor dtypes and mask lengths cannot silently select points."""
    mesh = _random_mesh()
    with pytest.raises(IndexError):
        torch.arange(mesh.n_points)[indices]
    with pytest.raises(IndexError):
        mesh.slice_points(indices)


@pytest.mark.parametrize(
    "indices",
    [
        torch.ones(2, 30, dtype=torch.bool),
        torch.tensor([[1, 2]], dtype=torch.long),
    ],
)
def test_rejects_multidimensional_selections(indices):
    """Point selection requires a single row-index or mask dimension."""
    with pytest.raises(IndexError, match="one-dimensional"):
        _random_mesh().slice_points(indices)


@pytest.mark.parametrize("indices", [60, -61, [0, 60], [0, -61]])
def test_rejects_out_of_range_indices(indices):
    """Negative normalization must not turn invalid indices into valid ones."""
    mesh = _random_mesh()
    with pytest.raises(IndexError):
        _reference_slice_points(mesh, indices)
    with pytest.raises(IndexError):
        mesh.slice_points(indices)


@pytest.mark.parametrize("indices", [slice(None, None, -1), slice(5, 1, -2)])
def test_rejects_negative_slice_steps_like_native_indexing(indices):
    """PyTorch's positive-step slice contract is retained."""
    mesh = _random_mesh()
    with pytest.raises(ValueError, match="step must be greater than zero"):
        _reference_slice_points(mesh, indices)
    with pytest.raises(ValueError, match="step must be greater than zero"):
        mesh.slice_points(indices)


@pytest.mark.parametrize(
    "indices",
    [
        [0, 1, 2],  # lookup table or search, by the forced ratio
        slice(100, 200),  # nothing kept: the empty branch
    ],
)
def test_int32_connectivity_remaps_to_int64_on_every_branch(indices):
    """Remapped connectivity is int64 whatever the input dtype, in all three
    branches, as the lookup table always produced."""
    mesh = _random_mesh()
    mesh = Mesh(
        points=mesh.points,
        cells=mesh.cells.to(torch.int32),
        point_data=mesh.point_data,
        cell_data=mesh.cell_data,
    )
    assert mesh.cells.dtype == torch.int32
    got = mesh.slice_points(indices)
    want = _reference_slice_points(mesh, indices)
    assert got.cells.dtype == torch.int64
    assert want.cells.dtype == torch.int64
    _assert_same_mesh(got, want)


def test_point_cloud_skips_the_remap():
    """A mesh without cells keeps its empty connectivity; only points are sliced."""
    torch.manual_seed(0)
    cloud = Mesh(points=torch.randn(50, 3), point_data={"f": torch.randn(50)})
    kept = torch.tensor([7, 3, 3, 40])
    got = cloud.slice_points(kept)
    assert got.n_points == 4
    assert got.n_cells == 0
    assert got.cells.shape == (0, 1)
    assert got.cells.dtype == torch.int64
    _assert_same_mesh(got, _reference_slice_points(cloud, kept))


def test_matches_lookup_table_on_random_selections():
    """Random unsorted subsets of a larger mesh."""
    mesh = _random_mesh(n_points=200, n_cells=400)
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        k = int(torch.randint(1, mesh.n_points + 1, (1,), generator=g))
        kept = torch.randperm(mesh.n_points, generator=g)[:k]
        _assert_same_mesh(mesh.slice_points(kept), _reference_slice_points(mesh, kept))


def test_matches_lookup_table_for_memmap_backed_mesh(tmp_path):
    """The reader's cell-subsampling path: a lazily loaded mesh sliced to the
    vertices that a block of cells references."""
    full = _random_mesh(n_points=500, n_cells=900)
    full.save(tmp_path / "m.pmsh")
    loaded = Mesh.load(tmp_path / "m.pmsh")
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        cell_block = torch.randperm(full.n_cells, generator=g)[:40]
        referenced = torch.unique(full.cells[cell_block])
        got = loaded.slice_cells(cell_block).slice_points(referenced)
        want = _reference_slice_points(full.slice_cells(cell_block), referenced)
        _assert_same_mesh(got, want)
        assert got.n_points == referenced.numel()
