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

"""Slicing preserves fields and only synchronizes for unknown output counts."""

import warnings
from contextlib import contextmanager

import pytest
import torch

import physicsnemo.mesh.mesh as mesh_module


@pytest.fixture
def mesh_with_fields(mesh_factory, device):
    """Two triangles with nested data and populated local geometry caches."""
    mesh = mesh_factory(3, 2, device=device)
    mesh.point_data["temperature"] = torch.arange(mesh.n_points, device=device)
    mesh.cell_data["pressure"] = torch.arange(mesh.n_cells, device=device)
    mesh.cell_data["nested", "velocity"] = torch.arange(
        3 * mesh.n_cells, device=device
    ).reshape(-1, 3)
    _ = mesh.cell_centroids, mesh.cell_areas, mesh.cell_normals
    return mesh


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8])
@pytest.mark.parametrize("kept", [[], [1], [0, 1]])
def test_cell_masks_preserve_data_and_caches(mesh_with_fields, device, dtype, kept):
    """Mask and integer selections agree, including empty and full selections."""
    mesh = mesh_with_fields
    indices = torch.tensor(kept, dtype=torch.long, device=device)
    mask = torch.zeros(mesh.n_cells, dtype=dtype, device=device)
    mask[indices] = 1
    expected = mesh.slice_cells(indices)

    # CPU masks also remain valid when the mesh is on CUDA.
    for selection in (mask, mask.cpu()):
        actual = mesh.slice_cells(selection)
        assert actual.points is mesh.points
        assert actual.point_data is mesh.point_data
        assert actual.global_data is mesh.global_data
        torch.testing.assert_close(actual.cells, expected.cells)
        assert actual.cell_data.batch_size == expected.cell_data.batch_size
        for key in expected.cell_data.keys(True, True):
            torch.testing.assert_close(actual.cell_data[key], expected.cell_data[key])
        for key in ("centroids", "areas", "normals"):
            torch.testing.assert_close(
                actual._cache["cell", key], expected._cache["cell", key]
            )


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8])
@pytest.mark.parametrize("length", [0, 1, 3])
def test_cell_mask_length_is_checked(mesh_with_fields, device, dtype, length):
    """Converting a mask to ids must not hide a mismatched mask length."""
    mask = torch.zeros(length, dtype=dtype, device=device)
    with pytest.raises(IndexError):
        mesh_with_fields.slice_cells(mask)


@pytest.mark.parametrize("case", ["empty", "cloud"])
def test_empty_cell_data_does_not_retain_storage(mesh_with_fields, device, case):
    """Point slicing releases cell-field storage when no cells remain."""
    mesh = mesh_with_fields
    if case == "cloud":
        mesh = mesh.slice_cells(slice(0, 0))
    indices = torch.tensor(
        [] if case == "empty" else [1, 2, 3], dtype=torch.long, device=device
    )

    actual = mesh.slice_points(indices)

    assert actual.n_cells == 0
    for key in mesh.cell_data.keys(True, True):
        torch.testing.assert_close(actual.cell_data[key], mesh.cell_data[key][:0])
        assert actual.cell_data[key].untyped_storage().nbytes() == 0


@pytest.mark.parametrize("device", ["cpu"])
def test_empty_memmap_selection_can_be_saved(mesh_with_fields, tmp_path):
    """Empty selections can be saved without copying the source memmap files."""
    source = tmp_path / "source.pmsh"
    destination = tmp_path / "empty.pmsh"
    mesh_with_fields.save(source)
    loaded = mesh_module.Mesh.load(source)

    selected = loaded.slice_points([])
    selected.save(destination)
    restored = mesh_module.Mesh.load(destination)

    torch.testing.assert_close(restored.points, selected.points)
    torch.testing.assert_close(restored.cells, selected.cells)
    for expected, actual in (
        (selected.point_data, restored.point_data),
        (selected.cell_data, restored.cell_data),
    ):
        assert set(actual.keys(True, True)) == set(expected.keys(True, True))
        for key in expected.keys(True, True):
            torch.testing.assert_close(actual[key], expected[key])


@contextmanager
def _cuda_sync_budget(max_syncs):
    """Count CUDA synchronization warnings without requiring profiler support."""
    previous = torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            yield
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    syncs = [w for w in caught if "synchronizing CUDA operation" in str(w.message)]
    assert len(syncs) <= max_syncs, (
        f"Expected at most {max_syncs} CUDA waits, got {len(syncs)}"
    )


@pytest.mark.cuda
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("selection", ["bool", "uint8", "indices", "slice"])
def test_slice_cells_cuda_syncs(mesh_with_fields, selection):
    """Fields and caches share the single wait needed for a CUDA mask's count."""
    mesh = mesh_with_fields
    indices = {
        "bool": torch.tensor([False, True], device="cuda"),
        "uint8": torch.tensor([0, 1], dtype=torch.uint8, device="cuda"),
        "indices": torch.tensor([1], device="cuda"),
        "slice": slice(1, 2),
    }[selection]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            mesh.slice_cells(indices)
        torch.cuda.synchronize()
        with _cuda_sync_budget(1 if selection in ("bool", "uint8") else 0):
            actual = mesh.slice_cells(indices)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual.cells, mesh.cells[1:2])
    assert actual.cell_data.batch_size == torch.Size([1])
    torch.testing.assert_close(
        actual.cell_data["pressure"], mesh.cell_data["pressure"][1:2]
    )
    for key in ("centroids", "areas", "normals"):
        torch.testing.assert_close(
            actual._cache["cell", key], mesh._cache["cell", key][1:2]
        )


@pytest.mark.cuda
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("search", [False, True], ids=["lookup_table", "binary_search"])
@pytest.mark.parametrize(
    "case,max_syncs", [("indices", 1), ("mask", 2), ("empty", 0), ("cloud", 0)]
)
def test_slice_points_cuda_syncs(
    mesh_with_fields, monkeypatch, search, case, max_syncs
):
    """Only unknown point/cell counts require a wait, independent of field count."""
    monkeypatch.setattr(mesh_module, "_SEARCH_REMAP_RATIO", 0 if search else 10**12)
    mesh = mesh_with_fields
    if case == "cloud":
        mesh = mesh.slice_cells(slice(0, 0))
    kept = torch.tensor(
        [] if case == "empty" else [1, 2, 3], dtype=torch.long, device="cuda"
    )
    if case == "mask":
        indices = torch.tensor([False, True, True, True], device="cuda")
    else:
        indices = kept
    expected_cells = (
        torch.empty((0, 3), dtype=torch.long, device="cuda")
        if case in ("cloud", "empty")
        else torch.tensor([[0, 2, 1]], device="cuda")
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            mesh.slice_points(indices)
        torch.cuda.synchronize()
        with _cuda_sync_budget(max_syncs):
            actual = mesh.slice_points(indices)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual.points, mesh.points[kept])
    torch.testing.assert_close(
        actual.point_data["temperature"], mesh.point_data["temperature"][kept]
    )
    torch.testing.assert_close(actual.cells, expected_cells)
    cell_selection = slice(0, 0) if case in ("cloud", "empty") else slice(1, 2)
    assert actual.cell_data.batch_size == torch.Size([expected_cells.shape[0]])
    for key in mesh.cell_data.keys(True, True):
        torch.testing.assert_close(
            actual.cell_data[key], mesh.cell_data[key][cell_selection]
        )
