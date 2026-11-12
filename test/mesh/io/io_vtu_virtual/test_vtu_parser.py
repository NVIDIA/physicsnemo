# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Tests for the virtual zarr view of ``.vtu`` files (prototype)."""

import numpy as np
import pytest
import torch

pv = pytest.importorskip("pyvista")
vtk = pytest.importorskip("vtk")
pytest.importorskip("virtualizarr")
pytest.importorskip("obstore")

from physicsnemo.datapipes.readers.mesh import _zarr_mesh_subsampled  # noqa: E402
from physicsnemo.mesh.io import from_pyvista  # noqa: E402
from physicsnemo.mesh.io.io_vtu_virtual import (  # noqa: E402
    VTUParser,
    open_vtu_virtual,
    vtu_to_mesh,
)
from physicsnemo.mesh.io.io_zarr import _TYPE_ATTR  # noqa: E402


def _tet_grid() -> "pv.UnstructuredGrid":
    grid = pv.ImageData(dimensions=(4, 3, 3)).triangulate()
    rng = np.random.default_rng(0)
    grid.point_data["p"] = rng.random(grid.n_points)
    grid.point_data["U"] = rng.random((grid.n_points, 3)).astype(np.float32)
    grid.point_data["idx"] = np.arange(grid.n_points, dtype=np.int32)
    grid.cell_data["c"] = rng.random(grid.n_cells)
    grid.cell_data["T"] = rng.random((grid.n_cells, 9))
    grid.field_data["meta"] = np.array([1.5, 2.5])
    return grid


def _write_vtu(
    grid, path, *, compress=False, base64=False, header64=False, big_endian=False
):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    writer.SetDataModeToAppended()
    if base64:
        writer.EncodeAppendedDataOn()
    else:
        writer.EncodeAppendedDataOff()
    if compress:
        writer.SetCompressorTypeToZLib()
    else:
        writer.SetCompressorTypeToNone()
    if header64:
        writer.SetHeaderTypeToUInt64()
    if big_endian:
        writer.SetByteOrderToBigEndian()
    writer.Write()
    return path


def _assert_same_mesh(mesh, ref):
    assert torch.equal(mesh.points.double(), ref.points.double())
    assert mesh.cells.shape == ref.cells.shape
    assert torch.equal(mesh.cells.long(), ref.cells.long())
    for group in ("point_data", "cell_data", "global_data"):
        got, want = getattr(mesh, group), getattr(ref, group)
        assert set(got.keys()) == set(want.keys()), group
        for key in want.keys():
            assert got[key].shape == want[key].shape, (group, key)
            assert torch.allclose(got[key].double(), want[key].double()), (group, key)


@pytest.mark.parametrize("header64", [False, True], ids=["UInt32", "UInt64"])
def test_tet_round_trip_matches_from_pyvista(tmp_path, header64):
    grid = _tet_grid()
    path = _write_vtu(grid, tmp_path / "tet.vtu", header64=header64)

    group = open_vtu_virtual(path)
    assert group.attrs[_TYPE_ATTR] == "Mesh"
    assert group["cells"].shape == (grid.n_cells, 4)
    assert group["points"].shape == (grid.n_points, 3)
    assert group["point_data"]["U"].shape == (grid.n_points, 3)
    assert group["cell_data"]["T"].shape == (grid.n_cells, 9)
    assert group["global_data"]["meta"].shape == (2,)

    _assert_same_mesh(vtu_to_mesh(path), from_pyvista(pv.read(str(path))))


def test_big_endian_file(tmp_path):
    grid = _tet_grid()
    path = _write_vtu(grid, tmp_path / "be.vtu", big_endian=True)
    _assert_same_mesh(vtu_to_mesh(path), from_pyvista(pv.read(str(path))))


def test_skip_arrays(tmp_path):
    path = _write_vtu(_tet_grid(), tmp_path / "tet.vtu")
    mesh = vtu_to_mesh(path, skip_arrays=["T", "idx"])
    assert set(mesh.point_data.keys()) == {"p", "U"}
    assert set(mesh.cell_data.keys()) == {"c"}


def test_existing_zarr_subsample_path_reads_virtual_store(tmp_path):
    grid = _tet_grid()
    path = _write_vtu(grid, tmp_path / "tet.vtu")
    group = open_vtu_virtual(path)
    sub = _zarr_mesh_subsampled(
        group, n_cells=7, n_points=None, generator=torch.Generator().manual_seed(0)
    )
    assert sub.cells.shape == (7, 4)
    assert int(sub.cells.max()) < sub.points.shape[0]
    assert set(sub.point_data.keys()) == {"p", "U", "idx"}
    assert sub.cell_data["T"].shape == (7, 9)


def test_zero_cell_file(tmp_path):
    cloud = pv.UnstructuredGrid()
    cloud.points = np.random.default_rng(1).random((6, 3))
    cloud.point_data["p"] = np.arange(6.0)
    path = _write_vtu(cloud, tmp_path / "cloud.vtu")
    mesh = vtu_to_mesh(path)
    assert mesh.n_points == 6
    assert mesh.n_cells == 0
    assert torch.equal(mesh.point_data["p"], torch.arange(6.0, dtype=torch.float64))


def test_rejects_compressed(tmp_path):
    path = _write_vtu(_tet_grid(), tmp_path / "z.vtu", compress=True)
    with pytest.raises(NotImplementedError, match="Compressed"):
        open_vtu_virtual(path)


def test_rejects_base64(tmp_path):
    path = _write_vtu(_tet_grid(), tmp_path / "b64.vtu", base64=True)
    with pytest.raises(NotImplementedError, match="encoding"):
        open_vtu_virtual(path)


def test_rejects_pyvista_default_save(tmp_path):
    path = tmp_path / "default.vtu"
    _tet_grid().save(path)
    with pytest.raises(NotImplementedError):
        open_vtu_virtual(path)


def test_rejects_non_simplex_cells(tmp_path):
    hexes = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    path = _write_vtu(hexes, tmp_path / "hex.vtu")
    with pytest.raises(NotImplementedError, match="simplex"):
        open_vtu_virtual(path)


def test_rejects_mixed_cells(tmp_path):
    tets = pv.ImageData(dimensions=(3, 3, 3)).triangulate()
    surface = tets.extract_surface(algorithm=None).cast_to_unstructured_grid()
    mixed = tets.merge(surface, merge_points=True)
    assert len(np.unique(mixed.celltypes)) == 2
    path = _write_vtu(mixed, tmp_path / "mixed.vtu")
    with pytest.raises(NotImplementedError, match="simplex"):
        open_vtu_virtual(path)


def test_parser_protocol_with_explicit_registry(tmp_path):
    import zarr
    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import LocalStore

    path = _write_vtu(_tet_grid(), tmp_path / "tet.vtu")
    registry = ObjectStoreRegistry(
        {tmp_path.as_uri(): LocalStore(prefix=str(tmp_path))}
    )
    store = VTUParser()(path.as_uri(), registry)
    group = zarr.open_group(store, mode="r")
    assert group["cells"].shape == (_tet_grid().n_cells, 4)
