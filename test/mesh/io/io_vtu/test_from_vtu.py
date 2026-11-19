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

"""Tests for the memory-mapped ``.vtu`` reader."""

import os
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest
import torch

pv = pytest.importorskip("pyvista")
vtk = pytest.importorskip("vtk")

from physicsnemo.datapipes.readers.mesh import MeshReader, _subsample_mesh  # noqa: E402
from physicsnemo.mesh import Mesh  # noqa: E402
from physicsnemo.mesh.io import from_pyvista, from_vtu  # noqa: E402
from physicsnemo.mesh.io.io_vtu import _parse_vtu_header, _resolve_cells  # noqa: E402


def _tet_grid(dims=(4, 3, 3)) -> "pv.UnstructuredGrid":
    grid = pv.ImageData(dimensions=dims).triangulate()
    rng = np.random.default_rng(0)
    grid.point_data["p"] = rng.random(grid.n_points)
    grid.point_data["U"] = rng.random((grid.n_points, 3)).astype(np.float32)
    grid.point_data["idx"] = np.arange(grid.n_points, dtype=np.int32)
    grid.cell_data["c"] = rng.random(grid.n_cells)
    grid.cell_data["T"] = rng.random((grid.n_cells, 9))
    grid.field_data["meta"] = np.array([1.5, 2.5])
    grid.field_data["tag"] = np.array([7], dtype=np.int32)
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


def _edit_header(path, old: bytes, new: bytes, count: int = 1):
    """Rewrite text in the XML header. Offsets are relative to the ``_``
    marker, so the header may change length freely."""
    data = path.read_bytes()
    split = data.index(b"<AppendedData")
    head, tail = data[:split], data[split:]
    assert head.count(old) >= count, (old, head.count(old))
    path.write_bytes(head.replace(old, new, count) + tail)
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
def test_round_trip_matches_from_pyvista(tmp_path, header64):
    grid = _tet_grid()
    path = _write_vtu(grid, tmp_path / "tet.vtu", header64=header64)
    mesh = from_vtu(path)
    _assert_same_mesh(mesh, from_pyvista(pv.read(str(path))))
    assert mesh.point_data["idx"].dtype == torch.int32
    assert mesh.cells.dtype == torch.int64


def test_tensors_are_views_of_one_file_map(tmp_path):
    grid = _tet_grid()
    path = _write_vtu(grid, tmp_path / "tet.vtu")
    file_size = os.path.getsize(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # from_numpy on a read-only map would warn
        mesh = from_vtu(path)
    # Every tensor is a view into one map of the file: its data pointer minus
    # its byte offset in the file is the same base address for all of them.
    with open(path, "rb") as f:
        header = _parse_vtu_header(f)
        connectivity = _resolve_cells(f, header)
    infos = {
        "points": header.points,
        "cells": connectivity,
        "U": next(a for a in header.point_data if a.name == "U"),
        "T": next(a for a in header.cell_data if a.name == "T"),
    }
    tensors = {
        "points": mesh.points,
        "cells": mesh.cells,
        "U": mesh.point_data["U"],
        "T": mesh.cell_data["T"],
    }
    bases = {k: tensors[k].data_ptr() - infos[k].payload_offset for k in infos}
    assert len(set(bases.values())) == 1, bases
    assert file_size > 0
    # Copy-on-write: an in-place edit neither crashes nor reaches the file.
    before = path.read_bytes()
    mesh.points[0, 0] = 123.0
    assert float(mesh.points[0, 0]) == 123.0
    assert path.read_bytes() == before


def test_big_endian_is_correct_but_copied(tmp_path):
    grid = _tet_grid()
    path = _write_vtu(grid, tmp_path / "be.vtu", big_endian=True)
    mesh = from_vtu(path)
    _assert_same_mesh(mesh, from_pyvista(pv.read(str(path))))
    # Big-endian data was byte-swapped into a private buffer, so the point
    # and cell tensors no longer share one base address.
    with open(path, "rb") as f:
        header = _parse_vtu_header(f)
        connectivity = _resolve_cells(f, header)
    assert (mesh.points.data_ptr() - header.points.payload_offset) != (
        mesh.cells.data_ptr() - connectivity.payload_offset
    )


def test_subsample_is_bit_identical_to_pdmsh(tmp_path):
    grid = _tet_grid(dims=(6, 5, 5))
    path = _write_vtu(grid, tmp_path / "tet.vtu")
    from_pyvista(pv.read(str(path))).save(tmp_path / "tet.pmsh")
    for n_cells in (9, 150):
        a = _subsample_mesh(
            from_vtu(path), n_cells=n_cells, generator=torch.Generator().manual_seed(0)
        )
        b = _subsample_mesh(
            Mesh.load(tmp_path / "tet.pmsh", mode="r"),
            n_cells=n_cells,
            generator=torch.Generator().manual_seed(0),
        )
        assert torch.equal(a.cells, b.cells)
        assert torch.equal(a.points, b.points)
        for key in b.point_data.keys():
            assert torch.equal(a.point_data[key], b.point_data[key]), key
        for key in b.cell_data.keys():
            assert torch.equal(a.cell_data[key], b.cell_data[key]), key


def test_mesh_reader_loads_vtu_samples(tmp_path):
    grid = _tet_grid()
    for i in range(3):
        (tmp_path / f"case_{i}").mkdir()
        _write_vtu(grid, tmp_path / f"case_{i}" / "mesh.vtu")
    reader = MeshReader(tmp_path, pattern="**/*.vtu", subsample_n_cells=7)
    assert len(reader) == 3
    sample, _ = reader[1]
    assert sample.cells.shape == (7, 4)
    assert "U" in sample.point_data
    full, _ = MeshReader(tmp_path, pattern="**/*.vtu")[0]
    _assert_same_mesh(full, from_pyvista(pv.read(str(tmp_path / "case_0/mesh.vtu"))))


def test_skip_arrays_in_every_section(tmp_path):
    path = _write_vtu(_tet_grid(), tmp_path / "tet.vtu")
    mesh = from_vtu(path, skip_arrays=["T", "idx", "meta"])
    assert set(mesh.point_data.keys()) == {"p", "U"}
    assert set(mesh.cell_data.keys()) == {"c"}
    assert set(mesh.global_data.keys()) == {"tag"}


def test_device_materializes(tmp_path):
    path = _write_vtu(_tet_grid(), tmp_path / "tet.vtu")
    mesh = from_vtu(path, device="cpu")
    assert mesh.points.device.type == "cpu"


def test_vertex_only_grid_is_a_point_cloud_like_from_pyvista(tmp_path):
    grid = pv.PolyData(
        np.random.default_rng(1).random((10, 3))
    ).cast_to_unstructured_grid()
    grid.point_data["p"] = np.arange(10.0)
    grid.cell_data["c"] = np.arange(10.0)
    path = _write_vtu(grid, tmp_path / "verts.vtu")
    mesh = from_vtu(path)
    ref = from_pyvista(pv.read(str(path)))
    assert mesh.n_cells == ref.n_cells == 0
    _assert_same_mesh(mesh, ref)


def test_zero_cell_and_empty_files(tmp_path):
    cloud = pv.UnstructuredGrid()
    cloud.points = np.random.default_rng(1).random((6, 3))
    cloud.point_data["p"] = np.arange(6.0)
    mesh = from_vtu(_write_vtu(cloud, tmp_path / "cloud.vtu"))
    assert mesh.n_points == 6 and mesh.n_cells == 0
    assert torch.equal(mesh.point_data["p"], torch.arange(6.0, dtype=torch.float64))
    empty = from_vtu(_write_vtu(pv.UnstructuredGrid(), tmp_path / "empty.vtu"))
    assert empty.points.shape == (0, 3) and empty.n_cells == 0


def test_only_header_types_and_one_offset_are_read(tmp_path, monkeypatch):
    import physicsnemo.mesh.io.io_vtu as mod

    path = _write_vtu(_tet_grid(), tmp_path / "tet.vtu")
    header = _parse_vtu_header(open(path, "rb"))
    reads = []
    real_open = open

    class Counting:
        def __init__(self, f):
            self._f = f

        def read(self, n=-1):
            data = self._f.read(n)
            reads.append(len(data))
            return data

        def __getattr__(self, name):
            return getattr(self._f, name)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self._f.close()

    monkeypatch.setattr(
        mod, "open", lambda *a, **k: Counting(real_open(*a, **k)), raising=False
    )
    from_vtu(path, skip_arrays=["p", "U", "idx", "c", "T", "meta", "tag"])
    n_header_blocks = len([r for r in reads if r == mod._HEADER_READ_BLOCK]) + 1
    assert len(reads) == n_header_blocks + 2, reads
    assert reads[-2] == header.types.payload_nbytes
    assert reads[-1] == header.offsets.dtype.itemsize


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(compress=True), "Compressed"),
        (dict(base64=True), "encoding"),
    ],
)
def test_rejects_non_raw_encodings(tmp_path, kwargs, match):
    path = _write_vtu(_tet_grid(), tmp_path / "x.vtu", **kwargs)
    with pytest.raises(NotImplementedError, match=match):
        from_vtu(path)


def test_rejects_pyvista_default_save(tmp_path):
    path = tmp_path / "default.vtu"
    _tet_grid().save(path)
    with pytest.raises(NotImplementedError, match="Compressed|AppendedData"):
        from_vtu(path)


def test_rejects_non_simplex_and_mixed_cells(tmp_path):
    hexes = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    with pytest.raises(NotImplementedError, match="simplex"):
        from_vtu(_write_vtu(hexes, tmp_path / "hex.vtu"))
    tets = pv.ImageData(dimensions=(3, 3, 3)).triangulate()
    surface = tets.extract_surface(algorithm=None).cast_to_unstructured_grid()
    mixed = tets.merge(surface, merge_points=True)
    assert len(np.unique(mixed.celltypes)) == 2
    with pytest.raises(NotImplementedError, match="simplex"):
        from_vtu(_write_vtu(mixed, tmp_path / "mixed.vtu"))


def test_pvtu_reports_file_type(tmp_path):
    path = tmp_path / "parallel.pvtu"
    path.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="PUnstructuredGrid" version="0.1">\n'
        '<PUnstructuredGrid GhostLevel="0"><Piece Source="p_0.vtu"/></PUnstructuredGrid>\n'
        "</VTKFile>\n"
    )
    with pytest.raises(ValueError, match="PUnstructuredGrid"):
        from_vtu(path)


def test_appended_tag_attribute_parsing_is_robust(tmp_path):
    path = _write_vtu(_tet_grid(), tmp_path / "tet.vtu")
    ref = from_pyvista(pv.read(str(path)))
    data = path.read_bytes().replace(
        b'<AppendedData encoding="raw">',
        b"<AppendedData my_attr=\"x_y\" encoding = 'raw' >",
        1,
    )
    path.write_bytes(data)
    _assert_same_mesh(from_vtu(path), ref)


@pytest.mark.parametrize(
    "old, new, match",
    [
        (b'Name="p"', b'Name="U [m/s]"', "path separators"),
        (b'Name="p"', b'Name=""', "no Name"),
        (b'Name="c"', b'Name="T"', "Duplicate.*CellData"),
    ],
)
def test_bad_array_names_are_rejected(tmp_path, old, new, match):
    path = _edit_header(_write_vtu(_tet_grid(), tmp_path / "tet.vtu"), old, new)
    with pytest.raises(ValueError, match=match):
        from_vtu(path)


def test_unknown_vtk_type_is_a_clear_error(tmp_path):
    path = _edit_header(
        _write_vtu(_tet_grid(), tmp_path / "tet.vtu"),
        b'type="Int32" Name="idx"',
        b'type="vtkIdType" Name="idx"',
    )
    with pytest.raises(NotImplementedError, match="vtkIdType"):
        from_vtu(path)


def test_out_of_range_connectivity_is_rejected(tmp_path):
    path = _write_vtu(_tet_grid(), tmp_path / "tet.vtu")
    header = _parse_vtu_header(open(path, "rb"))
    with open(path, "r+b") as f:
        f.seek(header.connectivity.payload_offset)
        f.write(np.array([10**6], dtype=header.connectivity.dtype).tobytes())
    with pytest.raises(ValueError, match="point ids"):
        from_vtu(path)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork start method unavailable")
def test_forked_worker_reads_parent_mesh(tmp_path):
    path = _write_vtu(_tet_grid(), tmp_path / "tet.vtu")
    script = textwrap.dedent(
        f"""
        import multiprocessing as mp, sys, torch
        from physicsnemo.mesh.io import from_vtu
        def child(mesh):
            print(tuple(mesh.points.clone().shape), flush=True)
        if __name__ == "__main__":
            mesh = from_vtu({str(path)!r})
            p = mp.get_context("fork").Process(target=child, args=(mesh,))
            p.start(); p.join(60)
            if p.is_alive():
                p.kill(); sys.exit(3)
            sys.exit(p.exitcode)
        """
    )
    proc = subprocess.run(  # noqa: S603 - interpreter/script are test constants
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-2000:])
    assert "(" in proc.stdout
