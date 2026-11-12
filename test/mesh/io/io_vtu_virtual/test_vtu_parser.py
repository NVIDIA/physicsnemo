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

import os
import subprocess
import sys
import textwrap

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
    _parse_vtu_header,
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
@pytest.mark.parametrize("big_endian", [False, True], ids=["LE", "BE"])
def test_tet_round_trip_matches_from_pyvista(tmp_path, header64, big_endian):
    grid = _tet_grid()
    path = _write_vtu(
        grid, tmp_path / "tet.vtu", header64=header64, big_endian=big_endian
    )

    group = open_vtu_virtual(path)
    assert group.attrs[_TYPE_ATTR] == "Mesh"
    assert group["cells"].shape == (grid.n_cells, 4)
    assert group["points"].shape == (grid.n_points, 3)
    assert group["point_data"]["U"].shape == (grid.n_points, 3)
    assert group["cell_data"]["T"].shape == (grid.n_cells, 9)
    assert group["global_data"]["meta"].shape == (2,)

    mesh = vtu_to_mesh(path)
    _assert_same_mesh(mesh, from_pyvista(pv.read(str(path))))
    # Native, contiguous tensors regardless of the file's byte order.
    assert mesh.points.is_contiguous()
    assert mesh.point_data["idx"].dtype == torch.int32


def test_skip_arrays_in_every_section(tmp_path):
    path = _write_vtu(_tet_grid(), tmp_path / "tet.vtu")
    mesh = vtu_to_mesh(path, skip_arrays=["T", "idx", "meta"])
    assert set(mesh.point_data.keys()) == {"p", "U"}
    assert set(mesh.cell_data.keys()) == {"c"}
    assert set(mesh.global_data.keys()) == {"tag"}


def test_existing_zarr_subsample_path_reads_virtual_store(tmp_path):
    grid = _tet_grid()
    path = _write_vtu(grid, tmp_path / "tet.vtu")
    group = open_vtu_virtual(path)
    gen = torch.Generator().manual_seed(0)
    sub = _zarr_mesh_subsampled(group, n_cells=7, n_points=None, generator=gen)
    full = vtu_to_mesh(path)
    assert sub.cells.shape == (7, 4)
    assert sub.cell_data["T"].shape == (7, 9)
    # The subsample is a compaction of real rows: every subsampled point and
    # its data must appear in the full mesh at the same row.
    for p, u in zip(sub.points, sub.point_data["U"]):
        matches = (full.points == p).all(dim=1).nonzero().flatten()
        assert len(matches) == 1
        assert torch.equal(full.point_data["U"][matches[0]], u)


@pytest.mark.parametrize("chunk_rows", [7, 1000, 10**9])
def test_row_chunking_round_trip(tmp_path, chunk_rows):
    grid = _tet_grid()
    path = _write_vtu(grid, tmp_path / "tet.vtu")
    group = open_vtu_virtual(path, chunk_rows=chunk_rows)
    for name, n in [("cells", grid.n_cells), ("points", grid.n_points)]:
        assert group[name].nchunks == -(-n // min(chunk_rows, n)), name
    assert group["point_data"]["U"].nchunks == -(
        -grid.n_points // min(chunk_rows, grid.n_points)
    )
    _assert_same_mesh(
        vtu_to_mesh(path, chunk_rows=chunk_rows), from_pyvista(pv.read(str(path)))
    )
    sub = _zarr_mesh_subsampled(
        group, n_cells=5, n_points=None, generator=torch.Generator().manual_seed(0)
    )
    ref = _zarr_mesh_subsampled(
        open_vtu_virtual(path),
        n_cells=5,
        n_points=None,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.equal(sub.cells, ref.cells)
    assert torch.equal(sub.points, ref.points)
    assert torch.equal(sub.point_data["p"], ref.point_data["p"])


def test_row_chunking_exact_divisor(tmp_path):
    grid = _tet_grid()
    path = _write_vtu(grid, tmp_path / "tet.vtu")
    group = open_vtu_virtual(path, chunk_rows=grid.n_cells)
    assert group["cells"].nchunks == 1
    _assert_same_mesh(
        vtu_to_mesh(path, chunk_rows=grid.n_cells), from_pyvista(pv.read(str(path)))
    )


def test_row_chunking_falls_back_when_trailing_chunk_overruns_file():
    from physicsnemo.mesh.io.io_vtu_virtual import _ArrayInfo, _manifest_array

    arr = _ArrayInfo("x", np.dtype("<f8"), 10, 1, payload_offset=100, payload_nbytes=80)
    # Room for the full trailing chunk: 2 chunks of 7 rows -> 112 bytes from 100.
    assert _manifest_array(
        "file:///f", arr, chunk_rows=7, file_size=212
    ).manifest.shape_chunk_grid == (2,)
    # One byte short: fall back to a single chunk.
    assert _manifest_array(
        "file:///f", arr, chunk_rows=7, file_size=211
    ).manifest.shape_chunk_grid == (1,)


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_chunk_rows_must_be_positive_int(bad):
    with pytest.raises(ValueError, match="chunk_rows"):
        VTUParser(chunk_rows=bad)


def test_path_with_spaces_and_unicode(tmp_path):
    d = tmp_path / "dir with space" / "ünïcødé+%"
    d.mkdir(parents=True)
    path = _write_vtu(_tet_grid(), d / "file name.vtu")
    _assert_same_mesh(vtu_to_mesh(path), from_pyvista(pv.read(str(path))))


def test_path_with_fragment_char_is_rejected(tmp_path):
    path = _write_vtu(_tet_grid(), tmp_path / "a#b.vtu")
    with pytest.raises(ValueError, match="'#'"):
        open_vtu_virtual(path)


def test_vertex_only_grid_is_a_point_cloud_like_from_pyvista(tmp_path):
    grid = pv.PolyData(
        np.random.default_rng(1).random((10, 3))
    ).cast_to_unstructured_grid()
    grid.point_data["p"] = np.arange(10.0)
    grid.cell_data["c"] = np.arange(10.0)
    path = _write_vtu(grid, tmp_path / "verts.vtu")
    mesh = vtu_to_mesh(path)
    ref = from_pyvista(pv.read(str(path)))
    assert mesh.n_cells == ref.n_cells == 0
    assert mesh.cells.shape == ref.cells.shape
    _assert_same_mesh(mesh, ref)


def test_zero_cell_file(tmp_path):
    cloud = pv.UnstructuredGrid()
    cloud.points = np.random.default_rng(1).random((6, 3))
    cloud.point_data["p"] = np.arange(6.0)
    path = _write_vtu(cloud, tmp_path / "cloud.vtu")
    mesh = vtu_to_mesh(path)
    assert mesh.n_points == 6
    assert mesh.n_cells == 0
    assert torch.equal(mesh.point_data["p"], torch.arange(6.0, dtype=torch.float64))


def test_empty_grid(tmp_path):
    path = _write_vtu(pv.UnstructuredGrid(), tmp_path / "empty.vtu")
    mesh = vtu_to_mesh(path)
    assert mesh.points.shape == (0, 3)
    assert mesh.n_cells == 0


def test_header_only_reads_for_local_file(tmp_path, monkeypatch):
    """Opening reads the header, the types array and one offset; never the data."""
    import physicsnemo.mesh.io.io_vtu_virtual as mod

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

    def counting_open(*a, **k):
        return Counting(real_open(*a, **k))

    monkeypatch.setattr(mod, "open", counting_open, raising=False)
    open_vtu_virtual(path)
    # Exactly three reads: header block(s), the types array, one offset.
    n_header_blocks = len([r for r in reads if r == mod._HEADER_READ_BLOCK]) + 1
    assert len(reads) == n_header_blocks + 2, reads
    assert reads[-2] == header.types.payload_nbytes
    assert reads[-1] == header.offsets.dtype.itemsize


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
    with pytest.raises(NotImplementedError, match="Compressed|AppendedData"):
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


def test_pvtu_reports_file_type_not_encoding(tmp_path):
    path = tmp_path / "parallel.pvtu"
    path.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="PUnstructuredGrid" version="0.1">\n'
        '<PUnstructuredGrid GhostLevel="0"><Piece Source="p_0.vtu"/></PUnstructuredGrid>\n'
        "</VTKFile>\n"
    )
    with pytest.raises(ValueError, match="PUnstructuredGrid"):
        open_vtu_virtual(path)


def test_appended_tag_attribute_parsing_is_robust(tmp_path):
    path = _write_vtu(_tet_grid(), tmp_path / "tet.vtu")
    data = path.read_bytes()
    data = data.replace(
        b'<AppendedData encoding="raw">',
        b"<AppendedData my_attr=\"x_y\" encoding = 'raw' >",
        1,
    )
    path.write_bytes(data)
    _assert_same_mesh(
        vtu_to_mesh(path), from_pyvista(pv.read(str(tmp_path / "tet.vtu")))
    )


def test_slash_in_array_name_is_rejected(tmp_path):
    path = _edit_header(
        _write_vtu(_tet_grid(), tmp_path / "tet.vtu"), b'Name="p"', b'Name="U [m/s]"'
    )
    with pytest.raises(ValueError, match="path separators"):
        open_vtu_virtual(path)


def test_empty_array_name_is_rejected(tmp_path):
    path = _edit_header(
        _write_vtu(_tet_grid(), tmp_path / "tet.vtu"), b'Name="p"', b'Name=""'
    )
    with pytest.raises(ValueError, match="no Name"):
        open_vtu_virtual(path)


def test_duplicate_array_names_are_rejected(tmp_path):
    path = _edit_header(
        _write_vtu(_tet_grid(), tmp_path / "tet.vtu"), b'Name="c"', b'Name="T"'
    )
    with pytest.raises(ValueError, match="Duplicate.*CellData"):
        open_vtu_virtual(path)


def test_unknown_vtk_type_is_a_clear_error(tmp_path):
    path = _edit_header(
        _write_vtu(_tet_grid(), tmp_path / "tet.vtu"),
        b'type="Int32" Name="idx"',
        b'type="vtkIdType" Name="idx"',
    )
    with pytest.raises(NotImplementedError, match="vtkIdType"):
        open_vtu_virtual(path)


def test_out_of_range_connectivity_is_rejected(tmp_path):
    path = _write_vtu(_tet_grid(), tmp_path / "tet.vtu")
    header = _parse_vtu_header(open(path, "rb"))
    with open(path, "r+b") as f:
        f.seek(header.connectivity.payload_offset)
        f.write(np.array([10**6], dtype=header.connectivity.dtype).tobytes())
    with pytest.raises(ValueError, match="point ids"):
        vtu_to_mesh(path)


def test_parser_protocol_with_explicit_registry(tmp_path):
    import zarr
    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import LocalStore

    grid = _tet_grid()
    path = _write_vtu(grid, tmp_path / "tet.vtu")
    registry = ObjectStoreRegistry(
        {"file://" + str(tmp_path): LocalStore(prefix=str(tmp_path))}
    )
    store = VTUParser()("file://" + str(path), registry)
    group = zarr.open_group(store, mode="r")
    assert group["cells"].shape == (grid.n_cells, 4)
    np.testing.assert_array_equal(
        group["point_data"]["idx"][...], np.arange(grid.n_points)
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork start method unavailable")
def test_forked_worker_can_read_after_parent_opened(tmp_path):
    """Opening in the parent must not issue an obstore read, or forked children hang."""
    path = _write_vtu(_tet_grid(), tmp_path / "tet.vtu")
    script = textwrap.dedent(
        f"""
        import multiprocessing as mp, sys
        from physicsnemo.mesh.io.io_vtu_virtual import open_vtu_virtual
        def child(group):
            print(group["points"][...].shape, flush=True)
        if __name__ == "__main__":
            group = open_vtu_virtual({str(path)!r})  # parent opens, does not read
            p = mp.get_context("fork").Process(target=child, args=(group,))
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
    assert "(102, 3)" in proc.stdout or "(" in proc.stdout
