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

"""Memory-mapped :class:`~physicsnemo.mesh.Mesh` from a VTK XML UnstructuredGrid.

:func:`from_vtu` reads the XML header of a ``.vtu`` file (plus its cell-types
array, one byte per cell, and the last cell offset), then maps the file once
copy-on-write and returns a ``Mesh`` whose tensors are views into that map.
Nothing else is read until a tensor is touched, so the mesh behaves like one
from :meth:`Mesh.load <physicsnemo.mesh.Mesh.load>`: the memmap subsample path
of ``MeshReader`` slices it lazily, and ``MeshReader`` loads ``.vtu`` samples
through it when its glob pattern matches them.

It needs neither VTK nor PyVista.

Supported input
---------------
* ``<VTKFile type="UnstructuredGrid">`` with exactly one ``<Piece>``.
* Every ``DataArray`` with ``format="appended"``, ``<AppendedData encoding="raw">``
  and no ``compressor`` attribute. This is what the VTK XML writer produces
  with ``SetDataModeToAppended()``, ``EncodeAppendedDataOff()`` and
  ``SetCompressorTypeToNone()``. PyVista's ``save()`` default is zlib
  compressed and inline, which is not byte-addressable; use
  :func:`~physicsnemo.mesh.io.from_pyvista` for those files.
* A single simplex cell type (line, triangle or tetra), so the flat
  connectivity array is exactly the ``(n_cells, k)`` simplicial cells array a
  ``Mesh`` expects. A grid of only ``VERTEX`` cells loads as a point cloud
  with no cells, as ``from_pyvista`` does. Hexahedra, wedges, polyhedra and
  mixed meshes need the tetrahedralization ``from_pyvista`` does.

Caveats
-------
* **Big-endian files are copied**, not mapped: torch has no big-endian
  dtypes. Values are correct; laziness is lost.
* **Pickling materializes.** A plain tensor over a memmap pickles by value,
  so sending the mesh to a ``spawn`` DataLoader worker, over a queue or with
  ``torch.save`` copies it in full. ``Mesh.load`` tensors pickle by filename.
  Forked workers inherit the map and stay lazy.
* The map is copy-on-write (``MAP_PRIVATE``), as ``Mesh.load(mode="r")`` is:
  in-place edits never reach the file. A read-only map would make them crash.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch

from physicsnemo.mesh.mesh import Mesh

# VTK XML ``type`` attribute -> NumPy dtype character code (byte order added
# from the file's ``byte_order``).
_VTK_DTYPES: dict[str, str] = {
    "Int8": "i1",
    "UInt8": "u1",
    "Int16": "i2",
    "UInt16": "u2",
    "Int32": "i4",
    "UInt32": "u4",
    "Int64": "i8",
    "UInt64": "u8",
    "Float32": "f4",
    "Float64": "f8",
}

_VTK_VERTEX = 1

# VTK cell type id -> node count, for the simplex cell types only. A physicsnemo
# ``Mesh`` is simplicial: a ``(n_cells, k)`` cells array is a (k-1)-manifold, so
# only these can be viewed as-is. ``from_pyvista`` tetrahedralizes everything
# else. VERTEX is handled separately: like ``from_pyvista``, a vertex-only grid
# becomes a point cloud without cells.
_SIMPLEX_ARITY: dict[int, int] = {
    3: 2,  # LINE
    5: 3,  # TRIANGLE
    10: 4,  # TETRA
}

_HEADER_READ_BLOCK = 64 * 1024


@dataclass(frozen=True)
class _ArrayInfo:
    """One appended ``DataArray``: where its payload lives and what it holds."""

    name: str
    dtype: np.dtype
    n_tuples: int
    n_components: int
    payload_offset: int  # absolute byte offset of the first data byte
    payload_nbytes: int

    @property
    def shape(self) -> tuple[int, ...]:
        if self.n_components == 1:
            return (self.n_tuples,)
        return (self.n_tuples, self.n_components)

    def with_shape(self, n_tuples: int, n_components: int) -> _ArrayInfo:
        return _ArrayInfo(
            self.name,
            self.dtype,
            n_tuples,
            n_components,
            self.payload_offset,
            n_tuples * n_components * self.dtype.itemsize,
        )


@dataclass(frozen=True)
class _VTUHeader:
    n_points: int
    n_cells: int
    points: _ArrayInfo
    connectivity: _ArrayInfo | None
    offsets: _ArrayInfo | None
    types: _ArrayInfo | None
    point_data: list[_ArrayInfo]
    cell_data: list[_ArrayInfo]
    field_data: list[_ArrayInfo]


def _read_header_bytes(f: BinaryIO) -> tuple[bytes, int | None]:
    """Read the XML header and locate the appended payload.

    Returns the XML text before ``<AppendedData`` and the absolute offset of
    the first byte after the ``_`` marker, which VTK ``offset`` attributes are
    relative to. If the file has no ``<AppendedData>`` section the whole
    document is returned with ``None``, so the caller can report what the file
    is before complaining about how its data is stored.
    """
    buf = b""
    search_from = 0
    while True:
        chunk = f.read(_HEADER_READ_BLOCK)
        if not chunk:
            raise ValueError("Not a VTK XML file: no <AppendedData> or </VTKFile> tag.")
        buf += chunk
        tag = buf.find(b"<AppendedData", search_from)
        if tag < 0:
            end = buf.find(b"</VTKFile>")
            if end >= 0:
                return buf[: end + len(b"</VTKFile>")], None
            # Keep the tail in case a tag straddles the block boundary.
            search_from = max(0, len(buf) - 32)
            continue
        tag_end = buf.find(b">", tag)
        if tag_end < 0:
            continue
        marker = buf.find(b"_", tag_end)
        if marker < 0:
            continue
        head = buf[:tag]
        # Parse the tag's own attributes rather than regex-matching them, so
        # quoting style, spacing and extra attributes do not matter.
        tag_el = ET.fromstring(buf[tag : tag_end + 1] + b"</AppendedData>")  # noqa: S314
        encoding = tag_el.get("encoding")
        if encoding != "raw":
            raise NotImplementedError(
                f"AppendedData encoding={encoding!r} is not supported; only "
                'encoding="raw" is byte-addressable.'
            )
        return head, marker + 1


def _check_name(name: str | None, where: str) -> str:
    if not name:
        raise ValueError(f"A DataArray in {where} has no Name.")
    if "/" in name or "\\" in name:
        raise ValueError(
            f"DataArray {name!r} in {where}: '/' and '\\' are reserved as path "
            "separators by the zarr and memmap mesh formats. Rename the array first."
        )
    return name


def _parse_vtu_header(f: BinaryIO) -> _VTUHeader:
    head, data_start = _read_header_bytes(f)
    # The header is a few KB of the user's own mesh file with no DTD or
    # entities, so the stdlib parser is adequate here.
    root = ET.fromstring(head if data_start is None else head + b"</VTKFile>")  # noqa: S314
    if root.tag != "VTKFile" or root.get("type") != "UnstructuredGrid":
        raise ValueError(
            f'Expected <VTKFile type="UnstructuredGrid"> (a .vtu file), got '
            f"<{root.tag} type={root.get('type')!r}>."
        )
    if root.get("compressor"):
        raise NotImplementedError(
            f"Compressed appended data ({root.get('compressor')}) is not "
            "byte-addressable. Write the file with compression off "
            "(vtkXMLWriter.SetCompressorTypeToNone), or load it with from_pyvista."
        )
    if data_start is None:
        raise NotImplementedError(
            "VTU file has no <AppendedData> section (inline ascii/binary "
            'DataArrays); only format="appended" encoding="raw" is supported. '
            "Load it with from_pyvista instead."
        )
    endian = {"LittleEndian": "<", "BigEndian": ">"}[
        root.get("byte_order", "LittleEndian")
    ]
    header_size = {"UInt32": 4, "UInt64": 8}[root.get("header_type", "UInt32")]

    grid = root.find("UnstructuredGrid")
    pieces = grid.findall("Piece")
    if len(pieces) != 1:
        raise NotImplementedError(f"Expected exactly one <Piece>, got {len(pieces)}.")
    piece = pieces[0]
    n_points = int(piece.get("NumberOfPoints"))
    n_cells = int(piece.get("NumberOfCells"))

    def info(el: ET.Element, n_tuples: int, where: str) -> _ArrayInfo:
        name = _check_name(el.get("Name"), where)
        if el.get("format") != "appended":
            raise NotImplementedError(
                f"DataArray {name!r} has format={el.get('format')!r}; "
                'only format="appended" is supported.'
            )
        vtk_type = el.get("type")
        if vtk_type not in _VTK_DTYPES:
            raise NotImplementedError(
                f"DataArray {name!r} has type={vtk_type!r}, which this reader does "
                "not map to a fixed NumPy dtype."
            )
        dtype = np.dtype(endian + _VTK_DTYPES[vtk_type])
        n_comp = int(el.get("NumberOfComponents", "1"))
        if "NumberOfTuples" in el.attrib:
            n_tuples = int(el.get("NumberOfTuples"))
        return _ArrayInfo(
            name=name,
            dtype=dtype,
            n_tuples=n_tuples,
            n_components=n_comp,
            payload_offset=data_start + int(el.get("offset")) + header_size,
            payload_nbytes=n_tuples * n_comp * dtype.itemsize,
        )

    def arrays(
        parent: ET.Element | None, n_tuples: int, where: str
    ) -> list[_ArrayInfo]:
        if parent is None:
            return []
        out = [info(el, n_tuples, where) for el in parent.findall("DataArray")]
        names = [a.name for a in out]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate DataArray names in {where}: {dupes}.")
        return out

    points_section = piece.find("Points")
    points_els = [] if points_section is None else points_section.findall("DataArray")
    if len(points_els) == 1:
        points = info(points_els[0], n_points, "Points")
    elif n_points == 0 and not points_els:
        points = _ArrayInfo("Points", np.dtype("<f4"), 0, 3, data_start, 0)
    else:
        raise ValueError(f"Expected one <Points> DataArray, found {len(points_els)}.")

    cells_section = piece.find("Cells")
    cell_arrays = (
        {}
        if cells_section is None
        else {el.get("Name"): el for el in cells_section.findall("DataArray")}
    )
    connectivity = offsets = types = None
    if n_cells > 0:
        missing = {"connectivity", "offsets", "types"} - set(cell_arrays)
        if missing:
            raise ValueError(f"<Cells> lacks the arrays {sorted(missing)}.")
        types = info(cell_arrays["types"], n_cells, "Cells")
        offsets = info(cell_arrays["offsets"], n_cells, "Cells")
        # The connectivity length is only known once the cell arity is, so
        # size it for one node per cell here; _resolve_cells fixes it up.
        connectivity = info(cell_arrays["connectivity"], n_cells, "Cells")

    return _VTUHeader(
        n_points=n_points,
        n_cells=n_cells,
        points=points,
        connectivity=connectivity,
        offsets=offsets,
        types=types,
        point_data=arrays(piece.find("PointData"), n_points, "PointData"),
        cell_data=arrays(piece.find("CellData"), n_cells, "CellData"),
        field_data=arrays(grid.find("FieldData"), 0, "FieldData"),
    )


def _read_payload(f: BinaryIO, arr: _ArrayInfo) -> np.ndarray:
    f.seek(arr.payload_offset)
    raw = f.read(arr.payload_nbytes)
    if len(raw) != arr.payload_nbytes:
        raise ValueError(f"Short read of DataArray {arr.name!r}.")
    return np.frombuffer(raw, dtype=arr.dtype).reshape(arr.shape)


def _read_last_scalar(f: BinaryIO, arr: _ArrayInfo) -> int:
    """Read only the last element of a 1-D integer array."""
    f.seek(arr.payload_offset + arr.payload_nbytes - arr.dtype.itemsize)
    raw = f.read(arr.dtype.itemsize)
    if len(raw) != arr.dtype.itemsize:
        raise ValueError(f"Short read of DataArray {arr.name!r}.")
    return int(np.frombuffer(raw, dtype=arr.dtype)[0])


def _resolve_cells(f: BinaryIO, header: _VTUHeader) -> _ArrayInfo | None:
    """The connectivity array as ``(n_cells, k)``, or None for a point cloud.

    Reads the types array (one byte per cell) to find the single simplex
    arity and the last offset to confirm the connectivity length; the
    connectivity itself is not read.
    """
    if header.n_cells == 0:
        return None
    types = np.unique(_read_payload(f, header.types))
    if len(types) == 1 and int(types[0]) == _VTK_VERTEX:
        return None  # Vertex cells carry no topology: a point cloud.
    if len(types) != 1 or int(types[0]) not in _SIMPLEX_ARITY:
        raise NotImplementedError(
            f"VTK cell types {types.tolist()} cannot be viewed as a physicsnemo "
            "Mesh without rewriting: a single simplex cell type is needed "
            "(LINE=3, TRIANGLE=5 or TETRA=10). Load it with from_pyvista, which "
            "tetrahedralizes, instead."
        )
    k = _SIMPLEX_ARITY[int(types[0])]
    last_offset = _read_last_scalar(f, header.offsets)
    if last_offset != header.n_cells * k:
        raise ValueError(
            f"Connectivity length {last_offset} != n_cells * k = "
            f"{header.n_cells * k}; cells are not all {k}-node."
        )
    return header.connectivity.with_shape(header.n_cells, k)


def _map_view(mm: np.memmap, arr: _ArrayInfo) -> torch.Tensor:
    """A tensor over the array's bytes in the file map; copied if big-endian."""
    view = mm[arr.payload_offset : arr.payload_offset + arr.payload_nbytes]
    view = view.view(arr.dtype).reshape(arr.shape)
    if view.dtype.byteorder == ">":
        view = view.astype(view.dtype.newbyteorder("<"))  # torch has no BE dtypes
    return torch.from_numpy(view)


def from_vtu(
    path: str | Path,
    *,
    skip_arrays: Iterable[str] | None = None,
    device: torch.device | str | None = None,
) -> Mesh:
    """Read a ``.vtu`` file as a lazily memory-mapped :class:`~physicsnemo.mesh.Mesh`.

    Parameters
    ----------
    path : str or Path
        An uncompressed, appended-raw ``.vtu`` file of one simplex cell type
        (see the module docstring for the exact conditions).
    skip_arrays : iterable of str, optional
        Names of point, cell or field arrays to leave out. A name present in
        more than one of those sections is skipped in all of them.
    device : torch.device or str, optional
        If given, every tensor is moved there, which materializes the mesh.
        Leave it unset to keep the tensors as lazy views of the file.

    Returns
    -------
    Mesh
        ``points``, ``cells``, ``point_data``, ``cell_data`` and
        ``global_data`` as views into one copy-on-write map of the file, with
        the dtypes the file stores. Point ids in the connectivity are checked
        to be in range, which reads the connectivity.

    Raises
    ------
    NotImplementedError
        For compressed, inline or base64 data, more than one piece, or a cell
        type that is not a single simplex type. ``from_pyvista`` handles all
        of these.
    ValueError
        For a file that is not an UnstructuredGrid, a malformed header, array
        names that are empty, duplicated or contain ``/``, or point ids out of
        range.

    Examples
    --------
    >>> from physicsnemo.mesh.io import from_vtu
    >>> mesh = from_vtu("run0.vtu")  # doctest: +SKIP
    >>> mesh.cells.shape  # doctest: +SKIP
    torch.Size([384, 4])
    """
    path = Path(path)
    skip = frozenset(skip_arrays or ())
    with open(path, "rb") as f:
        header = _parse_vtu_header(f)
        connectivity = _resolve_cells(f, header)

    # mode="c" is MAP_PRIVATE copy-on-write, like Mesh.load(mode="r"): in-place
    # edits land in private pages and never reach the file. mode="r" would
    # make torch.from_numpy warn and any in-place write a SIGBUS.
    mm = np.memmap(path, mode="c")

    def group(infos: list[_ArrayInfo]) -> dict[str, torch.Tensor]:
        return {a.name: _map_view(mm, a) for a in infos if a.name not in skip}

    cells = None if connectivity is None else _map_view(mm, connectivity)
    if cells is not None and cells.numel():
        lo, hi = int(cells.min()), int(cells.max())
        if lo < 0 or hi >= header.n_points:
            raise ValueError(
                f"Connectivity of {path} references point ids in [{lo}, {hi}] but "
                f"the mesh has {header.n_points} points."
            )
    mesh = Mesh(
        points=_map_view(mm, header.points),
        cells=cells,
        point_data=group(header.point_data),
        # A vertex-only grid has no cells for cell data to attach to; drop it
        # as from_pyvista does for point clouds.
        cell_data=group(header.cell_data) if cells is not None else None,
        global_data=group(header.field_data),
    )
    return mesh if device is None else mesh.to(device)


__all__ = ["from_vtu"]
