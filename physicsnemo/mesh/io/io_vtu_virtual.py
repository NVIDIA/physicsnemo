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

"""Virtual zarr view of a VTK XML UnstructuredGrid (``.vtu``) file.

**Prototype.** :class:`VTUParser` implements the VirtualiZarr parser protocol:
it reads only the XML header of a ``.vtu`` file and maps each appended binary
array to a byte range, producing a :class:`~virtualizarr.manifests.ManifestStore`
laid out exactly as :func:`physicsnemo.mesh.io.to_zarr` writes a ``Mesh``::

    /                     attrs: physicsnemo_mesh_type="Mesh"
    ├── points            (n_points, 3)
    ├── cells             (n_cells, k)     <- connectivity bytes, reshaped
    ├── point_data/<name> (n_points[, c])
    ├── cell_data/<name>  (n_cells[, c])
    └── global_data/<name>

Because the layout matches, the existing zarr readers (``_mesh_from_group``,
``_zarr_mesh_subsampled`` and hence ``MeshReader``) read the ``.vtu`` through
zarr without converting it, and without VTK or PyVista at read time.

Supported input
---------------
* ``<VTKFile type="UnstructuredGrid">`` with exactly one ``<Piece>``.
* Every ``DataArray`` with ``format="appended"``, ``<AppendedData encoding="raw">``
  and no ``compressor`` attribute. This is what the VTK XML writer produces
  with ``SetDataModeToAppended()``, ``EncodeAppendedDataOff()`` and
  ``SetCompressorTypeToNone()``. PyVista's ``save()`` default is zlib
  compressed and is **not** supported yet: VTK's block-compression header is
  not a standard zarr codec and would need a custom ``BytesBytesCodec``.
* A single simplex cell type (vertex, line, triangle or tetra), so the flat
  connectivity array is exactly the ``(n_cells, k)`` simplicial cells array a
  ``Mesh`` expects. Hexahedra, wedges, polyhedra and mixed meshes need the
  tetrahedralization ``from_pyvista`` does, which no byte-range view can.

Each VTK array is one contiguous blob, so each zarr array has one chunk: any
partial read fetches the whole array. The win over PyVista is therefore not
speed on local disk but (a) no VTK dependency in DataLoader workers, (b) byte-
range access on object storage via obstore, and (c) reuse of the zarr reader
and subsampling code paths.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import zarr
    from obspec_utils.registry import ObjectStoreRegistry
    from virtualizarr.manifests import ManifestStore

from physicsnemo.mesh.io.io_zarr import _TYPE_ATTR

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

# VTK cell type id -> node count, for the simplex cell types only. A physicsnemo
# ``Mesh`` is simplicial: a ``(n_cells, k)`` cells array is a (k-1)-manifold, so
# only these can be viewed as-is. ``from_pyvista`` tetrahedralizes everything
# else, which a byte-range view cannot do.
_SIMPLEX_ARITY: dict[int, int] = {
    1: 1,  # VERTEX
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


@dataclass(frozen=True)
class _VTUHeader:
    n_points: int
    n_cells: int
    points: _ArrayInfo
    connectivity: _ArrayInfo
    offsets: _ArrayInfo
    types: _ArrayInfo
    point_data: list[_ArrayInfo]
    cell_data: list[_ArrayInfo]
    field_data: list[_ArrayInfo]


def _read_header_bytes(f) -> tuple[bytes, int]:
    """Read the XML header up to ``<AppendedData ...>_`` and locate the payload.

    Returns the XML text before ``<AppendedData`` and the absolute offset of
    the first byte after the ``_`` marker, which VTK ``offset`` attributes are
    relative to.
    """
    buf = b""
    while True:
        chunk = f.read(_HEADER_READ_BLOCK)
        if not chunk:
            raise ValueError("Not a VTK XML appended-data file: no <AppendedData> tag.")
        buf += chunk
        tag = buf.find(b"<AppendedData")
        if tag < 0:
            if b"</VTKFile>" in buf:
                raise NotImplementedError(
                    "VTU file has no <AppendedData> section (inline ascii/binary "
                    'DataArrays); only format="appended" encoding="raw" is supported.'
                )
            continue
        marker = buf.find(b"_", tag)
        if marker < 0:
            continue
        head = buf[:tag]
        m = re.search(rb'encoding="(\w+)"', buf[tag:marker])
        if m is None or m.group(1) != b"raw":
            raise NotImplementedError(
                f"AppendedData encoding={m.group(1).decode() if m else '?'!r} is not "
                'supported; only encoding="raw" is byte-addressable.'
            )
        return head, marker + 1


def _parse_vtu_header(f) -> _VTUHeader:
    head, data_start = _read_header_bytes(f)
    # The header is a few KB of the user's own mesh file with no DTD or
    # entities, so the stdlib parser is adequate here.
    root = ET.fromstring(head + b"</VTKFile>")  # noqa: S314
    if root.tag != "VTKFile" or root.get("type") != "UnstructuredGrid":
        raise ValueError(
            f'Expected <VTKFile type="UnstructuredGrid">, got '
            f"<{root.tag} type={root.get('type')!r}>."
        )
    if root.get("compressor"):
        raise NotImplementedError(
            f"Compressed appended data ({root.get('compressor')}) is not supported: "
            "VTK's block-compression header is not a standard zarr codec. Write the "
            "file with compression off (vtkXMLWriter.SetCompressorTypeToNone)."
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

    def info(el: ET.Element, n_tuples: int) -> _ArrayInfo:
        if el.get("format") != "appended":
            raise NotImplementedError(
                f"DataArray {el.get('Name')!r} has format={el.get('format')!r}; "
                'only format="appended" is supported.'
            )
        dtype = np.dtype(endian + _VTK_DTYPES[el.get("type")])
        n_comp = int(el.get("NumberOfComponents", "1"))
        if "NumberOfTuples" in el.attrib:
            n_tuples = int(el.get("NumberOfTuples"))
        nbytes = n_tuples * n_comp * dtype.itemsize
        return _ArrayInfo(
            name=el.get("Name"),
            dtype=dtype,
            n_tuples=n_tuples,
            n_components=n_comp,
            payload_offset=data_start + int(el.get("offset")) + header_size,
            payload_nbytes=nbytes,
        )

    def arrays(parent: ET.Element | None, n_tuples: int) -> list[_ArrayInfo]:
        if parent is None:
            return []
        return [info(el, n_tuples) for el in parent.findall("DataArray")]

    cells = {el.get("Name"): el for el in piece.find("Cells").findall("DataArray")}
    (points_el,) = piece.find("Points").findall("DataArray")
    types = info(cells["types"], n_cells)
    offsets = info(cells["offsets"], n_cells)
    # The connectivity length is only known from the offsets array; it is
    # fixed up in ``_build_store`` after the cell types are read. Size it for
    # one node per cell here, which is the minimum.
    connectivity = info(cells["connectivity"], n_cells)
    return _VTUHeader(
        n_points=n_points,
        n_cells=n_cells,
        points=info(points_el, n_points),
        connectivity=connectivity,
        offsets=offsets,
        types=types,
        point_data=arrays(piece.find("PointData"), n_points),
        cell_data=arrays(piece.find("CellData"), n_cells),
        field_data=arrays(grid.find("FieldData"), 0),
    )


def _read_payload(f, arr: _ArrayInfo) -> np.ndarray:
    f.seek(arr.payload_offset)
    raw = f.read(arr.payload_nbytes)
    if len(raw) != arr.payload_nbytes:
        raise ValueError(f"Short read of DataArray {arr.name!r}.")
    return np.frombuffer(raw, dtype=arr.dtype).reshape(arr.shape)


def _manifest_array(
    url: str,
    arr: _ArrayInfo,
    shape: tuple[int, ...] | None = None,
    chunk_rows: int | None = None,
    file_size: int | None = None,
):
    """ManifestArray over the array's payload bytes.

    The payload is row-major and contiguous, so any run of leading-dimension
    rows is itself a contiguous byte range. With ``chunk_rows`` the array is
    declared as that many rows per chunk, each chunk pointing at its slice of
    the same bytes, so a row-window read fetches only the chunks it touches.
    Without it the whole array is one chunk.

    Zarr's bytes codec expects every stored chunk, including the trailing
    partial one, to hold a full ``chunk_shape`` of bytes. The trailing chunk
    is therefore declared over a full-size range that runs past the array
    into whatever follows it in the file; zarr discards the rows outside the
    array shape. If the file is too short for that, the array falls back to
    one chunk.
    """
    from virtualizarr.manifests import ChunkManifest, ManifestArray
    from virtualizarr.manifests.utils import create_v3_array_metadata

    shape = arr.shape if shape is None else shape
    if math.prod(shape) * arr.dtype.itemsize != arr.payload_nbytes:
        raise ValueError(
            f"DataArray {arr.name!r}: shape {shape} does not cover {arr.payload_nbytes} bytes."
        )
    n_rows = shape[0]
    rows_per_chunk = n_rows if chunk_rows is None else min(chunk_rows, n_rows)
    row_nbytes = arr.payload_nbytes // max(n_rows, 1)
    if (
        rows_per_chunk < n_rows
        and n_rows % rows_per_chunk
        and (
            file_size is None
            or arr.payload_offset
            + (-(-n_rows // rows_per_chunk)) * rows_per_chunk * row_nbytes
            > file_size
        )
    ):
        rows_per_chunk = n_rows  # Trailing chunk would overrun the file.
    chunk_shape = (max(rows_per_chunk, 1),) + tuple(max(s, 1) for s in shape[1:])
    metadata = create_v3_array_metadata(
        shape=shape,
        data_type=arr.dtype,
        # Zarr forbids a zero-length chunk edge; an empty array keeps a
        # positive chunk shape over an empty chunk grid.
        chunk_shape=chunk_shape,
    )
    if 0 in shape:
        manifest = ChunkManifest(entries={}, shape=(0,) * len(shape))
    else:
        n_chunks = -(-n_rows // rows_per_chunk)
        grid = (n_chunks,) + (1,) * (len(shape) - 1)
        starts = np.arange(n_chunks, dtype=np.uint64) * np.uint64(rows_per_chunk)
        # Every chunk, the trailing one included, is declared full-size.
        lengths = np.full(n_chunks, rows_per_chunk * row_nbytes, dtype=np.uint64)
        manifest = ChunkManifest.from_arrays(
            paths=np.full(grid, url, dtype=np.dtypes.StringDType()),
            offsets=(
                np.uint64(arr.payload_offset) + starts * np.uint64(row_nbytes)
            ).reshape(grid),
            lengths=lengths.reshape(grid),
            validate_paths=False,
        )
    return ManifestArray(metadata=metadata, chunkmanifest=manifest)


def _tensordict_attrs(batch_size: list[int]) -> dict[str, Any]:
    return {"__tensordict__": {"batch_size": batch_size, "version": 1}}


class VTUParser:
    """VirtualiZarr parser: ``.vtu`` file -> ManifestStore in physicsnemo's Mesh layout.

    Parameters
    ----------
    skip_arrays : iterable of str, optional
        Names of point/cell/field arrays to leave out of the store.

    Examples
    --------
    >>> from obspec_utils.registry import ObjectStoreRegistry  # doctest: +SKIP
    >>> from obstore.store import LocalStore  # doctest: +SKIP
    >>> registry = ObjectStoreRegistry({"file:///data": LocalStore(prefix="/data")})  # doctest: +SKIP
    >>> store = VTUParser()("file:///data/run0.vtu", registry)  # doctest: +SKIP
    >>> import zarr; group = zarr.open_group(store, mode="r")  # doctest: +SKIP
    >>> group["cells"].shape  # doctest: +SKIP
    (384, 4)
    """

    def __init__(self, skip_arrays=None, chunk_rows: int | None = None):
        self.skip_arrays = frozenset(skip_arrays or ())
        self.chunk_rows = chunk_rows

    def __call__(self, url: str, registry: "ObjectStoreRegistry") -> "ManifestStore":
        from obspec_utils.readers import BlockStoreReader
        from virtualizarr.manifests import ManifestGroup, ManifestStore

        store, path_in_store = registry.resolve(url)
        reader = BlockStoreReader(store=store, path=path_in_store)
        header = _parse_vtu_header(reader)
        file_size = reader.seek(0, 2) if self.chunk_rows else None

        # Cell topology: read the (small) ``types`` array to find the single
        # fixed arity, then view the connectivity blob as (n_cells, k).
        if header.n_cells > 0:
            types = np.unique(_read_payload(reader, header.types))
            if len(types) != 1 or int(types[0]) not in _SIMPLEX_ARITY:
                raise NotImplementedError(
                    f"VTK cell types {types.tolist()} cannot be viewed as a physicsnemo "
                    "Mesh without rewriting: a virtual store needs a single simplex "
                    "cell type (VERTEX=1, LINE=3, TRIANGLE=5 or TETRA=10). Use "
                    "from_pyvista (which tetrahedralizes) and to_zarr instead."
                )
            k = _SIMPLEX_ARITY[int(types[0])]
            conn = header.connectivity
            connectivity = _ArrayInfo(
                name=conn.name,
                dtype=conn.dtype,
                n_tuples=header.n_cells,
                n_components=k,
                payload_offset=conn.payload_offset,
                payload_nbytes=header.n_cells * k * conn.dtype.itemsize,
            )
            # Consistency check against the offsets array: last offset must be
            # the total connectivity length, i.e. k * n_cells.
            offsets = _read_payload(reader, header.offsets)
            if int(offsets[-1]) != header.n_cells * k:
                raise ValueError(
                    f"Connectivity length {int(offsets[-1])} != n_cells * k = "
                    f"{header.n_cells * k}; cells are not all {k}-node."
                )
            cells_shape = (header.n_cells, k)
        else:
            connectivity = header.connectivity
            cells_shape = (0, 1)

        def group(infos: list[_ArrayInfo], batch: list[int]) -> ManifestGroup:
            return ManifestGroup(
                arrays={
                    a.name: _manifest_array(
                        url, a, chunk_rows=self.chunk_rows, file_size=file_size
                    )
                    for a in infos
                    if a.name not in self.skip_arrays
                },
                attributes=_tensordict_attrs(batch),
            )

        root = ManifestGroup(
            arrays={
                "points": _manifest_array(
                    url,
                    header.points,
                    chunk_rows=self.chunk_rows,
                    file_size=file_size,
                ),
                "cells": _manifest_array(
                    url,
                    connectivity,
                    cells_shape,
                    chunk_rows=self.chunk_rows,
                    file_size=file_size,
                ),
            },
            groups={
                "point_data": group(header.point_data, [header.n_points]),
                "cell_data": group(header.cell_data, [header.n_cells]),
                "global_data": group(header.field_data, []),
            },
            attributes={**_tensordict_attrs([]), _TYPE_ATTR: "Mesh"},
        )
        return ManifestStore(group=root, registry=registry)


def open_vtu_virtual(path: str | Path, **parser_kwargs) -> "zarr.Group":
    """Open a local ``.vtu`` file as a read-only zarr group in physicsnemo's layout.

    Convenience wrapper around :class:`VTUParser` for ``file://`` paths. For
    object storage build the registry yourself and call the parser.
    """
    import zarr
    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import LocalStore

    path = Path(path).resolve()
    url = path.as_uri()
    registry = ObjectStoreRegistry(
        {Path(path.parent).as_uri(): LocalStore(prefix=str(path.parent))}
    )
    store = VTUParser(**parser_kwargs)(url, registry)
    return zarr.open_group(store, mode="r")


def vtu_to_mesh(path: str | Path, device=None, **parser_kwargs):
    """Read a ``.vtu`` file into a :class:`~physicsnemo.mesh.Mesh` via its virtual zarr view."""
    from physicsnemo.mesh.io.io_zarr import _mesh_from_group

    return _mesh_from_group(open_vtu_virtual(path, **parser_kwargs), device)


__all__ = ["VTUParser", "open_vtu_virtual", "vtu_to_mesh"]
