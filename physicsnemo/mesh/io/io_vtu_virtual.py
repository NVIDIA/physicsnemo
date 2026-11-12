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
it reads the XML header of a ``.vtu`` file, plus the cell-types array (one
byte per cell) and the last cell offset to validate the topology, and maps
each appended binary array to a byte range. The result is a
:class:`~virtualizarr.manifests.ManifestStore` laid out exactly as
:func:`physicsnemo.mesh.io.to_zarr` writes a ``Mesh``::

    /                     attrs: physicsnemo_mesh_type="Mesh"
    ├── points            (n_points, 3)
    ├── cells             (n_cells, k)     <- connectivity bytes, reshaped
    ├── point_data/<name> (n_points[, c])
    ├── cell_data/<name>  (n_cells[, c])
    └── global_data/<name>

Because the layout matches, the group-level zarr readers in
``physicsnemo.mesh.io.io_zarr`` and ``physicsnemo.datapipes.readers.mesh``
(``_mesh_from_group``, ``_zarr_mesh_subsampled``) read the ``.vtu`` through
zarr without converting it, and without VTK or PyVista at read time.
``MeshReader`` itself takes filesystem paths and cannot consume a
``ManifestStore`` yet.

Supported input
---------------
* ``<VTKFile type="UnstructuredGrid">`` with exactly one ``<Piece>``.
* Every ``DataArray`` with ``format="appended"``, ``<AppendedData encoding="raw">``
  and no ``compressor`` attribute. This is what the VTK XML writer produces
  with ``SetDataModeToAppended()``, ``EncodeAppendedDataOff()`` and
  ``SetCompressorTypeToNone()``. PyVista's ``save()`` default is zlib
  compressed and is **not** supported yet: VTK's block-compression header is
  not a standard zarr codec and would need a custom ``BytesBytesCodec``.
* A single simplex cell type (line, triangle or tetra), so the flat
  connectivity array is exactly the ``(n_cells, k)`` simplicial cells array a
  ``Mesh`` expects. A grid of only ``VERTEX`` cells loads as a point cloud
  with no cells, as :func:`~physicsnemo.mesh.io.from_pyvista` does. Hexahedra,
  wedges, polyhedra and mixed meshes need the tetrahedralization
  ``from_pyvista`` does, which no byte-range view can.

Chunking
--------
Each VTK array is one contiguous blob. By default each zarr array is one
chunk, so any partial read fetches the whole array. With
``VTUParser(chunk_rows=N)`` every array is declared as ``N``-row chunks over
sub-ranges of the same bytes, so a row-window read fetches only the chunks it
touches; the cost is per-chunk overhead on full reads. See
:func:`_manifest_array` for the trailing-chunk detail.

Process forking
---------------
``obstore`` starts a background runtime on its first read. A process that has
read through it cannot ``fork()`` children that read again: the child hangs.
For ``file://`` URLs this module therefore parses the header with plain file
I/O, so a parent that only opens the store never touches ``obstore`` and
forked DataLoader workers work. For remote URLs, or if the parent reads any
data itself, use the ``spawn`` or ``forkserver`` start method.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.mesh.io.io_zarr import _TYPE_ATTR

if TYPE_CHECKING:
    import zarr
    from obspec_utils.registry import ObjectStoreRegistry
    from virtualizarr.manifests import ManifestStore

_HINT = "Install with: pip install virtualizarr obstore"
_manifests = OptionalImport("virtualizarr.manifests", _HINT)
_manifest_utils = OptionalImport("virtualizarr.manifests.utils", _HINT)
_registry_mod = OptionalImport("obspec_utils.registry", _HINT)
_readers_mod = OptionalImport("obspec_utils.readers", _HINT)
_obstore_store = OptionalImport("obstore.store", _HINT)
_zarr = OptionalImport("zarr")

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
# else, which a byte-range view cannot do. VERTEX is handled separately: like
# ``from_pyvista``, a vertex-only grid becomes a point cloud without cells.
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


def _read_header_bytes(f) -> tuple[bytes, int | None]:
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
        raise ValueError(f"A DataArray in {where} has no Name; zarr needs one.")
    if "/" in name or "\\" in name:
        raise ValueError(
            f"DataArray {name!r} in {where}: '/' and '\\' are zarr path separators "
            "and would change the key. Rename the array first."
        )
    return name


def _parse_vtu_header(f) -> _VTUHeader:
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
            f"Compressed appended data ({root.get('compressor')}) is not supported: "
            "VTK's block-compression header is not a standard zarr codec. Write the "
            "file with compression off (vtkXMLWriter.SetCompressorTypeToNone)."
        )
    if data_start is None:
        raise NotImplementedError(
            "VTU file has no <AppendedData> section (inline ascii/binary "
            'DataArrays); only format="appended" encoding="raw" is supported.'
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
                f"DataArray {name!r} has type={vtk_type!r}, which this parser does "
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

    def empty(name: str, dtype: str, n_comp: int) -> _ArrayInfo:
        return _ArrayInfo(name, np.dtype(dtype), 0, n_comp, data_start, 0)

    points_section = piece.find("Points")
    points_els = [] if points_section is None else points_section.findall("DataArray")
    if len(points_els) == 1:
        points = info(points_els[0], n_points, "Points")
    elif n_points == 0 and not points_els:
        points = empty("Points", "<f4", 3)
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
        # size it for one node per cell here; VTUParser fixes it up.
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


def _read_payload(f, arr: _ArrayInfo) -> np.ndarray:
    f.seek(arr.payload_offset)
    raw = f.read(arr.payload_nbytes)
    if len(raw) != arr.payload_nbytes:
        raise ValueError(f"Short read of DataArray {arr.name!r}.")
    return np.frombuffer(raw, dtype=arr.dtype).reshape(arr.shape)


def _read_last_scalar(f, arr: _ArrayInfo) -> int:
    """Read only the last element of a 1-D integer array."""
    f.seek(arr.payload_offset + arr.payload_nbytes - arr.dtype.itemsize)
    raw = f.read(arr.dtype.itemsize)
    if len(raw) != arr.dtype.itemsize:
        raise ValueError(f"Short read of DataArray {arr.name!r}.")
    return int(np.frombuffer(raw, dtype=arr.dtype)[0])


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
    ChunkManifest, ManifestArray = _manifests.ChunkManifest, _manifests.ManifestArray
    create_v3_array_metadata = _manifest_utils.create_v3_array_metadata

    if chunk_rows is not None and chunk_rows < 1:
        raise ValueError(
            f"chunk_rows must be a positive integer or None, got {chunk_rows}."
        )
    shape = arr.shape if shape is None else shape
    if math.prod(shape) * arr.dtype.itemsize != arr.payload_nbytes:
        raise ValueError(
            f"DataArray {arr.name!r}: shape {shape} does not cover {arr.payload_nbytes} bytes."
        )
    n_rows = shape[0]
    rows_per_chunk = n_rows if chunk_rows is None else min(chunk_rows, n_rows)
    row_nbytes = arr.payload_nbytes // max(n_rows, 1)
    if (
        0 < rows_per_chunk < n_rows
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


def _local_path(url: str) -> str | None:
    """The filesystem path of a ``file://`` URL, or None for other schemes."""
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    return url2pathname(parsed.path)


class _HeaderSource:
    """File-like access to the header bytes, local or through the registry.

    A ``file://`` URL is opened with plain file I/O so the calling process
    never issues an ``obstore`` read (see *Process forking* in the module
    docstring). Anything else goes through the registry's store.
    """

    def __init__(self, url: str, registry: "ObjectStoreRegistry"):
        local = _local_path(url)
        if local is not None:
            self._f = open(local, "rb")  # noqa: SIM115
            self.size = os.fstat(self._f.fileno()).st_size
        else:
            store, path_in_store = registry.resolve(url)
            self._f = _readers_mod.BlockStoreReader(store=store, path=path_in_store)
            self.size = self._f.seek(0, 2)
            self._f.seek(0)

    def __enter__(self):
        return self._f

    def __exit__(self, *exc):
        close = getattr(self._f, "close", None)
        if close is not None:
            close()


class VTUParser:
    """VirtualiZarr parser: ``.vtu`` file -> ManifestStore in physicsnemo's Mesh layout.

    Parameters
    ----------
    skip_arrays : iterable of str, optional
        Names of point, cell or field arrays to leave out of the store. A name
        present in more than one of those sections is skipped in all of them.
    chunk_rows : int, optional
        Rows per zarr chunk for every array. ``None`` (default) makes each
        array a single chunk, which is cheapest for full reads. A value in the
        tens of thousands makes row-window reads, such as a cell subsample,
        fetch only the rows they touch.

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

    def __init__(
        self,
        skip_arrays: Iterable[str] | None = None,
        chunk_rows: int | None = None,
    ):
        if chunk_rows is not None and (
            not isinstance(chunk_rows, int) or chunk_rows < 1
        ):
            raise ValueError(
                f"chunk_rows must be a positive integer or None, got {chunk_rows!r}."
            )
        self.skip_arrays = frozenset(skip_arrays or ())
        self.chunk_rows = chunk_rows

    def __call__(self, url: str, registry: "ObjectStoreRegistry") -> "ManifestStore":
        ManifestGroup, ManifestStore = (
            _manifests.ManifestGroup,
            _manifests.ManifestStore,
        )

        source = _HeaderSource(url, registry)
        with source as reader:
            header = _parse_vtu_header(reader)
            file_size = source.size if self.chunk_rows is not None else None

            # Cell topology: read the types array (one byte per cell) to find
            # the single arity, and only the last offset to confirm the
            # connectivity length. Then view the connectivity blob as
            # (n_cells, k) without reading it.
            connectivity, cells_shape = None, (0, 1)
            if header.n_cells > 0:
                types = np.unique(_read_payload(reader, header.types))
                if len(types) == 1 and int(types[0]) == _VTK_VERTEX:
                    pass  # Vertex cells carry no topology: a point cloud.
                elif len(types) != 1 or int(types[0]) not in _SIMPLEX_ARITY:
                    raise NotImplementedError(
                        f"VTK cell types {types.tolist()} cannot be viewed as a "
                        "physicsnemo Mesh without rewriting: a virtual store needs a "
                        "single simplex cell type (LINE=3, TRIANGLE=5 or TETRA=10). "
                        "Use from_pyvista (which tetrahedralizes) and to_zarr instead."
                    )
                else:
                    k = _SIMPLEX_ARITY[int(types[0])]
                    last_offset = _read_last_scalar(reader, header.offsets)
                    if last_offset != header.n_cells * k:
                        raise ValueError(
                            f"Connectivity length {last_offset} != n_cells * k = "
                            f"{header.n_cells * k}; cells are not all {k}-node."
                        )
                    conn = header.connectivity
                    connectivity = _ArrayInfo(
                        name=conn.name,
                        dtype=conn.dtype,
                        n_tuples=header.n_cells,
                        n_components=k,
                        payload_offset=conn.payload_offset,
                        payload_nbytes=header.n_cells * k * conn.dtype.itemsize,
                    )
                    cells_shape = (header.n_cells, k)
            if connectivity is None:
                connectivity = _ArrayInfo(
                    "connectivity",
                    np.dtype("<i8"),
                    0,
                    1,
                    header.points.payload_offset,
                    0,
                )

        def array(a: _ArrayInfo, shape=None):
            return _manifest_array(
                url, a, shape, chunk_rows=self.chunk_rows, file_size=file_size
            )

        def group(infos: list[_ArrayInfo], batch: list[int]) -> ManifestGroup:
            return ManifestGroup(
                arrays={
                    a.name: array(a) for a in infos if a.name not in self.skip_arrays
                },
                attributes=_tensordict_attrs(batch),
            )

        # A vertex-only grid has no cells, so its cell data has nothing to
        # attach to; drop it as from_pyvista does for point clouds.
        cell_data = header.cell_data if cells_shape[0] > 0 else []
        root = ManifestGroup(
            arrays={
                "points": array(header.points),
                "cells": array(connectivity, cells_shape),
            },
            groups={
                "point_data": group(header.point_data, [header.n_points]),
                "cell_data": group(cell_data, [cells_shape[0]]),
                "global_data": group(header.field_data, []),
            },
            attributes={**_tensordict_attrs([]), _TYPE_ATTR: "Mesh"},
        )
        return ManifestStore(group=root, registry=registry)


def open_vtu_virtual(
    path: str | Path,
    *,
    skip_arrays: Iterable[str] | None = None,
    chunk_rows: int | None = None,
) -> "zarr.Group":
    """Open a local ``.vtu`` file as a read-only zarr group in physicsnemo's layout.

    Convenience wrapper around :class:`VTUParser` for local files. For object
    storage build the registry yourself and call the parser.
    """
    path = Path(path).resolve()
    if "#" in str(path) or "?" in str(path):
        raise ValueError(
            f"Path {str(path)!r} contains '#' or '?', which cannot be expressed in "
            "the file:// URL the object store registry keys on. Rename the file."
        )
    # Raw paths on both sides: the registry strips its key from the URL path
    # by plain string prefix, so the URL must not be percent-encoded.
    url = "file://" + str(path)
    registry = _registry_mod.ObjectStoreRegistry(
        {
            "file://" + str(path.parent): _obstore_store.LocalStore(
                prefix=str(path.parent)
            )
        }
    )
    store = VTUParser(skip_arrays=skip_arrays, chunk_rows=chunk_rows)(url, registry)
    return _zarr.open_group(store, mode="r")


def vtu_to_mesh(
    path: str | Path,
    *,
    device=None,
    skip_arrays: Iterable[str] | None = None,
    chunk_rows: int | None = None,
):
    """Read a ``.vtu`` file into a :class:`~physicsnemo.mesh.Mesh` via its virtual zarr view.

    Reads everything, so it is a drop-in for ``from_pyvista(pv.read(path))``
    on the files :class:`VTUParser` supports. Point ids in the connectivity
    are range-checked after the read, as ``from_pyvista`` does.
    """
    from physicsnemo.mesh.io.io_zarr import _mesh_from_group

    mesh = _mesh_from_group(
        open_vtu_virtual(path, skip_arrays=skip_arrays, chunk_rows=chunk_rows), device
    )
    if mesh.n_cells > 0:
        lo, hi = int(mesh.cells.min()), int(mesh.cells.max())
        if lo < 0 or hi >= mesh.n_points:
            raise ValueError(
                f"Connectivity of {path} references point ids in [{lo}, {hi}] but "
                f"the mesh has {mesh.n_points} points."
            )
    return mesh


__all__ = ["VTUParser", "open_vtu_virtual", "vtu_to_mesh"]
