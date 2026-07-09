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

"""
Zarr mesh reader - Load physicsnemo Mesh samples from Zarr groups.

Bridges the zarr storage world (chunked, compressed, cloud-capable; the
format produced by PhysicsNeMo-Curator) and the ``physicsnemo.mesh`` object
world consumed by mesh-native pipelines (``MeshDataset`` + mesh transforms).

Mesh schema (one Zarr group per sample)::

    sample_0000.zarr/                     attrs: format="physicsnemo-mesh-zarr",
        points               (n_points, n_spatial_dims)          layout, ...
        cells                (n_cells, nodes_per_cell)   [absent for point clouds]
        point_data/<field>   (n_points, ...)
        cell_data/<field>    (n_cells, ...)
        global_data/<field>  (...)

Data fields mirror the ``TensorDict`` tree: a ``<field>`` is either an array
(a tensor leaf) or a subgroup (a nested ``TensorDict``), recursively. Field
names must not contain ``"/"`` (the zarr path separator) and non-tensor
leaves (``NonTensorData``) are not representable in schema v1; the writers
reject both with clear errors rather than corrupting the store.

DomainMesh schema (one Zarr group per case, mirroring the ``.pdmsh`` tree)::

    run_1.zarr/                    attrs: format="physicsnemo-domainmesh-zarr"
        global_data/<field>        # case-level metadata (freestream, ...)
        interior/...               # a mesh-schema subgroup
        boundaries/<name>/...      # mesh-schema subgroups

``save_mesh_to_zarr`` / ``save_domain_mesh_to_zarr`` write these schemas
(they are also the reference implementation for curation tools emitting
them); ``ZarrMeshReader`` reads them and returns ``(Mesh, metadata)`` per
sample, so it drops into any pipeline that accepts
:class:`~physicsnemo.datapipes.readers.mesh.MeshReader`. Use ``subpath``
(e.g. ``"boundaries/vehicle"``) to read one mesh out of a DomainMesh group,
and ``merge_global_data_from`` (e.g. ``"../../global_data"``) to merge
case-level metadata at read time.

Scope note: the writers accept zarr StoreLike objects (object stores,
transactional stores), but the readers and validator navigate filesystem
paths in this version -- "cloud-capable" currently applies to the storage
format and writers, not to reading from object stores.

Subsampling reads a random contiguous cell block (sequential, chunk-aligned
I/O) plus the contiguous point range the block references. The point range
is tight only when the mesh's point ordering has locality (e.g. data curated
as a per-cell vertex soup via :func:`to_cell_soup`); for scattered point
orderings it degrades toward a full ``points`` read -- reorder or
denormalize at curation time for fast block reads. Groups whose verified
``layout`` attr is ``"soup"`` skip the ``cells`` read entirely (indices are
synthesized), removing both bytes and the dependent-read serialization.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch
from tensordict import TensorDictBase

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.datapipes.registry import register

zarr = OptionalImport("zarr")
tensorstore = OptionalImport("tensorstore")

logger = logging.getLogger(__name__)

DEFAULT_GROUP_PATTERN = "*.zarr"
_DATA_SUBGROUPS = ("point_data", "cell_data", "global_data")
MESH_FORMAT = "physicsnemo-mesh-zarr"
DOMAIN_MESH_FORMAT = "physicsnemo-domainmesh-zarr"
SCHEMA_VERSION = 1


def to_cell_soup(mesh):
    """Denormalize a mesh to a per-cell vertex soup.

    Points are re-gathered so cell ``i`` references points
    ``[nodes_per_cell * i, nodes_per_cell * (i + 1))`` -- every
    block-subsampled read becomes a contiguous range read, at the cost of
    duplicating shared vertices (and their ``point_data``). Connectivity
    between cells is lost; use only for cell-centric consumers.
    """
    from physicsnemo.mesh import Mesh

    if mesh.n_cells == 0:
        return mesh
    cells = mesh.cells
    n_cells, nodes_per_cell = cells.shape
    flat = cells.reshape(-1)
    points = mesh.points[flat].contiguous()
    # TensorDict batch-indexing gathers every leaf (nested included) at once.
    point_data = mesh.point_data[flat].contiguous()
    soup_cells = torch.arange(points.shape[0], dtype=torch.int64).reshape(
        n_cells, nodes_per_cell
    )
    return Mesh(
        points=points,
        cells=soup_cells,
        point_data=point_data,
        cell_data=mesh.cell_data,
        global_data=mesh.global_data,
    )


def _is_soup(mesh) -> bool:
    """True iff ``cells`` is exactly ``arange(n_points)`` reshaped."""
    if mesh.n_cells == 0:
        return False
    cells = mesh.cells
    if cells.numel() != mesh.n_points:
        return False
    expected = torch.arange(cells.numel(), dtype=cells.dtype, device=cells.device)
    return bool(torch.equal(cells.reshape(-1), expected))


def _open_group(path_or_store, mode: str):
    """Open a zarr group from a filesystem path or any zarr StoreLike.

    Accepting store objects keeps the writers forward-compatible with
    non-filesystem backends (object stores, transactional stores such as
    Icechunk) without an API change.
    """
    if isinstance(path_or_store, (str, Path)):
        return zarr.open_group(str(path_or_store), mode=mode)
    return zarr.open_group(store=path_or_store, mode=mode)


def _check_schema_version(attrs, where) -> None:
    """Reject groups written by a newer schema (see MESH_ZARR_SCHEMA.md)."""
    version = attrs.get("schema_version", SCHEMA_VERSION)
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"{where} was written with mesh-zarr schema_version={version}, "
            f"but this reader supports <= {SCHEMA_VERSION}. Upgrade "
            f"physicsnemo to read this store."
        )


def _make_compressors(compress: bool):
    if compress:
        from zarr.codecs import BloscCodec

        return BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    return None


def _write_array(target, name, tensor, chunk0: int, compressors) -> None:
    arr = tensor.detach().cpu().contiguous().numpy()
    # 0-d scalars and empty arrays fall back to zarr's auto chunking.
    chunks = (
        (min(chunk0, arr.shape[0]),) + arr.shape[1:]
        if arr.ndim >= 1 and arr.shape[0] > 0
        else "auto"
    )
    z = target.create_array(
        name,
        shape=arr.shape,
        dtype=arr.dtype,
        chunks=chunks,
        compressors=compressors,
    )
    if arr.size:
        z[...] = arr


def _write_tensordict(group, td, chunk0: int, compressors) -> None:
    """Write a TensorDict as a zarr group tree, mirroring its nesting.

    Tensor leaves become arrays; nested TensorDicts become subgroups,
    recursively. Rejects field names containing ``"/"`` (zarr's path
    separator -- it would silently create hierarchy instead of a field)
    and non-tensor leaves (``NonTensorData`` is not representable in
    schema v1; an attrs-based encoding is reserved for v2).
    """
    for key in td.keys():
        if not key or "/" in key:
            raise ValueError(
                f"field name {key!r} is not storable in the mesh-zarr "
                f"schema: names must be non-empty and must not contain "
                f"'/' (the zarr path separator). Rename the field before "
                f"saving (see MESH_ZARR_SCHEMA.md)."
            )
        value = td.get(key)
        if isinstance(value, TensorDictBase):
            _write_tensordict(group.create_group(key), value, chunk0, compressors)
        elif isinstance(value, torch.Tensor):
            _write_array(group, key, value, chunk0, compressors)
        else:
            raise TypeError(
                f"field {key!r} has unsupported type "
                f"{type(value).__name__}; mesh-zarr schema v1 stores "
                f"tensors and nested TensorDicts only (NonTensorData / "
                f"strings are reserved for v2, see MESH_ZARR_SCHEMA.md)."
            )


def _write_mesh_group(
    group, mesh, chunk_cells: int, chunk_points: int, compressors
) -> None:
    """Write one mesh-schema group (arrays + verified attrs)."""
    _write_array(group, "points", mesh.points, chunk_points, compressors)
    if mesh.n_cells > 0:
        _write_array(group, "cells", mesh.cells, chunk_cells, compressors)

    for sub_name, data, chunk0 in (
        ("point_data", mesh.point_data, chunk_points),
        ("cell_data", mesh.cell_data, chunk_cells),
        ("global_data", mesh.global_data, chunk_cells),
    ):
        if not len(data.keys()):
            continue
        _write_tensordict(group.create_group(sub_name), data, chunk0, compressors)

    group.attrs["format"] = MESH_FORMAT
    group.attrs["schema_version"] = SCHEMA_VERSION
    group.attrs["n_points"] = int(mesh.n_points)
    group.attrs["n_cells"] = int(mesh.n_cells)
    if mesh.n_cells > 0:
        group.attrs["nodes_per_cell"] = int(mesh.cells.shape[1])
        # Verified, not caller-asserted: readers rely on it to skip the
        # cells read and synthesize indices.
        group.attrs["layout"] = "soup" if _is_soup(mesh) else "indexed"


def save_mesh_to_zarr(
    mesh,
    path: Path | str,
    *,
    chunk_cells: int = 200_000,
    chunk_points: int = 600_000,
    compress: bool = True,
) -> None:
    """Write a :class:`~physicsnemo.mesh.Mesh` as a Zarr group.

    Parameters
    ----------
    mesh : Mesh
        Mesh to serialize. Points, cells, point_data, cell_data and
        global_data are written; the internal geometry cache is not.
        Nested TensorDicts are written as nested zarr groups. Field names
        containing ``"/"`` and non-tensor leaves (``NonTensorData``) are
        rejected (not representable in schema v1).
    path : Path, str, or zarr StoreLike
        Target Zarr group directory or store (created / overwritten).
    chunk_cells : int, default=200_000
        Chunk length along the cell axis (``cells`` and ``cell_data``).
        Align this with the reader's ``subsample_n_cells`` so one sample
        read touches at most two chunks per field.
    chunk_points : int, default=600_000
        Chunk length along the point axis (``points`` and ``point_data``).
    compress : bool, default=True
        If True, compress chunks with blosc-zstd (bitshuffle). Uncompressed
        stores trade disk/network bytes for zero decode cost.
    """
    if not zarr.available:
        zarr._get_module()  # Raises with install hint

    group = _open_group(path, mode="w")
    _write_mesh_group(
        group, mesh, chunk_cells, chunk_points, _make_compressors(compress)
    )


def save_domain_mesh_to_zarr(
    domain_mesh,
    path: Path | str,
    *,
    chunk_cells: int = 200_000,
    chunk_points: int = 600_000,
    compress: bool = True,
    soup_boundaries: bool = False,
) -> None:
    """Write a :class:`~physicsnemo.mesh.DomainMesh` as a Zarr group tree.

    Mirrors the ``.pdmsh`` structure: case-level ``global_data`` at the
    root (single source of truth -- NOT copied into each mesh), the
    interior mesh under ``interior/``, and each boundary under
    ``boundaries/<name>/``, all in the mesh schema. Read one mesh out of
    the tree with ``ZarrMeshReader(subpath=..., merge_global_data_from=...)``.

    Parameters
    ----------
    domain_mesh : DomainMesh
        Domain mesh to serialize.
    path : Path, str, or zarr StoreLike
        Target Zarr group directory or store (created / overwritten).
    chunk_cells, chunk_points, compress
        See :func:`save_mesh_to_zarr`.
    soup_boundaries : bool, default=False
        If True, boundary meshes are denormalized via :func:`to_cell_soup`
        before writing (recommended for cell-centric surface training on
        meshes without point-index locality). The interior is never souped.
    """
    if not zarr.available:
        zarr._get_module()

    root = _open_group(path, mode="w")
    compressors = _make_compressors(compress)
    root.attrs["format"] = DOMAIN_MESH_FORMAT
    root.attrs["schema_version"] = SCHEMA_VERSION
    root.attrs["boundary_names"] = list(domain_mesh.boundary_names)

    if len(domain_mesh.global_data.keys()):
        _write_tensordict(
            root.create_group("global_data"),
            domain_mesh.global_data,
            chunk_cells,
            compressors,
        )

    _write_mesh_group(
        root.create_group("interior"),
        domain_mesh.interior,
        chunk_cells,
        chunk_points,
        compressors,
    )
    boundaries = root.create_group("boundaries")
    for name in domain_mesh.boundary_names:
        mesh = domain_mesh.boundaries[name]
        if soup_boundaries:
            mesh = to_cell_soup(mesh)
        _write_mesh_group(
            boundaries.create_group(name), mesh, chunk_cells, chunk_points, compressors
        )


def _nest_tensors(items) -> dict[str, Any]:
    """Rebuild a nested dict of tensors from ``(leaf_path, ndarray)`` pairs.

    Inverts the group-tree layout written by :func:`_write_tensordict`:
    ``"turbulence/k"`` becomes ``{"turbulence": {"k": tensor}}``.
    """
    out: dict[str, Any] = {}
    for path, value in items:
        parts = path.split("/")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        # np.asarray: 0-d zarr reads come back as numpy scalars, which
        # torch.from_numpy rejects.
        node[parts[-1]] = torch.from_numpy(np.asarray(value))
    return out


class _ZarrBackend:
    """zarr-python array access for one sample group."""

    def __init__(self, path: Path) -> None:
        # The schema requires zarr format 3 (see MESH_ZARR_SCHEMA.md); a
        # format-2 store would otherwise fail opaquely inside the
        # tensorstore backend (its driver is pinned to zarr3).
        if not (path / "zarr.json").exists() and (path / ".zgroup").exists():
            raise ValueError(
                f"{path} is a zarr format 2 store; the mesh-zarr schema "
                f"requires zarr format 3 (see MESH_ZARR_SCHEMA.md)."
            )
        self._group = zarr.open_group(str(path), mode="r")

    def attrs(self) -> dict[str, Any]:
        return dict(self._group.attrs)

    def leaf_array_paths(self, subgroup: str | None = None) -> list[str]:
        """Slash-separated paths of all arrays under ``subgroup``, any depth."""
        node = self._group[subgroup] if subgroup else self._group
        paths: list[str] = []

        def _walk(g, prefix: str) -> None:
            for name, _ in g.arrays():
                paths.append(prefix + name)
            for name, child in g.groups():
                _walk(child, f"{prefix}{name}/")

        _walk(node, "")
        return sorted(paths)

    def has(self, name: str) -> bool:
        return name in self._group

    def shape(self, name: str) -> tuple[int, ...]:
        return self._group[name].shape

    def read(self, name: str, sl: slice | None = None) -> np.ndarray:
        arr = self._group[name]
        return arr[sl] if sl is not None else arr[...]

    def read_many(self, requests: dict[str, slice | None]) -> dict[str, np.ndarray]:
        return {name: self.read(name, sl) for name, sl in requests.items()}


class _TensorstoreBackend:
    """tensorstore array access: concurrent async chunk reads."""

    def __init__(self, path: Path) -> None:
        # Discover layout with zarr-python (cheap metadata-only), read with
        # tensorstore.
        self._meta = _ZarrBackend(path)
        self._path = path
        self._handles: dict[str, Any] = {}

    def attrs(self) -> dict[str, Any]:
        return self._meta.attrs()

    def leaf_array_paths(self, subgroup: str | None = None) -> list[str]:
        return self._meta.leaf_array_paths(subgroup)

    def has(self, name: str) -> bool:
        return self._meta.has(name)

    def shape(self, name: str) -> tuple[int, ...]:
        return self._meta.shape(name)

    def _handle(self, name: str):
        if name not in self._handles:
            spec = {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": str(self._path / name)},
            }
            self._handles[name] = tensorstore.open(spec, open=True).result()
        return self._handles[name]

    def read(self, name: str, sl: slice | None = None) -> np.ndarray:
        h = self._handle(name)
        return (h[sl] if sl is not None else h).read().result()

    def read_many(self, requests: dict[str, slice | None]) -> dict[str, np.ndarray]:
        futures = {
            name: (self._handle(name)[sl] if sl is not None else self._handle(name)).read()
            for name, sl in requests.items()
        }
        return {name: fut.result() for name, fut in futures.items()}


@register()
class ZarrMeshReader:
    r"""
    Read :class:`~physicsnemo.mesh.Mesh` samples from Zarr groups.

    Each Zarr group under ``path`` (see the module docstring for the schema)
    is one sample. Returns ``(Mesh, metadata)`` per index, making this a
    drop-in alternative to
    :class:`~physicsnemo.datapipes.readers.mesh.MeshReader` for pipelines
    whose data lives in zarr (e.g. PhysicsNeMo-Curator output) rather than
    ``.pmsh`` memmap stores.

    With ``subsample_n_cells`` set, each read fetches a random contiguous
    cell block plus the contiguous point range it references -- sequential,
    chunk-aligned I/O. Unlike ``MeshReader`` the point window is not
    compacted to referenced-only points; unreferenced points inside the
    window are retained (harmless to cell-centric consumers, and it avoids
    a ``unique``-based gather).

    Examples
    --------
    >>> reader = ZarrMeshReader("data_dir/", subsample_n_cells=200_000)  # doctest: +SKIP
    >>> mesh, metadata = reader[0]  # doctest: +SKIP

    In a Hydra pipeline config (after ``import physicsnemo.datapipes``)::

        reader:
          _target_: ${dp:ZarrMeshReader}
          path: ${train_datadir}
          subsample_n_cells: ${sampling_resolution}
    """

    def __init__(
        self,
        path: Path | str,
        *,
        pattern: str = DEFAULT_GROUP_PATTERN,
        subpath: str | None = None,
        merge_global_data_from: str | None = None,
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
        subsample_n_cells: int | None = None,
        subsample_n_points: int | None = None,
        backend: Literal["auto", "zarr", "tensorstore"] = "auto",
    ) -> None:
        """
        Initialize the zarr mesh reader.

        Parameters
        ----------
        path : Path or str
            Directory containing Zarr mesh groups.
        pattern : str, optional
            Glob pattern for group directories. Default ``*.zarr``
            (non-recursive, unlike ``MeshReader``'s ``**/*.pmsh``; use
            ``"**/*.zarr"`` for nested directory trees).
        subpath : str, optional
            Path of a mesh-schema subgroup inside each matched group, e.g.
            ``"boundaries/vehicle"`` to read one boundary out of
            DomainMesh-schema groups. ``None`` reads the group itself.
        merge_global_data_from : str, optional
            Path of a global_data group whose keys are merged into each
            sample's ``global_data`` at read time, relative to the sample's
            mesh group (after ``subpath``), e.g. ``"../../global_data"``
            for the case-level metadata of a DomainMesh-schema group. A key
            present on both sides raises ``ValueError`` (case-level
            metadata is single-sourced by definition). Mirrors the recipe's
            ``MeshReaderWithGlobalData`` semantics for ``.pdmsh`` data.
        pin_memory : bool, default=False
            If True, place tensors in pinned (page-locked) memory for
            faster async CPU->GPU transfers.
        include_index_in_metadata : bool, default=True
            If True, include sample index in metadata.
        subsample_n_cells : int, optional
            If set, read a random contiguous block of this many cells (plus
            the point range it references) instead of the full mesh. For
            groups with a verified ``layout="soup"`` attr the ``cells``
            read is skipped entirely (indices are synthesized) and all
            array reads issue concurrently.
        subsample_n_points : int, optional
            If set, read a random contiguous block of this many points.
            Only supported for point clouds (groups without ``cells``);
            combine with ``subsample_n_cells`` is not supported.
        backend : {"auto", "zarr", "tensorstore"}, default="auto"
            Array-read backend. ``auto`` prefers tensorstore (concurrent
            async chunk reads) when installed, else zarr-python.
        """
        if not zarr.available:
            zarr._get_module()

        if subsample_n_cells is not None and subsample_n_points is not None:
            raise NotImplementedError(
                "ZarrMeshReader supports subsample_n_cells or "
                "subsample_n_points, not both."
            )

        self._root = Path(path)
        if not self._root.exists():
            raise FileNotFoundError(f"Path not found: {self._root}")
        if not self._root.is_dir():
            raise ValueError(f"Path must be a directory: {self._root}")

        group_roots = sorted(
            p for p in self._root.glob(pattern) if self._is_zarr_group(p)
        )
        if not group_roots:
            raise ValueError(f"No Zarr groups matching {pattern!r} in {self._root}")

        self._subpath = subpath
        self._merge_rel_path = merge_global_data_from
        if subpath is not None:
            self._paths = []
            for root in group_roots:
                mesh_path = Path(os.path.normpath(root / subpath))
                if not self._is_zarr_group(mesh_path):
                    raise FileNotFoundError(
                        f"subpath {subpath!r} is not a zarr group in {root}"
                    )
                self._paths.append(mesh_path)
        else:
            self._paths = group_roots

        if backend == "auto":
            backend = "tensorstore" if tensorstore.available else "zarr"
        if backend == "tensorstore" and not tensorstore.available:
            tensorstore._get_module()
        self._backend_name = backend
        self._backend_cls = (
            _TensorstoreBackend if backend == "tensorstore" else _ZarrBackend
        )
        self._backends: dict[int, Any] = {}
        self._merge_backends: dict[Path, Any] = {}

        self.pin_memory = pin_memory
        self.include_index_in_metadata = include_index_in_metadata
        self.subsample_n_cells = subsample_n_cells
        self.subsample_n_points = subsample_n_points
        self._subsample_generator: torch.Generator | None = None
        self._subsample_base_seed: int | None = None

    @staticmethod
    def _is_zarr_group(path: Path) -> bool:
        return (path / "zarr.json").exists() or (path / ".zgroup").exists()

    def __len__(self) -> int:
        return len(self._paths)

    def set_generator(self, generator: torch.Generator) -> None:
        """Assign a ``torch.Generator`` for reproducible subsampling."""
        self._subsample_generator = generator
        # Capture the base seed once: reseeding from initial_seed() each
        # epoch would drift (manual_seed updates initial_seed), making the
        # epoch->seed mapping depend on call history and breaking
        # checkpoint-resume reproducibility.
        self._subsample_base_seed = generator.initial_seed()

    def set_epoch(self, epoch: int) -> None:
        """Reseed the subsample RNG for a new epoch (base seed + epoch)."""
        if self._subsample_generator is not None:
            self._subsample_generator.manual_seed(
                self._subsample_base_seed + epoch
            )

    def _get_backend(self, index: int):
        if index not in self._backends:
            self._backends[index] = self._backend_cls(self._paths[index])
        return self._backends[index]

    def _block(self, total: int, k: int) -> slice:
        if total <= k:
            return slice(0, total)
        start = torch.randint(
            0, total - k + 1, (1,), generator=self._subsample_generator
        ).item()
        return slice(start, start + k)

    def _field_requests(
        self, backend, subgroup: str, sl: slice | None
    ) -> dict[str, slice | None]:
        # Leaf paths at any depth: nested TensorDicts share the leading
        # batch dim (TensorDict enforces it), so one slice fits all leaves.
        if not backend.has(subgroup):
            return {}
        return {f"{subgroup}/{k}": sl for k in backend.leaf_array_paths(subgroup)}

    def _merged_global_data(self, index: int) -> dict[str, Any]:
        """Read the external global_data group for read-time merging."""
        merge_path = Path(
            os.path.normpath(self._paths[index] / self._merge_rel_path)
        )
        if merge_path not in self._merge_backends:
            if not merge_path.exists():
                raise FileNotFoundError(
                    f"merge_global_data_from path not found: {merge_path} "
                    f"(resolved from sample {self._paths[index]} + "
                    f"{self._merge_rel_path!r})"
                )
            self._merge_backends[merge_path] = self._backend_cls(merge_path)
        backend = self._merge_backends[merge_path]
        arrays = backend.read_many({k: None for k in backend.leaf_array_paths()})
        return _nest_tensors(arrays.items())

    def _load_sample(self, index: int):
        from physicsnemo.mesh import Mesh

        backend = self._get_backend(index)
        attrs = backend.attrs()
        _check_schema_version(attrs, self._paths[index])
        fmt = attrs.get("format")
        if fmt is not None and fmt != MESH_FORMAT:
            hint = (
                " (a DomainMesh group: pass subpath='interior' or "
                "subpath='boundaries/<name>' to select a member mesh, or "
                "use ZarrDomainMeshReader)"
                if fmt == DOMAIN_MESH_FORMAT
                else ""
            )
            raise ValueError(
                f"{self._paths[index]} has format={fmt!r}, expected "
                f"{MESH_FORMAT!r}{hint}"
            )
        has_cells = backend.has("cells")
        is_soup = attrs.get("layout") == "soup"

        if self.subsample_n_points is not None and has_cells:
            raise NotImplementedError(
                "subsample_n_points requires point-cloud samples (no cells); "
                "use subsample_n_cells for meshes with connectivity."
            )

        cell_sl: slice | None = None
        point_sl: slice | None = None
        point_offset = 0
        cells: torch.Tensor | None = None

        if has_cells and self.subsample_n_cells is not None and is_soup:
            # Soup fast path: cell block [s, e) owns points
            # [npc*s, npc*e) by construction -- skip the cells read and
            # its dependency; every array read issues concurrently.
            npc = int(attrs["nodes_per_cell"])
            n_cells = int(attrs["n_cells"])
            cell_sl = self._block(n_cells, self.subsample_n_cells)
            point_sl = slice(npc * cell_sl.start, npc * cell_sl.stop)
            k = cell_sl.stop - cell_sl.start
            cells = torch.arange(npc * k, dtype=torch.int64).reshape(k, npc)
            requests = {"points": point_sl}
            requests |= self._field_requests(backend, "cell_data", cell_sl)
            requests |= self._field_requests(backend, "point_data", point_sl)
            requests |= self._field_requests(backend, "global_data", None)
            arrays = backend.read_many(requests)
        elif has_cells and self.subsample_n_cells is not None:
            n_cells = backend.shape("cells")[0]
            cell_sl = self._block(n_cells, self.subsample_n_cells)
            # Dependent read: the cell block determines the point range.
            cells_np = backend.read("cells", cell_sl)
            point_offset = int(cells_np.min())
            point_sl = slice(point_offset, int(cells_np.max()) + 1)
            cells = torch.from_numpy(np.ascontiguousarray(cells_np)).long()
            if point_offset:
                cells = cells - point_offset
            requests = {"points": point_sl}
            requests |= self._field_requests(backend, "cell_data", cell_sl)
            requests |= self._field_requests(backend, "point_data", point_sl)
            requests |= self._field_requests(backend, "global_data", None)
            arrays = backend.read_many(requests)
        elif self.subsample_n_points is not None:
            n_points = backend.shape("points")[0]
            point_sl = self._block(n_points, self.subsample_n_points)
            requests = {"points": point_sl}
            requests |= self._field_requests(backend, "point_data", point_sl)
            requests |= self._field_requests(backend, "global_data", None)
            arrays = backend.read_many(requests)
        else:
            requests = {"points": None}
            if has_cells and not is_soup:
                requests["cells"] = None
            for sub in _DATA_SUBGROUPS:
                requests |= self._field_requests(backend, sub, None)
            arrays = backend.read_many(requests)
            if has_cells and is_soup:
                npc = int(attrs["nodes_per_cell"])
                n_cells = int(attrs["n_cells"])
                cells = torch.arange(npc * n_cells, dtype=torch.int64).reshape(
                    n_cells, npc
                )
            elif "cells" in arrays:
                cells = torch.from_numpy(
                    np.ascontiguousarray(arrays["cells"])
                ).long()

        def _sub(name: str) -> dict[str, Any]:
            prefix = f"{name}/"
            return _nest_tensors(
                (k[len(prefix):], v)
                for k, v in arrays.items()
                if k.startswith(prefix)
            )

        global_data = _sub("global_data")
        if self._merge_rel_path is not None:
            external = self._merged_global_data(index)
            collisions = sorted(set(external) & set(global_data))
            if collisions:
                raise ValueError(
                    f"global_data key collision merging "
                    f"{self._merge_rel_path!r} into sample "
                    f"{self._paths[index]}: keys {collisions} exist on both "
                    f"sides. Case-level metadata is single-sourced by "
                    f"definition; overlap indicates inconsistent data."
                )
            global_data |= external

        kwargs: dict[str, Any] = {
            "points": torch.from_numpy(arrays["points"]),
            "point_data": _sub("point_data"),
            "cell_data": _sub("cell_data"),
            "global_data": global_data,
        }
        if cells is not None:
            kwargs["cells"] = cells
        return Mesh(**kwargs)

    def __getitem__(self, index: int):
        mesh = self._load_sample(index)
        if self.pin_memory:
            mesh = mesh.pin_memory()
        metadata: dict[str, Any] = {"source_path": str(self._paths[index])}
        if self.include_index_in_metadata:
            metadata["index"] = index
        return mesh, metadata

    def __iter__(self) -> Iterator[tuple[Any, dict[str, Any]]]:
        for i in range(len(self)):
            try:
                yield self[i]
            except Exception as e:
                logger.error("Sample %s failed: %s", i, e)
                raise RuntimeError(f"Sample {i} failed: {e}") from e

    def __repr__(self) -> str:
        return (
            f"ZarrMeshReader(path={self._root!r}, len={len(self)}, "
            f"backend={self._backend_name!r})"
        )


@register()
class ZarrDomainMeshReader:
    r"""
    Read :class:`~physicsnemo.mesh.DomainMesh` samples from Zarr groups.

    Each DomainMesh-schema group under ``path`` (root ``global_data/`` +
    ``interior/`` + ``boundaries/<name>/``) is one sample. Returns
    ``(DomainMesh, metadata)`` per index, making this a drop-in
    alternative to
    :class:`~physicsnemo.datapipes.readers.mesh.DomainMeshReader` for
    cases curated to zarr (e.g. by PhysicsNeMo-Curator's
    ``DomainMeshZarrSink``).

    Subsampling picks per sub-mesh: meshes with cells use
    ``subsample_n_cells`` (contiguous block; soup fast path when the
    group's verified ``layout`` attr allows), point clouds use
    ``subsample_n_points``. Boundaries listed in
    ``full_resolution_boundaries`` are never subsampled -- use for
    geometry consumed by exact queries (e.g. an STL boundary feeding
    SDF computation).

    The dataset must be homogeneous: every case group must have the same
    ``boundary_names`` (and per-member cells-vs-points structure) as the
    first case, which defines the reader layout. A case with a missing
    boundary fails at construction; a case whose ``boundary_names``
    otherwise differ (e.g. extra boundaries) raises at read time rather
    than silently dropping data.

    Examples
    --------
    >>> reader = ZarrDomainMeshReader(  # doctest: +SKIP
    ...     "data_dir/",
    ...     subsample_n_points=200_000,
    ...     subsample_n_cells=200_000,
    ...     full_resolution_boundaries=["stl_geometry"],
    ... )
    >>> domain, metadata = reader[0]  # doctest: +SKIP
    """

    def __init__(
        self,
        path: Path | str,
        *,
        pattern: str = DEFAULT_GROUP_PATTERN,
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
        subsample_n_points: int | None = None,
        subsample_n_cells: int | None = None,
        full_resolution_boundaries: list[str] | None = None,
        backend: Literal["auto", "zarr", "tensorstore"] = "auto",
    ) -> None:
        """
        Initialize the zarr domain mesh reader.

        Parameters
        ----------
        path : Path or str
            Directory containing DomainMesh-schema Zarr groups.
        pattern : str, optional
            Glob pattern for group directories. Default ``*.zarr``.
        pin_memory : bool, default=False
            If True, place tensors in pinned memory.
        include_index_in_metadata : bool, default=True
            If True, include sample index in metadata.
        subsample_n_points : int, optional
            Contiguous block size for point-cloud sub-meshes (e.g. the
            volume interior).
        subsample_n_cells : int, optional
            Contiguous block size for sub-meshes with cells.
        full_resolution_boundaries : list[str], optional
            Boundary names loaded at full resolution (no subsampling).
        backend : {"auto", "zarr", "tensorstore"}, default="auto"
            Array-read backend (see :class:`ZarrMeshReader`).
        """
        if not zarr.available:
            zarr._get_module()

        self._root = Path(path)
        if not self._root.exists():
            raise FileNotFoundError(f"Path not found: {self._root}")
        self._paths = sorted(
            p for p in self._root.glob(pattern) if ZarrMeshReader._is_zarr_group(p)
        )
        if not self._paths:
            raise ValueError(f"No Zarr groups matching {pattern!r} in {self._root}")

        first = zarr.open_group(str(self._paths[0]), mode="r")
        _check_schema_version(dict(first.attrs), self._paths[0])
        if first.attrs.get("format") != DOMAIN_MESH_FORMAT:
            raise ValueError(
                f"{self._paths[0]} is not a {DOMAIN_MESH_FORMAT!r} group "
                f"(format={first.attrs.get('format')!r})"
            )
        self._boundary_names = list(first.attrs["boundary_names"])
        full_res = set(full_resolution_boundaries or [])
        unknown = full_res - set(self._boundary_names)
        if unknown:
            raise ValueError(
                f"full_resolution_boundaries {sorted(unknown)} not found; "
                f"available: {self._boundary_names}"
            )

        def _sub_reader(subpath: str, full_resolution: bool) -> ZarrMeshReader:
            n_cells = n_points = None
            if not full_resolution:
                # Pick the applicable subsample mode from the sub-mesh
                # schema (homogeneous across a curated dataset).
                if int(first[subpath].attrs.get("n_cells", 0)) > 0:
                    n_cells = subsample_n_cells
                else:
                    n_points = subsample_n_points
            return ZarrMeshReader(
                path,
                pattern=pattern,
                subpath=subpath,
                include_index_in_metadata=False,
                subsample_n_cells=n_cells,
                subsample_n_points=n_points,
                backend=backend,
            )

        self._interior_reader = _sub_reader("interior", False)
        self._boundary_readers = {
            name: _sub_reader(f"boundaries/{name}", name in full_res)
            for name in self._boundary_names
        }
        self._backend_cls = self._interior_reader._backend_cls
        self._root_backends: dict[int, Any] = {}
        self.pin_memory = pin_memory
        self.include_index_in_metadata = include_index_in_metadata

    def __len__(self) -> int:
        return len(self._paths)

    def set_generator(self, generator: torch.Generator) -> None:
        """Assign a shared ``torch.Generator`` for reproducible subsampling."""
        self._interior_reader.set_generator(generator)
        for reader in self._boundary_readers.values():
            reader.set_generator(generator)

    def set_epoch(self, epoch: int) -> None:
        """Reseed the subsample RNG for a new epoch."""
        self._interior_reader.set_epoch(epoch)
        # Sub-readers share one generator; a single reseed suffices, and
        # reseeding per reader would just repeat the same assignment.

    def _root_backend(self, index: int):
        if index not in self._root_backends:
            backend = self._backend_cls(self._paths[index])
            # Boundary structure (and each sub-reader's subsample mode) is
            # taken from the first case; verify instead of silently
            # dropping boundaries a later case might add.
            names = list(backend.attrs().get("boundary_names", []))
            if names != self._boundary_names:
                raise ValueError(
                    f"{self._paths[index]} has boundary_names {names}, but "
                    f"{self._paths[0]} (which defines the dataset "
                    f"structure) has {self._boundary_names}. "
                    f"ZarrDomainMeshReader requires a homogeneous dataset."
                )
            self._root_backends[index] = backend
        return self._root_backends[index]

    def _global_data(self, index: int) -> dict[str, Any]:
        backend = self._root_backend(index)
        if not backend.has("global_data"):
            return {}
        keys = backend.leaf_array_paths("global_data")
        arrays = backend.read_many({f"global_data/{k}": None for k in keys})
        return _nest_tensors(
            (k.split("/", 1)[1], v) for k, v in arrays.items()
        )

    def __getitem__(self, index: int):
        from physicsnemo.mesh import DomainMesh

        domain = DomainMesh(
            interior=self._interior_reader._load_sample(index),
            boundaries={
                name: reader._load_sample(index)
                for name, reader in self._boundary_readers.items()
            },
            global_data=self._global_data(index),
        )
        if self.pin_memory:
            domain = domain.pin_memory()
        metadata: dict[str, Any] = {
            "source_path": str(self._paths[index]),
            "boundary_names": self._boundary_names,
        }
        if self.include_index_in_metadata:
            metadata["index"] = index
        return domain, metadata

    def __iter__(self) -> Iterator[tuple[Any, dict[str, Any]]]:
        for i in range(len(self)):
            try:
                yield self[i]
            except Exception as e:
                logger.error("Sample %s failed: %s", i, e)
                raise RuntimeError(f"Sample {i} failed: {e}") from e

    def __repr__(self) -> str:
        return (
            f"ZarrDomainMeshReader(path={self._root!r}, len={len(self)}, "
            f"boundaries={self._boundary_names!r})"
        )


def validate_mesh_zarr(path: Path | str, *, _prefix: str = "") -> dict[str, Any]:
    """Validate a mesh-zarr or domainmesh-zarr group against the schema.

    Checks the requirements of ``MESH_ZARR_SCHEMA.md``: known ``format``,
    supported ``schema_version``, required arrays and attributes, attr /
    array-shape consistency, field-length consistency, and (sampled) that
    a ``layout="soup"`` claim is truthful. Raises ``ValueError`` with all
    problems found; returns a summary dict when valid.

    Parameters
    ----------
    path : Path or str
        Zarr group directory to validate.

    Returns
    -------
    dict
        Summary: ``{"format": ..., "schema_version": ..., ...}`` plus
        per-member summaries for DomainMesh groups.
    """
    if not zarr.available:
        zarr._get_module()

    group = zarr.open_group(str(path), mode="r")
    attrs = dict(group.attrs)
    problems: list[str] = []
    fmt = attrs.get("format")

    if fmt == DOMAIN_MESH_FORMAT:
        _check_schema_version(attrs, path)
        summary: dict[str, Any] = {
            "format": fmt,
            "schema_version": attrs.get("schema_version"),
        }
        names = attrs.get("boundary_names")
        if not isinstance(names, list):
            problems.append("missing/invalid attr 'boundary_names'")
            names = []
        if "interior" not in group:
            problems.append("missing 'interior' group")
        else:
            summary["interior"] = validate_mesh_zarr(
                Path(path) / "interior", _prefix="interior: "
            )
        boundaries: dict[str, Any] = {}
        for name in names:
            if f"boundaries/{name}" not in group:
                problems.append(f"boundary_names lists {name!r} but group is missing")
            else:
                boundaries[name] = validate_mesh_zarr(
                    Path(path) / "boundaries" / name, _prefix=f"boundaries/{name}: "
                )
        summary["boundaries"] = boundaries
        if problems:
            raise ValueError(
                f"{path} failed validation: " + "; ".join(problems)
            )
        return summary

    if fmt != MESH_FORMAT:
        raise ValueError(
            f"{_prefix}{path}: unknown format attr {fmt!r} "
            f"(expected {MESH_FORMAT!r} or {DOMAIN_MESH_FORMAT!r})"
        )
    _check_schema_version(attrs, path)

    if "points" not in group:
        problems.append("missing required array 'points'")
        n_points = None
    else:
        n_points = group["points"].shape[0]
        if attrs.get("n_points") != n_points:
            problems.append(
                f"attr n_points={attrs.get('n_points')} != points shape {n_points}"
            )

    n_cells = int(attrs.get("n_cells", 0))
    if n_cells > 0:
        layout = attrs.get("layout")
        npc = attrs.get("nodes_per_cell")
        if layout not in ("soup", "indexed"):
            problems.append(f"invalid layout attr {layout!r}")
        if not isinstance(npc, int):
            problems.append("missing attr 'nodes_per_cell'")
        if "cells" not in group:
            problems.append("n_cells > 0 but 'cells' array missing")
        else:
            cells = group["cells"]
            if cells.shape[0] != n_cells:
                problems.append(
                    f"attr n_cells={n_cells} != cells shape {cells.shape[0]}"
                )
            if isinstance(npc, int) and cells.shape[1] != npc:
                problems.append(
                    f"attr nodes_per_cell={npc} != cells shape {cells.shape[1]}"
                )
            # Sampled truthfulness check of a soup claim: first and last
            # window must be the identity arange. Full verification is
            # O(n_cells); the sampled check catches every honest mistake
            # short of an adversarial store.
            if layout == "soup" and n_points is not None:
                if n_points != n_cells * cells.shape[1]:
                    problems.append("layout=soup but n_points != n_cells*nodes_per_cell")
                else:
                    k = min(1000, n_cells)
                    head = np.asarray(cells[:k]).reshape(-1)
                    tail = np.asarray(cells[-k:]).reshape(-1)
                    w = cells.shape[1]
                    if not (
                        np.array_equal(head, np.arange(k * w))
                        and np.array_equal(
                            tail, np.arange((n_cells - k) * w, n_cells * w)
                        )
                    ):
                        problems.append("layout=soup but cells are not arange")

    def _iter_leaf_arrays(node, prefix: str = ""):
        """All arrays under a group, any depth (nested TensorDict fields)."""
        for name, arr in node.arrays():
            yield prefix + name, arr
        for name, child in node.groups():
            yield from _iter_leaf_arrays(child, f"{prefix}{name}/")

    for sub, expected in (("point_data", n_points), ("cell_data", n_cells or None)):
        if sub in group and expected is not None:
            for key, arr in _iter_leaf_arrays(group[sub]):
                got = arr.shape[0] if arr.ndim else None
                if got is not None and got != expected:
                    problems.append(
                        f"{sub}/{key} first dim {got} != {expected}"
                    )

    if problems:
        raise ValueError(f"{_prefix}{path} failed validation: " + "; ".join(problems))
    return {
        "format": fmt,
        "schema_version": attrs.get("schema_version"),
        "n_points": n_points,
        "n_cells": n_cells,
        "layout": attrs.get("layout"),
    }
