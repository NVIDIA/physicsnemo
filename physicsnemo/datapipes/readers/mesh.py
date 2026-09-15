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
Mesh readers - Load physicsnemo Mesh / DomainMesh from physicsnemo mesh format (.pmsh / .pdmsh).

MeshReader returns (Mesh, metadata) per sample.
DomainMeshReader returns (DomainMesh, metadata) per sample.
Both use tensorclass .load(path) directly; no conversion from other formats.
"""

from __future__ import annotations

import glob as _glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import torch
from tensordict import TensorDict

from physicsnemo.datapipes._domain_parallel import (
    DomainParallelConfig,
    PlacementName,
    ShardedProto,
    as_leaf_key,
    chunk_bounds,
    register_proto_kind,
)
from physicsnemo.datapipes._indexing import _cyclic_block_indices
from physicsnemo.datapipes._rng import spawn_generator
from physicsnemo.datapipes.registry import register
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import (
    MEASURE_WEIGHTS_KEY,
    compose_measure_weights,
)
from physicsnemo.mesh.io import io_zarr

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)

# Default extensions for physicsnemo mesh formats (tensordict/tensorclass layout).
# Do not hardcode elsewhere so format can evolve.
DEFAULT_MESH_EXTENSION = ".pmsh"
DEFAULT_DOMAIN_MESH_EXTENSION = ".pdmsh"


def _subsample_mesh_points(
    mesh: Mesh,
    n_points: int,
    generator: torch.Generator | None = None,
) -> Mesh:
    """Subsample a Mesh to *n_points* via a cyclic contiguous block read.

    Uses one or two contiguous runs for page-sequential I/O on memmap-backed
    data while giving every point the same inclusion probability.
    For point clouds (``n_cells == 0``) this avoids the heavy
    cell-remapping logic in :meth:`Mesh.slice_points` which allocates
    two *N*-element intermediate tensors.  For meshes with cells it
    falls back to ``slice_points``.

    Unlike :func:`_subsample_mesh_cells`, this does NOT maintain
    measure weights: dropping points removes cells implicitly, with
    no per-cell inclusion probability to invert.  Prefer cell
    subsampling when downstream code integrates over the mesh.
    """
    if mesh.n_points <= n_points:
        return mesh
    indices = _cyclic_block_indices(
        mesh.n_points,
        n_points,
        generator=generator,
        device=mesh.points.device,
    )
    if mesh.n_cells == 0:
        return Mesh(
            points=mesh.points[indices],
            cells=mesh.cells,
            point_data=mesh.point_data[indices],
            cell_data=mesh.cell_data,
            global_data=mesh.global_data,
        )
    return mesh.slice_points(indices)


def _subsample_mesh_cells(
    mesh: Mesh,
    n_cells: int,
    generator: torch.Generator | None = None,
) -> Mesh:
    """Subsample a Mesh to *n_cells* via a cyclic contiguous block read on cells.

    Preserves cell topology: each selected cell retains its full vertex
    connectivity.  Unreferenced points are compacted out.  Uses
    :func:`_cyclic_block_indices` for (page-)sequential I/O on
    memmap-backed cell tensors.

    Preserves the mesh's integration measure: every cell's inclusion
    probability is exactly ``k/N``, and the retained cells' measure
    weights (see :mod:`physicsnemo.mesh.calculus.measure`) are multiplied by
    ``N/k``, composing with any weights from earlier sampling stages.
    Consumers of the effective cell measure (see
    :mod:`physicsnemo.mesh.calculus.measure`) then see an unbiased estimate
    of the full-mesh measure rather than the ~``k/N`` retained fraction.

    Use this instead of :func:`_subsample_mesh_points` when the mesh
    has cell connectivity (triangulated surfaces, volume meshes) and
    downstream transforms or outputs depend on cell topology (e.g.
    surface normals, cell centroids, cell_data fields).
    """
    n_total = mesh.n_cells
    if n_total <= n_cells:
        return mesh
    indices = _cyclic_block_indices(
        n_total,
        n_cells,
        generator=generator,
        device=mesh.cells.device,
    )
    mesh = mesh.slice_cells(indices)
    # Compact: drop vertices not referenced by any surviving cell
    referenced = torch.unique(mesh.cells)
    if referenced.numel() < mesh.n_points:
        mesh = mesh.slice_points(referenced)
    ### Compose the Horvitz-Thompson weight for this sampling stage.
    ### slice_cells/slice_points returned fresh TensorDicts, so the
    ### in-place update cannot leak into the memmap-backed source.
    compose_measure_weights(mesh, n_total / n_cells)
    return mesh


def _indices_to_runs(indices: torch.Tensor) -> list[tuple[int, int]]:
    """Convert cyclic-block indices (1-2 ascending contiguous runs) to runs."""
    breaks = torch.nonzero(indices[1:] != indices[:-1] + 1).flatten()
    starts = [0] + [int(b) + 1 for b in breaks]
    ends = [int(b) + 1 for b in breaks] + [len(indices)]
    return [(int(indices[s]), int(indices[e - 1]) + 1) for s, e in zip(starts, ends)]


def _zarr_mesh_subsampled(
    group: Any,
    n_cells: int | None,
    n_points: int | None,
    generator: torch.Generator | None,
    *,
    drop_cells: bool = False,
) -> Mesh:
    """Partial-read a zarr mesh group: fetch only the subsample window.

    Reproduces :func:`_subsample_mesh` semantics (cyclic contiguous blocks,
    vertex compaction, Horvitz-Thompson measure weights) while reading only
    the selected rows from the store instead of materializing the full mesh.
    With ``drop_cells`` the group is read as a point cloud (cells and cell
    data are never fetched), matching the reader's ``drop_interior_cells``.
    """
    _ioz = io_zarr
    total_cells = (
        0 if drop_cells else (group["cells"].shape[0] if "cells" in group else 0)
    )
    total_points = group["points"].shape[0]

    if total_cells > 0 and n_cells is not None and total_cells > n_cells:
        indices = _cyclic_block_indices(total_cells, n_cells, generator=generator)
        runs = _indices_to_runs(indices)
        cells = _ioz._read_rows(group["cells"], runs)
        # Compact: gather only referenced vertices; remap connectivity to the
        # sorted-unique order, matching slice_cells + slice_points.
        referenced, inverse = torch.unique(cells, return_inverse=True)
        cells = inverse.reshape(cells.shape)
        ref_np = referenced.numpy()
        mesh = Mesh(
            points=_ioz._read_index(group["points"], ref_np),
            cells=cells,
            point_data=_ioz._read_tree(
                group, "point_data", leaf_reader=lambda a: _ioz._read_index(a, ref_np)
            ),
            cell_data=_ioz._read_tree(
                group, "cell_data", leaf_reader=lambda a: _ioz._read_rows(a, runs)
            ),
            global_data=_ioz._read_tree(group, "global_data"),
        )
        compose_measure_weights(mesh, total_cells / n_cells)
        if n_points is not None:
            mesh = _subsample_mesh_points(mesh, n_points, generator=generator)
        return mesh

    if total_cells == 0 and n_points is not None and total_points > n_points:
        indices = _cyclic_block_indices(total_points, n_points, generator=generator)
        runs = _indices_to_runs(indices)
        return Mesh(
            points=_ioz._read_rows(group["points"], runs),
            point_data=_ioz._read_tree(
                group, "point_data", leaf_reader=lambda a: _ioz._read_rows(a, runs)
            ),
            cell_data=(
                TensorDict({}, batch_size=[])
                if drop_cells
                else _ioz._read_tree(group, "cell_data")
            ),
            global_data=_ioz._read_tree(group, "global_data"),
        )

    if drop_cells:
        return Mesh(
            points=_ioz._read_rows(group["points"], [(0, total_points)]),
            point_data=_ioz._read_tree(group, "point_data"),
            global_data=_ioz._read_tree(group, "global_data"),
        )

    # No subsampling applies (small mesh, or unsupported combination):
    # eager full read keeps semantics identical to the memmap path.
    return _ioz._mesh_from_group(group, None)


def _subsample_mesh(
    mesh: Mesh,
    n_cells: int | None = None,
    n_points: int | None = None,
    generator: torch.Generator | None = None,
) -> Mesh:
    """Apply cell and/or point subsampling to a single Mesh.

    Cells are subsampled first (preserving topology) so that the
    subsequent point subsample operates on the already-reduced mesh.
    """
    if n_cells is not None:
        mesh = _subsample_mesh_cells(mesh, n_cells, generator=generator)
    if n_points is not None:
        mesh = _subsample_mesh_points(mesh, n_points, generator=generator)
    return mesh


# ---------------------------------------------------------------------------
# Domain-parallel (rank-local) reading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rank-local (domain-parallel) reading
# ---------------------------------------------------------------------------
#
# A mesh is read one of two ways: a lazy memmap ``Mesh`` (``Mesh.load``) whose
# entries can be sliced without materializing the file, or a zarr group whose
# entries are fetched with ``io_zarr._read_rows`` / ``_read_index``. ``_read_mesh_selection``
# hides that difference; everything above it plans in terms of dim-0 selections.
#
# Layout: one batch axis per mesh, always even chunks.
# - Point cloud (no cells): ``points`` / ``point_data`` chunk over the
#   (windowed) point range; ``global_data`` replicates.
# - Mesh with cells: ``cells`` / ``cell_data`` chunk over the (windowed) cell
#   range and ``points`` / ``point_data`` chunk over the point range -- after
#   the same global compaction onto referenced vertices the eager reader
#   performs when a cell window applies. Cells keep global vertex ids, so
#   ``points[cells]`` on the assembled mesh is ShardTensor's routed gather.
#   Point subsampling on a mesh with cells is not supported.
# Caches are dropped (they recompute lazily through ShardTensor ops).

IndexSelection = slice | torch.Tensor
ShardedMap = dict[tuple[str, ...], tuple[int, ...]]
MeshSource = Mesh | Any  # a lazy ``Mesh`` or a zarr group


def _is_zarr(src: MeshSource) -> bool:
    return not isinstance(src, Mesh)


def _mesh_counts(src: MeshSource, drop_cells: bool = False) -> tuple[int, int]:
    """``(n_points, n_cells)`` of a lazy mesh or zarr group; metadata only."""
    if _is_zarr(src):
        n_points = src["points"].shape[0]
        n_cells = src["cells"].shape[0] if "cells" in src else 0
    else:
        n_points, n_cells = src.n_points, src.n_cells
    return n_points, 0 if drop_cells else n_cells


def _read(t: torch.Tensor) -> torch.Tensor:
    # clone() materializes memmap entries into plain host memory NOW, on the
    # calling (worker) thread -- on a lazy mesh this is the actual disk read.
    return torch.as_tensor(t).clone()


def _read_leaves(td: TensorDict, selection: IndexSelection | None = None) -> TensorDict:
    """Read every (nested) leaf of a lazy TensorDict, optionally a dim-0 selection."""
    out = TensorDict({}, batch_size=[])
    for key, value in td.items(include_nested=True, leaves_only=True):
        out.set(key, _read(value if selection is None else value[selection]))
    return out


def _zarr_selection(arr: Any, selection: IndexSelection, n: int) -> torch.Tensor:
    """Read a selection of a zarr array: a contiguous run, 1-2 runs, or scattered ids."""
    if isinstance(selection, slice):
        start, stop, _ = selection.indices(n)
        return io_zarr._read_rows(arr, [(start, stop)])
    runs = _indices_to_runs(selection)
    if len(runs) <= 2:  # a cyclic-block window: page-sequential run reads
        return io_zarr._read_rows(arr, runs)
    return io_zarr._read_index(arr, selection.numpy())  # scattered ids


def _read_cells(src: MeshSource, selection: IndexSelection) -> torch.Tensor:
    """Connectivity only (used to compact a cell window before reading points)."""
    if _is_zarr(src):
        return _zarr_selection(src["cells"], selection, src["cells"].shape[0])
    return _read(src.cells[selection])


def _read_mesh_selection(
    src: MeshSource,
    point_selection: IndexSelection,
    cell_selection: IndexSelection,
    *,
    drop_cells: bool = False,
) -> Mesh:
    """Read a selection of points and cells of a mesh into memory.

    Parameters
    ----------
    src : Mesh or zarr group
        A lazy memmap ``Mesh`` or an open zarr mesh group.
    point_selection, cell_selection : slice or Tensor
        Dim-0 selection of the point and cell axes (``point_data`` /
        ``cell_data`` leaves follow their axis).
    drop_cells : bool, default=False
        Read the mesh as a point cloud: no cells or cell data.

    Returns
    -------
    Mesh
        Plain in-memory mesh holding exactly that selection; ``global_data``
        is read whole. Caches are not carried over.
    """
    empty = TensorDict({}, batch_size=[])
    if _is_zarr(src):
        n_points, n_cells = _mesh_counts(src)
        points = _zarr_selection(src["points"], point_selection, n_points)
        point_data = io_zarr._read_tree(
            src,
            "point_data",
            leaf_reader=lambda a: _zarr_selection(a, point_selection, n_points),
        )
        if drop_cells or n_cells == 0:
            cells, cell_data = None, empty
        else:
            cells = _zarr_selection(src["cells"], cell_selection, n_cells)
            cell_data = io_zarr._read_tree(
                src,
                "cell_data",
                leaf_reader=lambda a: _zarr_selection(a, cell_selection, n_cells),
            )
        global_data = io_zarr._read_tree(src, "global_data")
    else:
        points = _read(src.points[point_selection])
        point_data = _read_leaves(src.point_data, point_selection)
        if drop_cells or src.n_cells == 0:
            cells, cell_data = None, empty
        else:
            cells = _read(src.cells[cell_selection])
            cell_data = _read_leaves(src.cell_data, cell_selection)
        global_data = _read_leaves(src.global_data)
    return Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        global_data=global_data,
    )


def _require_seed(generator: torch.Generator | None) -> None:
    """Domain-parallel subsampling needs a seed: every rank must draw the same window.

    Without one the draw falls back to each process's global RNG, so ranks
    would read different entries while agreeing on the global shapes -- a
    silently corrupt sample.
    """
    if generator is None:
        raise ValueError(
            "domain-parallel reading with subsampling requires a seed so every "
            "rank draws the same window: call set_generator on the dataset (the "
            "DataLoader does this when given a seed)"
        )


# ---- placement per sub-mesh -------------------------------------------------


def mesh_axis_names(path: str) -> tuple[str, str]:
    r"""``(points_axis, cells_axis)`` names for the sub-mesh at *path*.

    ``path`` is ``""`` for a standalone mesh, ``"interior"`` or
    ``"boundaries.<name>"`` inside a domain mesh. A mesh has exactly one
    batch axis: ``points`` for a point cloud, ``cells`` otherwise (its
    vertices follow the cell partition).
    """
    prefix = f"{path}." if path else ""
    return f"{prefix}points", f"{prefix}cells"


def resolve_mesh_placements(
    shapes: dict[str, tuple[int, int]],
    config: DomainParallelConfig,
    pinned: dict[str, PlacementName] | None = None,
) -> dict[str, tuple[bool, bool]]:
    r"""Decide ``(shard_points, shard_cells)`` per sub-mesh from global counts.

    Parameters
    ----------
    shapes : dict[str, tuple[int, int]]
        ``path -> (n_points, n_cells)`` for every sub-mesh, where the counts
        are the effective global counts (the subsample window length when
        one applies).
    config : DomainParallelConfig
        Parsed configuration bound to the device mesh.
    pinned : dict[str, PlacementName], optional
        Reader-supplied structural pins (``"boundaries.stl": "replicate"``).

    Returns
    -------
    dict[str, tuple[bool, bool]]
        Per-path ``(shard_points, shard_cells)``. A point cloud is gated on
        its ``points`` axis and never shards cells; a mesh with cells is
        gated on its ``cells`` axis and its vertices follow that decision.
    """
    axes: dict[str, int] = {}
    for path, (n_points, n_cells) in shapes.items():
        points_axis, cells_axis = mesh_axis_names(path)
        if n_cells > 0:
            axes[cells_axis] = n_cells
        else:
            axes[points_axis] = n_points
    decisions = config.decide(axes, pinned)
    out: dict[str, tuple[bool, bool]] = {}
    for path, (_n_points, n_cells) in shapes.items():
        points_axis, cells_axis = mesh_axis_names(path)
        if n_cells > 0:
            out[path] = (decisions[cells_axis], decisions[cells_axis])
        else:
            out[path] = (decisions[points_axis], False)
    return out


# ---- selection plan and read ------------------------------------------------


@dataclass(frozen=True)
class MeshSelectionPlan:
    r"""What this rank selects of one (sub-)mesh, decided from global counts only.

    Parameters
    ----------
    shard_points, shard_cells : bool
        Placement of the two batch axes. For a mesh with cells they are equal
        (both follow the ``cells`` axis decision).
    n_points_src, n_cells_src : int
        Point and cell counts of the source mesh.
    point_window : Tensor or None
        Point-subsample cyclic block (point clouds only). The global point
        count is then the window length.
    cell_window : Tensor or None
        Cell-subsample cyclic block (meshes with cells only). Selecting a
        window compacts the point set to the vertices the window references
        -- the same global operation the eager reader performs -- so every
        rank reads the whole window of ``cells`` (integers, cheap) to derive
        the identical referenced set, then reads only its chunk of it.
    measure_factor : float or None
        Horvitz-Thompson factor for a cell window (``n_cells_src / n_cells``).
    device_mesh : DeviceMesh
        1-D mesh the chunks are taken against.
    """

    shard_points: bool
    shard_cells: bool
    n_points_src: int
    n_cells_src: int
    point_window: torch.Tensor | None
    cell_window: torch.Tensor | None
    measure_factor: float | None
    device_mesh: DeviceMesh

    @property
    def n_cells(self) -> int:
        """Number of cells after windowing: the window length, else the source count."""
        return (
            len(self.cell_window) if self.cell_window is not None else self.n_cells_src
        )


def plan_mesh_selection(
    n_points_src: int,
    n_cells_src: int,
    shard_points: bool,
    shard_cells: bool,
    device_mesh: DeviceMesh,
    point_window: torch.Tensor | None = None,
    cell_window: torch.Tensor | None = None,
) -> MeshSelectionPlan:
    r"""Validate and bundle a :class:`MeshSelectionPlan`; see the class for the fields."""
    if n_cells_src > 0 and point_window is not None:
        raise ValueError(
            "point subsampling on a mesh with cells is not supported under "
            "domain parallelism (it would remap connectivity globally); "
            "subsample cells instead, or drop the cells"
        )
    if n_cells_src == 0 and cell_window is not None:
        raise ValueError("cell_window applies to meshes with cells only")
    if n_cells_src > 0:
        shard_points = shard_cells  # vertices follow the cells axis
    return MeshSelectionPlan(
        shard_points=shard_points,
        shard_cells=shard_cells,
        n_points_src=n_points_src,
        n_cells_src=n_cells_src,
        point_window=point_window,
        cell_window=cell_window,
        measure_factor=(
            n_cells_src / len(cell_window) if cell_window is not None else None
        ),
        device_mesh=device_mesh,
    )


def compact_cells(cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""``(referenced_vertex_ids, remapped_cells)``: remap *cells* onto their vertices.

    ``referenced_vertex_ids`` is sorted, matching what ``Mesh.slice_cells`` +
    ``slice_points`` produce; ``remapped_cells`` index into it.
    """
    referenced, inverse = torch.unique(cells, return_inverse=True)
    return referenced, inverse.reshape(cells.shape)


def _chunk(n: int, shard: bool, device_mesh: DeviceMesh) -> slice:
    return slice(*chunk_bounds(n, device_mesh)) if shard else slice(None)


def _select(window: torch.Tensor | None, share: slice) -> IndexSelection:
    return window[share] if window is not None else share


def mesh_selection_to_proto(
    src: MeshSource, plan: MeshSelectionPlan, prefix: tuple[str, ...] = ()
) -> tuple[TensorDict, ShardedMap]:
    r"""Execute a :class:`MeshSelectionPlan` against a lazy mesh or zarr group.

    Three layouts, all even ``torch.chunk``-style splits so the assembly wrap
    needs no communication:

    - point cloud: this rank's share of the (windowed) point range;
    - mesh with cells, no window: this rank's share of the cell range and of
      the point range -- cells keep their global vertex ids;
    - mesh with cells and a cell window: the whole window of ``cells`` is
      read (integers) and compacted onto its referenced vertices identically
      on every rank; this rank then keeps its share of the remapped cells and
      cell data and reads its share of the referenced vertices.

    Returns the nested proto TensorDict and its sharded-leaf map
    (``path -> global shape``). Cells always index the global (compacted)
    vertex space; ``points[cells]`` on the assembled mesh is the routed gather.
    """
    device_mesh = plan.device_mesh
    drop_cells = plan.n_cells_src == 0
    if drop_cells:
        n_points = (
            len(plan.point_window)
            if plan.point_window is not None
            else plan.n_points_src
        )
        p = _select(plan.point_window, _chunk(n_points, plan.shard_points, device_mesh))
        local = _read_mesh_selection(src, p, slice(0, 0), drop_cells=True)
        cells = torch.zeros(0, 1, dtype=torch.long)
    elif plan.cell_window is None:
        n_points = plan.n_points_src
        c = _chunk(plan.n_cells_src, plan.shard_cells, device_mesh)
        p = _chunk(n_points, plan.shard_points, device_mesh)
        local = _read_mesh_selection(src, p, c)
        cells = local.cells
    else:
        window = plan.cell_window
        referenced, remapped = compact_cells(_read_cells(src, window))
        n_points = len(referenced)
        if plan.shard_points and n_points < device_mesh.size(0):
            raise ValueError(
                f"the cell window references only {n_points} vertices, fewer than "
                f"the {device_mesh.size(0)} ranks sharding them; use a larger "
                "subsample_n_cells or replicate this mesh"
            )
        c = _chunk(len(window), plan.shard_cells, device_mesh)
        p_ids = referenced[_chunk(n_points, plan.shard_points, device_mesh)]
        local = _read_mesh_selection(src, p_ids, window[c])
        cells = remapped[c]

    cell_data = local.cell_data
    if plan.measure_factor is not None:
        weights = cell_data.get(MEASURE_WEIGHTS_KEY, None)
        if weights is None:
            weights = torch.ones(cells.shape[0], dtype=local.points.dtype)
        cell_data[MEASURE_WEIGHTS_KEY] = weights * plan.measure_factor

    tensors = TensorDict(
        {
            "points": local.points,
            "cells": cells,
            "point_data": local.point_data,
            "cell_data": cell_data,
            "global_data": local.global_data,
        },
        batch_size=[],
    )

    def leaves(td: TensorDict):
        for key, value in td.items(include_nested=True, leaves_only=True):
            yield as_leaf_key(key), value

    sharded: ShardedMap = {}
    if plan.shard_points:
        sharded[(*prefix, "points")] = (n_points, *local.points.shape[1:])
        for k, v in leaves(local.point_data):
            sharded[(*prefix, "point_data", *k)] = (n_points, *v.shape[1:])
    if plan.shard_cells:
        sharded[(*prefix, "cells")] = (plan.n_cells, *cells.shape[1:])
        for k, v in leaves(cell_data):
            sharded[(*prefix, "cell_data", *k)] = (plan.n_cells, *v.shape[1:])
    return tensors, sharded


# ---- per sub-mesh preparation (phase A: metadata and windows only) ------------


@dataclass
class _PreparedSubmesh:
    """A sub-mesh ready for rank-local reading: source + effective global counts.

    ``src`` is a lazy memmap ``Mesh`` or an open zarr group. ``point_window`` /
    ``cell_window`` are the subsample cyclic blocks (point clouds / meshes
    with cells respectively); the effective global count of the windowed axis
    is the window length.
    """

    src: MeshSource
    n_points: int
    n_cells: int
    point_window: torch.Tensor | None = None
    cell_window: torch.Tensor | None = None


def _prepare_submesh(
    src: MeshSource,
    *,
    n_cells_sub: int | None,
    n_points_sub: int | None,
    generator: torch.Generator | None,
    drop_cells: bool = False,
) -> _PreparedSubmesh:
    """Phase A of a rank-local read: subsample decisions from metadata only.

    Nothing is read here. A mesh with cells draws its cell window; a point
    cloud draws its point window. Point subsampling on a mesh with cells is
    rejected (it remaps connectivity globally). The generator draw order
    matches :func:`_subsample_mesh` (cells, then points), so every rank
    derives the same windows.
    """
    total_points, total_cells = _mesh_counts(src, drop_cells)

    if total_cells > 0:
        if n_points_sub is not None:
            raise NotImplementedError(
                "subsample_n_points on a mesh with cells is not supported under "
                "domain-parallel reading; use subsample_n_cells (or drop the "
                "cells to read the mesh as a point cloud)"
            )
        cell_window = None
        if n_cells_sub is not None and total_cells > n_cells_sub:
            _require_seed(generator)
            cell_window = _cyclic_block_indices(
                total_cells, n_cells_sub, generator=generator
            )
        return _PreparedSubmesh(
            src=src,
            n_points=total_points,
            n_cells=total_cells if cell_window is None else len(cell_window),
            cell_window=cell_window,
        )

    point_window = None
    if n_points_sub is not None and total_points > n_points_sub:
        _require_seed(generator)
        point_window = _cyclic_block_indices(
            total_points, n_points_sub, generator=generator
        )
    return _PreparedSubmesh(
        src=src,
        n_points=total_points if point_window is None else len(point_window),
        n_cells=0,
        point_window=point_window,
    )


def _read_prepared(
    prep: _PreparedSubmesh,
    shard_points: bool,
    shard_cells: bool,
    device_mesh: DeviceMesh,
    prefix: tuple[str, ...] = (),
) -> tuple[TensorDict, ShardedMap]:
    """Phase B: plan this rank's selection and read it into proto tensors."""
    plan = plan_mesh_selection(
        *_mesh_counts(prep.src, drop_cells=prep.n_cells == 0),
        shard_points,
        shard_cells,
        device_mesh,
        point_window=prep.point_window,
        cell_window=prep.cell_window,
    )
    return mesh_selection_to_proto(prep.src, plan, prefix)


# ---- rebuild (device side, after the ShardTensor wrap) --------------------------


def _rebuild_mesh(td: TensorDict) -> Mesh:
    return Mesh(
        points=td["points"],
        cells=td["cells"],
        point_data=td["point_data"],
        cell_data=td["cell_data"],
        global_data=td["global_data"],
    )


def _rebuild_domain_mesh(td: TensorDict) -> DomainMesh:
    boundaries = td["boundaries"]
    return DomainMesh(
        interior=_rebuild_mesh(td["interior"]),
        boundaries={
            name: _rebuild_mesh(boundaries[name]) for name in boundaries.keys()
        },
        global_data=td["global_data"],
    )


register_proto_kind("mesh", _rebuild_mesh)
register_proto_kind("domain_mesh", _rebuild_domain_mesh)


class _MeshReaderBase:
    """State and helpers shared by :class:`MeshReader` and :class:`DomainMeshReader`.

    Per-sample RNG (base seed + epoch), the cached zarr group handles, and
    the domain-parallel configuration.
    """

    def _init_common(
        self, domain_parallel: dict | None, device_mesh: DeviceMesh | None
    ) -> None:
        self._domain_parallel = DomainParallelConfig.from_dict(
            domain_parallel, device_mesh
        )
        # Base seed + epoch for deterministic per-index RNG (see
        # :meth:`set_generator`). ``None`` means unseeded.
        self._seed_base: int | None = None
        self._epoch: int = 0
        self._zarr_groups: dict[Path, Any] = {}

    def _zarr_group(self, path: Path) -> Any:
        """Open (and cache) the zarr group at *path*.

        Re-opening walks the store's group-metadata chain, and on networked
        filesystems every uncached lookup is a metadata-server round-trip
        per draw.
        """
        group = self._zarr_groups.get(path)
        if group is None:
            group = self._zarr_groups[path] = io_zarr._open_group(path)
        return group

    def _generator(self, index: int) -> torch.Generator | None:
        """Per-sample generator from ``(base_seed, epoch, index)``; ``None`` if unseeded."""
        return (
            None
            if self._seed_base is None
            else spawn_generator(self._seed_base, self._epoch, index)
        )

    def set_generator(self, generator: torch.Generator) -> None:
        """Assign a base seed for reproducible, order-independent subsampling.

        Called by :class:`MeshDataset` when the DataLoader provides a
        seed.  Stores ``generator.initial_seed()`` as the base seed; each
        sample then derives its own generator from
        ``(base_seed, epoch, index)``, so subsampling is reproducible
        regardless of read order or worker thread. Required for
        domain-parallel subsampling, where every rank must draw the same
        window.

        Parameters
        ----------
        generator : torch.Generator
            Generator whose ``initial_seed()`` seeds all per-sample RNG.
        """
        self._seed_base = generator.initial_seed()

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to vary per-sample RNG deterministically.

        The epoch is folded into each sample's derived seed, producing a
        different (but deterministic) sequence of contiguous blocks each
        epoch when a base seed has been assigned via :meth:`set_generator`.
        """
        self._epoch = epoch

    def close(self) -> None:
        """Release cached zarr store handles (``MeshDataset.close`` calls this)."""
        self._zarr_groups.clear()


@register()
class MeshReader(_MeshReaderBase):
    r"""
    Read single-mesh samples from directories of physicsnemo mesh files.

    Each sample is one Mesh. Returns (Mesh, metadata) per index.
    Uses Mesh.load(path) for physicsnemo mesh format (.pmsh).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        pattern: str = f"**/*{DEFAULT_MESH_EXTENSION}",
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
        subsample_n_points: int | None = None,
        subsample_n_cells: int | None = None,
        domain_parallel: dict | None = None,
        device_mesh: "torch.distributed.device_mesh.DeviceMesh | None" = None,
    ) -> None:
        """
        Initialize the mesh reader.

        Parameters
        ----------
        path : Path or str
            Root directory containing mesh files (e.g. .pmsh directories).
        pattern : str, optional
            Glob pattern for mesh paths under ``path``. Default matches ``**/*.pmsh``.
        pin_memory : bool, default=False
            If True, place tensors in pinned (page-locked) memory for faster
            async CPU→GPU transfers.
        include_index_in_metadata : bool, default=True
            If True, include sample index in metadata.
        subsample_n_points : int, optional
            If set, subsample the mesh to this many points *before*
            ``pin_memory``.  Uses cyclic contiguous block reads for
            page-sequential I/O on memmap-backed data, with uniform point
            inclusion probability.  Appropriate for point clouds
            or meshes where cell topology is not needed downstream.
            For best results, pre-shuffle the on-disk point order so
            that a contiguous block is spatially representative.
        subsample_n_cells : int, optional
            If set, subsample the mesh to this many cells *before*
            ``pin_memory``.  Uses cyclic contiguous block reads on the
            cell tensor for sequential I/O, then compacts unreferenced
            vertices.  Preserves cell topology and is the correct
            choice for triangulated surface meshes where downstream
            transforms depend on cells (e.g. surface normals, cell
            centroids, cell_data fields).  Records the inverse inclusion
            probability as measure weights, preserving the integration
            measure (see :mod:`physicsnemo.mesh.calculus.measure`).  Applied before
            ``subsample_n_points`` when both are set.
        domain_parallel : dict, optional
            Optional dict to configure domain-parallel (rank-local)
            reading; see :mod:`physicsnemo.datapipes._domain_parallel` for
            the schema. The mesh has two batch axes, ``points`` (points +
            point_data) and ``cells`` (cell_data), each gated by
            ``auto_shard_size`` or pinned via ``placements``
            (``{"points": "shard"}``). Composes with the ``subsample_*``
            options: a point cloud reads only this rank's chunk of the
            (rank-consistent) subsample window; a mesh with cells is
            subsampled in full first, since cell subsampling compacts the
            point set globally. The sample returns as a proto payload that
            ``MeshDataset`` assembles into a ShardTensor-backed ``Mesh`` on
            the GPU.
        device_mesh : torch.distributed.device_mesh.DeviceMesh, optional
            1-D device mesh for domain-parallel reading; required with
            ``domain_parallel``. Constructed and injected at runtime.
        """
        self._root = Path(path)
        self._pattern = pattern
        self.pin_memory = pin_memory
        self.include_index_in_metadata = include_index_in_metadata
        self.subsample_n_points = subsample_n_points
        self.subsample_n_cells = subsample_n_cells
        self._init_common(domain_parallel, device_mesh)

        if not self._root.exists():
            raise FileNotFoundError(f"Path not found: {self._root}")
        if not self._root.is_dir():
            raise ValueError(f"Path must be a directory: {self._root}")

        # glob.glob instead of Path.glob: the latter re-stats each entry and
        # silently drops entries under Lustre metadata-server load.
        self._paths = sorted(
            Path(p) for p in _glob.glob(str(self._root / pattern), recursive=True)
        )
        if not self._paths:
            raise ValueError(f"No paths matching {pattern!r} found in {self._root}")

    def _load_sample(self, index: int) -> Mesh:
        """Load a single Mesh from disk."""
        mesh_path = self._paths[index]
        if (mesh_path / "zarr.json").exists():
            from physicsnemo.mesh.io import from_zarr

            if (
                self.subsample_n_cells is not None
                or self.subsample_n_points is not None
            ):
                # Push the subsample into the read: fetch only the selected
                # window from the store. The generator derivation matches
                # __getitem__, so the draw is identical to subsampling after
                # an eager load (whose subsample then no-ops).
                return _zarr_mesh_subsampled(
                    self._zarr_group(mesh_path),
                    self.subsample_n_cells,
                    self.subsample_n_points,
                    self._generator(index),
                )
            return from_zarr(mesh_path)
        return Mesh.load(mesh_path)

    def _load_domain_parallel(self, index: int) -> ShardedProto:
        """Rank-local read: this rank's share of the (windowed) mesh as a proto."""
        mesh_path = self._paths[index]
        sub_kw = dict(
            n_cells_sub=self.subsample_n_cells,
            n_points_sub=self.subsample_n_points,
            generator=self._generator(index),
        )
        if (mesh_path / "zarr.json").exists():
            prep = _prepare_submesh(self._zarr_group(mesh_path), **sub_kw)
        else:
            # Through ``_load_sample`` (a lazy memmap load) so subclass hooks
            # that enrich the sample (e.g. merging external global_data)
            # apply under domain-parallel reading too.
            prep = _prepare_submesh(self._load_sample(index), **sub_kw)

        device_mesh = self._domain_parallel.device_mesh
        (shard_points, shard_cells) = resolve_mesh_placements(
            {"": (prep.n_points, prep.n_cells)}, self._domain_parallel
        )[""]
        tensors, sharded = _read_prepared(prep, shard_points, shard_cells, device_mesh)
        return ShardedProto(
            tensors=tensors, sharded=sharded, device_mesh=device_mesh, kind="mesh"
        )

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return metadata for the sample (e.g. source path)."""
        return {"source_path": str(self._paths[index])}

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> tuple[Mesh, dict[str, Any]]:
        metadata = self._get_sample_metadata(index)
        if self.include_index_in_metadata:
            metadata["index"] = index

        if self._domain_parallel is not None:
            # Rank-local read: only this rank's share leaves the store. The
            # window is rank-consistent by the (seed, epoch, index) RNG
            # scheme; the dataset assembles the ShardTensors on the GPU.
            proto = self._load_domain_parallel(index)
            return (proto.pin_memory() if self.pin_memory else proto), metadata

        mesh = self._load_sample(index)
        mesh = _subsample_mesh(
            mesh,
            self.subsample_n_cells,
            self.subsample_n_points,
            generator=self._generator(index),
        )
        if self.pin_memory:
            mesh = mesh.pin_memory()
        return mesh, metadata

    def __iter__(self) -> Iterator[tuple[Mesh, dict[str, Any]]]:
        for i in range(len(self)):
            try:
                yield self[i]
            except Exception as e:
                logger.error("Sample %s failed: %s", i, e)
                raise RuntimeError(f"Sample {i} failed: {e}") from e

    def __repr__(self) -> str:
        return f"MeshReader(path={self._root!r}, len={len(self)})"


@register()
class DomainMeshReader(_MeshReaderBase):
    r"""
    Read DomainMesh samples from a directory of physicsnemo mesh files.

    Each sample is one DomainMesh (interior + named boundaries + global_data).
    Returns (DomainMesh, metadata) per index.
    Uses DomainMesh.load(path) for physicsnemo mesh format (.pdmsh).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        pattern: str = f"**/*{DEFAULT_DOMAIN_MESH_EXTENSION}",
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
        subsample_n_points: int | None = None,
        subsample_n_cells: int | None = None,
        extra_boundaries: dict[str, dict] | None = None,
        drop_interior_cells: bool = False,
        drop_in_file_boundaries: bool = False,
        domain_parallel: dict | None = None,
        device_mesh: "torch.distributed.device_mesh.DeviceMesh | None" = None,
    ) -> None:
        """
        Initialize the domain mesh reader.

        Parameters
        ----------
        path : Path or str
            Root directory containing DomainMesh files (e.g. .pdmsh archives).
        pattern : str, optional
            Glob pattern for DomainMesh paths under ``path``.
            Default matches ``**/*.pdmsh``.
        pin_memory : bool, default=False
            If True, place tensors in pinned (page-locked) memory for faster
            async CPU→GPU transfers.
        include_index_in_metadata : bool, default=True
            If True, include sample index in metadata.
        subsample_n_points : int, optional
            If set, subsample the interior and each boundary mesh to
            at most this many points *before* ``pin_memory``.  Uses
            cyclic contiguous block reads for page-sequential I/O on
            memmap-backed data, with uniform point inclusion probability.
            Appropriate for point clouds or meshes where cell topology is
            not needed downstream.  For best results,
            pre-shuffle the on-disk point order so that a contiguous
            block is spatially representative.
        subsample_n_cells : int, optional
            If set, subsample the interior and each boundary mesh to
            at most this many cells *before* ``pin_memory``.  Uses
            cyclic contiguous block reads on cell tensors for
            sequential I/O, then compacts unreferenced vertices.
            Preserves cell topology and is the correct choice when
            downstream transforms depend on cells.  Records the
            inverse inclusion probability as measure weights, preserving
            the integration measure (see
            :mod:`physicsnemo.mesh.calculus.measure`).  Applied
            before
            ``subsample_n_points`` when both are set.
        extra_boundaries : dict[str, dict] or None, optional
            Load additional sibling meshes as extra boundaries on each
            sample.  Each key is the boundary name to assign; each value
            is a dict with a ``"pattern"`` key giving a glob pattern
            (relative to the sample's parent directory) to find the mesh
            file.  These meshes are loaded at full resolution and are
            **not** subsampled, making them suitable for geometric
            queries like SDF computation.

            Example::

                extra_boundaries:
                  stl_geometry:
                    pattern: "*_single_solid.stl.pmsh"
        drop_interior_cells : bool, default=False
            If True, discard the interior mesh's cell connectivity (and
            cell_data) immediately after load, turning it into a point
            cloud.  This makes ``subsample_n_points`` take the cheap
            contiguous-block path instead of the expensive
            ``slice_points`` remap (which allocates an ``n_points`` map
            and scatter-reads the full cell array from the memmap).  Use
            for point-based models that consume only ``interior.points``
            and ``interior.point_data`` (e.g. GeoTransolver volume) and
            never the interior tet/cell topology.  Boundaries are
            unaffected, so surface normals etc. still work.
        drop_in_file_boundaries : bool, default=False
            If True, discard the boundaries stored *in* the DomainMesh
            file immediately after load (before subsampling and pinning).
            ``extra_boundaries`` are added afterwards and are therefore
            unaffected.  Use when the model consumes only the interior
            (plus any ``extra_boundaries``) and never the in-file
            boundaries -- e.g. a volume pipeline whose SDF comes from an
            injected STL, where the in-file car-surface boundary would
            otherwise be subsampled (an expensive ``slice_points`` remap,
            GIL-held, that blocks worker-thread overlap) and pinned every
            sample for nothing.
        domain_parallel : dict, optional
            Optional dict to configure domain-parallel (rank-local)
            reading; see :class:`MeshReader`. Every sub-mesh's two batch
            axes are gated independently and addressable in
            ``placements`` by path: ``interior.points``,
            ``boundaries.<name>.cells``, or ``boundaries.<name>`` for both.
            ``extra_boundaries`` are pinned to replicate by default (they
            exist for whole-geometry queries such as SDF) but an explicit
            ``placements`` entry overrides that.
        device_mesh : torch.distributed.device_mesh.DeviceMesh, optional
            1-D device mesh for domain-parallel reading; required with
            ``domain_parallel``. Constructed and injected at runtime.
        """
        self._root = Path(path)
        self._pattern = pattern
        self.pin_memory = pin_memory
        self.include_index_in_metadata = include_index_in_metadata
        self.drop_interior_cells = drop_interior_cells
        self.drop_in_file_boundaries = drop_in_file_boundaries
        self.subsample_n_points = subsample_n_points
        self.subsample_n_cells = subsample_n_cells
        self._init_common(domain_parallel, device_mesh)
        self._extra_boundaries = extra_boundaries or {}

        if not self._root.exists():
            raise FileNotFoundError(f"Path not found: {self._root}")
        if not self._root.is_dir():
            raise ValueError(f"Path must be a directory: {self._root}")

        # glob.glob instead of Path.glob: the latter re-stats each entry and
        # silently drops entries under Lustre metadata-server load.
        self._paths = sorted(
            Path(p) for p in _glob.glob(str(self._root / pattern), recursive=True)
        )
        if not self._paths:
            raise ValueError(f"No paths matching {pattern!r} found in {self._root}")

    def _load_sample(self, index: int) -> DomainMesh:
        """Load a single DomainMesh from disk."""
        path = self._paths[index]
        if (path / "zarr.json").exists():
            from physicsnemo.mesh.io import from_zarr

            if (
                self.subsample_n_cells is not None
                or self.subsample_n_points is not None
            ):
                # Push the subsample into the read (window reads per
                # sub-mesh); drop flags are honored at read time so skipped
                # data is never fetched. Generator derivation and sub-mesh
                # order match __getitem__, whose subsample then no-ops.
                generator = self._generator(index)
                root = self._zarr_group(path)
                interior = _zarr_mesh_subsampled(
                    root["interior"],
                    self.subsample_n_cells,
                    self.subsample_n_points,
                    generator,
                    drop_cells=self.drop_interior_cells,
                )
                boundaries = {}
                if not self.drop_in_file_boundaries and "boundaries" in root:
                    boundaries = {
                        name: _zarr_mesh_subsampled(
                            grp,
                            self.subsample_n_cells,
                            self.subsample_n_points,
                            generator,
                        )
                        for name, grp in root["boundaries"].groups()
                    }
                return DomainMesh(
                    interior=interior,
                    boundaries=boundaries,
                    global_data=io_zarr._read_tree(root, "global_data"),
                )
            return from_zarr(path)
        return DomainMesh.load(path)

    def __len__(self) -> int:
        return len(self._paths)

    def _load_domain_parallel(self, index: int) -> ShardedProto:
        """Rank-local read of every sub-mesh; one proto for the whole domain.

        Phase A prepares each sub-mesh (subsample decisions, window draws)
        from metadata in the same order as the eager path draws its
        generator; placements are then resolved jointly from the effective
        global counts; phase B reads only this rank's share. Extra boundaries
        are pinned to replicate unless ``placements`` says otherwise.
        """
        path = self._paths[index]
        generator = self._generator(index)
        sub_kw = dict(
            n_cells_sub=self.subsample_n_cells,
            n_points_sub=self.subsample_n_points,
            generator=generator,
        )

        prepared: dict[str, _PreparedSubmesh] = {}
        if (path / "zarr.json").exists():
            root = self._zarr_group(path)
            prepared["interior"] = _prepare_submesh(
                root["interior"], drop_cells=self.drop_interior_cells, **sub_kw
            )
            if not self.drop_in_file_boundaries and "boundaries" in root:
                for name, grp in root["boundaries"].groups():
                    prepared[f"boundaries.{name}"] = _prepare_submesh(grp, **sub_kw)
            global_data = io_zarr._read_tree(root, "global_data")
        else:
            # Through ``_load_sample`` (a lazy memmap load) so subclass hooks
            # that enrich the sample apply under domain-parallel reading too.
            domain = self._load_sample(index)
            prepared["interior"] = _prepare_submesh(
                domain.interior, drop_cells=self.drop_interior_cells, **sub_kw
            )
            if not self.drop_in_file_boundaries:
                for name in domain.boundary_names:
                    prepared[f"boundaries.{name}"] = _prepare_submesh(
                        domain.boundaries[name], **sub_kw
                    )
            global_data = domain.global_data

        # Extra boundaries: full resolution, never subsampled, replicate by
        # default -- they exist for whole-geometry queries (e.g. SDF).
        pinned: dict[str, PlacementName] = {}
        for name, mesh in self._load_extra_boundary_meshes(index).items():
            prepared[f"boundaries.{name}"] = _PreparedSubmesh(
                src=mesh, n_points=mesh.n_points, n_cells=mesh.n_cells
            )
            pinned[f"boundaries.{name}"] = "replicate"

        device_mesh = self._domain_parallel.device_mesh
        decisions = resolve_mesh_placements(
            {p: (prep.n_points, prep.n_cells) for p, prep in prepared.items()},
            self._domain_parallel,
            pinned=pinned,
        )

        sharded: ShardedMap = {}
        boundaries: dict[str, TensorDict] = {}
        interior = None
        for p, prep in prepared.items():
            shard_points, shard_cells = decisions[p]
            prefix = tuple(p.split("."))
            td, sub = _read_prepared(
                prep, shard_points, shard_cells, device_mesh, prefix
            )
            sharded.update(sub)
            if p == "interior":
                interior = td
            else:
                boundaries[prefix[1]] = td

        tensors = TensorDict(
            {
                "interior": interior,
                "boundaries": TensorDict(boundaries, batch_size=[]),
                # Nesting-aware: global_data may hold sub-TensorDicts.
                "global_data": _read_leaves(global_data),
            },
            batch_size=[],
        )
        return ShardedProto(
            tensors=tensors,
            sharded=sharded,
            device_mesh=device_mesh,
            kind="domain_mesh",
        )

    def __getitem__(self, index: int) -> tuple[DomainMesh, dict[str, Any]]:
        if self._domain_parallel is not None:
            proto = self._load_domain_parallel(index)
            metadata: dict[str, Any] = {
                "source_path": str(self._paths[index]),
                "boundary_names": sorted(proto.tensors["boundaries"].keys()),
            }
            if self.include_index_in_metadata:
                metadata["index"] = index
            return (proto.pin_memory() if self.pin_memory else proto), metadata

        dm = self._load_sample(index)

        # Trim unused data before subsample/pin. Both references are lazy (no
        # memmap materialization here):
        #  - drop_interior_cells: turn the interior into a point cloud so its
        #    point subsample takes the cheap contiguous-block path instead of a
        #    full slice_points remap + scattered reads.
        #  - drop_in_file_boundaries: skip the in-file boundaries entirely so we
        #    don't subsample (an expensive, GIL-held slice_points remap that
        #    starves worker-thread overlap) or pin a surface the model ignores.
        if (self.drop_interior_cells and dm.interior.n_cells > 0) or (
            self.drop_in_file_boundaries and len(dm.boundary_names) > 0
        ):
            interior = dm.interior
            if self.drop_interior_cells and interior.n_cells > 0:
                interior = Mesh(
                    points=interior.points,
                    point_data=interior.point_data,
                    global_data=interior.global_data,
                )
            boundaries = {} if self.drop_in_file_boundaries else dm.boundaries
            dm = DomainMesh(
                interior=interior,
                boundaries=boundaries,
                global_data=dm.global_data,
            )

        if self.subsample_n_cells is not None or self.subsample_n_points is not None:
            generator = self._generator(index)
            sub_kw = dict(
                n_cells=self.subsample_n_cells,
                n_points=self.subsample_n_points,
                generator=generator,
            )
            interior = _subsample_mesh(dm.interior, **sub_kw)
            boundaries = {
                name: _subsample_mesh(dm.boundaries[name], **sub_kw)
                for name in dm.boundary_names
            }
            dm = DomainMesh(
                interior=interior,
                boundaries=boundaries,
                global_data=dm.global_data,
            )

        # Load extra boundary meshes (full resolution, no subsampling).
        if self._extra_boundaries:
            dm = self._load_extra_boundaries(dm, index)

        metadata: dict[str, Any] = {
            "source_path": str(self._paths[index]),
            "boundary_names": dm.boundary_names,
        }
        if self.include_index_in_metadata:
            metadata["index"] = index

        if self.pin_memory:
            dm = dm.pin_memory()

        return dm, metadata

    def _load_extra_boundaries(self, dm: DomainMesh, index: int) -> DomainMesh:
        """Attach the sibling meshes from :meth:`_load_extra_boundary_meshes`."""
        return DomainMesh(
            interior=dm.interior,
            boundaries={
                **dict(dm.boundaries),
                **self._load_extra_boundary_meshes(index),
            },
            global_data=dm.global_data,
        )

    def _load_extra_boundary_meshes(self, index: int) -> dict[str, Mesh]:
        """Find and load sibling meshes configured as extra boundaries.

        Extra boundaries are loaded at full resolution (no subsampling)
        so they are suitable for geometric queries like SDF computation.
        """
        case_dir = Path(self._paths[index]).parent
        new_boundaries: dict[str, Mesh] = {}

        for bnd_name, bnd_cfg in self._extra_boundaries.items():
            glob_pattern = bnd_cfg["pattern"]
            matches = sorted(case_dir.glob(glob_pattern))
            if not matches:
                raise FileNotFoundError(
                    f"No mesh matching {glob_pattern!r} found in "
                    f"{case_dir} for extra boundary {bnd_name!r}"
                )
            if len(matches) > 1:
                logger.warning(
                    "Multiple meshes found for extra boundary %r in %s "
                    "matching %r; using %s",
                    bnd_name,
                    case_dir,
                    glob_pattern,
                    matches[0],
                )
            if (matches[0] / "zarr.json").exists():
                from physicsnemo.mesh.io import from_zarr

                new_boundaries[bnd_name] = from_zarr(matches[0])
            else:
                new_boundaries[bnd_name] = Mesh.load(matches[0])

        return new_boundaries

    def __iter__(self) -> Iterator[tuple[DomainMesh, dict[str, Any]]]:
        for i in range(len(self)):
            try:
                yield self[i]
            except Exception as e:
                logger.error("Sample %s failed: %s", i, e)
                raise RuntimeError(f"Sample {i} failed: {e}") from e

    def __repr__(self) -> str:
        return f"DomainMeshReader(path={self._root!r}, len={len(self)})"
