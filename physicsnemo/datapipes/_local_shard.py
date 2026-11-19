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

r"""The ``"local"`` domain-parallel layout of a mesh with cells.

Under ``domain_parallel={"mesh_layout": "local"}`` (the ``MeshReader``
default), each rank of the 1-D device mesh reads its contiguous
``torch.chunk``-style share of the (windowed) cells together with exactly the
vertices those cells reference, renumbered so that its cells index its own
points. The reader returns that plain, rank-local :class:`~physicsnemo.mesh.Mesh`
in a :class:`LocalMeshShard`; ``MeshDataset`` runs the transform chain on it
with plain tensors (no ShardTensor dispatch, no collectives) and only then
wraps the result into ``Shard(0)`` ShardTensors with explicit shard shapes
(:func:`wrap_local_sample`, communication-free).

Layout of the wrapped sample
----------------------------
- **Cell axis**: ``cells``, ``cell_data`` and every per-cell output of the
  transforms (cell centroids as an interior point cloud, a
  ``[n_cells, C]`` feature tensor, ...) are ``Shard(0)`` over the global
  (windowed) cell count with ``torch.chunk`` shard sizes, exactly the rows of
  the unsharded sample.
- **Point axis**: ``points`` / ``point_data`` are each rank's referenced
  vertices, sorted by their global id. Vertices shared by cells on two ranks
  (the seam) appear on both, so the global point axis is the concatenation of
  the per-rank vertex sets, *not* the unsharded point set: reductions over the
  points count seam vertices once per rank holding them. ``cells`` are offset
  into this concatenated axis, so ``points[cells]`` on the wrapped mesh still
  resolves every cell's own vertices (the routed gather finds them on the
  same rank).

Transforms
----------
Transforms see the plain local mesh. Per-point and per-cell transforms
(field ops, normalization, affine maps, augmentations whose parameters come
from the rank-identical generators, cell normals / centroids / areas,
``MeshToDomainMesh``) give exactly the rows of the unsharded result. A
transform that needs a statistic over the whole mesh implements
``required_shard_stats`` / ``apply_local_shard`` (see
:class:`~physicsnemo.datapipes.transforms.mesh.base.MeshTransform`); the
reader computes the requested statistics for every rank from the same read,
so no rank ever communicates. :class:`~physicsnemo.datapipes.transforms.mesh.CenterMesh`
is the built-in user. Transforms that aggregate over neighbouring cells of
a vertex (point normals) or draw a global random subset (``SubsampleMesh``
below the sample size) cannot run locally and raise; use
``mesh_layout: "global"`` for those chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from tensordict import TensorDict
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.mesh import DomainMesh, Mesh

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

# Whole-mesh statistics a local-layout reader can compute for the transforms.
# ``point_mean``: mean of the sample's points (the window's referenced
# vertices under cell subsampling). ``area_weighted_center``: cell-area
# weighted mean of the cell centroids. Both are what ``CenterMesh`` computes
# on the unsharded sample.
POINT_MEAN = "point_mean"
AREA_WEIGHTED_CENTER = "area_weighted_center"
SHARD_STATS = frozenset({POINT_MEAN, AREA_WEIGHTED_CENTER})

Axis = Literal["cells", "points"]


# ---------------------------------------------------------------------------
# Row reads
# ---------------------------------------------------------------------------


def read_rows(
    t: torch.Tensor, selection: slice | torch.Tensor | None = None
) -> torch.Tensor:
    r"""``t[selection]`` (all rows for ``None``) materialized in host memory.

    Parameters
    ----------
    t : torch.Tensor
        Source tensor, typically a lazily loaded memmap leaf.
    selection : slice or Tensor or None
        Slice, 1-D integer index, or ``None`` for every row.

    Returns
    -------
    torch.Tensor
        A plain tensor owning its memory.
    """
    return torch.as_tensor(t if selection is None else t[selection]).clone()


# ---------------------------------------------------------------------------
# Host payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalShardInfo:
    r"""Global context of one rank's local share of a mesh with cells.

    Parameters
    ----------
    device_mesh : DeviceMesh
        1-D device mesh the sample is sharded over.
    rank : int
        This rank's index on ``device_mesh``.
    cell_sizes : tuple[int, ...]
        Cells per rank (``torch.chunk``-style split of the global cell count).
    point_sizes : tuple[int, ...]
        Referenced vertices per rank (seam vertices counted on every rank).
    n_points : int
        Point count of the unsharded sample (the window's referenced vertices
        under cell subsampling, else every point of the source).
    """

    device_mesh: DeviceMesh
    rank: int
    cell_sizes: tuple[int, ...]
    point_sizes: tuple[int, ...]
    n_points: int

    @property
    def n_cells(self) -> int:
        """Global (windowed) cell count."""
        return sum(self.cell_sizes)


@dataclass(frozen=True)
class LocalMeshShard:
    r"""Host payload of a ``"local"``-layout read: a plain rank-local mesh.

    Parameters
    ----------
    mesh : Mesh
        This rank's cells with the vertices they reference; ``cells`` index
        ``points`` locally. Cell-subsampling measure corrections are applied.
    info : LocalShardInfo
        Global sizes needed to wrap the transformed sample.
    stats : TensorDict
        Whole-mesh statistics requested by the transforms (see
        :data:`SHARD_STATS`), identical on every rank.
    """

    mesh: Mesh
    info: LocalShardInfo
    stats: TensorDict

    def to(self, device: torch.device, non_blocking: bool = False) -> LocalMeshShard:
        r"""Return a copy with the mesh and statistics on *device*."""
        return LocalMeshShard(
            mesh=self.mesh.to(device, non_blocking=non_blocking),
            info=self.info,
            stats=self.stats.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> LocalMeshShard:
        r"""Return a copy with the mesh and statistics in pinned host memory."""
        return LocalMeshShard(
            mesh=self.mesh.pin_memory(), info=self.info, stats=self.stats.pin_memory()
        )


# ---------------------------------------------------------------------------
# Wrap: transformed local sample -> ShardTensor-backed sample
# ---------------------------------------------------------------------------


def _shard(t: torch.Tensor, sizes: tuple[int, ...], info: LocalShardInfo):
    trailing = tuple(t.shape[1:])
    return ShardTensor.from_local(
        t,
        info.device_mesh,
        (Shard(0),),
        sharding_shapes={0: [(n, *trailing) for n in sizes]},
    )


def _sizes(axis: Axis, info: LocalShardInfo) -> tuple[int, ...]:
    return info.cell_sizes if axis == "cells" else info.point_sizes


def _axis_of_length(n: int, info: LocalShardInfo, what: str) -> Axis | None:
    """The axis a dim-0 length belongs to on this rank, ``None`` for neither."""
    is_cells = n == info.cell_sizes[info.rank]
    is_points = n == info.point_sizes[info.rank]
    if is_cells and is_points:
        raise ValueError(
            f"local mesh layout: cannot tell whether {what} (length {n}) is per "
            f"cell or per point: rank {info.rank} holds {n} cells and {n} points. "
            'Use domain_parallel={"mesh_layout": "global"} for this dataset.'
        )
    if is_cells:
        return "cells"
    if is_points:
        return "points"
    return None


def _wrap_tree(td: TensorDict, axis: Axis | None, info: LocalShardInfo) -> TensorDict:
    """Wrap every leaf on *axis*; ``None`` classifies each leaf by its length."""
    out: dict[str, Any] = {}
    for key, value in td.items():
        if isinstance(value, TensorDict):
            out[key] = _wrap_tree(value, axis, info)
            continue
        leaf_axis = axis
        if leaf_axis is None and value.ndim > 0:
            leaf_axis = _axis_of_length(value.shape[0], info, repr(key))
        out[key] = (
            value if leaf_axis is None else _shard(value, _sizes(leaf_axis, info), info)
        )
    return TensorDict(out, batch_size=[], device=td.device)


def _mesh_axes(
    n_points: int, n_cells: int, info: LocalShardInfo, what: str
) -> tuple[Axis | None, bool]:
    """``(points_axis, cells_sharded)`` of a (sub-)mesh of the transformed sample."""
    if n_cells > 0:
        cells, points = info.cell_sizes[info.rank], info.point_sizes[info.rank]
        if n_cells != cells or n_points != points:
            raise ValueError(
                f"local mesh layout: {what} has {n_cells} cells and {n_points} "
                f"points, but this rank's share has {cells} cells and {points} "
                "points; a transform changed the shard's cells, which the local "
                'layout cannot follow. Use domain_parallel={"mesh_layout": '
                '"global"} for this transform chain.'
            )
        return "points", True
    # Point cloud: cell centroids (cell axis), vertices (point axis) or
    # anything else (replicated).
    return _axis_of_length(n_points, info, f"{what} points"), False


def _wrap_mesh(mesh: Mesh, info: LocalShardInfo, what: str) -> Mesh:
    points_axis, cells_sharded = _mesh_axes(mesh.n_points, mesh.n_cells, info, what)
    point_offset = sum(info.point_sizes[: info.rank])

    def points_leaf(t):
        return t if points_axis is None else _shard(t, _sizes(points_axis, info), info)

    return Mesh(
        points=points_leaf(mesh.points),
        # Offset into the concatenated point axis: each cell's vertices are
        # rows of its own rank's points.
        cells=(
            _shard(mesh.cells + point_offset, info.cell_sizes, info)
            if cells_sharded
            else mesh.cells
        ),
        point_data=(
            mesh.point_data
            if points_axis is None
            else _wrap_tree(mesh.point_data, points_axis, info)
        ),
        cell_data=(
            _wrap_tree(mesh.cell_data, "cells", info)
            if cells_sharded
            else mesh.cell_data
        ),
        global_data=mesh.global_data,
    )


def _wrap_tensordict(td: TensorDict, info: LocalShardInfo) -> TensorDict:
    """A ``MeshToTensorDict``-style layout is wrapped like a mesh; other leaves by length."""
    keys = set(td.keys())
    if "points" in keys and not isinstance(td["points"], TensorDict):
        cells = td.get("cells", None)
        n_cells = 0 if cells is None else cells.shape[0]
        points_axis, cells_sharded = _mesh_axes(
            td["points"].shape[0], n_cells, info, "tensordict"
        )
        point_offset = sum(info.point_sizes[: info.rank])
        out: dict[str, Any] = {}
        for key, value in td.items():
            if key == "global_data":
                out[key] = value
            elif key == "points":
                out[key] = (
                    value
                    if points_axis is None
                    else _shard(value, _sizes(points_axis, info), info)
                )
            elif key == "cells":
                out[key] = (
                    _shard(value + point_offset, info.cell_sizes, info)
                    if cells_sharded
                    else value
                )
            elif key == "point_data" and isinstance(value, TensorDict):
                out[key] = (
                    value
                    if points_axis is None
                    else _wrap_tree(value, points_axis, info)
                )
            elif key == "cell_data" and isinstance(value, TensorDict):
                out[key] = _wrap_tree(value, "cells", info) if cells_sharded else value
            elif isinstance(value, TensorDict):
                out[key] = _wrap_tensordict(value, info)
            else:
                out[key] = _wrap_tree(TensorDict({key: value}), None, info)[key]
        return TensorDict(out, batch_size=[], device=td.device)
    out = {}
    for key, value in td.items():
        if key == "global_data":
            out[key] = value
        elif isinstance(value, TensorDict):
            out[key] = _wrap_tensordict(value, info)
        else:
            out[key] = _wrap_tree(TensorDict({key: value}), None, info)[key]
    return TensorDict(out, batch_size=[], device=td.device)


def wrap_local_sample(sample: Any, info: LocalShardInfo) -> Any:
    r"""Wrap a transformed local sample into ``Shard(0)`` ShardTensors.

    Communication-free: every rank knows every rank's cell and point counts
    (:class:`LocalShardInfo`). Classification is structural where the
    structure says it: a (sub-)mesh with cells must be this rank's shard (its
    ``cells`` / ``cell_data`` go on the cell axis, ``points`` /
    ``point_data`` on the point axis); a point cloud and a free-standing
    leaf go on the axis whose local length they have (cell centroids on the
    cell axis, vertices on the point axis), and stay plain (replicated) when
    they match neither. ``global_data`` stays plain. Mesh caches are dropped.

    Parameters
    ----------
    sample : Mesh or DomainMesh or TensorDict
        Output of the transform chain on the local mesh.
    info : LocalShardInfo
        The read's global sizes.

    Returns
    -------
    Mesh or DomainMesh or TensorDict
        The same structure with sharded leaves wrapped.

    Raises
    ------
    ValueError
        If a mesh with cells is not this rank's shard (a transform changed
        the cells), or a length is ambiguous (a rank holds as many points as
        cells).
    TypeError
        For any other sample type.
    """
    if isinstance(sample, (Mesh, DomainMesh, TensorDict)):
        # Sharded leaves land on the device mesh's device type (as
        # ``ShardTensor.from_local`` moves them); keep the plain ones beside them.
        device_type = info.device_mesh.device_type
        if isinstance(sample, Mesh):
            leaf = sample.points
        elif isinstance(sample, DomainMesh):
            leaf = sample.interior.points
        else:
            leaf = next(iter(sample.values(True, True)), None)
        if leaf is not None and leaf.device.type != device_type:
            sample = sample.to(device_type)
    if isinstance(sample, Mesh):
        return _wrap_mesh(sample, info, "the mesh")
    if isinstance(sample, DomainMesh):
        return DomainMesh(
            interior=_wrap_mesh(sample.interior, info, "the interior"),
            boundaries={
                name: _wrap_mesh(sample.boundaries[name], info, f"boundary {name!r}")
                for name in sample.boundary_names
            },
            global_data=sample.global_data,
        )
    if isinstance(sample, TensorDict):
        return _wrap_tensordict(sample, info)
    raise TypeError(
        "local mesh layout: the transform chain must return a Mesh, DomainMesh "
        f"or TensorDict to be wrapped into ShardTensors, got {type(sample).__name__}"
    )
