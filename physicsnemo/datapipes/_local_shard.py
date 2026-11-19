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

Each rank reads its ``torch.chunk``-style share of the (windowed) cells plus
the vertices they reference, renumbered locally, as a plain
:class:`LocalMeshShard`. ``MeshDataset`` runs the transforms on it without
ShardTensor dispatch, then wraps the result into ``Shard(0)`` ShardTensors
(:func:`wrap_local_sample`, communication-free).

Cells and per-cell outputs are ``Shard(0)`` over the global cell count. The
point axis is the concatenation of each rank's referenced vertices (sorted by
global id): seam vertices appear on every rank holding them, so reductions over
points count them once per rank. ``cells`` are offset into that axis, so
``points[cells]`` still resolves locally. Transforms needing whole-mesh
statistics implement ``required_shard_stats`` / ``apply_local_shard``; point
normals and random ``SubsampleMesh`` cannot run locally (use ``"global"``).
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

# Whole-mesh statistics a local-layout reader can compute (as ``CenterMesh`` does).
POINT_MEAN = "point_mean"
AREA_WEIGHTED_CENTER = "area_weighted_center"
SHARD_STATS = frozenset({POINT_MEAN, AREA_WEIGHTED_CENTER})

# Per-rank float64 partial sums, combined by :func:`reduce_shard_stats`.
POINT_SUM = "point_sum"
POINT_COUNT = "point_count"
AREA_MOMENT = "area_moment"
AREA_TOTAL = "area_total"
_PARTIALS = (
    (POINT_MEAN, POINT_SUM, POINT_COUNT),
    (AREA_WEIGHTED_CENTER, AREA_MOMENT, AREA_TOTAL),
)

Axis = Literal["cells", "points"]


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
        Per-rank partial sums for the statistics requested by the
        transforms (:data:`POINT_SUM`, ...); after :func:`reduce_shard_stats`
        the whole-mesh statistics (:data:`SHARD_STATS`), identical on every
        rank.
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


def new_shard_stats_group(device_mesh):
    r"""Create a process group with the same ranks as ``device_mesh``'s group.

    Collective over the whole world: every rank must call it. Ranks gather
    every domain group's rank list, then all create all groups in the same
    order (``new_group`` requires that); this rank keeps its own.
    """
    mine = tuple(torch.distributed.get_process_group_ranks(device_mesh.get_group()))
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, mine)
    mine_group = None
    for ranks in sorted(set(gathered)):
        group = torch.distributed.new_group(ranks=list(ranks))
        if ranks == mine:
            mine_group = group
    return mine_group


def reduce_shard_stats(shard: LocalMeshShard, group=None) -> LocalMeshShard:
    r"""Combine the per-rank partial sums of *shard* into whole-sample statistics.

    One float64 ``all_reduce`` (SUM); no collective if there are no stats.

    Parameters
    ----------
    shard : LocalMeshShard
        Payload whose ``stats`` hold partial keys (:data:`POINT_SUM`, ...).
    group : ProcessGroup, optional
        Group to reduce over; default the device mesh's group.

    Returns
    -------
    LocalMeshShard
        The same shard with ``stats`` holding :data:`POINT_MEAN` and/or
        :data:`AREA_WEIGHTED_CENTER` (float64).
    """
    keys = set(shard.stats.keys())
    if not keys:
        return shard
    present = [p for p in _PARTIALS if p[1] in keys]
    parts = [shard.stats[k].reshape(-1) for _, num, den in present for k in (num, den)]
    sizes = [p.numel() for p in parts]
    buffer = torch.cat([p.to(torch.float64) for p in parts])
    out_device = buffer.device
    device_mesh = shard.info.device_mesh
    if buffer.device.type != device_mesh.device_type:
        dev = torch.device(device_mesh.device_type)
        if dev.type == "cuda":
            dev = torch.device("cuda", torch.cuda.current_device())
        buffer = buffer.to(dev)
    torch.distributed.all_reduce(
        buffer, group=group if group is not None else device_mesh.get_group()
    )
    values = buffer.to(out_device).split(sizes)
    stats = {
        key: values[2 * i] / values[2 * i + 1] for i, (key, _, _) in enumerate(present)
    }
    return LocalMeshShard(
        mesh=shard.mesh, info=shard.info, stats=TensorDict(stats, batch_size=[])
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


def _wrap_parts(points, cells, point_data, cell_data, info, what):
    """Wrap the points / cells / point_data / cell_data of a (sub-)mesh."""
    n_cells = 0 if cells is None else cells.shape[0]
    points_axis, cells_sharded = _mesh_axes(points.shape[0], n_cells, info, what)
    point_offset = sum(info.point_sizes[: info.rank])
    if points_axis is not None:
        points = _shard(points, _sizes(points_axis, info), info)
        if point_data is not None:
            point_data = _wrap_tree(point_data, points_axis, info)
    if cells_sharded:
        # Offset into the concatenated point axis.
        cells = _shard(cells + point_offset, info.cell_sizes, info)
        if cell_data is not None:
            cell_data = _wrap_tree(cell_data, "cells", info)
    return points, cells, point_data, cell_data


def _wrap_mesh(mesh: Mesh, info: LocalShardInfo, what: str) -> Mesh:
    points, cells, point_data, cell_data = _wrap_parts(
        mesh.points, mesh.cells, mesh.point_data, mesh.cell_data, info, what
    )
    return Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        global_data=mesh.global_data,
    )


def _wrap_tensordict(td: TensorDict, info: LocalShardInfo) -> TensorDict:
    """A ``MeshToTensorDict``-style layout is wrapped like a mesh; other leaves by length."""
    out: dict[str, Any] = {}
    if "points" in td.keys() and not isinstance(td["points"], TensorDict):
        pd, cd = td.get("point_data", None), td.get("cell_data", None)
        pd = pd if isinstance(pd, TensorDict) else None
        cd = cd if isinstance(cd, TensorDict) else None
        points, cells, pd, cd = _wrap_parts(
            td["points"], td.get("cells", None), pd, cd, info, "tensordict"
        )
        wrapped = {"points": points, "cells": cells, "point_data": pd, "cell_data": cd}
        out = {k: v for k, v in wrapped.items() if v is not None}
    for key, value in td.items():
        if key in out:
            continue
        if key == "global_data":
            out[key] = value
        elif isinstance(value, TensorDict):
            out[key] = _wrap_tensordict(value, info)
        else:
            out[key] = _wrap_tree(TensorDict({key: value}), None, info)[key]
    return TensorDict(out, batch_size=[], device=td.device)


def wrap_local_sample(sample: Any, info: LocalShardInfo) -> Any:
    r"""Wrap a transformed local sample into ``Shard(0)`` ShardTensors.

    Communication-free. A mesh with cells must be this rank's shard; point
    clouds and free-standing leaves go on the axis matching their local
    length, else stay plain. ``global_data`` stays plain.

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
