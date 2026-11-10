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
"""Penetration depths, the envelope projection and the rigid-wall clamp.

These are the evaluation-side consumers of the pair search: per-node depth
fields for penetration metrics, a geometric post-processing step that moves
penetrating nodes back onto the shell thickness envelope, and the clamp of
positions to the admissible side of an analytic rigid wall. None of them
learns anything; all of them are pure functions of tensors.
"""

from __future__ import annotations

import math

import torch

from .search import ShellContactSearch, shell_contact_pairs


def node_maximum(
    num_nodes: int, index: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    r"""Per-node maximum of pair values, zero for nodes without a pair.

    Parameters
    ----------
    num_nodes : int
        Number of nodes ``N``.
    index : torch.Tensor
        ``[E]`` node index of every value.
    values : torch.Tensor
        ``[E]`` nonnegative values.

    Returns
    -------
    torch.Tensor
        ``[N]`` scattered maxima in the dtype of ``values``.
    """
    out = values.new_zeros(num_nodes)
    if index.numel():
        out = out.scatter_reduce(0, index, values, reduce="amax", include_self=True)
    return out


def shell_penetration_depths(
    pairs, num_nodes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Envelope and crossing depth of every node from its contact pairs.

    The envelope depth is the deepest violation of the thickness envelope,
    ``max over pairs of relu(-gap)``; the crossing depth is the distance a
    node has travelled through a shell's mid-surface,
    ``max over pairs of relu(-(gap + half_thickness))``, zero unless a pair
    is crossed.

    Parameters
    ----------
    pairs : ShellContactPairs
        Output of :func:`~physicsnemo.experimental.nn.shell_contact.shell_contact_pairs`.
    num_nodes : int
        Number of nodes ``N``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``[N]`` envelope depths and ``[N]`` crossing depths.
    """
    envelope = node_maximum(num_nodes, pairs.query, (-pairs.gap).clamp_min(0.0))
    crossing = node_maximum(
        num_nodes, pairs.query, (-(pairs.gap + pairs.half_thickness)).clamp_min(0.0)
    )
    return envelope, crossing


def wall_penetration_depths(
    positions: torch.Tensor,
    thickness: torch.Tensor,
    wall_position: float,
    *,
    axis: int = 0,
    sign: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Envelope and crossing depth of every node against a rigid wall plane.

    The wall is the plane ``x[axis] = wall_position``; ``sign=+1`` means the
    admissible half space is ``x[axis] <= wall_position``, ``sign=-1`` the
    opposite. The envelope depth is the violation of the half-thickness
    clearance, the crossing depth the distance beyond the plane itself.

    Parameters
    ----------
    positions : torch.Tensor
        ``[N, 3]`` node positions in the wall's units.
    thickness : torch.Tensor
        ``[N]`` nodal shell thickness.
    wall_position : float
        Plane coordinate along ``axis``.
    axis : int, optional
        Coordinate axis of the wall normal (0, 1 or 2).
    sign : float, optional
        ``+1`` or ``-1``, the side of the plane that is admissible.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``[N]`` envelope depths and ``[N]`` crossing depths.
    """
    _check_wall(axis, sign)
    clearance = float(sign) * (float(wall_position) - positions[:, axis])
    envelope = (0.5 * thickness - clearance).clamp_min(0.0)
    crossing = (-clearance).clamp_min(0.0)
    return envelope, crossing


def clamp_to_wall(
    positions: torch.Tensor,
    wall_position: float | torch.Tensor,
    *,
    axis: int = 0,
    sign: float = 1.0,
) -> torch.Tensor:
    r"""Move every point onto the admissible side of a rigid wall plane.

    Nothing but the wall coordinate changes, and only for points beyond the
    plane: ``min(x[axis], wall)`` for ``sign=+1``, ``max`` for ``sign=-1``.
    The plane position is taken in the coordinates of ``positions``
    (physical or normalized alike).

    Parameters
    ----------
    positions : torch.Tensor
        ``[N, 3]`` points.
    wall_position : float or torch.Tensor
        Plane coordinate along ``axis`` (a scalar tensor is accepted).
    axis : int, optional
        Coordinate axis of the wall normal (0, 1 or 2).
    sign : float, optional
        ``+1`` when the admissible half space is ``x[axis] <= wall_position``.

    Returns
    -------
    torch.Tensor
        ``[N, 3]`` clamped points (a new tensor).
    """
    _check_wall(axis, sign)
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [N, 3]")
    limit = torch.as_tensor(
        wall_position, dtype=positions.dtype, device=positions.device
    )
    column = positions[:, axis]
    column = torch.minimum(column, limit) if sign > 0 else torch.maximum(column, limit)
    return torch.cat(
        (positions[:, :axis], column.unsqueeze(-1), positions[:, axis + 1 :]), dim=-1
    )


@torch.no_grad()
def project_shell_penetrations(
    positions: torch.Tensor,
    faces: torch.Tensor,
    thickness: torch.Tensor,
    reference_positions: torch.Tensor,
    *,
    search: ShellContactSearch,
    iterations: int,
    tolerance: float = 0.0,
    damping: float = 1.0,
    crossed_only: bool = False,
    excluded_pairs: torch.Tensor | None = None,
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Move penetrating nodes onto the shell thickness envelope.

    Each iteration discovers the contact pairs of the current positions with
    ``search`` (which must use the reference side test, so a node that has
    passed through a shell is moved back through it rather than pushed out
    the far side), selects every node whose deepest pair violates the
    envelope by more than ``tolerance``, and moves it along its offset from
    that facet's closest point to the envelope on the admissible side. A
    ``damping`` below one applies that fraction of the move per iteration;
    with 0.5 both sides of a mutual crossing meet at the envelope instead of
    overshooting. The search is rebuilt every iteration because moved nodes
    are facet vertices too; the loop stops early once nothing penetrates.
    Geometry only, no gradient.

    Parameters
    ----------
    positions : torch.Tensor
        ``[N, 3]`` physical node positions to correct.
    faces : torch.Tensor
        ``[F, 3|4]`` facet connectivity.
    thickness : torch.Tensor
        ``[N]`` nodal shell thickness.
    reference_positions : torch.Tensor
        ``[N, 3]`` reference configuration of the side test.
    search : ShellContactSearch
        Discovery settings with ``side_from_reference=True``.
    iterations : int
        Number of search-and-move passes (0 returns the input unchanged).
    tolerance : float, optional
        Envelope violation that is left in place (physical length).
    damping : float, optional
        Fraction of the computed move applied per iteration, in ``(0, 1]``.
    crossed_only : bool, optional
        Move only nodes whose deepest pair has crossed the shell, leaving
        envelope overlaps without a crossing to the model.
    excluded_pairs : torch.Tensor, optional
        ``[2, X]`` static node/facet exclusions of the search.
    batch : torch.Tensor, optional
        ``[N]`` graph index for batched meshes.

    Returns
    -------
    torch.Tensor
        ``[N, 3]`` corrected positions (a new tensor).
    """
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 0:
        raise ValueError("iterations must be an integer >= 0")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and >= 0")
    if not 0.0 < float(damping) <= 1.0:
        raise ValueError("damping must be in (0, 1]")
    if not search.side_from_reference:
        raise ValueError(
            "the projection needs the reference side test: side_from_reference=True"
        )
    positions = positions.clone()
    n = positions.shape[0]
    for _ in range(int(iterations)):
        pairs = shell_contact_pairs(
            positions,
            faces,
            thickness,
            reference_positions=reference_positions,
            excluded_pairs=excluded_pairs,
            batch=batch,
            search=search,
        )
        gap = pairs.gap
        query = pairs.query
        # gap + half is the signed distance to the facet, negative once the
        # node has crossed it.
        pair_half = pairs.half_thickness
        depth = (-gap - tolerance).clamp_min(0.0)
        if crossed_only:
            depth = torch.where(gap + pair_half < 0, depth, torch.zeros_like(depth))
        if not bool((depth > 0).any()):
            break
        node_depth = node_maximum(n, query, depth)
        deepest = (depth > 0) & (depth >= node_depth[query])
        sentinel = query.numel()
        chosen = torch.full(
            (n,), sentinel, device=query.device, dtype=torch.long
        ).scatter_reduce(
            0,
            query[deepest],
            torch.arange(sentinel, device=query.device)[deepest],
            reduce="amin",
            include_self=True,
        )
        nodes = (chosen < sentinel).nonzero().flatten()
        selected = chosen[nodes]
        relative = pairs.closest[selected] - positions[nodes]
        closest = positions[nodes] + relative
        distance = torch.linalg.vector_norm(relative, dim=-1)
        half = pair_half[selected]
        # gap + half = (+/-) distance: the sign says which side the node is on.
        side = torch.sign(gap[selected] + half)
        side = torch.where(side == 0, torch.ones_like(side), side)
        movable = distance > 1e-6
        direction = -relative / distance.clamp_min(1e-6).unsqueeze(-1)
        moved = closest + (side * half).unsqueeze(-1) * direction
        moved = positions[nodes] + float(damping) * (moved - positions[nodes])
        positions[nodes[movable]] = moved[movable]
    return positions


def _check_wall(axis: int, sign: float) -> None:
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    if float(sign) not in (1.0, -1.0):
        raise ValueError("sign must be +1 or -1")
