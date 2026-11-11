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
"""Penetration depths and the rigid-wall clamp.

These are the evaluation-side consumers of the pair search: per-node depth
fields for penetration metrics and the clamp of positions to the admissible
side of an analytic rigid wall. None of them learns anything; all of them are
pure functions of tensors.
"""

from __future__ import annotations

import torch


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
        ``[N]`` nodal shell thickness, or a scalar shared by every node.
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
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [N, 3]")
    thickness = torch.as_tensor(
        thickness, dtype=positions.dtype, device=positions.device
    )
    if thickness.ndim != 0 and thickness.shape != positions.shape[:1]:
        raise ValueError(
            "thickness must have shape [N] or be a scalar; got "
            f"{tuple(thickness.shape)} for {positions.shape[0]} nodes"
        )
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
    (physical or normalized alike). The clamp targets the crossing family
    of :func:`wall_penetration_depths`: it puts the mid-surface on the plane,
    so a clamped node still reports a thickness-envelope depth of half its
    thickness, as the envelope is measured there.

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


def _check_wall(axis: int, sign: float) -> None:
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    if float(sign) not in (1.0, -1.0):
        raise ValueError("sign must be +1 or -1")
