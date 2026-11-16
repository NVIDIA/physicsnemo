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
"""Facet projections and the reference side test of shell contact pairs.

The functions here are the geometric primitives behind the pair search: the
live closest point of a node on a facet, the detached projection of the same
pair in the reference configuration, and the discrete test that decides
whether a node has passed through a shell since that configuration. None of
them discovers pairs; :func:`~physicsnemo.experimental.nn.shell_contact.shell_contact_pairs`
does.
"""

from __future__ import annotations

import torch

from ._closest_point import (
    TriangleProjection,
    closest_point_facet,
)


def project_onto_facets(
    positions: torch.Tensor,
    query: torch.Tensor,
    source_nodes: torch.Tensor,
    *,
    material_fan: bool,
) -> TriangleProjection:
    r"""Live closest point of every query node on its paired facet.

    Parameters
    ----------
    positions : torch.Tensor
        ``[N, 3]`` node positions.
    query : torch.Tensor
        ``[E]`` node index of every pair.
    source_nodes : torch.Tensor
        ``[E, 3|4]`` vertex indices of the paired facet; a triangle in a
        four-wide array repeats its third vertex in slot four.
    material_fan : bool
        Transport the reference centre fan of four-node facets with the
        vertices (the material surface contract) instead of validating the
        current quad shape.

    Returns
    -------
    TriangleProjection
        Closest point, barycentric weights over the facet vertices, distance
        and facet normal, all live in ``positions``.
    """
    triangle_mask = (
        (source_nodes[:, 2] == source_nodes[:, 3])
        if source_nodes.shape[1] == 4
        else None
    )
    points = positions[query]
    facets = positions[source_nodes]
    return closest_point_facet(
        points,
        facets,
        triangle_mask=triangle_mask,
        validate_quads=False,
        material_fan=material_fan,
    )


def facet_side(
    points: torch.Tensor,
    closest: torch.Tensor,
    normal: torch.Tensor,
    distance: torch.Tensor,
    interior_only: bool = False,
) -> torch.Tensor:
    r"""Side of a facet's oriented plane every point lies on: -1, 0 or +1.

    The sign is zero (inconclusive) within a thousandth of the distance of the
    facet plane and on a degenerate facet. With ``interior_only`` it is also
    zero unless the closest point is an interior projection (the offset from
    the closest point is parallel to the normal within 2 percent), that is
    unless the point is over the facet rather than beside one of its edges.

    Parameters
    ----------
    points, closest, normal, distance : torch.Tensor
        ``[E, 3]`` points, their closest points and facet normals, and the
        ``[E]`` distances of a projection.
    interior_only : bool, optional
        Require an interior projection for a conclusive side.

    Returns
    -------
    torch.Tensor
        ``[E]`` signs in the dtype of ``points``.
    """
    signed = ((points - closest) * normal).sum(-1)
    inconclusive = signed.abs() <= 1e-3 * distance
    if interior_only:
        inconclusive = inconclusive | (signed.abs() < 0.98 * distance)
    return torch.where(inconclusive, torch.zeros_like(signed), torch.sign(signed))


@torch.no_grad()
def reference_facet_projection(
    reference_positions: torch.Tensor,
    thickness: torch.Tensor,
    query: torch.Tensor,
    source_nodes: torch.Tensor,
    *,
    material_fan: bool,
) -> tuple[torch.Tensor, TriangleProjection]:
    r"""Detached closest point and shell clearance of pairs in the reference configuration.

    Parameters
    ----------
    reference_positions : torch.Tensor
        ``[N, 3]`` reference node positions.
    thickness : torch.Tensor
        ``[N]`` nodal shell thickness.
    query, source_nodes : torch.Tensor
        The pairs, as in :func:`project_onto_facets`.
    material_fan : bool
        Surface contract of four-node facets.

    Returns
    -------
    tuple[torch.Tensor, TriangleProjection]
        ``[E]`` reference activation gap (distance minus the half-thickness
        sum) and the reference projection, both without gradient.
    """
    projection = project_onto_facets(
        reference_positions, query, source_nodes, material_fan=material_fan
    )
    surface_thickness = (thickness[source_nodes] * projection.barycentric).sum(-1)
    gap = projection.distance - (thickness[query] + surface_thickness) * 0.5
    return gap, projection


@torch.no_grad()
def crossing_flags(
    points: torch.Tensor,
    projection: TriangleProjection,
    reference_points: torch.Tensor,
    reference_projection: TriangleProjection,
    *,
    symmetric: bool,
) -> torch.Tensor:
    r"""Discrete flag of the pairs whose node has passed through the facet.

    A node has crossed when it lies on the other side of the facet's oriented
    plane than in the reference configuration and projects onto the facet's
    interior now (a node beside a facet can change plane side by sliding past
    an edge). With ``symmetric`` the reference configuration must show an
    interior projection as well: a node that was beside the facet and lies
    under its interior now has slid under the edge, not through the shell.

    Parameters
    ----------
    points : torch.Tensor
        ``[E, 3]`` current positions of the query nodes.
    projection : TriangleProjection
        Their current projection onto the paired facets.
    reference_points : torch.Tensor
        ``[E, 3]`` reference positions of the query nodes.
    reference_projection : TriangleProjection
        The reference projection of the same pairs.
    symmetric : bool
        Ask the reference configuration for an interior projection too.

    Returns
    -------
    torch.Tensor
        ``[E]`` boolean crossing flags, without gradient.
    """
    before = facet_side(
        reference_points,
        reference_projection.closest,
        reference_projection.normal,
        reference_projection.distance,
        interior_only=symmetric,
    )
    now = facet_side(
        points.detach(),
        projection.closest.detach(),
        projection.normal.detach(),
        projection.distance.detach(),
        interior_only=True,
    )
    return before * now < 0
