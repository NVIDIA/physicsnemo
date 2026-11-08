# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-Torch deformation-energy kernels on normalized tensor inputs."""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float, Int

from physicsnemo.utils._small_linalg import small_det


def _safe_topology_indices(
    topology: Int[torch.Tensor, "num_primitives vertices_per_primitive"],
    num_points: int,
) -> tuple[
    Int[torch.Tensor, "num_primitives vertices_per_primitive"],
    Bool[torch.Tensor, " num_primitives"],
]:
    """Return gather-safe indices and a per-primitive bounds mask.

    Bounds remain a tensor-side concern so CUDA callers do not synchronize to
    validate connectivity. Invalid indices are replaced only for the gather;
    callers use the returned mask to expose the affected primitive as NaN.
    """

    in_bounds = (topology >= 0) & (topology < num_points)
    return torch.where(in_bounds, topology, torch.zeros_like(topology)), in_bounds.all(
        dim=1
    )


def _gather_vertices(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    safe_topology: Int[torch.Tensor, "num_primitives vertices_per_primitive"],
) -> Float[torch.Tensor, "batch num_primitives vertices_per_primitive num_dims"]:
    """Gather vertices, including the zero-point invalid-topology edge case."""

    if points.shape[1] == 0:
        vertices = points.new_full(
            (
                points.shape[0],
                safe_topology.shape[0],
                safe_topology.shape[1],
                points.shape[2],
            ),
            torch.nan,
        )
        return vertices + 0.0 * points.sum()
    return points[:, safe_topology, :]


def _simplex_edges(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    safe_simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
) -> Float[torch.Tensor, "batch num_simplices simplex_dimension num_dims"]:
    """Gather an equivalent edge basis suited to slender simplices.

    Edges connect consecutive cell vertices. This unit-determinant basis
    change preserves orientation, measure, and strain while avoiding
    cancellation when neighboring vertices are close together.
    """

    vertices = _gather_vertices(points, safe_simplices)
    return vertices[:, :, 1:, :] - vertices[:, :, :-1, :]


def _canonical_simplex_indices(
    safe_simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
) -> Int[torch.Tensor, "num_simplices vertices_per_simplex"]:
    """Choose one connectivity-order-independent simplex vertex order.

    Current and reference geometry share this permutation, so relative signed
    measures are unchanged while low-precision arithmetic becomes independent
    of the caller's cell ordering.
    """

    return torch.sort(safe_simplices, dim=1).values


def _simplex_factorial(simplex_dimension: int) -> float:
    """Return ``m!`` for supported dimensions without a Python factorial."""

    # This polynomial equals m! for the supported dimensions m in {1, 2, 3}
    # and remains traceable when torch.compile represents m as a SymInt.
    return (3 * simplex_dimension * simplex_dimension - 7 * simplex_dimension + 6) / 2


def _unsigned_simplex_measure(
    edges: Float[torch.Tensor, "batch num_simplices simplex_dimension num_dims"],
    simplex_dimension: int,
) -> Float[torch.Tensor, "batch num_simplices"]:
    """Return intrinsic simplex measure using modified Gram--Schmidt.

    Computing orthogonal residuals avoids the squared condition number and
    subtractive determinant cancellation of a Gram matrix. At exact current
    collapse, vector norms provide the conventional zero subgradient.
    """

    edge_0 = edges[..., 0, :]
    length_0 = torch.linalg.vector_norm(edge_0, dim=-1)
    if simplex_dimension == 1:
        return length_0

    safe_length_0 = torch.where(length_0 > 0.0, length_0, torch.ones_like(length_0))
    basis_0 = edge_0 / safe_length_0.unsqueeze(-1)
    edge_1 = edges[..., 1, :]
    orthogonal_1 = edge_1 - ((edge_1 * basis_0).sum(dim=-1, keepdim=True) * basis_0)
    length_1 = torch.linalg.vector_norm(orthogonal_1, dim=-1)
    if simplex_dimension == 2:
        return 0.5 * length_0 * length_1

    safe_length_1 = torch.where(length_1 > 0.0, length_1, torch.ones_like(length_1))
    basis_1 = orthogonal_1 / safe_length_1.unsqueeze(-1)
    edge_2 = edges[..., 2, :]
    orthogonal_2 = edge_2 - (
        (edge_2 * basis_0).sum(dim=-1, keepdim=True) * basis_0
        + (edge_2 * basis_1).sum(dim=-1, keepdim=True) * basis_1
    )
    length_2 = torch.linalg.vector_norm(orthogonal_2, dim=-1)
    return length_0 * length_1 * length_2 / 6.0


def _reference_measure(
    reference_edges: Float[
        torch.Tensor, "batch num_simplices simplex_dimension num_dims"
    ],
    simplex_dimension: int,
) -> tuple[
    Float[torch.Tensor, "batch num_simplices"],
    Bool[torch.Tensor, "batch num_simplices"],
]:
    """Return reference measure and its strict nondegeneracy mask."""

    measure = _unsigned_simplex_measure(reference_edges, simplex_dimension)
    valid = (
        (measure > 0.0)
        & torch.isfinite(measure)
        & torch.isfinite(reference_edges).all(dim=(-1, -2))
    )
    return torch.where(valid, measure, torch.full_like(measure, torch.nan)), valid


def _reference_frame_cauchy_green(
    edges: Float[torch.Tensor, "batch num_simplices simplex_dimension num_dims"],
    reference_edges: Float[
        torch.Tensor, "batch num_simplices simplex_dimension num_dims"
    ],
    simplex_dimension: int,
) -> tuple[
    Float[torch.Tensor, "batch num_simplices simplex_dimension simplex_dimension"],
    Float[torch.Tensor, "batch num_simplices simplex_dimension"],
    Float[torch.Tensor, "batch num_simplices"],
]:
    """Return the right Cauchy--Green tensor in a reference-orthonormal frame.

    Modified Gram--Schmidt factors the reference edge rows as ``E0 = L Q``
    without forming ``E0 E0^T``; forward substitution ``L H = E`` then gives
    ``C = H H^T``. Also returns the diagonal of ``L`` for the scale-normalized
    reference edges, and the reference measure. Degenerate reference cells use
    unit Gram--Schmidt denominators so their masked terms and gradients stay
    finite.
    """

    # C is invariant to a common scale of both edge sets. Normalizing by the
    # largest reference edge coordinate keeps the squared norms of tiny or huge
    # cells in range; the scale is detached because every normalized
    # expression equals the unnormalized one for any fixed positive scale.
    scale = reference_edges.abs().amax(dim=(-1, -2)).detach()
    scale = torch.where(
        (scale > 0.0) & torch.isfinite(scale), scale, torch.ones_like(scale)
    )
    edges = edges / scale[..., None, None]
    reference_edges = reference_edges / scale[..., None, None]

    basis: list[torch.Tensor] = []
    frame_edges: list[torch.Tensor] = []
    lengths: list[torch.Tensor] = []
    for row in range(simplex_dimension):
        residual = reference_edges[..., row, :]
        frame_edge = edges[..., row, :]
        for basis_vector, previous_frame_edge in zip(basis, frame_edges):
            coefficient = (residual * basis_vector).sum(dim=-1, keepdim=True)
            residual = residual - coefficient * basis_vector
            frame_edge = frame_edge - coefficient * previous_frame_edge
        length = torch.linalg.vector_norm(residual, dim=-1)
        safe_length = torch.where(length > 0.0, length, torch.ones_like(length))
        basis.append(residual / safe_length.unsqueeze(-1))
        frame_edges.append(frame_edge / safe_length.unsqueeze(-1))
        lengths.append(length)

    # H: (batch, num_simplices, simplex_dimension, num_dims)
    frame_edge_rows = torch.stack(frame_edges, dim=-2)
    # Broadcast-and-sum: a batched matmul of millions of tiny matrices is slow.
    right_cauchy_green = (
        frame_edge_rows.unsqueeze(-2) * frame_edge_rows.unsqueeze(-3)
    ).sum(dim=-1)
    diagonal = torch.stack(lengths, dim=-1)
    reference_measure = (
        diagonal.prod(dim=-1)
        / _simplex_factorial(simplex_dimension)
        * scale**simplex_dimension
    )
    return right_cauchy_green, diagonal, reference_measure


def simplex_stvk_terms_torch(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    lame_lambda: float | Float[torch.Tensor, ""],
    shear_modulus: float | Float[torch.Tensor, ""],
) -> Float[torch.Tensor, "batch num_simplices"]:
    r"""Return reference-integrated St. Venant--Kirchhoff simplex terms.

    Inputs are normalized to ``(B, N, D)`` and the output has shape ``(B, M)``.
    The intrinsic strain is computed in a reference-orthonormal simplex frame,
    so the expression applies to edges, triangles, and tetrahedra embedded in
    any coordinate dimension ``D >= m``.

    Degenerate reference simplices return NaN rather than receiving an implicit
    regularization. This makes invalid rest geometry visible to an optimizer.
    """

    safe_simplices, topology_valid = _safe_topology_indices(simplices, points.shape[1])
    safe_simplices = _canonical_simplex_indices(safe_simplices)
    edges = _simplex_edges(points, safe_simplices)
    reference_edges = _simplex_edges(reference_points, safe_simplices)
    simplex_dimension = simplices.shape[1] - 1
    right_cauchy_green, diagonal, reference_measure = _reference_frame_cauchy_green(
        edges, reference_edges, simplex_dimension
    )
    identity = torch.eye(
        simplex_dimension,
        dtype=points.dtype,
        device=points.device,
    )
    strain = 0.5 * (right_cauchy_green - identity)
    trace = torch.diagonal(strain, dim1=-2, dim2=-1).sum(dim=-1)
    # This equivalent deviatoric/volumetric split remains manifestly
    # nonnegative for stable negative Lamé parameters at large strain.
    mean_trace = trace / simplex_dimension
    deviatoric_strain = strain - mean_trace[..., None, None] * identity
    deviatoric_square = deviatoric_strain.square().sum(dim=(-1, -2))
    volumetric_coefficient = 0.5 * lame_lambda + shear_modulus / simplex_dimension
    terms = reference_measure * (
        shear_modulus * deviatoric_square + volumetric_coefficient * trace.square()
    )

    valid = (
        topology_valid.unsqueeze(0)
        & torch.isfinite(edges).all(dim=(-1, -2))
        & torch.isfinite(reference_edges).all(dim=(-1, -2))
        & torch.isfinite(diagonal).all(dim=-1)
        & (diagonal != 0.0).all(dim=-1)
        & torch.isfinite(reference_measure)
        & (reference_measure > 0.0)
    )
    return torch.where(valid, terms, torch.full_like(terms, torch.nan))


def simplex_measure_components_torch(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
) -> tuple[
    Float[torch.Tensor, "batch num_simplices"],
    Float[torch.Tensor, "batch num_simplices"],
]:
    r"""Return relative measure and reference measure for each simplex.

    The ratio is signed for full-dimensional simplices,
    ``det(E) / det(E0)``, so reflected elements remain distinguishable. For an
    embedded simplex it is the unsigned intrinsic ratio. Both outputs have
    shape ``(B, M)``.
    """

    safe_simplices, topology_valid = _safe_topology_indices(simplices, points.shape[1])
    safe_simplices = _canonical_simplex_indices(safe_simplices)
    edges = _simplex_edges(points, safe_simplices)
    reference_edges = _simplex_edges(reference_points, safe_simplices)
    simplex_dimension = simplices.shape[1] - 1
    coordinate_dimension = points.shape[-1]
    current_finite = torch.isfinite(edges).all(dim=(-1, -2))

    if simplex_dimension == coordinate_dimension:
        determinant = small_det(edges)
        reference_determinant = small_det(reference_edges)
        reference_measure = reference_determinant.abs() / _simplex_factorial(
            simplex_dimension
        )
        reference_valid = (
            (reference_determinant != 0.0)
            & torch.isfinite(reference_determinant)
            & torch.isfinite(reference_edges).all(dim=(-1, -2))
        )
        reference_measure = torch.where(
            reference_valid,
            reference_measure,
            torch.full_like(reference_measure, torch.nan),
        )
        ratio = determinant / reference_determinant
        valid = reference_valid
    else:
        reference_measure, reference_valid = _reference_measure(
            reference_edges, simplex_dimension
        )
        current_measure = _unsigned_simplex_measure(edges, simplex_dimension)
        ratio = current_measure / reference_measure
        valid = reference_valid

    topology_valid = topology_valid.unsqueeze(0)
    valid = valid & topology_valid & current_finite & torch.isfinite(ratio)
    nan = torch.full_like(ratio, torch.nan)
    reference_measure = torch.where(topology_valid, reference_measure, nan)
    return torch.where(valid, ratio, nan), reference_measure


def simplex_inversion_terms_torch(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    minimum_jacobian: float | Float[torch.Tensor, ""],
) -> Float[torch.Tensor, "batch num_simplices"]:
    r"""Return signed-Jacobian inversion penalties for full-dimensional cells.

    Each term is ``0.5 * V0 * relu(minimum_jacobian - J)^2``. Embedded
    simplices have no intrinsic orientation sign and are rejected.
    """

    simplex_dimension = simplices.shape[1] - 1
    if simplex_dimension != points.shape[-1]:
        raise ValueError(
            "simplex inversion energy requires full-dimensional simplices, got "
            f"simplex dimension {simplex_dimension} in D={points.shape[-1]}"
        )
    ratio, reference_measure = simplex_measure_components_torch(
        points, reference_points, simplices
    )
    violation = torch.relu(minimum_jacobian - ratio)
    return 0.5 * reference_measure * violation.square()


def _signed_dihedral(
    vertices: Float[torch.Tensor, "batch num_hinges 4 3"],
) -> tuple[
    Float[torch.Tensor, "batch num_hinges"],
    Float[torch.Tensor, "batch num_hinges"],
    Float[torch.Tensor, "batch num_hinges"],
    Float[torch.Tensor, "batch num_hinges"],
]:
    """Return signed hinge angle, edge length, and adjacent doubled areas."""

    point_i, point_j, point_k, point_l = vertices.unbind(dim=2)
    edge = point_j - point_i
    left_offset = point_k - point_i
    right_offset = point_l - point_j

    # The angle and l^2 / A weight are invariant to a common coordinate scale.
    # Normalize before products and norms so finite hinges remain representable
    # when their coordinates are very small or large. Detaching the scale keeps
    # the analytical gradients of the original scale-invariant expression.
    coordinate_scale = (
        torch.stack((edge, left_offset, right_offset), dim=-2)
        .abs()
        .amax(dim=(-1, -2))
        .detach()
    )
    safe_scale = torch.where(
        (coordinate_scale > 0.0) & torch.isfinite(coordinate_scale),
        coordinate_scale,
        torch.ones_like(coordinate_scale),
    )
    edge = edge / safe_scale.unsqueeze(-1)
    left_offset = left_offset / safe_scale.unsqueeze(-1)
    right_offset = right_offset / safe_scale.unsqueeze(-1)
    left_normal = torch.linalg.cross(edge, left_offset, dim=-1)
    right_normal = torch.linalg.cross(-edge, right_offset, dim=-1)

    edge_length = torch.linalg.vector_norm(edge, dim=-1)
    left_double_area = torch.linalg.vector_norm(left_normal, dim=-1)
    right_double_area = torch.linalg.vector_norm(right_normal, dim=-1)
    unit_edge = edge / edge_length.unsqueeze(-1)
    unit_left = left_normal / left_double_area.unsqueeze(-1)
    unit_right = right_normal / right_double_area.unsqueeze(-1)
    sine = (unit_edge * torch.linalg.cross(unit_left, unit_right, dim=-1)).sum(dim=-1)
    cosine = (unit_left * unit_right).sum(dim=-1)
    return (
        torch.atan2(sine, cosine),
        edge_length,
        left_double_area,
        right_double_area,
    )


def hinge_bending_terms_torch(
    points: Float[torch.Tensor, "batch num_points 3"],
    reference_points: Float[torch.Tensor, "batch num_points 3"],
    hinges: Int[torch.Tensor, "num_hinges 4"],
) -> Float[torch.Tensor, "batch num_hinges"]:
    r"""Return reference-relative discrete hinge bending terms.

    A hinge row ``(i, j, k, l)`` denotes oriented faces ``(i, j, k)`` and
    ``(j, i, l)``. The returned ``(B, H)`` terms are

    ``0.5 * l0^2 / (A0_left + A0_right) * wrap(theta-theta0)^2``.

    Reference hinges with a zero-length edge or degenerate adjacent triangle
    return NaN. Current degeneracy likewise remains visible rather than being
    hidden by an epsilon-clamped normalization.
    """

    safe_hinges, topology_valid = _safe_topology_indices(hinges, points.shape[1])
    vertices = _gather_vertices(points, safe_hinges)
    reference_vertices = _gather_vertices(reference_points, safe_hinges)
    angle, _, current_left_double_area, current_right_double_area = _signed_dihedral(
        vertices
    )
    (
        reference_angle,
        reference_edge_length,
        reference_left_double_area,
        reference_right_double_area,
    ) = _signed_dihedral(reference_vertices)

    angle_difference = angle - reference_angle
    wrapped_difference = torch.atan2(
        torch.sin(angle_difference),
        torch.cos(angle_difference),
    )
    reference_area_sum = 0.5 * (
        reference_left_double_area + reference_right_double_area
    )
    weight = reference_edge_length.square() / reference_area_sum
    terms = 0.5 * weight * wrapped_difference.square()

    valid = (
        topology_valid.unsqueeze(0)
        & (reference_edge_length > 0.0)
        & (reference_left_double_area > 0.0)
        & (reference_right_double_area > 0.0)
        & torch.isfinite(reference_edge_length)
        & torch.isfinite(reference_left_double_area)
        & torch.isfinite(reference_right_double_area)
        & (current_left_double_area > 0.0)
        & (current_right_double_area > 0.0)
    )
    return torch.where(valid, terms, torch.full_like(terms, torch.nan))


def closed_surface_volume_contributions_torch(
    points: Float[torch.Tensor, "batch num_points 3"],
    reference_points: Float[torch.Tensor, "batch num_points 3"],
    triangles: Int[torch.Tensor, "num_triangles 3"],
) -> tuple[
    Float[torch.Tensor, "batch num_triangles"],
    Float[torch.Tensor, "batch num_triangles"],
]:
    r"""Return signed per-triangle enclosed-volume contributions.

    Triangles must form a closed, consistently oriented surface. The tensor
    primitive assumes that discrete topology contract has already been checked.
    Coordinates are shifted by one common detached origin per batch for
    numerical stability. Each contribution is
    ``dot(y0, cross(y1, y2)) / 6`` and has shape ``(B,F)``; the closed-surface
    sum is independent of that origin.
    """

    # A detached, topologically used origin leaves the closed-surface sum and
    # its gradient unchanged while avoiding cancellation between large world
    # coordinates and small extents.
    safe_triangles, topology_valid = _safe_topology_indices(triangles, points.shape[1])
    origin_index = safe_triangles[:1, :1].reshape(-1)
    if points.shape[1] == 0:
        nan = points.new_full((points.shape[0], triangles.shape[0]), torch.nan)
        return (
            nan + 0.0 * points.sum(),
            nan.clone() + 0.0 * reference_points.sum(),
        )
    origin = points.index_select(1, origin_index).detach().unsqueeze(2)
    reference_origin = (
        reference_points.index_select(1, origin_index).detach().unsqueeze(2)
    )
    vertices = _gather_vertices(points, safe_triangles) - origin
    reference_vertices = (
        _gather_vertices(reference_points, safe_triangles) - reference_origin
    )
    current = small_det(vertices) / 6.0
    reference = small_det(reference_vertices) / 6.0
    topology_valid = topology_valid.unsqueeze(0)
    nan = torch.full_like(current, torch.nan)
    return (
        torch.where(topology_valid, current, nan),
        torch.where(topology_valid, reference, nan),
    )
