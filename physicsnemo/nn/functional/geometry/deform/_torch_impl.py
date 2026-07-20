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

"""Pure-Torch point displacement, Shepard morphing, and lattice FFD kernels."""

from __future__ import annotations

from math import isqrt

import torch
from jaxtyping import Bool, Float
from torch.utils.checkpoint import checkpoint

from ._utils import _ffd_window_size, _zero_dependency

# Bound all live pairwise temporaries, rather than only the nominal
# ``(B, query_chunk, control_chunk, D)`` difference tensor. Profiling the
# numerically robust distance calculation shows a peak of roughly 24 value-sized
# pairwise tensors in eager mode. The larger factor leaves headroom for allocator
# granularity and autograd bookkeeping while making float64 blocks half as large
# as float32 blocks.
_PAIRWISE_TEMPORARY_BYTE_BUDGET = 256 * 1024 * 1024
_PAIRWISE_LIVE_VALUE_FACTOR = 32


def displace_points_torch(
    points: torch.Tensor,
    displacement: torch.Tensor,
    point_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Apply dense displacement to normalized rank-3 point tensors."""

    if point_weights is None:
        return points + displacement
    if point_weights.dtype == torch.bool:
        displacement = torch.where(point_weights.unsqueeze(-1), displacement, 0.0)
        return points + displacement
    if points.is_cuda:
        # Avoid materializing the weighted displacement in eager CUDA execution.
        return torch.addcmul(points, point_weights.unsqueeze(-1), displacement)
    return points + point_weights.unsqueeze(-1) * displacement


def _chunk_sizes(
    batch_size: int,
    num_points: int,
    num_controls: int,
    num_dims: int,
    element_size: int,
) -> tuple[int, int]:
    """Choose query/control block sizes within the live temporary byte budget."""

    if num_points == 0 or num_controls == 0:
        return max(num_points, 1), max(num_controls, 1)

    pair_budget = max(
        1,
        _PAIRWISE_TEMPORARY_BYTE_BUDGET
        // _PAIRWISE_LIVE_VALUE_FACTOR
        // max(element_size, 1)
        // max(batch_size, 1)
        // max(num_dims, 1),
    )
    query_chunk = min(num_points, max(1, isqrt(pair_budget)))
    control_chunk = min(num_controls, max(1, pair_budget // query_chunk))

    # If all controls fit, use the remaining budget for more query points.
    if control_chunk == num_controls:
        query_chunk = min(num_points, max(1, pair_budget // num_controls))
    return query_chunk, control_chunk


def _requires_chunk_checkpoint(
    batch_size: int,
    num_points: int,
    num_controls: int,
    num_dims: int,
    element_size: int,
) -> bool:
    """Whether retaining the complete eager graph would exceed the byte budget."""

    estimated_bytes = (
        batch_size
        * num_points
        * num_controls
        * num_dims
        * element_size
        * _PAIRWISE_LIVE_VALUE_FACTOR
    )
    return estimated_bytes > _PAIRWISE_TEMPORARY_BYTE_BUDGET


def _normalized_distance(
    points: torch.Tensor,
    controls: torch.Tensor,
    radius: torch.Tensor,
    *,
    compute_q_squared: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return coincidence flags and overflow-safe normalized distances.

    Forming ``sum((x-c)**2) / radius**2`` can overflow even when the final
    normalized distance is inside the support. Divide componentwise first,
    replace overflowed differences with finite boundary sentinels, and clamp
    components before the norm because only ``q < 1`` is relevant.
    """

    query = points.unsqueeze(2)
    control = controls.unsqueeze(1)
    radii = radius.unsqueeze(1).unsqueeze(-1)
    coordinate_exact = (query == control).all(dim=-1)
    difference = query - control

    # Scale numerator and denominator by the same detached factor before the
    # division. Besides avoiding forward overflow, this matters in backward:
    # the naive derivative of ``difference / radius`` forms
    # ``difference / radius**2``, which can overflow for subnormal radii even
    # when the final field derivative is finite. Keeping the scaled radius in
    # the normal range lets ordinary Torch autograd preserve that cancellation.
    finfo = torch.finfo(points.dtype)
    infinite_radius = torch.isinf(radii)
    small_radius = (radii < finfo.tiny) & ~infinite_radius
    normal_radius = torch.where(
        small_radius | infinite_radius, torch.ones_like(radii), radii
    )
    ratio_scale = torch.where(
        small_radius,
        torch.full_like(radii, finfo.max),
        normal_radius.reciprocal(),
    ).detach()
    # For finite coordinates, a nonfinite subtraction can only result from
    # overflow. Replacing it with a finite boundary sentinel keeps every
    # finite-radius pair outside its support and gives that branch a zero
    # geometry derivative. With an infinite radius the sentinel divides to
    # zero, which is the correct limiting influence.
    safe_difference = torch.nan_to_num(
        difference,
        nan=finfo.max,
        posinf=finfo.max,
        neginf=-finfo.max,
    )
    normalized = (safe_difference * ratio_scale) / (radii * ratio_scale)
    normalized = torch.nan_to_num(normalized, nan=1.0, posinf=1.0, neginf=-1.0).clamp(
        -1.0, 1.0
    )
    numerically_zero = (normalized == 0).all(dim=-1)
    coincident = coordinate_exact | numerically_zero
    # Evaluate the norm after a detached common rescaling. ``torch.hypot`` is
    # value-stable but its backward squares the tiny result, producing NaNs for
    # otherwise representable multi-handle derivatives. This form keeps the
    # square root near unit scale and also handles subnormal components without
    # ever forming an overflowing reciprocal.
    seed = torch.zeros_like(normalized)
    seed[..., 0] = 1
    norm_input = torch.where(coincident.unsqueeze(-1), seed, normalized)
    maximum = norm_input.abs().amax(dim=-1, keepdim=True)
    normal_maximum = torch.where(
        maximum < finfo.tiny, torch.ones_like(maximum), maximum
    )
    norm_scale = torch.where(
        maximum < finfo.tiny,
        torch.full_like(maximum, finfo.max),
        normal_maximum.reciprocal(),
    ).detach()
    scaled_norm = torch.sqrt((norm_input * norm_scale).square().sum(dim=-1))
    q = scaled_norm / norm_scale.squeeze(-1)
    q = torch.where(coincident, torch.zeros_like(q), q)
    q_squared = normalized.square().sum(dim=-1) if compute_q_squared else None
    # A nonzero coordinate delta can normalize below the smallest representable
    # value. Its field and geometry derivatives are indistinguishable from the
    # exact branch at the working precision, so use the defined coincidence
    # subgradient instead of allowing a zero denominator.
    return coincident, q, q_squared


def _compact_shepard_query_chunk(
    query: torch.Tensor,
    control_points: torch.Tensor,
    control_displacements: torch.Tensor,
    radius: torch.Tensor,
    control_indices: torch.Tensor,
    control_chunk: int | None,
) -> torch.Tensor:
    """Evaluate one query block, optionally under activation checkpointing."""

    batch_size, chunk_points, num_dims = query.shape
    num_controls = control_points.shape[1]
    minimum_q = torch.full(
        (batch_size, chunk_points),
        torch.inf,
        dtype=query.dtype,
        device=query.device,
    )
    minimum_index = torch.zeros(
        (batch_size, chunk_points),
        dtype=torch.long,
        device=query.device,
    )
    reference_q_squared = torch.zeros(
        (batch_size, chunk_points),
        dtype=query.dtype,
        device=query.device,
    )
    exact_count = torch.zeros(
        (batch_size, chunk_points),
        dtype=torch.int64,
        device=query.device,
    )
    exact_displacement_sum = torch.zeros(
        (batch_size, chunk_points, num_dims),
        dtype=query.dtype,
        device=query.device,
    )

    control_starts = (
        (0,) if control_chunk is None else range(0, num_controls, control_chunk)
    )

    # First pass: find the stable common scale and collect exact controls.
    for control_start in control_starts:
        control_stop = (
            num_controls
            if control_chunk is None
            else min(control_start + control_chunk, num_controls)
        )
        controls = control_points[:, control_start:control_stop]
        displacements = control_displacements[:, control_start:control_stop]
        radii = radius[:, control_start:control_stop]
        exact, q, q_squared = _normalized_distance(query, controls, radii)

        exact_count = exact_count + exact.sum(dim=-1)
        exact_displacement_sum = exact_displacement_sum + torch.bmm(
            exact.to(dtype=query.dtype), displacements
        )

        active = (~exact) & (q < 1)
        masked_q = torch.where(active, q, torch.full_like(q, torch.inf))
        local_minimum, local_index = masked_q.min(dim=-1)
        update_minimum = local_minimum < minimum_q
        local_q_squared = torch.gather(
            q_squared, dim=-1, index=local_index.unsqueeze(-1)
        ).squeeze(-1)
        minimum_index = torch.where(
            update_minimum, local_index + control_start, minimum_index
        )
        reference_q_squared = torch.where(
            update_minimum, local_q_squared, reference_q_squared
        )
        minimum_q = torch.where(update_minimum, local_minimum, minimum_q)

    has_active = torch.isfinite(minimum_q)
    reference_q = torch.where(has_active, minimum_q, torch.zeros_like(minimum_q))
    reference_t = (1 - reference_q).clamp_min(0)
    reference_phi = reference_t.pow(4) * (4 * reference_q + 1)
    reference_index = minimum_index.unsqueeze(-1).expand(
        batch_size, chunk_points, num_dims
    )
    reference_displacement = torch.gather(
        control_displacements, dim=1, index=reference_index
    )
    reference_displacement = torch.where(
        has_active.unsqueeze(-1),
        reference_displacement,
        torch.zeros_like(reference_displacement),
    )
    # Scale every term by the detached minimum active q^2, then divide the
    # quotient by the selected reference handle's scaled influence. The common
    # detached scale cancels algebraically, leaving a stable reference weight of
    # one and background q_ref^2 / phi_ref. Unlike an inverse-square form, this
    # retains O(q) derivatives even when q^2 underflows in the forward value.
    background = torch.where(
        has_active,
        reference_q_squared / reference_phi,
        torch.zeros_like(reference_q),
    )
    safe_reference_q = torch.where(
        has_active, reference_q, torch.ones_like(reference_q)
    )
    safe_reference_phi = torch.where(
        has_active, reference_phi, torch.ones_like(reference_phi)
    )

    # All non-reference weights share the inverse reference influence. Factor
    # its geometry dependence after summing those weights to avoid subtracting
    # O(1/q_ref) gradient terms when tied controls cancel.
    common_log_scale = (
        2 * (torch.log(safe_reference_q) - torch.log(safe_reference_q.detach()))
        + torch.log(safe_reference_phi.detach())
        - torch.log(safe_reference_phi)
    )
    common_scale = torch.exp(common_log_scale)
    other_denominator = torch.zeros_like(background)
    other_correction = torch.zeros_like(reference_displacement)

    # Second pass: accumulate the stably scaled compact Shepard quotient.
    for control_start in control_starts:
        control_stop = (
            num_controls
            if control_chunk is None
            else min(control_start + control_chunk, num_controls)
        )
        controls = control_points[:, control_start:control_stop]
        displacements = control_displacements[:, control_start:control_stop]
        radii = radius[:, control_start:control_stop]
        exact, q, _ = _normalized_distance(
            query, controls, radii, compute_q_squared=False
        )
        active = (~exact) & (q < 1)

        safe_q = torch.where(active, q, torch.ones_like(q))
        one_minus_q = (1 - safe_q).clamp_min(0)
        phi = one_minus_q.pow(4) * (4 * safe_q + 1)
        is_reference = (
            minimum_index.unsqueeze(-1) == control_indices[control_start:control_stop]
        )
        ratio_squared = torch.exp(
            2 * (torch.log(safe_reference_q.detach().unsqueeze(-1)) - torch.log(safe_q))
        )
        relative_phi = phi / safe_reference_phi.detach().unsqueeze(-1)
        other_influence = torch.where(
            active & ~is_reference,
            ratio_squared * relative_phi,
            torch.zeros_like(q),
        )
        chunk_denominator = other_influence.sum(dim=-1)
        other_denominator = other_denominator + chunk_denominator
        other_correction = (
            other_correction
            + torch.bmm(other_influence, displacements)
            - chunk_denominator.unsqueeze(-1) * reference_displacement
        )

    denominator = background + 1 + common_scale * other_denominator
    correction_numerator = (
        -background.unsqueeze(-1) * reference_displacement
        + common_scale.unsqueeze(-1) * other_correction
    )
    denominator = torch.where(has_active, denominator, torch.ones_like(denominator))
    field = reference_displacement + correction_numerator / denominator.unsqueeze(-1)
    has_exact = exact_count > 0
    exact_field = exact_displacement_sum / exact_count.clamp_min(1).unsqueeze(-1)
    return torch.where(has_exact.unsqueeze(-1), exact_field, field)


def compact_shepard_field_torch(
    points: torch.Tensor,
    control_points: torch.Tensor,
    control_displacements: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate a compact Shepard displacement field on normalized inputs.

    For non-coincident controls, the raw influence is

    .. math::

       a_j = \frac{\phi(q_j)}{q_j^2}, \qquad
       \phi(q) = (1-q)^4(4q+1), \quad 0 < q < 1,

    and the stationary background has weight one. The displacement is

    .. math::

       u(x) = \frac{\sum_j a_j d_j}{1 + \sum_j a_j}.

    Numerically, every term is multiplied by the detached minimum active
    :math:`q^2`. This leaves the quotient unchanged while preventing overflow
    near a control. At exact control coincidences, the mean displacement of all
    coincident controls is returned. With no controls inside the compact support,
    the field is zero.

    All inputs use normalized backend shapes: points ``(B, N, D)``, controls and
    control displacements ``(B, C, D)``, and radius ``(B, C)``.
    """

    batch_size, num_points, num_dims = points.shape
    num_controls = control_points.shape[1]
    if batch_size == 0 or num_points == 0 or num_controls == 0:
        # Keep every field input connected to autograd so empty-control and
        # empty-query cases report defined zero gradients, matching Warp.
        zero = _zero_dependency(control_points, control_displacements, radius)
        return points * 0 + zero

    # Dynamo cannot unroll shape-dependent Python chunk loops for symbolic
    # dimensions. Let Inductor see one vectorized block while compiling. Eager
    # execution retains byte-aware blocking below. Activation checkpointing is
    # still required for AOTAutograd training: without it, compiled backward
    # retains the complete O(B*N*C*D) pairwise graph.
    if torch.compiler.is_compiling():
        control_indices = torch.arange(num_controls, device=points.device)
        if torch.is_grad_enabled() and any(
            tensor.requires_grad
            for tensor in (points, control_points, control_displacements, radius)
        ):
            return checkpoint(
                _compact_shepard_query_chunk,
                points,
                control_points,
                control_displacements,
                radius,
                control_indices,
                None,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return _compact_shepard_query_chunk(
            points,
            control_points,
            control_displacements,
            radius,
            control_indices,
            None,
        )

    query_chunk, control_chunk = _chunk_sizes(
        batch_size, num_points, num_controls, num_dims, points.element_size()
    )
    checkpoint_chunks = (
        torch.is_grad_enabled()
        and any(
            tensor.requires_grad
            for tensor in (points, control_points, control_displacements, radius)
        )
        and _requires_chunk_checkpoint(
            batch_size,
            num_points,
            num_controls,
            num_dims,
            points.element_size(),
        )
    )
    control_indices = torch.arange(num_controls, device=points.device)
    chunks: list[torch.Tensor] = []

    for query_start in range(0, num_points, query_chunk):
        query_stop = min(query_start + query_chunk, num_points)
        query = points[:, query_start:query_stop]
        if checkpoint_chunks:
            field = checkpoint(
                _compact_shepard_query_chunk,
                query,
                control_points,
                control_displacements,
                radius,
                control_indices,
                control_chunk,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            field = _compact_shepard_query_chunk(
                query,
                control_points,
                control_displacements,
                radius,
                control_indices,
                control_chunk,
            )
        chunks.append(field)

    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)


# FFD temporaries scale with the per-point basis window (all lattice nodes for
# Bernstein, 4 per axis for B-splines) rather than with pairwise distances, so
# fewer value-sized tensors stay live than in the Shepard path.
_FFD_TEMPORARY_BYTE_BUDGET = 256 * 1024 * 1024
_FFD_LIVE_VALUE_FACTOR = 8


def _bernstein_rows(
    u: Float[torch.Tensor, "*batch"], degree: int
) -> Float[torch.Tensor, "*batch n_basis"]:
    """Evaluate all Bernstein polynomials of ``degree`` at ``u``.

    Uses the de Casteljau recurrence instead of the closed form
    ``C(p, i) u^i (1-u)^(p-i)``: the backward of ``pow`` with a zero exponent
    forms ``0 * u**-1``, which is NaN at the lattice faces ``u in {0, 1}``.
    """

    complement = 1 - u
    rows = [torch.ones_like(u)]
    for _ in range(degree):
        raised = [complement * rows[0]]
        raised.extend(
            complement * rows[i] + u * rows[i - 1] for i in range(1, len(rows))
        )
        raised.append(u * rows[-1])
        rows = raised
    return torch.stack(rows, dim=-1)


def _bspline_rows(
    t: Float[torch.Tensor, "*batch"],
) -> Float[torch.Tensor, "*batch 4"]:
    """Evaluate the four uniform cubic B-spline weights at cell parameter ``t``."""

    complement = 1 - t
    t_squared = t * t
    t_cubed = t_squared * t
    return torch.stack(
        [
            complement * complement * complement / 6,
            (3 * t_cubed - 6 * t_squared + 4) / 6,
            (-3 * t_cubed + 3 * t_squared + 3 * t + 1) / 6,
            t_cubed / 6,
        ],
        dim=-1,
    )


def _interpolating_rows(
    t: Float[torch.Tensor, "*batch"], basis: str
) -> Float[torch.Tensor, "*batch 2"]:
    """Evaluate a two-node interpolating basis at cell coordinate ``t``."""

    if basis == "linear":
        upper = t
    elif basis == "cubic_hermite":
        upper = t * t * (3 - 2 * t)
    else:
        upper = t * t * t * (t * (6 * t - 15) + 10)
    return torch.stack((1 - upper, upper), dim=-1)


def _ffd_field_chunk(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    lattice_displacements: Float[torch.Tensor, "batch lattice_nodes num_dims"],
    origin: Float[torch.Tensor, "batch num_dims"],
    extent: Float[torch.Tensor, "batch num_dims"],
    resolution: tuple[int, ...],
    basis: str,
) -> Float[torch.Tensor, "batch num_points num_dims"]:
    """Evaluate the lattice FFD displacement field for one query block."""

    batch_size, num_points, num_dims = points.shape
    u = (points - origin.unsqueeze(1)) / extent.unsqueeze(1)
    inside = ((u >= 0) & (u <= 1)).all(dim=-1)
    # Outside points are masked below, but their raw lattice coordinates can be
    # arbitrarily large or non-finite. Substituting a constant keeps every basis
    # value finite so masked rows backpropagate exact zeros instead of NaNs.
    u = torch.where(inside.unsqueeze(-1), u, 0.0)

    if basis == "bernstein":
        weights = points.new_ones(batch_size, num_points, 1)
        for d, size in enumerate(resolution):
            rows = _bernstein_rows(u[..., d], size - 1)
            weights = (weights.unsqueeze(-1) * rows.unsqueeze(-2)).reshape(
                batch_size, num_points, -1
            )
        field = torch.bmm(weights, lattice_displacements)
    else:
        interpolating = basis != "bspline"
        nodes_per_axis = 2 if interpolating else 4
        strides: list[int] = []
        stride = 1
        for size in reversed(resolution):
            strides.append(stride)
            stride *= size
        strides.reverse()
        weights = points.new_ones(batch_size, num_points, 1)
        flat_index = torch.zeros(
            (batch_size, num_points, 1), dtype=torch.long, device=points.device
        )
        offsets = torch.arange(nodes_per_axis, device=points.device)
        for d, size in enumerate(resolution):
            span = size - 1 if interpolating else size - 3
            s = u[..., d] * span
            # The clamp keeps the final span active on the inclusive upper
            # lattice face, where the raw cell index equals the span count.
            cell = s.detach().clamp(0, span - 1).floor()
            cell_coordinate = s - cell
            rows = (
                _interpolating_rows(cell_coordinate, basis)
                if interpolating
                else _bspline_rows(cell_coordinate)
            )
            node = cell.to(torch.long).unsqueeze(-1) + offsets
            weights = (weights.unsqueeze(-1) * rows.unsqueeze(-2)).reshape(
                batch_size, num_points, -1
            )
            flat_index = (
                flat_index.unsqueeze(-1) + (node * strides[d]).unsqueeze(-2)
            ).reshape(batch_size, num_points, -1)
        window = weights.shape[-1]
        gathered = torch.gather(
            lattice_displacements,
            1,
            flat_index.reshape(batch_size, num_points * window, 1).expand(
                -1, -1, num_dims
            ),
        ).reshape(batch_size, num_points, window, num_dims)
        field = (weights.unsqueeze(-1) * gathered).sum(dim=2)

    # ``where`` returns exact zeros for outside rows, even when lattice
    # displacements are non-finite. Multiplying by a mask would propagate NaNs.
    return torch.where(inside.unsqueeze(-1), field, 0.0)


def _ffd_chunk_size(
    batch_size: int,
    num_points: int,
    window: int,
    num_dims: int,
    element_size: int,
) -> int:
    """Choose a query block size within the live temporary byte budget."""

    per_point_bytes = (
        _FFD_LIVE_VALUE_FACTOR
        * element_size
        * max(batch_size, 1)
        * window
        * (num_dims + 1)
    )
    return min(num_points, max(1, _FFD_TEMPORARY_BYTE_BUDGET // per_point_bytes))


def ffd_points_torch(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    control_displacements: Float[torch.Tensor, "batch lattice_nodes num_dims"],
    origin: Float[torch.Tensor, "batch num_dims"],
    extent: Float[torch.Tensor, "batch num_dims"],
    resolution: tuple[int, ...],
    basis: str,
    point_weights: Bool[torch.Tensor, "batch num_points"]
    | Float[torch.Tensor, "batch num_points"]
    | None,
) -> Float[torch.Tensor, "batch num_points num_dims"]:
    """Deform normalized rank-3 points with a lattice free-form field."""

    batch_size, num_points, num_dims = points.shape
    if batch_size == 0 or num_points == 0:
        # The point identity supplies its gradient directly. Connect every other
        # differentiable input through one scalar zero instead of evaluating an
        # empty field.
        zero = _zero_dependency(control_displacements, point_weights)
        return points + zero

    needs_grad = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (points, control_displacements)
    )
    if torch.compiler.is_compiling():
        # Dynamo cannot unroll shape-dependent Python chunk loops for symbolic
        # dimensions. Let Inductor see one vectorized block while compiling.
        # activation checkpointing reduces saved activations for training but
        # does not enforce the eager path's temporary-memory budget.
        if needs_grad:
            field = checkpoint(
                _ffd_field_chunk,
                points,
                control_displacements,
                origin,
                extent,
                resolution,
                basis,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            field = _ffd_field_chunk(
                points, control_displacements, origin, extent, resolution, basis
            )
        return displace_points_torch(points, field, point_weights)

    window = _ffd_window_size(resolution, basis)
    query_chunk = _ffd_chunk_size(
        batch_size, num_points, window, num_dims, points.element_size()
    )
    checkpoint_chunks = needs_grad and query_chunk < num_points
    chunks: list[torch.Tensor] = []
    for query_start in range(0, num_points, query_chunk):
        query = points[:, query_start : query_start + query_chunk]
        if checkpoint_chunks:
            field = checkpoint(
                _ffd_field_chunk,
                query,
                control_displacements,
                origin,
                extent,
                resolution,
                basis,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            field = _ffd_field_chunk(
                query, control_displacements, origin, extent, resolution, basis
            )
        chunks.append(field)

    field = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)
    return displace_points_torch(points, field, point_weights)


def morph_points_torch(
    points: torch.Tensor,
    control_points: torch.Tensor,
    control_displacements: torch.Tensor,
    radius: torch.Tensor,
    point_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Morph normalized rank-3 points with a compact Shepard field."""

    if points.shape[0] == 0 or points.shape[1] == 0 or control_points.shape[1] == 0:
        # The point identity supplies its gradient directly. Connect every other
        # differentiable input through one scalar zero instead of constructing a
        # full zero field and running dense displacement over it.
        zero = _zero_dependency(
            control_points, control_displacements, radius, point_weights
        )
        return points + zero

    field = compact_shepard_field_torch(
        points, control_points, control_displacements, radius
    )
    return displace_points_torch(points, field, point_weights)


__all__ = [
    "compact_shepard_field_torch",
    "displace_points_torch",
    "ffd_points_torch",
    "morph_points_torch",
]
