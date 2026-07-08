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

"""Normal-projected, smoothed, and geometrically guarded mesh updates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Bool, Float

from physicsnemo.mesh.transformations.geometric import _resolve_point_field
from physicsnemo.mesh.utilities._index_tuple_ops import unique_index_tuples
from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


ValidationReport = Mapping[str, bool | int | float | torch.Tensor]


@dataclass(frozen=True)
class NormalUpdateDiagnostics:
    """Diagnostics from :func:`guarded_normal_update`.

    Top-level scalar values are detached and converted to Python numbers.
    Tensors inside ``validation`` are detached and remain on the mesh device.
    The mapping describes the accepted trial, or the last rejected trial if no
    guarded step was found.

    Attributes
    ----------
    accepted : bool
        Whether a geometrically valid update was found and applied.
    n_backtracks : int
        Number of scale reductions used by the accepted trial, or the maximum
        number attempted when no trial was accepted.
    backtracking_scale : float
        Geometric backtracking scale applied after clipping. Zero when no trial
        was accepted.
    clip_scale : float
        Global scale introduced by ``max_step`` before backtracking.
    raw_max_norm : float
        Largest norm in the supplied direction field.
    projected_max_norm : float
        Largest absolute normal component before smoothing.
    smoothed_max_norm : float
        Largest absolute normal component after smoothing.
    proposed_max_step : float
        Largest point displacement before clipping and backtracking.
    applied_max_step : float
        Largest point displacement in the accepted update, or zero when no
        trial was accepted.
    n_masked_points : int
        Number of points whose resolved point weight is zero.
    validation : Mapping[str, bool | int | float | torch.Tensor]
        Validation report for the accepted trial, or the last rejected trial.
    """

    accepted: bool
    n_backtracks: int
    backtracking_scale: float
    clip_scale: float
    raw_max_norm: float
    projected_max_norm: float
    smoothed_max_norm: float
    proposed_max_step: float
    applied_max_step: float
    n_masked_points: int
    validation: ValidationReport


def _require_finite_tensor(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _resolve_direction(
    mesh: "Mesh",
    direction: str | tuple[str, ...] | Float[torch.Tensor, "n_points n_spatial_dims"],
) -> Float[torch.Tensor, "n_points n_spatial_dims"]:
    resolved = _resolve_point_field(mesh, direction, argument_name="direction")
    if resolved.shape != mesh.points.shape:
        raise ValueError(
            "direction must have the same shape as mesh.points, got "
            f"{tuple(resolved.shape)} and {tuple(mesh.points.shape)}"
        )
    if resolved.device != mesh.points.device:
        raise ValueError(
            "direction and mesh points must be on the same device, got "
            f"{resolved.device} and {mesh.points.device}"
        )
    if resolved.dtype != mesh.points.dtype:
        raise TypeError(
            "direction and mesh points must have the same dtype, got "
            f"{resolved.dtype} and {mesh.points.dtype}"
        )
    _require_finite_tensor(resolved, "direction")
    return resolved


def _resolve_point_weights(
    mesh: "Mesh",
    point_weights: str
    | tuple[str, ...]
    | Bool[torch.Tensor, " n_points"]
    | Float[torch.Tensor, " n_points"]
    | None,
) -> Bool[torch.Tensor, " n_points"] | Float[torch.Tensor, " n_points"] | None:
    if point_weights is None:
        return None

    resolved = _resolve_point_field(mesh, point_weights, argument_name="point_weights")
    expected_shape = (mesh.n_points,)
    if resolved.shape != expected_shape:
        raise ValueError(
            "point_weights must have shape "
            f"{expected_shape}, got {tuple(resolved.shape)}"
        )
    if resolved.device != mesh.points.device:
        raise ValueError(
            "point_weights and mesh points must be on the same device, got "
            f"{resolved.device} and {mesh.points.device}"
        )
    if resolved.dtype not in (torch.bool, mesh.points.dtype):
        raise TypeError(
            "point_weights must have bool dtype or the same dtype as mesh points, got "
            f"{resolved.dtype} and {mesh.points.dtype}"
        )
    if resolved.dtype != torch.bool:
        _require_finite_tensor(resolved, "point_weights")

    return resolved


def _validate_parameters(
    *,
    smoothing_iterations: int,
    smoothing_relaxation: float,
    max_step: float | None,
    backtracking_factor: float,
    max_backtracks: int,
    validation_tolerance: float | None,
    min_cell_measure_ratio: float,
) -> None:
    if not isinstance(smoothing_iterations, int) or isinstance(
        smoothing_iterations, bool
    ):
        raise TypeError(
            "smoothing_iterations must be an integer, got "
            f"{type(smoothing_iterations).__name__}"
        )
    if smoothing_iterations < 0:
        raise ValueError(
            f"smoothing_iterations must be >= 0, got {smoothing_iterations=}"
        )
    if not isinstance(smoothing_relaxation, Real) or isinstance(
        smoothing_relaxation, bool
    ):
        raise TypeError(
            "smoothing_relaxation must be a finite real scalar, got "
            f"{type(smoothing_relaxation).__name__}"
        )
    if not math.isfinite(float(smoothing_relaxation)):
        raise ValueError(
            f"smoothing_relaxation must be finite, got {smoothing_relaxation=}"
        )
    if not 0.0 <= float(smoothing_relaxation) <= 1.0:
        raise ValueError(
            f"smoothing_relaxation must be in [0, 1], got {smoothing_relaxation=}"
        )

    if max_step is not None:
        if not isinstance(max_step, Real) or isinstance(max_step, bool):
            raise TypeError(
                "max_step must be a positive finite real scalar or None, got "
                f"{type(max_step).__name__}"
            )
        if not math.isfinite(float(max_step)) or float(max_step) <= 0.0:
            raise ValueError(f"max_step must be positive and finite, got {max_step=}")

    if not isinstance(backtracking_factor, Real) or isinstance(
        backtracking_factor, bool
    ):
        raise TypeError(
            "backtracking_factor must be a finite real scalar, got "
            f"{type(backtracking_factor).__name__}"
        )
    if not math.isfinite(float(backtracking_factor)):
        raise ValueError(
            f"backtracking_factor must be finite, got {backtracking_factor=}"
        )
    if not 0.0 < float(backtracking_factor) < 1.0:
        raise ValueError(
            "backtracking_factor must be strictly between 0 and 1, got "
            f"{backtracking_factor=}"
        )
    if not isinstance(max_backtracks, int) or isinstance(max_backtracks, bool):
        raise TypeError(
            f"max_backtracks must be an integer, got {type(max_backtracks).__name__}"
        )
    if max_backtracks < 0:
        raise ValueError(f"max_backtracks must be >= 0, got {max_backtracks=}")

    if validation_tolerance is not None:
        if not isinstance(validation_tolerance, Real) or isinstance(
            validation_tolerance, bool
        ):
            raise TypeError(
                "validation_tolerance must be a real scalar or None, got "
                f"{type(validation_tolerance).__name__}"
            )
        if (
            not math.isfinite(float(validation_tolerance))
            or float(validation_tolerance) <= 0.0
        ):
            raise ValueError(
                "validation_tolerance must be positive and finite, got "
                f"{validation_tolerance=}"
            )

    if not isinstance(min_cell_measure_ratio, Real) or isinstance(
        min_cell_measure_ratio, bool
    ):
        raise TypeError(
            "min_cell_measure_ratio must be a finite real scalar, got "
            f"{type(min_cell_measure_ratio).__name__}"
        )
    if not math.isfinite(float(min_cell_measure_ratio)):
        raise ValueError(
            f"min_cell_measure_ratio must be finite, got {min_cell_measure_ratio=}"
        )
    if not 0.0 <= float(min_cell_measure_ratio) <= 1.0:
        raise ValueError(
            f"min_cell_measure_ratio must be in [0, 1], got {min_cell_measure_ratio=}"
        )


def _orientation_defects(mesh: "Mesh") -> tuple[torch.Tensor, torch.Tensor]:
    """Return nonmanifold and inconsistently oriented codimension-one facets."""
    n_cell_vertices = mesh.cells.shape[1]
    oriented_facets = []
    orientation_signs = []
    for omitted_vertex in range(n_cell_vertices):
        facet = torch.cat(
            (
                mesh.cells[:, :omitted_vertex],
                mesh.cells[:, omitted_vertex + 1 :],
            ),
            dim=-1,
        )
        n_inversions = torch.zeros(
            mesh.n_cells, dtype=mesh.cells.dtype, device=mesh.cells.device
        )
        for left in range(mesh.n_manifold_dims):
            for right in range(left + 1, mesh.n_manifold_dims):
                n_inversions += facet[:, left] > facet[:, right]
        permutation_sign = torch.where(
            n_inversions.remainder(2) == 0,
            torch.ones_like(n_inversions),
            -torch.ones_like(n_inversions),
        )
        boundary_sign = 1 if omitted_vertex % 2 == 0 else -1
        oriented_facets.append(torch.sort(facet, dim=-1).values)
        orientation_signs.append(boundary_sign * permutation_sign)

    canonical_facets = torch.cat(oriented_facets, dim=0)
    signs = torch.cat(orientation_signs, dim=0)
    unique_facets, inverse, counts = unique_index_tuples(
        canonical_facets,
        index_bound=mesh.n_points,
        return_inverse=True,
        return_counts=True,
    )
    orientation_sum = torch.zeros_like(counts).index_add(0, inverse, signs)
    return (
        unique_facets[counts > 2],
        unique_facets[(counts == 2) & (orientation_sum != 0)],
    )


def _normal_update_validation_report(
    candidate: "Mesh",
    *,
    reference_cell_measures: torch.Tensor,
    reference_cell_normals: torch.Tensor,
    validation_tolerance: float | None,
    min_cell_measure_ratio: float,
) -> dict[str, bool | int | float | torch.Tensor]:
    """Validate a trial with one host synchronization for its scalar summary."""
    finite_points = torch.isfinite(candidate.points).all(dim=-1)
    cell_measures = candidate.cell_areas
    finite_cell_measures = torch.isfinite(cell_measures)
    relative_measures = cell_measures / reference_cell_measures.clamp_min(
        torch.finfo(reference_cell_measures.dtype).tiny
    )
    small_cells = relative_measures < min_cell_measure_ratio
    tolerance = (
        safe_eps(candidate.points.dtype)
        if validation_tolerance is None
        else validation_tolerance
    )
    degenerate_cells = cell_measures < tolerance**candidate.n_manifold_dims

    normal_alignment = (candidate.cell_normals * reference_cell_normals).sum(dim=-1)
    finite_normal_alignment = torch.isfinite(normal_alignment)
    flipped_cells = normal_alignment <= 0.0

    counts = torch.stack(
        (
            (~finite_points).sum(),
            (~finite_cell_measures).sum(),
            degenerate_cells.sum(),
            small_cells.sum(),
            (~finite_normal_alignment).sum(),
            flipped_cells.sum(),
        )
    )
    count_values = tuple(int(value) for value in counts.detach().cpu().tolist())
    (
        n_nonfinite_points,
        n_nonfinite_cell_measures,
        n_degenerate_cells,
        n_small_cells,
        n_nonfinite_normal_alignments,
        n_flipped_cells,
    ) = count_values
    report: dict[str, bool | int | float | torch.Tensor] = {
        "valid": all(value == 0 for value in count_values),
        "n_nonfinite_points": n_nonfinite_points,
        "n_nonfinite_cell_measures": n_nonfinite_cell_measures,
        "n_degenerate_cells": n_degenerate_cells,
        "n_small_relative_cells": n_small_cells,
        "n_nonfinite_cell_normal_alignments": n_nonfinite_normal_alignments,
        "n_flipped_cells": n_flipped_cells,
        "minimum_relative_cell_measure": torch.nan_to_num(
            relative_measures, nan=-math.inf
        )
        .amin()
        .detach(),
        "minimum_cell_normal_alignment": torch.nan_to_num(
            normal_alignment, nan=-math.inf
        )
        .amin()
        .detach(),
    }
    masks_and_keys = (
        (~finite_points, "nonfinite_point_indices"),
        (~finite_cell_measures, "nonfinite_cell_measure_indices"),
        (degenerate_cells, "degenerate_cell_indices"),
        (small_cells, "small_relative_cell_indices"),
        (~finite_normal_alignment, "nonfinite_cell_normal_alignment_indices"),
        (flipped_cells, "flipped_cell_indices"),
    )
    for count, (mask, key) in zip(count_values, masks_and_keys, strict=True):
        if count:
            report[key] = torch.where(mask)[0]
    return report


def guarded_normal_update(
    mesh: "Mesh",
    direction: str | tuple[str, ...] | Float[torch.Tensor, "n_points n_spatial_dims"],
    *,
    point_weights: str
    | tuple[str, ...]
    | Bool[torch.Tensor, " n_points"]
    | Float[torch.Tensor, " n_points"]
    | None = None,
    smoothing_iterations: int = 5,
    smoothing_relaxation: float = 0.2,
    max_step: float | None = None,
    backtracking_factor: float = 0.5,
    max_backtracks: int = 8,
    validation_tolerance: float | None = None,
    min_cell_measure_ratio: float = 1.0e-6,
    implementation: Literal["torch"] | None = None,
) -> tuple["Mesh", NormalUpdateDiagnostics]:
    r"""Apply a guarded, normal-only update to a codimension-one mesh.

    ``direction`` includes both sign and magnitude. For minimization with an
    ordinary objective gradient, pass a scaled ``-gradient``. A sensitivity
    such as DoMINO's :math:`d(-C_D)/dX` can be supplied directly.

    The update pipeline projects the point vectors onto the current surface
    normals, smooths the resulting scalar normal field, reconstructs a strictly
    normal vector field, applies optional point weights, and globally clips the
    largest effective point displacement. Candidate meshes are then validated;
    geometrically invalid trials are retried with a smaller global scale.

    Parameters
    ----------
    mesh : Mesh
        Locally manifold, consistently oriented codimension-one mesh to update.
        The source mesh is not modified.
    direction : str, tuple[str, ...], or Float[torch.Tensor, "n_points n_spatial_dims"]
        Signed point update vectors with the same shape, dtype, and device as
        ``mesh.points``, or a key/path resolving to such a point-data tensor.
        Float32 and float64 are supported. Scale this tensor before calling to
        control the step magnitude.
    point_weights : str, tuple[str, ...], Bool[torch.Tensor, " n_points"], Float[torch.Tensor, " n_points"], or None, optional
        Optional bool or floating-point multiplier with shape
        ``(mesh.n_points,)``, or a point-data key/path. Zero freezes a point;
        floating values are applied as supplied and may be signed or greater
        than one. Default is ``None``.
    smoothing_iterations : int, optional
        Number of normalized edge-Laplacian iterations applied to the scalar
        normal field. Default is ``5``.
    smoothing_relaxation : float, optional
        Neighbor-averaging fraction per smoothing iteration in ``[0, 1]``.
        Default is ``0.2``.
    max_step : float or None, optional
        Maximum effective displacement norm in mesh coordinate units. The
        entire field is scaled uniformly when clipping is needed, preserving
        relative update magnitudes. Default is ``None``.
    backtracking_factor : float, optional
        Scale multiplier for each rejected trial, strictly between zero and
        one. Default is ``0.5``.
    max_backtracks : int, optional
        Maximum reductions after the initial full-scale validation attempt.
        Default is ``8``.
    validation_tolerance : float or None, optional
        Absolute tolerance passed to :meth:`Mesh.validate`. ``None`` selects
        its dtype-aware tolerance. Default is ``None``.
    min_cell_measure_ratio : float, optional
        Reject cells whose measure falls below this fraction of its source
        measure. Must be in ``[0, 1]``. Default is ``1e-6``.
    implementation : {"torch"} or None, optional
        Dense-displacement backend forwarded to :meth:`Mesh.displace`.

    Returns
    -------
    tuple[Mesh, NormalUpdateDiagnostics]
        The accepted mesh and diagnostics. If every candidate is rejected, the
        original mesh is returned with ``diagnostics.accepted == False``.

    Raises
    ------
    TypeError
        If an input has an unsupported type or tensor dtype.
    ValueError
        If the mesh is empty, has zero-dimensional cells, is not codimension
        one, has invalid or inconsistently oriented source geometry, or an
        input has an invalid shape, device, value, or parameter range.

    Notes
    -----
    Backtracking selection and diagnostics are discrete and non-differentiable.
    Candidate trials are detached, then the accepted mesh is reconstructed from
    the original tensors, preserving autograd through the selected update.

    This host-controlled routine is intended as an eager optimization-step
    boundary; the backtracking loop is not compatible with full-graph capture.
    The geometric guard detects degenerate cells, large relative cell collapse,
    and cell-normal reversals relative to the source. It does not detect global
    self-intersections, and it does not evaluate whether an optimization
    objective improved. Attached data is retained with Lagrangian semantics by
    :meth:`Mesh.displace`; a stored sensitivity becomes stale after the update.
    """
    ### Validate the public contract
    if mesh.codimension != 1 or mesh.n_manifold_dims < 1:
        raise ValueError(
            "guarded_normal_update requires a positive-dimensional, "
            "codimension-one mesh with unique point normals, got "
            f"{mesh.n_manifold_dims=} and {mesh.n_spatial_dims=}"
        )
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError("guarded_normal_update requires a non-empty mesh")
    if mesh.points.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "guarded_normal_update requires mesh points with dtype "
            f"torch.float32 or torch.float64, got {mesh.points.dtype}"
        )
    _require_finite_tensor(mesh.points, "mesh.points")

    _validate_parameters(
        smoothing_iterations=smoothing_iterations,
        smoothing_relaxation=smoothing_relaxation,
        max_step=max_step,
        backtracking_factor=backtracking_factor,
        max_backtracks=max_backtracks,
        validation_tolerance=validation_tolerance,
        min_cell_measure_ratio=min_cell_measure_ratio,
    )
    direction_t = _resolve_direction(mesh, direction)
    point_weights_t = _resolve_point_weights(mesh, point_weights)

    ### Validate detached source geometry
    # A separate mesh keeps validation caches detached from the differentiable
    # update and lets every backtracking trial avoid copying attached fields.
    from physicsnemo.mesh.mesh import Mesh

    with torch.no_grad():
        baseline = Mesh(points=mesh.points.detach(), cells=mesh.cells)
        baseline_report = baseline.validate(
            check_degenerate_cells=True,
            check_duplicate_vertices=False,
            check_inverted_cells=False,
            check_out_of_bounds=True,
            check_manifoldness=False,
            tolerance=validation_tolerance,
            raise_on_error=False,
        )
        if not baseline_report["valid"]:
            raise ValueError(
                "guarded_normal_update requires locally valid source geometry; "
                f"validation report: {baseline_report}"
            )
        nonmanifold_facets, inconsistent_facets = _orientation_defects(baseline)
        if len(nonmanifold_facets) > 0:
            raise ValueError(
                "guarded_normal_update requires locally manifold source "
                "topology; facets shared by more than two cells include "
                f"{nonmanifold_facets[:10].tolist()}"
            )
        if len(inconsistent_facets) > 0:
            raise ValueError(
                "guarded_normal_update requires consistently oriented cells; "
                "shared facets with equal induced orientation include "
                f"{inconsistent_facets[:10].tolist()}"
            )
        reference_cell_measures = baseline.cell_areas
        reference_cell_normals = baseline.cell_normals
        finite_reference_geometry = torch.isfinite(reference_cell_measures).all()
        finite_reference_geometry &= torch.isfinite(reference_cell_normals).all()
        if not bool(finite_reference_geometry.item()):
            raise ValueError(
                "guarded_normal_update requires finite source cell measures and normals"
            )

    ### Build the differentiable normal update from fresh geometry caches
    # The caller may already have consumed an objective graph stored in the
    # input mesh's lazy geometry caches. Recomputing avoids a second-backward
    # failure while preserving gradients to the original point tensor.
    geometry_mesh = mesh.strip_caches()
    normals = geometry_mesh.point_normals
    normal_norms = normals.norm(dim=-1)
    undefined_normals = ~torch.isfinite(normals).all(dim=-1)
    undefined_normals |= normal_norms <= safe_eps(mesh.points.dtype)
    if bool(undefined_normals.any().item()):
        raise ValueError(
            "guarded_normal_update requires finite, nonzero point normals; "
            "repair ambiguous orientation or unreferenced points at indices "
            f"{torch.where(undefined_normals)[0][:10].tolist()}"
        )
    # ``Mesh.point_normals`` uses an absolute normalization floor, so valid
    # microscale meshes can return finite normals shorter than one. Restore unit
    # length explicitly after rejecting truly undefined normals.
    normals = normals / normal_norms.unsqueeze(-1)
    normal_direction = (direction_t * normals).sum(dim=-1)
    from physicsnemo.mesh.smoothing import smooth_point_field

    smoothed_normal_direction = smooth_point_field(
        geometry_mesh,
        normal_direction,
        n_iter=smoothing_iterations,
        relaxation_factor=float(smoothing_relaxation),
    )
    effective_displacement = smoothed_normal_direction.unsqueeze(-1) * normals
    if point_weights_t is not None:
        effective_displacement = effective_displacement * point_weights_t.to(
            dtype=mesh.points.dtype
        ).unsqueeze(-1)
    _require_finite_tensor(effective_displacement, "effective displacement")

    proposed_max_step_t = effective_displacement.norm(dim=-1).amax()
    _require_finite_tensor(proposed_max_step_t, "proposed maximum step")
    clip_scale_t = torch.ones((), dtype=mesh.points.dtype, device=mesh.points.device)
    if max_step is not None:
        max_step_t = torch.as_tensor(
            max_step, dtype=mesh.points.dtype, device=mesh.points.device
        )
        clip_scale_t = torch.clamp(
            max_step_t
            / proposed_max_step_t.clamp_min(torch.finfo(mesh.points.dtype).tiny),
            max=1.0,
        )
    clipped_displacement = effective_displacement * clip_scale_t

    ### Select the largest locally valid backtracking scale
    accepted_scale = 0.0
    accepted_attempt = max_backtracks
    validation: dict[str, bool | int | float | torch.Tensor] = dict(baseline_report)
    for attempt in range(max_backtracks + 1):
        trial_scale = float(backtracking_factor) ** attempt
        with torch.no_grad():
            trial = baseline.displace(
                trial_scale * clipped_displacement.detach(),
                implementation=implementation,
            )
            validation = _normal_update_validation_report(
                trial,
                reference_cell_measures=reference_cell_measures,
                reference_cell_normals=reference_cell_normals,
                validation_tolerance=validation_tolerance,
                min_cell_measure_ratio=float(min_cell_measure_ratio),
            )
        if validation["valid"]:
            accepted_scale = trial_scale
            accepted_attempt = attempt
            break

    ### Reconstruct the selected differentiable branch and diagnostics
    accepted = accepted_scale > 0.0
    if accepted:
        updated = mesh.displace(
            accepted_scale * clipped_displacement,
            implementation=implementation,
        )
        applied_max_step_t = proposed_max_step_t * clip_scale_t * accepted_scale
    else:
        updated = mesh
        applied_max_step_t = torch.zeros_like(proposed_max_step_t)

    n_masked_points = (
        0
        if point_weights_t is None
        else int((point_weights_t == 0).sum().detach().cpu().item())
    )
    diagnostic_values = (
        torch.stack(
            (
                clip_scale_t,
                direction_t.norm(dim=-1).amax(),
                normal_direction.abs().amax(),
                smoothed_normal_direction.abs().amax(),
                proposed_max_step_t,
                applied_max_step_t,
            )
        )
        .detach()
        .cpu()
        .tolist()
    )
    (
        clip_scale,
        raw_max_norm,
        projected_max_norm,
        smoothed_max_norm,
        proposed_max_step,
        applied_max_step,
    ) = diagnostic_values
    diagnostics = NormalUpdateDiagnostics(
        accepted=accepted,
        n_backtracks=accepted_attempt,
        backtracking_scale=accepted_scale,
        clip_scale=float(clip_scale),
        raw_max_norm=float(raw_max_norm),
        projected_max_norm=float(projected_max_norm),
        smoothed_max_norm=float(smoothed_max_norm),
        proposed_max_step=float(proposed_max_step),
        applied_max_step=float(applied_max_step),
        n_masked_points=n_masked_points,
        validation=validation,
    )
    return updated, diagnostics


__all__ = ["NormalUpdateDiagnostics", "guarded_normal_update"]
