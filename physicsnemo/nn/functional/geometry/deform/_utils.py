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

"""Shared structural validation and normalization for deformation backends."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Literal, TypeAlias

import torch
from jaxtyping import Bool, Float

_FFDBasis: TypeAlias = Literal[
    "bernstein", "bspline", "linear", "cubic_hermite", "quintic_hermite"
]

_FFD_MIN_NODES: dict[_FFDBasis, int] = {
    "bernstein": 2,
    "bspline": 4,
    "linear": 2,
    "cubic_hermite": 2,
    "quintic_hermite": 2,
}


def _ffd_window_size(resolution: Sequence[int], basis: str) -> int:
    """Return the number of lattice nodes influencing one query point."""

    if basis == "bspline":
        return 4 ** len(resolution)
    if basis != "bernstein":
        return 2 ** len(resolution)
    return math.prod(resolution)


def _zero_dependency(
    tensor: torch.Tensor,
    *others: torch.Tensor | None,
) -> torch.Tensor:
    """Return scalar zero connected to every differentiable tensor argument."""

    dependency = tensor.sum()
    for other in others:
        if other is not None and other.dtype != torch.bool:
            dependency = dependency + other.sum()
    return dependency * 0


def _validate_points(tensor: torch.Tensor, name: str) -> None:
    """Validate an independent point-coordinate tensor."""

    if tensor.ndim not in (2, 3):
        raise ValueError(
            f"{name} must have shape (N, D) or (B, N, D), got {tuple(tensor.shape)}"
        )
    if tensor.shape[-1] < 1:
        raise ValueError(f"{name} coordinate dimension must be at least 1")
    if tensor.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            f"{name} must have dtype torch.float32 or torch.float64, got {tensor.dtype}"
        )


def _validate_layout(
    tensor: torch.Tensor,
    reference: torch.Tensor,
    names: str,
    same_shape: bool = False,
) -> None:
    """Validate tensor layout constraints that must not broadcast or promote."""

    if same_shape and tensor.shape != reference.shape:
        raise ValueError(
            f"{names} must have identical shapes, got "
            f"{tuple(tensor.shape)} and {tuple(reference.shape)}"
        )
    if tensor.device != reference.device:
        raise ValueError(
            f"{names} must be on the same device, got "
            f"{tensor.device} and {reference.device}"
        )
    if tensor.dtype != reference.dtype:
        raise TypeError(
            f"{names} must have the same dtype, got "
            f"{tensor.dtype} and {reference.dtype}"
        )


def _as_batched(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Normalize ``(N, D)`` to ``(1, N, D)`` and retain rank information."""

    if tensor.ndim == 2:
        return tensor.unsqueeze(0), True
    return tensor, False


def _normalize_point_weights(
    point_weights: torch.Tensor | None,
    points: torch.Tensor,
    was_unbatched: bool,
) -> torch.Tensor | None:
    """Normalize optional per-point weights to ``(B, N)`` without copying."""

    if point_weights is None:
        return None

    batch_size, num_points = points.shape[:2]
    expected = (num_points,) if was_unbatched else (batch_size, num_points)
    if tuple(point_weights.shape) != expected:
        raise ValueError(
            "point_weights must have shape "
            f"{expected}, got {tuple(point_weights.shape)}"
        )
    if point_weights.device != points.device:
        raise ValueError(
            "point_weights and points must be on the same device, got "
            f"{point_weights.device} and {points.device}"
        )
    if point_weights.dtype not in (torch.bool, points.dtype):
        raise TypeError(
            "point_weights must have bool dtype or the same dtype as points, got "
            f"{point_weights.dtype} and {points.dtype}"
        )
    return point_weights.unsqueeze(0) if was_unbatched else point_weights


def normalize_displace_inputs(
    points: torch.Tensor,
    displacement: torch.Tensor,
    point_weights: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    bool,
]:
    """Validate and normalize dense-displacement inputs."""

    _validate_points(points, "points")
    _validate_layout(displacement, points, "points and displacement", True)

    points_b3, was_unbatched = _as_batched(points)
    displacement_b3, _ = _as_batched(displacement)
    return (
        points_b3,
        displacement_b3,
        _normalize_point_weights(point_weights, points_b3, was_unbatched),
        was_unbatched,
    )


def _normalize_radius(
    radius: float | torch.Tensor,
    controls: torch.Tensor,
    was_unbatched: bool,
) -> torch.Tensor:
    """Normalize scalar, per-control, or aligned-batch radii to ``(B, C)``."""

    batch_size, num_controls = controls.shape[:2]
    if isinstance(radius, torch.Tensor):
        _validate_layout(radius, controls, "tensor-valued radius and controls")
        if radius.ndim == 0:
            normalized = radius.reshape(1, 1).expand(batch_size, num_controls)
        elif tuple(radius.shape) == (num_controls,):
            normalized = radius.unsqueeze(0).expand(batch_size, num_controls)
        elif not was_unbatched and tuple(radius.shape) == (
            batch_size,
            num_controls,
        ):
            normalized = radius
        else:
            expected = (
                "a scalar or shape (C,)"
                if was_unbatched
                else (
                    "a scalar, shape (C,), or aligned shape "
                    f"(B, C)={(batch_size, num_controls)}"
                )
            )
            raise ValueError(f"radius must be {expected}, got {tuple(radius.shape)}")
    elif isinstance(radius, Real) and not isinstance(radius, bool):
        if num_controls > 0:
            finfo = torch.finfo(controls.dtype)
            if torch.compiler.is_compiling():
                # Dynamo may generalize a call-time Python scalar internally
                # while leaving its source-level type as ``float``. Ask its
                # symbolic-shape machinery whether each predicate is statically
                # known instead of relying on ``isinstance(SymFloat)``. Keep
                # the predicates on ``radius`` itself: converting a SymFloat
                # with ``float`` would specialize the graph to each call-time
                # value. This validates literals during tracing and skips only
                # predicates that depend on a generalized runtime scalar.
                from torch.fx.experimental.symbolic_shapes import (
                    statically_known_true,
                )

                # Preserve eager's ValueError through Dynamo tracing.
                if statically_known_true(radius != radius) or statically_known_true(
                    abs(radius) == math.inf
                ):
                    torch._check_value(False, lambda: "radius must be finite")
                if statically_known_true(radius <= 0):
                    torch._check_value(
                        False, lambda: "radius must be strictly positive"
                    )
                if statically_known_true(radius > finfo.max):
                    torch._check_value(
                        False, lambda: "radius must be finite in the control dtype"
                    )
                if statically_known_true(radius < finfo.tiny * finfo.eps):
                    torch._check_value(
                        False,
                        lambda: "radius must be strictly positive in the control dtype",
                    )
            else:
                radius_value = float(radius)
                if not math.isfinite(radius_value):
                    raise ValueError("radius must be finite")
                if radius_value <= 0:
                    raise ValueError("radius must be strictly positive")
                if radius_value > finfo.max:
                    raise ValueError("radius must be finite in the control dtype")
                if radius_value < finfo.tiny * finfo.eps:
                    raise ValueError(
                        "radius must be strictly positive in the control dtype"
                    )
        # ``as_tensor`` specializes a Dynamo SymFloat to its current value.
        # ``tensor`` keeps generalized call-time radii in one graph.
        normalized = (
            torch.tensor(radius, dtype=controls.dtype, device=controls.device)
            .reshape(1, 1)
            .expand(batch_size, num_controls)
        )
    else:
        raise TypeError(
            "radius must be a positive finite Python real scalar or floating-point tensor"
        )

    return normalized


def normalize_morph_inputs(
    points: torch.Tensor,
    control_points: torch.Tensor,
    control_displacements: torch.Tensor,
    radius: float | torch.Tensor,
    point_weights: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    bool,
]:
    """Validate and normalize compact-Shepard morphing inputs."""

    _validate_points(points, "points")
    _validate_points(control_points, "control_points")
    _validate_layout(
        control_displacements,
        control_points,
        "control_points and control_displacements",
        True,
    )
    if points.ndim != control_points.ndim:
        raise ValueError(
            "points and controls must both be unbatched or both be batched. Got ranks "
            f"{points.ndim} and {control_points.ndim}"
        )
    if points.shape[-1] != control_points.shape[-1]:
        raise ValueError(
            "points and control_points must have the same coordinate dimension, got "
            f"{points.shape[-1]} and {control_points.shape[-1]}"
        )
    if points.ndim == 3 and points.shape[0] != control_points.shape[0]:
        raise ValueError(
            "batched points and controls must have aligned batch sizes, got "
            f"{points.shape[0]} and {control_points.shape[0]}"
        )
    _validate_layout(control_points, points, "points and control_points")

    points_b3, was_unbatched = _as_batched(points)
    controls_b3, _ = _as_batched(control_points)
    control_displacements_b3, _ = _as_batched(control_displacements)
    return (
        points_b3,
        controls_b3,
        control_displacements_b3,
        _normalize_radius(radius, controls_b3, was_unbatched),
        _normalize_point_weights(point_weights, points_b3, was_unbatched),
        was_unbatched,
    )


def _normalize_lattice_box(
    value: Float[torch.Tensor, "*box_batch num_dims"] | Sequence[float],
    name: str,
    points: Float[torch.Tensor, "batch num_points num_dims"],
    was_unbatched: bool,
    require_positive: bool,
) -> Float[torch.Tensor, "batch num_dims"]:
    """Normalize one lattice-box vector to ``(B, D)`` without copying tensors.

    Tensor values are trusted at runtime because validating them would force a
    device synchronization. Sequences are converted on the host and validated.
    """

    batch_size, num_dims = points.shape[0], points.shape[-1]
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError(
                f"{name} is non-differentiable lattice configuration and must not "
                "require grad. Optimize control_displacements instead"
            )
        _validate_layout(value, points, f"points and {name}")
        if tuple(value.shape) == (num_dims,):
            return value.reshape(1, num_dims).expand(batch_size, num_dims)
        if not was_unbatched and tuple(value.shape) == (batch_size, num_dims):
            return value
        expected = (
            f"shape ({num_dims},)"
            if was_unbatched
            else f"shape ({num_dims},) or aligned shape ({batch_size}, {num_dims})"
        )
        raise ValueError(f"{name} must have {expected}, got {tuple(value.shape)}")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != num_dims or not all(
            isinstance(entry, Real) and not isinstance(entry, bool) for entry in value
        ):
            raise TypeError(
                f"{name} must contain exactly {num_dims} real values, got {value!r}"
            )
        finfo = torch.finfo(points.dtype)
        if torch.compiler.is_compiling():
            from torch.fx.experimental.symbolic_shapes import statically_known_true

            for entry in value:
                if statically_known_true(entry != entry) or statically_known_true(
                    abs(entry) == math.inf
                ):
                    torch._check_value(False, lambda: f"{name} values must be finite")
                if require_positive and statically_known_true(entry <= 0):
                    torch._check_value(
                        False, lambda: f"{name} values must be strictly positive"
                    )
                if statically_known_true(abs(entry) > finfo.max):
                    torch._check_value(
                        False,
                        lambda: (
                            f"{name} values must be finite in the points dtype "
                            f"{points.dtype}"
                        ),
                    )
                if require_positive and statically_known_true(
                    entry < finfo.tiny * finfo.eps
                ):
                    torch._check_value(
                        False,
                        lambda: (
                            f"{name} values must be strictly positive in the points "
                            f"dtype {points.dtype}"
                        ),
                    )
            values = value
        else:
            values = [float(entry) for entry in value]
            if not all(math.isfinite(entry) for entry in values):
                raise ValueError(f"{name} values must be finite, got {value!r}")
            if require_positive and not all(entry > 0 for entry in values):
                raise ValueError(
                    f"{name} values must be strictly positive, got {value!r}"
                )
            if any(abs(entry) > finfo.max for entry in values):
                raise ValueError(
                    f"{name} values must be finite in the points dtype "
                    f"{points.dtype}, got {value!r}"
                )
            if require_positive and any(
                entry < finfo.tiny * finfo.eps for entry in values
            ):
                raise ValueError(
                    f"{name} values must be strictly positive in the points dtype "
                    f"{points.dtype}, got {value!r}"
                )
        return (
            torch.tensor(values, dtype=points.dtype, device=points.device)
            .reshape(1, num_dims)
            .expand(batch_size, num_dims)
        )
    raise TypeError(
        f"{name} must be a torch.Tensor or a sequence of {num_dims} reals, got "
        f"{type(value).__name__}"
    )


def normalize_ffd_inputs(
    points: Float[torch.Tensor, "*batch num_points num_dims"],
    control_displacements: Float[torch.Tensor, "*lattice_resolution num_dims"],
    origin: Float[torch.Tensor, "*box_batch num_dims"] | Sequence[float],
    extent: Float[torch.Tensor, "*box_batch num_dims"] | Sequence[float],
    basis: str,
    point_weights: Bool[torch.Tensor, "*batch num_points"]
    | Float[torch.Tensor, "*batch num_points"]
    | None,
) -> tuple[
    Float[torch.Tensor, "batch num_points num_dims"],
    Float[torch.Tensor, "batch lattice_nodes num_dims"],
    Float[torch.Tensor, "batch num_dims"],
    Float[torch.Tensor, "batch num_dims"],
    tuple[int, ...],
    Bool[torch.Tensor, "batch num_points"]
    | Float[torch.Tensor, "batch num_points"]
    | None,
    bool,
]:
    """Validate and normalize lattice free-form deformation inputs."""

    _validate_points(points, "points")
    num_dims = points.shape[-1]
    _validate_layout(control_displacements, points, "points and control_displacements")
    points_unbatched = points.ndim == 2
    expected_rank = num_dims + (1 if points_unbatched else 2)
    if control_displacements.ndim != expected_rank:
        # Dynamo mis-traces tuple membership over symbolic ranks, so compare
        # ranks explicitly.
        other_rank = num_dims + (2 if points_unbatched else 1)
        if control_displacements.ndim == other_rank:
            raise ValueError(
                "points and control_displacements must both be unbatched or both "
                f"be batched. Got shapes {tuple(points.shape)} and "
                f"{tuple(control_displacements.shape)}"
            )
        raise ValueError(
            "control_displacements must have shape (n_1, ..., n_D, D) or "
            "(B, n_1, ..., n_D, D) with one displacement vector per lattice node. "
            f"got shape {tuple(control_displacements.shape)} for D={num_dims}"
        )
    if control_displacements.shape[-1] != num_dims:
        raise ValueError(
            "control_displacements must store one displacement vector with "
            f"{num_dims} components per lattice node, got last dimension "
            f"{control_displacements.shape[-1]}"
        )
    if not points_unbatched and control_displacements.shape[0] != points.shape[0]:
        raise ValueError(
            "batched points and control_displacements must have aligned batch "
            f"sizes, got {points.shape[0]} and {control_displacements.shape[0]}"
        )
    resolution = tuple(
        int(size)
        for size in (
            control_displacements.shape[:-1]
            if points_unbatched
            else control_displacements.shape[1:-1]
        )
    )
    min_nodes = _FFD_MIN_NODES[basis]
    if any(size < min_nodes for size in resolution):
        raise ValueError(
            f"basis '{basis}' requires at least {min_nodes} lattice nodes per "
            f"axis, got lattice resolution {resolution}"
        )

    points_b3, was_unbatched = _as_batched(points)
    lattice = (
        control_displacements.unsqueeze(0) if was_unbatched else control_displacements
    )
    lattice_b3 = lattice.reshape(points_b3.shape[0], math.prod(resolution), num_dims)
    return (
        points_b3,
        lattice_b3,
        _normalize_lattice_box(origin, "origin", points_b3, was_unbatched, False),
        _normalize_lattice_box(extent, "extent", points_b3, was_unbatched, True),
        resolution,
        _normalize_point_weights(point_weights, points_b3, was_unbatched),
        was_unbatched,
    )


def restore_point_rank(points: torch.Tensor, was_unbatched: bool) -> torch.Tensor:
    """Restore an originally unbatched output to rank two."""

    return points.squeeze(0) if was_unbatched else points


__all__ = [
    "normalize_displace_inputs",
    "normalize_ffd_inputs",
    "normalize_morph_inputs",
    "restore_point_rank",
]
