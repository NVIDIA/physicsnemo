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

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp

_SUPPORTED_ORDERS = (2, 4)
_SUPPORTED_DERIVATIVE_ORDERS = (1, 2)

### Warp runtime initialization for custom kernels.
wp.init()
wp.config.quiet = True

### Optional launch block size override; <=0 uses Warp default autotuning.
_WARP_BLOCK_DIM = -1


def _normalize_spacing(
    spacing: float | Sequence[float], ndim: int
) -> tuple[float, ...]:
    ### Normalize scalar/list spacing into one value per axis.
    if isinstance(spacing, (float, int)):
        return tuple(float(spacing) for _ in range(ndim))
    spacing_tuple = tuple(float(x) for x in spacing)
    if len(spacing_tuple) != ndim:
        raise ValueError(
            f"spacing must have {ndim} entries for a {ndim}D field, got {len(spacing_tuple)}"
        )
    return spacing_tuple


def _validate_order(order: int) -> int:
    ### Validate finite-difference order selection.
    if not isinstance(order, int):
        raise TypeError(f"order must be an integer, got {type(order)}")
    if order not in _SUPPORTED_ORDERS:
        raise ValueError(
            f"uniform_grid_gradient supports {list(_SUPPORTED_ORDERS)} central orders, got order={order}"
        )
    return order


def _validate_derivative_order(derivative_order: int) -> int:
    ### Validate derivative-order selection (first vs pure second derivative).
    if not isinstance(derivative_order, int):
        raise TypeError(
            f"derivative_order must be an integer, got {type(derivative_order)}"
        )
    if derivative_order not in _SUPPORTED_DERIVATIVE_ORDERS:
        raise ValueError(
            "uniform_grid_gradient supports derivative_order in [1, 2], "
            f"got derivative_order={derivative_order}"
        )
    return derivative_order


def _validate_include_mixed(
    *,
    derivative_order: int,
    include_mixed: bool,
) -> None:
    ### Phase-1 guard: mixed second derivatives are intentionally not yet exposed.
    if not isinstance(include_mixed, bool):
        raise TypeError(f"include_mixed must be a bool, got {type(include_mixed)}")
    if include_mixed and derivative_order != 2:
        raise ValueError("include_mixed is only valid when derivative_order=2")
    if include_mixed:
        raise NotImplementedError(
            "include_mixed=True is not yet supported; phase-1 supports pure axis-wise "
            "second derivatives only"
        )


def _validate_field(field: torch.Tensor) -> None:
    ### Validate field shape and dtype.
    if field.ndim < 1 or field.ndim > 3:
        raise ValueError(
            f"uniform_grid_gradient supports 1D-3D fields, got {field.shape=}"
        )
    if not torch.is_floating_point(field):
        raise TypeError("field must be a floating-point tensor")


def _wp_launch(
    *,
    kernel,
    dim,
    inputs,
    device,
    stream,
) -> None:
    ### Launch a Warp kernel, optionally overriding block size.
    if _WARP_BLOCK_DIM > 0:
        wp.launch(
            kernel=kernel,
            dim=dim,
            inputs=inputs,
            device=device,
            stream=stream,
            block_dim=_WARP_BLOCK_DIM,
        )
        return
    wp.launch(
        kernel=kernel,
        dim=dim,
        inputs=inputs,
        device=device,
        stream=stream,
    )


def _warp_launch_context(field: torch.Tensor):
    ### Resolve warp launch context without per-call dynamic imports.
    if field.device.type == "cuda":
        return None, wp.stream_from_torch(torch.cuda.current_stream(field.device))
    return "cpu", None


def _launch_dim(shape: torch.Size) -> int | tuple[int, ...]:
    """Return Warp launch dimensions for 1D vs ND kernels."""
    return shape[0] if len(shape) == 1 else tuple(shape)


def _inverse_spacings(
    spacing_tuple: tuple[float, ...],
    *,
    power: int,
) -> list[float]:
    """Compute inverse spacing terms with optional square for second derivatives."""
    if power == 1:
        return [1.0 / float(dx) for dx in spacing_tuple]
    return [1.0 / float(dx * dx) for dx in spacing_tuple]


def _mixed_inverse_spacings(spacing_tuple: tuple[float, ...]) -> list[float]:
    """Compute inverse mixed spacing terms in axis-pair order."""
    return [
        1.0 / float(spacing_tuple[i] * spacing_tuple[j])
        for i in range(len(spacing_tuple))
        for j in range(i + 1, len(spacing_tuple))
    ]


def _to_wp_components(
    components: Sequence[torch.Tensor],
    count: int,
) -> list[wp.array]:
    """Convert leading tensor components to Warp arrays."""
    return [wp.from_torch(components[i], dtype=wp.float32) for i in range(count)]


def _to_wp_tensor(component: torch.Tensor) -> wp.array:
    """Convert a single tensor component to a Warp array."""
    return wp.from_torch(component, dtype=wp.float32)
