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
from itertools import combinations

import torch


def _normalize_lengths(
    lengths: float | Sequence[float], ndim: int
) -> tuple[float, ...]:
    """Normalize periodic lengths into one positive entry per axis."""
    if isinstance(lengths, (float, int)):
        lengths_tuple = tuple(float(lengths) for _ in range(ndim))
    else:
        lengths_tuple = tuple(float(v) for v in lengths)
        if len(lengths_tuple) != ndim:
            raise ValueError(
                f"lengths must have {ndim} entries for a {ndim}D field, got {len(lengths_tuple)}"
            )

    for axis, length in enumerate(lengths_tuple):
        if length <= 0.0:
            raise ValueError(f"lengths[{axis}] must be strictly positive")
    return lengths_tuple


def _validate_inputs(
    field: torch.Tensor,
    lengths: float | Sequence[float],
    order: int,
    return_mixed_derivs: bool,
) -> tuple[tuple[float, ...], torch.Tensor]:
    """Validate spectral-gradient inputs and return normalized parameters."""
    if field.ndim < 1 or field.ndim > 3:
        raise ValueError(
            f"spectral_grid_gradient supports 1D-3D fields, got field.shape={tuple(field.shape)}"
        )
    if not torch.is_floating_point(field):
        raise TypeError("field must be a floating-point tensor")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    if return_mixed_derivs and order != 2:
        raise ValueError("return_mixed_derivs=True requires order=2")
    if return_mixed_derivs and field.ndim == 1:
        raise ValueError("mixed derivatives require at least 2D fields")

    lengths_tuple = _normalize_lengths(lengths=lengths, ndim=field.ndim)

    if field.dtype in (torch.float16, torch.bfloat16):
        field_eval = field.to(torch.float32)
    else:
        field_eval = field
    return lengths_tuple, field_eval


def _wavenumbers(
    shape: Sequence[int],
    lengths: Sequence[float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Build broadcastable angular wavenumber tensors for each axis."""
    ks: list[torch.Tensor] = []
    for axis, (n_axis, length_axis) in enumerate(zip(shape, lengths)):
        freq_axis = torch.fft.fftfreq(
            n_axis,
            d=length_axis / float(n_axis),
            device=device,
            dtype=dtype,
        )
        k_axis = 2.0 * torch.pi * freq_axis
        view_shape = [1] * len(shape)
        view_shape[axis] = n_axis
        ks.append(k_axis.reshape(view_shape))
    return ks


def spectral_grid_gradient_torch(
    field: torch.Tensor,
    lengths: float | Sequence[float] = 1.0,
    order: int = 1,
    return_mixed_derivs: bool = False,
) -> torch.Tensor:
    """Compute periodic grid derivatives with spectral differentiation.

    This implementation assumes periodic boundaries along each axis and computes
    derivatives in Fourier space, then transforms back with inverse FFT.
    """
    lengths_tuple, field_eval = _validate_inputs(
        field=field,
        lengths=lengths,
        order=order,
        return_mixed_derivs=return_mixed_derivs,
    )
    ndim = field_eval.ndim

    u_hat = torch.fft.fftn(field_eval, dim=tuple(range(ndim)))
    k_axes = _wavenumbers(
        field_eval.shape,
        lengths_tuple,
        device=field_eval.device,
        dtype=field_eval.dtype,
    )

    derivatives: list[torch.Tensor] = []
    if order == 1:
        for axis in range(ndim):
            deriv_hat = (1j * k_axes[axis]) * u_hat
            derivatives.append(torch.fft.ifftn(deriv_hat, dim=tuple(range(ndim))).real)
    else:
        for axis in range(ndim):
            deriv_hat = -(k_axes[axis] * k_axes[axis]) * u_hat
            derivatives.append(torch.fft.ifftn(deriv_hat, dim=tuple(range(ndim))).real)

        if return_mixed_derivs:
            for axis_i, axis_j in combinations(range(ndim), 2):
                deriv_hat = -(k_axes[axis_i] * k_axes[axis_j]) * u_hat
                derivatives.append(
                    torch.fft.ifftn(deriv_hat, dim=tuple(range(ndim))).real
                )

    output = torch.stack(derivatives, dim=0)
    if output.dtype != field.dtype:
        return output.to(dtype=field.dtype)
    return output
