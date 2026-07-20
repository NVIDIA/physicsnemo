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

import math
from collections.abc import Sequence

import torch


def _normalize_lengths(
    lengths: float | Sequence[float], ndim: int
) -> tuple[float, ...]:
    """Normalize periodic lengths into one finite positive entry per axis."""
    if isinstance(lengths, (float, int)):
        lengths_tuple = tuple(float(lengths) for _ in range(ndim))
    else:
        lengths_tuple = tuple(float(value) for value in lengths)
        if len(lengths_tuple) != ndim:
            raise ValueError(
                f"lengths must have {ndim} entries for a {ndim}D field, "
                f"got {len(lengths_tuple)}"
            )

    for axis, length in enumerate(lengths_tuple):
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError(f"lengths[{axis}] must be finite and strictly positive")
    return lengths_tuple


def _validate_inputs(
    field: torch.Tensor,
    lengths: float | Sequence[float],
) -> tuple[tuple[float, ...], torch.Tensor]:
    """Validate an unbatched 1D-3D scalar field."""
    if field.ndim < 1 or field.ndim > 3:
        raise ValueError(
            "spectral_grid_laplacian supports 1D-3D fields, "
            f"got field.shape={tuple(field.shape)}"
        )
    if not torch.is_floating_point(field):
        raise TypeError("field must be a floating-point tensor")

    lengths_tuple = _normalize_lengths(lengths, field.ndim)
    field_eval = (
        field.to(torch.float32)
        if field.dtype in (torch.float16, torch.bfloat16)
        else field
    )
    return lengths_tuple, field_eval


def _wavenumbers(
    shape: Sequence[int],
    lengths: Sequence[float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Build broadcastable angular wavenumbers for each spatial axis."""
    wavenumbers: list[torch.Tensor] = []
    for axis, (axis_size, axis_length) in enumerate(zip(shape, lengths)):
        frequency = torch.fft.fftfreq(
            axis_size,
            d=axis_length / float(axis_size),
            device=device,
            dtype=dtype,
        )
        wavenumber = 2.0 * torch.pi * frequency
        view_shape = [1] * len(shape)
        view_shape[axis] = axis_size
        wavenumbers.append(wavenumber.reshape(view_shape))
    return wavenumbers


def spectral_grid_laplacian_torch(
    field: torch.Tensor,
    lengths: float | Sequence[float] = 1.0,
) -> torch.Tensor:
    """Compute a periodic scalar Laplacian directly in Fourier space."""
    lengths_tuple, field_eval = _validate_inputs(field, lengths)
    grid_ndim = field_eval.ndim
    spatial_dims = tuple(range(grid_ndim))
    field_hat = torch.fft.fftn(field_eval, dim=spatial_dims)
    wavenumbers = _wavenumbers(
        field_eval.shape,
        lengths_tuple,
        device=field_eval.device,
        dtype=field_eval.dtype,
    )

    # In Fourier space, the scalar Laplacian has the symbol -|k|^2.
    squared_wavenumber = wavenumbers[0].square()
    for axis in range(1, grid_ndim):
        squared_wavenumber = squared_wavenumber + wavenumbers[axis].square()
    output = torch.fft.ifftn(
        -squared_wavenumber * field_hat,
        dim=spatial_dims,
    ).real
    if output.dtype != field.dtype:
        return output.to(dtype=field.dtype)
    return output
