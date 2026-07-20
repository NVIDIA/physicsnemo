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
    vector_field: torch.Tensor,
    lengths: float | Sequence[float],
) -> tuple[int, tuple[float, ...], torch.Tensor]:
    """Validate a channel-first 2D or 3D vector field."""
    grid_ndim = vector_field.ndim - 1
    if grid_ndim not in (2, 3):
        raise ValueError(
            "spectral_grid_curl supports 2D or 3D vector fields, "
            f"got vector_field.shape={tuple(vector_field.shape)}"
        )
    if vector_field.shape[0] != grid_ndim:
        raise ValueError(
            "vector_field.shape[0] must equal the number of spatial dimensions "
            f"({grid_ndim}), got {vector_field.shape[0]}"
        )
    if not torch.is_floating_point(vector_field):
        raise TypeError("vector_field must be a floating-point tensor")

    lengths_tuple = _normalize_lengths(lengths, grid_ndim)
    vector_eval = (
        vector_field.to(torch.float32)
        if vector_field.dtype in (torch.float16, torch.bfloat16)
        else vector_field
    )
    return grid_ndim, lengths_tuple, vector_eval


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


def spectral_grid_curl_torch(
    vector_field: torch.Tensor,
    lengths: float | Sequence[float] = 1.0,
) -> torch.Tensor:
    """Compute periodic curl directly in Fourier space."""
    grid_ndim, lengths_tuple, vector_eval = _validate_inputs(
        vector_field,
        lengths,
    )
    # Transform spatial axes only; the leading component axis is a batch axis.
    spatial_dims = tuple(range(1, vector_eval.ndim))
    vector_hat = torch.fft.fftn(vector_eval, dim=spatial_dims)
    k = _wavenumbers(
        vector_eval.shape[1:],
        lengths_tuple,
        device=vector_eval.device,
        dtype=vector_eval.dtype,
    )

    # In Fourier space, curl has the symbol i * k cross u_hat.
    if grid_ndim == 2:
        curl_hat = 1j * (k[0] * vector_hat[1] - k[1] * vector_hat[0])
        output = torch.fft.ifftn(curl_hat, dim=(0, 1)).real
    else:
        curl_hat = torch.stack(
            (
                1j * (k[1] * vector_hat[2] - k[2] * vector_hat[1]),
                1j * (k[2] * vector_hat[0] - k[0] * vector_hat[2]),
                1j * (k[0] * vector_hat[1] - k[1] * vector_hat[0]),
            ),
            dim=0,
        )
        output = torch.fft.ifftn(curl_hat, dim=(1, 2, 3)).real
    if output.dtype != vector_field.dtype:
        return output.to(dtype=vector_field.dtype)
    return output
