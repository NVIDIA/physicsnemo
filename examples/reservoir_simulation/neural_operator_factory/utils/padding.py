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

"""Padding utilities for reservoir simulation tensors.

Design goals for this repo:
- Provide **right-side padding** helpers that are:
  - **shape-safe** across 3D datasets (2D space + time) and 4D datasets (3D space + time)
  - compatible with PyTorch limitations around non-constant padding modes
  - explicit about what dimensions are being padded

Tensor layouts used in this codebase:
- 3D dataset samples (CO2-style): (B, H, W, T, C)
- 4D dataset samples (Norne-style): (B, X, Y, Z, T, C)

Important subtlety (xFNO vs xDeepONet):
- For xDeepONet, padding must be **spatial-only** because time is handled by the trunk/query.
- For xFNO, the operator is learned over the full domain, so the convolved dimensions
  include time (e.g., SpectralConv3d over H,W,T; SpectralConv4d over X,Y,Z,T). Therefore,
  "spatial_ndim" in xFNO wrappers may include the time dimension by design.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_right_pad_to_multiple(
    spatial_shape: Sequence[int],
    *,
    multiple: int = 8,
    min_right_pad: int = 0,
) -> Tuple[int, ...]:
    """Compute right-side padding to reach a multiple (optionally with a minimum)."""
    if multiple <= 0:
        raise ValueError(f"multiple must be > 0, got {multiple}")
    if min_right_pad < 0:
        raise ValueError(f"min_right_pad must be >= 0, got {min_right_pad}")

    pads = []
    for d in spatial_shape:
        if d <= 0:
            raise ValueError(
                f"spatial dimensions must be positive, got {spatial_shape}"
            )
        to_mult = (multiple - (d % multiple)) % multiple
        # Guarantee:
        # - (d + pad) is divisible by `multiple`
        # - pad >= min_right_pad
        if to_mult >= min_right_pad:
            pad = to_mult
        else:
            # Increase by whole multiples so the final size stays aligned.
            deficit = min_right_pad - to_mult
            k = (deficit + multiple - 1) // multiple
            pad = to_mult + k * multiple
        pads.append(int(pad))
    return tuple(pads)


def compute_right_pad_to_multiple_per_dim(
    spatial_shape: Sequence[int],
    *,
    multiple: int = 8,
    min_right_pad: Union[int, Sequence[int]] = 0,
) -> Tuple[int, ...]:
    """Like `compute_right_pad_to_multiple`, but supports per-dimension minimum padding."""
    if isinstance(min_right_pad, int):
        mins = [min_right_pad] * len(spatial_shape)
    else:
        mins = list(min_right_pad)
        if len(mins) != len(spatial_shape):
            raise ValueError(
                f"min_right_pad length must match spatial_shape length "
                f"({len(mins)} vs {len(spatial_shape)})"
            )
    return tuple(
        compute_right_pad_to_multiple((d,), multiple=multiple, min_right_pad=m)[0]
        for d, m in zip(spatial_shape, mins)
    )


def pad_right_nd(
    x: Tensor,
    *,
    dims: Sequence[int],
    right_pad: Sequence[int],
    mode: str = "replicate",
    constant_value: float = 0.0,
) -> Tensor:
    """Right-pad arbitrary dims for tensors of any rank.

    This is implemented manually so it works for `mode="replicate"` even when
    PyTorch's `F.pad` doesn't support the tensor rank (e.g. 6D+ tensors).
    """
    if len(dims) != len(right_pad):
        raise ValueError("dims and right_pad must have the same length")
    if not dims:
        return x

    for dim, pad in zip(dims, right_pad):
        pad = int(pad)
        if pad <= 0:
            continue
        if dim < 0:
            dim = x.dim() + dim
        if dim < 0 or dim >= x.dim():
            raise ValueError(f"invalid dim {dim} for x.dim()={x.dim()}")

        if mode == "constant":
            pad_shape = list(x.shape)
            pad_shape[dim] = pad
            pad_tensor = torch.full(
                pad_shape, float(constant_value), dtype=x.dtype, device=x.device
            )
            x = torch.cat([x, pad_tensor], dim=dim)
            continue

        if mode != "replicate":
            raise ValueError(
                f"pad_right_nd currently supports mode='replicate' or 'constant', got {mode}"
            )

        last = x.select(dim, x.size(dim) - 1).unsqueeze(dim)  # singleton at dim
        expand_shape = list(x.shape)
        expand_shape[dim] = pad
        pad_tensor = last.expand(*expand_shape)
        x = torch.cat([x, pad_tensor], dim=dim)

    return x


def pad_spatial_right(
    x: Tensor,
    *,
    spatial_ndim: int,
    right_pad: Sequence[int],
    mode: str = "replicate",
    constant_value: float = 0.0,
) -> Tensor:
    """Pad only the first `spatial_ndim` dims after batch on the right.

    Assumes `x` is shaped:
      (B, *spatial, *rest)
    """
    if spatial_ndim not in (2, 3, 4):
        raise ValueError(f"spatial_ndim must be 2, 3, or 4, got {spatial_ndim}")
    if len(right_pad) != spatial_ndim:
        raise ValueError(
            f"right_pad must have length {spatial_ndim}, got {len(right_pad)}"
        )
    if x.dim() < 1 + spatial_ndim:
        raise ValueError(
            f"expected x.dim() >= {1 + spatial_ndim}, got x.dim()={x.dim()}"
        )
    if all(int(p) == 0 for p in right_pad):
        return x

    # For 4 spatial dims, fall back to generic implementation (works for 6D tensors).
    if spatial_ndim == 4:
        dims = [1, 2, 3, 4]
        return pad_right_nd(
            x, dims=dims, right_pad=right_pad, mode=mode, constant_value=constant_value
        )

    # For 2D/3D spatial, use a reshape trick so we can call F.pad with replicate.
    b = x.shape[0]
    spatial_shape = x.shape[1 : 1 + spatial_ndim]
    rest_shape = x.shape[1 + spatial_ndim :]
    rest_prod = (
        1 if len(rest_shape) == 0 else int(torch.tensor(rest_shape).prod().item())
    )

    # (B, *spatial, *rest) -> (B, rest_prod, *spatial)
    x_reshaped = x.reshape(b, *spatial_shape, rest_prod).permute(
        0, spatial_ndim + 1, *range(1, 1 + spatial_ndim)
    )

    if spatial_ndim == 2:
        pad_h, pad_w = (int(p) for p in right_pad)
        pad = (0, pad_w, 0, pad_h)
    else:
        pad_x, pad_y, pad_z = (int(p) for p in right_pad)
        pad = (0, pad_z, 0, pad_y, 0, pad_x)

    if mode == "constant":
        x_padded = F.pad(x_reshaped, pad, mode="constant", value=float(constant_value))
    else:
        x_padded = F.pad(x_reshaped, pad, mode=mode)

    padded_spatial = x_padded.shape[2 : 2 + spatial_ndim]
    return x_padded.permute(0, *range(2, 2 + spatial_ndim), 1).reshape(
        b, *padded_spatial, *rest_shape
    )
