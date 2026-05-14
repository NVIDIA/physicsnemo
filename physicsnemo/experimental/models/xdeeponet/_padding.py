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

"""Right-side spatial padding helpers used by the xDeepONet packed-input
forward path.

When :class:`~physicsnemo.experimental.models.xdeeponet.DeepONet` is
constructed with ``auto_pad=True`` it aligns spatial dimensions to a
multiple (typically 8) so that spectral and convolutional sub-branches
operate on compatible shapes.  These helpers are dimension-agnostic and
support 2D, 3D, or 4D spatial layouts.

Tensor layouts used here:
- 2D spatial samples:  ``(B, H, W, T, C)``
- 3D spatial samples:  ``(B, X, Y, Z, T, C)``

This module is private (leading underscore): the helpers are part of the
xdeeponet package's internal API surface only and may be renamed or
restructured without notice.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from jaxtyping import Shaped
from torch import Tensor


def compute_right_pad_to_multiple(
    spatial_shape: Sequence[int],
    *,
    multiple: int = 8,
    min_right_pad: int = 0,
) -> tuple[int, ...]:
    """Compute right-side padding to reach a multiple of *multiple*.

    Parameters
    ----------
    spatial_shape : Sequence[int]
        Current spatial dimension sizes.
    multiple : int, optional
        Target alignment (default ``8``).
    min_right_pad : int, optional
        Minimum right-side padding applied per dimension (default ``0``).

    Returns
    -------
    tuple[int, ...]
        Right-side padding per dimension such that ``(d + pad)`` is a multiple
        of *multiple* and ``pad >= min_right_pad``.
    """
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
        if to_mult >= min_right_pad:
            pad = to_mult
        else:
            deficit = min_right_pad - to_mult
            k = (deficit + multiple - 1) // multiple
            pad = to_mult + k * multiple
        pads.append(int(pad))
    return tuple(pads)


def pad_right_nd(
    x: Shaped[Tensor, "..."],
    *,
    dims: Sequence[int],
    right_pad: Sequence[int],
    mode: Literal["replicate", "constant"] = "replicate",
    constant_value: float = 0.0,
) -> Shaped[Tensor, "..."]:
    """Right-pad arbitrary dimensions of an N-D tensor.

    Implemented manually so it works for ``mode="replicate"`` even when
    :func:`torch.nn.functional.pad` does not support the tensor rank
    (e.g. 6D tensors in the 3D-spatial case).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of any rank and dtype.
    dims : Sequence[int]
        Dimensions to right-pad.  Negative indices are supported.
    right_pad : Sequence[int]
        Right-side padding amounts per ``dims`` entry.  Non-positive
        entries are no-ops.
    mode : str, optional
        ``"replicate"`` (default) repeats the last slice along each
        padded dim; ``"constant"`` uses ``constant_value``.
    constant_value : float, optional
        Fill value when ``mode="constant"`` (default ``0.0``).

    Returns
    -------
    torch.Tensor
        Tensor of the same rank and dtype as ``x`` with the specified
        dimensions right-padded.
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
                f"pad_right_nd supports mode='replicate' or 'constant', got {mode}"
            )

        last = x.select(dim, x.size(dim) - 1).unsqueeze(dim)
        expand_shape = list(x.shape)
        expand_shape[dim] = pad
        pad_tensor = last.expand(*expand_shape)
        x = torch.cat([x, pad_tensor], dim=dim)

    return x


def pad_spatial_right(
    x: Shaped[Tensor, "..."],
    *,
    spatial_ndim: int,
    right_pad: Sequence[int],
    mode: Literal["replicate", "constant"] = "replicate",
    constant_value: float = 0.0,
) -> Shaped[Tensor, "..."]:
    """Right-pad the first *spatial_ndim* dimensions after the batch dim.

    Assumes ``x`` is shaped ``(B, *spatial, *rest)``.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor shaped ``(B, *spatial, *rest)``; any dtype is
        accepted.  Must satisfy ``x.dim() >= 1 + spatial_ndim``.
    spatial_ndim : int
        Number of spatial dimensions immediately following the batch
        dim.  Must be ``2``, ``3``, or ``4``.
    right_pad : Sequence[int]
        Right-side padding amounts per spatial dimension; must have
        length ``spatial_ndim``.  Non-positive entries are no-ops.
    mode : str, optional
        ``"replicate"`` (default) or ``"constant"``.
    constant_value : float, optional
        Fill value when ``mode="constant"`` (default ``0.0``).

    Returns
    -------
    torch.Tensor
        Tensor of the same rank and dtype as ``x`` with the spatial
        dimensions right-padded.
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

    # For 4 spatial dims fall back to the generic implementation (works for 6D+).
    if spatial_ndim == 4:
        dims = [1, 2, 3, 4]
        return pad_right_nd(
            x,
            dims=dims,
            right_pad=right_pad,
            mode=mode,
            constant_value=constant_value,
        )

    # For 2D/3D spatial, use a reshape trick so F.pad(replicate) applies.
    b = x.shape[0]
    spatial_shape = x.shape[1 : 1 + spatial_ndim]
    rest_shape = x.shape[1 + spatial_ndim :]
    rest_prod = math.prod(rest_shape)

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


__all__ = [
    "compute_right_pad_to_multiple",
    "pad_right_nd",
    "pad_spatial_right",
]
