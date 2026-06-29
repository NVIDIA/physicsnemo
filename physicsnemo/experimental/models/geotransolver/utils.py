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

r"""Small tensor helpers shared across GeoTransolver context projectors."""

from __future__ import annotations

import torch
from jaxtyping import Float


def structured_grid_to_conv_input(
    x: Float[torch.Tensor, "batch tokens channels"],
    spatial_shape: tuple[int, ...],
) -> Float[torch.Tensor, "batch channels ..."]:
    r"""Reshape a flat token tensor to spatial layout for Conv2d/Conv3d.

    Converts :math:`(B, N, C)` to :math:`(B, C, H, W)` (2D) or
    :math:`(B, C, H, W, D)` (3D) so structured projectors can apply spatial
    convolutions.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape :math:`(B, N, C)`.
    spatial_shape : tuple[int, ...]
        :math:`(H, W)` for 2D or :math:`(H, W, D)` for 3D. The product must
        equal :math:`N`.

    Returns
    -------
    torch.Tensor
        Tensor of shape :math:`(B, C, H, W)` or :math:`(B, C, H, W, D)`.

    Raises
    ------
    ValueError
        If ``spatial_shape`` is not length 2 or 3, or its product does not
        match the token dimension :math:`N`.
    """
    batch, tokens, channels = x.shape
    expected = 1
    for s in spatial_shape:
        expected *= s
    if tokens != expected:
        raise ValueError(
            f"Expected N={expected} tokens for grid {tuple(spatial_shape)}, "
            f"got N={tokens}"
        )

    if len(spatial_shape) == 2:
        H, W = spatial_shape
        return x.view(batch, H, W, channels).permute(0, 3, 1, 2)
    if len(spatial_shape) == 3:
        H, W, D = spatial_shape
        return x.view(batch, H, W, D, channels).permute(0, 4, 1, 2, 3)
    raise ValueError(
        f"spatial_shape must have length 2 or 3, got {tuple(spatial_shape)}"
    )


def tensors_alias(
    a: Float[torch.Tensor, "..."],
    b: Float[torch.Tensor, "..."],
) -> bool:
    r"""Return ``True`` when ``a`` and ``b`` are guaranteed to hold identical data.

    This is a sync-free, *sufficient* aliasing test: it confirms the two tensors
    are the same object, or distinct views over the same storage with matching
    shape, dtype, stride, and offset. A plain ``is`` check is not enough because
    callers may pass separately-created views of the same storage; a value
    comparison is avoided because it would force a host sync.

    Parameters
    ----------
    a, b : torch.Tensor
        Candidate tensors to compare.

    Returns
    -------
    bool
        ``True`` if ``a`` and ``b`` are element-for-element equal.
    """
    return a is b or (
        a.shape == b.shape
        and a.dtype == b.dtype
        and a.stride() == b.stride()
        and a.storage_offset() == b.storage_offset()
        and a.data_ptr() == b.data_ptr()
    )
