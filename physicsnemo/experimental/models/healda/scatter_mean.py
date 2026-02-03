# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
import math

import torch


def _compute_row_major_strides(shape):
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.insert(0, stride)
        stride *= size
    return strides


def scatter_mean(
    tensor: torch.Tensor,
    index: torch.Tensor,
    shape: tuple[int, ...],
    fill_value: float = float("nan"),
) -> torch.Tensor:
    """Scatter-mean values onto a multi-dimensional grid

    Args:
        tensor: [N, c] observation feature vectors
        index: [N, d] 1D grid cell index for each value in tensor
        shape: d-tuple. The size of the non value dimensions of the output array

    Returns:
        aggregated: [*shape, c] with mean-aggregated values,
            filled with fill_value at grid cells with no values
        present: (*shape) bool mask indicating which grid cells have values
    """
    strides = _compute_row_major_strides(shape)
    # manually implement the dot product since matmul doesn't support long tensors on cuda
    # avoids RuntimeError: "addmv_impl_cuda" not implemented for 'Long'
    grid_indices_flat = (index * torch.tensor(strides, device=index.device)).sum(dim=-1)
    grid_size = math.prod(shape)

    device = tensor.device
    dtype = tensor.dtype
    embedding_dim = tensor.shape[1]

    # Initialize with fill_value (typically NaN)
    values_mean = torch.full(
        (grid_size, embedding_dim), fill_value, device=device, dtype=dtype
    )

    # Use scatter_reduce with mean, expanding indices to match tensor dimensions
    grid_indices_flat_expanded = grid_indices_flat.unsqueeze(-1).expand(
        -1, embedding_dim
    )
    values_mean.scatter_reduce_(
        0, grid_indices_flat_expanded, tensor, reduce="mean", include_self=False
    )

    # Compute present mask (cells that are not fill_value)
    if math.isnan(fill_value):
        present = ~torch.isnan(values_mean[:, 0])
    else:
        present = values_mean[:, 0] != fill_value

    # Reshape
    aggregated = values_mean.view(*shape, embedding_dim)
    present = present.view(shape)

    return aggregated, present
