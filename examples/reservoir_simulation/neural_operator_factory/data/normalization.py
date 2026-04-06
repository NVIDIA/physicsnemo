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

"""Z-score normalization utilities for reservoir simulation tensors."""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass
class NormStats:
    """Per-channel z-score statistics.

    Attributes
    ----------
    input_mean, input_std : Tensor
        Shape ``(1, *[1]*spatial, 1, C)`` — broadcastable to input samples.
    output_mean, output_std : Tensor
        Scalar tensors for the output variable.
    """

    input_mean: Tensor
    input_std: Tensor
    output_mean: Tensor
    output_std: Tensor

    def as_tuple(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Unpack for ``set_normalization(*stats.as_tuple())``."""
        return self.input_mean, self.input_std, self.output_mean, self.output_std


def compute_norm_stats(input_data: Tensor, output_data: Tensor) -> NormStats:
    """Compute z-score statistics from a training split.

    Parameters
    ----------
    input_data : Tensor
        ``(N, *spatial, T, C)`` — full training input tensor.
    output_data : Tensor
        ``(N, *spatial, T)`` — full training output tensor.

    Returns
    -------
    NormStats
    """
    reduce_dims = tuple(range(input_data.dim() - 1))

    input_mean = input_data.mean(dim=reduce_dims, keepdim=True)
    input_std = input_data.std(dim=reduce_dims, keepdim=True)
    output_mean = output_data.mean()
    output_std = output_data.std()

    input_std = torch.where(input_std > 1e-6, input_std, torch.ones_like(input_std))
    if output_std < 1e-6:
        output_std = torch.tensor(1.0)

    return NormStats(input_mean, input_std, output_mean, output_std)


def identity_norm_stats(input_ndim: int, num_channels: int) -> NormStats:
    """Identity (no-op) statistics for val/test before sharing.

    Parameters
    ----------
    input_ndim : int
        Dimensionality of a full input tensor (5 for 3D, 6 for 4D).
    num_channels : int
        Number of input channels (last dimension).
    """
    shape = [1] * (input_ndim - 1) + [num_channels]
    return NormStats(
        input_mean=torch.zeros(shape),
        input_std=torch.ones(shape),
        output_mean=torch.tensor(0.0),
        output_std=torch.tensor(1.0),
    )


def normalize_sample(
    input_sample: Tensor,
    output_sample: Tensor,
    stats: NormStats,
) -> Tuple[Tensor, Tensor]:
    """Apply z-score normalization to a single sample.

    Parameters
    ----------
    input_sample : Tensor
        ``(*spatial, T, C)`` — one sample without batch dim.
    output_sample : Tensor
        ``(*spatial, T)`` — one sample without batch dim.
    stats : NormStats

    Returns
    -------
    Tuple[Tensor, Tensor]
        Normalized ``(input, output)``.
    """
    mean = stats.input_mean.squeeze(0).to(input_sample.device)
    std = stats.input_std.squeeze(0).to(input_sample.device)
    o_mean = stats.output_mean.to(output_sample.device)
    o_std = stats.output_std.to(output_sample.device)

    return (input_sample - mean) / std, (output_sample - o_mean) / o_std
