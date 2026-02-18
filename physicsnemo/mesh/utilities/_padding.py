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

import torch


def _pad_by_tiling_last(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Pads a tensor along its first dimension by tiling the last element.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to pad.
    size : int
        Target size for first dimension. Also accepts SymInt for torch.compile.

    Returns
    -------
    torch.Tensor
        Padded tensor.
    """
    pad_count = size - tensor.shape[0]
    padding = tensor[-1:].expand(pad_count, -1)
    return torch.cat([tensor, padding], dim=0)


def _pad_with_value(tensor: torch.Tensor, size: int, value: float) -> torch.Tensor:
    """Pads a tensor along its first dimension with a constant value.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to pad.
    size : int
        Target size for first dimension. Also accepts SymInt for torch.compile.
    value : float
        Fill value for padding.

    Returns
    -------
    torch.Tensor
        Padded tensor.
    """
    pad_count = size - tensor.shape[0]
    padding_shape = (pad_count,) + tensor.shape[1:]
    # Use new_full which inherits dtype/device from tensor
    padding = tensor.new_full(padding_shape, fill_value=value)
    return torch.cat([tensor, padding], dim=0)
