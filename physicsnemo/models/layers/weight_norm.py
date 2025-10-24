# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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
import torch.nn as nn
import torch.nn.functional as F

import physicsnemo  # noqa: F401 for docs

Tensor = torch.Tensor


class WeightNormLinear(nn.Module):
    """Weight Norm Layer for 1D Tensors

    Parameters
    ----------
    in_features : int
        Size of the input features
    out_features : int
        Size of the output features
    bias : bool, optional
        Apply the bias to the output of linear layer, by default True

    Example
    -------
    >>> wnorm = physicsnemo.models.layers.WeightNormLinear(2,4)
    >>> input = torch.rand(2,2)
    >>> output = wnorm(input)
    >>> output.size()
    torch.Size([2, 4])
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_g = nn.Parameter(torch.empty((out_features, 1)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset normalization weights"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.weight_g, 1.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input: Tensor) -> Tensor:
        norm = self.weight.norm(dim=1, p=2, keepdim=True)
        weight = self.weight_g * self.weight / norm
        return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        """Print information about weight norm"""
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
