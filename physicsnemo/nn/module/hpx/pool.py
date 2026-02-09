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
from jaxtyping import Float

from .layers import HEALPixLayer


class HEALPixMaxPool(torch.nn.Module):
    r"""
    HEALPix-aware max pooling wrapper around ``torch.nn.MaxPool2d``.

    Parameters
    ----------
    geometry_layer : torch.nn.Module, optional
        Wrapper that applies pooling per HEALPix face. Defaults to :class:`HEALPixLayer`.
    pooling : int, optional
        Kernel size and stride for pooling. Defaults to ``2``.
    enable_nhwc : bool, optional
        If ``True``, operate on channels-last tensors.
    enable_healpixpad : bool, optional
        Enable HEALPix padding when available.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B \cdot F, C, H, W)`.

    Outputs
    -------
    torch.Tensor
        Pooled tensor with spatial dimensions downsampled by ``pooling``.
    """

    def __init__(
        self,
        geometry_layer: torch.nn.Module = HEALPixLayer,
        pooling: int = 2,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ) -> None:
        super().__init__()
        self.maxpool = geometry_layer(
            layer=torch.nn.MaxPool2d,
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )

    def forward(
        self, x: Float[torch.Tensor, "batch_faces channels height width"]
    ) -> Float[torch.Tensor, "batch_faces channels pooled_height pooled_width"]:
        return self.maxpool(x)


class HEALPixAvgPool(torch.nn.Module):
    r"""
    HEALPix-aware average pooling wrapper around ``torch.nn.AvgPool2d``.

    Parameters
    ----------
    geometry_layer : torch.nn.Module, optional
        Wrapper that applies pooling per HEALPix face. Defaults to :class:`HEALPixLayer`.
    pooling : int, optional
        Kernel size and stride for pooling. Defaults to ``2``.
    enable_nhwc : bool, optional
        If ``True``, operate on channels-last tensors.
    enable_healpixpad : bool, optional
        Enable HEALPix padding when available.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B \cdot F, C, H, W)`.

    Outputs
    -------
    torch.Tensor
        Pooled tensor with spatial dimensions downsampled by ``pooling``.
    """

    def __init__(
        self,
        geometry_layer: torch.nn.Module = HEALPixLayer,
        pooling: int = 2,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ) -> None:
        super().__init__()
        self.avgpool = geometry_layer(
            layer=torch.nn.AvgPool2d,
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )

    def forward(
        self, x: Float[torch.Tensor, "batch_faces channels height width"]
    ) -> Float[torch.Tensor, "batch_faces channels pooled_height pooled_width"]:
        return self.avgpool(x)
