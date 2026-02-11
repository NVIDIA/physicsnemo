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

"""Reusable HEALPix tensor utilities and padding layers."""

import torch
from jaxtyping import Float

from physicsnemo.core.version_check import check_version_spec

from .padding import HEALPixPadding, HEALPixPaddingv2

HEALPIXPAD_AVAILABLE = check_version_spec("earth2grid", "0.1.0", hard_fail=False)


class HEALPixLayer(torch.nn.Module):
    r"""
    Wrapper that applies an arbitrary 2D layer to each HEALPix face independently.

    Parameters
    ----------
    layer : torch.nn.Module
        A callable layer class such as ``torch.nn.Conv2d``.
    **kwargs
        Parameters forwarded to ``layer``. Recognized extras:

        - ``enable_nhwc``: bool, optional
            If ``True``, use channels-last memory format.
        - ``enable_healpixpad``: bool, optional
            If ``True`` and CUDA ``earth2grid`` is available, use accelerated padding.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(..., F=12, H, W)`.

    Outputs
    -------
    torch.Tensor
        Tensor with the same leading dimensions and transformed spatial dimensions.

    Examples
    --------
    >>> conv = HEALPixLayer(torch.nn.Conv2d, in_channels=3, out_channels=8, kernel_size=3)
    >>> x = torch.randn(24, 3, 16, 16)
    >>> conv(x).shape
    torch.Size([24, 8, 16, 16])
    """

    def __init__(self, layer, **kwargs) -> None:
        super().__init__()
        layers = []

        enable_nhwc = kwargs.pop("enable_nhwc", False)
        enable_healpixpad = kwargs.pop("enable_healpixpad", False)

        if (
            layer.__bases__[0] is torch.nn.modules.conv._ConvNd
            and kwargs.get("kernel_size", 3) > 1
        ):
            kwargs["padding"] = 0
            kernel_size = kwargs.get("kernel_size", 3)
            dilation = kwargs.get("dilation", 1)
            padding = ((kernel_size - 1) // 2) * dilation
            if (
                enable_healpixpad
                and HEALPIXPAD_AVAILABLE
                and torch.cuda.is_available()
                and not enable_nhwc
            ):  # pragma: no cover
                layers.append(HEALPixPaddingv2(padding=padding))
            else:
                layers.append(HEALPixPadding(padding=padding, enable_nhwc=enable_nhwc))

        layers.append(layer(**kwargs))
        self.layers = torch.nn.Sequential(*layers)

        if enable_nhwc:
            self.layers = self.layers.to(memory_format=torch.channels_last)

    def forward(
        self, x: Float[torch.Tensor, "batch_faces channels height width"]
    ) -> Float[torch.Tensor, "batch_faces channels height width"]:
        r"""Forward pass for the HEALPix layer wrapper."""
        return self.layers(x)
