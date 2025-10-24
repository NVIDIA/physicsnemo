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

from dataclasses import dataclass
from typing import Any, Literal, Tuple, Union

import numpy as np
import torch

from physicsnemo.models.diffusion.preconditioning import EDMPrecondSuperResolution
from physicsnemo.models.meta import ModelMetaData


@dataclass
class tEDMPrecondSuperResMetaData(ModelMetaData):
    """tEDMPrecondSuperRes meta data"""

    name: str = "tEDMPrecondSuperRes"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class tEDMPrecondSuperRes(EDMPrecondSuperResolution):
    """
    Preconditioning proposed in the paper `Heavy-Tailed Diffusion Models,
    Pandey et al. <https://arxiv.org/abs/2410.14171>`_ (t-EDM). A variant of
    :class:`~physicsnemo.models.diffusion.preconditioning.EDMPrecondSuperResolution`
    that replaces the traditional Gaussian noise with a noise sampled from a
    Student-t distribution.

    Parameters
    ----------
    img_resolution : Union[int, Tuple[int, int]]
        Spatial resolution :math:`(H, W)` of the image. If a single int is provided,
        the image is assumed to be square.
    img_in_channels : int
        Number of input channels in the low-resolution input image.
    img_out_channels : int
        Number of output channels in the high-resolution output image.
    use_fp16 : bool, optional
        Whether to use half-precision floating point (FP16) for model execution,
        by default False.
    model_type : str, optional
        Class name of the underlying model. Must be one of the following:
        'SongUNet', 'SongUNetPosEmbd', 'SongUNetPosLtEmbd', 'DhariwalUNet'.
        Defaults to 'SongUNetPosEmbd'.
    sigma_data : float, optional
        Expected standard deviation of the training data, by default 0.5.
    sigma_min : float, optional
        Minimum supported noise level, by default 0.0.
    sigma_max : float, optional
        Maximum supported noise level, by default inf.
    nu : int, optional, default=10
        Number of degrees of freedom used for the Student-t distribution.
        Must be strictly greater than 2.
    **model_kwargs : dict
        Keyword arguments passed to the underlying model `__init__` method.
    """

    def __init__(
        self,
        img_resolution: Union[int, Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        model_type: Literal[
            "SongUNetPosEmbd", "SongUNetPosLtEmbd", "SongUNet", "DhariwalUNet"
        ] = "SongUNetPosEmbd",
        sigma_data: float = 0.5,
        sigma_min=0.0,
        sigma_max=float("inf"),
        nu: int = 10,
        **model_kwargs: Any,
    ):
        # NOTE: Check if nu is greater than 2. This is to ensure the variance of the
        # Student-t prior during sampling is finite.
        if nu <= 2:
            raise ValueError(f"Expected nu > 2, but got {nu}.")

        super().__init__(
            img_resolution=img_resolution,
            img_in_channels=img_in_channels,
            img_out_channels=img_out_channels,
            use_fp16=use_fp16,
            model_type=model_type,
            sigma_data=sigma_data,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            **model_kwargs,
        )
        self.nu = nu
        self.meta = tEDMPrecondSuperResMetaData()

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        sigma: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: dict,
    ):

        # Rescale sigma to account for nu scaling
        sigma *= np.sqrt(self.nu / (self.nu - 2))

        return super().forward(x, img_lr, sigma, force_fp32, **model_kwargs)
