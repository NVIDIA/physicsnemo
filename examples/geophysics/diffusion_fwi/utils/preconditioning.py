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

from typing import Callable, Dict, Any, Union, Tuple

import torch


def edm_precond(
    model: Union[torch.nn.Module, Callable[..., torch.Tensor]],
    x: torch.Tensor,
    sigma: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    sigma_data: float = 0.5,
    model_args: Tuple = (),
    model_kwargs: Dict[str, Any] = {},
) -> torch.Tensor:
    r"""
    Computes parameters :math:`(c_{skip}, c_{out}, c_{in}, c_{noise})`, and
    uses them to apply EDM preconditioning to a diffusion model.

    Parameters
    ----------
    model : Union[torch.nn.Module, Callable[..., torch.Tensor]]
        Diffusion model to be wrapped with EDM preconditioning.
    x : torch.Tensor
        Latent state :math:`\mathbf{x}_t` of shape :math:`(B, *)`.
    sigma : torch.Tensor
        Noise level :math:`\sigma_t`. Should be of shape :math:`(B,)`.
    cond : Dict[str, torch.Tensor]
        Dictionary of conditioning information for the model. The keys should
        be the names of the conditioning variables to the model, and the values
        should be the tensors passed as conditioning data.
    sigma_data : float, optional, default=0.5
        Expected standard deviation of the training data.
    model_args : Tuple, optional, default=()
        Additional positional arguments to pass to the wrapped model.
    model_kwargs : Dict[str, Any], optional, default={}
        Additional keyword arguments to pass to the wrapped model.

    Returns
    -------
    torch.Tensor
        The output tensor from the diffusion model, with the same shape
        :math:`(B, *)` as the latent state ``x``.
    """

    # Compute conditioning parameters
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in = 1 / (sigma_data**2 + sigma**2).sqrt()
    c_noise = sigma.log() / 4

    # Apply conditioning to input
    x_precond = c_in.view(-1, *[1] * (x.dim() - 1)) * x

    # Call model with conditioned input
    F_x = model(
        x_precond,
        c_noise.flatten(),
        cond,
        *model_args,
        **model_kwargs,
    )

    # Apply output conditioning
    D_x = c_skip.view(-1, *[1] * (x.dim() - 1)) * x + c_out.view(
        -1, *[1] * (x.dim() - 1)
    ) * F_x.to(torch.float32)

    return D_x
