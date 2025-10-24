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
from typing import Any, Callable, Dict, Tuple, Union


class EDMLoss:
    """
    Loss function proposed in the EDM paper for denoising score matching. It
    samples a noise level ``sigma``, generates an associated noisy sample ``x + n``
    and then calls the diffusion model ``model`` to predict the denoised
    sample. The loss is then computed as a noise-dependent weighted squared error
    between the predicted and the ground truth.

    The diffusion model is expected to be called with:
    ``model(x, t cond, *model_args, **model_kwargs)`` (see below for details on
    the expected arguments). It is expected to return a tensor of same shape as
    ``x``.

    Parameters
    ----------
    P_mean: float, optional, default=-1.2
        Mean value for noise level computation.
    P_std: float, optional, default=1.2
        Standard deviation for noise level computation.
    sigma_data: float, optional, default=0.5
        Standard deviation for data.

    Forward
    -------
    model : torch.nn.Module
        The diffusion model that predicts denoised latent state.
    x : torch.Tensor
        Noisy latent state of shape :math:`(B, *)`, passed as first positional
        argument to ``model``.
    cond : Dict[str, torch.Tensor], optional, default={}
        Dictionary of conditioning information for the diffusion model.
        The keys should be the names of the conditioning variables to the
        diffusion model, and the values should be the tensors passed as
        conditioning data.
    model_args : tuple, optional, default=()
        Additional positional arguments to pass to the diffusion model.
    model_kwargs : dict, optional, default={}
        Additional keyword arguments to pass to the diffusion model.

    Outputs
    -------
    torch.Tensor
        A tensor with the same shape :math:`(B, *)` as the latent state,
        representing the pixel-wise loss (not reduced).

    """

    def __init__(
        self,
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(
        self,
        model: Union[torch.nn.Module, Callable],
        x: torch.Tensor,
        cond: Dict[str, torch.Tensor] = {},
        model_args: Tuple = (),
        model_kwargs: Dict[str, Any] = {},
    ) -> torch.Tensor:
        # Sample noise level
        rnd_normal = torch.randn([x.shape[0]] + [1] * (x.ndim - 1), device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # Sample noise
        n: torch.Tensor = torch.randn_like(x) * sigma

        # Denoising step
        x_pred = model(
            x + n,
            sigma,
            cond,
            *model_args,
            **model_kwargs,
        )
        loss = weight * ((x_pred - x) ** 2)
        return loss
