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

import numpy as np
import torch
from torch import Tensor

from physicsnemo.metrics.diffusion.loss import ResidualLoss


class tEDMResidualLoss(ResidualLoss):
    """
    Loss function for denoising score matching proposed in the paper
    `Heavy-Tailed Diffusion Models, Pandey et al. <https://arxiv.org/abs/2410.14171>`_
    (t-EDM). A variant of :class:`~physicsnemo.metrics.diffusion.loss.ResidualLoss`
    that uses a Student-t distribution for the noise. The loss function uses a
    pre-trained regression model to compute residuals and computes the denoising
    score matching loss on the latent state formed by the residuals.

    Parameters
    ----------
    regression_net : torch.nn.Module
        The regression network used for computing residuals.
    P_mean : float
        Mean value for noise level computation.
    P_std : float
        Standard deviation for noise level computation.
    sigma_data : float
        Standard deviation for data weighting.
    hr_mean_conditioning : bool
        Flag indicating whether to use high-resolution mean for conditioning.
    nu : int, optional, default=10
        Number of degrees of freedom used for the Student-t distribution.
        Must be strictly greater than 2.
    """

    def __init__(
        self,
        regression_net: torch.nn.Module,
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hr_mean_conditioning: bool = False,
        nu: int = 10,
    ):
        if nu <= 2:
            raise ValueError(f"Expected nu > 2, but got {nu}.")
        super().__init__(
            regression_net,
            P_mean=P_mean,
            P_std=P_std,
            sigma_data=sigma_data,
            hr_mean_conditioning=hr_mean_conditioning,
        )
        self.nu = nu
        self.chi_dist = torch.distributions.Chi2(self.nu)

    def get_noise_params(self, y: Tensor) -> Tensor:
        """
        Compute the noise parameters to apply denoising score matching.

        Parameters
        ----------
        y : torch.Tensor
            Latent state of shape :math:`(B, *)`. Only used to determine the shape of
            the noise and create tensors on the same device.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Noise ``n`` of shape :math:`(B, *)` to be added to the latent state.
            - Noise level ``sigma`` of shape :math:`(B, 1, 1, 1)`.
            - Weight ``weight`` of shape :math:`(B, 1, 1, 1)` to multiply the loss.
        """
        # Sample noise level
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=y.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # NOTE: Scale sigma with the a scaling factor to account for nu
        sigma_scaled = sigma * np.sqrt(self.nu / (self.nu - 2))
        weight = (sigma_scaled**2 + self.sigma_data**2) / (sigma_scaled * self.sigma_data) ** 2
        # Sample Student-t noise
        kappa = self.chi_dist.sample((y.shape[0],)).to(y.device) / self.nu
        kappa = kappa.view(y.shape[0], 1, 1, 1)
        n = (torch.randn_like(y) / torch.sqrt(kappa)) * sigma
        return n, sigma, weight
