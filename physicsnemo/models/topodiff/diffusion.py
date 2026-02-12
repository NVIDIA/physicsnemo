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
import torch.nn as nn
import torch.nn.functional as F


class Diffusion:
    r"""Diffusion scheduler for TopoDiff."""

    def __init__(
        self,
        n_steps: int = 1000,
        min_beta: float = 10**-4,
        max_beta: float = 0.02,
        device: str = "cpu",
    ):
        r"""Initialize the diffusion schedule.

        Parameters
        ----------
        n_steps : int, optional, default=1000
            Number of diffusion steps.
        min_beta : float, optional, default=1e-4
            Minimum beta in the linear schedule.
        max_beta : float, optional, default=0.02
            Maximum beta in the linear schedule.
        device : str, optional, default="cpu"
            Target device string for tensors.
        """
        self.n_steps = n_steps
        self.device = device

        self.betas = torch.linspace(min_beta, max_beta, self.n_steps).to(device)

        self.alphas = 1 - self.betas

        self.alpha_bars = torch.cumprod(self.alphas, 0).to(device)

        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], [1, 0], "constant", 0)

        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )

        self.loss = nn.MSELoss()

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Diffuse the input data (forward process).

        Parameters
        ----------
        x0 : torch.Tensor
            Clean samples :math:`(N, C, H, W)`.
        t : torch.Tensor
            Timestep indices :math:`(N,)`.
        noise : torch.Tensor, optional
            Optional noise tensor; if ``None`` sampled from standard normal.

        Returns
        -------
        torch.Tensor
            Noised samples :math:`x_t`.
        """

        if noise is None:
            noise = torch.rand_like(x0).to(self.device)

        alpha_bars = self.alpha_bars[t]

        x = (
            alpha_bars.sqrt()[:, None, None, None] * x0
            + (1 - alpha_bars).sqrt()[:, None, None, None] * noise
        )

        return x

    def p_sample(
        self, model, xt: torch.Tensor, t: torch.Tensor, cons: torch.Tensor
    ) -> torch.Tensor:
        r"""Predict noise using the model (reverse process).

        Parameters
        ----------
        model : torch.nn.Module
            Denoiser that predicts noise given ``(x_t, cons, t)``.
        xt : torch.Tensor
            Noised samples :math:`x_t`.
        t : torch.Tensor
            Timestep indices :math:`(N,)`.
        cons : torch.Tensor
            Constraint tensor concatenated in the model.

        Returns
        -------
        torch.Tensor
            Predicted noise tensor.
        """

        return model(xt, cons, t)

    def train_loss(self, model, x0: torch.Tensor, cons: torch.Tensor) -> torch.Tensor:
        r"""Compute training loss for diffusion denoiser.

        Parameters
        ----------
        model : torch.nn.Module
            Denoiser model.
        x0 : torch.Tensor
            Clean inputs :math:`(N, C, H, W)`.
        cons : torch.Tensor
            Constraint tensor :math:`(N, C_{cons}, H, W)`.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """

        b, c, w, h = x0.shape
        noise = torch.randn_like(x0).to(self.device)

        t = torch.randint(0, self.n_steps, (b,)).to(self.device)

        xt = self.q_sample(x0, t, noise)

        pred_noise = self.p_sample(model, xt, t, cons)

        return self.loss(pred_noise, noise)
