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


class Diffusion:
    """
    Diffusion model for TopoDiff.
    """

    def __init__(self, n_steps=1000, min_beta=10**-4, max_beta=0.02, device="cpu"):

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

    def q_sample(self, x0, t, noise=None):

        """
        Diffuse the input data.
        """

        if noise is None:
            noise = torch.rand_like(x0).to(self.device)

        alpha_bars = self.alpha_bars[t]

        x = (
            alpha_bars.sqrt()[:, None, None, None] * x0
            + (1 - alpha_bars).sqrt()[:, None, None, None] * noise
        )

        return x

    def p_sample(self, model, xt, t, cons):

        """
        Sample from the posterior distribution.
        """

        return model(xt, cons, t)

    def train_loss(self, model, x0, cons):

        """
        Compute the training loss.
        """

        b, c, w, h = x0.shape
        noise = torch.randn_like(x0).to(self.device)

        t = torch.randint(0, self.n_steps, (b,)).to(self.device)

        xt = self.q_sample(x0, t, noise)

        pred_noise = self.p_sample(model, xt, t, cons)

        return self.loss(pred_noise, noise)
