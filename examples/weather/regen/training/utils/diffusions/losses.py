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
"""Code from https://github.com/NVlabs/edm/tree/main

"""
import torch
from utils.diffusions import networks


class EDMLoss:
    """
    Improved loss function proposed in the paper "Elucidating the Design Space
    of Diffusion-Based Generative Models" (EDM).
    """

    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        print("using P_mean value of {}".format(P_mean))

    def __call__(
        self,
        net: networks.EDMPrecond,
        x,
        condition=None,
        class_labels=None,
        augment_pipe=None,
    ):
        """
        Args:
            net:
            x: The latent data (to be denoised). shape [batch_size, target_channels, w, h]
            class_labels: optional, shape [batch_size, label_dim]
            condition: optional, the conditional inputs,
                shape=[batch_size, condition_channel, w, h]
        Returns:
            out: loss function with shape [batch_size, target_channels, w, h]
                This should be averaged to get the mean loss for gradient descent.
        """
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(
            y + n,
            sigma,
            condition=condition,
            class_labels=class_labels,
            augment_labels=augment_labels,
        )
        loss = weight * ((D_yn - y) ** 2)
        return loss
