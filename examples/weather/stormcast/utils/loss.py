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


"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

from typing import Callable

import numpy as np
import torch
from torch.distributed.tensor.placement_types import Replicate


from physicsnemo.domain_parallel.shard_tensor import scatter_tensor


class EDMLoss:
    """
    Loss function proposed in the EDM paper.

    Parameters
    ----------
    P_mean: float, optional
        Mean value for `sigma` computation, by default -1.2.
    P_std: float, optional:
        Standard deviation for `sigma` computation, by default 1.2.
    sigma_data: float | torch.Tensor, optional
        Standard deviation for data, by default 0.5. Can also be a tensor; to use
        per-channel sigma_data, pass a tensor of shape (1, number_of_channels, 1, 1).

    Note
    ----
    Reference: Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    """

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float | torch.Tensor = 0.5,
        sigma_source_rank: int | None = None,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_source_rank = sigma_source_rank

    def get_noise_level(self, y: torch.Tensor) -> torch.Tensor:
        """Sample the sigma noise parameter for each sample."""
        shape = (y.shape[0], 1, 1, 1)
        rnd_normal = torch.randn(shape, device=y.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        return sigma

    def get_loss_weight(self, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute loss weight for each sample."""
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        return weight

    def sample_noise(self, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Sample the noise."""
        return torch.randn_like(y) * sigma

    def replicate_in_mesh(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if (hasattr(y, "_local_tensor") or hasattr(y, "placements")) and hasattr(
            y, "device_mesh"
        ):
            # y is sharded - need to replicate sigma to be compatible with DTensor operations
            # Get the source rank for replication
            # Replicate sigma across the domain mesh
            # This ensures all spatial shards see the same sigma values
            x = scatter_tensor(
                x,
                self.sigma_source_rank,
                y.device_mesh,
                placements=(Replicate(),),
                global_shape=x.shape,
                dtype=x.dtype,
            )
        return x

    def __call__(
        self,
        net: torch.nn.Module,
        images: torch.Tensor,
        condition: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        augment_pipe: Callable | None = None,
        lead_time_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calculate and return the loss corresponding to the EDM formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the input images.
        The noise level is determined by 'sigma', which is drawn from the `get_noise_level`
        function. The calculated loss is weighted as a function of 'sigma' and 'sigma_data'.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        images: torch.Tensor
            Input images to the neural network.

        condition: torch.Tensor
            Condition to be passed to the `condition` argument of `net.forward`.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        lead_time_label: torch.Tensor, optional
            Lead-time labels to pass to the model, shape ``(batch_size,)``.
            If not provided, the model is called without a lead-time label input.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        sigma = self.get_noise_level(y)
        sigma = self.replicate_in_mesh(sigma, y)
        weight = self.get_loss_weight(y, sigma)

        n = self.sample_noise(y, sigma)

        optional_args = {
            "augment_labels": augment_labels,
            "lead_time_label": lead_time_label,
            "class_labels": labels,
        }
        # drop None items to support models that don't have these arguments in `forward`
        optional_args = {k: v for (k, v) in optional_args.items() if v is not None}
        if condition is not None:
            D_yn = net(
                y + n,
                sigma.flatten(),
                condition=condition,
                # class_labels=labels,
                **optional_args,
            )
        else:
            D_yn = net(y + n, sigma, labels, **optional_args)
        loss = weight * ((D_yn - y) ** 2)
        return loss


class EDMLossLogUniform(EDMLoss):
    """
    EDM Loss with log-uniform sampling for `sigma`.

    Parameters
    ----------
    sigma_min: float, optional
        Minimum value for `sigma` computation, by default 0.02.
    sigma_max: float, optional:
        Minimum value for `sigma` computation, by default 1000.
    sigma_data: float | torch.Tensor, optional
        Standard deviation for data, by default 0.5. Can also be a tensor; to use
        per-channel sigma_data, pass a tensor of shape (1, number_of_channels, 1, 1).
    """

    def __init__(
        self,
        sigma_min: float = 0.02,
        sigma_max: float = 1000,
        sigma_data: float | torch.Tensor = 0.5,
        sigma_source_rank: int | None = None,
    ):
        self.sigma_data = sigma_data
        self.sigma_source_rank = sigma_source_rank
        self.log_sigma_min = float(np.log(sigma_min))
        self.log_sigma_diff = float(np.log(sigma_max)) - self.log_sigma_min

    def get_noise_level(self, y: torch.Tensor) -> torch.Tensor:
        """Sample the sigma noise parameter for each sample."""
        shape = (y.shape[0], 1, 1, 1)
        rnd_uniform = torch.rand(shape, device=y.device)
        sigma = (self.log_sigma_min + rnd_uniform * self.log_sigma_diff).exp()
        return sigma
