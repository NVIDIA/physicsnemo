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

"""Variance Exploding (VE) noise scheduler."""

import math

import torch
from jaxtyping import Float
from torch import Tensor

from .linear_gaussian import LinearGaussianNoiseScheduler


class VENoiseScheduler(LinearGaussianNoiseScheduler):
    r"""
    Variance Exploding (VE) noise scheduler.

    Implements the VE formulation with :math:`\sigma(t) = \sqrt{t}` and
    :math:`\alpha(t) = 1` (no signal attenuation).

    **Sampling time-steps** use geometric spacing in :math:`\sigma^2` space:

    .. math::
        \sigma_i^2 = \sigma_{\max}^2 \cdot
        \left(\frac{\sigma_{\min}^2}{\sigma_{\max}^2}\right)^{i/(N-1)}

    **Training times** are sampled log-uniformly between ``sigma_min`` and
    ``sigma_max``, then mapped to time via :math:`t = \sigma^2`.

    Parameters
    ----------
    sigma_min : float, optional
        Minimum noise level, by default 0.02.
    sigma_max : float, optional
        Maximum noise level, by default 100.

    Note
    ----
    Reference: `Score-Based Generative Modeling through Stochastic
    Differential Equations <https://arxiv.org/abs/2011.13456>`_

    Examples
    --------
    Basic training and sampling workflow using the VE noise scheduler:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import VENoiseScheduler
    >>>
    >>> scheduler = VENoiseScheduler(sigma_min=0.02, sigma_max=100.0)
    >>>
    >>> # Training: sample times and add noise
    >>> x0 = torch.randn(4, 3, 8, 8)  # Clean data
    >>> t = scheduler.sample_time(4)    # Sample diffusion times
    >>> x_t = scheduler.add_noise(x0, t)  # Create noisy samples
    >>> x_t.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Sampling: generate timesteps and initial latents
    >>> t_steps = scheduler.timesteps(10)
    >>> tN = t_steps[0].expand(4)  # Initial time for batch of 4
    >>> xN = scheduler.init_latents((3, 8, 8), tN)  # Initial noise
    >>> xN.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Convert x0-predictor to denoiser for sampling
    >>> x0_predictor = lambda x, t: x / (1 + t.view(-1, 1, 1, 1)**2)  # Toy x0-predictor
    >>> denoiser = scheduler.get_denoiser(x0_predictor=x0_predictor)
    >>> denoiser(xN, tN).shape  # ODE RHS for sampling
    torch.Size([4, 3, 8, 8])
    """

    def __init__(
        self,
        sigma_min: float = 0.02,
        sigma_max: float = 100.0,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""VE noise coefficient: :math:`\sigma(t) = \sqrt{t}`."""
        return t.sqrt()

    def sigma_inv(
        self,
        sigma: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Inverse VE mapping: :math:`t = \sigma^2`."""
        return sigma**2

    def sigma_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Time derivative: :math:`\dot{\sigma}(t) = 1/(2\sqrt{t})`."""
        return 0.5 / t.sqrt()

    def alpha(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Constant signal coefficient: :math:`\alpha(t) = 1`."""
        return torch.ones_like(t)

    def alpha_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Zero derivative: :math:`\dot{\alpha}(t) = 0`."""
        return torch.zeros_like(t)

    def timesteps(
        self,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N+1"]:
        r"""
        Generate VE time-steps with geometric spacing in :math:`\sigma^2`.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        torch.Tensor
            Time-steps tensor of shape :math:`(N + 1,)`.
        """
        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        ratio = self.sigma_min**2 / self.sigma_max**2
        exponent = step_indices / (num_steps - 1)
        t_steps = (self.sigma_max**2) * (ratio**exponent)
        zero = torch.zeros(1, dtype=dtype, device=device)
        return torch.cat([t_steps, zero])

    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""
        Sample N diffusion times log-uniformly in sigma space, mapped to time.

        Parameters
        ----------
        N : int
            Number of time values to sample.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Sampled diffusion times of shape :math:`(N,)`.
        """
        u = torch.rand(N, device=device, dtype=dtype)
        log_ratio = math.log(self.sigma_max / self.sigma_min)
        sigma = self.sigma_min * torch.exp(u * log_ratio)
        return self.sigma_inv(sigma)

    def loss_weight(
        self,
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"]:
        r"""
        Compute VE loss weight: :math:`w(t) = 1 / \sigma(t)^2`.

        .. important::

            This loss weight is designed for training an x0-predictor
            (clean data predictor) wrapped with
            :class:`~physicsnemo.diffusion.preconditioners.VEPreconditioner`.
            It is not suitable for training a score-predictor, or a model
            without a pre-conditioner.

        Parameters
        ----------
        t : Tensor
            Diffusion time values of shape :math:`(N,)`.

        Returns
        -------
        Tensor
            Loss weight of shape :math:`(N,)`.
        """
        return 1 / self.sigma(t) ** 2
