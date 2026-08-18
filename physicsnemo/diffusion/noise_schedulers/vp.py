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

"""Variance Preserving (VP) noise scheduler."""

import torch
from jaxtyping import Float
from torch import Tensor

from .linear_gaussian import LinearGaussianNoiseScheduler


class VPNoiseScheduler(LinearGaussianNoiseScheduler):
    r"""
    Variance Preserving (VP) noise scheduler.

    Implements the VP formulation where the total variance is preserved:
    :math:`\alpha(t)^2 + \sigma(t)^2 = 1`. This is based on a linear beta
    schedule: :math:`\beta(t) = \beta_{\min} + t \cdot \beta_d`.

    The noise and signal coefficients are:

    .. math::
        \alpha(t) = \exp\left(-\frac{1}{2}
        \left(\frac{\beta_d}{2} t^2 + \beta_{\min} t\right)\right)

    .. math::
        \sigma(t) = \sqrt{1 - \alpha(t)^2}
        = \sqrt{1 - \exp\left(-\frac{\beta_d}{2} t^2
        - \beta_{\min} t\right)}

    **Sampling time-steps** are linearly spaced from ``t_max`` (usually 1) to
    ``epsilon_s`` (small positive value to avoid singularities).

    **Training times** are sampled uniformly between ``epsilon_s`` and
    ``t_max``.

    Parameters
    ----------
    beta_min : float, optional
        Minimum beta value for the linear schedule, by default 0.1.
    beta_d : float, optional
        Beta slope (delta) for the linear schedule, by default 19.1.
    epsilon_s : float, optional
        Small positive value for minimum time, by default 1e-3.
    t_max : float, optional
        Maximum diffusion time, by default 1.0.

    Note
    ----
    Reference: `Score-Based Generative Modeling through Stochastic
    Differential Equations <https://arxiv.org/abs/2011.13456>`_

    Examples
    --------
    Basic training and sampling workflow using the VP noise scheduler:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import VPNoiseScheduler
    >>>
    >>> scheduler = VPNoiseScheduler(beta_min=0.1, beta_d=19.1)
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
    >>> x0_predictor = lambda x, t: x * 0.9  # Toy x0-predictor
    >>> denoiser = scheduler.get_denoiser(x0_predictor=x0_predictor)
    >>> denoiser(xN, tN).shape  # ODE RHS for sampling
    torch.Size([4, 3, 8, 8])
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_d: float = 19.1,
        epsilon_s: float = 1e-3,
        t_max: float = 1.0,
    ) -> None:
        self.beta_min = beta_min
        self.beta_d = beta_d
        self.epsilon_s = epsilon_s
        self.t_max = t_max

    def alpha(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Signal coefficient: :math:`\alpha(t) = \exp(-a(t)/2)`."""
        return torch.exp(-0.5 * (0.5 * self.beta_d * t**2 + self.beta_min * t))

    def alpha_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Derivative: :math:`\dot{\alpha}(t) = -\frac{\beta(t)}{2} \alpha(t)`."""
        beta_t = self.beta_min + self.beta_d * t
        return -0.5 * beta_t * self.alpha(t)

    def sigma(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Noise level: :math:`\sigma(t) = \sqrt{1 - \alpha(t)^2}`."""
        alpha_sq = self.alpha(t) ** 2
        return torch.sqrt(1 - alpha_sq)

    def sigma_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Derivative: :math:`\dot{\sigma}(t) = -\alpha(t) \dot{\alpha}(t) / \sigma(t)`."""  # noqa: E501
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        alpha_dot_t = self.alpha_dot(t)
        # d/dt sqrt(1 - alpha^2) = -alpha * alpha_dot / sqrt(1 - alpha^2)
        return -alpha_t * alpha_dot_t / sigma_t

    def sigma_inv(
        self,
        sigma: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""
        Inverse mapping from sigma to time.

        Solves: :math:`\sigma^2 = 1 - \exp(-a(t))` for :math:`t`.
        """
        # sigma^2 = 1 - exp(-a) => a = -log(1 - sigma^2)
        # a = beta_d/2 * t^2 + beta_min * t
        # Quadratic: beta_d * t^2 + 2*beta_min * t + 2*log(1-sigma^2) = 0
        log_term = torch.log(1 - sigma**2 + 1e-8)  # small eps for stability
        discriminant = self.beta_min**2 - 2 * self.beta_d * log_term
        return (-self.beta_min + torch.sqrt(discriminant.clamp(min=0))) / self.beta_d

    def timesteps(
        self,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N+1"]:
        r"""
        Generate VP time-steps with linear spacing.

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
        # Linear spacing from t_max to epsilon_s
        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        frac = step_indices / (num_steps - 1)
        t_steps = self.t_max + frac * (self.epsilon_s - self.t_max)
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
        Sample N diffusion times uniformly in :math:`[\epsilon_s, t_{max}]`.

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
        return self.epsilon_s + u * (self.t_max - self.epsilon_s)

    def loss_weight(
        self,
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"]:
        r"""
        Compute VP loss weight: :math:`w(t) = \alpha(t)^2 / \sigma(t)^2`.

        .. important::

            This loss weight is designed for training an x0-predictor
            (clean data predictor) wrapped with
            :class:`~physicsnemo.diffusion.preconditioners.VPPreconditioner`.
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
        return (self.alpha(t) / self.sigma(t)) ** 2
