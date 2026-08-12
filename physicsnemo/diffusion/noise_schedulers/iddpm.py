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

"""Improved DDPM (iDDPM) noise scheduler."""

import math

import torch
from jaxtyping import Float
from torch import Tensor

from .linear_gaussian import LinearGaussianNoiseScheduler


class IDDPMNoiseScheduler(LinearGaussianNoiseScheduler):
    r"""
    Improved DDPM (iDDPM) noise scheduler.

    Uses identity mappings :math:`\sigma(t) = t` and :math:`\alpha(t) = 1`.
    The key feature is a precomputed noise level schedule derived from a
    cosine schedule, providing improved sample quality in comparison to
    original DDPM.

    **Sampling time-steps** are selected from a precomputed schedule of
    :math:`M` discrete noise levels, subsampled to ``num_steps``.

    **Training times** are sampled uniformly from the precomputed schedule.

    Parameters
    ----------
    sigma_min : float, optional
        Minimum noise level for filtering, by default 0.002.
    sigma_max : float, optional
        Maximum noise level for filtering, by default 81.
    C_1 : float, optional
        Clipping threshold for alpha ratio, by default 0.001.
    C_2 : float, optional
        Cosine schedule parameter, by default 0.008.
    M : int, optional
        Number of precomputed discretization steps, by default 1000.

    Note
    ----
    Reference: `Improved Denoising Diffusion Probabilistic Models
    <https://arxiv.org/abs/2102.09672>`_

    Examples
    --------
    Basic training and sampling workflow using the iDDPM noise scheduler:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import IDDPMNoiseScheduler
    >>>
    >>> scheduler = IDDPMNoiseScheduler(C_1=0.001, C_2=0.008, M=1000)
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
        sigma_min: float = 0.002,
        sigma_max: float = 81.0,
        C_1: float = 0.001,
        C_2: float = 0.008,
        M: int = 1000,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M

        # Precompute the noise level schedule u_j, j = 0, ..., M
        self._u = self._compute_u_schedule()

    def _compute_u_schedule(self) -> Tensor:
        """Precompute the iDDPM noise level schedule."""
        u = torch.zeros(self.M + 1)
        for j in range(self.M, 0, -1):
            angle_j = 0.5 * math.pi * j / self.M / (self.C_2 + 1)
            angle_jm1 = 0.5 * math.pi * (j - 1) / self.M / (self.C_2 + 1)
            alpha_bar_j = math.sin(angle_j) ** 2
            alpha_bar_jm1 = math.sin(angle_jm1) ** 2
            alpha_ratio = alpha_bar_jm1 / alpha_bar_j
            val = (u[j] ** 2 + 1) / max(alpha_ratio, self.C_1) - 1
            u[j - 1] = val.sqrt()
        return u

    def sigma(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""For iDDPM, :math:`\sigma(t) = t` (identity mapping)."""
        return t

    def sigma_inv(
        self,
        sigma: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""For iDDPM, :math:`t = \sigma` (identity mapping)."""
        return sigma

    def sigma_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Constant derivative: :math:`\dot{\sigma}(t) = 1`."""
        return torch.ones_like(t)

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
        Generate iDDPM time-steps from precomputed schedule.

        Subsamples ``num_steps`` values from the precomputed schedule of
        :math:`M` noise levels.

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
        u = self._u.to(device=device, dtype=dtype)
        # Filter to valid sigma range
        in_range = torch.logical_and(u >= self.sigma_min, u <= self.sigma_max)
        u_filtered = u[in_range]

        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        scale = (len(u_filtered) - 1) / (num_steps - 1)
        indices = (scale * step_indices).round().to(torch.int64)
        sigma_steps = u_filtered[indices]

        zero = torch.zeros(1, dtype=dtype, device=device)
        return torch.cat([sigma_steps, zero])

    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""
        Sample N diffusion times uniformly from precomputed schedule.

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
        u = self._u.to(device=device, dtype=dtype)
        in_range = torch.logical_and(u >= self.sigma_min, u <= self.sigma_max)
        u_filtered = u[in_range]
        # Sample random indices
        indices = torch.randint(0, len(u_filtered), (N,), device=device)
        return u_filtered[indices]

    def loss_weight(
        self,
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"]:
        r"""
        Compute iDDPM loss weight: :math:`w(t) = 1 / \sigma(t)^2`.

        .. important::

            This loss weight is designed for training an x0-predictor
            (clean data predictor) wrapped with
            :class:`~physicsnemo.diffusion.preconditioners.IDDPMPreconditioner`.
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
