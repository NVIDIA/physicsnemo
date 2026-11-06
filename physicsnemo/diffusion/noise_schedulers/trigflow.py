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

r"""Trigonometric interpolation between data and Gaussian noise."""

import math

import torch
from jaxtyping import Float
from torch import Tensor

from .linear_gaussian import LinearGaussianNoiseScheduler


class TrigFlowNoiseScheduler(LinearGaussianNoiseScheduler):
    r"""Interpolate between clean data and Gaussian noise along a trigonometric path.

    Use this scheduler for TrigFlow denoising or consistency training, paired
    with ``TrigFlowPreconditioner`` and the same ``sigma_data``. Diffusion time
    is an angle in radians, whereas EDM uses the noise standard deviation.
    Training draws diffusion times uniformly. Inference uses evenly spaced
    diffusion times ending at zero.

    .. math::

        x_t = \cos(t)x_0 + \sigma_d \sin(t)\epsilon,
        \qquad \epsilon \sim \mathcal{N}(0,I).

    Parameters
    ----------
    sigma_data : float, optional, default=0.5
        Data scale and noise standard deviation at :math:`t=\pi/2`.
        Must be finite and positive.
    t_min : float, optional, default=0.001
        Lower training diffusion time in radians, with
        :math:`0 \leq t_{\min} < t_{\max}`. Use a positive value for
        denoising loss weights, which diverge at zero.
    t_max : float, optional, default=1.55
        Upper training diffusion time and first inference diffusion time,
        in radians. Must be below ``pi / 2``. The default stays below that
        endpoint in reduced precision.

    Examples
    --------
    Use this scheduler when the model treats diffusion time as an angle and predicts
    TrigFlow velocity. The example checks noisy-state shape, reverse-time grid size, and
    the inverse signal-to-noise mapping.

    >>> scheduler = TrigFlowNoiseScheduler()
    >>> t = scheduler.sample_time(2)
    >>> scheduler.add_noise(torch.zeros(2, 3), t).shape
    torch.Size([2, 3])
    >>> scheduler.timesteps(4).shape
    torch.Size([5])
    >>> # Use interior angles for the inverse SNR check.
    >>> t = torch.tensor([0.2, 0.7])
    >>> torch.allclose(scheduler.snr_inv(scheduler.snr(t)), t)
    True

    Note
    ----
    Coefficient functions accept angular times without clipping. The training
    bounds do not limit coefficient evaluation. At zero, score conversions
    and denoising weights are singular. At ``pi / 2``, conversions that divide
    by the signal coefficient are singular.
    """

    def __init__(
        self, sigma_data: float = 0.5, t_min: float = 0.001, t_max: float = 1.55
    ) -> None:
        if not math.isfinite(sigma_data) or sigma_data <= 0:
            raise ValueError(f"Expected finite sigma_data > 0, got {sigma_data}.")
        if not 0 <= t_min < t_max < math.pi / 2:
            raise ValueError(
                f"Expected 0 <= t_min < t_max < pi/2, got {t_min=}, {t_max=}."
            )
        self.sigma_data = sigma_data
        self.t_min = t_min
        self.t_max = t_max

    def alpha(self, t: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Signal coefficient: :math:`\alpha(t)=\cos(t)`."""
        return t.cos()

    def alpha_dot(self, t: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Signal derivative: :math:`\dot\alpha(t)=-\sin(t)`."""
        return -t.sin()

    def sigma(self, t: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Noise coefficient: :math:`\sigma(t)=\sigma_d\sin(t)`."""
        return self.sigma_data * t.sin()

    def sigma_dot(self, t: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Noise derivative: :math:`\dot\sigma(t)=\sigma_d\cos(t)`."""
        return self.sigma_data * t.cos()

    def sigma_inv(self, sigma: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Find diffusion time :math:`\arcsin(\sigma/\sigma_d)` for noise scales in ``[0, sigma_data]``."""
        return (sigma / self.sigma_data).asin()

    def snr_inv(self, q: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Find diffusion time :math:`\arctan(1/(\sigma_d q))` for nonnegative amplitude ratios."""
        return torch.atan2(torch.ones_like(q), self.sigma_data * q)

    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""Draw training diffusion times uniformly between ``t_min`` and ``t_max``.

        Parameters
        ----------
        N : int
            Number of training examples.
        device : torch.device, optional
            Device for the sampled times.
        dtype : torch.dtype, optional
            Floating-point type for the sampled times.

        Returns
        -------
        Tensor
            Training diffusion times in radians with shape :math:`(N,)`.
        """
        return self.t_min + (self.t_max - self.t_min) * torch.rand(
            N, device=device, dtype=dtype
        )

    def timesteps(
        self,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N+1"]:
        r"""Return evenly spaced inference diffusion times from ``t_max`` to zero.

        Parameters
        ----------
        num_steps : int
            Number of solver steps, at least one.
        device : torch.device, optional
            Device for the inference grid.
        dtype : torch.dtype, optional
            Floating-point type for the grid.

        Returns
        -------
        Tensor
            Decreasing diffusion times in radians with shape
            :math:`(\mathrm{num\_steps}+1,)`.
        """
        if num_steps < 1:
            raise ValueError(f"Expected num_steps >= 1, got {num_steps}.")
        return torch.linspace(self.t_max, 0, num_steps + 1, device=device, dtype=dtype)

    def loss_weight(self, t: Float[Tensor, " N"]) -> Float[Tensor, " N"]:
        r"""Weight denoising residuals by :math:`1/(\sigma_d\sin(t))^2`.

        Use with ``TrigFlowPreconditioner`` to give unit weight to the
        normalized-velocity residual. Use positive diffusion times to avoid
        the singularity at zero.

        Parameters
        ----------
        t : Tensor
            Training diffusion times in radians with shape :math:`(N,)`.

        Returns
        -------
        Tensor
            Per-example denoising weights.
        """
        return self.sigma(t).square().reciprocal()
