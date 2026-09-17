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

"""EDM noise scheduler."""

import torch
from jaxtyping import Float
from torch import Tensor

from .linear_gaussian import LinearGaussianNoiseScheduler


class EDMNoiseScheduler(LinearGaussianNoiseScheduler):
    r"""
    EDM noise scheduler.

    The EDM formulation uses :math:`\alpha(t) = 1` (no signal attenuation)
    and :math:`\sigma(t) = t` (identity mapping between time and noise level).

    **Sampling time-steps** are computed with polynomial spacing:

    .. math::
        t_i = \left(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}
        \left(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}\right)
        \right)^{\rho}

    **Training times** are sampled from a log-normal distribution with
    mean :math:`P_{\text{mean}}` and standard deviation :math:`P_{\text{std}}`.

    Parameters
    ----------
    sigma_min : float, optional
        Minimum noise level for sampling time-steps, by default 0.002.
    sigma_max : float, optional
        Maximum noise level for sampling time-steps, by default 80.
    rho : float, optional
        Exponent controlling time-step spacing. Larger values concentrate more
        steps at lower noise levels (better for fine details). By default 7.
    sigma_data : float or Tensor, optional
        Expected standard deviation of the training data, by default 0.5.
        Used by :meth:`loss_weight` to compute the per-sample loss weight.
        When a scalar ``float`` is given, it is stored as a 0-D tensor and
        the same value is applied to all channels.  When a 1-D ``Tensor``
        of shape :math:`(C,)` is given, each channel receives its own
        weight and :meth:`loss_weight` returns shape :math:`(N, C)` instead
        of :math:`(N,)`.
    P_mean : float, optional
        Mean of the log-normal distribution used to sample training times,
        by default -1.2.
    P_std : float, optional
        Standard deviation of the log-normal distribution used to sample
        training times, by default 1.2.

    Note
    ----
    Reference: `Elucidating the Design Space of Diffusion-Based
    Generative Models <https://arxiv.org/abs/2206.00364>`_

    Examples
    --------
    Basic training and sampling workflow using the EDM noise scheduler:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>>
    >>> scheduler = EDMNoiseScheduler(sigma_min=0.002, sigma_max=80.0, rho=7)
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

    Per-channel ``sigma_data`` for heterogeneous channels:

    >>> sigma_per_ch = torch.tensor([0.3, 0.5, 0.7])
    >>> scheduler_ch = EDMNoiseScheduler(sigma_data=sigma_per_ch)
    >>> t = scheduler_ch.sample_time(4)
    >>> w = scheduler_ch.loss_weight(t)
    >>> w.shape
    torch.Size([4, 3])
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float | Float[Tensor, " C"] = 0.5,
        P_mean: float = -1.2,
        P_std: float = 1.2,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data: Tensor = (
            sigma_data
            if isinstance(sigma_data, Tensor)
            else torch.as_tensor(sigma_data, dtype=torch.float32)
        )
        self._per_channel: bool = self.sigma_data.ndim > 0
        self.P_mean = P_mean
        self.P_std = P_std

    def sigma(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Identity mapping: :math:`\sigma(t) = t`."""
        return t

    def sigma_inv(
        self,
        sigma: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Identity mapping: :math:`t = \sigma`."""
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
        Generate EDM time-steps with polynomial spacing.

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
            Time-steps tensor of shape :math:`(N + 1,)` where :math:`N` is
            ``num_steps``.
        """
        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        smax_inv_rho = self.sigma_max ** (1 / self.rho)
        smin_inv_rho = self.sigma_min ** (1 / self.rho)
        frac = step_indices / (num_steps - 1)
        interp = smax_inv_rho + frac * (smin_inv_rho - smax_inv_rho)
        t_steps = interp**self.rho
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
        Sample N diffusion times from a log-normal distribution:
        :math:`\ln(\sigma) \sim \mathcal{N}(P_{\text{mean}},
        P_{\text{std}}^2)`.

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
        rnd_normal = torch.randn(N, device=device, dtype=dtype)
        return (rnd_normal * self.P_std + self.P_mean).exp()

    def loss_weight(
        self,
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"] | Float[Tensor, " N C"]:
        r"""
        Compute EDM loss weight.

        .. math::
            w(t) = \frac{\sigma(t)^2 + \sigma_{\text{data}}^2}
            {\left(\sigma(t) \cdot \sigma_{\text{data}}\right)^2}

        .. important::

            This loss weight is designed for training an x0-predictor
            (clean data predictor) wrapped with
            :class:`~physicsnemo.diffusion.preconditioners.EDMPreconditioner`.
            It is not suitable for training a score-predictor, or a model
            without a pre-conditioner.

        Parameters
        ----------
        t : Tensor
            Diffusion time values of shape :math:`(N,)`.

        Returns
        -------
        Tensor
            Loss weight of shape :math:`(N,)` when ``sigma_data`` is a
            scalar, or :math:`(N, C)` when ``sigma_data`` is per-channel.
        """
        sigma = self.sigma(t)
        sd = self.sigma_data.to(device=sigma.device, dtype=sigma.dtype)
        if self._per_channel:
            # Per-channel: sigma (N,) → (N, 1);  sd (C,) → (1, C)
            sigma = sigma.unsqueeze(-1)
            sd = sd.unsqueeze(0)
        return (sigma**2 + sd**2) / (sigma * sd) ** 2
