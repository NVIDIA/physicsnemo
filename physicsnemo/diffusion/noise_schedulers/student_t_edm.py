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

"""Student-t EDM noise scheduler."""

from typing import Tuple

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from .linear_gaussian import LinearGaussianNoiseScheduler


class StudentTEDMNoiseScheduler(LinearGaussianNoiseScheduler):
    r"""
    Student-t EDM noise scheduler for heavy-tailed diffusion models.

    This scheduler is a variant of :class:`EDMNoiseScheduler` that uses
    Student-t noise instead of Gaussian noise. It is useful for modeling
    heavy-tailed distributions and can improve sample quality for certain
    data types.

    .. important::

        Despite inheriting from :class:`LinearGaussianNoiseScheduler`, this
        scheduler is **not truly Gaussian**. It uses the same linear structure
        (identity mappings :math:`\sigma(t) = t` and :math:`\alpha(t) = 1`) but
        replaces Gaussian noise with Student-t noise. The "Linear" part of
        :class:`LinearGaussianNoiseScheduler` still applies, but the "Gaussian"
        part does not.

    This scheduler uses a non-gaussian forward process:

    .. math::
        \mathbf{x}(t) = \mathbf{x}_0 + \sigma(t) \mathbf{n}, \quad
        \mathbf{n} \sim \text{Student-}t(\nu)

    The marginal distribution :math:`p(\mathbf{x}_t | \mathbf{x}_0)` is
    therefore a scaled Student-t distribution, not Gaussian.

    **Comparison with EDMNoiseScheduler:**

    This scheduler shares the same time-to-noise mappings as
    :class:`EDMNoiseScheduler`.
    The only differences are in :meth:`add_noise` and :meth:`init_latents`,
    which use Student-t noise instead of Gaussian noise.

    Parameters
    ----------
    sigma_min : float, optional
        Minimum noise level for sampling time-steps, by default 0.002.
    sigma_max : float, optional
        Maximum noise level for sampling time-steps, by default 80.
    rho : float, optional
        Exponent controlling time-step spacing. Larger values concentrate more
        steps at lower noise levels (better for fine details). By default 7.
    nu : int, optional
        Degrees of freedom for Student-t distribution. Must be > 2.
        As ``nu`` increases, the distribution approaches Gaussian. Lower values
        produce heavier tails. By default 10.
    sigma_data : float or Tensor, optional
        Expected standard deviation of the training data, by default 0.5.
        Used by :meth:`loss_weight` to compute the per-sample loss weight.
        When a 1-D ``Tensor`` of shape :math:`(C,)` is given, each channel
        receives its own weight and :meth:`loss_weight` returns shape
        :math:`(N, C)` instead of :math:`(N,)`.
    P_mean : float, optional
        Mean of the log-normal distribution used to sample training times,
        by default -1.2.
    P_std : float, optional
        Standard deviation of the log-normal distribution used to sample
        training times, by default 1.2.

    Note
    ----
    Reference: `Heavy-Tailed Diffusion Models
    <https://arxiv.org/abs/2410.14171>`_

    Examples
    --------
    Basic training and sampling workflow with Student-t noise:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import (
    ...     StudentTEDMNoiseScheduler,
    ... )
    >>>
    >>> scheduler = StudentTEDMNoiseScheduler(nu=10)
    >>>
    >>> # Training: sample times and add Student-t noise
    >>> x0 = torch.randn(4, 3, 8, 8)  # Clean data
    >>> t = scheduler.sample_time(4)    # Sample diffusion times
    >>> x_t = scheduler.add_noise(x0, t)  # Adds Student-t noise
    >>> x_t.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Sampling: generate timesteps and Student-t initial latents
    >>> t_steps = scheduler.timesteps(10)
    >>> tN = t_steps[0].expand(4)
    >>> xN = scheduler.init_latents((3, 8, 8), tN)  # Student-t latents
    >>> xN.shape
    torch.Size([4, 3, 8, 8])
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        nu: int = 10,
        sigma_data: float | Float[Tensor, " C"] = 0.5,
        P_mean: float = -1.2,
        P_std: float = 1.2,
    ) -> None:
        if nu <= 2:
            raise ValueError(f"nu must be > 2, got {nu}")
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.nu = nu
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
        Compute Student-t EDM loss weight: :math:`w(t) = \frac{\tilde{\sigma}(t)^2 + \sigma_{\text{data}}^2}
        {\left(\tilde{\sigma}(t) \cdot \sigma_{\text{data}}\right)^2}`

        where :math:`\tilde{\sigma}(t) = \sigma(t) \cdot \sqrt{\frac{\nu}{\nu -
        2}}` is the scaled noise level.

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
        sigma = self.sigma(t) * np.sqrt(self.nu / (self.nu - 2))
        sd = self.sigma_data.to(device=sigma.device, dtype=sigma.dtype)
        if self._per_channel:
            # Per-channel: sigma (N,) → (N, 1);  sd (C,) → (1, C)
            sigma = sigma.unsqueeze(-1)
            sd = sd.unsqueeze(0)
        return (sigma**2 + sd**2) / (sigma * sd) ** 2

    def _sample_student_t(
        self,
        *shape: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        r"""
        Sample from standard Student-t distribution.

        Student-t samples are generated as: :math:`X / \sqrt{V / \nu}` where
        :math:`X \sim \mathcal{N}(0, 1)` and :math:`V \sim \chi^2(\nu)`.

        Parameters
        ----------
        *shape : int
            Shape of the output tensor.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Student-t samples of the specified shape.
        """
        normal = torch.randn(*shape, device=device, dtype=dtype)

        nu = torch.tensor(self.nu, device=device, dtype=dtype)
        chi2_dist = torch.distributions.Chi2(df=nu)
        chi2_samples = chi2_dist.sample((shape[0], *([1] * (len(shape) - 1))))
        kappa = chi2_samples / nu

        return normal / torch.sqrt(kappa)

    def add_noise(
        self,
        x0: Float[Tensor, " B *dims"],
        time: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Add Student-t noise to clean data at the given diffusion times.

        Unlike the Gaussian case in :class:`LinearGaussianNoiseScheduler`,
        this method uses Student-t noise:

        .. math::
            \mathbf{x}(t) = \mathbf{x}_0 + \sigma(t) \mathbf{n}, \quad
            \mathbf{n} \sim \text{Student-}t(\nu)

        Parameters
        ----------
        x0 : Tensor
            Clean latent state of shape :math:`(B, *)`.
        time : Tensor
            Diffusion time values of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Noisy latent state of shape :math:`(B, *)`.
        """
        expected_shape = (-1,) + (1,) * (x0.ndim - 1)
        t_bc = time.reshape(expected_shape)
        sigma_t_bc = self.sigma(t_bc)
        noise = self._sample_student_t(*x0.shape, device=x0.device, dtype=x0.dtype)
        return x0 + sigma_t_bc * noise

    def init_latents(
        self,
        spatial_shape: Tuple[int, ...],
        tN: Float[Tensor, " B"],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " B *spatial_shape"]:
        r"""
        Initialize noisy latent state with Student-t noise.

        Unlike the Gaussian case in :class:`LinearGaussianNoiseScheduler`,
        this method uses Student-t noise:

        .. math::
            \mathbf{x}_N = \sigma(t_N) \cdot \mathbf{n}, \quad
            \mathbf{n} \sim \text{Student-}t(\nu)

        Parameters
        ----------
        spatial_shape : Tuple[int, ...]
            Spatial shape of the latent state, e.g., ``(C, H, W)``.
        tN : Tensor
            Initial diffusion time of shape :math:`(B,)`.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Initial noisy latent of shape :math:`(B, *spatial\_shape)`.
        """
        B = tN.shape[0]
        noise = self._sample_student_t(B, *spatial_shape, device=device, dtype=dtype)
        expected_shape = (-1,) + (1,) * len(spatial_shape)
        tN_bc = tN.reshape(expected_shape)
        sigma_tN_bc = self.sigma(tN_bc)
        return sigma_tN_bc * noise
