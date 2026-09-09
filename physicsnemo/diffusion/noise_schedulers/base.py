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

"""Base protocol for diffusion noise schedulers."""

from typing import Any, Protocol, Tuple, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser


@runtime_checkable
class NoiseScheduler(Protocol):
    r"""
    Protocol defining the minimal interface for noise schedulers.

    A noise scheduler defines methods for training (adding noise, sampling
    diffusion time) and for sampling (generating diffusion time-steps,
    initializing latent state, obtaining a denoiser). This interface is generic
    and does not assume any specific form of noise schedule.

    Any object that implements this interface can be used with the diffusion
    training and sampling utilities.

    **Training methods:**

    - :meth:`sample_time`: Sample diffusion time values for training
    - :meth:`add_noise`: Add noise to clean data at given diffusion time
    - :meth:`loss_weight`: Compute per-sample loss weight for training

    **Sampling methods:**

    - :meth:`timesteps`: Generate discrete time-steps for sampling
    - :meth:`init_latents`: Initialize noisy latent state :math:`\mathbf{x}_N`
    - :meth:`get_denoiser`: Convert a predictor (e.g. model that predicts
         clean, data, score, etc.) to a sampling-compatible denoiser

    See Also
    --------
    :class:`LinearGaussianNoiseScheduler` : base abstract class for
        linear-Gaussian schedules. Implements the NoiseScheduler protocol.
    :func:`~physicsnemo.diffusion.samplers.sample` : sampling function for
        generating data samples from a diffusion model.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import NoiseScheduler
    >>>
    >>> class MyScheduler:
    ...     def sample_time(self, N, device=None, dtype=None):
    ...         return torch.rand(N, device=device, dtype=dtype)
    ...     def add_noise(self, x0, time):
    ...         return x0 + time.view(-1, 1) * torch.randn_like(x0)
    ...     def timesteps(self, num_steps, device=None, dtype=None):
    ...         return torch.linspace(1, 0, num_steps + 1, device=device)
    ...     def init_latents(self, spatial_shape, tN, device=None, dtype=None):
    ...         return torch.randn(tN.shape[0], *spatial_shape, device=device)
    ...     def get_denoiser(self, x0_predictor=None, score_predictor=None, **kwargs):
    ...         def denoiser(x, t):
    ...             if x0_predictor is not None:
    ...                 return (x - x0_predictor(x, t)) / (t.view(-1, 1))
    ...             elif score_predictor is not None:
    ...                 return -score_predictor(x, t) * t.view(-1, 1)
    ...         return denoiser
    ...     def loss_weight(self, t):
    ...         return 1 / t**2
    ...
    >>> scheduler = MyScheduler()
    >>> isinstance(scheduler, NoiseScheduler)
    True
    """

    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""
        Sample N diffusion time values for training.

        Used in training to sample random diffusion times, typically in the
        denoising score matching loss.

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
        ...

    def add_noise(
        self,
        x0: Float[Tensor, " B *dims"],
        time: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Add noise to clean data at the given diffusion times.

        Used in training to create noisy samples from clean data.

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
        ...

    def timesteps(
        self,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N+1"]:
        r"""
        Generate discrete time-steps for sampling.

        Used in sampling to produce the sequence of diffusion times.

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
        Tensor
            Time-steps tensor of shape :math:`(N + 1,)` in decreasing order,
            with the last element being 0.
        """
        ...

    def init_latents(
        self,
        spatial_shape: Tuple[int, ...],
        tN: Float[Tensor, " B"],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " B *spatial_shape"]:
        r"""
        Initialize the noisy latent state :math:`\mathbf{x}_N` for sampling.

        Used in sampling to generate the initial condition at diffusion time
        ``tN``.

        Parameters
        ----------
        spatial_shape : Tuple[int, ...]
            Spatial shape of the latent state, e.g., ``(C, H, W)``.
        tN : Tensor
            Initial diffusion time of shape :math:`(B,)`. Determines the noise
            level for the initial latent state.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Initial noisy latent state of shape :math:`(B, *spatial\_shape)`.
        """
        ...

    def get_denoiser(
        self,
        **kwargs: Any,
    ) -> Denoiser:
        r"""
        Factory that converts a predictor into a denoiser for sampling.

        Used in sampling to transform a :class:`Predictor` (e.g., x0-predictor,
        score-predictor) into a :class:`Denoiser` that returns the
        update term compatible with the solver. The exact transformation
        depends on the noise scheduler implementation.

        Parameters
        ----------
        **kwargs : Any
            Implementation-specific keyword arguments. Concrete
            implementations typically accept keyword-only predictor arguments
            (e.g., ``score_predictor``, ``x0_predictor``). See concrete classes
            docstrings for details (e.g.
            :meth:`LinearGaussianNoiseScheduler.get_denoiser`).

        Returns
        -------
        Denoiser
            A callable that implements the
            :class:`~physicsnemo.diffusion.Denoiser` interface, for use
            with solvers and the
            :func:`~physicsnemo.diffusion.samplers.sample` function.
        """
        ...

    def loss_weight(
        self,
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"] | Float[Tensor, " N C"]:
        r"""
        Compute loss weight for denoising score matching training.

        Used in training to weight the per-sample loss in
        :class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss`.

        Parameters
        ----------
        t : Tensor
            Diffusion time values of shape :math:`(N,)`.

        Returns
        -------
        Tensor
            Loss weight with leading dimension :math:`N`.  Shape is
            :math:`(N,)` for scalar ``sigma_data``, or :math:`(N, C)`
            when the scheduler uses per-channel ``sigma_data`` (see
            :class:`EDMNoiseScheduler`).
        """
        ...
