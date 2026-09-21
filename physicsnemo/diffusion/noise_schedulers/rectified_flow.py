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

"""Rectified flow noise scheduler."""

import warnings

import torch
from jaxtyping import Float
from torch import Tensor

from .linear_gaussian import LinearGaussianNoiseScheduler


class RectifiedFlowNoiseScheduler(LinearGaussianNoiseScheduler):
    r"""
    Rectified flow noise scheduler.

    The rectified-flow formulation uses :math:`\alpha(t) = 1 - t` and
    :math:`\sigma(t) = t` for :math:`t \in [0, 1]`. The state corresponds to
    clean data at :math:`t = 0` and standard Gaussian noise at :math:`t = 1`.
    This interpolation has the flow (velocity)

    .. math::
        \mathbf{v} = \frac{d\mathbf{x}(t)}{dt}
        = \boldsymbol{\epsilon} - \mathbf{x}_0.

    A model can learn this velocity field directly and use the same
    parameterization for training and sampling.

    Sampled times during training follow a uniform distribution over
    :math:`[t_{\min}, t_{\max}]`.

    Sampling uses equidistant time steps that decrease from ``t_max`` to zero.
    The sampling grid does not use :math:`t_{\min}`.

    .. warning::

        The endpoints require care:

        - At :math:`t = 0`: the direct flow parameterization remains finite,
          but :meth:`x0_to_score`, :meth:`epsilon_to_score`,
          :meth:`x0_to_epsilon`, :meth:`x0_to_flow`, and
          :meth:`flow_to_score` become singular. Keep the default
          ``t_min=0.0`` unless training uses one of these conversions at this
          endpoint. In that case, set ``t_min`` slightly above zero.
        - At :math:`t = 1`: :meth:`drift`, :meth:`diffusion`,
          :meth:`score_to_x0`, :meth:`epsilon_to_x0`, and
          :meth:`score_to_flow` become singular. Keep ``t_max`` below one for
          sampling. The default ``t_max=0.99`` is safe for ``bfloat16``; use
          ``t_max=1.0`` only for training operations that remain finite at this
          endpoint.

    Parameters
    ----------
    t_min : float, optional
        Lower bound for times sampled during training by :meth:`sample_time`,
        by default 0.0. Requires ``0 <= t_min < t_max``. Most flow-matching
        workflows can keep the default; see the warning above for when to use a
        positive value.
    t_max : float, optional
        Upper bound for times sampled during training and initial diffusion
        time :math:`t_N` of the sampling grid returned by :meth:`timesteps`, by
        default 0.99. Requires ``t_min < t_max <= 1``. Keep this value below one
        for sampling.

    Note
    ----
    References: `Flow Matching for Generative Modeling
    <https://arxiv.org/abs/2210.02747>`_, `Flow Straight and Fast: Learning to
    Generate and Transfer Data with Rectified Flow
    <https://arxiv.org/abs/2209.03003>`_

    Examples
    --------
    Construct noisy states for training, then initialize sampling with the same
    flow parameterization:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import (
    ...     RectifiedFlowNoiseScheduler,
    ... )
    >>>
    >>> scheduler = RectifiedFlowNoiseScheduler()
    >>>
    >>> # Training: sample times and interpolate towards noise
    >>> x0 = torch.randn(4, 3, 8, 8)  # Clean data
    >>> t = scheduler.sample_time(4)    # Uniform times in [0, t_max]
    >>> x_t = scheduler.add_noise(x0, t)  # (1 - t) * x0 + t * noise
    >>> x_t.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Sampling: generate timesteps and initial latents
    >>> t_steps = scheduler.timesteps(10)
    >>> tN = t_steps[0].expand(4)  # Initial time (t=0.99) for batch of 4
    >>> xN = scheduler.init_latents((3, 8, 8), tN)  # Near-pure Gaussian noise
    >>> xN.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Convert flow-predictor to denoiser for sampling
    >>> flow_predictor = lambda x, t: -x  # Toy flow-predictor
    >>> denoiser = scheduler.get_denoiser(flow_predictor=flow_predictor)
    >>> denoiser(xN, tN).shape
    torch.Size([4, 3, 8, 8])
    """

    def __init__(
        self,
        t_min: float = 0.0,
        t_max: float = 0.99,
    ) -> None:
        if not 0.0 <= t_min < t_max <= 1.0:
            raise ValueError(
                f"t_min and t_max must satisfy 0 <= t_min < t_max <= 1, "
                f"got t_min={t_min}, t_max={t_max}."
            )
        if t_max >= 1.0:
            warnings.warn(
                "RectifiedFlowNoiseScheduler was constructed with t_max=1.0: "
                "the reverse-process drift is undefined at t=1. Use a value "
                "below 1 for sampling; the default is 0.99. Avoid 0.999 "
                "because bfloat16 rounds it to 1.0. The endpoint remains valid "
                "for operations that stay finite there.",
                UserWarning,
                stacklevel=2,
            )
        self.t_min = t_min
        self.t_max = t_max

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
        r"""Linearly decaying signal coefficient: :math:`\alpha(t) = 1 - t`."""
        return 1 - t

    def alpha_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Constant derivative: :math:`\dot{\alpha}(t) = -1`."""
        return -torch.ones_like(t)

    def timesteps(
        self,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N+1"]:
        r"""
        Generate linearly spaced time-steps from ``t_max`` down to 0.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Tensor data type.

        Returns
        -------
        torch.Tensor
            Sampling times of shape :math:`(N + 1,)` in decreasing order. The
            last value is zero.
        """
        return torch.linspace(
            self.t_max, 0.0, num_steps + 1, device=device, dtype=dtype
        )

    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""
        Sample :math:`N` time values uniformly from
        :math:`[t_{\min}, t_{\max}]`.

        Parameters
        ----------
        N : int
            Number of time values to sample.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Tensor data type.

        Returns
        -------
        Tensor
            Sampled time values of shape :math:`(N,)`.
        """
        u = torch.rand(N, device=device, dtype=dtype)
        return self.t_min + u * (self.t_max - self.t_min)

    def loss_weight(
        self,
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"]:
        r"""
        Compute flow matching loss weight: :math:`w(t) = 1`.

        Rectified flow applies equal weight to every sampled time.

        Parameters
        ----------
        t : Tensor
            Time values of shape :math:`(N,)`.

        Returns
        -------
        Tensor
            Loss weight of shape :math:`(N,)`, all ones.
        """
        return torch.ones_like(t)
