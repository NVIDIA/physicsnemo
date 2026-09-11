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

    Implements the linear interpolation path used in flow matching, with
    :math:`\alpha(t) = 1 - t` and :math:`\sigma(t) = t` for
    :math:`t \in [0, 1]`:

    .. math::
        \mathbf{x}(t) = (1 - t)\, \mathbf{x}_0 + t\, \boldsymbol{\epsilon},
        \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})

    At :math:`t = 0` the state is clean data, and at :math:`t = 1` pure
    Gaussian noise. The flow (velocity) field of this path,

    .. math::
        \mathbf{v} = \frac{d\mathbf{x}(t)}{dt}
        = \boldsymbol{\epsilon} - \mathbf{x}_0,

    is the regression target of
    :class:`~physicsnemo.diffusion.metrics.losses.FlowMatchingLoss`
    (``prediction_type="flow"``). Sampling time-steps run linearly from
    ``t_max`` down to 0; training times follow a uniform distribution on
    :math:`[t_{\min}, t_{\max}]`.

    .. warning::

        The reverse-process drift :math:`\dot{\alpha}(t)/\alpha(t)` is
        singular at :math:`t = 1`. The default ``t_max=0.99`` keeps
        sampling clear of it (``0.999`` would round to exactly ``1.0`` in
        ``bfloat16``); constructing with ``t_max=1.0`` emits a
        ``UserWarning``. Training with
        :class:`~physicsnemo.diffusion.metrics.losses.FlowMatchingLoss`
        stays exact at any ``t_max``, including ``1.0``.

    Parameters
    ----------
    t_min : float, optional
        Smallest training time, by default 0.0. Requires
        ``0 <= t_min < t_max``. Set slightly above 0 (e.g. ``1e-3``) for
        x0-predictors, whose flow conversion is singular at :math:`t = 0`.
    t_max : float, optional
        Largest diffusion time, by default 0.99: the first sampling
        time-step and the upper bound for training times. Requires
        ``t_min < t_max <= 1``; ``t_max=1.0`` is valid for training but
        emits a ``UserWarning`` (see the warning above).

    Note
    ----
    References: `Flow Matching for Generative Modeling
    <https://arxiv.org/abs/2210.02747>`_, `Flow Straight and Fast: Learning to
    Generate and Transfer Data with Rectified Flow
    <https://arxiv.org/abs/2209.03003>`_

    Examples
    --------
    Basic training and sampling workflow:

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
    >>> denoiser(xN, tN).shape  # ODE RHS for sampling
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # An x0-predictor works as well (score/x0 conversions still apply)
    >>> x0_predictor = lambda x, t: x * 0.9  # Toy x0-predictor
    >>> denoiser = scheduler.get_denoiser(x0_predictor=x0_predictor)
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
                "the reverse-process drift is singular at t=1, so sampling "
                "with get_denoiser starting from t_max=1.0 will produce "
                "non-finite values. Use t_max slightly below 1 for sampling "
                "(the default is 0.99; avoid 0.999, which rounds to 1.0 in "
                "bfloat16). Training with FlowMatchingLoss is unaffected.",
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
            Time-steps tensor of shape :math:`(N + 1,)` in decreasing order,
            with the last element being 0.
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
        Sample N diffusion times uniformly in :math:`[t_{\min}, t_{\max}]`.

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
            Sampled diffusion times of shape :math:`(N,)`.
        """
        u = torch.rand(N, device=device, dtype=dtype)
        return self.t_min + u * (self.t_max - self.t_min)

    def loss_weight(
        self,
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"]:
        r"""
        Compute flow matching loss weight: :math:`w(t) = 1`.

        Standard flow matching uses uniform weighting across
        diffusion times.

        Parameters
        ----------
        t : Tensor
            Diffusion time values of shape :math:`(N,)`.

        Returns
        -------
        Tensor
            Loss weight of shape :math:`(N,)`, all ones.
        """
        return torch.ones_like(t)
