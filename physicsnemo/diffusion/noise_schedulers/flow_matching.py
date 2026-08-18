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

"""Flow matching (rectified flow) noise scheduler."""

from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser, Predictor

from .linear_gaussian import LinearGaussianNoiseScheduler


class FlowMatchingNoiseScheduler(LinearGaussianNoiseScheduler):
    r"""
    Flow matching (rectified flow / conditional optimal transport) noise
    scheduler.

    Implements the linear interpolation path used in flow matching, with
    :math:`\alpha(t) = 1 - t` and :math:`\sigma(t) = t` for
    :math:`t \in [0, 1]`:

    .. math::
        \mathbf{x}(t) = (1 - t)\, \mathbf{x}_0 + t\, \boldsymbol{\epsilon},
        \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})

    At :math:`t = 0` the state is clean data, and at :math:`t = 1` it is pure
    Gaussian noise. The conditional velocity field associated with this path is

    .. math::
        \mathbf{v} = \frac{d\mathbf{x}(t)}{dt}
        = \boldsymbol{\epsilon} - \mathbf{x}_0

    which is the regression target of the flow matching training objective
    (see :class:`~physicsnemo.diffusion.metrics.losses.FlowMatchingLoss`).

    **Sampling time-steps** are linearly spaced from ``t_max`` down to 0.

    **Training times** are sampled uniformly in :math:`[t_{\min}, t_{\max}]`.

    Since this path is linear-Gaussian, this scheduler is fully interoperable
    with the rest of the diffusion framework (losses, samplers, solvers,
    domain-parallel wrapper). In particular a model trained with denoising
    score matching on this schedule and a model trained with flow matching
    parameterize the same probability-flow ODE.

    .. note::

        :meth:`get_denoiser` is overridden with closed-form expressions for
        the probability-flow ODE right-hand side (the velocity field), which
        avoids the :math:`\dot{\alpha}(t)/\alpha(t)` singularity of the
        generic formulation at :math:`t = 1`. Sampling can therefore start
        exactly at :math:`t = 1` (pure noise) with ``x0_predictor`` or
        ``velocity_predictor``. For ``epsilon_predictor``,
        ``score_predictor``, or ``denoising_type="sde"``, the reverse process
        is inherently singular at :math:`t = 1` and ``t_max`` must be set
        strictly below 1 (e.g. ``t_max=0.999``).

    Parameters
    ----------
    t_min : float, optional
        Minimum diffusion time used when sampling training times, by
        default 0.0. Must satisfy ``0 <= t_min < t_max``. Set slightly
        above 0 (e.g. ``1e-3``) when training an x0-predictor, since the
        x0-to-velocity conversion is singular at :math:`t = 0`.
    t_max : float, optional
        Maximum diffusion time, by default 1.0. Used as the initial time for
        sampling time-steps and as the upper bound for training times. Must
        satisfy ``t_min < t_max <= 1``.

    Note
    ----
    References: `Flow Matching for Generative Modeling
    <https://arxiv.org/abs/2210.02747>`_, `Flow Straight and Fast: Learning to
    Generate and Transfer Data with Rectified Flow
    <https://arxiv.org/abs/2209.03003>`_

    Examples
    --------
    Basic training and sampling workflow using the flow matching noise
    scheduler:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import (
    ...     FlowMatchingNoiseScheduler,
    ... )
    >>>
    >>> scheduler = FlowMatchingNoiseScheduler()
    >>>
    >>> # Training: sample times and interpolate towards noise
    >>> x0 = torch.randn(4, 3, 8, 8)  # Clean data
    >>> t = scheduler.sample_time(4)    # Uniform times in [0, 1]
    >>> x_t = scheduler.add_noise(x0, t)  # (1 - t) * x0 + t * noise
    >>> x_t.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Sampling: generate timesteps and initial latents
    >>> t_steps = scheduler.timesteps(10)
    >>> tN = t_steps[0].expand(4)  # Initial time (t=1) for batch of 4
    >>> xN = scheduler.init_latents((3, 8, 8), tN)  # Pure Gaussian noise
    >>> xN.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Convert velocity-predictor to denoiser for sampling
    >>> velocity_predictor = lambda x, t: -x  # Toy velocity-predictor
    >>> denoiser = scheduler.get_denoiser(velocity_predictor=velocity_predictor)
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
        t_max: float = 1.0,
    ) -> None:
        if not 0.0 <= t_min < t_max <= 1.0:
            raise ValueError(
                f"t_min and t_max must satisfy 0 <= t_min < t_max <= 1, "
                f"got t_min={t_min}, t_max={t_max}."
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
            Data type of the tensor.

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
            Data type of the tensor.

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

        The standard flow matching objective uses uniform weighting across
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

    def get_denoiser(
        self,
        *,
        score_predictor: Predictor | None = None,
        x0_predictor: Predictor | None = None,
        epsilon_predictor: Predictor | None = None,
        velocity_predictor: Predictor | None = None,
        denoising_type: Literal["ode", "sde"] = "ode",
        **kwargs: Any,
    ) -> Denoiser:
        r"""
        Factory that converts a predictor to a denoiser for sampling.

        Accepts exactly one of **velocity-predictor**, **x0-predictor**,
        **epsilon-predictor**, or **score-predictor**. The returned denoiser
        computes the right-hand side of the reverse ODE or SDE using
        closed-form expressions for the flow matching path.

        For the ODE (``denoising_type="ode"``), the right-hand side is the
        velocity field :math:`\mathbf{v}(\mathbf{x}, t)`:

        .. math::
            \frac{d\mathbf{x}}{dt} = \mathbf{v}(\mathbf{x}, t) =
            \begin{cases}
                \hat{\mathbf{v}} & \text{(velocity)} \\
                (\mathbf{x} - \hat{\mathbf{x}}_0) / t & \text{(x0)} \\
                (\hat{\boldsymbol{\epsilon}} - \mathbf{x}) / (1 - t)
                & \text{(epsilon)} \\
                -(\mathbf{x} + t\, \hat{s}) / (1 - t) & \text{(score)}
            \end{cases}

        For the SDE (``denoising_type="sde"``), the deterministic part of
        the drift is:

        .. math::
            \mathbf{v}(\mathbf{x}, t) - \frac{t}{1 - t}\,
            s(\mathbf{x}, t)

        where :math:`s` is the score (the stochastic term :math:`g(t)
        d\mathbf{W}` is handled by the solver).

        .. warning::

            The velocity and x0 parameterizations are regular at
            :math:`t = 1`, so ODE sampling can start from pure noise
            (``t_max=1``, the default). The epsilon and score
            parameterizations, as well as ``denoising_type="sde"``, are
            singular at :math:`t = 1`; use them with ``t_max < 1``.

        Parameters
        ----------
        score_predictor : Predictor, optional
            A score-predictor that takes ``(x_t, t)`` and returns the score.
            Mutually exclusive with the other predictor arguments.
        x0_predictor : Predictor, optional
            An x0-predictor that takes ``(x_t, t)`` and returns an estimate
            of clean data :math:`\hat{\mathbf{x}}_0`. Mutually exclusive with
            the other predictor arguments.
        epsilon_predictor : Predictor, optional
            An epsilon-predictor that takes ``(x_t, t)`` and returns an
            estimate of the noise :math:`\hat{\boldsymbol{\epsilon}}`.
            Mutually exclusive with the other predictor arguments.
        velocity_predictor : Predictor, optional
            A velocity-predictor that takes ``(x_t, t)`` and returns an
            estimate of the velocity :math:`\hat{\mathbf{v}}` (e.g. a model
            trained with
            :class:`~physicsnemo.diffusion.metrics.losses.FlowMatchingLoss`).
            Mutually exclusive with the other predictor arguments.
        denoising_type : {"ode", "sde"}, default="ode"
            Type of reverse process. Use ``"ode"`` for deterministic sampling,
            ``"sde"`` for stochastic sampling.
        **kwargs : Any
            Ignored.

        Returns
        -------
        Denoiser
            A denoiser computing the RHS of the reverse ODE/SDE. Implements
            the :class:`~physicsnemo.diffusion.Denoiser` interface.

        Raises
        ------
        ValueError
            If not exactly one of ``score_predictor``, ``x0_predictor``,
            ``epsilon_predictor``, or ``velocity_predictor`` is provided.
        ValueError
            If ``denoising_type`` is not ``"ode"`` or ``"sde"``.
        ValueError
            If ``denoising_type="sde"`` and this scheduler's ``t_max >= 1``,
            since the SDE drift is singular at :math:`t = 1`.

        Examples
        --------
        Generate ODE RHS from a velocity-predictor:

        >>> import torch
        >>> scheduler = FlowMatchingNoiseScheduler()
        >>> velocity_pred = lambda x, t: -x  # Toy velocity-predictor
        >>> denoiser = scheduler.get_denoiser(velocity_predictor=velocity_pred)
        >>> x = torch.randn(2, 3, 8, 8)
        >>> t = torch.ones(2)
        >>> dx_dt = denoiser(x, t)  # Returns ODE RHS for sampling
        >>> dx_dt.shape
        torch.Size([2, 3, 8, 8])
        """
        provided = sum(
            p is not None
            for p in (
                score_predictor,
                x0_predictor,
                epsilon_predictor,
                velocity_predictor,
            )
        )
        if provided != 1:
            raise ValueError(
                "Exactly one of 'score_predictor', 'x0_predictor', "
                "'epsilon_predictor', or 'velocity_predictor' must be provided."
            )
        if denoising_type not in ("ode", "sde"):
            raise ValueError(
                f"denoising_type must be 'ode' or 'sde', got '{denoising_type}'"
            )
        if denoising_type == "sde" and self.t_max >= 1.0:
            raise ValueError(
                "denoising_type='sde' is singular at t=1 for the flow matching "
                f"path (division by 1 - t), but this scheduler has t_max="
                f"{self.t_max} >= 1. Construct FlowMatchingNoiseScheduler with "
                "t_max < 1 (e.g. t_max=0.999) to use SDE sampling."
            )

        def _bc(t: Tensor, ndim: int) -> Tensor:
            return t.reshape((-1,) + (1,) * (ndim - 1))

        # Closed-form velocity field (probability-flow ODE RHS)
        if velocity_predictor is not None:
            velocity_fn = velocity_predictor
        elif x0_predictor is not None:

            def velocity_fn(
                x: Float[Tensor, " B *dims"],
                t: Float[Tensor, " B"],
            ) -> Float[Tensor, " B *dims"]:
                x0 = x0_predictor(x, t)
                return (x - x0) / _bc(t, x.ndim)

        elif epsilon_predictor is not None:

            def velocity_fn(
                x: Float[Tensor, " B *dims"],
                t: Float[Tensor, " B"],
            ) -> Float[Tensor, " B *dims"]:
                eps = epsilon_predictor(x, t)
                return (eps - x) / (1 - _bc(t, x.ndim))

        else:

            def velocity_fn(
                x: Float[Tensor, " B *dims"],
                t: Float[Tensor, " B"],
            ) -> Float[Tensor, " B *dims"]:
                score = score_predictor(x, t)
                t_bc = _bc(t, x.ndim)
                return -(x + t_bc * score) / (1 - t_bc)

        if denoising_type == "ode":
            return velocity_fn

        # SDE: deterministic drift = velocity - (1/2) g^2 * score, with
        # g^2 = 2t / (1 - t) for the flow matching path.
        if score_predictor is not None:
            score_fn = score_predictor
        elif x0_predictor is not None:
            x0_to_score = self.x0_to_score

            def score_fn(
                x: Float[Tensor, " B *dims"],
                t: Float[Tensor, " B"],
            ) -> Float[Tensor, " B *dims"]:
                return x0_to_score(x0_predictor(x, t), x, t)

        elif epsilon_predictor is not None:
            epsilon_to_score = self.epsilon_to_score

            def score_fn(
                x: Float[Tensor, " B *dims"],
                t: Float[Tensor, " B"],
            ) -> Float[Tensor, " B *dims"]:
                return epsilon_to_score(epsilon_predictor(x, t), t)

        else:
            velocity_to_x0 = self.velocity_to_x0
            x0_to_score_v = self.x0_to_score

            def score_fn(
                x: Float[Tensor, " B *dims"],
                t: Float[Tensor, " B"],
            ) -> Float[Tensor, " B *dims"]:
                x0 = velocity_to_x0(velocity_predictor(x, t), x, t)
                return x0_to_score_v(x0, x, t)

        def sde_denoiser(
            x: Float[Tensor, " B *dims"],
            t: Float[Tensor, " B"],
        ) -> Float[Tensor, " B *dims"]:
            t_bc = _bc(t, x.ndim)
            return velocity_fn(x, t) - (t_bc / (1 - t_bc)) * score_fn(x, t)

        return sde_denoiser
