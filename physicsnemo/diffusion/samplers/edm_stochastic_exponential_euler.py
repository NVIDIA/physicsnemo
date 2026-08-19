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

"""First-order stochastic exponential Euler sampler with EDM-style churn."""

import math
from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser

from .base import Solver


class EDMStochasticExponentialEulerSolver(Solver):
    r"""
    Add controlled noise to exponential Euler sampling.

    This solver combines the first-order
    :class:`ExponentialEulerSolver` update with two optional forms of noise
    injection:

    - EDM-style churn perturbs the state before the integration step.
    - ``renoise`` replaces part of the noise at the end of the step with a
      fresh sample.

    Use these controls to increase sample diversity while retaining the
    semi-linear update:

    .. math::
        \frac{d\mathbf{x}}{dt} = D(\mathbf{x}, t)
        = A(t) \, \mathbf{x} + N(\mathbf{x}, t)

    The ``denoiser`` provides the full right-hand side :math:`D`, while
    ``linear_fn`` provides :math:`A(t)`. The solver derives the nonlinear term
    :math:`N` automatically. A noise scheduler can provide both callables.

    .. important::

        This is **not** a true SDE solver. It performs ad-hoc noise injection
        at each step to improve sample diversity, but the underlying
        integration is still an ODE step, so the denoiser should return the
        right-hand side of the **ODE**, not the SDE.

    By default, the solver applies churn directly in diffusion-time space. If
    diffusion time differs from noise level, as it does for a VP schedule,
    provide ``sigma_fn`` and ``sigma_inv_fn`` to apply churn at the intended
    noise levels. Use ``diffusion_fn`` when the churn strength must follow the
    schedule's diffusion coefficient.

    The ``renoise`` value :math:`r \in [0, 1]` controls how much noise the
    solver refreshes at the arrival point:

    - ``renoise=0`` keeps all the carried noise and recovers the churn-style
      samplers of the EDM paper.
    - Intermediate values mix carried and fresh noise, producing ancestral
      sampling variants.
    - ``renoise=1`` replaces all carried noise and gives the re-noising
      sampler used by distilled and consistency models, also known as
      stochastic DDIM.

    The optional callables have the signatures:

    .. code-block:: python

        def linear_fn(
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # linear coefficient A(t), shape: (B,)

        def sigma_fn(
            t: Tensor,  # shape: (B,) or broadcastable
        ) -> Tensor: ...  # noise level, same shape as t

        def sigma_inv_fn(
            sigma: Tensor,  # shape: (B,) or broadcastable
        ) -> Tensor: ...  # diffusion time, same shape as sigma

        def diffusion_fn(
            x: Tensor,  # shape: (B, *dims)
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # g^2(x, t), broadcastable to shape of x

    Parameters
    ----------
    denoiser : Denoiser
        Right-hand side of the **ODE**, following the
        :class:`~physicsnemo.diffusion.Denoiser` interface. Do not provide an
        SDE right-hand side because this solver adds noise internally. In most
        workflows, get the callable from
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`
        with ``denoising_type="ode"``.
    linear_fn : Callable[[Tensor], Tensor] | None, optional
        Linear coefficient :math:`A(t)` used by the exponential Euler update.
        Use the signature above. In most workflows, get it from
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.get_linear_denoiser`,
        using the same predictor parameterization as ``denoiser``. The default
        is ``None``, which uses an explicit Euler update.
    S_churn : float, optional
        Controls the amount of noise added at each step. Higher values add
        more stochasticity. By default 0 (no churn), in which case this
        solver with ``renoise=0`` matches the deterministic
        :class:`ExponentialEulerSolver`.
    S_min : float, optional
        Smallest diffusion time (or noise level when using ``sigma_fn`` and
        ``sigma_inv_fn``) subject to churn. By default 0.
    S_max : float, optional
        Largest diffusion time (or noise level when using ``sigma_fn`` and
        ``sigma_inv_fn``) subject to churn. By default ``float("inf")``.
    S_noise : float, optional
        Scales the churn noise. Larger values add more noise to the latent
        state.
        By default 1.
    num_steps : int, optional
        Total number of sampling steps, used to scale churn. By default 18.
    sigma_fn : Callable[[Tensor], Tensor] | None, optional
        Maps time to noise level :math:`\sigma(t)`. Useful for linear-Gaussian
        schedules where :math:`\sigma(t) \neq t`. Typically
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.sigma`.
        Requires ``sigma_inv_fn``. By default ``None`` (identity mapping).
    sigma_inv_fn : Callable[[Tensor], Tensor] | None, optional
        Maps noise level back to time. Typically
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.sigma_inv`.
        Requires ``sigma_fn``. By default ``None`` (identity mapping).
    diffusion_fn : Callable[[Tensor, Tensor], Tensor] | None, optional
        Time-dependent scaling for churn, applied on top of ``S_noise``. Use
        the squared diffusion coefficient :math:`g^2(\mathbf{x}, t)` from the
        reverse SDE, available from
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.diffusion`.
        By default ``None`` (:math:`g^2 = 2t`), which corresponds to an
        EDM-like noise schedule.
    renoise : float, optional
        Fraction :math:`r \in [0, 1]` of arrival noise replaced with fresh
        noise at each step. Use ``0`` to keep the carried noise, ``1`` to
        replace the carried noise, or an intermediate value to mix the two. By
        default 0.

    Note
    ----
    References:

    - EDM: `Elucidating the Design Space of Diffusion-Based Generative
      Models <https://arxiv.org/abs/2206.00364>`_
    - `Denoising Diffusion Implicit Models <https://arxiv.org/abs/2010.02502>`_
    - `Consistency Models <https://arxiv.org/abs/2303.01469>`_

    Examples
    --------
    Add churn while sampling the probability-flow ODE of an EDM schedule:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.samplers import (
    ...     EDMStochasticExponentialEulerSolver,
    ... )
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>> x0_pred = lambda x, t: x * 0.1  # Toy x0-predictor
    >>> solver = EDMStochasticExponentialEulerSolver(
    ...     scheduler.get_denoiser(x0_predictor=x0_pred),
    ...     linear_fn=scheduler.get_linear_denoiser(prediction_type="x0"),
    ...     S_churn=40,
    ...     num_steps=18,
    ... )
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> x_tm1 = solver.step(x_t, torch.tensor([5.0]), torch.tensor([2.5]))
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])

    With a VP schedule and an x0-predictor, full noise renewal gives a
    re-noising sampler suitable for distilled few-step and consistency models:

    >>> from physicsnemo.diffusion.noise_schedulers import VPNoiseScheduler
    >>> scheduler = VPNoiseScheduler()
    >>> x0_pred = lambda x, t: x * 0.1  # Toy x0-predictor
    >>> solver = EDMStochasticExponentialEulerSolver(
    ...     scheduler.get_denoiser(x0_predictor=x0_pred),
    ...     linear_fn=scheduler.get_linear_denoiser(prediction_type="x0"),
    ...     sigma_fn=scheduler.sigma,
    ...     sigma_inv_fn=scheduler.sigma_inv,
    ...     renoise=1.0,
    ... )
    >>> x_tm1 = solver.step(x_t, torch.tensor([0.6]), torch.tensor([0.3]))
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        denoiser: Denoiser,
        linear_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
        num_steps: int = 18,
        sigma_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        sigma_inv_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        diffusion_fn: Callable[
            [Float[Tensor, " B *dims"], Float[Tensor, " B"]], Float[Tensor, " B *_"]
        ]
        | None = None,
        renoise: float = 0,
    ) -> None:
        self.denoiser = denoiser
        if linear_fn is None:
            self.linear_fn = lambda t: torch.zeros_like(t)
        else:
            self.linear_fn = linear_fn
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.num_steps = num_steps
        if not 0 <= renoise <= 1:
            raise ValueError(f"renoise must be in [0, 1], got {renoise}")
        self.renoise = renoise
        # Noise level kept by the deterministic stage, so that the renewed
        # noise restores the exact arrival level
        self._kept_fraction = math.sqrt(1 - renoise**2)
        if sigma_fn is None and sigma_inv_fn is None:
            self.sigma_fn = lambda t: t
            self.sigma_inv_fn = lambda sigma: sigma
        elif sigma_fn is not None and sigma_inv_fn is not None:
            self.sigma_fn = sigma_fn
            self.sigma_inv_fn = sigma_inv_fn
        else:
            raise ValueError(
                "sigma_fn and sigma_inv_fn must both be provided or both None."
            )
        if diffusion_fn is None:
            self.diffusion_fn = lambda x, t: 2 * t.reshape(-1, *([1] * (x.ndim - 1)))
        else:
            self.diffusion_fn = diffusion_fn
        # Bind the deterministic-stage target at construction: renoise=0
        # skips the noise-level round-trip so that it matches the churn-only
        # path exactly
        if renoise == 0:
            self._t_dn_fn = lambda t_next: t_next
        else:
            sigma_fn = self.sigma_fn
            sigma_inv_fn = self.sigma_inv_fn
            kept_fraction = self._kept_fraction
            self._t_dn_fn = lambda t_next: sigma_inv_fn(
                kept_fraction * sigma_fn(t_next)
            )

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one stochastic exponential Euler sampling step.

        Parameters
        ----------
        x : Tensor
            Current noisy latent state :math:`\mathbf{x}_n` of shape
            :math:`(B, *)` where :math:`B` is the batch size.
        t_cur : Tensor
            Current diffusion time :math:`t_n` of shape :math:`(B,)`.
        t_next : Tensor
            Target diffusion time :math:`t_{n-1}` of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Updated latent state :math:`\mathbf{x}_{n-1}` at time
            ``t_next``, same shape as ``x``.
        """
        # Ensure contiguous strides so successive denoiser calls (across
        # sampling steps) present the same stride layout to torch.compile,
        # avoiding spurious recompilations / silently divergent traces.
        t_cur = t_cur.contiguous()
        t_next = t_next.contiguous()

        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        expected_shape = (-1,) + (1,) * (x.ndim - 1)
        t_cur_bc = t_cur.reshape(expected_shape)

        gamma_base = min(self.S_churn / self.num_steps, math.sqrt(2) - 1)

        # Compute perturbed time t_hat with increased noise
        # NOTE: sigma_fn and sigma_inv_fn are identity if not provided (stays
        # in time-step space). diffusion_fn defaults to g^2 = 2t (EDM-like
        # noise schedule).
        sigma_cur_bc = self.sigma_fn(t_cur_bc)
        # Mask: apply churn only where S_min <= sigma <= S_max
        churn_mask = (sigma_cur_bc >= self.S_min) & (sigma_cur_bc <= self.S_max)
        gamma_bc = torch.where(churn_mask, gamma_base, 0.0)
        sigma_hat_bc = sigma_cur_bc + gamma_bc * sigma_cur_bc
        t_hat_bc = self.sigma_inv_fn(sigma_hat_bc)
        # Noise scale: sqrt(sigma_hat^2 - sigma_cur^2) * S_noise * g(x,t) / sqrt(2*t)
        g_sq_bc = self.diffusion_fn(x, t_cur)
        safe_t_cur_bc = torch.where(t_cur_bc == 0, torch.ones_like(t_cur_bc), t_cur_bc)
        noise_scale_bc = (
            (sigma_hat_bc**2 - sigma_cur_bc**2).clamp(min=0).sqrt()
            * self.S_noise
            * (g_sq_bc / (2 * safe_t_cur_bc)).sqrt()
        )
        noise_scale_bc = torch.where(
            t_cur_bc == 0, torch.zeros_like(noise_scale_bc), noise_scale_bc
        )

        # Perturb latent with noise
        x_hat = x + noise_scale_bc * torch.randn_like(x)

        # The deterministic stage aims at the reduced arrival level kept by
        # the renoise dial
        t_dn = self._t_dn_fn(t_next)

        # Exponential Euler stage from t_hat, isolating the nonlinear part
        # of the RHS: N = D - A x
        t_hat = t_hat_bc.reshape(x.shape[0])
        h_bc = (t_dn - t_hat).reshape(expected_shape)
        d_cur = self.denoiser(x_hat, t_hat)
        a_bc = self.linear_fn(t_hat).reshape(expected_shape)
        n_cur = d_cur - a_bc * x_hat

        # h * phi1(h A) = expm1(h A) / A; the A -> 0 limit equals h
        z = h_bc * a_bc
        a_safe = torch.where(a_bc == 0, torch.ones_like(a_bc), a_bc)
        h_phi1 = torch.where(a_bc == 0, h_bc, torch.expm1(z) / a_safe)

        x_next = torch.exp(z) * x_hat + h_phi1 * n_cur

        # The zero-renoise branch skips the fresh draw so that renoise=0
        # consumes the same random sequence as the churn-only sampler, which
        # keeps seeded trajectories reproducible across the two
        if self.renoise != 0:
            sigma_next_bc = self.sigma_fn(t_next).reshape(expected_shape)
            x_next = x_next + self.renoise * sigma_next_bc * torch.randn_like(x)

        return x_next
