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

"""First-order exponential Euler solver for semi-linear diffusion ODEs."""

from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser

from ._utils import gauss_legendre
from .base import Solver

# Number of Gauss-Legendre points used by the internal quadrature of this
# solver
_NUM_QUADRATURE_POINTS = 4

# Fractional offset from t_next toward t_cur of the fallback probe for the
# slope-to-bias ratio; matches the innermost node of the 4-point rule
_RATIO_PROBE_OFFSET = 0.0694318442029737


def _nonlinear_weight(
    t_cur: Float[Tensor, " B"],
    t_next: Float[Tensor, " B"],
    bias_fn: Callable[[Tensor], Tensor],
    bias_int_fn: Callable[[Tensor], Tensor],
    slope_fn: Callable[[Tensor], Tensor],
) -> Float[Tensor, " B"]:
    r"""
    Compute the exponential-kernel weight of the nonlinear term over one step.
    """
    bias_int_next = bias_int_fn(t_next)

    # Finite estimate of the slope-to-bias ratio b / a near t_next: prefer
    # the endpoint value; when not finite (an indeterminate ratio at a
    # vanishing noise level, or a zero bias), fall back to a probe point
    # near t_next, then to zero (pure quadrature)
    t_star = t_next + _RATIO_PROBE_OFFSET * (t_cur - t_next)
    ratio_next = slope_fn(t_next) / bias_fn(t_next)
    ratio_star = slope_fn(t_star) / bias_fn(t_star)
    ratio = torch.where(torch.isfinite(ratio_next), ratio_next, ratio_star)
    ratio = torch.where(torch.isfinite(ratio), ratio, torch.zeros_like(ratio))

    def integrand(s: Tensor) -> Tensor:
        return torch.exp(bias_int_next - bias_int_fn(s)) * (
            slope_fn(s) - ratio * bias_fn(s)
        )

    exact_part = ratio * torch.expm1(bias_int_next - bias_int_fn(t_cur))
    return exact_part + gauss_legendre(integrand, t_cur, t_next, _NUM_QUADRATURE_POINTS)


class ExponentialEulerSolver(Solver):
    r"""
    Integrate diffusion ODEs with exponential Euler.

    This first-order solver integrates ODEs with an extended semi-linear
    right-hand side:

    .. math::
        \frac{d\mathbf{x}}{dt} = D(\mathbf{x}, t)
        = a(t) \, \mathbf{x} + b(t) \, N(\mathbf{x}, t)

    The ``denoiser`` provides the full right-hand side :math:`D`, while
    ``bias_fn`` and ``bias_int_fn`` provide the bias coefficient :math:`a(t)`
    and its antiderivative :math:`\mathcal{A}(t)`
    (:math:`\mathcal{A}'(t) = a(t)`), and ``slope_fn`` provides the slope
    coefficient :math:`b(t)`. A noise scheduler can provide these callables.

    Each step from :math:`t_n` to :math:`t_{n-1}` treats the linear dynamics
    exactly and freezes the nonlinear term:

    .. math::
        \mathbf{x}_{n-1} = e^{\mathcal{A}(t_{n-1}) - \mathcal{A}(t_n)}
        \, \mathbf{x}_n + J \, N(\mathbf{x}_n, t_n),
        \qquad
        J = \int_{t_n}^{t_{n-1}}
        e^{\mathcal{A}(t_{n-1}) - \mathcal{A}(s)} \, b(s) \, ds

    Exponential Euler requires one denoiser evaluation per step and can take
    advantage of known linear dynamics in the diffusion process. If you omit
    ``bias_fn``, ``bias_int_fn``, and ``slope_fn``, the solver uses explicit
    Euler.

    For linear-Gaussian probability-flow ODEs, this method corresponds to the
    first-order DPM-Solver. With the bias and slope callbacks from a noise
    scheduler, it reproduces DDIM, the standard sampler of distilled
    few-step models.

    The ``bias_fn``, ``bias_int_fn``, and ``slope_fn`` callables have the
    signatures:

    .. code-block:: python

        def bias_fn(
            t: Tensor,  # shape: (B,) or broadcastable
        ) -> Tensor: ...  # bias coefficient a(t), same shape as t

        def bias_int_fn(
            t: Tensor,  # shape: (B,) or broadcastable
        ) -> Tensor: ...  # antiderivative of a(t), same shape as t

        def slope_fn(
            t: Tensor,  # shape: (B,) or broadcastable
        ) -> Tensor: ...  # slope coefficient b(t), same shape as t

    Parameters
    ----------
    denoiser : Denoiser
        Right-hand side :math:`D(x, t)` of the ODE to integrate, following
        the :class:`~physicsnemo.diffusion.Denoiser` interface. Typically from
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`,
        but any callable with the correct signature works.
    bias_fn : Callable[[Tensor], Tensor] | None, optional
        Bias coefficient :math:`a(t)`, with the signature shown above.
        Requires ``bias_int_fn``. Typically from
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.get_linear_denoiser`
        with the same predictor parameterization as ``denoiser``; any
        callable with the correct signature works. The default is ``None``,
        which uses a zero bias.
    bias_int_fn : Callable[[Tensor], Tensor] | None, optional
        Antiderivative :math:`\mathcal{A}(t)` of the bias, with the signature
        shown above. Requires ``bias_fn``. The default is ``None``, which
        uses a zero bias.
    slope_fn : Callable[[Tensor], Tensor] | None, optional
        Slope coefficient :math:`b(t)` of the nonlinear term, with the
        signature shown above. The default is ``None``, which uses a constant
        slope (:math:`b = 1`) and treats the full nonlinear term as the frozen
        quantity.

    Note
    ----
    References:

    - `Denoising Diffusion Implicit Models
      <https://arxiv.org/abs/2010.02502>`_
    - `DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model
      Sampling <https://arxiv.org/abs/2206.00927>`_

    Examples
    --------
    Reproduce a DDIM-like sampler on an EDM schedule. Pair the scheduler's
    denoiser with its bias and slope callbacks (here for an x0-predictor) so
    that the solver integrates the linear dynamics exactly and only freezes
    the data prediction. This generalized DDIM is the standard sampler of
    distilled few-step models. The original DDIM paper used a VP schedule,
    but any linear-Gaussian schedule works the same way:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.samplers import ExponentialEulerSolver
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>> x0_pred = lambda x, t: x * 0.1  # Toy x0-predictor
    >>> bias, bias_int, slope = scheduler.get_linear_denoiser(
    ...     prediction_type="x0"
    ... )
    >>> solver = ExponentialEulerSolver(
    ...     scheduler.get_denoiser(x0_predictor=x0_pred),
    ...     bias_fn=bias,
    ...     bias_int_fn=bias_int,
    ...     slope_fn=slope,
    ... )
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> x_tm1 = solver.step(x_t, torch.tensor([5.0]), torch.tensor([2.5]))
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        denoiser: Denoiser,
        bias_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
        bias_int_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
        slope_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
    ) -> None:
        self.denoiser = denoiser
        if bias_fn is None and bias_int_fn is None:
            self.bias_fn = lambda t: torch.zeros_like(t)
            self.bias_int_fn = lambda t: torch.zeros_like(t)
        elif bias_fn is not None and bias_int_fn is not None:
            self.bias_fn = bias_fn
            self.bias_int_fn = bias_int_fn
        else:
            raise ValueError(
                "bias_fn and bias_int_fn must both be provided or both None."
            )
        if slope_fn is None:
            self.slope_fn = lambda t: torch.ones_like(t)
        else:
            self.slope_fn = slope_fn

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one exponential Euler integration step.

        Parameters
        ----------
        x : Tensor
            Current noisy latent state :math:`\mathbf{x}_{n}` of shape
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

        # Shape for broadcasting time-only quantities: (B,) -> (B, 1, ..., 1)
        expected_shape = (-1,) + (1,) * (x.ndim - 1)

        # Recover the predictor-like term of the RHS: N = (D - a x) / b; the
        # zero-slope guard drops the nonlinear term instead of dividing by
        # zero
        d_cur = self.denoiser(x, t_cur)
        a_bc = self.bias_fn(t_cur).reshape(expected_shape)
        b_bc = self.slope_fn(t_cur).reshape(expected_shape)
        b_safe = torch.where(b_bc == 0, torch.ones_like(b_bc), b_bc)
        inv_b_bc = torch.where(b_bc == 0, torch.zeros_like(b_bc), 1 / b_safe)
        n_cur = (d_cur - a_bc * x) * inv_b_bc

        # Exact linear propagator from the antiderivative of the bias, and
        # exponential-kernel weight of the nonlinear term
        e_bc = torch.exp(
            (self.bias_int_fn(t_next) - self.bias_int_fn(t_cur)).reshape(expected_shape)
        )
        j_bc = _nonlinear_weight(
            t_cur, t_next, self.bias_fn, self.bias_int_fn, self.slope_fn
        ).reshape(expected_shape)

        x_next = e_bc * x + j_bc * n_cur

        return x_next
