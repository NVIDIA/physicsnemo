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

"""DPM-Solver++(2M) with the UniC-2 corrector for diffusion ODEs."""

import math
from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser

from ._utils import gauss_legendre
from .base import Solver

# Number of Gauss-Legendre points used by the internal quadratures of this
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


def _nonlinear_moment(
    order: int,
    t_cur: Float[Tensor, " B"],
    t_next: Float[Tensor, " B"],
    lam_cur: Float[Tensor, " B"],
    bias_int_fn: Callable[[Tensor], Tensor],
    slope_fn: Callable[[Tensor], Tensor],
    lambda_fn: Callable[[Tensor], Tensor],
) -> Float[Tensor, " B"]:
    r"""
    Compute a propagator-weighted moment of the nonlinear term over one step:

    .. math::
        J_{k} = \int_{t_n}^{t_{n-1}}
        e^{\mathcal{A}(t_{n-1}) - \mathcal{A}(s)} \, b(s)
        \frac{[\lambda(s) - \lambda(t_n)]^k}{k!} \, ds

    The higher moments (:math:`k = 1, 2`) provide the derivative information
    that the UniC-2 corrector needs without differentiating the predictor.
    """
    bias_int_next = bias_int_fn(t_next)
    factorial = float(math.factorial(order))

    def integrand(s: Tensor) -> Tensor:
        kernel = torch.exp(bias_int_next - bias_int_fn(s))
        dz = lambda_fn(s) - lam_cur
        return kernel * slope_fn(s) * dz**order / factorial

    return gauss_legendre(integrand, t_cur, t_next, _NUM_QUADRATURE_POINTS)


class DPMPlusPlus2MUniC2(Solver):
    r"""
    Sample diffusion ODEs with DPM-Solver++(2M) and the UniC-2 corrector.

    This is a predictor-corrector scheme built on top of
    :class:`DPMPlusPlus2M`: a second-order DPM-Solver++(2M) predictor with a
    UniC-2 corrector stage that raises the update to third order while
    keeping one denoiser evaluation per step. The extra
    history the corrector reuses raises the memory footprint slightly
    compared with :class:`DPMPlusPlus2M`.

    The solver integrates ODEs with an extended semi-linear right-hand side:

    .. math::
        \frac{d\mathbf{x}}{dt} = D(\mathbf{x}, t)
        = a(t) \, \mathbf{x} + b(t) \, N(\mathbf{x}, t)

    The ``denoiser`` provides the full right-hand side :math:`D`;
    ``bias_fn`` and ``bias_int_fn`` provide the bias coefficient
    :math:`a(t)` and its antiderivative :math:`\mathcal{A}(t)`
    (:math:`\mathcal{A}'(t) = a(t)`); ``slope_fn`` provides the slope
    coefficient :math:`b(t)`. A noise scheduler can provide all these
    callables.

    Each step advances the state from the current time :math:`t_n` to the
    target time :math:`t_{n-1}` (sampling proceeds from large to small
    times). The corrected update applies the exact linear propagator
    :math:`E_n = e^{\mathcal{A}(t_{n-1}) - \mathcal{A}(t_n)}` to the state
    and combines the current predictor-like evaluation with the two most
    recent ones:

    .. math::
        \mathbf{x}_{n-1} = E_n \, \mathbf{x}_n + J_{0, n} N_n
        + C_{\mathrm{old}} \left( N_{n+1} - N_n \right)
        + C_{\mathrm{new}} \left( N_{n-1}^{P} - N_n \right)

    where :math:`N_{n-1}^{P}` is the predictor-like term evaluated at the
    predicted endpoint, :math:`J_{0, n}` weights the current evaluation
    over the step, and :math:`C_{\mathrm{old}}`, :math:`C_{\mathrm{new}}`
    are corrector coefficients determined by the numerical scheme from the
    step sizes measured in the extrapolation coordinate :math:`\lambda(t)`.

    ``lambda_fn`` sets the extrapolation coordinate :math:`\lambda(t)`. The
    original DPM-Solver++ methods use the schedule's log-SNR,
    :math:`\lambda(t) = \log(\alpha(t) / \sigma(t))`; the default
    :math:`\lambda(t) = t` extrapolates in diffusion time.

    The ``bias_fn``, ``bias_int_fn``, ``slope_fn``, and ``lambda_fn``
    callables have the signatures:

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

        def lambda_fn(
            t: Tensor,  # shape: (B,) or broadcastable
        ) -> Tensor: ...  # extrapolation coordinate lambda(t), same shape as t

    .. note::

        This solver is **stateful**: a single instance tracks a single
        batch of trajectories, caching two latent-shaped tensors. Call
        :meth:`reset` before reusing an instance on a new trajectory.

    Parameters
    ----------
    denoiser : Denoiser
        Right-hand side :math:`D(x, t)` of the ODE to integrate, following the
        :class:`~physicsnemo.diffusion.Denoiser` interface. Typically from
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`,
        but any callable with the correct signature works.
    bias_fn : Callable[[Tensor], Tensor] | None, optional
        Bias coefficient :math:`a(t)`, with the signature shown above.
        Requires ``bias_int_fn`` when provided. Typically from
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.get_linear_denoiser`
        with the same predictor parameterization as ``denoiser``; any
        callable with the correct signature works. The default is ``None``,
        which corresponds to a zero bias.
    bias_int_fn : Callable[[Tensor], Tensor] | None, optional
        Antiderivative :math:`\mathcal{A}(t)` of the bias, with the signature
        shown above. Requires ``bias_fn`` when provided. The default is
        ``None``, which corresponds to a zero bias.
    slope_fn : Callable[[Tensor], Tensor] | None, optional
        Slope coefficient :math:`b(t)` of the nonlinear term, with the
        signature shown above. The default is ``None``, which uses a constant
        slope (:math:`b = 1`).
    lambda_fn : Callable[[Tensor], Tensor] | None, optional
        Extrapolation coordinate :math:`\lambda(t)`, with the signature shown
        above. For the classical diffusion method, pass the schedule's
        log-SNR: ``lambda t: torch.log(scheduler.snr(t))``. The default is
        ``None``, which extrapolates in diffusion time.

    Note
    ----
    References:

    - `UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of
      Diffusion Models <https://arxiv.org/abs/2302.04867>`_
    - `DPM-Solver++: Fast Solver for Guided Sampling of Diffusion
      Probabilistic Models <https://arxiv.org/abs/2211.01095>`_

    Examples
    --------

    This class can express different predictor-corrector samplers through
    its callback configuration. The examples below add a UniC-2 corrector
    stage to two predictors: a classical two-step Adams-Bashforth scheme,
    and the original DPM-Solver++(2M).

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.samplers import DPMPlusPlus2MUniC2
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>> x0_pred = lambda x, t: x * 0.1  # Toy x0-predictor
    >>> # Both configurations use the same EDM probability-flow ODE.
    >>> denoiser = scheduler.get_denoiser(x0_predictor=x0_pred)
    >>> # Add a UniC-2 corrector stage to a two-step Adams-Bashforth predictor.
    >>> ab2_pc_solver = DPMPlusPlus2MUniC2(denoiser)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> x_mid = ab2_pc_solver.step(x_t, torch.tensor([5.0]), torch.tensor([2.5]))
    >>> x_next = ab2_pc_solver.step(x_mid, torch.tensor([2.5]), torch.tensor([1.0]))
    >>> x_next.shape
    torch.Size([1, 3, 8, 8])

    To add the same corrector stage to the original DPM-Solver++(2M)
    instead, provide the scheduler's bias and slope callbacks and use its
    log-SNR as the extrapolation coordinate:

    >>> # Separate the known affine structure from the denoiser output.
    >>> bias, bias_int, slope = scheduler.get_linear_denoiser(
    ...     prediction_type="x0"
    ... )
    >>> # Extrapolation variable is the schedule's log-SNR.
    >>> log_snr = lambda t: torch.log(scheduler.snr(t))
    >>> unic2_solver = DPMPlusPlus2MUniC2(
    ...     denoiser,
    ...     bias_fn=bias,
    ...     bias_int_fn=bias_int,
    ...     slope_fn=slope,
    ...     lambda_fn=log_snr,
    ... )
    >>> x_mid = unic2_solver.step(x_t, torch.tensor([5.0]), torch.tensor([2.5]))
    >>> x_next = unic2_solver.step(x_mid, torch.tensor([2.5]), torch.tensor([1.0]))
    >>> x_next.shape
    torch.Size([1, 3, 8, 8])
    >>> unic2_solver.reset()  # Before reusing it on a new trajectory
    """

    def __init__(
        self,
        denoiser: Denoiser,
        bias_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
        bias_int_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
        slope_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
        lambda_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
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
        if lambda_fn is None:
            self.lambda_fn = lambda t: t
        else:
            self.lambda_fn = lambda_fn
        self._n_cur: Tensor | None = None
        self._n_old: Tensor | None = None
        self._lam_old: Tensor | None = None

    def reset(self) -> None:
        r"""
        Clear the cached history from the previous trajectory.

        Call this method before starting a new trajectory with an existing
        solver instance. The next :meth:`step` call uses a first-order update
        because no previous predictor-like evaluation is available.

        Returns
        -------
        None
            This method updates the solver state in place.
        """
        self._n_cur = None
        self._n_old = None
        self._lam_old = None

    def _predictor_value(
        self,
        x: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
        expected_shape: tuple[int, ...],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Recover the predictor-like term of the RHS at ``(x, t)``:
        :math:`N = (D - a \mathbf{x}) / b`, with a zero-slope guard that
        drops the nonlinear term instead of dividing by zero.
        """
        d = self.denoiser(x, t)
        a_bc = self.bias_fn(t).reshape(expected_shape)
        b_bc = self.slope_fn(t).reshape(expected_shape)
        b_safe = torch.where(b_bc == 0, torch.ones_like(b_bc), b_bc)
        inv_b_bc = torch.where(b_bc == 0, torch.zeros_like(b_bc), 1 / b_safe)
        return (d - a_bc * x) * inv_b_bc

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Advance one step with DPM-Solver++(2M) and the UniC-2 corrector.

        Successive calls must belong to a single trajectory with consecutive
        time intervals (``t_cur`` equal to the previous call's ``t_next``);
        see :meth:`reset`.

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
        lam_cur = self.lambda_fn(t_cur)
        lam_cur_bc = lam_cur.reshape(expected_shape)
        lam_next_bc = self.lambda_fn(t_next).reshape(expected_shape)

        # The denoiser is generally singular at the terminal time t = 0, and
        # the final step cannot reuse its endpoint evaluation anyway, so
        # steps that land exactly at t = 0 skip the corrector and take the
        # endpoint value at the current time instead. This keeps the update
        # and its gradients finite on the last step of a trajectory.
        t_eval = torch.where(t_next == 0, t_cur, t_next)
        at_zero_bc = (t_next == 0).reshape(expected_shape)

        # Exact linear propagator from the antiderivative of the bias, and
        # exponential-kernel weight of the nonlinear term
        e_bc = torch.exp(
            (self.bias_int_fn(t_next) - self.bias_int_fn(t_cur)).reshape(expected_shape)
        )
        j0_bc = _nonlinear_weight(
            t_cur, t_next, self.bias_fn, self.bias_int_fn, self.slope_fn
        ).reshape(expected_shape)

        if self._n_cur is None or self._n_old is None or self._lam_old is None:
            # First step: exponential Euler, then seed the history with the
            # endpoint evaluation so that the corrector can start on the
            # second step with one new denoiser evaluation per step. Later
            # steps update the caches in place, keeping their storage stable
            # for torch.compile.
            n_cur = self._predictor_value(x, t_cur, expected_shape)
            x_next = e_bc * x + j0_bc * n_cur
            n_pred = self._predictor_value(x_next, t_eval, expected_shape)
            # Keep the current value where the endpoint evaluation is not
            # finite (for example the recovery of N at a vanishing noise
            # level)
            n_pred = torch.where(torch.isfinite(n_pred), n_pred, n_cur)
            self._n_old = n_cur.clone()
            self._lam_old = lam_cur_bc.clone()
            self._n_cur = n_pred.clone()
            return x_next

        # Steady state: the current predictor-like value is the endpoint
        # evaluation cached by the previous step
        n_cur = self._n_cur
        n_old = self._n_old

        # Signed increments of the extrapolation coordinate, masked to fall
        # back to first order on repeated or non-monotone nodes, on
        # non-finite lambda values (for example the log-SNR at t = 0), and
        # on the terminal time. Use finite dummy denominators because
        # torch.where evaluates both branches.
        h_new_bc = lam_next_bc - lam_cur_bc
        h_old_bc = lam_cur_bc - self._lam_old
        ok = (
            torch.isfinite(h_new_bc)
            & torch.isfinite(h_old_bc)
            & (h_old_bc * h_new_bc > 0)
            & ~at_zero_bc
        )
        h_new_safe = torch.where(ok, h_new_bc, torch.ones_like(h_new_bc))
        h_old_safe = torch.where(ok, h_old_bc, torch.ones_like(h_old_bc))
        q_half_bc = torch.where(
            ok, h_new_safe / h_old_safe / 2, torch.zeros_like(h_new_bc)
        )

        # DPM-Solver++(2M) predictor and its endpoint evaluation
        x_pred = e_bc * x + j0_bc * (n_cur + q_half_bc * (n_cur - n_old))
        n_pred = self._predictor_value(x_pred, t_eval, expected_shape)
        n_pred = torch.where(torch.isfinite(n_pred), n_pred, n_cur)

        # Corrector weights from the normalized propagator-weighted moments
        j1_bc = _nonlinear_moment(
            1, t_cur, t_next, lam_cur, self.bias_int_fn, self.slope_fn, self.lambda_fn
        ).reshape(expected_shape)
        j2_bc = _nonlinear_moment(
            2, t_cur, t_next, lam_cur, self.bias_int_fn, self.slope_fn, self.lambda_fn
        ).reshape(expected_shape)
        m1_bc = j1_bc / h_new_safe
        m2_bc = 2 * j2_bc / h_new_safe**2
        rho_bc = h_old_safe / h_new_safe
        c_old_bc = torch.where(
            ok, (m2_bc - m1_bc) / (rho_bc * (rho_bc + 1)), torch.zeros_like(m1_bc)
        )
        c_new_bc = torch.where(
            ok, (rho_bc * m1_bc + m2_bc) / (rho_bc + 1), torch.zeros_like(m1_bc)
        )

        x_next = (
            e_bc * x
            + j0_bc * n_cur
            + c_old_bc * (n_old - n_cur)
            + c_new_bc * (n_pred - n_cur)
        )
        self._n_old.copy_(n_cur)
        self._lam_old.copy_(lam_cur_bc)
        self._n_cur.copy_(n_pred)

        return x_next
