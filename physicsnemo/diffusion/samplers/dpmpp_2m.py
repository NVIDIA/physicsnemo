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

"""DPM-Solver++(2M) multistep sampler for diffusion ODEs."""

from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser

from ._utils import _nonlinear_weight
from .base import Solver


class DPMPlusPlus2M(Solver):
    r"""
    Sample diffusion ODEs with DPM-Solver++(2M).

    This solver offers second-order sampling with one denoiser evaluation per
    step. Use it as a computationally efficient alternative to
    :class:`HeunSolver`, which requires two evaluations per step.

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
    target time :math:`t_{n-1}`, with :math:`t_{n+1}` denoting the time of
    the previous step (sampling proceeds from large to small times):

    .. math::
        \mathbf{x}_{n-1} = E_n \, \mathbf{x}_n + J_n \left[ N_n
        + \frac{1}{2}
        \frac{\lambda(t_{n-1}) - \lambda(t_n)}{\lambda(t_n) - \lambda(t_{n+1})}
        \left( N_n - N_{n+1} \right) \right]

    where :math:`E_n = e^{\mathcal{A}(t_{n-1}) - \mathcal{A}(t_n)}` is the
    exact linear propagator and

    .. math::
        J_n = \int_{t_n}^{t_{n-1}} e^{\mathcal{A}(t_{n-1}) - \mathcal{A}(s)}
        \, b(s) \, ds

    is the weight of the nonlinear term.

    The function :math:`\lambda(t)`, provided by ``lambda_fn``, is the
    extrapolation coordinate: the solver extrapolates
    :math:`N` linearly in :math:`\lambda(t)` rather than in :math:`t`.
    The original DPM-Solver++(2M) measures the extrapolation in the log-SNR
    :math:`\lambda(t) = \log(\alpha(t) / \sigma(t))` of the noise schedule.
    The default :math:`\lambda(t) = t` extrapolates in diffusion time.

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
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # extrapolation coordinate lambda(t), shape: (B,)

    .. note::

        This solver is **stateful**: it caches the previous predictor-like
        evaluation across calls to :meth:`step`, so a single instance tracks
        a single trajectory. Call :meth:`reset` before reusing an instance on
        a new trajectory. String-key selection in
        :func:`~physicsnemo.diffusion.samplers.sample` constructs a fresh
        instance for each call, which is always safe.

    Parameters
    ----------
    denoiser : Denoiser
        Right-hand side :math:`D(x, t)` of the ODE to integrate, following
        the :class:`~physicsnemo.diffusion.Denoiser` interface. Typically from
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
        slope (:math:`b = 1`). A zero value disables the nonlinear term; use it
        only when the denoiser is fully described by the linear bias.
    lambda_fn : Callable[[Tensor], Tensor] | None, optional
        Extrapolation coordinate :math:`\lambda(t)`, with the signature shown
        above. For DPM-Solver++(2M), pass the schedule's log-SNR:
        ``lambda t: torch.log(scheduler.snr(t))``. The default is ``None``,
        which extrapolates in diffusion time.

    Note
    ----
    Reference: `DPM-Solver++: Fast Solver for Guided Sampling of Diffusion
    Probabilistic Models <https://arxiv.org/abs/2211.01095>`_

    Examples
    --------

    This class can express different multistep samplers through its callback
    configuration. The examples below show a classical Adams-Bashforth solver
    and the original DPM-Solver++(2M) method.

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.samplers import DPMPlusPlus2M
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>> x0_pred = lambda x, t: x * 0.1  # Toy x0-predictor
    >>> # Both configurations use the same EDM probability-flow ODE.
    >>> denoiser = scheduler.get_denoiser(x0_predictor=x0_pred)
    >>> # The default callbacks recover classical two-step Adams-Bashforth.
    >>> ab2_solver = DPMPlusPlus2M(denoiser)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> x_mid = ab2_solver.step(x_t, torch.tensor([5.0]), torch.tensor([2.5]))
    >>> x_next = ab2_solver.step(x_mid, torch.tensor([2.5]), torch.tensor([1.0]))
    >>> x_next.shape
    torch.Size([1, 3, 8, 8])

    To use DPM-Solver++(2M) instead, provide the scheduler's bias and slope
    callbacks and use its log-SNR as the extrapolation coordinate:

    >>> # Separate the known affine structure from the denoiser output.
    >>> bias, bias_int, slope = scheduler.get_linear_denoiser(
    ...     prediction_type="x0"
    ... )
    >>> # Extrapolation variable is the schedule's log-SNR.
    >>> log_snr = lambda t: torch.log(scheduler.snr(t))
    >>> dpmpp_solver = DPMPlusPlus2M(
    ...     denoiser,
    ...     bias_fn=bias,
    ...     bias_int_fn=bias_int,
    ...     slope_fn=slope,
    ...     lambda_fn=log_snr,
    ... )
    >>> x_mid = dpmpp_solver.step(x_t, torch.tensor([5.0]), torch.tensor([2.5]))
    >>> x_next = dpmpp_solver.step(x_mid, torch.tensor([2.5]), torch.tensor([1.0]))
    >>> x_next.shape
    torch.Size([1, 3, 8, 8])
    >>> dpmpp_solver.reset()  # Before reusing it on a new trajectory
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
        self._n_prev: Tensor | None = None
        self._lam_prev: Tensor | None = None

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
        self._n_prev = None
        self._lam_prev = None

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Advance one step with DPM-Solver++(2M).

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
        lam_cur_bc = self.lambda_fn(t_cur).reshape(expected_shape)

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

        if self._n_prev is None or self._lam_prev is None:
            # Seed placeholder history on the first step: the equal previous
            # lambda zeroes the masked extrapolation below, so the update
            # degenerates to a first-order exponential Euler step.
            n_prev = n_cur
            lam_prev = lam_cur_bc
        else:
            n_prev = self._n_prev
            lam_prev = self._lam_prev

        # Extrapolation ratio of successive steps, measured in the lambda
        # coordinate and masked to fall back to first order on repeated
        # nodes or non-finite lambda values (for example the log-SNR at
        # t = 0). Use finite dummy denominators because torch.where
        # evaluates both branches.
        num_bc = self.lambda_fn(t_next).reshape(expected_shape) - lam_cur_bc
        den_bc = lam_cur_bc - lam_prev
        ok = torch.isfinite(num_bc) & torch.isfinite(den_bc) & (den_bc != 0)
        r_safe = torch.where(ok, num_bc, torch.zeros_like(num_bc)) / torch.where(
            ok, den_bc, torch.ones_like(den_bc)
        )
        q_half_bc = torch.where(ok, r_safe / 2, torch.zeros_like(r_safe))

        x_next = e_bc * x + j_bc * (n_cur + q_half_bc * (n_cur - n_prev))

        # Rebind immutable history instead of mutating graph-connected
        # tensors in place. Clone only the batch-sized lambda value so a
        # callback returning its input cannot alias caller-owned storage.
        self._n_prev = n_cur
        self._lam_prev = lam_cur_bc.clone()

        return x_next
