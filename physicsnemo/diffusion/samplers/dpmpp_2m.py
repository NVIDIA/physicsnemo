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

from .base import Solver


class DPMPlusPlus2M(Solver):
    r"""
    Sample diffusion ODEs with DPM-Solver++(2M).

    This solver offers second-order sampling with one denoiser evaluation per
    step. Use it as a computationally efficient alternative to
    :class:`HeunSolver`, which requires two evaluations per step.

    The solver integrates ODEs that separate the right-hand side into a term
    that is linear in the state and a remaining nonlinear term:

    .. math::
        \frac{d\mathbf{x}}{dt} = D(\mathbf{x}, t)
        = A(t) \, \mathbf{x} + N(\mathbf{x}, t)

    The ``denoiser`` provides the full right-hand side :math:`D`, while
    ``linear_fn`` provides :math:`A(t)`. The solver then derives the nonlinear
    term :math:`N` automatically. A noise scheduler can provide both callables.

    The ``linear_fn`` callable has the signature:

    .. code-block:: python

        def linear_fn(
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # linear coefficient A(t), shape: (B,)

    .. note::

        This solver is **stateful**: it caches the previous denoiser
        evaluation across calls to :meth:`step`, so a single instance tracks
        a single trajectory. Call :meth:`reset` before reusing an instance on
        a new trajectory. String-key selection in
        :func:`~physicsnemo.diffusion.samplers.sample` constructs a fresh
        instance for each call, which is always safe.

    Parameters
    ----------
    denoiser : Denoiser
        Right-hand side of the ODE to integrate, following the
        :class:`~physicsnemo.diffusion.Denoiser` interface. In most workflows,
        get it from
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`,
        but any callable with the correct signature works.
    linear_fn : Callable[[Tensor], Tensor] | None, optional
        Linear coefficient :math:`A(t)`. Use the signature above. In most
        workflows, get it from
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.get_linear_denoiser`,
        using the same predictor parameterization as ``denoiser``. The default
        is ``None``, which corresponds to a zero linear coefficient (classical
        two-step Adams-Bashforth method).

    Note
    ----
    Reference: `DPM-Solver++: Fast Solver for Guided Sampling of Diffusion
    Probabilistic Models <https://arxiv.org/abs/2211.01095>`_

    Examples
    --------
    Sample the probability-flow ODE of an EDM schedule:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.samplers import DPMPlusPlus2M
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>> x0_pred = lambda x, t: x * 0.1  # Toy x0-predictor
    >>> solver = DPMPlusPlus2M(
    ...     scheduler.get_denoiser(x0_predictor=x0_pred),
    ...     linear_fn=scheduler.get_linear_denoiser(prediction_type="x0"),
    ... )
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> x_1 = solver.step(x_t, torch.tensor([5.0]), torch.tensor([2.5]))
    >>> x_0 = solver.step(x_1, torch.tensor([2.5]), torch.tensor([0.0]))
    >>> x_0.shape
    torch.Size([1, 3, 8, 8])
    >>> solver.reset()  # Before reusing the instance on a new trajectory
    """

    def __init__(
        self,
        denoiser: Denoiser,
        linear_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
    ) -> None:
        self.denoiser = denoiser
        if linear_fn is None:
            self.linear_fn = lambda t: torch.zeros_like(t)
        else:
            self.linear_fn = linear_fn
        self._n_prev: Tensor | None = None
        self._t_prev: Tensor | None = None

    def reset(self) -> None:
        r"""
        Clear the cached history from the previous trajectory.

        Call this method before starting a new trajectory with an existing
        solver instance. The next :meth:`step` call uses a first-order update
        because no previous denoiser evaluation is available.

        Returns
        -------
        None
            This method updates the solver state in place.
        """
        self._n_prev = None
        self._t_prev = None

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
        t_cur_bc = t_cur.reshape(expected_shape)

        h_bc = (t_next - t_cur).reshape(expected_shape)

        # Isolate the nonlinear part of the RHS: N = D - A x
        d_cur = self.denoiser(x, t_cur)
        a_bc = self.linear_fn(t_cur).reshape(expected_shape)
        n_cur = d_cur - a_bc * x

        # h * phi1(h A) = expm1(h A) / A; the A -> 0 limit equals h
        z = h_bc * a_bc
        a_safe = torch.where(a_bc == 0, torch.ones_like(a_bc), a_bc)
        h_phi1 = torch.where(a_bc == 0, h_bc, torch.expm1(z) / a_safe)

        if self._n_prev is None or self._t_prev is None:
            # No history yet: first-order exponential Euler step
            x_next = torch.exp(z) * x + h_phi1 * n_cur
            # Build the history caches on the first step; later steps update
            # them in place for torch.compile compatibility
            self._n_prev = n_cur.clone()
            self._t_prev = t_cur_bc.clone()
        else:
            # Extrapolation ratio of successive steps, masked to fall back to
            # first order on repeated nodes. Use finite dummy denominators
            # because torch.where evaluates both branches.
            den_bc = t_cur_bc - self._t_prev
            ok = torch.isfinite(h_bc) & torch.isfinite(den_bc) & (den_bc != 0)
            r_safe = torch.where(ok, h_bc, torch.zeros_like(h_bc)) / torch.where(
                ok, den_bc, torch.ones_like(den_bc)
            )
            # DPM-Solver++ approximates the first exponential moment of the
            # slope term by h * phi1(h A) / 2
            slope_bc = torch.where(ok, h_phi1 / 2 * r_safe, torch.zeros_like(h_bc))
            x_next = (
                torch.exp(z) * x + h_phi1 * n_cur + slope_bc * (n_cur - self._n_prev)
            )
            self._n_prev.copy_(n_cur)
            self._t_prev.copy_(t_cur_bc)

        return x_next
