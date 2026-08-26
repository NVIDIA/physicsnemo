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

"""Second-order Heun solver for diffusion ODEs."""

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser

from .base import Solver


class HeunSolver(Solver):
    r"""
    Second-order Heun solver for diffusion ODEs.

    This method requires two denoiser evaluations per step but usually produces
    higher quality samples than :class:`EulerSolver`.

    Parameters
    ----------
    denoiser : Denoiser
        A callable implementing the
        :class:`~physicsnemo.diffusion.Denoiser` interface. Here it is
        expected to return the right hand side of the ODE. Typically obtained
        via
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`,
        but any callable with the correct signature can be used.
    alpha : float, optional
        Interpolation parameter for the corrector step, must be in (0, 1].
        ``alpha=1`` gives the standard Heun method (trapezoidal rule),
        ``alpha=0.5`` gives the midpoint method. By default 1.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.samplers import HeunSolver
    >>>
    >>> denoiser = lambda x, t: x / (1 + t.view(-1, 1, 1, 1)**2)  # Toy denoiser
    >>> solver = HeunSolver(denoiser)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> t_cur = torch.tensor([1.0])
    >>> t_next = torch.tensor([0.5])
    >>> x_tm1 = solver.step(x_t, t_cur, t_next)
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        denoiser: Denoiser,
        alpha: float = 1.0,
    ) -> None:
        self.denoiser = denoiser
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one Heun integration step.

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
        # Ensure contiguous strides so that both denoiser calls (at t_cur
        # and at the intermediate t_prime) present the same stride layout
        # to torch.compile, avoiding spurious recompilations.
        t_cur = t_cur.contiguous()
        t_next = t_next.contiguous()

        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        expected_shape = (-1,) + (1,) * (x.ndim - 1)
        t_cur_bc = t_cur.reshape(expected_shape)
        t_next_bc = t_next.reshape(expected_shape)

        h_bc = t_next_bc - t_cur_bc

        # First RHS evaluation
        d_cur = self.denoiser(x, t_cur)

        # Predictor step to intermediate point
        t_prime_bc = t_cur_bc + self.alpha * h_bc
        x_prime = x + self.alpha * h_bc * d_cur

        # Mask for elements where t_next != 0 (need 2nd order correction)
        # Shape: (B, 1, ..., 1) for broadcasting
        mask_bc = (t_next_bc != 0).float()

        # Second RHS evaluation (compute everywhere, masked later)
        # Avoid division by zero in denoiser by using t_cur where t_prime is 0
        t_prime = t_prime_bc.reshape(x.shape[0])
        t_prime_safe = torch.where(t_prime == 0, t_cur, t_prime)
        d_prime = self.denoiser(x_prime, t_prime_safe)

        # Apply 2nd order correction only where t_next != 0
        # Where t_next == 0, use first-order Euler step
        w_cur = 1 - 1 / (2 * self.alpha)
        w_prime = 1 / (2 * self.alpha)
        x_euler = x + h_bc * d_cur
        x_heun = x + h_bc * (w_cur * d_cur + w_prime * d_prime)
        x_next = mask_bc * x_heun + (1 - mask_bc) * x_euler

        return x_next
