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

"""First-order Euler solver for diffusion ODEs."""

from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser

from .base import Solver


class EulerSolver(Solver):
    r"""
    First-order Euler solver for diffusion ODEs.

    This is a fast solver with one denoiser evaluation per step, but typically
    produces lower quality samples compared to higher-order methods.

    Parameters
    ----------
    denoiser : Denoiser
        A callable implementing the
        :class:`~physicsnemo.diffusion.Denoiser` interface. Here it is
        expected to return the right hand side of the ODE. Typically obtained
        via
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`,
        but any callable with the correct signature can be used.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.samplers import EulerSolver
    >>>
    >>> denoiser = lambda x, t: x / (1 + t.view(-1, 1, 1, 1)**2)  # Toy denoiser
    >>> solver = EulerSolver(denoiser)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> t_cur = torch.tensor([1.0])
    >>> t_next = torch.tensor([0.5])
    >>> x_tm1 = solver.step(x_t, t_cur, t_next)
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])
    >>> isinstance(solver, Solver)
    True
    """

    def __init__(self, denoiser: Denoiser) -> None:
        self.denoiser = denoiser

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one Euler integration step.

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

        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        expected_shape = (-1,) + (1,) * (x.ndim - 1)
        t_cur_bc = t_cur.reshape(expected_shape)
        t_next_bc = t_next.reshape(expected_shape)

        # RHS evaluation and step update
        d_cur = self.denoiser(x, t_cur)
        x_next = x + (t_next_bc - t_cur_bc) * d_cur

        return x_next
