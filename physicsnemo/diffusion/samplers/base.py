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

"""Base protocol for diffusion ODE/SDE solvers."""

from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor


@runtime_checkable
class Solver(Protocol):
    r"""
    Protocol defining the interface for diffusion solvers.

    A solver implements a numerical method to integrate the diffusion process
    from a noisy state to a less noisy (or clean) state. Each call to
    :meth:`step` advances the state from time ``t_cur`` (:math:`t_n`) to
    ``t_next`` (:math:`t_{n-1}`).

    This is the minimal interface required for sampling from a diffusion model,
    and any object that implements this interface can be used as a solver in
    sampling utilities.

    The update rule applied by the sampler is roughly:

    .. math::
        \mathbf{x}_{n-1} = \text{Step}(F(\mathbf{x}_n, t_n); \mathbf{x}_n, t_n, t_{n-1})

    where :math:`F` is the denoiser (e.g. the right hand side in the case of
    ODE/SDE-based sampling, the denoised latent state in the case of discrete
    Markov chain-based sampling, etc.) and :math:`\text{Step}` is
    the update rule of the solver, implemented by the :meth:`step` method.

    See Also
    --------
    :func:`~physicsnemo.diffusion.samplers.sample` : The sampling function that
        uses solvers to generate samples.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.samplers import Solver
    >>>
    >>> class SimpleEuler:
    ...     def __init__(self, denoiser):
    ...         self.denoiser = denoiser
    ...     def step(self, x, t_cur, t_next):
    ...         d = (x - self.denoiser(x, t_cur)) / t_cur
    ...         return x + (t_next - t_cur) * d
    ...
    >>> denoiser = lambda x, t: x / (1 + t.view(-1, 1)**2)  # Toy denoiser
    >>> solver = SimpleEuler(denoiser)
    >>> isinstance(solver, Solver)
    True
    """

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one integration step from ``t_cur`` to ``t_next``.

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
        ...
