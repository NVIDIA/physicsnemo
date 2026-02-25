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

import torch
import torch.nn as nn
from jaxtyping import Float


def gumbel_softmax(
    logits: Float[torch.Tensor, "... num_categories"],
    tau: torch.Tensor | float = 1.0,
) -> Float[torch.Tensor, "... num_categories"]:
    r"""
    Implementation of Gumbel Softmax from Transolver++.

    Applies a differentiable approximation to sampling from a categorical
    distribution using the Gumbel-Softmax trick.

    Original code: https://github.com/thuml/Transolver_plus/blob/main/models/Transolver_plus.py#L69

    Parameters
    ----------
    logits : torch.Tensor
        Input logits tensor of shape :math:`(*, K)` where :math:`K` is the
        number of categories.
    tau : torch.Tensor | float, optional, default=1.0
        Temperature parameter. Lower values make the distribution more
        concentrated.

    Returns
    -------
    torch.Tensor
        Gumbel-Softmax output of the same shape as ``logits``.
    """
    # Sample Gumbel noise
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)

    # Add noise and apply temperature-scaled softmax
    y = logits + gumbel_noise
    y = y / tau
    y = torch.nn.functional.softmax(y, dim=-1)

    return y


class GumbelSoftmax(nn.Module):
    r"""Gumbel-Softmax module for differentiable categorical sampling.

    This module wraps the :func:`gumbel_softmax` function as an ``nn.Module``,
    allowing it to be used as a layer in neural network architectures.

    The Gumbel-Softmax trick provides a differentiable approximation to sampling
    from a categorical distribution, enabling end-to-end training of models with
    discrete latent variables.

    Parameters
    ----------
    tau : float, optional, default=1.0
        Initial temperature parameter. Lower values make the distribution more
        concentrated (closer to one-hot). Can be modified after initialization.
    learnable : bool, optional, default=False
        If ``True``, the temperature parameter is registered as a learnable
        ``nn.Parameter``. If ``False``, it is a fixed buffer.

    Examples
    --------
    >>> import torch
    >>> gs = GumbelSoftmax(tau=0.5)
    >>> logits = torch.randn(2, 10)  # batch_size=2, num_categories=10
    >>> probs = gs(logits)
    >>> probs.shape
    torch.Size([2, 10])
    >>> torch.allclose(probs.sum(dim=-1), torch.ones(2))  # Each row sums to 1
    True

    >>> # With learnable temperature
    >>> gs_learnable = GumbelSoftmax(tau=1.0, learnable=True)
    >>> gs_learnable.tau.requires_grad
    True

    See Also
    --------
    :func:`gumbel_softmax` : Functional implementation of Gumbel-Softmax.
    """

    def __init__(self, tau: float = 1.0, learnable: bool = False):
        super().__init__()
        if learnable:
            self.tau = nn.Parameter(torch.tensor(tau))
        else:
            self.register_buffer("tau", torch.tensor(tau))

    def forward(
        self, logits: Float[torch.Tensor, "... num_categories"]
    ) -> Float[torch.Tensor, "... num_categories"]:
        r"""Apply Gumbel-Softmax to input logits.

        Parameters
        ----------
        logits : torch.Tensor
            Input logits tensor of shape :math:`(*, K)` where :math:`K` is the
            number of categories.

        Returns
        -------
        torch.Tensor
            Gumbel-Softmax output of the same shape as ``logits``.
        """
        return gumbel_softmax(logits, tau=self.tau)
