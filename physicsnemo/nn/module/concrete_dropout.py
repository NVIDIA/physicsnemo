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

r"""Concrete Dropout for learned per-layer dropout rates.

Implements the concrete relaxation of dropout (Gal, Hron & Kendall, 2017) which
learns the optimal dropout probability for each layer during training. This enables
Monte Carlo dropout uncertainty quantification without manual tuning of
per-layer dropout rates.

References
----------
.. [1] Y. Gal, J. Hron, A. Kendall, "Concrete Dropout", NeurIPS 2017.
   https://arxiv.org/abs/1705.07832
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConcreteDropout(nn.Module):
    r"""Concrete Dropout layer with a learnable dropout probability.

    Uses the concrete (Gumbel-softmax) relaxation to make the dropout mask
    differentiable with respect to the dropout probability ``p``. During
    training, ``p`` is optimized jointly with the model parameters via a
    regularization loss that balances data fit against model complexity.

    Parameters
    ----------
    in_features : int
        Number of input features (used for logging in ``extra_repr``).
    init_p : float, optional
        Initial dropout probability. Default is ``0.1``.
    temperature : float, optional
        Temperature for the concrete relaxation. Lower values produce
        masks closer to binary. Default is ``0.1``.

    Forward
    -------
    x : torch.Tensor
        Input tensor of any shape.

    Outputs
    -------
    torch.Tensor
        Tensor with concrete dropout applied (same shape as input).

    Notes
    -----
    The regularization loss from :meth:`regularization_loss` must be added
    to the training objective for the dropout rate to be learned properly.
    Use :func:`collect_concrete_dropout_losses` to gather losses from all
    ``ConcreteDropout`` modules in a model.

    During inference with ``model.train()`` (MC-dropout mode), the concrete
    relaxation produces stochastic masks for uncertainty quantification.
    During ``model.eval()``, dropout is disabled and the layer is a no-op,
    identical to standard ``nn.Dropout`` behavior.

    Examples
    --------
    >>> import torch
    >>> cd = ConcreteDropout(in_features=256)
    >>> x = torch.randn(2, 100, 256)
    >>> out = cd(x)
    >>> out.shape
    torch.Size([2, 100, 256])
    >>> cd.regularization_loss()
    tensor(..., grad_fn=<AddBackward0>)
    """

    def __init__(
        self,
        in_features: int,
        init_p: float = 0.1,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()

        self.temperature = temperature

        # Learnable dropout logit, initialized to match init_p
        # sigmoid(p_logit) = init_p  =>  p_logit = log(p / (1-p))
        init_p = max(min(init_p, 0.99), 0.01)  # clamp for numerical safety
        p_logit = torch.log(torch.tensor(init_p / (1.0 - init_p)))
        self.p_logit = nn.Parameter(p_logit)

        self._in_features = in_features

    @property
    def p(self) -> torch.Tensor:
        """Current dropout probability (always in (0, 1))."""
        return torch.sigmoid(self.p_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply concrete dropout to the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            Input with concrete dropout mask applied during training,
            or unchanged input during evaluation.
        """
        if not self.training:
            return x

        # Clamp logit to prevent sigmoid saturation to exact 0 or 1.
        # sigmoid(±10) ≈ 4.5e-5 / 0.99995, keeping p strictly in (0, 1)
        # with non-zero sigmoid gradient so recovery remains possible.
        self.p_logit.data.clamp_(-10.0, 10.0)

        p = self.p

        # Concrete relaxation: differentiable approximation to Bernoulli
        eps = 1e-7
        u = torch.rand_like(x).clamp(eps, 1.0 - eps)

        # Compute concrete (binary concrete) random variable
        # This is the sigmoid of (logit(u) + logit(p)) / temperature
        # Use p_logit directly instead of log(p) - log(1-p) to avoid
        # numerical instability when sigmoid saturates near 0 or 1.
        drop_prob = (
            torch.log(u) - torch.log(1.0 - u) + self.p_logit
        ) / self.temperature
        mask = torch.sigmoid(drop_prob)

        # Apply mask with inverted dropout scaling
        # mask values near 1 = drop, near 0 = keep
        return x * (1.0 - mask) / (1.0 - p + eps)

    def regularization_loss(self) -> torch.Tensor:
        r"""Compute the regularization loss for this layer.

        The loss is the negative entropy of a Bernoulli(p) distribution,
        which encourages the learned probability away from trivial solutions
        (0 or 1).

        Returns
        -------
        torch.Tensor
            Scalar regularization loss.
        """
        p = self.p
        eps = 1e-7

        # Entropy of Bernoulli(p) -- negative because we minimize
        dropout_entropy = p * torch.log(p + eps) + (1.0 - p) * torch.log(1.0 - p + eps)

        return dropout_entropy

    def extra_repr(self) -> str:
        return f"in_features={self._in_features}, p={self.p.item():.4f}"


def collect_concrete_dropout_losses(model: nn.Module) -> torch.Tensor:
    r"""Collect regularization losses from all ConcreteDropout modules.

    Parameters
    ----------
    model : nn.Module
        The model containing ConcreteDropout layers.

    Returns
    -------
    torch.Tensor
        Sum of all ConcreteDropout regularization losses. Returns
        ``torch.tensor(0.0)`` if no ConcreteDropout modules are found.

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> model = nn.Sequential(ConcreteDropout(in_features=8), nn.Linear(8, 1))
    >>> reg_loss = collect_concrete_dropout_losses(model)
    >>> reg_loss.shape
    torch.Size([])
    """
    reg_loss = torch.tensor(0.0)
    for module in model.modules():
        if isinstance(module, ConcreteDropout):
            loss = module.regularization_loss()
            # Ensure device compatibility
            if reg_loss.device != loss.device:
                reg_loss = reg_loss.to(loss.device)
            reg_loss = reg_loss + loss
    return reg_loss


def get_concrete_dropout_rates(model: nn.Module) -> dict[str, float]:
    r"""Extract learned dropout rates from all ConcreteDropout modules.

    Useful for monitoring and logging during training.

    Parameters
    ----------
    model : nn.Module
        The model containing ConcreteDropout layers.

    Returns
    -------
    dict[str, float]
        Dictionary mapping module names to their learned dropout
        probabilities.

    Examples
    --------
    >>> import torch.nn as nn
    >>> model = nn.Sequential(ConcreteDropout(in_features=8), nn.Linear(8, 1))
    >>> rates = get_concrete_dropout_rates(model)
    >>> len(rates) == 1
    True
    """
    rates = {}
    for name, module in model.named_modules():
        if isinstance(module, ConcreteDropout):
            rates[name] = module.p.item()
    return rates
