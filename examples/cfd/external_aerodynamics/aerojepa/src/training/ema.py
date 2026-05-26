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

r"""Exponential moving average bookkeeping for model parameters."""

from __future__ import annotations

import torch


class ExponentialMovingAverage:
    r"""Exponential moving average over a model's ``state_dict``.

    Maintains a shadow copy of every tensor in ``model.state_dict()`` and
    updates it with the rule
    ``shadow = decay * shadow + (1 - decay) * model_value``. Call
    :meth:`apply_to` before evaluation to swap the EMA weights into the
    model, and :meth:`restore` afterwards to put the live weights back.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose ``state_dict`` is shadowed.
    decay : float, optional
        Smoothing factor in ``[0, 1)``. Higher = slower averaging.
        Default ``0.999``.

    Examples
    --------
    >>> import torch
    >>> from src.training.ema import ExponentialMovingAverage
    >>> model = torch.nn.Linear(4, 4)
    >>> ema = ExponentialMovingAverage(model, decay=0.99)
    >>> ema.update(model)
    >>> ema.apply_to(model)
    >>> ema.restore(model)
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError(f"decay must lie in [0, 1); got {decay!r}")
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        r"""Pull one update step from ``model`` into the shadow weights."""
        state = model.state_dict()
        for k, v in state.items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        r"""Swap the model's weights for the EMA shadow (saving the live weights)."""
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        r"""Restore the live weights saved by :meth:`apply_to`."""
        if self.backup:
            model.load_state_dict(self.backup, strict=True)
            self.backup = {}
