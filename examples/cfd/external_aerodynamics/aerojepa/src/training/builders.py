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

r"""Hydra-friendly builders for the AeroJEPA training stack.

These helpers keep ``train.py`` short: they delegate optimizer
construction to ``hydra.utils.instantiate`` and provide the small
linear-warmup helper used to ramp the JEPA loss weights at the start
of training.
"""

from __future__ import annotations

from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


def build_optimizer(
    model: torch.nn.Module,
    optimizer_cfg: DictConfig | dict[str, Any],
    *,
    extra_params: list[torch.nn.Parameter] | None = None,
) -> torch.optim.Optimizer:
    r"""Instantiate an optimizer from a Hydra config.

    Calls :func:`hydra.utils.instantiate` on ``optimizer_cfg`` with
    ``params=model.parameters() (+ extra_params)`` injected at call
    time. The config is expected to carry a ``_target_`` pointing at
    an optimizer class (e.g. ``torch.optim.AdamW``) and the optimizer's
    own kwargs (``lr``, ``weight_decay``, ``betas`` …).

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters are optimised.
    optimizer_cfg : DictConfig or dict
        Hydra group with ``_target_`` and optimizer kwargs.
    extra_params : list of torch.nn.Parameter, optional
        Additional parameter tensors to optimise alongside the model
        (used by prototype / anchor heads that live outside the main
        module). Default ``None``.

    Returns
    -------
    torch.optim.Optimizer
        The instantiated optimizer.
    """
    params = list(model.parameters())
    if extra_params:
        params.extend(extra_params)
    return instantiate(optimizer_cfg, params=params)


def linear_warmup_weight(
    target_weight: float,
    warmup_epochs: float,
    current_epoch: float,
) -> float:
    r"""Linearly ramp a loss weight from ``0`` to ``target_weight``.

    The JEPA training loss combines several terms (reconstruction,
    latent MSE / cosine, SIGReg). The non-reconstruction terms benefit
    from a short linear warmup so the predictor first learns to match
    the target latents at small scale before the regulariser kicks in.

    Parameters
    ----------
    target_weight : float
        Final weight after warmup completes.
    warmup_epochs : float
        Length of the warmup ramp in epochs. ``<= 0`` means no warmup
        (returns ``target_weight`` immediately).
    current_epoch : float
        Current epoch counter (fractional epochs are fine).

    Returns
    -------
    float
        Weight at ``current_epoch``.
    """
    if float(warmup_epochs) <= 0.0:
        return float(target_weight)
    progress = max(0.0, min(1.0, float(current_epoch) / float(warmup_epochs)))
    return float(target_weight) * progress
