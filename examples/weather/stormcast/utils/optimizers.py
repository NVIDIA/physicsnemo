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

r"""
Optimizer utilities for StormCast training.

Provides a factory function to build optimizers from configuration.
"""

from __future__ import annotations
from typing import Iterable, Dict, Any
import torch
from optimi import StableAdamW


def build_optimizer(
    params: Iterable,
    cfg: Dict[str, Any],
    *,
    lr: float,
) -> torch.optim.Optimizer:
    r"""
    Construct an optimizer from a config dict.

    Parameters
    ----------
    params : Iterable
        Model parameters to optimize (typically ``model.parameters()``).
    cfg : dict
        Optimizer configuration dict with keys:

        - ``name`` : str
            Optimizer type: "adam", "adamw", or "stableadamw".
        - ``betas`` : list of float, optional
            Adam beta parameters [beta1, beta2], default [0.9, 0.999].
        - ``weight_decay`` : float, optional
            Weight decay (L2 regularization), default 0.0.
        - ``eps`` : float, optional
            Adam epsilon for numerical stability, default 1e-8.
        - ``fused`` : bool, optional
            Use fused CUDA kernel for better performance, default True.

    lr : float
        Learning rate for the optimizer.

    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer instance.

    Raises
    ------
    ValueError
        If an unsupported optimizer name is provided.

    Examples
    --------
    >>> optimizer = build_optimizer(
    ...     model.parameters(),
    ...     {"name": "adamw", "weight_decay": 0.01},
    ...     lr=1e-4,
    ... )
    """
    name = (cfg.get("name") or "adam").lower()
    betas = tuple(cfg.get("betas", [0.9, 0.999]))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    eps = float(cfg.get("eps", 1e-8))
    fused = bool(cfg.get("fused", True))

    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=fused,
        )
    elif name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=fused,
            amsgrad=False,
        )
    elif name == "stableadamw":
        return StableAdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer '{name}'")
