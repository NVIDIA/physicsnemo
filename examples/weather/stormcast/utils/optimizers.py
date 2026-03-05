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

from collections.abc import Iterable

import torch

try:
    from optimi import StableAdamW

    IS_OPTIMI_AVAILABLE = True
except ImportError:
    IS_OPTIMI_AVAILABLE = False

from utils.config import OptimizerConfig


def build_optimizer(params: Iterable, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    r"""
    Construct an optimizer from a config dict.

    Parameters
    ----------
    params : Iterable
        Model parameters to optimize (typically ``model.parameters()``).
    cfg : OptimizerConfig
        Optimizer configuration with keys:

        - ``name`` : str
            Optimizer type: "adam", "adamw", or "stableadamw".
        - ``lr`` : float, optional
            Learning rate (overrides default_lr if specified).
        - ``betas`` : list of float, optional
            Adam beta parameters [beta1, beta2], default [0.9, 0.999].
        - ``weight_decay`` : float, optional
            Weight decay (L2 regularization), default 0.0.
        - ``eps`` : float, optional
            Adam epsilon for numerical stability, default 1e-8.
        - ``fused`` : bool, optional
            Use fused CUDA kernel for better performance, default True.

    default_lr : float
        Default learning rate used if not specified in cfg.

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
    ...     default_lr=1e-4,
    ... )
    """
    name = cfg.name

    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            fused=cfg.fused,
        )
    elif name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            fused=cfg.fused,
            amsgrad=False,
        )
    elif name == "stableadamw":  # TODO: StableAdamW doesn't work with DTensor
        if not IS_OPTIMI_AVAILABLE:
            raise ImportError(
                "Selected 'stableadamw' optimizer but optimi is not installed."
            )
        return StableAdamW(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
    elif isinstance(name, tuple):
        # if you know what you're doing, you can pass the name and kwargs or any torch optimizer
        (name, opt_kwargs) = name
        opt_cls = getattr(torch.optim, name)
        return opt_cls(params, lr=cfg.lr, **opt_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer '{name}'")
