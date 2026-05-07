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

"""Shared utilities for the unified training recipe."""

from __future__ import annotations

import random
from typing import Literal, TypeAlias

import numpy as np
import torch
from omegaconf import DictConfig

from physicsnemo.optim import CombinedOptimizer

### Recipe-wide type aliases. Re-exported for use in loss.py, metrics.py,
### output_normalize.py, forward_kwargs.py, collate.py, train.py, and the
### tests so that ``target_config`` values share a single source of truth.
FieldType: TypeAlias = Literal["scalar", "vector"]


def set_seed(seed: int | None, rank: int = 0) -> None:
    """Pin all RNG states for reproducible training.

    When *seed* is not None, seeds Python, NumPy, and PyTorch (CPU + all
    CUDA devices) with ``seed + rank`` so that different ranks diverge
    deterministically.  When *seed* is None this function is a no-op,
    preserving the current (non-deterministic) behaviour.
    """
    if seed is None:
        return
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed % (1 << 31))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_muon_optimizer(
    model: torch.nn.Module, cfg: DictConfig, *, compile_optimizer: bool = False
) -> torch.optim.Optimizer:
    """Build Muon + AdamW combined optimizer.

    Muon handles 2-D parameters (linear/attention weight matrices) while AdamW
    handles everything else (biases, layer-norm, embeddings, etc.).

    Args:
        model: The model (may be DDP-wrapped).
        cfg: Full Hydra config. Reads ``cfg.training.optimizer.*`` for lr,
            weight_decay, betas, and eps.
        compile_optimizer: If True, compile the optimizer step functions
            with ``torch.compile``.
    """
    base_model = model.module if hasattr(model, "module") else model
    muon_params = [p for p in base_model.parameters() if p.ndim == 2]
    other_params = [p for p in base_model.parameters() if p.ndim != 2]

    opt_cfg = cfg.training.optimizer
    lr = opt_cfg.lr
    weight_decay = opt_cfg.get("weight_decay", 1e-4)
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = opt_cfg.get("eps", 1e-8)

    compile_kwargs = {} if compile_optimizer else None

    if muon_params and other_params:
        return CombinedOptimizer(
            [
                torch.optim.Muon(
                    muon_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    adjust_lr_fn="match_rms_adamw",
                ),
                torch.optim.AdamW(
                    other_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=betas,
                    eps=eps,
                ),
            ],
            torch_compile_kwargs=compile_kwargs,
        )
    elif muon_params:
        opt = torch.optim.Muon(
            muon_params,
            lr=lr,
            weight_decay=weight_decay,
            adjust_lr_fn="match_rms_adamw",
        )
        if compile_optimizer:
            opt.step = torch.compile(opt.step)
        return opt
    else:
        opt = torch.optim.AdamW(
            other_params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
        )
        if compile_optimizer:
            opt.step = torch.compile(opt.step)
        return opt


# ---------------------------------------------------------------------------
# Field type helpers for target configurations
# ---------------------------------------------------------------------------


def field_dim(field_type: FieldType, n_spatial_dims: int = 3) -> int:
    """Number of channels a single ``"scalar"`` or ``"vector"`` field occupies.

    The type tag is always lowercase by contract -- the recipe normalises
    YAML inputs at the LossCalculator / MetricCalculator boundary. Pass
    pre-lowercased strings here.

    Args:
        field_type: ``"scalar"`` or ``"vector"``.
        n_spatial_dims: Dimensionality of vector fields. Default 3.

    Raises:
        ValueError: If ``field_type`` is not ``"scalar"`` or ``"vector"``.
    """
    if field_type == "scalar":
        return 1
    if field_type == "vector":
        return n_spatial_dims
    raise ValueError(
        f"Unknown field type {field_type!r}. Expected 'scalar' or 'vector'."
    )


def align_scalar_shapes(
    p: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align a ``(...)`` / ``(..., 1)`` shape mismatch by squeezing one side.

    Used in scalar-field loss / metric paths where the prediction may
    arrive as ``(B, N, 1)`` (sliced from a concatenated ``(B, N, C)``
    tensor before squeeze) while the target is ``(B, N)`` (per-element
    scalar from a TensorDict), or vice versa. After alignment both
    tensors share the same shape (or were already equal-shape).
    """
    if p.ndim > t.ndim and p.shape[-1] == 1:
        p = p.squeeze(-1)
    elif t.ndim > p.ndim and t.shape[-1] == 1:
        t = t.squeeze(-1)
    return p, t
