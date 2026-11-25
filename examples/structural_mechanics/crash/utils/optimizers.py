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

"""Optimizers of the crash recipes: Muon with AdamW for the remaining parameters."""

import torch

from physicsnemo.optim import CombinedOptimizer


def build_muon_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """
    Build Muon + AdamW combined optimizer (Muon for 2D params, AdamW for others).

    Muon requires PyTorch >= 2.9. Pass the underlying model (unwrap DDP if needed).
    Frozen parameters (a residual model's teacher) are not optimized.
    ``training.muon_min_dim`` (default 0): 2-D parameters whose smaller side is
    below it go to AdamW instead of being orthogonalized by Muon (tiny matrices
    such as an 8 x 32 per-head bias projection).
    """
    if not hasattr(torch.optim, "Muon"):
        raise ImportError(
            "Muon optimizer requires PyTorch >= 2.9. "
            "Install a newer PyTorch or use optimizer=adam."
        )
    base_model = model.module if hasattr(model, "module") else model
    min_dim = int(cfg.training.get("muon_min_dim", 0))
    named = [(n, p) for n, p in base_model.named_parameters() if p.requires_grad]
    muon_params = [p for _, p in named if p.ndim == 2 and min(p.shape) >= min_dim]
    other_params = [
        p for _, p in named if not (p.ndim == 2 and min(p.shape) >= min_dim)
    ]
    weight_decay = cfg.training.get("optimizer_weight_decay", 1e-4)
    lr = cfg.training.start_lr
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
                    betas=(0.9, 0.999),
                    eps=1.0e-8,
                ),
            ]
        )
    elif muon_params:
        return torch.optim.Muon(
            muon_params,
            lr=lr,
            weight_decay=weight_decay,
            adjust_lr_fn="match_rms_adamw",
        )
    else:
        return torch.optim.AdamW(other_params, lr=lr, weight_decay=weight_decay)
