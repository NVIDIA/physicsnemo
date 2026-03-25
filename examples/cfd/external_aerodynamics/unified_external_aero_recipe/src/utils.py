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

from dataclasses import dataclass
from typing import Literal

import torch

from omegaconf import DictConfig
from physicsnemo.optim import CombinedOptimizer


def build_muon_optimizer(
    model: torch.nn.Module, cfg: DictConfig
) -> torch.optim.Optimizer:
    """Build Muon + AdamW combined optimizer.

    Muon handles 2-D parameters (linear/attention weight matrices) while AdamW
    handles everything else (biases, layer-norm, embeddings, etc.).

    Parameters
    ----------
    model : torch.nn.Module
        The model (may be DDP-wrapped).
    cfg : DictConfig
        Full Hydra config.  Reads ``cfg.training.optimizer.*`` for lr,
        weight_decay, betas, and eps.
    """
    base_model = model.module if hasattr(model, "module") else model
    muon_params = [p for p in base_model.parameters() if p.ndim == 2]
    other_params = [p for p in base_model.parameters() if p.ndim != 2]

    opt_cfg = cfg.training.optimizer
    lr = opt_cfg.lr
    weight_decay = opt_cfg.get("weight_decay", 1e-4)
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = opt_cfg.get("eps", 1e-8)

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
        return torch.optim.AdamW(
            other_params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
        )


# ---------------------------------------------------------------------------
# Field specification for target configurations
# ---------------------------------------------------------------------------


@dataclass
class FieldSpec:
    """Specification for a single target field.

    Attributes:
        name: Human-readable name for the field (used in metric/loss keys).
        field_type: Either "scalar" or "vector".
        start_index: Starting index in the channel dimension.
        end_index: Ending index (exclusive) in the channel dimension.
    """

    name: str
    field_type: Literal["scalar", "vector"]
    start_index: int
    end_index: int

    @property
    def dim(self) -> int:
        """Number of channels for this field."""
        return self.end_index - self.start_index


def parse_target_config(
    target_config: dict[str, str], vector_dim: int = 3
) -> list[FieldSpec]:
    """Parse target configuration to field specifications.

    Args:
        target_config: Mapping of field names to types ("scalar" or "vector").
                      Order determines channel indices.
        vector_dim: Dimensionality of vector fields. Default is 3.

    Returns:
        List of FieldSpec objects describing each field.

    Raises:
        ValueError: If an unknown field type is specified.

    Example:
        >>> config = {"pressure": "scalar", "velocity": "vector"}
        >>> specs = parse_target_config(config)
        >>> specs[0]
        FieldSpec(name='pressure', field_type='scalar', start_index=0, end_index=1)
        >>> specs[1]
        FieldSpec(name='velocity', field_type='vector', start_index=1, end_index=4)
    """
    specs = []
    current_index = 0

    for name, field_type in target_config.items():
        field_type = field_type.lower()
        if field_type == "scalar":
            dim = 1
        elif field_type == "vector":
            dim = vector_dim
        else:
            raise ValueError(
                f"Unknown field type '{field_type}' for field '{name}'. "
                "Expected 'scalar' or 'vector'."
            )

        specs.append(
            FieldSpec(
                name=name,
                field_type=field_type,
                start_index=current_index,
                end_index=current_index + dim,
            )
        )
        current_index += dim

    return specs
