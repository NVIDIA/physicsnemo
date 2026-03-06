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

import json
import os

import torch

from physicsnemo.optim import CombinedOptimizer


def load_global_features(json_path: str) -> dict[str, dict[str, float]]:
    """
    Load global features JSON once.

    Returns:
        dict[str, dict[str, float]]:
            Mapping run_id -> global feature dict
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Global features file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError("Global features JSON must be a dict keyed by run_id")

    # Optional: sanity check values
    for run_id, features in data.items():
        if not isinstance(features, dict):
            raise TypeError(f"Global features for run '{run_id}' must be a dict")

    return data


def get_global_features_for_run(
    all_global_features: dict[str, dict[str, float]],
    run_id: str,
) -> dict[str, float]:
    """
    Fetch global features for a single run.

    Args:
        all_global_features: output of load_global_features
        run_id: key identifying the run (e.g. derived from filename)

    Returns:
        dict[str, float]: global scalar features for this run
    """
    try:
        return all_global_features[run_id]
    except KeyError:
        raise KeyError(f"run_id '{run_id}' not found in global features file")


def build_muon_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """
    Build Muon + AdamW combined optimizer (Muon for 2D params, AdamW for others).

    Muon requires PyTorch >= 2.9. Pass the underlying model (unwrap DDP if needed).
    """
    if not hasattr(torch.optim, "Muon"):
        raise ImportError(
            "Muon optimizer requires PyTorch >= 2.9. "
            "Install a newer PyTorch or use optimizer=adam."
        )
    base_model = model.module if hasattr(model, "module") else model
    muon_params = [p for p in base_model.parameters() if p.ndim == 2]
    other_params = [p for p in base_model.parameters() if p.ndim != 2]
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
