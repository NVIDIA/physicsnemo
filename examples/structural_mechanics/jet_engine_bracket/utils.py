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

r"""Helpers shared by the workflows of this example."""

from __future__ import annotations

import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from datapipe import model_inputs

from physicsnemo.optim import CombinedOptimizer, Muon

EXAMPLE_DIR = Path(__file__).resolve().parent


def resolve(path: str | Path) -> Path:
    r"""``path`` as is if absolute, else relative to this example."""
    path = Path(path)
    return path if path.is_absolute() else EXAMPLE_DIR / path


def set_seed(seed: int) -> None:
    r"""Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast(device: torch.device, precision: str):
    r"""bfloat16 autocast context for ``precision="bfloat16"``, else a no-op."""
    if precision == "bfloat16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def warmup_cosine(step: int, warmup: int, total: int, final_ratio: float) -> float:
    r"""Learning-rate factor: linear warm-up over ``warmup`` steps, then cosine decay to ``final_ratio``."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return final_ratio + (1.0 - final_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def muon_adamw(
    model: torch.nn.Module, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    r"""Muon for the weight matrices and AdamW for every other parameter."""
    matrices = [p for p in model.parameters() if p.requires_grad and p.ndim == 2]
    others = [p for p in model.parameters() if p.requires_grad and p.ndim != 2]
    return CombinedOptimizer(
        [
            Muon(
                matrices,
                lr=lr,
                weight_decay=weight_decay,
                adjust_lr_fn="match_rms_adamw",
            ),
            torch.optim.AdamW(others, lr=lr, weight_decay=weight_decay),
        ]
    )


def weighted_mse(
    prediction: torch.Tensor, target: torch.Tensor, weights: dict[str, float]
) -> torch.Tensor:
    r"""Weighted mean squared error of the normalized displacement (channels 0-2) and stress (channel 3)."""
    error = (prediction.float() - target) ** 2  # (B, N, 4)
    return (
        weights["displacement"] * error[..., :3].mean()
        + weights["von_mises"] * error[..., 3].mean()
    )


def case_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    r"""Relative L2 errors of displacement and von Mises stress, and the stress :math:`R^2`, of one sample.

    Both tensors are :math:`(N, 4)` in the same (normalized or physical) units; relative errors do
    not depend on the units.
    """
    p, t = prediction.double(), target.double()
    stress = t[:, 3]
    return {
        "displacement_rel_l2": float(
            torch.linalg.norm(p[:, :3] - t[:, :3]) / torch.linalg.norm(t[:, :3])
        ),
        "von_mises_rel_l2": float(
            torch.linalg.norm(p[:, 3] - stress) / torch.linalg.norm(stress)
        ),
        "von_mises_r2": float(
            1.0
            - torch.sum((p[:, 3] - stress) ** 2)
            / torch.sum((stress - stress.mean()) ** 2)
        ),
    }


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    precision: str = "float32",
) -> torch.Tensor:
    r"""Normalized prediction :math:`(N, 4)` for a batch of one sample."""
    with autocast(device, precision):
        return model(**model_inputs(batch, device))[0].float()


def newest(directory: str | Path, pattern: str) -> Path:
    r"""The most recently modified file matching ``pattern`` in ``directory``."""
    files = sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No {pattern} in {directory}")
    return files[-1]
