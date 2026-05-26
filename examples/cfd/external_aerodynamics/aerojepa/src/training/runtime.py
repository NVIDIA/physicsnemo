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

r"""Training-runtime helpers: seeding, device transfer, autocast, scheduler."""

from __future__ import annotations

import contextlib
import math
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    r"""Seed Python, NumPy, and PyTorch RNGs (and CUDA, if available).

    Parameters
    ----------
    seed : int
        Seed value applied to all four RNGs.
    """
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    r"""Move every tensor value in ``batch`` to ``device``.

    Non-tensor values (metadata strings, ints, floats, lists) are passed
    through unchanged.

    Parameters
    ----------
    batch : dict
        Batch dict produced by the dataset's collate function.
    device : torch.device
        Target device.

    Returns
    -------
    dict
        New dict with tensors moved; non-tensor entries shared.
    """
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def get_autocast_context(
    device: torch.device, precision: str
) -> contextlib.AbstractContextManager:
    r"""Return an autocast context manager matching ``precision``.

    Returns a no-op context on CPU or when ``precision`` is unrecognised
    (so callers can wrap blocks unconditionally).

    Parameters
    ----------
    device : torch.device
        Device the wrapped block runs on.
    precision : str
        One of ``"bf16"``, ``"fp16"``, ``"fp32"`` (case-insensitive).

    Returns
    -------
    contextlib.AbstractContextManager
        Autocast context; a no-op when CUDA isn't in use or the
        precision is fp32.
    """
    precision_l = str(precision).lower()
    if device.type != "cuda":
        return contextlib.nullcontext()
    if precision_l == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision_l == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: float = 5.0,
    warmup_steps: int | None = None,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    r"""Build a learning-rate scheduler.

    Supports ``"none"`` / ``"constant"`` (returns ``None``) and
    ``"warmup_cosine"``: linear warmup for ``warmup_epochs`` epochs (or
    a step count via ``warmup_steps``), then a half-cosine decay from
    the peak LR down to ``peak_lr * min_lr_ratio``.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to wrap.
    name : str
        Scheduler name. ``"warmup_cosine"`` is the supported non-trivial
        option; ``"none"`` / ``"constant"`` return ``None``.
    epochs : int
        Total number of training epochs.
    steps_per_epoch : int
        Number of optimizer steps per epoch.
    warmup_epochs : float, optional
        Linear-warmup length in epochs. Ignored when ``warmup_steps``
        is supplied. Default ``5.0``.
    warmup_steps : int, optional
        Linear-warmup length in optimizer steps. When set, overrides
        ``warmup_epochs``.
    min_lr_ratio : float, optional
        Floor on the LR multiplier at the end of cosine decay. Clipped
        to ``[0, 1]``. Default ``0.05``.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR or None
        The scheduler (or ``None`` for constant LR).

    Raises
    ------
    ValueError
        If ``name`` is not one of the supported strings.
    """
    name_l = str(name).lower()
    if name_l in {"none", "constant"}:
        return None
    if name_l != "warmup_cosine":
        raise ValueError(f"Unsupported lr scheduler: {name!r}")

    total_steps = max(1, int(epochs) * max(1, int(steps_per_epoch)))
    if warmup_steps is None:
        warmup_steps = int(
            round(max(0.0, float(warmup_epochs)) * max(1, int(steps_per_epoch)))
        )
    warmup_steps = (
        min(max(0, int(warmup_steps)), total_steps - 1) if total_steps > 1 else 0
    )
    min_ratio = max(0.0, min(1.0, float(min_lr_ratio)))

    def _lambda(step: int) -> float:
        s = int(step)
        if warmup_steps > 0 and s < warmup_steps:
            return float(s + 1) / float(max(1, warmup_steps))
        denom = max(1, total_steps - warmup_steps)
        progress = float(s - warmup_steps) / float(denom)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lambda)
