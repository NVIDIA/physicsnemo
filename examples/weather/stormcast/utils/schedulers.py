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
Learning rate scheduler utilities.

Note: Warmup is handled manually in the trainer via linear scaling.
Schedulers here handle the decay phase after warmup.
"""

import torch.optim.lr_scheduler as lr_scheduler_module
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from utils.config import SchedulerConfig
from utils.logging import ExperimentLogger


def init_scheduler(
    optimizer, cfg: SchedulerConfig, *, total_steps: int, logger: ExperimentLogger
) -> tuple[LRScheduler | None, str | None]:
    r"""
    Create a scheduler from config.

    Warmup is handled by the trainer via linear scaling, not by the scheduler.
    Supports any torch.optim.lr_scheduler class by name.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule.
    cfg : SchedulerConfig
        Scheduler configuration:
        - ``name``: Any ``torch.optim.lr_scheduler`` class name (e.g., "CosineAnnealingLR")
        - Other scheduler-specific parameters passed to the constructor
    warmup_steps : int
        Number of warmup steps (used to compute default T_max for cosine schedulers).
    total_steps : int
        Total training steps.

    Returns
    -------
    scheduler : LRScheduler or None
        The configured scheduler, or None if cfg is empty.
    scheduler_name : str or None
        Name of the scheduler class, or None if no scheduler.

    Raises
    ------
    ValueError
        If the scheduler name is not found in torch.optim.lr_scheduler.

    Examples
    --------
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    >>> scheduler, name = init_scheduler(
    ...     optimizer,
    ...     {"name": "CosineAnnealingLR", "eta_min": 1e-6},
    ...     warmup_steps=1000,
    ...     total_steps=10000,
    ... )
    >>> name
    'CosineAnnealingLR'
    """
    if cfg.name is None:
        return None, None

    # Get scheduler class from torch
    scheduler_cls = getattr(lr_scheduler_module, cfg.name)
    if not isinstance(scheduler_cls, type) or not issubclass(
        scheduler_cls, LRScheduler
    ):
        raise ValueError(f"{cfg.name} is not a learning rate scheduler")

    # Set default T_max for cosine schedulers (decay steps after warmup)
    if cfg.name == "CosineAnnealingLR" and not hasattr(cfg, "T_max"):
        cfg.T_max = total_steps - cfg.lr_rampup_steps

    scheduler_kwargs = cfg.__dict__.copy()
    del scheduler_kwargs["name"], scheduler_kwargs["lr_rampup_steps"]
    scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
    logger.info(f"Initialized scheduler: {cfg.name} with params: {scheduler_kwargs}")
    return scheduler, cfg.name


def step_scheduler(
    scheduler: LRScheduler | None,
    *,
    total_steps: int,
    warmup_steps: int,
    metric: float | None = None,
    logger: ExperimentLogger,
) -> None:
    r"""
    Advance the scheduler by one step.

    Only steps the scheduler after the warmup period. For ReduceLROnPlateau,
    requires a metric value to determine learning rate reduction.

    Parameters
    ----------
    scheduler : LRScheduler or None
        The scheduler to step. If None, this function is a no-op.
    total_steps : int
        Current total step count.
    warmup_steps : int
        Number of warmup steps. Scheduler only steps when total_steps >= warmup_steps.
    metric : float, optional
        Validation metric value. Required for ReduceLROnPlateau scheduler.

    Examples
    --------
    >>> step_scheduler(scheduler, total_steps=1500, warmup_steps=1000)
    >>> # For ReduceLROnPlateau:
    >>> step_scheduler(scheduler, total_steps=1500, warmup_steps=1000, metric=0.05)
    """
    if scheduler is None:
        return

    # Only step after warmup (warmup handled by trainer)
    if total_steps < warmup_steps:
        return

    # ReduceLROnPlateau needs a metric and only steps during validation
    if isinstance(scheduler, ReduceLROnPlateau):
        if metric is not None:
            # Get LR before stepping to detect changes
            old_lr = scheduler.optimizer.param_groups[0]["lr"]
            scheduler.step(metric)
            new_lr = scheduler.optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                logger.info(
                    f"ReduceLROnPlateau triggered: LR reduced from {old_lr:.6g} to {new_lr:.6g} "
                    f"(metric={metric:.6f})"
                )
            else:
                logger.debug(
                    f"ReduceLROnPlateau stepped: metric={metric:.6f}, LR={new_lr:.6g}, "
                    f"num_bad_epochs={scheduler.num_bad_epochs}/{scheduler.patience}"
                )
    else:
        # Other schedulers step once per training step (when metric is None)
        # Skip during validation (when metric is provided) to avoid double-stepping
        if metric is None:
            scheduler.step()
