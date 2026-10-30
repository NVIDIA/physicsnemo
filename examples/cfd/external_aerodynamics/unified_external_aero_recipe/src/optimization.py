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

"""Configurable optimization for the AB-UPT paper-style training recipe.

Lion follows Chen et al., *Symbolic Discovery of Optimization Algorithms*,
Algorithm 2 (https://arxiv.org/abs/2302.06675). No model research code is used.
"""

from __future__ import annotations

import math

import torch


class Lion(torch.optim.Optimizer):
    """Sign-momentum optimizer with decoupled weight decay.

    Parameters are normally maintained in float32 under autocast. Sparse
    gradients are unsupported. ``betas`` control update and momentum averaging,
    respectively; they are not the two Adam moment coefficients.
    """

    def __init__(self, params, lr=5e-5, betas=(0.9, 0.99), weight_decay=0.05):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        if len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        """Apply one Lion update and return the optional closure loss."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta_update, beta_momentum = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")
                state = self.state[parameter]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(parameter)
                momentum = state["momentum"]
                direction = (
                    momentum.mul(beta_update)
                    .add(gradient, alpha=1 - beta_update)
                    .sign_()
                )
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(direction, alpha=-group["lr"])
                momentum.mul_(beta_momentum).add_(gradient, alpha=1 - beta_momentum)
        return loss


class WarmupCosine(torch.optim.lr_scheduler.LRScheduler):
    """Per-update linear warmup followed by cosine decay.

    ``total_steps`` counts optimizer updates over the complete run, including
    warmup. The first update uses peak_lr / warmup_steps, the last warmup
    update uses peak_lr, and the final planned update uses min_lr. A resumed
    run restores the scheduler state instead of recomputing a shorter schedule.
    """

    def __init__(
        self, optimizer, total_steps, warmup_steps, min_lr=1e-6, last_epoch=-1
    ):
        if (
            isinstance(total_steps, bool)
            or not isinstance(total_steps, int)
            or total_steps < 2
        ):
            raise ValueError("total_steps must be an integer >= 2")
        if (
            isinstance(warmup_steps, bool)
            or not isinstance(warmup_steps, int)
            or not 0 <= warmup_steps < total_steps
        ):
            raise ValueError("warmup_steps must be an integer in [0, total_steps)")
        if not math.isfinite(min_lr) or min_lr < 0:
            raise ValueError("min_lr must be finite and nonnegative")
        if any(min_lr > group["lr"] for group in optimizer.param_groups):
            raise ValueError("min_lr must not exceed any group's peak learning rate")
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        """Return each group learning rate for the next optimizer update."""
        step = self.last_epoch
        if step < self.warmup_steps:
            return [lr * (step + 1) / self.warmup_steps for lr in self.base_lrs]
        if self.warmup_steps:
            progress = (step + 1 - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
        else:
            progress = step / (self.total_steps - 1)
        factor = 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return [self.min_lr + (lr - self.min_lr) * factor for lr in self.base_lrs]
