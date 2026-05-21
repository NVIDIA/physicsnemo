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

"""Per-batch training step for the active-learning loop.

The structure of this step (encoder forward + reduced-embedding +
GP head + optional consistency penalty + grad sync for non-DDP
modules) is the *generic* AL training recipe. The CFD-specific bits
live in ``aero_physics`` (drag-target extraction, Monte-Carlo drag
from subsampled outputs); to adapt the recipe to a different
quantity of interest, swap those calls.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.uq import VariationalGPHead

from utils import cast_precisions, get_autocast_context, loss_fn
from gp_utils import sync_non_ddp_gradients
from aero_physics import (
    compute_drag_from_subsampled_outputs,
    compute_drag_target_from_batch,
)


def train_one_batch(
    batch: dict[str, Any],
    backbone_model: nn.Module,
    embedding_reduction: nn.Module,
    gp: VariationalGPHead,
    surface_factors: dict[str, torch.Tensor],
    device: torch.device,
    precision: str,
    optimizer: torch.optim.Optimizer,
    lambda_gp: float,
    lambda_consistency: float,
    consistency_detach: bool,
    consistency_every_n: int,
    step_idx: int,
    dist_manager: DistributedManager,
    accumulation_steps: int = 1,
) -> float:
    """Run a single training step and return the (unscaled) loss value.

    The total loss is::

        L = field_mse + lambda_gp * head_loss
              + lambda_consistency * consistency_mse

    where the consistency term is evaluated only when
    ``surface_*_sub`` are present in the batch and only every
    ``consistency_every_n`` steps.

    With ``accumulation_steps > 1`` the loss is scaled by
    ``1/accumulation_steps`` and ``zero_grad`` / non-DDP grad sync /
    ``optimizer.step`` only fire on cycle boundaries (``step_idx``
    counted across all batches in the epoch). Partial cycles at the
    end of an epoch are dropped; size your batch count accordingly.
    """
    features = cast_precisions(batch["fx"], precision)
    embeddings = cast_precisions(batch["embeddings"], precision)
    geometry = (
        cast_precisions(batch["geometry"], precision) if "geometry" in batch else None
    )

    with get_autocast_context(precision):
        local_positions = embeddings[:, :, :3]
        outputs, embedding_states = backbone_model(
            global_embedding=features,
            local_embedding=embeddings,
            geometry=geometry,
            local_positions=local_positions,
            return_embedding_states=True,
        )
        mse_loss = torch.mean(loss_fn(outputs, batch["fields"]))

    reduced = embedding_reduction(embedding_states.flatten(1, 2))
    drag_target = compute_drag_target_from_batch(batch, surface_factors, device).to(
        reduced.dtype
    )

    head_mean, head_loss = gp.forward_and_loss(reduced, drag_target)

    c_loss = torch.tensor(0.0, device=device)
    if lambda_consistency > 0 and (step_idx % consistency_every_n == 0):
        if "surface_areas_sub" in batch and "surface_normals_sub" in batch:
            trans_cd = compute_drag_from_subsampled_outputs(
                outputs, batch, surface_factors, device
            ).to(head_mean.dtype)
            if consistency_detach:
                trans_cd = trans_cd.detach()
            c_loss = F.mse_loss(head_mean, trans_cd)

    total_loss = mse_loss + lambda_gp * head_loss + lambda_consistency * c_loss

    if step_idx % accumulation_steps == 0:
        optimizer.zero_grad()

    (total_loss / accumulation_steps).backward()

    if (step_idx + 1) % accumulation_steps == 0:
        if dist_manager.world_size > 1:
            sync_non_ddp_gradients([embedding_reduction, gp], dist_manager.world_size)
        optimizer.step()

    return total_loss.item()
