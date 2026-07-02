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

"""PhysicsNeMo-native training loop for the FP-DDM thermal PINO."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import LaunchLogger

from .data import create_dataloaders
from .model import ThermalPINO, thermal_losses


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _core_model(model: torch.nn.Module) -> ThermalPINO:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _batch_losses(
    model: torch.nn.Module,
    batch: torch.Tensor,
    pde_weight: float,
    boundary_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    core = _core_model(model)
    normalized = core.normalize_input(batch)
    prediction = model(normalized)
    pde_loss, boundary_loss = thermal_losses(
        prediction,
        normalized,
        source_scale=core.nondimensional_source_scale,
    )
    objective = pde_weight * pde_loss + boundary_weight * boundary_loss
    pde_metric = torch.sqrt(pde_weight * pde_loss)
    boundary_metric = torch.sqrt(boundary_weight * boundary_loss)
    return objective, {
        "loss": pde_metric + boundary_metric,
        "pde": pde_metric,
        "boundary": boundary_metric,
    }


def _reduce_metrics(
    totals: dict[str, torch.Tensor], batches: int, dist: DistributedManager
) -> dict[str, float]:
    values = torch.stack([totals[name] for name in ("loss", "pde", "boundary")])
    count = torch.tensor(float(batches), device=dist.device)
    if dist.distributed:
        torch.distributed.all_reduce(values)
        torch.distributed.all_reduce(count)
    values /= count.clamp_min(1.0)
    return {
        name: float(value.detach().cpu())
        for name, value in zip(("loss", "pde", "boundary"), values)
    }


def _run_epoch(
    model: torch.nn.Module,
    loader,
    dist: DistributedManager,
    *,
    optimizer: torch.optim.Optimizer | None,
    pde_weight: float,
    boundary_weight: float,
    logger: LaunchLogger | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        name: torch.tensor(0.0, device=dist.device)
        for name in ("loss", "pde", "boundary")
    }
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(dist.device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            objective, metrics = _batch_losses(
                model, batch, pde_weight, boundary_weight
            )
            if optimizer is not None:
                objective.backward()
                optimizer.step()
            for name, value in metrics.items():
                totals[name] += value.detach()
            if logger is not None:
                logger.log_minibatch(
                    {
                        "loss": metrics["loss"].detach(),
                        "loss_pde": metrics["pde"].detach(),
                        "loss_boundary": metrics["boundary"].detach(),
                    }
                )
            batches += 1
    return _reduce_metrics(totals, batches, dist)


def train_model(config: Mapping[str, object], dist: DistributedManager) -> Path:
    """Train the thermal PINO and return its best checkpoint directory."""

    training = config["training"]
    dataset = config["dataset"]
    model_config = config["model"]
    if not all(
        isinstance(section, Mapping) for section in (training, dataset, model_config)
    ):
        raise TypeError("training, dataset, and model configs must be mappings")

    seed = int(config.get("seed", 10))
    set_seed(seed + dist.rank)
    dataset_config = dict(dataset)
    dataset_config["n_samples"] = int(training["samples"])
    train_loader, validation_loader, test_loader = create_dataloaders(
        dataset_config,
        int(training["batch_size"]),
        seed=seed,
        num_workers=int(training.get("num_workers", 10)),
        distributed=dist.distributed,
        rank=dist.rank,
        world_size=dist.world_size,
    )

    model = ThermalPINO(model_config).to(dist.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"])
    )
    metadata: dict[str, object] = {}
    loaded_epoch = 0
    resume_dir = training.get("resume_dir")
    if resume_dir:
        loaded_epoch = load_checkpoint(
            Path(str(resume_dir)),
            models=model,
            optimizer=optimizer,
            metadata_dict=metadata,
            device=dist.device,
        )

    if dist.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank] if dist.device.type == "cuda" else None,
            output_device=dist.local_rank if dist.device.type == "cuda" else None,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
        )

    output_dir = Path(str(training["output_dir"]))
    best_dir = output_dir / "checkpoints" / "best"
    latest_dir = output_dir / "checkpoints" / "latest"
    best_loss = float(metadata.get("best_validation_loss", float("inf")))
    max_minutes = float(training.get("max_minutes", 0.0))
    deadline = time.monotonic() + 60.0 * max_minutes if max_minutes > 0 else None
    pde_weight = float(training.get("pde_weight", 1.0))
    boundary_weight = float(training.get("boundary_weight", 1.0))
    last_epoch = loaded_epoch

    for epoch in range(loaded_epoch + 1, int(training["epochs"]) + 1):
        last_epoch = epoch
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(train_loader)
        ) as logger:
            train_metrics = _run_epoch(
                model,
                train_loader,
                dist,
                optimizer=optimizer,
                pde_weight=pde_weight,
                boundary_weight=boundary_weight,
                logger=logger,
            )
            logger.log_epoch(
                {
                    **train_metrics,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        with LaunchLogger("valid", epoch=epoch) as logger:
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                dist,
                optimizer=None,
                pde_weight=pde_weight,
                boundary_weight=boundary_weight,
            )
            logger.log_epoch(validation_metrics)

        if dist.rank == 0:
            core = _core_model(model)
            save_checkpoint(
                latest_dir,
                models=core,
                optimizer=optimizer,
                epoch=epoch,
                metadata={
                    "best_validation_loss": min(best_loss, validation_metrics["loss"])
                },
            )
            if validation_metrics["loss"] <= best_loss:
                best_loss = validation_metrics["loss"]
                save_checkpoint(
                    best_dir,
                    models=core,
                    epoch=epoch,
                    metadata={"best_validation_loss": best_loss},
                )

        if deadline is not None and time.monotonic() >= deadline:
            break

    with LaunchLogger("test", epoch=last_epoch) as logger:
        test_metrics = _run_epoch(
            model,
            test_loader,
            dist,
            optimizer=None,
            pde_weight=pde_weight,
            boundary_weight=boundary_weight,
        )
        logger.log_epoch(test_metrics)
    return best_dir
