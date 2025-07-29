# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import logging
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from dataset import ProcessedVTPDataset
from model import MoEGatingNet

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Trainer")


def entropy(weights, eps=1e-8):
    return -torch.sum(weights * torch.log(weights + eps), dim=-1).mean()


def train_gating_networks(dataset, cfg):
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    rank = dist.rank
    world_size = dist.world_size

    if rank == 0:
        logger.info(f"Distributed training: {world_size} GPUs")

    # Distributed sampler for shuffling and partitioning data
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        collate_fn=lambda x: x,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=True,
        drop_last=False,
    )
    if rank == 0:
        logger.info("DataLoader initialized.")

    pressure_gating = MoEGatingNet(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_experts=cfg.num_experts,
        num_feature_per_expert=cfg.num_feature_per_expert_pressure,
        use_moe_bias=cfg.use_moe_bias,
        include_normals=cfg.include_normals,
    ).to(device)
    shear_gating = MoEGatingNet(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_experts=cfg.num_experts,
        num_feature_per_expert=cfg.num_feature_per_expert_shear,
        use_moe_bias=cfg.use_moe_bias,
        include_normals=cfg.include_normals,
    ).to(device)

    # Wrap with DistributedDataParallel
    if world_size > 1:
        pressure_gating = torch.nn.parallel.DistributedDataParallel(
            pressure_gating, device_ids=[dist.local_rank], output_device=dist.local_rank
        )
        shear_gating = torch.nn.parallel.DistributedDataParallel(
            shear_gating, device_ids=[dist.local_rank], output_device=dist.local_rank
        )

    optimizer = torch.optim.Adam(
        list(pressure_gating.parameters()) + list(shear_gating.parameters()),
        lr=cfg.start_lr,
    )

    # Learning rate scheduler: CosineAnnealingLR from 1e-3 to 5e-6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=cfg.end_lr
    )

    # Initialize mixed precision training
    scaler = GradScaler(enabled=cfg.use_amp)
    if rank == 0:
        logger.info(
            f"Mixed precision training: {'enabled' if cfg.use_amp else 'disabled'}"
        )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # --- Restart logic: load latest checkpoint if exists ---
    start_epoch = load_checkpoint(
        cfg.checkpoint_dir,
        models=[pressure_gating, shear_gating],
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )

    for epoch in range(start_epoch, cfg.num_epochs):
        pressure_losses = []
        shear_losses = []
        sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(
                f"\nEpoch {epoch+1}/{cfg.num_epochs} -----------------------------"
            )
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")
        else:
            pbar = dataloader

        for batch_idx, batch in enumerate(pbar):
            sample = batch[0]
            # Pressure
            p_preds = (
                torch.from_numpy(
                    np.concatenate(
                        [
                            sample["p_pred_xmgn"],
                            sample["p_pred_fignet"],
                            sample["p_pred_domino"],
                        ],
                        axis=1,
                    )
                )
                .float()
                .to(device)
            )  # (N, 3)
            p_true = torch.from_numpy(sample["p"]).float().to(device)  # (N, 1)

            # Use only normals as additional features
            normals = torch.from_numpy(sample["normals"]).float().to(device)  # (N, 3)
            p_gating_input = torch.cat([p_preds, normals], dim=1)  # (N, 6)

            # Shear
            shear_preds = (
                torch.from_numpy(
                    np.stack(
                        [
                            sample["shear_pred_xmgn"],
                            sample["shear_pred_fignet"],
                            sample["shear_pred_domino"],
                        ],
                        axis=1,
                    )
                )
                .float()
                .to(device)
            )  # (N, 3, 3)
            shear_preds_flat = shear_preds.reshape(shear_preds.shape[0], -1)  # (N, 9)
            shear_true = torch.from_numpy(sample["shear"]).float().to(device)  # (N, 3)
            shear_gating_input = torch.cat(
                [shear_preds_flat, normals], dim=1
            )  # (N, 12)

            # --- Forward  ---
            with autocast(enabled=cfg.use_amp):
                gate_p = pressure_gating(p_gating_input)  # (N, 3)
                pred_p = torch.sum(gate_p * p_preds, dim=1, keepdim=True)  # (N, 1)

                gate_shear = shear_gating(shear_gating_input)  # (N, 3)
                pred_shear = torch.sum(
                    gate_shear.unsqueeze(-1) * shear_preds, dim=1
                )  # (N, 3)

                # --- Loss ---
                loss_p = F.mse_loss(pred_p, p_true)
                loss_shear = F.mse_loss(pred_shear, shear_true)
                entropy_p = entropy(gate_p)  # gate_p: (N, 3)
                entropy_shear = entropy(gate_shear)  # gate_shear: (N, 3)
                loss = (
                    loss_p
                    + loss_shear
                    - cfg.lambda_entropy * (entropy_p + entropy_shear)
                )

            # --- Backward ---
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pressure_losses.append(loss_p.item())
            shear_losses.append(loss_shear.item())

            if rank == 0:
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(
                        {
                            "PressureLoss": f"{loss_p.item():.4e}",
                            "ShearLoss": f"{loss_shear.item():.4e}",
                        }
                    )

            # Avoid memory leaks
            del sample, p_preds, p_true, normals, p_gating_input
            del shear_preds, shear_preds_flat, shear_true, shear_gating_input
            del gate_p, pred_p, gate_shear, pred_shear, loss_p, loss_shear, loss
            torch.cuda.empty_cache()

        # Step the scheduler at the end of each epoch
        scheduler.step()

        # Optionally, gather and print mean losses across all ranks
        mean_pressure_loss = np.mean(pressure_losses)
        mean_shear_loss = np.mean(shear_losses)
        if world_size > 1:
            # Reduce mean losses across all processes
            mean_pressure_loss_tensor = torch.tensor(mean_pressure_loss, device=device)
            mean_shear_loss_tensor = torch.tensor(mean_shear_loss, device=device)
            torch.distributed.all_reduce(
                mean_pressure_loss_tensor, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                mean_shear_loss_tensor, op=torch.distributed.ReduceOp.SUM
            )
            mean_pressure_loss = mean_pressure_loss_tensor.item() / world_size
            mean_shear_loss = mean_shear_loss_tensor.item() / world_size

        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1}/{cfg.num_epochs} | Mean Pressure Loss: {mean_pressure_loss:.6f} | Mean Shear Loss: {mean_shear_loss:.6f} | LR: {current_lr:.6e}"
            )

            # --- Save model after each epoch  ---
            save_checkpoint(
                cfg.checkpoint_dir,
                models=[pressure_gating, shear_gating],
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
            )
            logger.info(
                f"Saved model checkpoint for epoch {epoch+1} to {cfg.checkpoint_dir}"
            )

    if rank == 0:
        logger.info("Training complete.")
    return pressure_gating, shear_gating


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(
        f"Loading dataset from {to_absolute_path(cfg.preprocessed_data_dir)} with normalization stats from {os.path.join(to_absolute_path(cfg.preprocessed_data_dir), 'stats')}..."
    )
    dataset = ProcessedVTPDataset(
        os.path.join(to_absolute_path(cfg.preprocessed_data_dir), "train"),
        os.path.join(to_absolute_path(cfg.preprocessed_data_dir), "stats"),
        normalize=True,
    )
    logger.info("Dataset loaded.")
    train_gating_networks(dataset, cfg)


if __name__ == "__main__":
    main()
