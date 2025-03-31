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

"""
This code defines a distributed training pipeline for training UNet at scale,
which operates on partitioned voxel girds for the AWS drivaer dataset. It includes
loading voxels grids from h5 files, partitioning them, normalizing node and edge features using
precomputed statistics, and training the model in parallel using DistributedDataParallel
across multiple GPUs. The training loop involves computing predictions for each
partition, calculating loss, and updating model parameters using mixed precision.
Periodic checkpointing is performed to save the model, optimizer state, and training
progress. Validation is also conducted every few epochs, where predictions are compared
against ground truth values, and results are saved as point clouds. The code logs training
and validation metrics to TensorBoard and optionally integrates with Weights and Biases for
experiment tracking.
"""

import os
import sys
import pyvista as pv
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from physicsnemo.launch.logging.wandb import initialize_wandb
import json
import wandb as wb
import hydra

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from physicsnemo.distributed import DistributedManager
from physicsnemo.models.unet import UNet
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from dataloader import create_dataloader
from partition import parallel_partitioning

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils import (
    find_h5_files,
    save_checkpoint,
    load_checkpoint,
    count_trainable_params,
    calculate_continuity_loss,
)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = cfg.enable_cudnn_benchmark

    # Instantiate the distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    print(f"Rank {dist.rank} of {dist.world_size}")

    # Instantiate the writers
    if dist.rank == 0:
        writer = SummaryWriter(log_dir="tensorboard")
        initialize_wandb(
            project="aws_drivaer",
            entity="PhysicsNeMo",
            name="aws_drivaer",
            mode="disabled",
            group="group",
            save_code=True,
        )

    # AMP Configs
    amp_dtype = torch.float16  # UNet does not work with bfloat16
    amp_device = "cuda"

    # Find all .h5 files in the directory
    train_dataset = find_h5_files(to_absolute_path(cfg.h5_path))
    valid_dataset = find_h5_files(to_absolute_path(cfg.validation_h5_path))

    # Prepare the stats
    with open(to_absolute_path(cfg.stats_file), "r") as f:
        stats = json.load(f)
    mean = np.array(stats["mean"])
    std = np.array(stats["std_dev"])
    mean_tensor = torch.from_numpy(mean).to(device)
    std_tensor = torch.from_numpy(std).to(device)

    # Create DataLoader
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.world_size,
        rank=dist.rank,
        shuffle=True,
        drop_last=True,
    )
    train_dataloader = create_dataloader(
        train_dataset, mean, std, batch_size=1, num_workers=1, sampler=sampler
    )
    valid_dataloader = create_dataloader(
        valid_dataset, mean, std, batch_size=1, num_workers=1, sampler=None
    )
    print(f"Training dataset size: {len(train_dataloader)*dist.world_size}")
    print(f"Validation dataset size: {len(valid_dataloader)}")

    # Partitioning
    print("Partitioning started")
    data, filter, phys_filter, _ = parallel_partitioning(
        train_dataloader,
        num_partitions=cfg.num_partitions,
        partition_width=cfg.partition_width,
        halo_width=cfg.halo_width,
    )
    vdata, vfilter, _, vbatch = parallel_partitioning(
        valid_dataloader,
        num_partitions=cfg.num_partitions,
        partition_width=cfg.partition_width,
        halo_width=cfg.halo_width,
    )
    print("Partitioning completed")

    ######################################
    # Training #
    ######################################

    # Initialize model, loss function, and optimizer
    h = cfg.initial_hidden_dim
    model = UNet(
        in_channels=25,
        out_channels=4,
        model_depth=3,
        feature_map_channels=[h, h, 2 * h, 2 * h, 8 * h, 8 * h],
        num_conv_blocks=2,
        kernel_size=3,
        stride=1,
        conv_activation=cfg.activation,
        padding=1,
        padding_mode="replicate",
        pooling_type="MaxPool3d",
        pool_size=2,
        normalization="layernorm",
        use_attn_gate=cfg.use_attn_gate,
        attn_decoder_feature_maps=[8 * h, 2 * h],
        attn_feature_map_channels=[2 * h, h],
        attn_intermediate_channels=cfg.attn_intermediate_channels,
        gradient_checkpointing=cfg.gradient_checkpointing,
    ).to(device)
    print(f"Number of trainable parameters: {count_trainable_params(model)}")

    # DistributedDataParallel wrapper
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.start_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=cfg.end_lr
    )
    scaler = GradScaler()
    print("Instantiated the model and optimizer")

    # Check if there's a checkpoint to resume from
    start_epoch, _ = load_checkpoint(
        model, optimizer, scaler, scheduler, cfg.checkpoint_filename
    )

    # Training loop
    print("Training started")
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        total_loss_data = 0
        total_loss_continuity = 0
        for i in range(len(data)):
            optimizer.zero_grad()

            for idx, part in enumerate(
                data[i]
            ):  # (x, y, z, sdf, dsdf_dx, dsdf_dy, dsdf_dz, u_x, u_y, u_z, p)
                with torch.autocast(amp_device, enabled=True, dtype=amp_dtype):
                    part = part.to(device)
                    inp = torch.cat(
                        [
                            part[:, 0:7],
                            torch.sin(np.pi * part[:, 0:3]),
                            torch.cos(np.pi * part[:, 0:3]),
                            torch.sin(2 * np.pi * part[:, 0:3]),
                            torch.cos(2 * np.pi * part[:, 0:3]),
                            torch.sin(4 * np.pi * part[:, 0:3]),
                            torch.cos(4 * np.pi * part[:, 0:3]),
                        ],
                        dim=1,
                    )
                    pred = model(inp)
                    pred_filtered = pred[:, :, list(filter[i][idx])]
                    data_filtered = part[:, 7:, list(filter[i][idx])]
                    sdf_filtered = part[:, 3, list(filter[i][idx])]
                    sdf_filtered_denormalized = sdf_filtered * std[3] + mean[3]
                    mask = (sdf_filtered_denormalized > 0).squeeze()
                    pred_masked = pred_filtered[:, :, mask]
                    data_masked = data_filtered[:, :, mask]
                    loss_data = torch.mean((pred_masked - data_masked) ** 2) / len(
                        data[0]
                    )
                    total_loss_data += loss_data.item()
                    pred_phys_filtered = pred[:, :, list(phys_filter[i][idx])]
                    pred_phys_filtered_denormalized = (
                        pred_phys_filtered[:, 0:3]
                        * std_tensor[None, 7:10, None, None, None]
                        + mean_tensor[None, 7:10, None, None, None]
                    )
                    sdf_phys_filtered = part[:, 3, list(phys_filter[i][idx])]
                    sdf_phys_filtered_denormalized = (
                        sdf_phys_filtered * std[3] + mean[3]
                    )
                    loss_continuity = calculate_continuity_loss(
                        pred_phys_filtered_denormalized, sdf_phys_filtered_denormalized
                    ) / len(data[0])
                    total_loss_continuity += loss_continuity.item()
                    loss = loss_data * cfg.continuity_lambda * loss_continuity
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 32.0)
            scaler.step(optimizer)
            scaler.update()
        # Update scheduler after each epoch
        scheduler.step()

        if dist.rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}, Learning Rate: {current_lr}, Data Loss: {total_loss_data/len(data)}, Continuity Loss: {cfg.continuity_lambda * total_loss_continuity/len(data)}, Total Loss: {(total_loss_data + cfg.continuity_lambda * total_loss_continuity) / len(data)}"
            )
            writer.add_scalar("training_data_loss", total_loss_data / len(data), epoch)
            writer.add_scalar(
                "training_continuity_loss",
                cfg.continuity_lambda * total_loss_continuity / len(data),
                epoch,
            )
            writer.add_scalar(
                "total_loss",
                (total_loss_data + cfg.continuity_lambda * total_loss_continuity)
                / len(data),
                epoch,
            )
            writer.add_scalar("learning_rate", current_lr, epoch)
            # wb.log({"training loss": total_loss / len(data), "learning_rate": current_lr}, step=epoch)

        # Save checkpoint periodically
        if (epoch) % cfg.save_checkpoint_frequency == 0:
            if dist.world_size > 1:
                torch.distributed.barrier()
            if dist.rank == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    scheduler,
                    epoch + 1,
                    loss.item(),
                    cfg.checkpoint_filename,
                )

        ######################################
        # Validation #
        ######################################

        if dist.rank == 0 and (epoch) % cfg.validation_freq == 0:
            with torch.no_grad():
                with torch.autocast(amp_device, enabled=True, dtype=amp_dtype):
                    valid_loss = 0
                    for i in range(len(vdata)):
                        pred_list = []
                        for idx, part in enumerate(vdata[i]):
                            part = part.to(device)
                            inp = torch.cat(
                                [
                                    part[:, 0:7],
                                    torch.sin(np.pi * part[:, 0:3]),
                                    torch.cos(np.pi * part[:, 0:3]),
                                    torch.sin(2 * np.pi * part[:, 0:3]),
                                    torch.cos(2 * np.pi * part[:, 0:3]),
                                    torch.sin(4 * np.pi * part[:, 0:3]),
                                    torch.cos(4 * np.pi * part[:, 0:3]),
                                ],
                                dim=1,
                            )
                            pred = model(inp)
                            pred_filtered = pred[:, :, list(vfilter[i][idx])]
                            data_filtered = part[:, 7:, list(vfilter[i][idx])]
                            sdf_filtered = part[:, 3, list(vfilter[i][idx])]
                            sdf_filtered_denormalized = sdf_filtered * std[3] + mean[3]
                            mask = (sdf_filtered_denormalized > 0).squeeze()
                            pred_masked = pred_filtered[:, :, mask]
                            data_masked = data_filtered[:, :, mask]
                            pred_list.append(pred_filtered)
                        pred = torch.cat(pred_list, dim=2)
                        err = torch.mean((pred_masked - data_masked) ** 2) / len(
                            vdata[0]
                        )
                        valid_loss += err
                        pred = pred.to(torch.float32).cpu().numpy()
            print(f"Epoch {epoch+1}, Validation Error: {valid_loss/len(vdata)}")
            writer.add_scalar("validation_loss", valid_loss / len(vdata), epoch)
            wb.log({"Validation Error": valid_loss / len(vdata)}, step=epoch)

            # Define the dimensions and grid spacing
            x_dim, y_dim, z_dim = cfg.num_voxels_x, cfg.num_voxels_y, cfg.num_voxels_z
            dims = np.array(
                [x_dim, y_dim, z_dim]
            )  # The number of voxels in each direction
            spacing = (cfg.spacing, cfg.spacing, cfg.spacing)  # Grid spacing

            # Create a uniform grid
            grid = pv.ImageData()
            grid.dimensions = dims
            grid.spacing = spacing  # Spacing between grid points
            cbatch = vbatch[-1].to(device)
            cbatch = cbatch.clone().cpu().numpy()
            grid.origin = (cfg.grid_origin_x, cfg.grid_origin_y, cfg.grid_origin_z)

            # Add the scalar data to the grid (flatten the array as point data)
            # TODO denormalize the data
            grid.point_data["p"] = pred[:, -1].squeeze().flatten(order="F")
            grid.point_data["true_p"] = cbatch[:, -1].squeeze().flatten(order="F")
            grid.point_data["u_z"] = pred[:, -2].squeeze().flatten(order="F")
            grid.point_data["true_u_z"] = cbatch[:, -2].squeeze().flatten(order="F")
            grid.point_data["u_y"] = pred[:, -3].squeeze().flatten(order="F")
            grid.point_data["true_u_y"] = cbatch[:, -3].squeeze().flatten(order="F")
            grid.point_data["u_x"] = pred[:, -4].squeeze().flatten(order="F")
            grid.point_data["true_u_x"] = cbatch[:, -4].squeeze().flatten(order="F")
            grid.point_data["dsdf_dz"] = cbatch[:, -5].squeeze().flatten(order="F")
            grid.point_data["dsdf_dy"] = cbatch[:, -6].squeeze().flatten(order="F")
            grid.point_data["dsdf_dx"] = cbatch[:, -7].squeeze().flatten(order="F")
            grid.point_data["sdf"] = cbatch[:, -8].squeeze().flatten(order="F")
            grid.point_data["z"] = cbatch[:, -9].squeeze().flatten(order="F")
            grid.point_data["y"] = cbatch[:, -10].squeeze().flatten(order="F")
            grid.point_data["x"] = cbatch[:, -11].squeeze().flatten(order="F")

            # Save the grid to a .vti file
            grid.save("output.vti")
            print("Saved the vti file")

    # Save final checkpoint
    if dist.world_size > 1:
        torch.distributed.barrier()
    if dist.rank == 0:
        save_checkpoint(
            model,
            optimizer,
            scaler,
            scheduler,
            cfg.num_epochs,
            loss.item(),
            "final_model_checkpoint.pth",
        )
        print("Training complete")


if __name__ == "__main__":
    main()
