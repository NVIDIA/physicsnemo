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
This code defines a distributed pipeline for re-training the DoMINO model on
CFD datasets starting from a pre-trained checkpoint. The model is retrained 
with a very small learning rate on the new dataset. The train tab in 
config.yaml can be used to specify batch size, number of epochs and 
other training parameters.
"""

import time
import os
import re
import torch
import torchinfo

import apex
import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

from physicsnemo.datapipes.cae.domino_datapipe import create_domino_dataset
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils.domino.utils import *


def relative_loss_fn(output, target, padded_value=-10):
    mask = abs(target - padded_value) > 1e-3
    masked_loss = torch.sum(((output - target) ** 2.0) * mask, (0, 1)) / torch.sum(
        mask, (0, 1)
    )
    masked_truth = torch.sum(((target) ** 2.0) * mask, (0, 1)) / torch.sum(mask, (0, 1))
    loss = torch.mean(masked_loss / masked_truth)
    return loss


def mse_loss_fn(output, target, padded_value=-10):
    mask = abs(target - padded_value) > 1e-3
    masked_loss = torch.sum(((output - target) ** 2.0) * mask, (0, 1)) / torch.sum(
        mask, (0, 1)
    )
    masked_truth = torch.sum(((target) ** 2.0) * mask, (0, 1)) / torch.sum(mask, (0, 1))
    loss = torch.mean(masked_loss)
    return loss


def mse_loss_fn_surface(output, target, normals, padded_value=-10):
    ws_pred = torch.sqrt(
        output[:, :, 1:2] ** 2.0 + output[:, :, 2:3] ** 2.0 + output[:, :, 3:4] ** 2.0
    )
    ws_true = torch.sqrt(
        target[:, :, 1:2] ** 2.0 + target[:, :, 2:3] ** 2.0 + target[:, :, 3:4] ** 2.0
    )

    masked_loss_ws = torch.mean(((ws_pred - ws_true) ** 2.0), (0, 1))

    masked_loss_pres = torch.mean(
        ((output[:, :, :1] - target[:, :, :1]) ** 2.0), (0, 1)
    )

    pres_x_true = target[:, :, :1] * normals[:, :, 0:1]
    pres_x_pred = output[:, :, :1] * normals[:, :, 0:1]

    masked_loss_pres_x = torch.mean(((pres_x_pred - pres_x_true) ** 2.0), (0, 1))

    ws_x_true = target[:, :, 1:2]
    ws_x_pred = output[:, :, 1:2]
    masked_loss_ws_x = torch.mean(((ws_x_pred - ws_x_true) ** 2.0), (0, 1))

    ws_y_true = target[:, :, 2:3]
    ws_y_pred = output[:, :, 2:3]
    masked_loss_ws_y = torch.mean(((ws_y_pred - ws_y_true) ** 2.0), (0, 1))

    ws_z_true = target[:, :, 3:4]
    ws_z_pred = output[:, :, 3:4]
    masked_loss_ws_z = torch.mean(((ws_z_pred - ws_z_true) ** 2.0), (0, 1))

    loss = (
        torch.mean(masked_loss_pres)
        + torch.mean(masked_loss_ws_x)
        + torch.mean(masked_loss_ws_y)
        + torch.mean(masked_loss_ws_z)
    )
    loss = loss / 4
    return loss


def relative_loss_fn_surface(output, target, normals, padded_value=-10):
    ws_pred = torch.sqrt(
        output[:, :, 1:2] ** 2.0 + output[:, :, 2:3] ** 2.0 + output[:, :, 3:4] ** 2.0
    )
    ws_true = torch.sqrt(
        target[:, :, 1:2] ** 2.0 + target[:, :, 2:3] ** 2.0 + target[:, :, 3:4] ** 2.0
    )

    masked_loss_ws = torch.mean(((ws_pred - ws_true) ** 2.0), (0, 1)) / torch.mean(
        ((ws_true) ** 2.0), (0, 1)
    )
    masked_loss_pres = torch.mean(
        ((output[:, :, :1] - target[:, :, :1]) ** 2.0), (0, 1)
    ) / torch.mean(((target[:, :, :1]) ** 2.0), (0, 1))

    pres_x_true = target[:, :, :1] * normals[:, :, 0:1]
    pres_x_pred = output[:, :, :1] * normals[:, :, 0:1]

    masked_loss_pres_x = torch.mean(
        ((pres_x_pred - pres_x_true) ** 2.0), (0, 1)
    ) / torch.mean(((pres_x_true) ** 2.0), (0, 1))

    ws_x_true = target[:, :, 1:2]
    ws_x_pred = output[:, :, 1:2]
    masked_loss_ws_x = torch.mean(
        ((ws_x_pred - ws_x_true) ** 2.0), (0, 1)
    ) / torch.mean(((ws_x_true) ** 2.0), (0, 1))

    ws_y_true = target[:, :, 2:3]
    ws_y_pred = output[:, :, 2:3]
    masked_loss_ws_y = torch.mean(
        ((ws_y_pred - ws_y_true) ** 2.0), (0, 1)
    ) / torch.mean(((ws_y_true) ** 2.0), (0, 1))

    ws_z_true = target[:, :, 3:4]
    ws_z_pred = output[:, :, 3:4]
    masked_loss_ws_z = torch.mean(
        ((ws_z_pred - ws_z_true) ** 2.0), (0, 1)
    ) / torch.mean(((ws_z_true) ** 2.0), (0, 1))

    loss = (
        torch.mean(masked_loss_pres)
        + torch.mean(masked_loss_ws_x)
        + torch.mean(masked_loss_ws_y)
        + torch.mean(masked_loss_ws_z)
    )
    loss = loss / 4
    return loss


def relative_loss_fn_area(output, target, normals, area, padded_value=-10):
    scale_factor = 1.0  # Get this from the dataset
    area = area * 10**4
    ws_pred = torch.sqrt(
        output[:, :, 1:2] ** 2.0 + output[:, :, 2:3] ** 2.0 + output[:, :, 3:4] ** 2.0
    )
    ws_true = torch.sqrt(
        target[:, :, 1:2] ** 2.0 + target[:, :, 2:3] ** 2.0 + target[:, :, 3:4] ** 2.0
    )

    masked_loss_ws = torch.mean(
        (
            (
                ws_pred * area * scale_factor**2.0
                - ws_true * area * scale_factor**2.0
            )
            ** 2.0
        ),
        (0, 1),
    ) / torch.mean(((ws_true * area) ** 2.0), (0, 1))
    masked_loss_pres = torch.mean(
        (
            (
                output[:, :, :1] * area * scale_factor**2.0
                - target[:, :, :1] * area * scale_factor**2.0
            )
            ** 2.0
        ),
        (0, 1),
    ) / torch.mean(((target[:, :, :1] * area) ** 2.0), (0, 1))

    pres_x_true = target[:, :, :1] * normals[:, :, 0:1] * area * scale_factor**2.0
    pres_x_pred = output[:, :, :1] * normals[:, :, 0:1] * area * scale_factor**2.0

    masked_loss_pres_x = torch.mean(
        ((pres_x_pred - pres_x_true) ** 2.0), (0, 1)
    ) / torch.mean(((pres_x_true) ** 2.0), (0, 1))

    ws_x_true = target[:, :, 1:2] * area * scale_factor**2.0
    ws_x_pred = output[:, :, 1:2] * area * scale_factor**2.0
    masked_loss_ws_x = torch.mean(
        ((ws_x_pred - ws_x_true) ** 2.0), (0, 1)
    ) / torch.mean(((ws_x_true) ** 2.0), (0, 1))

    ws_y_true = target[:, :, 2:3] * area * scale_factor**2.0
    ws_y_pred = output[:, :, 2:3] * area * scale_factor**2.0
    masked_loss_ws_y = torch.mean(
        ((ws_y_pred - ws_y_true) ** 2.0), (0, 1)
    ) / torch.mean(((ws_y_true) ** 2.0), (0, 1))

    ws_z_true = target[:, :, 3:4] * area * scale_factor**2.0
    ws_z_pred = output[:, :, 3:4] * area * scale_factor**2.0
    masked_loss_ws_z = torch.mean(
        ((ws_z_pred - ws_z_true) ** 2.0), (0, 1)
    ) / torch.mean(((ws_z_true) ** 2.0), (0, 1))

    loss = (
        torch.mean(masked_loss_pres_x)
        + torch.mean(masked_loss_ws_x)
        + torch.mean(masked_loss_ws_y)
        + torch.mean(masked_loss_ws_z)
    )
    loss = loss / 4
    return loss


def mse_loss_fn_area(output, target, normals, area, padded_value=-10):
    scale_factor = 1.0  # Get this from the dataset
    area = area * 10**4
    ws_pred = torch.sqrt(
        output[:, :, 1:2] ** 2.0 + output[:, :, 2:3] ** 2.0 + output[:, :, 3:4] ** 2.0
    )
    ws_true = torch.sqrt(
        target[:, :, 1:2] ** 2.0 + target[:, :, 2:3] ** 2.0 + target[:, :, 3:4] ** 2.0
    )

    masked_loss_ws = torch.mean(
        (
            (
                ws_pred * area * scale_factor**2.0
                - ws_true * area * scale_factor**2.0
            )
            ** 2.0
        ),
        (0, 1),
    )
    masked_loss_pres = torch.mean(
        (
            (
                output[:, :, :1] * area * scale_factor**2.0
                - target[:, :, :1] * area * scale_factor**2.0
            )
            ** 2.0
        ),
        (0, 1),
    )

    pres_x_true = target[:, :, :1] * normals[:, :, 0:1] * area * scale_factor**2.0
    pres_x_pred = output[:, :, :1] * normals[:, :, 0:1] * area * scale_factor**2.0

    masked_loss_pres_x = torch.mean(((pres_x_pred - pres_x_true) ** 2.0), (0, 1))

    ws_x_true = target[:, :, 1:2] * area * scale_factor**2.0
    ws_x_pred = output[:, :, 1:2] * area * scale_factor**2.0
    masked_loss_ws_x = torch.mean(((ws_x_pred - ws_x_true) ** 2.0), (0, 1))

    ws_y_true = target[:, :, 2:3] * area * scale_factor**2.0
    ws_y_pred = output[:, :, 2:3] * area * scale_factor**2.0
    masked_loss_ws_y = torch.mean(((ws_y_pred - ws_y_true) ** 2.0), (0, 1))

    ws_z_true = target[:, :, 3:4] * area * scale_factor**2.0
    ws_z_pred = output[:, :, 3:4] * area * scale_factor**2.0
    masked_loss_ws_z = torch.mean(((ws_z_pred - ws_z_true) ** 2.0), (0, 1))

    loss = (
        torch.mean(masked_loss_pres_x)
        + torch.mean(masked_loss_ws_x)
        + torch.mean(masked_loss_ws_y)
        + torch.mean(masked_loss_ws_z)
    )
    loss = loss / 4
    return loss


def integral_loss_fn(output, target, area, normals, padded_value=-10):
    vel_inlet = 30.0  # Get this from the dataset
    mask = abs(target - padded_value) > 1e-3
    area = torch.unsqueeze(area, -1)
    output_true = target * mask * area * (vel_inlet) ** 2.0
    output_pred = output * mask * area * (vel_inlet) ** 2.0

    output_true[:, :, 0] = output_true[:, :, 0] * normals[:, :, 0]
    output_pred[:, :, 0] = output_pred[:, :, 0] * normals[:, :, 0]

    masked_pred = torch.sum(output_pred, (1))
    masked_truth = torch.sum(output_true, (1))

    loss = (masked_pred - masked_truth) ** 2.0
    loss = torch.mean(loss)
    return loss


def integral_loss_fn_new(output, target, area, normals, padded_value=-10):
    drag_loss = drag_loss_fn(output, target, area, normals, padded_value=-10)
    lift_loss = lift_loss_fn(output, target, area, normals, padded_value=-10)
    return lift_loss + drag_loss


def lift_loss_fn(output, target, area, normals, padded_value=-10):
    vel_inlet = 30.0  # Get this from the dataset
    mask = abs(target - padded_value) > 1e-3
    area = torch.unsqueeze(area, -1)
    output_true = target * mask * area * (vel_inlet) ** 2.0
    output_pred = output * mask * area * (vel_inlet) ** 2.0

    pres_true = output_true[:, :, 0] * normals[:, :, 2]
    pres_pred = output_pred[:, :, 0] * normals[:, :, 2]

    wz_true = output_true[:, :, -1]
    wz_pred = output_pred[:, :, -1]

    masked_pred = torch.sum(pres_pred + wz_pred, (1)) / (
        torch.sum(area) * (vel_inlet) ** 2.0
    )
    masked_truth = torch.sum(pres_true + wz_true, (1)) / (
        torch.sum(area) * (vel_inlet) ** 2.0
    )

    loss = (masked_pred - masked_truth) ** 2.0
    loss = torch.mean(loss)
    return loss


def drag_loss_fn(output, target, area, normals, padded_value=-10):
    vel_inlet = 30.0  # Get this from the dataset
    mask = abs(target - padded_value) > 1e-3
    area = torch.unsqueeze(area, -1)
    output_true = target * mask * area * (vel_inlet) ** 2.0
    output_pred = output * mask * area * (vel_inlet) ** 2.0

    pres_true = output_true[:, :, 0] * normals[:, :, 0]
    pres_pred = output_pred[:, :, 0] * normals[:, :, 0]

    wx_true = output_true[:, :, 1]
    wx_pred = output_pred[:, :, 1]

    masked_pred = torch.sum(pres_pred + wx_pred, (1)) / (
        torch.sum(area) * (vel_inlet) ** 2.0
    )
    masked_truth = torch.sum(pres_true + wx_true, (1)) / (
        torch.sum(area) * (vel_inlet) ** 2.0
    )

    loss = (masked_pred - masked_truth) ** 2.0
    loss = torch.mean(loss)
    return loss


def validation_step(
    dataloader,
    model,
    device,
    use_sdf_basis=False,
    use_surface_normals=False,
    integral_scaling_factor=1.0,
    loss_fn_type="mse",
):
    running_vloss = 0.0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            sampled_batched = dict_to_device(sample_batched, device)

            prediction_vol, prediction_surf = model(sampled_batched)

            if prediction_vol is not None:
                target_vol = sampled_batched["volume_fields"]
                if loss_fn_type == "rmse":
                    loss_norm_vol = relative_loss_fn(
                        prediction_vol, target_vol, padded_value=-10
                    )
                else:
                    loss_norm_vol = mse_loss_fn(
                        prediction_vol, target_vol, padded_value=-10
                    )

            if prediction_surf is not None:
                target_surf = sampled_batched["surface_fields"]
                surface_normals = sampled_batched["surface_normals"]
                surface_areas = sampled_batched["surface_areas"]
                if loss_fn_type == "rmse":
                    loss_norm_surf = relative_loss_fn_surface(
                        prediction_surf, target_surf, surface_normals, padded_value=-10
                    )
                    loss_norm_surf_area = relative_loss_fn_area(
                        prediction_surf,
                        target_surf,
                        surface_normals,
                        surface_areas,
                        padded_value=-10,
                    )
                else:
                    loss_norm_surf = mse_loss_fn_surface(
                        prediction_surf, target_surf, surface_normals, padded_value=-10
                    )
                    loss_norm_surf_area = mse_loss_fn_area(
                        prediction_surf,
                        target_surf,
                        surface_normals,
                        surface_areas,
                        padded_value=-10,
                    )
                loss_integral = (
                    integral_loss_fn_new(
                        prediction_surf,
                        target_surf,
                        surface_areas,
                        surface_normals,
                        padded_value=-10,
                    )
                ) * integral_scaling_factor

            if prediction_surf is not None and prediction_vol is not None:
                vloss = (
                    loss_norm_vol
                    + 1.0 * loss_norm_surf
                    + loss_integral
                    + 0.0 * loss_norm_surf_area
                )
            elif prediction_vol is not None:
                vloss = loss_norm_vol
            elif prediction_surf is not None:
                vloss = 1.0 * loss_norm_surf + loss_integral + 0.0 * loss_norm_surf_area

            running_vloss += vloss

    avg_vloss = running_vloss / (i_batch + 1)

    return avg_vloss


def train_epoch(
    dataloader,
    model,
    optimizer,
    scaler,
    tb_writer,
    epoch_index,
    device,
    integral_scaling_factor,
    loss_fn_type,
):

    running_loss = 0.0
    last_loss = 0.0
    loss_interval = 1

    for i_batch, sample_batched in enumerate(dataloader):

        sampled_batched = dict_to_device(sample_batched, device)

        with autocast(enabled=False):
            prediction_vol, prediction_surf = model(sampled_batched)

            if prediction_vol is not None:
                target_vol = sampled_batched["volume_fields"]
                if loss_fn_type == "rmse":
                    loss_norm_vol = relative_loss_fn(
                        prediction_vol, target_vol, padded_value=-10
                    )
                else:
                    loss_norm_vol = mse_loss_fn(
                        prediction_vol, target_vol, padded_value=-10
                    )

            if prediction_surf is not None:

                target_surf = sampled_batched["surface_fields"]
                surface_areas = sampled_batched["surface_areas"]
                surface_normals = sampled_batched["surface_normals"]
                if loss_fn_type == "rmse":
                    loss_norm_surf = relative_loss_fn_surface(
                        prediction_surf, target_surf, surface_normals, padded_value=-10
                    )
                    loss_norm_surf_area = relative_loss_fn_area(
                        prediction_surf,
                        target_surf,
                        surface_normals,
                        surface_areas,
                        padded_value=-10,
                    )
                else:
                    loss_norm_surf = mse_loss_fn_surface(
                        prediction_surf, target_surf, surface_normals, padded_value=-10
                    )
                    loss_norm_surf_area = mse_loss_fn_area(
                        prediction_surf,
                        target_surf,
                        surface_normals,
                        surface_areas,
                        padded_value=-10,
                    )
                loss_integral = (
                    integral_loss_fn_new(
                        prediction_surf,
                        target_surf,
                        surface_areas,
                        surface_normals,
                        padded_value=-10,
                    )
                ) * integral_scaling_factor

            if prediction_vol is not None and prediction_surf is not None:
                loss_norm = (
                    loss_norm_vol
                    + 1.0 * loss_norm_surf
                    + loss_integral
                    + 0.0 * loss_norm_surf_area
                )
            elif prediction_vol is not None:
                loss_norm = loss_norm_vol
            elif prediction_surf is not None:
                loss_norm = (
                    0.5 * loss_norm_surf + loss_integral + 0.5 * loss_norm_surf_area
                )

        loss = loss_norm
        loss = loss / loss_interval
        scaler.scale(loss).backward()

        if ((i_batch + 1) % loss_interval == 0) or (i_batch + 1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        # Gather data and report
        running_loss += loss.item()

        if prediction_vol is not None and prediction_surf is not None:
            print(
                f"Device {device}, batch processed: {i_batch + 1}, loss volume: {loss_norm_vol:.5f} \
            , loss surface: {loss_norm_surf:.5f}, loss integral: {loss_integral:.5f}, loss surface area: {loss_norm_surf_area:.5f}"
            )
        elif prediction_vol is not None:
            print(
                f"Device {device}, batch processed: {i_batch + 1}, loss volume: {loss_norm_vol:.5f}"
            )
        elif prediction_surf is not None:
            print(
                f"Device {device}, batch processed: {i_batch + 1} \
            , loss surface: {loss_norm_surf:.5f}, loss integral: {loss_integral:.5f}, loss surface area: {loss_norm_surf_area:.5f}"
            )

    last_loss = running_loss / (i_batch + 1)  # loss per batch
    print(f" Device {device},  batch: {i_batch + 1}, loss norm: {loss:.5f}")
    tb_x = epoch_index * len(dataloader) + i_batch + 1
    tb_writer.add_scalar("Loss/train", last_loss, tb_x)

    return last_loss


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    model_type = cfg.model.model_type

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    print(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    num_vol_vars = 0
    volume_variable_names = []
    if model_type == "volume" or model_type == "combined":
        volume_variable_names = list(cfg.variables.volume.solution.keys())
        for j in volume_variable_names:
            if cfg.variables.volume.solution[j] == "vector":
                num_vol_vars += 3
            else:
                num_vol_vars += 1
    else:
        num_vol_vars = None

    num_surf_vars = 0
    surface_variable_names = []
    if model_type == "surface" or model_type == "combined":
        surface_variable_names = list(cfg.variables.surface.solution.keys())
        num_surf_vars = 0
        for j in surface_variable_names:
            if cfg.variables.surface.solution[j] == "vector":
                num_surf_vars += 3
            else:
                num_surf_vars += 1
    else:
        num_surf_vars = None

    vol_save_path = os.path.join(cfg.project_dir, "volume_scaling_factors.npy")
    surf_save_path = os.path.join(cfg.project_dir, "surface_scaling_factors.npy")
    if os.path.exists(vol_save_path) and os.path.exists(surf_save_path):
        vol_factors = np.load(vol_save_path)
        surf_factors = np.load(surf_save_path)
    else:
        vol_factors = None
        surf_factors = None

    train_dataset = create_domino_dataset(
        cfg,
        "train",
        volume_variable_names,
        surface_variable_names,
        vol_factors,
        surf_factors,
    )
    val_dataset = create_domino_dataset(
        cfg,
        "val",
        volume_variable_names,
        surface_variable_names,
        vol_factors,
        surf_factors,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.world_size,
        rank=dist.rank,
        **cfg.train.sampler,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.world_size,
        rank=dist.rank,
        **cfg.val.sampler,
    )

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, **cfg.train.dataloader
    )
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, **cfg.val.dataloader)

    model = DoMINO(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        model_parameters=cfg.model,
    ).to(dist.device)
    model = torch.compile(model, disable=True)  # TODO make this configurable

    # Print model summary (structure and parmeter count).
    print(f"Model summary:\n{torchinfo.summary(model, verbose=0, depth=2)}\n")

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

    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 150, 200, 250, 300, 350, 400], gamma=0.8
    )

    # Initialize the scaler for mixed precision
    scaler = GradScaler()

    writer = SummaryWriter(os.path.join(cfg.output, "tensorboard"))

    epoch_number = 0

    model_save_path = os.path.join(cfg.output, "models")
    param_save_path = os.path.join(cfg.output, "param")
    best_model_path = os.path.join(model_save_path, "best_model")
    if dist.rank == 0:
        create_directory(model_save_path)
        create_directory(param_save_path)
        create_directory(best_model_path)

    if dist.world_size > 1:
        torch.distributed.barrier()

    init_epoch = load_checkpoint(
        to_absolute_path(cfg.resume_dir),
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=dist.device,
    )

    if init_epoch != 0:
        init_epoch += 1  # Start with the next epoch
    epoch_number = init_epoch

    if epoch_number == 0:
        init_epoch = load_checkpoint(
            to_absolute_path(cfg.train.checkpoint_dir),
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=dist.device,
        )
        optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[25, 50, 75, 100, 250, 300, 350, 400], gamma=0.5
        )
        init_epoch = 0
        print("Pretrained checkpoint loaded ...")

    # retrive the smallest validation loss if available
    numbers = []
    for filename in os.listdir(best_model_path):
        match = re.search(r"\d+\.\d*[1-9]\d*", filename)
        if match:
            number = float(match.group(0))
            numbers.append(number)

    best_vloss = min(numbers) if numbers else 1_000_000.0

    initial_integral_factor_orig = cfg.model.integral_loss_scaling_factor

    for epoch in range(init_epoch, cfg.train.epochs):
        start_time = time.time()
        print(f"Device {dist.device}, epoch {epoch_number}:")

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        initial_integral_factor = initial_integral_factor_orig

        model.train(True)
        avg_loss = train_epoch(
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            tb_writer=writer,
            epoch_index=epoch,
            device=dist.device,
            integral_scaling_factor=initial_integral_factor,
            loss_fn_type=cfg.model.loss_function,
        )

        model.eval()
        avg_vloss = validation_step(
            dataloader=val_dataloader,
            model=model,
            device=dist.device,
            use_sdf_basis=cfg.model.use_sdf_in_basis_func,
            use_surface_normals=cfg.model.use_surface_normals,
            integral_scaling_factor=initial_integral_factor,
            loss_fn_type=cfg.model.loss_function,
        )

        scheduler.step()
        print(
            f"Device {dist.device} "
            f"LOSS train {avg_loss:.5f} "
            f"valid {avg_vloss:.5f} "
            f"Current lr {scheduler.get_last_lr()[0]}"
            f"Integral factor {initial_integral_factor}"
        )

        if dist.rank == 0:
            writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch_number,
            )
            writer.flush()

        # Track best performance, and save the model's state
        if dist.world_size > 1:
            torch.distributed.barrier()

        if avg_vloss < best_vloss:  # This only considers GPU: 0, is that okay?
            best_vloss = avg_vloss
            # if dist.rank == 0:
            save_checkpoint(
                to_absolute_path(best_model_path),
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=str(
                    best_vloss.item()
                ),  # hacky way of using epoch to store metadata
            )
        print(
            f"Device { dist.device}, Best val loss {best_vloss}, Time taken {time.time() - start_time}"
        )

        if dist.rank == 0 and (epoch + 1) % cfg.train.checkpoint_interval == 0.0:
            save_checkpoint(
                to_absolute_path(model_save_path),
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
            )

        epoch_number += 1

        if scheduler.get_last_lr()[0] == 1e-6:
            print("Training ended")
            exit()


if __name__ == "__main__":
    main()
