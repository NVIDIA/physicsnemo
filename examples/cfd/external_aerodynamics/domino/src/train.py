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
This code defines a distributed pipeline for training the DoMINO model on
CFD datasets. It includes the computation of scaling factors, instantiating 
the DoMINO model and datapipe, automatically loading the most recent checkpoint, 
training the model in parallel using DistributedDataParallel across multiple 
GPUs, calculating the loss and updating model parameters using mixed precision. 
This is a common recipe that enables training of combined models for surface and 
volume as well either of them separately. Validation is also conducted every epoch, 
where predictions are compared against ground truth values. The code logs training
and validation metrics to TensorBoard. The train tab in config.yaml can be used to 
specify batch size, number of epochs and other training parameters.
"""

import time
import os
import re
import torch
import torchinfo

from typing import Literal

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
from nvtx import annotate as nvtx_annotate
import torch.cuda.nvtx as nvtx

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.datapipes.cae.domino_datapipe import (
    DoMINODataPipe,
    compute_scaling_factors,
    create_domino_dataset,
    # domino_collate_fn,
)
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils.domino.utils import *

# This is included for GPU memory tracking:
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import time

# Initialize NVML
nvmlInit()


from physicsnemo.utils.profiling import profile, Profiler

# Profiler().enable("line_profiler")
# Profiler().initialize()


def loss_fn(
    output: torch.Tensor,
    target: torch.Tensor,
    loss_type: Literal["mse", "rmse"],
    padded_value: float = -10,
) -> torch.Tensor:
    """Calculate mean squared error or root mean squared error with masking for padded values.

    Args:
        output: Predicted values from the model
        target: Ground truth values
        loss_type: Type of loss to calculate ("mse" or "rmse")
        padded_value: Value used for padding in the tensor

    Returns:
        Calculated loss as a scalar tensor
    """
    mask = abs(target - padded_value) > 1e-3

    if loss_type == "rmse":
        dims = (0, 1)
    else:
        dims = None

    num = torch.sum(mask * (output - target) ** 2.0, dims)
    if loss_type == "rmse":
        denom = torch.sum(mask * target**2.0, dims)
    else:
        denom = torch.sum(mask)

    return torch.mean(num / denom)


def loss_fn_surface(
    output: torch.Tensor, target: torch.Tensor, loss_type: Literal["mse", "rmse"]
) -> torch.Tensor:
    """Calculate loss for surface data by handling scalar and vector components separately.

    Args:
        output: Predicted surface values from the model
        target: Ground truth surface values
        loss_type: Type of loss to calculate ("mse" or "rmse")

    Returns:
        Combined scalar and vector loss as a scalar tensor
    """
    # Separate the scalar and vector components:
    output_scalar, output_vector = torch.split(output, [1, 3], dim=2)
    target_scalar, target_vector = torch.split(target, [1, 3], dim=2)

    numerator = torch.mean((output_scalar - target_scalar) ** 2.0)
    vector_diff_sq = torch.mean((target_vector - output_vector) ** 2.0, (0, 1))
    if loss_type == "mse":
        masked_loss_pres = numerator
        masked_loss_ws = torch.sum(vector_diff_sq)
    else:
        denom = torch.mean((target_scalar) ** 2.0)
        masked_loss_pres = numerator / denom

        # Compute the mean diff**2 of the vector component, leave the last dimension:
        masked_loss_ws_num = vector_diff_sq
        masked_loss_ws_denom = torch.mean((target_vector) ** 2.0, (0, 1))
        masked_loss_ws = torch.sum(masked_loss_ws_num / masked_loss_ws_denom)

    loss = masked_loss_pres + masked_loss_ws

    return loss / 4.0


def loss_fn_area(
    output: torch.Tensor,
    target: torch.Tensor,
    normals: torch.Tensor,
    area: torch.Tensor,
    area_scaling_factor: float,
    loss_type: Literal["mse", "rmse"],
) -> torch.Tensor:
    """Calculate area-weighted loss for surface data considering normal vectors.

    Args:
        output: Predicted surface values from the model
        target: Ground truth surface values
        normals: Normal vectors for the surface
        area: Area values for surface elements
        area_scaling_factor: Scaling factor for area weighting
        loss_type: Type of loss to calculate ("mse" or "rmse")

    Returns:
        Area-weighted loss as a scalar tensor
    """
    area = area * area_scaling_factor
    area_scale_factor = area

    # Separate the scalar and vector components.
    target_scalar, target_vector = torch.split(
        target * area_scale_factor, [1, 3], dim=2
    )
    output_scalar, output_vector = torch.split(
        output * area_scale_factor, [1, 3], dim=2
    )

    # Apply the normals to the scalar components (only [:,:,0]):
    normals, _ = torch.split(normals, [1, normals.shape[-1] - 1], dim=2)
    target_scalar = target_scalar * normals
    output_scalar = output_scalar * normals

    # Compute the mean diff**2 of the scalar component:
    masked_loss_pres = torch.mean(((output_scalar - target_scalar) ** 2.0), dim=(0, 1))
    if loss_type == "rmse":
        masked_loss_pres /= torch.mean(target_scalar**2.0, dim=(0, 1))

    # Compute the mean diff**2 of the vector component, leave the last dimension:
    masked_loss_ws = torch.mean((target_vector - output_vector) ** 2.0, (0, 1))

    if loss_type == "rmse":
        masked_loss_ws /= torch.mean((target_vector) ** 2.0, (0, 1))

    # Combine the scalar and vector components:
    loss = 0.25 * (masked_loss_pres + torch.sum(masked_loss_ws))

    return loss


def integral_loss_fn(
    output, target, area, normals, stream_velocity=None, padded_value=-10
):
    drag_loss = drag_loss_fn(
        output, target, area, normals, stream_velocity, padded_value=-10
    )
    lift_loss = lift_loss_fn(
        output, target, area, normals, stream_velocity, padded_value=-10
    )
    return lift_loss + drag_loss


def lift_loss_fn(output, target, area, normals, stream_velocity=None, padded_value=-10):
    vel_inlet = stream_velocity  # Get this from the dataset
    mask = abs(target - padded_value) > 1e-3

    output_true = target * mask * area * (vel_inlet) ** 2.0
    output_pred = output * mask * area * (vel_inlet) ** 2.0

    pres_true = output_true[:, :, 0] * normals[:, :, 2]
    pres_pred = output_pred[:, :, 0] * normals[:, :, 2]

    wz_true = output_true[:, :, -1]
    wz_pred = output_pred[:, :, -1]

    masked_pred = torch.mean(pres_pred + wz_pred, (1))
    masked_truth = torch.mean(pres_true + wz_true, (1))

    loss = (masked_pred - masked_truth) ** 2.0
    loss = torch.mean(loss)
    return loss


def drag_loss_fn(output, target, area, normals, stream_velocity=None, padded_value=-10):
    vel_inlet = stream_velocity  # Get this from the dataset
    mask = abs(target - padded_value) > 1e-3
    output_true = target * mask * area * (vel_inlet) ** 2.0
    output_pred = output * mask * area * (vel_inlet) ** 2.0

    pres_true = output_true[:, :, 0] * normals[:, :, 0]
    pres_pred = output_pred[:, :, 0] * normals[:, :, 0]

    wx_true = output_true[:, :, 1]
    wx_pred = output_pred[:, :, 1]

    masked_pred = torch.mean(pres_pred + wx_pred, (1))
    masked_truth = torch.mean(pres_true + wx_true, (1))

    loss = (masked_pred - masked_truth) ** 2.0
    loss = torch.mean(loss)
    return loss


def validation_step(
    dataloader,
    model,
    device,
    logger,
    use_sdf_basis=False,
    use_surface_normals=False,
    integral_scaling_factor=1.0,
    loss_fn_type=None,
    vol_loss_scaling=None,
    surf_loss_scaling=None,
):
    running_vloss = 0.0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            sampled_batched = dict_to_device(sample_batched, device)

            with autocast(enabled=True):

                prediction_vol, prediction_surf = model(sampled_batched)
                total_loss_terms = []
                if prediction_vol is not None:
                    target_vol = sampled_batched["volume_fields"]

                    alternate_loss_vol = loss_fn(
                        prediction_vol,
                        target_vol,
                        loss_fn_type.loss_type,
                        padded_value=-10,
                    )
                    total_loss_terms.append(alternate_loss_vol)

                if prediction_surf is not None:
                    target_surf = sampled_batched["surface_fields"]
                    surface_normals = sampled_batched["surface_normals"]
                    surface_areas = sampled_batched["surface_areas"]
                    stream_velocity = sampled_batched["stream_velocity"]
                    surface_areas = torch.unsqueeze(surface_areas, -1)

                    loss_integral = (
                        integral_loss_fn(
                            prediction_surf,
                            target_surf,
                            surface_areas,
                            surface_normals,
                            stream_velocity,
                            padded_value=-10,
                        )
                    ) * integral_scaling_factor  # * 0.0

                    alternate_loss_surf = loss_fn_surface(
                        prediction_surf,
                        target_surf,
                        loss_fn_type.loss_type,
                    )
                    alternate_loss_surf_area = loss_fn_area(
                        prediction_surf,
                        target_surf,
                        surface_normals,
                        surface_areas,
                        area_scaling_factor=loss_fn_type.area_weighing_factor,
                        loss_type=loss_fn_type.loss_type,
                    )
                    if loss_fn_type.loss_type == "mse":
                        alternate_loss_surf = alternate_loss_surf * surf_loss_scaling
                        alternate_loss_surf_area = (
                            alternate_loss_surf_area * surf_loss_scaling
                        )

                    total_loss_terms.append(0.5 * alternate_loss_surf)
                    total_loss_terms.append(0.5 * alternate_loss_surf_area)
                    total_loss_terms.append(loss_integral)

                total_loss = sum(total_loss_terms)

            running_vloss += total_loss.item()

    avg_vloss = running_vloss / (i_batch + 1)

    return avg_vloss


@profile
def train_epoch(
    dataloader,
    model,
    optimizer,
    scaler,
    tb_writer,
    logger,
    gpu_handle,
    epoch_index,
    device,
    integral_scaling_factor,
    loss_fn_type,
    vol_loss_scaling=None,
    surf_loss_scaling=None,
):

    dist = DistributedManager()

    running_loss = 0.0
    last_loss = 0.0
    loss_interval = 1

    gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)
    for i_batch, sample_batched in enumerate(dataloader):

        sampled_batched = dict_to_device(sample_batched, device)

        with autocast(enabled=True):
            with nvtx.range("Model Forward Pass"):
                prediction_vol, prediction_surf = model(sampled_batched)
            total_loss_terms = []
            nvtx.range_push("Loss Calculation")
            if prediction_vol is not None:
                target_vol = sampled_batched["volume_fields"]

                alternate_loss_vol = loss_fn(
                    prediction_vol, target_vol, loss_fn_type.loss_type, padded_value=-10
                )
                total_loss_terms.append(alternate_loss_vol)

            if prediction_surf is not None:

                target_surf = sampled_batched["surface_fields"]
                surface_areas = sampled_batched["surface_areas"]
                surface_areas = torch.unsqueeze(surface_areas, -1)
                surface_normals = sampled_batched["surface_normals"]
                stream_velocity = sampled_batched["stream_velocity"]
                alternate_loss_surf = loss_fn_surface(
                    prediction_surf,
                    target_surf,
                    loss_fn_type.loss_type,
                )
                alternate_loss_surf_area = loss_fn_area(
                    prediction_surf,
                    target_surf,
                    surface_normals,
                    surface_areas,
                    area_scaling_factor=loss_fn_type.area_weighing_factor,
                    loss_type=loss_fn_type.loss_type,
                )

                if loss_fn_type.loss_type == "mse":
                    alternate_loss_surf = alternate_loss_surf * surf_loss_scaling
                    alternate_loss_surf_area = (
                        alternate_loss_surf_area * surf_loss_scaling
                    )

                total_loss_terms.append(0.5 * alternate_loss_surf)
                total_loss_terms.append(0.5 * alternate_loss_surf_area)
                loss_integral = (
                    integral_loss_fn(
                        prediction_surf,
                        target_surf,
                        surface_areas,
                        surface_normals,
                        stream_velocity,
                        padded_value=-10,
                    )
                ) * integral_scaling_factor  # * 0.0
                total_loss_terms.append(loss_integral)

            total_loss = sum(total_loss_terms)

            nvtx.range_pop()

        # loss = loss_norm
        loss = total_loss
        loss = loss / loss_interval
        scaler.scale(loss).backward()

        if ((i_batch + 1) % loss_interval == 0) or (i_batch + 1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        # Gather data and report
        running_loss += loss.item()

        gpu_end_info = nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_memory_used = gpu_end_info.used / (1024**3)
        gpu_memory_delta = (gpu_end_info.used - gpu_start_info.used) / (1024**3)

        logging_string = f"Device {device}, batch processed: {i_batch + 1}\n"
        logging_string += f"  total loss: {total_loss.item():.5f}\n"
        if prediction_vol is not None:
            logging_string += f"    loss volume: {alternate_loss_vol.item():.5f}\n"
        if prediction_surf is not None:
            logging_string += f"    loss surface: {alternate_loss_surf.item():.5f}\n"
            logging_string += (
                f"    loss surface area: {alternate_loss_surf_area.item():.5f}\n"
            )
            logging_string += f"    loss integral: {loss_integral.item():.5f}\n"
        logging_string += f"  GPU memory used: {gpu_memory_used} Gb\n"
        logging_string += f"  GPU memory delta: {gpu_memory_delta} Gb\n"
        logger.info(logging_string)
        gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)

    last_loss = running_loss / (i_batch + 1)  # loss per batch
    if dist.rank == 0:
        logger.info(
            f" Device {device},  batch: {i_batch + 1}, loss norm: {loss.item():.5f}"
        )
        tb_x = epoch_index * len(dataloader) + i_batch + 1
        tb_writer.add_scalar("Loss/train", last_loss, tb_x)

    return last_loss


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    gpu_handle = nvmlDeviceGetHandleByIndex(dist.device.index)

    compute_scaling_factors(
        cfg, cfg.data_processor.output_dir, use_cache=cfg.data_processor.use_cache
    )
    model_type = cfg.model.model_type

    logger = PythonLogger("Train")
    logger = RankZeroLoggingWrapper(logger, dist)

    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

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

    vol_save_path = os.path.join(
        "outputs", cfg.project.name, "volume_scaling_factors.npy"
    )
    surf_save_path = os.path.join(
        "outputs", cfg.project.name, "surface_scaling_factors.npy"
    )
    if os.path.exists(vol_save_path):
        vol_factors = np.load(vol_save_path)
    else:
        vol_factors = None

    if os.path.exists(surf_save_path):
        surf_factors = np.load(surf_save_path)
    else:
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
        train_dataset,
        sampler=train_sampler,
        **cfg.train.dataloader,
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        **cfg.val.dataloader,
    )

    model = DoMINO(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        model_parameters=cfg.model,
    ).to(dist.device)
    model = torch.compile(model, disable=True)  # TODO make this configurable

    # Print model summary (structure and parmeter count).
    logger.info(f"Model summary:\n{torchinfo.summary(model, verbose=0, depth=2)}\n")

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

    # optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 200, 300, 400, 500, 600, 700, 800], gamma=0.5
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
        start_time = time.perf_counter()
        logger.info(f"Device {dist.device}, epoch {epoch_number}:")

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        initial_integral_factor = initial_integral_factor_orig

        if epoch > 250:
            surface_scaling_loss = 1.0 * cfg.model.surf_loss_scaling
        else:
            surface_scaling_loss = cfg.model.surf_loss_scaling

        model.train(True)
        epoch_start_time = time.perf_counter()
        avg_loss = train_epoch(
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            tb_writer=writer,
            logger=logger,
            gpu_handle=gpu_handle,
            epoch_index=epoch,
            device=dist.device,
            integral_scaling_factor=initial_integral_factor,
            loss_fn_type=cfg.model.loss_function,
            vol_loss_scaling=cfg.model.vol_loss_scaling,
            surf_loss_scaling=surface_scaling_loss,
        )
        epoch_end_time = time.perf_counter()
        logger.info(
            f"Device {dist.device}, Epoch {epoch_number} took {epoch_end_time - epoch_start_time} seconds"
        )
        epoch_end_time = time.perf_counter()

        model.eval()
        avg_vloss = validation_step(
            dataloader=val_dataloader,
            model=model,
            device=dist.device,
            logger=logger,
            use_sdf_basis=cfg.model.use_sdf_in_basis_func,
            use_surface_normals=cfg.model.use_surface_normals,
            integral_scaling_factor=initial_integral_factor,
            loss_fn_type=cfg.model.loss_function,
            vol_loss_scaling=cfg.model.vol_loss_scaling,
            surf_loss_scaling=surface_scaling_loss,
        )

        scheduler.step()
        logger.info(
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

        print(f"Device { dist.device}, Best val loss {best_vloss}")

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
