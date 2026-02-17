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
from typing import Literal, Any
from tabulate import tabulate

import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# This will set up the cupy-ecosystem and pytorch to share memory pools
from physicsnemo.utils.memory import unified_gpu_memory

import torchinfo
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import distribute_module

from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from nvtx import annotate as nvtx_annotate
import torch.cuda.nvtx as nvtx


from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.datapipes.cae.domino_datapipe import (
    DoMINODataPipe,
    create_domino_dataset,
)
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.models.domino.utils import create_directory

from utils import ScalingFactors, get_keys_to_read, coordinate_distributed_environment

# This is included for GPU memory tracking:
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import time


# Initialize NVML
nvmlInit()


from physicsnemo.utils.profiling import profile, Profiler


from loss import compute_loss_dict
from utils import get_num_vars, load_scaling_factors, compute_l2, all_reduce_dict


def validation_step(
    dataloader,
    model,
    device,
    logger,
    tb_writer,
    epoch_index,
    use_sdf_basis=False,
    use_surface_normals=False,
    integral_scaling_factor=1.0,
    loss_fn_type=None,
    vol_loss_scaling=None,
    surf_loss_scaling=None,
    first_deriv: torch.nn.Module | None = None,
    eqn: Any = None,
    bounding_box: torch.Tensor | None = None,
    vol_factors: torch.Tensor | None = None,
    add_physics_loss=False,
    autocast_enabled=None,
):
    dm = DistributedManager()
    running_vloss = 0.0
    with torch.no_grad():
        metrics = None

        for i_batch, sampled_batched in enumerate(dataloader):
            with autocast("cuda", enabled=autocast_enabled, cache_enabled=False):
                if add_physics_loss:
                    prediction_vol, prediction_surf = model(
                        sampled_batched, return_volume_neighbors=True
                    )
                else:
                    prediction_vol, prediction_surf = model(sampled_batched)

                loss, loss_dict = compute_loss_dict(
                    prediction_vol,
                    prediction_surf,
                    sampled_batched,
                    loss_fn_type,
                    integral_scaling_factor,
                    surf_loss_scaling,
                    vol_loss_scaling,
                    first_deriv,
                    eqn,
                    bounding_box,
                    vol_factors,
                    add_physics_loss,
                )

            running_vloss += loss.item()
            local_metrics = compute_l2(
                prediction_surf, prediction_vol, sampled_batched, dataloader
            )
            if metrics is None:
                metrics = local_metrics
            else:
                metrics = {
                    key: metrics[key] + local_metrics[key] for key in metrics.keys()
                }

    avg_vloss = running_vloss / (i_batch + 1)
    metrics = {key: metrics[key] / (i_batch + 1) for key in metrics.keys()}

    metrics = all_reduce_dict(metrics, dm)

    if dm.rank == 0:
        logger.info(
            f" Device {device},  batch: {i_batch + 1}, VAL loss norm: {loss.detach().item():.5f}"
        )
        tb_x = epoch_index
        for key in metrics.keys():
            tb_writer.add_scalar(f"L2 Metrics/val/{key}", metrics[key], tb_x)

        metrics_table = tabulate(
            [[k, v] for k, v in metrics.items()],
            headers=["Metric", "Average Value"],
            tablefmt="pretty",
        )
        logger.info(
            f"\nEpoch {epoch_index} VALIDATION Average Metrics:\n{metrics_table}\n"
        )

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
    first_deriv: torch.nn.Module | None = None,
    eqn: Any = None,
    bounding_box: torch.Tensor | None = None,
    vol_factors: torch.Tensor | None = None,
    surf_factors: torch.Tensor | None = None,
    add_physics_loss=False,
    autocast_enabled=None,
    grad_clip_enabled=None,
    grad_max_norm=None,
):
    dm = DistributedManager()

    running_loss = 0.0
    last_loss = 0.0
    loss_interval = 1

    gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)
    start_time = time.perf_counter()
    with Profiler():
        io_start_time = time.perf_counter()
        metrics = None
        for i_batch, sampled_batched in enumerate(dataloader):
            io_end_time = time.perf_counter()
            if add_physics_loss:
                autocast_enabled = False

            with autocast("cuda", enabled=autocast_enabled, cache_enabled=False):
                with nvtx.range("Model Forward Pass"):
                    if add_physics_loss:
                        prediction_vol, prediction_surf = model(
                            sampled_batched, return_volume_neighbors=True
                        )
                    else:
                        prediction_vol, prediction_surf = model(sampled_batched)

                loss, loss_dict = compute_loss_dict(
                    prediction_vol,
                    prediction_surf,
                    sampled_batched,
                    loss_fn_type,
                    integral_scaling_factor,
                    surf_loss_scaling,
                    vol_loss_scaling,
                    first_deriv,
                    eqn,
                    bounding_box,
                    vol_factors,
                    add_physics_loss,
                )

                # Compute metrics:
                if isinstance(prediction_vol, tuple):
                    # This is if return_neighbors is on for volume:
                    prediction_vol = prediction_vol[0]

                local_metrics = compute_l2(
                    prediction_surf, prediction_vol, sampled_batched, dataloader
                )
                if metrics is None:
                    metrics = local_metrics
                else:
                    # Sum the running total:
                    metrics = {
                        key: metrics[key] + local_metrics[key] for key in metrics.keys()
                    }

            loss = loss / loss_interval
            scaler.scale(loss).backward()

            if ((i_batch + 1) % loss_interval == 0) or (i_batch + 1 == len(dataloader)):
                if grad_clip_enabled:
                    # Unscales the gradients of optimizer's assigned params in-place.
                    scaler.unscale_(optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Gather data and report
            running_loss += loss.detach().item()
            elapsed_time = time.perf_counter() - start_time
            io_time = io_end_time - io_start_time
            start_time = time.perf_counter()
            gpu_end_info = nvmlDeviceGetMemoryInfo(gpu_handle)
            gpu_memory_used = gpu_end_info.used / (1024**3)
            gpu_memory_delta = (gpu_end_info.used - gpu_start_info.used) / (1024**3)

            logging_string = f"Device {device}, batch processed: {i_batch + 1}\n"
            # Format the loss dict into a string:
            loss_string = (
                "  "
                + "\t".join(
                    [f"{key.replace('loss_', ''):<10}" for key in loss_dict.keys()]
                )
                + "\n"
            )
            loss_string += (
                "  "
                + f"\t".join(
                    [f"{l.detach().item():<10.3e}" for l in loss_dict.values()]
                )
                + "\n"
            )

            logging_string += loss_string
            logging_string += f"  GPU memory used: {gpu_memory_used:.3f} Gb (delta: {gpu_memory_delta:.3f})\n"
            logging_string += f"  Timings: (IO: {io_time:.2f}, Model: {elapsed_time - io_time:.2f}, Total: {elapsed_time:.2f})s\n"
            logger.info(logging_string)
            gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)
            io_start_time = time.perf_counter()

    last_loss = running_loss / (i_batch + 1)  # loss per batch
    # Normalize metrics:
    metrics = {key: metrics[key] / (i_batch + 1) for key in metrics.keys()}
    # reduce metrics across batch:
    metrics = all_reduce_dict(metrics, dm)
    if dm.rank == 0:
        logger.info(
            f" Device {device},  batch: {i_batch + 1}, loss norm: {loss.detach().item():.5f}"
        )
        tb_x = epoch_index * len(dataloader) + i_batch + 1
        tb_writer.add_scalar("Loss/train", last_loss, tb_x)
        for key in metrics.keys():
            tb_writer.add_scalar(f"L2 Metrics/train/{key}", metrics[key], epoch_index)

        metrics_table = tabulate(
            [[k, v] for k, v in metrics.items()],
            headers=["Metric", "Average Value"],
            tablefmt="pretty",
        )
        logger.info(f"\nEpoch {epoch_index} Average Metrics:\n{metrics_table}\n")

    return last_loss


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    ######################################################
    # initialize distributed manager
    ######################################################
    DistributedManager.initialize()
    dist = DistributedManager()

    # DoMINO supports domain parallel training.  This function helps coordinate
    # how to set that up, if needed.
    domain_mesh, data_mesh, placements = coordinate_distributed_environment(cfg)

    if data_mesh is not None:
        data_replica_size = data_mesh.size()
        data_rank = data_mesh.get_local_rank()
    else:
        data_replica_size = dist.world_size
        data_rank = dist.rank

    ################################
    # Initialize NVML
    ################################
    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(dist.device.index)

    ######################################################
    # Initialize logger
    ######################################################

    logger = PythonLogger("Train")
    logger = RankZeroLoggingWrapper(logger, dist)

    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    ######################################################
    # Get scaling factors - precompute them if this fails!
    ######################################################
    vol_factors, surf_factors = load_scaling_factors(cfg)

    ######################################################
    # Configure the model
    ######################################################
    model_type = cfg.model.model_type
    num_vol_vars, num_surf_vars, num_global_features = get_num_vars(cfg, model_type)

    if model_type == "combined" or model_type == "surface":
        surface_variable_names = list(cfg.variables.surface.solution.keys())
    else:
        surface_variable_names = []

    if model_type == "combined" or model_type == "volume":
        volume_variable_names = list(cfg.variables.volume.solution.keys())
    else:
        volume_variable_names = []

    ######################################################
    # Configure physics loss
    # Unless enabled, these are null-ops
    ######################################################
    add_physics_loss = getattr(cfg.train, "add_physics_loss", False)

    if add_physics_loss:
        from physicsnemo.sym.eq.pde import PDE
        from physicsnemo.sym.eq.ls.grads import FirstDeriv
        from physicsnemo.sym.eq.pdes.navier_stokes import IncompressibleNavierStokes
    else:
        PDE = FirstDeriv = IncompressibleNavierStokes = None

    # Initialize physics components conditionally
    first_deriv = None
    eqn = None
    if add_physics_loss:
        first_deriv = FirstDeriv(dim=3, direct_input=True)
        eqn = IncompressibleNavierStokes(rho=1.226, nu="nu", dim=3, time=False)
        eqn = eqn.make_nodes(return_as_dict=True)

    # The bounding box is used in calculating the physics loss:
    bounding_box = None
    if add_physics_loss:
        bounding_box = cfg.data.bounding_box
        bounding_box = (
            torch.from_numpy(
                np.stack([bounding_box["max"], bounding_box["min"]], axis=0)
            )
            .to(vol_factors.dtype)
            .to(dist.device)
        )

    ######################################################
    # Configure the dataset
    ######################################################

    # This helper function is to determine which keys to read from the data
    # (and which to use default values for, if they aren't present - like
    # air_density, for example)
    keys_to_read, keys_to_read_if_available = get_keys_to_read(
        cfg, model_type, get_ground_truth=True
    )

    # The dataset actually works in two pieces
    # The core dataset just reads data from disk, and puts it on the GPU if needed.
    # The data processesing pipeline will preprocess that data and prepare it for the model.
    # Obviously, you need both, so this function will return the datapipeline in
    # a way that can be iterated over.
    #
    # To properly shuffle the data, we use a distributed sampler too.
    # It's configured properly for optional domain parallelism, and you have
    # to make sure to call set_epoch below.

    train_dataloader = create_domino_dataset(
        cfg,
        phase="train",
        keys_to_read=keys_to_read,
        keys_to_read_if_available=keys_to_read_if_available,
        vol_factors=vol_factors,
        surf_factors=surf_factors,
        device_mesh=domain_mesh,
        placements=placements,
        normalize_coordinates=cfg.data.normalize_coordinates,
        sample_in_bbox=cfg.data.sample_in_bbox,
        sampling=cfg.data.sampling,
    )
    train_sampler = DistributedSampler(
        train_dataloader,
        num_replicas=data_replica_size,
        rank=data_rank,
        **cfg.train.sampler,
    )

    val_dataloader = create_domino_dataset(
        cfg,
        phase="val",
        keys_to_read=keys_to_read,
        keys_to_read_if_available=keys_to_read_if_available,
        vol_factors=vol_factors,
        surf_factors=surf_factors,
        device_mesh=domain_mesh,
        placements=placements,
        normalize_coordinates=cfg.data.normalize_coordinates,
        sample_in_bbox=cfg.data.sample_in_bbox,
        sampling=cfg.data.sampling,
    )
    val_sampler = DistributedSampler(
        val_dataloader,
        num_replicas=data_replica_size,
        rank=data_rank,
        **cfg.val.sampler,
    )

    ######################################################
    # Configure the model
    ######################################################
    model = DoMINO(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        global_features=num_global_features,
        model_parameters=cfg.model,
    ).to(dist.device)

    # Print model summary (structure and parmeter count).
    logger.info(f"Model summary:\n{torchinfo.summary(model, verbose=0, depth=2)}\n")

    if dist.world_size > 1:
        if domain_mesh is None:
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        else:
            model = distribute_module(
                model,
                device_mesh=domain_mesh,
            )
            model = fully_shard(model, mesh=data_mesh)

    ######################################################
    # Initialize optimzer and gradient scaler
    ######################################################

    optimizer_class = None
    if cfg.train.optimizer.name == "Adam":
        optimizer_class = torch.optim.Adam
    elif cfg.train.optimizer.name == "AdamW":
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer.name}")
    optimizer = optimizer_class(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        weight_decay=cfg.train.optimizer.weight_decay,
    )
    if cfg.train.lr_scheduler.name == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.train.lr_scheduler.milestones,
            gamma=cfg.train.lr_scheduler.gamma,
        )
    elif cfg.train.lr_scheduler.name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.lr_scheduler.T_max,
            eta_min=cfg.train.lr_scheduler.eta_min,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.train.lr_scheduler.name}")

    # Initialize the scaler for mixed precision
    scaler = GradScaler()

    ######################################################
    # Initialize output tools
    ######################################################

    # Tensorboard Writer to track training.
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

    ######################################################
    # Load checkpoint if available
    ######################################################
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

    ######################################################
    # Begin Training loop over epochs
    ######################################################

    for epoch in range(init_epoch, cfg.train.epochs):
        start_time = time.perf_counter()
        logger.info(f"Device {dist.device}, epoch {epoch_number}:")

        if epoch == init_epoch and add_physics_loss:
            logger.info(
                "Physics loss enabled - mixed precision (autocast) will be disabled as physics loss computation is not supported with mixed precision"
            )

        # This controls what indices to use for each epoch.
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        train_dataloader.dataset.set_indices(list(train_sampler))
        val_dataloader.dataset.set_indices(list(val_sampler))

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
            first_deriv=first_deriv,
            eqn=eqn,
            bounding_box=bounding_box,
            vol_factors=vol_factors,
            add_physics_loss=add_physics_loss,
            autocast_enabled=cfg.train.amp.enabled,
            grad_clip_enabled=cfg.train.amp.clip_grad,
            grad_max_norm=cfg.train.amp.grad_max_norm,
        )
        epoch_end_time = time.perf_counter()
        logger.info(
            f"Device {dist.device}, Epoch {epoch_number} took {epoch_end_time - epoch_start_time:.3f} seconds"
        )
        epoch_end_time = time.perf_counter()

        model.eval()
        avg_vloss = validation_step(
            dataloader=val_dataloader,
            model=model,
            device=dist.device,
            logger=logger,
            tb_writer=writer,
            epoch_index=epoch,
            use_sdf_basis=cfg.model.use_sdf_in_basis_func,
            use_surface_normals=cfg.model.use_surface_normals,
            integral_scaling_factor=initial_integral_factor,
            loss_fn_type=cfg.model.loss_function,
            vol_loss_scaling=cfg.model.vol_loss_scaling,
            surf_loss_scaling=surface_scaling_loss,
            first_deriv=first_deriv,
            eqn=eqn,
            bounding_box=bounding_box,
            vol_factors=vol_factors,
            add_physics_loss=add_physics_loss,
            autocast_enabled=cfg.train.amp.enabled,
        )

        scheduler.step()
        logger.info(
            f"Device {dist.device} "
            f"LOSS train {avg_loss:.5f} "
            f"valid {avg_vloss:.5f} "
            f"Current lr {scheduler.get_last_lr()[0]} "
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

        if dist.rank == 0:
            print(f"Device {dist.device}, Best val loss {best_vloss}")

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
