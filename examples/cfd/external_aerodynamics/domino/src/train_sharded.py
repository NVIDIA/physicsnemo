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

import apex
import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from typing import Literal

from physicsnemo.distributed import ShardTensor

from torch.cuda.amp import GradScaler, autocast

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)

from contextlib import nullcontext

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.tensor import distribute_module
from torch.utils.tensorboard import SummaryWriter
from nvtx import annotate as nvtx_annotate
import torch.cuda.nvtx as nvtx

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.datapipes.cae.domino_datapipe import (
    compute_scaling_factors,
    create_domino_dataset,
)
from physicsnemo.datapipes.cae.domino_sharded_datapipe import (
    create_sharded_domino_dataset,
)

from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils.domino.utils import *

# Bring these from the single-gpu script.
from train import (
    compute_loss_dict,
)

# This is included for GPU memory tracking:
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import time


from physicsnemo.utils.profiling import profile, Profiler


def validation_step(
    dataloader,
    model,
    device,
    use_sdf_basis=False,
    use_surface_normals=False,
    integral_scaling_factor=1.0,
    loss_fn_type=None,
    vol_loss_scaling=None,
    surf_loss_scaling=None,
):
    running_vloss = 0.0
    with torch.no_grad():
        for i_batch, sampled_batched in enumerate(dataloader):
            # sampled_batched = dict_to_device(sample_batched, device)

            with autocast(enabled=True):

                prediction_vol, prediction_surf = model(sampled_batched)
                loss, loss_dict = compute_loss_dict(
                    prediction_vol,
                    prediction_surf,
                    sampled_batched,
                    loss_fn_type,
                    integral_scaling_factor,
                    surf_loss_scaling,
                    vol_loss_scaling,
                )
            running_vloss += loss.full_tensor()

    avg_vloss = running_vloss / (i_batch + 1)

    return avg_vloss.item()


@profile
def train_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    tb_writer: SummaryWriter,
    logger: PythonLogger,
    gpu_handles: List[int],
    epoch_index: int,
    device: torch.device,
    integral_scaling_factor: float,
    loss_fn_type: Literal["mse", "rmse"],
    vol_loss_scaling: Optional[float] = None,
    surf_loss_scaling: Optional[float] = None,
) -> float:
    """
    Train a single epoch of the model.

    Args:
        dataloader: DataLoader for the training data, preprocessing w. DoMINO Pipeline
        model: DoMINO model to train
        optimizer: Optimizer for training
        scaler: GradScaler for mixed precision training
        tb_writer: SummaryWriter for logging to TensorBoard
        logger: PythonLogger for logging to console
        gpu_handles: List of GPU handles from pynvml for tracking GPU memory
        epoch_index: Index of the current epoch
        device: Device to run the model on
        integral_scaling_factor: Scaling factor for the integral loss
        loss_fn_type: Type of loss function to use
        vol_loss_scaling: Scaling factor for the volume loss
        surf_loss_scaling: Scaling factor for the surface loss

    Returns:
        Average loss for the epoch
    """

    dist = DistributedManager()

    running_loss = 0.0
    last_loss = 0.0
    loss_interval = 1

    gpu_start_info = [nvmlDeviceGetMemoryInfo(gpu_handle) for gpu_handle in gpu_handles]
    start_time = time.perf_counter()
    for i_batch, sample_batched in enumerate(dataloader):

        sampled_batched = sample_batched

        with autocast(enabled=True):
            with nvtx.range("Model Forward Pass"):
                prediction_vol, prediction_surf = model(sampled_batched)

            nvtx.range_push("Loss Calculation")
            # The loss calculation is the same as singel GPU
            loss, loss_dict = compute_loss_dict(
                prediction_vol,
                prediction_surf,
                sampled_batched,
                loss_fn_type,
                integral_scaling_factor,
                surf_loss_scaling,
                vol_loss_scaling,
            )

            loss = loss / loss_interval
            scaler.scale(loss).backward()

        if ((i_batch + 1) % loss_interval == 0) or (i_batch + 1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        # Gather data and report
        running_loss += loss.full_tensor().item()

        gpu_end_info = [
            nvmlDeviceGetMemoryInfo(gpu_handle) for gpu_handle in gpu_handles
        ]
        gpu_memory_used = [
            gpu_end_info.used / (1024**3) for gpu_end_info in gpu_end_info
        ]
        gpu_memory_delta = [
            (gpu_end_info.used - gpu_start_info.used) / (1024**3)
            for gpu_end_info, gpu_start_info in zip(gpu_end_info, gpu_start_info)
        ]
        elapsed_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        logging_string = f"Device {device}, batch processed: {i_batch + 1}\n"

        # Format the loss dict into a string (use full_tensor to reduce across the domain.):
        # **** Note ****
        # We have to use full_tensor to reduce across the domain.
        # You could use `.to_local()` to use just the local gpus version.
        # the full_tensor() reduction is only over the mesh domain.`
        loss_string = (
            "  "
            + "\t".join([f"{key.replace('loss_',''):<10}" for key in loss_dict.keys()])
            + "\n"
        )
        loss_string += (
            "  "
            + f"\t".join(
                [f"{l.full_tensor().item():<10.2e}" for l in loss_dict.values()]
            )
            + "\n"
        )
        logging_string += loss_string

        mem_used_str = " ".join(
            [f"{gpu_memory_used[i]:.2f}" for i in range(len(gpu_memory_used))]
        )
        mem_delta_str = " ".join(
            [f"{gpu_memory_delta[i]:.2f}" for i in range(len(gpu_memory_delta))]
        )
        logging_string += f"  GPU memory used: {mem_used_str} Gb\n"
        logging_string += f"  GPU memory delta: {mem_delta_str} Gb\n"
        logging_string += f"  Elapsed time: {elapsed_time:.2f} seconds\n"
        logger.info(logging_string)
        gpu_start_info = [
            nvmlDeviceGetMemoryInfo(gpu_handle) for gpu_handle in gpu_handles
        ]

    last_loss = running_loss / (i_batch + 1)  # loss per batch
    if dist.rank == 0:
        logger.info(
            f" Device {device},  batch: {i_batch + 1}, loss norm: {loss.full_tensor().item():.5f}"
        )
        tb_x = epoch_index * len(dataloader) + i_batch + 1
        tb_writer.add_scalar("Loss/train", last_loss, tb_x)

    return last_loss


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize NVML
    nvmlInit()

    # Use this to monitor GPU memory usage for visible GPUs:
    gpu_count = torch.cuda.device_count()
    # This will allocate a little memory on all visible GPUS:
    # Change to just the local GPU if you don't want that.
    gpu_handles = [
        nvmlDeviceGetHandleByIndex(dist.local_rank),
    ]
    # gpu_handles = [nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)]

    #################################
    # Mesh Creation
    # For Sharded training, we utilize pytorch's device mesh.
    # The distributed manager can create it for us.  We'll use a mesh
    # with two devices and the rest of the GPUs are the data-parallel
    # dimension.
    #################################

    # The global mesh represents all the GPUs in the process, in a multi-dimensional grid.
    # Think of the global mesh as a tensor, with rank = len(mesh_shape)
    domain_size = int(cfg.domain_parallelism.domain_size)
    # You can use -1 to one axis to indicate that you want to use all the GPUs in that dimension.
    mesh = dist.initialize_mesh(
        mesh_shape=(-1, domain_size), mesh_dim_names=("ddp", "domain")
    )
    # This is a subset of all the GPUs, and will vary depending on the process.
    # Think of this as slicing the global mesh along the domain axis.
    # It will contain only the GPUs that this process is sharing data with.
    domain_mesh = mesh["domain"]

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

    num_global_features = 0
    global_params_names = list(cfg.variables.global_parameters.keys())
    for param in global_params_names:
        if cfg.variables.global_parameters[param].type == "vector":
            num_global_features += len(cfg.variables.global_parameters[param].reference)
        elif cfg.variables.global_parameters[param].type == "scalar":
            num_global_features += 1
        else:
            raise ValueError(f"Unknown global parameter type")

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
        phase="train",
        volume_variable_names=volume_variable_names,
        surface_variable_names=surface_variable_names,
        vol_factors=vol_factors,
        surf_factors=surf_factors,
    )
    val_dataset = create_domino_dataset(
        cfg,
        phase="val",
        volume_variable_names=volume_variable_names,
        surface_variable_names=surface_variable_names,
        vol_factors=vol_factors,
        surf_factors=surf_factors,
    )

    #################################
    # Using a Sharded Dataset
    #################################
    # Physicsnemo has a built-in wrapper for the DoMino dataset
    # that allows for sharding the dataset across multiple GPUs.
    # (it's nothing fancy - each rank that shares data loads the entire image,
    # and then slices to it's own chunks)
    train_dataset = create_sharded_domino_dataset(
        train_dataset,
        domain_mesh,  # The dataloader needs to know the mesh for sharing data.
        shard_point_cloud=cfg.domain_parallelism.shard_points,  # We can shard the point
        shard_grid=cfg.domain_parallelism.shard_grid,  # Or the grid (or both)
    )

    val_dataset = create_sharded_domino_dataset(
        val_dataset,
        domain_mesh,
        shard_point_cloud=cfg.domain_parallelism.shard_points,
        shard_grid=cfg.domain_parallelism.shard_grid,
    )

    # The distributed sampler needs to know that the dataset is not
    # being used in a usual way.  We have to tell it how many "real"
    # times the dataset is sharded (world size / shard_size).
    # It also needs to know its rank in the global "ddp" dimension.
    sampler_num_replicas = mesh["ddp"].size()
    sampler_rank = mesh["ddp"].get_local_rank()

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=sampler_num_replicas,
        rank=sampler_rank,
        **cfg.train.sampler,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=sampler_num_replicas,
        rank=sampler_rank,
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
        global_features=num_global_features,
        model_parameters=cfg.model,
    ).to(dist.device)
    model = torch.compile(model, disable=True)  # TODO make this configurable

    # Print model summary (structure and parmeter count).
    logger.info(f"Model summary:\n{torchinfo.summary(model, verbose=0, depth=2)}\n")

    if dist.world_size > 1:
        # Instead of DDP, for sharding we use FSDP.  It's possible to use FSDP in the DDP
        # mode, but since it's not pure data parallel we have to me more careful.

        # First, distribute the model so that each GPU has the copy with DTensor weights:
        model = distribute_module(model, domain_mesh)

        model = FSDP(
            model,
            device_mesh=mesh["ddp"],
            sharding_strategy=ShardingStrategy.NO_SHARD,
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
            gpu_handles=gpu_handles,
            epoch_index=epoch,
            device=dist.device,
            integral_scaling_factor=initial_integral_factor,
            loss_fn_type=cfg.model.loss_function,
            vol_loss_scaling=cfg.model.vol_loss_scaling,
            surf_loss_scaling=surface_scaling_loss,
        )
        epoch_end_time = time.perf_counter()
        logger.info(
            f"Device {dist.device}, Epoch {epoch_number} took {epoch_end_time - epoch_start_time:.3f} seconds"
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
                {
                    "Training": avg_loss,
                    # "Validation": avg_vloss
                },
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
                epoch=str(best_vloss),  # hacky way of using epoch to store metadata
            )
        if dist.rank == 0:
            print(
                f"Device { dist.device}, Best val loss {best_vloss}, Time taken {time.perf_counter() - start_time:.3f}"
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
