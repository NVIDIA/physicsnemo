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

import argparse
import importlib
import logging
import random
import time

import numpy as np
import torch
import zarr
from data import HRRRSurfaceDataset
from nn import HRRRUnconditionalUNet
from tensordict import TensorDict
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from physicsnemo.diffusion.multi_diffusion import (
    MultiDiffusionModel2D,
    MultiDiffusionMSEDSMLoss,
)
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
from physicsnemo.diffusion.preconditioners import EDMPreconditioner
from physicsnemo.diffusion.utils.utils import InfiniteSampler
from physicsnemo.distributed import DistributedManager
from physicsnemo.distributed.utils import reduce_loss
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

# Compilation settings
torch._dynamo.reset()
torch._dynamo.config.cache_size_limit = 264
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = False
torch._logging.set_logs(recompiles=True, graph_breaks=True)


def parse_args():
    """Parse command-line arguments with sensible defaults."""
    p = argparse.ArgumentParser(description="HRRR surface diffusion SDA training")
    p.add_argument("--load-checkpoint", action="store_true", default=False)
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    p.add_argument("--batch-size-per-gpu", type=int, default=2)
    p.add_argument("--P-mean", type=float, default=-0.8)
    p.add_argument("--P-std", type=float, default=1.6)
    return p.parse_args()


def main():
    args = parse_args()

    # Configuration
    img_resolution = [1059, 1799]
    img_channels = 16
    num_condition_channels = 3
    batch_size_per_gpu = args.batch_size_per_gpu
    num_patches_per_sample = 4
    patch_shape = (448, 448)
    load_checkpoint_from_file = args.load_checkpoint
    checkpoint_dir = args.checkpoint_dir
    max_training_samples = 10000000
    checkpoint_frequency = 100000
    validation_frequency = 1000
    num_validation_samples = 100
    logging_frequency = 1000
    use_apex = True

    # Set seeds for reproducibility
    seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)

    # Initialize distributed environment
    DistributedManager.initialize()
    dist = DistributedManager()

    # Setup logging
    logger = PythonLogger("main")
    logger.logger.setLevel(logging.INFO)
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)

    # Setup model
    num_grid_channels, time_embed_channels = 20, 8
    backbone = HRRRUnconditionalUNet(
        img_resolution=list(patch_shape),
        img_channels=img_channels,
        num_condition_channels=num_condition_channels,
        num_grid_channels=num_grid_channels,
        time_embed_channels=time_embed_channels,
        use_apex_gn=use_apex,
        amp_mode=True,
    )
    preconditioner = EDMPreconditioner(backbone, sigma_data=1.0)

    md_model = MultiDiffusionModel2D(
        model=preconditioner,
        global_spatial_shape=(img_resolution[0], img_resolution[1]),
        positional_embedding="learnable",
        channels_positional_embedding=num_grid_channels,
        condition_patch={"cond_concat": True},
    )
    md_model.set_random_patching(
        patch_shape=patch_shape, patch_num=num_patches_per_sample
    )

    model = md_model.to(dist.device).to(memory_format=torch.channels_last)
    rank_zero_logger.info(f"Training model with {model.num_parameters()} parameters.")

    if load_checkpoint_from_file:
        load_checkpoint(checkpoint_dir, models=model)

    # Compile inner model before DDP (recommended by PyTorch docs)
    model = torch.compile(model)

    # Setup DDP for multi-GPU training
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=True,
            bucket_cap_mb=35,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    # Determine resume point for the data sampler
    current_samples_trained = 0
    if load_checkpoint_from_file:
        metadata = {}
        load_checkpoint(checkpoint_dir, metadata_dict=metadata)
        current_samples_trained = metadata.get("current_samples_trained", 0)
        rank_zero_logger.info(
            f"Resuming from samples trained: {current_samples_trained}"
        )
    sampler_start_idx = current_samples_trained // num_patches_per_sample

    # Create data loaders
    # Needs zarr 3.0
    root = zarr.open_group(
        store="s3://hrrr-surface-sda/zarr-v2",
        mode="r",
        storage_options={
            "endpoint_url": "https://pdx.s8k.io",
            "profile": "physicsnemo",
        },
    )
    time_coord = root["time"][:]
    sidx = np.where(time_coord == np.datetime64("2021-01-01T00:00:00"))[0][0]
    eidx = np.where(time_coord == np.datetime64("2024-12-31T23:00:00"))[0][0]
    time_idx = np.arange(sidx, eidx)
    dataset = HRRRSurfaceDataset(
        "s3://hrrr-surface-sda/zarr-v2",
        time_idx,
        storage_options={
            "endpoint_url": "https://pdx.s8k.io",
            "profile": "physicsnemo",
        },
    )
    train_iter = iter(
        DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            sampler=InfiniteSampler(
                dataset=dataset,
                shuffle=True,
                rank=dist.rank,
                num_replicas=dist.world_size,
                start_idx=sampler_start_idx,
            ),
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            timeout=0,
            prefetch_factor=4,
            persistent_workers=False,
        )
    )
    num_training_samples = len(dataset)

    sidx = np.where(time_coord == np.datetime64("2025-01-01T00:00:00"))[0][0]
    eidx = np.where(time_coord == np.datetime64("2025-12-31T00:00:00"))[0][0]
    time_idx = np.arange(sidx, eidx, 25)
    val_dataset = HRRRSurfaceDataset(
        "s3://hrrr-surface-sda/zarr-v2",
        time_idx,
        storage_options={
            "endpoint_url": "https://pdx.s8k.io",
            "profile": "physicsnemo",
        },
    )
    val_iter = iter(
        DataLoader(
            val_dataset,
            batch_size=batch_size_per_gpu,
            sampler=InfiniteSampler(
                dataset=val_dataset,
                shuffle=True,
                rank=dist.rank,
                num_replicas=dist.world_size,
            ),
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            timeout=0,
            prefetch_factor=4,
            persistent_workers=False,
        )
    )

    # Create loss function with multi-diffusion support
    noise_scheduler = EDMNoiseScheduler(P_mean=args.P_mean, P_std=args.P_std, sigma_data=1.0)
    loss_fn = MultiDiffusionMSEDSMLoss(
        model=model,
        noise_scheduler=noise_scheduler,
    )

    # Initialize optimizer
    if use_apex:
        FusedAdam = getattr(importlib.import_module("apex.optimizers"), "FusedAdam")
        optimizer = FusedAdam(
            model.parameters(),
            lr=5e-4,
            weight_decay=0.0,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-4,
            weight_decay=0.0,
        )

    # Initialize learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max_training_samples // num_training_samples,
        eta_min=5e-6,
    )

    # Load optimizer/scheduler state from checkpoint
    if dist.world_size > 1:
        torch.distributed.barrier()
    if load_checkpoint_from_file:
        load_checkpoint(
            checkpoint_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            device=dist.device,
        )

    # Training loop (batch-based with InfiniteSampler)
    rank_zero_logger.info("Training started...")

    # Running average for loss
    loss_running_mean = 0.0
    n_loss_running_mean = 1

    total_batch_size = batch_size_per_gpu * dist.world_size * num_patches_per_sample
    rank_zero_logger.info(f"Total batch size: {total_batch_size}")

    # Counters for periodic tasks
    samples_since_scheduler_update = 0
    samples_since_logging = 0
    samples_since_validation = 0
    samples_since_checkpoint = 0

    tick_start_time = time.time()
    while current_samples_trained < max_training_samples:

        model.train()

        # Get next batch from infinite sampler
        x, cond_spatial, cond_time = next(train_iter)
        x = x.to(dist.device, non_blocking=True).to(memory_format=torch.channels_last)
        cond_spatial = cond_spatial.to(dist.device, non_blocking=True).to(
            memory_format=torch.channels_last
        )
        cond_time = cond_time.to(dist.device, non_blocking=True).float()
        batch_size = x.shape[0]

        condition = TensorDict(
            {"cond_concat": cond_spatial, "cond_time": cond_time},
            batch_size=[batch_size],
        )

        # Forward pass
        optimizer.zero_grad(**({} if use_apex else {"set_to_none": True}))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(x, condition=condition)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        mean_loss = reduce_loss(loss.item() * batch_size, dst_rank=0)

        # Update running mean of loss (only meaningful on rank 0)
        if dist.rank == 0:
            loss_running_mean += (
                mean_loss / total_batch_size - loss_running_mean
            ) / n_loss_running_mean
            n_loss_running_mean += 1
        current_samples_trained += total_batch_size

        # Update scheduler periodically
        samples_since_scheduler_update += total_batch_size
        if samples_since_scheduler_update >= num_training_samples:
            scheduler.step()
            samples_since_scheduler_update = 0

        # Periodic logging
        samples_since_logging += total_batch_size
        if samples_since_logging >= logging_frequency:
            elapsed = time.time() - tick_start_time
            steps = samples_since_logging / total_batch_size
            rank_zero_logger.info(
                f"Samples trained: {current_samples_trained}, "
                f"loss: {loss_running_mean:.3e}, "
                f"learning rate: {optimizer.param_groups[0]['lr']:.2e}, "
                f"throughput: {samples_since_logging / elapsed / 1000:.4f} ksamples/s, "
                f"time/step: {elapsed / steps:.3f}s"
            )
            # Reset running mean after logging
            loss_running_mean = 0.0
            n_loss_running_mean = 1
            tick_start_time = time.time()
            samples_since_logging = 0

        # Validation step
        samples_since_validation += total_batch_size
        if samples_since_validation >= validation_frequency:
            model.eval()
            val_loss_running_mean = 0.0
            n_val_loss_running_mean = 1
            current_validation_samples = 0
            with torch.no_grad():
                while current_validation_samples < num_validation_samples:
                    x, cs, ct = next(val_iter)
                    x = x.to(dist.device, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                    cs = cs.to(dist.device, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                    ct = ct.to(dist.device, non_blocking=True).float()
                    batch_size = x.shape[0]

                    val_condition = TensorDict(
                        {"cond_concat": cs, "cond_time": ct},
                        batch_size=[batch_size],
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        val_loss = loss_fn(x, condition=val_condition)
                    mean_val_loss = reduce_loss(
                        val_loss.item() * batch_size, dst_rank=0
                    )
                    if dist.rank == 0:
                        val_loss_running_mean += (
                            mean_val_loss / total_batch_size - val_loss_running_mean
                        ) / n_val_loss_running_mean
                        n_val_loss_running_mean += 1
                    current_validation_samples += total_batch_size
            rank_zero_logger.info(
                f"Samples trained: {current_samples_trained}, "
                f"val_loss: {val_loss_running_mean:.3e}, "
            )
            samples_since_validation = 0

        # Periodic checkpoint
        samples_since_checkpoint += total_batch_size
        if samples_since_checkpoint >= checkpoint_frequency:
            if dist.world_size > 1:
                torch.distributed.barrier()
            if dist.rank == 0:
                save_checkpoint(
                    checkpoint_dir,
                    models=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metadata={
                        "current_samples_trained": current_samples_trained,
                    },
                )
                rank_zero_logger.info(
                    f"Saved checkpoint at samples trained: {current_samples_trained}"
                )
            samples_since_checkpoint = 0

    # Cleanup
    rank_zero_logger.info("Training completed!")


if __name__ == "__main__":
    main()
