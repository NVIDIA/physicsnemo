# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Training script for diffusion models on 2D flow fields using Modulus."""

import os
import time
import psutil
import hydra
import torch
import tqdm

from modulus.models.diffusion import EDMPrecond
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.metrics.diffusion import EDMLoss
from modulus.launch.utils import load_checkpoint, save_checkpoint

from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from datasets.dataset import get_dataset_and_dataloader_from_config
from helpers.train_helpers import (
    set_seed,
    configure_cuda_for_consistent_precision,
    handle_and_clip_gradients,
    is_time_for_periodic_task_epoch,
)


# Train the CorrDiff model using the configurations in "conf/config_training.yaml"
@hydra.main(version_base="1.2", config_path="conf", config_name="config_training_uflow")
def main(cfg: DictConfig) -> None:
    """Train a diffusion model for 2D flow field generation.

    This function initializes the distributed training environment, sets up
    the model, optimizer, loss function, and data loaders, and trains the
    diffusion model using the EDM (Elucidating Diffusion Models) framework.
    Supports distributed data parallel training across multiple GPUs.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing model architecture, training
        hyperparameters, dataset paths, and logging settings.
    """
    # Initialize distributed environment for training
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers
    if dist.rank == 0:
        writer = SummaryWriter(log_dir="tensorboard")
    logger = PythonLogger("main")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger

    # Resolve and parse configs
    OmegaConf.resolve(cfg)
    dataset_cfg = OmegaConf.to_container(
        cfg.dataset, resolve=True
    )  # TODO needs better handling
    del dataset_cfg[
        "dataset_features"
    ]  # Because we cannot pass dataset features into dataloader, can implement better later

    if hasattr(cfg, "validation"):
        OmegaConf.to_container(cfg.validation)
    else:
        pass

    fp_optimizations = cfg.training.perf.fp_optimizations
    fp16 = fp_optimizations == "fp16"  # flag to use use fp16
    enable_amp = fp_optimizations.startswith("amp")  # Flag for mixed precesion
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16

    logger.info(f"Saving the outputs in {os.getcwd()}")

    checkpoint_dir = os.path.join(
        cfg.training.io.get("checkpoint_dir", "."), f"checkpoints_{cfg.model.name}"
    )

    # Set seeds and configure CUDA and cuDNN settings to ensure consistent precision
    set_seed(dist.rank)
    configure_cuda_for_consistent_precision()

    # Instantiate the dataset
    data_loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.training.perf.dataloader_workers,
        "prefetch_factor": 2,
    }

    dataset, DataLoader = get_dataset_and_dataloader_from_config(
        dataset_cfg,
        data_loader_kwargs,
        batch_size=cfg.training.hp.batch_size_per_gpu,
        seed=dist.rank,
        shuffle=True,
        dist=dist,
        Train=True,
    )

    dataset_channels = dataset.num_channels()
    img_shape = dataset.image_shape()

    model = EDMPrecond(
        img_resolution=list(img_shape),
        img_channels=dataset_channels,
        model_channels=cfg.model.model_args.model_channels,
        channel_mult=cfg.model.model_args.channel_mult,
        attn_resolutions=cfg.model.model_args.attn_resolutions,
        use_fp16=fp16,
        num_blocks=cfg.model.model_args.num_blocks,
        dropout=cfg.model.model_args.dropout,
        model_type="SongUNet",  # TODO: check if dhariwalUnet can be used
        channel_mult_emb=cfg.model.model_args.channel_mult_emb,
    )

    model.train().requires_grad_(True).to(dist.device)

    # Enable distributed data parallel if applicable
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=dist.find_unused_parameters,
        )

    loss_fn = EDMLoss()

    # Instantiate the optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=cfg.training.hp.lr, betas=[0.9, 0.999], eps=1e-8
    )

    # Record the current time to measure the duration of subsequent operations.
    start_time = time.time()
    ## Resume training from previous checkpoints if exists
    if dist.world_size > 1:
        torch.distributed.barrier()
    try:
        epoch = load_checkpoint(
            path=checkpoint_dir,
            models=model,
            optimizer=optimizer,
            device=dist.device,
        )
    except Exception:
        epoch = 0

    if dist.rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params / 1000000}M")

    ############################################################################
    #                            MAIN TRAINING LOOP                            #
    ############################################################################

    batch_size_per_gpu = cfg.training.hp.batch_size_per_gpu
    logger0.info(
        f"Training for {cfg.training.hp.epochs} epochs...Starting from epoch {epoch}"
    )
    done = False

    for ep in range(epoch, cfg.training.hp.epochs + 1):
        tick_start_time = time.time()
        loss_accum = 0
        num_batches = len(DataLoader)
        pbar = tqdm.tqdm(enumerate(DataLoader), total=len(DataLoader), disable=True)

        lr_decay = cfg.training.hp.lr_decay
        decay_from = cfg.training.hp.lr_decay_from

        for g in optimizer.param_groups:
            # Apply learning rate decay after ramp-up
            if ep >= decay_from:
                g["lr"] = cfg.training.hp.lr * (
                    lr_decay ** ((ep - decay_from) // decay_from)
                )

            current_lr = g["lr"]

        for batch_index, batch in pbar:
            # Compute & accumulate gradients
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(dist.device).to(torch.float32).contiguous()

            with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                loss = loss_fn(
                    net=model,
                    images=batch,
                    augment_pipe=None,
                    labels=None,
                )
            loss = loss.sum() / batch_size_per_gpu
            loss.backward()

            # Clip gradients
            handle_and_clip_gradients(
                model, grad_clip_threshold=cfg.training.hp.grad_clip_threshold
            )

            optimizer.step()
            loss_accum += loss / num_batches
            # Update the progress bar description with your custom message
            # pbar.set_description(f"Rank:{dist.rank}, LocalRank:{dist.local_rank}, Epoch: {ep}, Batch_index: {batch_index} , Loss: {loss}")
            # Done.

        loss_sum = torch.tensor([loss_accum], device=dist.device)

        if dist.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
        average_loss = (loss_sum / dist.world_size).cpu().item()

        if dist.rank == 0:
            writer.add_scalar("training_loss", average_loss, ep)

        is_time_for_periodic_task_epoch(
            ep,
            cfg.training.io.print_progress_freq,
            done,
            dist.rank,
            rank_0_only=True,
        )

        done = ep >= cfg.training.hp.epochs

        if is_time_for_periodic_task_epoch(
            ep,
            cfg.training.io.print_progress_freq,
            done,
            dist.rank,
            rank_0_only=True,
        ):
            batch_size = cfg.training.hp.batch_size_per_gpu
            tick_end_time = time.time()
            fields = []
            fields += [f"epoch {ep:<6}"]  # Replace cur_nimg with epoch-based tracking
            fields += [f"avg_training_loss {average_loss:<7.2f}"]
            fields += [f"batch_size:{dist.world_size:<3.1f}x{batch_size:<3.1f}"]
            fields += [f"learning_rate {current_lr:<7.8f}"]
            fields += [f"total_sec {(tick_end_time - start_time):<7.1f}"]
            fields += [f"sec_per_epoch {(tick_end_time - tick_start_time):<7.1f}"]
            fields += [
                f"cpu_mem_gb {(psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
            ]
            fields += [
                f"peak_gpu_mem_gb {(torch.cuda.max_memory_allocated(dist.device) / 2**30):<6.2f}"
            ]
            fields += [
                f"peak_gpu_mem_reserved_gb {(torch.cuda.max_memory_reserved(dist.device) / 2**30):<6.2f}"
            ]
            logger0.info(" ".join(fields))
            torch.cuda.reset_peak_memory_stats()

        original_args = model.module._args
        original_args = OmegaConf.create(original_args)
        model.module._args = OmegaConf.to_container(original_args, resolve=True)

        # Save checkpoints
        if dist.world_size > 1:
            torch.distributed.barrier()
        if is_time_for_periodic_task_epoch(
            ep,
            cfg.training.io.save_checkpoint_freq,
            done,
            dist.rank,
            rank_0_only=True,
        ):
            save_checkpoint(
                path=checkpoint_dir,
                models=model,
                optimizer=optimizer,
                epoch=ep,
            )

    logger0.info("Training Completed.")


if __name__ == "__main__":
    main()
