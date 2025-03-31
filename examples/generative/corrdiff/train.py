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

import os, time, psutil, hydra, torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from physicsnemo import Module
from physicsnemo.models.diffusion import UNet, EDMPrecondSR
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.metrics.diffusion import RegressionLoss, ResLoss, RegressionLossCE
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from datasets.dataset import init_train_valid_datasets_from_config
from helpers.train_helpers import (
    set_patch_shape,
    set_seed,
    configure_cuda_for_consistent_precision,
    compute_num_accumulation_rounds,
    handle_and_clip_gradients,
    is_time_for_periodic_task,
)


# Train the CorrDiff model using the configurations in "conf/config_training.yaml"
@hydra.main(version_base="1.2", config_path="conf", config_name="config_training")
def main(cfg: DictConfig) -> None:

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
    dataset_cfg = OmegaConf.to_container(cfg.dataset)  # TODO needs better handling
    if hasattr(cfg, "validation"):
        train_test_split = True
        validation_dataset_cfg = OmegaConf.to_container(cfg.validation)
    else:
        train_test_split = False
        validation_dataset_cfg = None
    fp_optimizations = cfg.training.perf.fp_optimizations
    songunet_checkpoint_level = cfg.training.perf.songunet_checkpoint_level
    fp16 = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16
    logger.info(f"Saving the outputs in {os.getcwd()}")
    checkpoint_dir = os.path.join(
        cfg.training.io.get("checkpoint_dir", "."), f"checkpoints_{cfg.model.name}"
    )
    if cfg.training.hp.batch_size_per_gpu == "auto":
        cfg.training.hp.batch_size_per_gpu = (
            cfg.training.hp.total_batch_size // dist.world_size
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
    (
        dataset,
        dataset_iterator,
        validation_dataset,
        validation_dataset_iterator,
    ) = init_train_valid_datasets_from_config(
        dataset_cfg,
        data_loader_kwargs,
        batch_size=cfg.training.hp.batch_size_per_gpu,
        seed=0,
        validation_dataset_cfg=validation_dataset_cfg,
        train_test_split=train_test_split,
    )

    # Parse image configuration & update model args
    dataset_channels = len(dataset.input_channels())
    img_in_channels = dataset_channels
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())
    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels

    if cfg.model.name == "lt_aware_ce_regression":
        prob_channels = dataset.get_prob_channel_index()
    else:
        prob_channels = None

    # Parse the patch shape
    if (
        cfg.model.name == "patched_diffusion"
        or cfg.model.name == "lt_aware_patched_diffusion"
    ):
        patch_shape_x = cfg.training.hp.patch_shape_x
        patch_shape_y = cfg.training.hp.patch_shape_y
    else:
        patch_shape_x = None
        patch_shape_y = None
    patch_shape = (patch_shape_y, patch_shape_x)
    img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if patch_shape != img_shape:
        logger0.info("Patch-based training enabled")
    else:
        logger0.info("Patch-based training disabled")
    # interpolate global channel if patch-based model is used
    if img_shape[1] != patch_shape[1]:
        img_in_channels += dataset_channels

    # Instantiate the model and move to device.
    if cfg.model.name not in (
        "regression",
        "lt_aware_ce_regression",
        "diffusion",
        "patched_diffusion",
        "lt_aware_patched_diffusion",
    ):
        raise ValueError("Invalid model")
    model_args = {  # default parameters for all networks
        "img_out_channels": img_out_channels,
        "img_resolution": list(img_shape),
        "use_fp16": fp16,
    }
    standard_model_cfgs = {  # default parameters for different network types
        "regression": {
            "img_channels": 4,
            "N_grid_channels": 4,
            "embedding_type": "zero",
            "checkpoint_level": songunet_checkpoint_level,
        },
        "lt_aware_ce_regression": {
            "img_channels": 4,
            "N_grid_channels": 4,
            "embedding_type": "zero",
            "lead_time_channels": 4,
            "lead_time_steps": 9,
            "prob_channels": prob_channels,
            "checkpoint_level": songunet_checkpoint_level,
            "model_type": "SongUNetPosLtEmbd",
        },
        "diffusion": {
            "img_channels": img_out_channels,
            "gridtype": "sinusoidal",
            "N_grid_channels": 4,
            "checkpoint_level": songunet_checkpoint_level,
        },
        "patched_diffusion": {
            "img_channels": img_out_channels,
            "gridtype": "learnable",
            "N_grid_channels": 100,
            "checkpoint_level": songunet_checkpoint_level,
        },
        "lt_aware_patched_diffusion": {
            "img_channels": img_out_channels,
            "gridtype": "learnable",
            "N_grid_channels": 100,
            "lead_time_channels": 20,
            "lead_time_steps": 9,
            "checkpoint_level": songunet_checkpoint_level,
            "model_type": "SongUNetPosLtEmbd",
        },
    }
    model_args.update(standard_model_cfgs[cfg.model.name])
    if cfg.model.name in (
        "diffusion",
        "patched_diffusion",
        "lt_aware_patched_diffusion",
    ):
        model_args["scale_cond_input"] = cfg.model.scale_cond_input
    if hasattr(cfg.model, "model_args"):  # override defaults from config file
        model_args.update(OmegaConf.to_container(cfg.model.model_args))
    if cfg.model.name == "regression":
        model = UNet(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
    elif cfg.model.name == "lt_aware_ce_regression":
        model = UNet(
            img_in_channels=img_in_channels
            + model_args["N_grid_channels"]
            + model_args["lead_time_channels"],
            **model_args,
        )
    elif cfg.model.name == "lt_aware_patched_diffusion":
        model = EDMPrecondSR(
            img_in_channels=img_in_channels
            + model_args["N_grid_channels"]
            + model_args["lead_time_channels"],
            **model_args,
        )
    else:  # diffusion or patched diffusion
        model = EDMPrecondSR(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
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

    # Load the regression checkpoint if applicable
    if hasattr(cfg.training.io, "regression_checkpoint_path"):
        regression_checkpoint_path = to_absolute_path(
            cfg.training.io.regression_checkpoint_path
        )
        if not os.path.exists(regression_checkpoint_path):
            raise FileNotFoundError(
                f"Expected this regression checkpoint but not found: {regression_checkpoint_path}"
            )
        regression_net = Module.from_checkpoint(regression_checkpoint_path)
        regression_net.eval().requires_grad_(False).to(dist.device)
        logger0.success("Loaded the pre-trained regression model")

    # Instantiate the loss function
    patch_num = getattr(cfg.training.hp, "patch_num", 1)
    if cfg.model.name in (
        "diffusion",
        "patched_diffusion",
        "lt_aware_patched_diffusion",
    ):
        loss_fn = ResLoss(
            regression_net=regression_net,
            img_shape_x=img_shape[1],
            img_shape_y=img_shape[0],
            patch_shape_x=patch_shape[1],
            patch_shape_y=patch_shape[0],
            patch_num=patch_num,
            hr_mean_conditioning=cfg.model.hr_mean_conditioning,
        )
    elif cfg.model.name == "regression":
        loss_fn = RegressionLoss()
    elif cfg.model.name == "lt_aware_ce_regression":
        loss_fn = RegressionLossCE(prob_channels=prob_channels)

    # Instantiate the optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=cfg.training.hp.lr, betas=[0.9, 0.999], eps=1e-8
    )

    # Record the current time to measure the duration of subsequent operations.
    start_time = time.time()

    # Compute the number of required gradient accumulation rounds
    # It is automatically used if batch_size_per_gpu * dist.world_size < total_batch_size
    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.training.hp.total_batch_size,
        cfg.training.hp.batch_size_per_gpu,
        dist.world_size,
    )
    batch_size_per_gpu = cfg.training.hp.batch_size_per_gpu
    logger0.info(f"Using {num_accumulation_rounds} gradient accumulation rounds")

    ## Resume training from previous checkpoints if exists
    if dist.world_size > 1:
        torch.distributed.barrier()
    try:
        cur_nimg = load_checkpoint(
            path=checkpoint_dir,
            models=model,
            optimizer=optimizer,
            device=dist.device,
        )
    except:
        cur_nimg = 0

    ############################################################################
    #                            MAIN TRAINING LOOP                            #
    ############################################################################

    logger0.info(f"Training for {cfg.training.hp.training_duration} images...")
    done = False

    # init variables to monitor running mean of average loss since last periodic
    average_loss_running_mean = 0
    n_average_loss_running_mean = 1

    while not done:
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        # Compute & accumulate gradients
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0
        for _ in range(num_accumulation_rounds):
            img_clean, img_lr, labels, *lead_time_label = next(dataset_iterator)
            img_clean = img_clean.to(dist.device).to(torch.float32).contiguous()
            img_lr = img_lr.to(dist.device).to(torch.float32).contiguous()
            labels = labels.to(dist.device).contiguous()
            loss_fn_kwargs = {
                "net": model,
                "img_clean": img_clean,
                "img_lr": img_lr,
                "labels": labels,
                "augment_pipe": None,
            }
            if lead_time_label:
                lead_time_label = lead_time_label[0].to(dist.device).contiguous()
                loss_fn_kwargs.update({"lead_time_label": lead_time_label})
            else:
                lead_time_label = None
            with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                loss = loss_fn(**loss_fn_kwargs)
            loss = loss.sum() / batch_size_per_gpu
            loss_accum += loss / num_accumulation_rounds
            loss.backward()

        loss_sum = torch.tensor([loss_accum], device=dist.device)
        if dist.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
        average_loss = (loss_sum / dist.world_size).cpu().item()

        # update running mean of average loss since last periodic task
        average_loss_running_mean += (
            average_loss - average_loss_running_mean
        ) / n_average_loss_running_mean
        n_average_loss_running_mean += 1

        if dist.rank == 0:
            writer.add_scalar("training_loss", average_loss, cur_nimg)
            writer.add_scalar(
                "training_loss_running_mean", average_loss_running_mean, cur_nimg
            )

        ptt = is_time_for_periodic_task(
            cur_nimg,
            cfg.training.io.print_progress_freq,
            done,
            cfg.training.hp.total_batch_size,
            dist.rank,
            rank_0_only=True,
        )
        if ptt:
            # reset running mean of average loss
            average_loss_running_mean = 0
            n_average_loss_running_mean = 1

        # Update weights.
        lr_rampup = cfg.training.hp.lr_rampup  # ramp up the learning rate
        for g in optimizer.param_groups:
            if lr_rampup > 0:
                g["lr"] = cfg.training.hp.lr * min(cur_nimg / lr_rampup, 1)
            if cur_nimg >= lr_rampup:
                g["lr"] *= cfg.training.hp.lr_decay ** ((cur_nimg - lr_rampup) // 5e6)
            current_lr = g["lr"]
            if dist.rank == 0:
                writer.add_scalar("learning_rate", current_lr, cur_nimg)
        handle_and_clip_gradients(
            model, grad_clip_threshold=cfg.training.hp.grad_clip_threshold
        )
        optimizer.step()

        cur_nimg += cfg.training.hp.total_batch_size
        done = cur_nimg >= cfg.training.hp.training_duration

        # Validation
        if validation_dataset_iterator is not None:
            valid_loss_accum = 0
            if is_time_for_periodic_task(
                cur_nimg,
                cfg.training.io.validation_freq,
                done,
                cfg.training.hp.total_batch_size,
                dist.rank,
            ):
                with torch.no_grad():
                    for _ in range(cfg.training.io.validation_steps):
                        img_clean_valid, img_lr_valid, labels_valid = next(
                            validation_dataset_iterator
                        )

                        img_clean_valid = (
                            img_clean_valid.to(dist.device)
                            .to(torch.float32)
                            .contiguous()
                        )
                        img_lr_valid = (
                            img_lr_valid.to(dist.device).to(torch.float32).contiguous()
                        )
                        labels_valid = labels_valid.to(dist.device).contiguous()
                        loss_valid = loss_fn(
                            net=model,
                            img_clean=img_clean_valid,
                            img_lr=img_lr_valid,
                            labels=labels_valid,
                            augment_pipe=None,
                        )
                        loss_valid = (
                            (loss_valid.sum() / batch_size_per_gpu).cpu().item()
                        )
                        valid_loss_accum += (
                            loss_valid / cfg.training.io.validation_steps
                        )
                    valid_loss_sum = torch.tensor(
                        [valid_loss_accum], device=dist.device
                    )
                    if dist.world_size > 1:
                        torch.distributed.barrier()
                        torch.distributed.all_reduce(
                            valid_loss_sum, op=torch.distributed.ReduceOp.SUM
                        )
                    average_valid_loss = valid_loss_sum / dist.world_size
                    if dist.rank == 0:
                        writer.add_scalar(
                            "validation_loss", average_valid_loss, cur_nimg
                        )

        if is_time_for_periodic_task(
            cur_nimg,
            cfg.training.io.print_progress_freq,
            done,
            cfg.training.hp.total_batch_size,
            dist.rank,
            rank_0_only=True,
        ):
            # Print stats if we crossed the printing threshold with this batch
            tick_end_time = time.time()
            fields = []
            fields += [f"samples {cur_nimg:<9.1f}"]
            fields += [f"training_loss {average_loss:<7.2f}"]
            fields += [f"training_loss_running_mean {average_loss_running_mean:<7.2f}"]
            fields += [f"learning_rate {current_lr:<7.8f}"]
            fields += [f"total_sec {(tick_end_time - start_time):<7.1f}"]
            fields += [f"sec_per_tick {(tick_end_time - tick_start_time):<7.1f}"]
            fields += [
                f"sec_per_sample {((tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg)):<7.2f}"
            ]
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

        # Save checkpoints
        if dist.world_size > 1:
            torch.distributed.barrier()
        if is_time_for_periodic_task(
            cur_nimg,
            cfg.training.io.save_checkpoint_freq,
            done,
            cfg.training.hp.total_batch_size,
            dist.rank,
            rank_0_only=True,
        ):
            save_checkpoint(
                path=checkpoint_dir,
                models=model,
                optimizer=optimizer,
                epoch=cur_nimg,
            )

    # Done.
    logger0.info("Training Completed.")


if __name__ == "__main__":
    main()
