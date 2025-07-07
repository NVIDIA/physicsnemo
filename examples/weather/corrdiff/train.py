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
import time
import psutil
from contextlib import nullcontext

import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import nvtx
import wandb

from physicsnemo import Module
from physicsnemo.models.diffusion import UNet, EDMPrecondSuperResolution
from physicsnemo.distributed import DistributedManager
from physicsnemo.metrics.diffusion import RegressionLoss, ResidualLoss, RegressionLossCE
from physicsnemo.utils.patching import RandomPatching2D
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.utils import (
    load_checkpoint,
    save_checkpoint,
    get_checkpoint_dir,
)

from datasets.dataset import init_train_valid_datasets_from_config, register_dataset
from helpers.train_helpers import (
    set_patch_shape,
    set_seed,
    configure_cuda_for_consistent_precision,
    compute_num_accumulation_rounds,
    handle_and_clip_gradients,
    is_time_for_periodic_task,
)

torch._dynamo.reset()
# Increase the cache size limit
torch._dynamo.config.cache_size_limit = 264  # Set to a higher value
torch._dynamo.config.verbose = True  # Enable verbose logging
torch._dynamo.config.suppress_errors = False  # Forces the error to show all details
torch._logging.set_logs(recompiles=True, graph_breaks=True)


def checkpoint_list(path, suffix=".mdlus"):
    """Helper function to return sorted list, in ascending order, of checkpoints in a path"""
    checkpoints = []
    for file in os.listdir(path):
        if file.endswith(suffix):
            # Split the filename and extract the index
            try:
                index = int(file.split(".")[-2])
                checkpoints.append((index, file))
            except ValueError:
                continue

    # Sort by index and return filenames
    checkpoints.sort(key=lambda x: x[0])
    return [file for _, file in checkpoints]


# Define safe CUDA profiler tools that fallback to no-ops when CUDA is not available
def cuda_profiler():
    if torch.cuda.is_available():
        return torch.cuda.profiler.profile()
    else:
        return nullcontext()


def cuda_profiler_start():
    if torch.cuda.is_available():
        torch.cuda.profiler.start()


def cuda_profiler_stop():
    if torch.cuda.is_available():
        torch.cuda.profiler.stop()


def profiler_emit_nvtx():
    if torch.cuda.is_available():
        return torch.autograd.profiler.emit_nvtx()
    else:
        return nullcontext()


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
    initialize_wandb(
        project="Modulus-Launch",
        entity="Modulus",
        name=f"CorrDiff-Training-{HydraConfig.get().job.name}",
        group="CorrDiff-DDP-Group",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg),
        results_dir=cfg.wandb.results_dir,
    )

    # Resolve and parse configs
    OmegaConf.resolve(cfg)
    dataset_cfg = OmegaConf.to_container(cfg.dataset)  # TODO needs better handling

    # Register custom dataset if specified in config
    register_dataset(cfg.dataset.type)
    logger0.info(f"Using dataset: {cfg.dataset.type}")

    if hasattr(cfg, "validation"):
        validation = True
        validation_dataset_cfg = OmegaConf.to_container(cfg.validation)
    else:
        validation = False
        validation_dataset_cfg = None
    fp_optimizations = cfg.training.perf.fp_optimizations
    songunet_checkpoint_level = cfg.training.perf.songunet_checkpoint_level
    fp16 = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16
    logger.info(f"Saving the outputs in {os.getcwd()}")
    checkpoint_dir = get_checkpoint_dir(
        str(cfg.training.io.get("checkpoint_dir", ".")), cfg.model.name
    )
    if cfg.training.hp.batch_size_per_gpu == "auto":
        cfg.training.hp.batch_size_per_gpu = (
            cfg.training.hp.total_batch_size // dist.world_size
        )

    # Load the current number of images for resuming
    try:
        cur_nimg = load_checkpoint(
            path=checkpoint_dir,
        )
    except Exception:
        cur_nimg = 0

    # Set seeds and configure CUDA and cuDNN settings to ensure consistent precision
    set_seed(dist.rank + cur_nimg)
    configure_cuda_for_consistent_precision()

    # Instantiate the dataset
    data_loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.training.perf.dataloader_workers,
        "prefetch_factor": 2 if cfg.training.perf.dataloader_workers > 0 else None,
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
        validation=validation,
        sampler_start_idx=cur_nimg,
    )

    # Parse image configuration & update model args
    dataset_channels = len(dataset.input_channels())
    img_in_channels = dataset_channels
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())
    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels

    # Handle patch shape
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
    if (
        patch_shape_x
        and patch_shape_y
        and patch_shape_y >= img_shape[0]
        and patch_shape_x >= img_shape[1]
    ):
        logger0.warning(
            f"Patch shape {patch_shape_y}x{patch_shape_x} is larger than \
            the image shape {img_shape[0]}x{img_shape[1]}. Patching will not be used."
        )
    patch_shape = (patch_shape_y, patch_shape_x)
    use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if use_patching:
        # Utility to perform patches extraction and batching
        patching = RandomPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            patch_num=getattr(cfg.training.hp, "patch_num", 1),
        )
        logger0.info("Patch-based training enabled")
    else:
        patching = None
        logger0.info("Patch-based training disabled")
    # interpolate global channel if patch-based model is used
    if use_patching:
        img_in_channels += dataset_channels

    # Instantiate the model and move to device.
    model_args = {  # default parameters for all networks
        "img_out_channels": img_out_channels,
        "img_resolution": list(img_shape),
        "use_fp16": fp16,
        "checkpoint_level": songunet_checkpoint_level,
    }
    if cfg.model.name == "lt_aware_ce_regression":
        model_args["prob_channels"] = prob_channels
    if hasattr(cfg.model, "model_args"):  # override defaults from config file
        model_args.update(OmegaConf.to_container(cfg.model.model_args))

    use_torch_compile = False
    use_apex_gn = False
    profile_mode = False

    if hasattr(cfg.training.perf, "torch_compile"):
        use_torch_compile = cfg.training.perf.torch_compile
    if hasattr(cfg.training.perf, "use_apex_gn"):
        use_apex_gn = cfg.training.perf.use_apex_gn
        model_args["use_apex_gn"] = use_apex_gn

    if hasattr(cfg.training.perf, "profile_mode"):
        profile_mode = cfg.training.perf.profile_mode
        model_args["profile_mode"] = profile_mode

    if enable_amp:
        model_args["amp_mode"] = enable_amp

    if cfg.model.name == "regression":
        model = UNet(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
    elif (
        cfg.model.name == "lt_aware_ce_regression"
        or cfg.model.name == "lt_aware_regression"
    ):
        model = UNet(
            img_in_channels=img_in_channels
            + model_args["N_grid_channels"]
            + model_args["lead_time_channels"],
            **model_args,
        )
    elif cfg.model.name == "lt_aware_patched_diffusion":
        model = EDMPrecondSuperResolution(
            img_in_channels=img_in_channels
            + model_args["N_grid_channels"]
            + model_args["lead_time_channels"],
            **model_args,
        )
    elif cfg.model.name == "diffusion":
        model = EDMPrecondSuperResolution(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
    elif cfg.model.name == "patched_diffusion":
        model = EDMPrecondSuperResolution(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
    else:
        raise ValueError(f"Invalid model: {cfg.model.name}")

    model.train().requires_grad_(True).to(dist.device)

    if use_apex_gn:
        model.to(memory_format=torch.channels_last)

    # Check if regression model is used with patching
    if (
        cfg.model.name
        in ["regression", "lt_aware_regression", "lt_aware_ce_regression"]
        and patching is not None
    ):
        raise ValueError(
            f"Regression model ({cfg.model.name}) cannot be used with patch-based training. "
        )

    # Enable distributed data parallel if applicable
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=True,  # dist.find_unused_parameters,
            bucket_cap_mb=35,
            gradient_as_bucket_view=True,
        )
    if cfg.wandb.watch_model and dist.rank == 0:
        wandb.watch(model)

    # Load the model checkpoint if applicable
    try:
        load_checkpoint(path=checkpoint_dir, models=model)
    except Exception:
        pass

    # Load the regression checkpoint if applicable
    if (
        hasattr(cfg.training.io, "regression_checkpoint_path")
        and cfg.training.io.regression_checkpoint_path is not None
    ):
        regression_checkpoint_path = to_absolute_path(
            cfg.training.io.regression_checkpoint_path
        )
        if not os.path.exists(regression_checkpoint_path):
            raise FileNotFoundError(
                f"Expected this regression checkpoint but not found: {regression_checkpoint_path}"
            )
        regression_net = Module.from_checkpoint(
            regression_checkpoint_path, override_args={"use_apex_gn": use_apex_gn}
        )
        regression_net.amp_mode = enable_amp
        regression_net.profile_mode = profile_mode
        regression_net.eval().requires_grad_(False).to(dist.device)
        if use_apex_gn:
            regression_net.to(memory_format=torch.channels_last)
        logger0.success("Loaded the pre-trained regression model")
    else:
        regression_net = None

    # Compile the model and regression net if applicable
    if use_torch_compile:
        model = torch.compile(model)
        if regression_net:
            regression_net = torch.compile(regression_net)

    # Compute the number of required gradient accumulation rounds
    # It is automatically used if batch_size_per_gpu * dist.world_size < total_batch_size
    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.training.hp.total_batch_size,
        cfg.training.hp.batch_size_per_gpu,
        dist.world_size,
    )
    batch_size_per_gpu = cfg.training.hp.batch_size_per_gpu
    logger0.info(f"Using {num_accumulation_rounds} gradient accumulation rounds")

    # calculate patch per iter
    patch_num = getattr(cfg.training.hp, "patch_num", 1)
    if hasattr(cfg.training.hp, "max_patch_per_gpu"):
        max_patch_per_gpu = cfg.training.hp.max_patch_per_gpu
        if max_patch_per_gpu // batch_size_per_gpu < 1:
            raise ValueError(
                f"max_patch_per_gpu ({max_patch_per_gpu}) must be greater or equal to batch_size_per_gpu ({batch_size_per_gpu})."
            )
        max_patch_num_per_iter = min(
            patch_num, (max_patch_per_gpu // batch_size_per_gpu)
        )
        patch_iterations = (
            patch_num + max_patch_num_per_iter - 1
        ) // max_patch_num_per_iter
        patch_nums_iter = [
            min(max_patch_num_per_iter, patch_num - i * max_patch_num_per_iter)
            for i in range(patch_iterations)
        ]
        logger0.info(
            f"max_patch_num_per_iter is {max_patch_num_per_iter}, patch_iterations is {patch_iterations}, patch_nums_iter is {patch_nums_iter}"
        )
    else:
        patch_nums_iter = [patch_num]

    # Set patch gradient accumulation only for patched diffusion models
    if cfg.model.name in {
        "patched_diffusion",
        "lt_aware_patched_diffusion",
    }:
        if len(patch_nums_iter) > 1:
            if not patching:
                logger0.info(
                    "Patching is not enabled: patch gradient accumulation automatically disabled."
                )
                use_patch_grad_acc = False
            else:
                use_patch_grad_acc = True
        else:
            use_patch_grad_acc = False
    # Automatically disable patch gradient accumulation for non-patched models
    else:
        logger0.info(
            "Training a non-patched model: patch gradient accumulation automatically disabled."
        )
        use_patch_grad_acc = None

    # Instantiate the loss function
    if cfg.model.name in (
        "diffusion",
        "patched_diffusion",
        "lt_aware_patched_diffusion",
    ):
        loss_fn = ResidualLoss(
            regression_net=regression_net,
            hr_mean_conditioning=cfg.model.hr_mean_conditioning,
        )
    elif cfg.model.name == "regression" or cfg.model.name == "lt_aware_regression":
        loss_fn = RegressionLoss()
    elif cfg.model.name == "lt_aware_ce_regression":
        loss_fn = RegressionLossCE(prob_channels=prob_channels)

    # Instantiate the optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.training.hp.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
        fused=True,
    )

    # Record the current time to measure the duration of subsequent operations.
    start_time = time.time()

    ## Load optimizer checkpoint if exists
    if dist.world_size > 1:
        torch.distributed.barrier()
    try:
        load_checkpoint(
            path=checkpoint_dir,
            optimizer=optimizer,
            device=dist.device,
        )
    except Exception:
        pass

    ############################################################################
    #                            MAIN TRAINING LOOP                            #
    ############################################################################

    logger0.info(f"Training for {cfg.training.hp.training_duration} images...")
    done = False

    # init variables to monitor running mean of average loss since last periodic
    average_loss_running_mean = 0
    n_average_loss_running_mean = 1
    start_nimg = cur_nimg
    input_dtype = torch.float32
    if enable_amp:
        input_dtype = torch.float32
    elif fp16:
        input_dtype = torch.float16

    # enable profiler:
    with cuda_profiler():
        with profiler_emit_nvtx():

            while not done:
                tick_start_nimg = cur_nimg
                tick_start_time = time.time()

                if cur_nimg - start_nimg == 24 * cfg.training.hp.total_batch_size:
                    logger0.info(f"Starting Profiler at {cur_nimg}")
                    cuda_profiler_start()

                if cur_nimg - start_nimg == 25 * cfg.training.hp.total_batch_size:
                    logger0.info(f"Stopping Profiler at {cur_nimg}")
                    cuda_profiler_stop()

                with nvtx.annotate("Training iteration", color="green"):
                    # Compute & accumulate gradients
                    optimizer.zero_grad(set_to_none=True)
                    loss_accum = 0
                    for n_i in range(num_accumulation_rounds):
                        with nvtx.annotate(
                            f"accumulation round {n_i}", color="Magenta"
                        ):
                            with nvtx.annotate("loading data", color="green"):
                                img_clean, img_lr, *lead_time_label = next(
                                    dataset_iterator
                                )
                                if use_apex_gn:
                                    img_clean = img_clean.to(
                                        dist.device,
                                        dtype=input_dtype,
                                        non_blocking=True,
                                    ).to(memory_format=torch.channels_last)
                                    img_lr = img_lr.to(
                                        dist.device,
                                        dtype=input_dtype,
                                        non_blocking=True,
                                    ).to(memory_format=torch.channels_last)
                                else:
                                    img_clean = (
                                        img_clean.to(dist.device)
                                        .to(input_dtype)
                                        .contiguous()
                                    )
                                    img_lr = (
                                        img_lr.to(dist.device)
                                        .to(input_dtype)
                                        .contiguous()
                                    )
                            loss_fn_kwargs = {
                                "net": model,
                                "img_clean": img_clean,
                                "img_lr": img_lr,
                                "augment_pipe": None,
                            }
                            if use_patch_grad_acc is not None:
                                loss_fn_kwargs[
                                    "use_patch_grad_acc"
                                ] = use_patch_grad_acc

                            if lead_time_label:
                                lead_time_label = (
                                    lead_time_label[0].to(dist.device).contiguous()
                                )
                                loss_fn_kwargs.update(
                                    {"lead_time_label": lead_time_label}
                                )
                            else:
                                lead_time_label = None
                            if use_patch_grad_acc:
                                loss_fn.y_mean = None

                            for patch_num_per_iter in patch_nums_iter:
                                if patching is not None:
                                    patching.set_patch_num(patch_num_per_iter)
                                    loss_fn_kwargs.update({"patching": patching})
                                with nvtx.annotate(f"loss forward", color="green"):
                                    with torch.autocast(
                                        "cuda", dtype=amp_dtype, enabled=enable_amp
                                    ):
                                        loss = loss_fn(**loss_fn_kwargs)

                                loss = loss.sum() / batch_size_per_gpu
                                loss_accum += (
                                    loss
                                    / num_accumulation_rounds
                                    / len(patch_nums_iter)
                                )
                                with nvtx.annotate(f"loss backward", color="yellow"):
                                    loss.backward()

                    with nvtx.annotate(f"loss aggregate", color="green"):
                        loss_sum = torch.tensor([loss_accum], device=dist.device)
                        if dist.world_size > 1:
                            torch.distributed.barrier()
                            torch.distributed.all_reduce(
                                loss_sum, op=torch.distributed.ReduceOp.SUM
                            )
                        average_loss = (loss_sum / dist.world_size).cpu().item()

                    # update running mean of average loss since last periodic task
                    average_loss_running_mean += (
                        average_loss - average_loss_running_mean
                    ) / n_average_loss_running_mean
                    n_average_loss_running_mean += 1

                    if dist.rank == 0:
                        writer.add_scalar("training_loss", average_loss, cur_nimg)
                        writer.add_scalar(
                            "training_loss_running_mean",
                            average_loss_running_mean,
                            cur_nimg,
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
                    with nvtx.annotate("update weights", color="blue"):
                        lr_rampup = (
                            cfg.training.hp.lr_rampup
                        )  # ramp up the learning rate
                        for g in optimizer.param_groups:
                            if lr_rampup > 0:
                                g["lr"] = cfg.training.hp.lr * min(
                                    cur_nimg / lr_rampup, 1
                                )
                            if cur_nimg >= lr_rampup:
                                g["lr"] *= cfg.training.hp.lr_decay ** (
                                    (cur_nimg - lr_rampup)
                                    // cfg.training.hp.lr_decay_rate
                                )
                            current_lr = g["lr"]
                            if dist.rank == 0:
                                writer.add_scalar("learning_rate", current_lr, cur_nimg)
                        handle_and_clip_gradients(
                            model,
                            grad_clip_threshold=cfg.training.hp.grad_clip_threshold,
                        )
                    with nvtx.annotate("optimizer step", color="blue"):
                        optimizer.step()

                    cur_nimg += cfg.training.hp.total_batch_size
                    done = cur_nimg >= cfg.training.hp.training_duration

                with nvtx.annotate("validation", color="red"):
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
                                    (
                                        img_clean_valid,
                                        img_lr_valid,
                                        *lead_time_label_valid,
                                    ) = next(validation_dataset_iterator)

                                    if use_apex_gn:
                                        img_clean_valid = img_clean_valid.to(
                                            dist.device,
                                            dtype=input_dtype,
                                            non_blocking=True,
                                        ).to(memory_format=torch.channels_last)
                                        img_lr_valid = img_lr_valid.to(
                                            dist.device,
                                            dtype=input_dtype,
                                            non_blocking=True,
                                        ).to(memory_format=torch.channels_last)

                                    else:
                                        img_clean_valid = (
                                            img_clean_valid.to(dist.device)
                                            .to(input_dtype)
                                            .contiguous()
                                        )
                                        img_lr_valid = (
                                            img_lr_valid.to(dist.device)
                                            .to(input_dtype)
                                            .contiguous()
                                        )

                                    loss_valid_kwargs = {
                                        "net": model,
                                        "img_clean": img_clean_valid,
                                        "img_lr": img_lr_valid,
                                        "augment_pipe": None,
                                        "use_patch_grad_acc": use_patch_grad_acc,
                                    }
                                    if use_patch_grad_acc is not None:
                                        loss_valid_kwargs[
                                            "use_patch_grad_acc"
                                        ] = use_patch_grad_acc
                                    if lead_time_label_valid:
                                        lead_time_label_valid = (
                                            lead_time_label_valid[0]
                                            .to(dist.device)
                                            .contiguous()
                                        )
                                        loss_valid_kwargs.update(
                                            {"lead_time_label": lead_time_label_valid}
                                        )
                                    if use_patch_grad_acc:
                                        loss_fn.y_mean = None

                                    for patch_num_per_iter in patch_nums_iter:
                                        if patching is not None:
                                            patching.set_patch_num(patch_num_per_iter)
                                            loss_fn_kwargs.update(
                                                {"patching": patching}
                                            )
                                        with torch.autocast(
                                            "cuda", dtype=amp_dtype, enabled=enable_amp
                                        ):
                                            loss_valid = loss_fn(**loss_valid_kwargs)

                                        loss_valid = (
                                            (loss_valid.sum() / batch_size_per_gpu)
                                            .cpu()
                                            .item()
                                        )
                                        valid_loss_accum += (
                                            loss_valid
                                            / cfg.training.io.validation_steps
                                        )
                                valid_loss_sum = torch.tensor(
                                    [valid_loss_accum], device=dist.device
                                )
                                if dist.world_size > 1:
                                    torch.distributed.barrier()
                                    torch.distributed.all_reduce(
                                        valid_loss_sum,
                                        op=torch.distributed.ReduceOp.SUM,
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
                    fields += [
                        f"training_loss_running_mean {average_loss_running_mean:<7.2f}"
                    ]
                    fields += [f"learning_rate {current_lr:<7.8f}"]
                    fields += [f"total_sec {(tick_end_time - start_time):<7.1f}"]
                    fields += [
                        f"sec_per_tick {(tick_end_time - tick_start_time):<7.1f}"
                    ]
                    fields += [
                        f"sec_per_sample {((tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg)):<7.2f}"
                    ]
                    fields += [
                        f"cpu_mem_gb {(psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
                    ]
                    if torch.cuda.is_available():
                        fields += [
                            f"peak_gpu_mem_gb {(torch.cuda.max_memory_allocated(dist.device) / 2**30):<6.2f}"
                        ]
                        fields += [
                            f"peak_gpu_mem_reserved_gb {(torch.cuda.max_memory_reserved(dist.device) / 2**30):<6.2f}"
                        ]
                        torch.cuda.reset_peak_memory_stats()
                    logger0.info(" ".join(fields))

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

            # Retain only the recent n checkpoints, if desired
            if cfg.training.io.save_n_recent_checkpoints > 0:
                for suffix in [".mdlus", ".pt"]:
                    ckpts = checkpoint_list(checkpoint_dir, suffix=suffix)
                    while len(ckpts) > cfg.training.io.save_n_recent_checkpoints:
                        os.remove(os.path.join(checkpoint_dir, ckpts[0]))
                        ckpts = ckpts[1:]

    # Done.
    logger0.info("Training Completed.")


if __name__ == "__main__":
    main()
