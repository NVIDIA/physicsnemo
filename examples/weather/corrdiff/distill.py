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

import os
import time
from contextlib import nullcontext

import psutil
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from scipy.signal import windows
import nvtx
import wandb
from typing import Callable
import numpy as np
import gc

from physicsnemo import Module
from physicsnemo.diffusion.preconditioners import EDMPrecondSuperResolution

from physicsnemo.distributed import DistributedManager
from physicsnemo.diffusion.metrics import RegressionLoss, ResidualLoss, RegressionLossCE
from physicsnemo.diffusion.multi_diffusion import RandomPatching2D
from physicsnemo.utils.logging.wandb import initialize_wandb
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils import (
    load_checkpoint,
    save_checkpoint,
    get_checkpoint_dir,
)
from physicsnemo.experimental.metrics.diffusion import tEDMResidualLoss
from physicsnemo.experimental.models.diffusion.preconditioning import (
    tEDMPrecondSuperRes,
)

from fastgen.callbacks.ct_schedule import CTScheduleCallback
from fastgen.callbacks.ema import EMACallback
from fastgen.utils.distributed.ddp import DDPWrapper
from fastgen.utils import lr_scheduler
from helpers.distill_helpers import (
    MODEL_MAP,
    PRECISION_MAP,
    DistillLoss,
    get_scheduler,
    get_window_function,
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
    """
    Safe CUDA profiler tool that falls back to no-op when CUDA is not available.
    """
    if torch.cuda.is_available():
        return torch.cuda.profiler.profile()
    else:
        return nullcontext()


def cuda_profiler_start():
    """
    Start CUDA profiler.
    """
    if torch.cuda.is_available():
        torch.cuda.profiler.start()


def cuda_profiler_stop():
    """
    Stop CUDA profiler.
    """
    if torch.cuda.is_available():
        torch.cuda.profiler.stop()


def profiler_emit_nvtx():
    """
    Emit NVTX markers for CUDA profiler.
    """
    if torch.cuda.is_available():
        return torch.autograd.profiler.emit_nvtx()
    else:
        return nullcontext()


# Distill the CorrDiff Diffusion model
@hydra.main(
    version_base="1.2", config_path="conf", config_name="config_distill_mini_diffusion"
)
def main(cfg: DictConfig) -> None:
    """
    Entry point for CorrDiff distillation training.
    """
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
    fp_optimizations = cfg.distill.perf.fp_optimizations
    songunet_checkpoint_level = cfg.distill.perf.songunet_checkpoint_level
    fp16 = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16
    logger.info(f"Saving the outputs in {os.getcwd()}")
    checkpoint_dir = get_checkpoint_dir(
        str(cfg.distill.io.get("checkpoint_dir", ".")), cfg.model.name
    )
    if cfg.distill.hp.batch_size_per_gpu == "auto":
        cfg.distill.hp.batch_size_per_gpu = (
            cfg.distill.hp.total_batch_size // dist.world_size
        )

    # Load the current number of images for resuming
    try:
        cur_nimg = load_checkpoint(
            path=checkpoint_dir,
        )
    except Exception:
        cur_nimg = 0

    # Distillation Callbacks
    callbacks = []
    distill_cfg = getattr(cfg.distill.hp, cfg.distill.hp.mode)
    callbacks_cfg = distill_cfg.get("callbacks", {})
    if "ema" in callbacks_cfg:
        callbacks.append(EMACallback(**distill_cfg.callbacks.ema))
    if "ct_schedule" in callbacks_cfg:
        callbacks.append(
            CTScheduleCallback(
                **callbacks_cfg.ct_schedule, batch_size=cfg.distill.hp.total_batch_size
            )
        )
    for callback in callbacks:
        callback.on_app_begin()

    # Set seeds and configure CUDA and cuDNN settings to ensure consistent precision
    set_seed(dist.rank + cur_nimg)
    configure_cuda_for_consistent_precision()

    # Instantiate the dataset
    for callback in callbacks:
        callback.on_dataloader_init_start(None)

    # Instantiate the dataset
    data_loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.distill.perf.dataloader_workers,
        "prefetch_factor": 2 if cfg.distill.perf.dataloader_workers > 0 else None,
    }
    (
        dataset,
        dataset_iterator,
        validation_dataset,
        validation_dataset_iterator,
    ) = init_train_valid_datasets_from_config(
        dataset_cfg,
        data_loader_kwargs,
        batch_size=cfg.distill.hp.batch_size_per_gpu,
        seed=0,
        validation_dataset_cfg=validation_dataset_cfg,
        validation=validation,
        sampler_start_idx=cur_nimg,
    )
    # Callbacks
    for callback in callbacks:
        callback.on_dataloader_init_end(
            None, dataset_iterator, validation_dataset_iterator
        )

    # Parse image configuration & update model args
    dataset_channels = len(dataset.input_channels())
    img_in_channels = dataset_channels
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())
    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels

    # Handle distribution type
    distribution = getattr(cfg.distill.hp, "distribution", None)
    student_t_nu = getattr(cfg.distill.hp, "student_t_nu", None)
    residual_loss, edm_precond_super_res = ResidualLoss, EDMPrecondSuperResolution
    if distribution is not None and cfg.model.name not in [
        "diffusion",
        "patched_diffusion",
        "lt_aware_patched_diffusion",
    ]:
        raise ValueError(
            f"cfg.distill.distribution should only be specified for diffusion models."
        )
    if distribution not in ["normal", "student_t", None]:
        raise ValueError(f"Invalid distribution {distribution}")
    if distribution == "student_t":
        if student_t_nu is None:
            raise ValueError(
                "student_t_nu must be provided in cfg.distill.hp.student_t_nu for student_t distribution"
            )
        elif student_t_nu <= 2:
            raise ValueError(f"Expected nu > 2, but got {student_t_nu}.")
        # Reassign models and class for student-t distribution
        else:
            residual_loss, edm_precond_super_res = tEDMResidualLoss, tEDMPrecondSuperRes
            logger0.info(
                f"Using student-t distribution with nu={student_t_nu}. "
                f"This is an experimental feature and APIs may change without notice."
            )

    # Parse P_mean and P_std
    P_mean = getattr(cfg.distill.hp, "P_mean", None)
    P_std = getattr(cfg.distill.hp, "P_std", None)

    # Handle patch shape
    if cfg.model.name == "lt_aware_ce_regression":
        prob_channels = dataset.get_prob_channel_index()
    else:
        prob_channels = None

    # Parse the patch shape - superpatch vs patch training for distillation
    is_superpatch = False
    if cfg.distill.hp.get("patching", None) is not None:
        patch_shape_x = cfg.distill.hp.patching.patch_shape_x
        patch_shape_y = cfg.distill.hp.patching.patch_shape_y
        # compute super-patch shape for distillation
        subpatch_num = cfg.distill.hp.patching.get("subpatch_num", 2)
        overlap_pix = cfg.distill.hp.patching.get("overlap_pix", 32)
        super_patch_shape_x = subpatch_num * (patch_shape_x - overlap_pix) + overlap_pix
        super_patch_shape_y = subpatch_num * (patch_shape_y - overlap_pix) + overlap_pix
        patching_cfg = {
            "patch_shape": (patch_shape_y, patch_shape_x),
            "overlap_pix": overlap_pix,
        }
        is_superpatch = True
    else:
        patch_shape_x = None
        patch_shape_y = None
        super_patch_shape_x = None
        super_patch_shape_y = None
        patching_cfg = {}
    if (
        super_patch_shape_x
        and super_patch_shape_y
        and super_patch_shape_y >= img_shape[0]
        and super_patch_shape_x >= img_shape[1]
    ):
        logger0.warning(
            f"Patch shape {super_patch_shape_y}x{super_patch_shape_x} is larger than \
            the image shape {img_shape[0]}x{img_shape[1]}. Patching will not be used."
        )
    super_patch_shape = (super_patch_shape_y, super_patch_shape_x)
    use_patching, img_shape, super_patch_shape = set_patch_shape(
        img_shape, super_patch_shape
    )
    if use_patching:
        # Utility to perform patches extraction and batching
        patching = RandomPatching2D(
            img_shape=img_shape,
            patch_shape=super_patch_shape,
            patch_num=cfg.distill.hp.patching.get("patch_num", 1),
        )
        logger0.info(
            f"Patch-based training enabled with patch shape {super_patch_shape} and patch num {cfg.distill.hp.patching.get('patch_num', 1)}."
        )
    else:
        patching = None
        logger0.info("Patch-based training disabled")

    # set window function with superpatch
    window = None
    if is_superpatch:
        window_function = cfg.distill.hp.patching.get("window_function", None)
        window_alpha = cfg.distill.hp.patching.get("window_alpha", 1)
        if window_function is not None:
            logger0.info(
                f"Enabling window function {window_function} with alpha {window_alpha} in superpatch training"
            )
            window = get_window_function(
                patch_shape_x=patch_shape_x,
                patch_shape_y=patch_shape_y,
                window_alpha=window_alpha,
                type=window_function,
                dtype=torch.float32,
                device=dist.device,
            )
            window = window.reshape((1, 1, window.shape[0], window.shape[1]))
    else:
        logger0.info("Window function is not used with regular patch training")

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
    if student_t_nu is not None:
        model_args["nu"] = student_t_nu
    if cfg.model.name == "lt_aware_ce_regression":
        model_args["prob_channels"] = prob_channels
    if hasattr(cfg.model, "model_args"):  # override defaults from config file
        model_args.update(OmegaConf.to_container(cfg.model.model_args))

    use_torch_compile = getattr(cfg.distill.perf, "torch_compile", False)
    use_apex_gn = getattr(cfg.distill.perf, "use_apex_gn", False)
    profile_mode = getattr(cfg.distill.perf, "profile_mode", False)

    model_args["use_apex_gn"] = use_apex_gn
    model_args["profile_mode"] = profile_mode

    if enable_amp:
        model_args["amp_mode"] = enable_amp

    # Load the diffusion checkpoint for distillation
    if (
        hasattr(cfg.distill.io, "diffusion_checkpoint_path")
        and cfg.distill.io.diffusion_checkpoint_path is not None
    ):
        diffusion_checkpoint_path = to_absolute_path(
            cfg.distill.io.diffusion_checkpoint_path
        )
        if not os.path.exists(diffusion_checkpoint_path):
            raise FileNotFoundError(
                f"Expected this diffusion checkpoint but not found: {diffusion_checkpoint_path}"
            )
        diffusion_model = Module.from_checkpoint(
            diffusion_checkpoint_path, override_args={"use_apex_gn": use_apex_gn}
        )
        diffusion_model.amp_mode = enable_amp
        diffusion_model.profile_mode = profile_mode
        diffusion_model.eval().requires_grad_(False).to(dist.device)
        if use_apex_gn:
            diffusion_model.to(memory_format=torch.channels_last)
        logger0.success("Loaded the pre-trained diffusion model")
    else:
        raise ValueError(
            "A diffusion checkpoint must be provided for distillation training. "
            "Set cfg.distill.io.diffusion_checkpoint_path."
        )
        diffusion_model = None

    if cfg.wandb.watch_model and dist.rank == 0:
        wandb.watch(diffusion_model)

    # Load the regression checkpoint if applicable
    if (
        hasattr(cfg.distill.io, "regression_checkpoint_path")
        and cfg.distill.io.regression_checkpoint_path is not None
    ):
        regression_checkpoint_path = to_absolute_path(
            cfg.distill.io.regression_checkpoint_path
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

    # Compile the teacher diffusion model and regression net if applicable
    if use_torch_compile:
        logger0.info("Compiling the diffusion model and regression net...")
        if diffusion_model:
            diffusion_model = torch.compile(diffusion_model)
        if regression_net:
            regression_net = torch.compile(regression_net)

    input_dtype = torch.float32
    if enable_amp:
        input_dtype = torch.float32
    elif fp16:
        input_dtype = torch.float16

    # Instantiate the loss function for distillation
    loss_fn = DistillLoss(
        regression_net=regression_net,
        hr_mean_conditioning=cfg.model.hr_mean_conditioning,
    )

    # Instantiate the FastGenNet for distillation
    model_cfg_update = DictConfig(
        {
            "precision": PRECISION_MAP[fp_optimizations],
            "precision_infer": PRECISION_MAP[str(input_dtype)],
            "input_shape": (diffusion_model.img_out_channels, *super_patch_shape),
            "window": window,
            "device": dist.device,
            "net": diffusion_model,
            **patching_cfg,
        },
        flags={"allow_objects": True},
    )
    model_cfg = OmegaConf.merge(model_cfg_update, distill_cfg.model)
    unwrapped_model = MODEL_MAP[cfg.distill.hp.mode](model_cfg)

    for callback in callbacks:
        callback.on_model_init_start(unwrapped_model)
    unwrapped_model.on_train_begin()

    if dist.world_size > 1:
        model = DDPWrapper(
            unwrapped_model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    else:
        model = unwrapped_model

    # Enable distributed data parallel if applicable
    # move to device and initialize before DDP
    save_models = [unwrapped_model.net.net, unwrapped_model.net.logvar_linear]
    for name in ["fake_score", "discriminator"]:
        if hasattr(unwrapped_model, name):
            save_models.append(getattr(unwrapped_model, name))
            logger0.info(f"Saving {name} model")
    if unwrapped_model.use_ema:
        save_models += [unwrapped_model.ema.net, unwrapped_model.ema.logvar_linear]
        logger0.info("Saving EMA model")

    # Compute the number of required gradient accumulation rounds
    # It is automatically used if batch_size_per_gpu * dist.world_size < total_batch_size
    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.distill.hp.total_batch_size,
        cfg.distill.hp.batch_size_per_gpu,
        dist.world_size,
    )
    batch_size_per_gpu = cfg.distill.hp.batch_size_per_gpu
    logger0.info(f"Using {num_accumulation_rounds} gradient accumulation rounds")

    # calculate patch per iter
    patch_num = getattr(cfg.distill.hp.patching, "patch_num", 1)
    if hasattr(cfg.distill.hp.patching, "max_patch_per_gpu"):
        max_patch_per_gpu = cfg.distill.hp.patching.max_patch_per_gpu
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

    # Distillation only support diffusion models.
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

    # Instantiate the optimizer
    for callback in callbacks:
        callback.on_optimizer_init_start(unwrapped_model)

    optim_cls = getattr(torch.optim, cfg.distill.hp.get("optimizer_name", "Adam"))
    optimizer = optim_cls(
        params=model.parameters(),
        **OmegaConf.to_container(cfg.distill.hp.optimizer, resolve=True),
    )

    scheduler = get_scheduler(
        name=cfg.distill.hp.scheduler_name,
        # cfg=getattr(cfg.distill.hp.scheduler, cfg.distill.hp.scheduler_name),
        cfg=OmegaConf.to_container(
            getattr(cfg.distill.hp.scheduler, cfg.distill.hp.scheduler_name),
            resolve=True,
        ),
        optimizer=optimizer,
    )
    for callback in callbacks:
        callback.on_optimizer_init_end(unwrapped_model)

    # Record the current time to measure the duration of subsequent operations.
    start_time = time.time()

    ## Load optimizer checkpoint if exists
    if dist.world_size > 1:
        torch.distributed.barrier()
    # Distill callback
    for callback in callbacks:
        callback.on_load_checkpoint_start(unwrapped_model)

    cur_nimg = load_checkpoint(
        path=checkpoint_dir,
        models=save_models,
        optimizer=optimizer,
        scheduler=scheduler,
        device=dist.device,
    )
    logger0.info(f"Resuming training from {cur_nimg} images...")

    for callback in callbacks:
        callback.on_load_checkpoint_end(unwrapped_model)

    ############################################################################
    #                            MAIN TRAINING LOOP                            #
    ############################################################################

    logger0.info(f"Training for {cfg.distill.hp.training_duration} images...")
    done = False

    # FastGen Initialization
    for callback in callbacks:
        callback.on_train_begin(
            unwrapped_model, iteration=cur_nimg // cfg.distill.hp.total_batch_size
        )

    # init variables to monitor running mean of average loss since last periodic
    average_loss_running_mean = 0
    n_average_loss_running_mean = 1
    start_nimg = cur_nimg

    # enable profiler:
    with cuda_profiler():
        with profiler_emit_nvtx():
            while not done:
                tick_start_nimg = cur_nimg
                tick_start_time = time.time()

                if cur_nimg - start_nimg == 24 * cfg.distill.hp.total_batch_size:
                    logger0.info(f"Starting Profiler at {cur_nimg}")
                    cuda_profiler_start()

                if cur_nimg - start_nimg == 25 * cfg.distill.hp.total_batch_size:
                    logger0.info(f"Stopping Profiler at {cur_nimg}")
                    cuda_profiler_stop()

                with nvtx.annotate("Training iteration", color="green"):
                    # Compute & accumulate gradients
                    optimizer.zero_grad(set_to_none=True)
                    loss_accum = 0
                    for callback in callbacks:
                        callback.on_training_step_begin(
                            unwrapped_model,
                            iteration=cur_nimg // cfg.distill.hp.total_batch_size,
                        )
                    for n_i in range(num_accumulation_rounds):
                        with nvtx.annotate(
                            f"accumulation round {n_i}", color="Magenta"
                        ):
                            with nvtx.annotate("loading data", color="green"):
                                img_clean, img_lr, *lead_time_label = next(
                                    dataset_iterator
                                )
                                data_batch = {
                                    "img_clean": img_clean,
                                    "img_lr": img_lr,
                                    "lead_time_label": lead_time_label,
                                }
                                for callback in callbacks:
                                    callback.on_training_accum_step_begin(
                                        unwrapped_model,
                                        data_batch=data_batch,
                                        iteration=cur_nimg
                                        // cfg.distill.hp.total_batch_size,
                                        accum_iter=n_i,
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
                                "iteration": cur_nimg
                                // cfg.distill.hp.total_batch_size,
                            }
                            if use_patch_grad_acc is not None:
                                loss_fn_kwargs["use_patch_grad_acc"] = (
                                    use_patch_grad_acc
                                )

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
                                        loss, loss_map, output_batch = loss_fn(
                                            **loss_fn_kwargs
                                        )

                                # loss is averaged in the loss_fn (different from train.py); we need to sum it up and divide by num_accumulation_rounds * num_patches
                                assert loss.ndim == 0, (
                                    f"Loss has {loss.ndim} dimensions, expected 0"
                                )
                                loss = (
                                    loss
                                    * patch_num_per_iter
                                    / (num_accumulation_rounds * patch_num)
                                )
                                loss_accum += loss.detach()
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
                        loss_map = {f"training/{k}": v for k, v in loss_map.items()}
                        loss_map.update(
                            {
                                "training/loss": average_loss,
                                "training/loss_running_mean": average_loss_running_mean,
                            }
                        )
                        if hasattr(unwrapped_model, "ratio"):
                            loss_map["schedule/ratio"] = unwrapped_model.ratio
                        for k, v in loss_map.items():
                            writer.add_scalar(k, v, cur_nimg)
                        if wandb.run is not None:
                            wandb.log(loss_map, step=cur_nimg)

                    ptt = is_time_for_periodic_task(
                        cur_nimg,
                        cfg.distill.io.print_progress_freq,
                        done,
                        cfg.distill.hp.total_batch_size,
                        dist.rank,
                        rank_0_only=True,
                    )
                    if ptt:
                        # reset running mean of average loss
                        average_loss_running_mean = 0
                        n_average_loss_running_mean = 1

                    # Update weights.
                    with nvtx.annotate("update weights", color="blue"):
                        if scheduler is None:
                            assert cfg.distill.hp.scheduler_name == "modulus_default"
                            scheduler_cfg = cfg.distill.hp.scheduler.modulus_default
                            lr_rampup = (
                                scheduler_cfg.lr_rampup
                            )  # ramp up the learning rate
                            for g in optimizer.param_groups:
                                if lr_rampup > 0:
                                    g["lr"] = cfg.distill.hp.opt.lr * min(
                                        cur_nimg / lr_rampup, 1
                                    )
                                if cur_nimg >= lr_rampup:
                                    g["lr"] *= scheduler_cfg.lr_decay ** (
                                        (cur_nimg - lr_rampup) // 5e6
                                    )
                                current_lr = g["lr"]
                        else:
                            scheduler.step()
                            current_lr = scheduler.get_last_lr()[0]
                        if dist.rank == 0:
                            writer.add_scalar("learning_rate", current_lr, cur_nimg)
                            if wandb.run is not None:
                                wandb.log(
                                    {"learning_rate_decay": current_lr}, step=cur_nimg
                                )

                        handle_and_clip_gradients(
                            model,
                            grad_clip_threshold=cfg.distill.hp.grad_clip_threshold,
                        )
                    with nvtx.annotate("optimizer step", color="blue"):
                        for callback in callbacks:
                            callback.on_optimizer_step_begin(
                                unwrapped_model,
                                iteration=cur_nimg // cfg.distill.hp.total_batch_size,
                            )
                        optimizer.step()

                    cur_nimg += cfg.distill.hp.total_batch_size
                    done = cur_nimg >= cfg.distill.hp.training_duration

                    for callback in callbacks:
                        callback.on_training_step_end(
                            unwrapped_model,
                            data_batch=data_batch,
                            output_batch=output_batch,
                            loss_dict=loss_map,
                            iteration=cur_nimg // cfg.distill.hp.total_batch_size,
                        )
                    del loss, loss_sum, loss_map, output_batch

                with nvtx.annotate("validation", color="red"):
                    # Validation
                    if validation_dataset_iterator is not None:
                        valid_loss_accum = 0
                        if is_time_for_periodic_task(
                            cur_nimg,
                            cfg.distill.io.validation_freq,
                            done,
                            cfg.distill.hp.total_batch_size,
                            dist.rank,
                        ):
                            for callback in callbacks:
                                callback.on_validation_begin(
                                    unwrapped_model,
                                    iteration=cur_nimg
                                    // cfg.distill.hp.total_batch_size,
                                )
                            with torch.no_grad():
                                for _ in range(cfg.distill.io.validation_steps):
                                    (
                                        img_clean_valid,
                                        img_lr_valid,
                                        *lead_time_label_valid,
                                    ) = next(validation_dataset_iterator)

                                    for callback in callbacks:
                                        callback.on_validation_step_begin(
                                            unwrapped_model,
                                            data_batch=data_batch,
                                            iteration=cur_nimg
                                            // cfg.distill.hp.total_batch_size,
                                        )

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
                                        "iteration": cur_nimg
                                        // cfg.distill.hp.total_batch_size,
                                    }
                                    if use_patch_grad_acc is not None:
                                        loss_valid_kwargs["use_patch_grad_acc"] = (
                                            use_patch_grad_acc
                                        )
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
                                            loss_valid_kwargs.update(
                                                {"patching": patching}
                                            )
                                        with torch.autocast(
                                            "cuda", dtype=amp_dtype, enabled=enable_amp
                                        ):
                                            (
                                                loss_valid,
                                                loss_map_valid,
                                                output_batch_valid,
                                            ) = loss_fn(**loss_valid_kwargs)

                                        loss_valid = (
                                            (loss_valid.sum() / batch_size_per_gpu)
                                            .cpu()
                                            .item()
                                        )
                                        valid_loss_accum += (
                                            loss_valid
                                            / cfg.distill.io.validation_steps
                                            / len(patch_nums_iter)
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
                                    loss_map_valid = {
                                        f"valid/{k}": v
                                        for k, v in loss_map_valid.items()
                                    }
                                    loss_map_valid.update(
                                        {
                                            "valid/loss": average_valid_loss,
                                        }
                                    )
                                    for k, v in loss_map_valid.items():
                                        writer.add_scalar(k, v, cur_nimg)
                                    if wandb.run is not None:
                                        wandb.log(loss_map_valid, step=cur_nimg)

                                    # generate images and log to wandb
                                    diff_out = output_batch_valid["gen_rand"]
                                    if isinstance(diff_out, Callable):
                                        with torch.autocast(
                                            "cuda", dtype=amp_dtype, enabled=enable_amp
                                        ):
                                            diff_out = diff_out()
                                    assert isinstance(diff_out, torch.Tensor)
                                    y_mean = output_batch_valid["y_mean"]
                                    if patching is not None:
                                        y_mean = patching.apply(y_mean)
                                        img_clean_valid = patching.apply(
                                            img_clean_valid
                                        )

                                    image_out = diff_out + y_mean

                                    # log first element in batch
                                    images = {
                                        "mean": y_mean,
                                        "diff": diff_out,
                                        "pred": image_out,
                                        "truth": img_clean_valid,
                                    }
                                    images = {
                                        name: validation_dataset.denormalize_output(
                                            img.float().cpu().numpy()
                                        )
                                        for name, img in images.items()
                                    }

                                    wandb_log = {"valid/loss": average_valid_loss}
                                    for batch_idx in range(images["pred"].shape[0]):
                                        for channel_idx in range(
                                            images["pred"].shape[1]
                                        ):
                                            info = validation_dataset.output_channels()[
                                                channel_idx
                                            ]
                                            channel_name = info.name + info.level
                                            channel_min = np.min(
                                                images["truth"][batch_idx, channel_idx]
                                            )
                                            channel_max = np.max(
                                                images["truth"][batch_idx, channel_idx]
                                            )
                                            span = (channel_max - channel_min) * 1.5
                                            channel_images = []
                                            for name, img in images.items():
                                                img = img[batch_idx, channel_idx]
                                                img = (
                                                    img - channel_min + 0.25 * span
                                                ) / (1.5 * span)
                                                img = (
                                                    (img * 255)
                                                    .clip(0, 255)
                                                    .astype(np.uint8)
                                                )
                                                channel_images.append(
                                                    wandb.Image(img, caption=name)
                                                )
                                            wandb_log[
                                                f"images/{channel_name}_{batch_idx}"
                                            ] = channel_images
                                    wandb.log(wandb_log, step=cur_nimg)

                                    # free memory on rank 0
                                    del images, image_out, diff_out, y_mean, wandb_log

                                for callback in callbacks:
                                    callback.on_validation_step_end(
                                        unwrapped_model,
                                        data_batch=data_batch,
                                        output_batch=output_batch_valid,
                                        loss_dict=loss_map_valid,
                                        iteration=cur_nimg
                                        // cfg.distill.hp.total_batch_size,
                                    )

                                # free memory after validation
                                loss_fn.y_mean = None
                                del img_clean_valid, img_lr_valid, lead_time_label_valid
                                del (
                                    loss_valid,
                                    valid_loss_sum,
                                    valid_loss_accum,
                                    average_valid_loss,
                                    loss_map_valid,
                                    output_batch_valid,
                                )
                                gc.collect()
                                with torch.cuda.device(dist.device):
                                    torch.cuda.empty_cache()

                            for callback in callbacks:
                                callback.on_validation_end(
                                    unwrapped_model,
                                    iteration=cur_nimg
                                    // cfg.distill.hp.total_batch_size,
                                )

                if is_time_for_periodic_task(
                    cur_nimg,
                    cfg.distill.io.print_progress_freq,
                    done,
                    cfg.distill.hp.total_batch_size,
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
                    cfg.distill.io.save_checkpoint_freq,
                    done,
                    cfg.distill.hp.total_batch_size,
                    dist.rank,
                    rank_0_only=True,
                ):
                    for callback in callbacks:
                        callback.on_save_checkpoint_start(
                            unwrapped_model,
                            iteration=cur_nimg // cfg.distill.hp.total_batch_size,
                        )
                    save_checkpoint(
                        path=checkpoint_dir,
                        models=save_models,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=cur_nimg,
                    )
                    for callback in callbacks:
                        callback.on_save_checkpoint_success(
                            unwrapped_model,
                            iteration=cur_nimg // cfg.distill.hp.total_batch_size,
                            path=checkpoint_dir,
                        )
                        callback.on_save_checkpoint_end(
                            unwrapped_model,
                            iteration=cur_nimg // cfg.distill.hp.total_batch_size,
                        )

                    # Retain only the recent n checkpoints, if desired
                    if cfg.distill.io.save_n_recent_checkpoints > 0:
                        for suffix in [".mdlus", ".pt"]:
                            ckpts = checkpoint_list(checkpoint_dir, suffix=suffix)
                            while len(ckpts) > cfg.distill.io.save_n_recent_checkpoints:
                                os.remove(os.path.join(checkpoint_dir, ckpts[0]))
                                ckpts = ckpts[1:]

    # Done.
    for callback in callbacks:
        callback.on_train_end(
            unwrapped_model, iteration=cur_nimg // cfg.distill.hp.total_batch_size
        )
        callback.on_app_end(
            unwrapped_model, iteration=cur_nimg // cfg.distill.hp.total_batch_size
        )
    logger0.info("Training Completed.")


if __name__ == "__main__":
    main()
