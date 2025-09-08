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

"""Main training loop."""

import os
import time
import numpy as np
import torch
import psutil
from physicsnemo.models import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.metrics.diffusion import EDMLoss
from physicsnemo.utils.diffusion import InfiniteSampler

from physicsnemo.launch.utils import save_checkpoint, load_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from utils.nn import (
    diffusion_model_forward,
    regression_loss_fn,
    get_preconditioned_architecture,
    build_network_condition_and_target,
    unpack_batch,
)
from utils.plots import validation_plot
from datasets import dataset_classes
from datasets.dataset import worker_init
import matplotlib.pyplot as plt
import wandb
from utils.spectrum import ps1d_plots
from torch.nn.utils import clip_grad_norm_


logger = PythonLogger("train")


def training_loop(cfg):
    # Initialize.
    start_time = time.time()
    dist = DistributedManager()
    device = dist.device
    logger0 = RankZeroLoggingWrapper(logger, dist)

    # Shorthand for config items
    batch_size = cfg.training.batch_size
    if cfg.training.batch_size_per_gpu == "auto":
        local_batch_size = batch_size // dist.world_size
    else:
        local_batch_size = cfg.training.batch_size_per_gpu
    assert batch_size % (local_batch_size * dist.world_size) == 0
    # gradient accumulation rounds per step
    num_accumulation_rounds = batch_size // (local_batch_size * dist.world_size)

    log_to_wandb = cfg.training.log_to_wandb

    loss_type = cfg.training.loss
    if loss_type == "regression":
        net_name = "regression"
    elif loss_type == "edm":
        net_name = "diffusion"

    condition_list = (
        cfg.model.regression_conditions
        if net_name == "regression"
        else cfg.model.diffusion_conditions
    )

    # Seed and Performance settings
    np.random.seed((cfg.training.seed * dist.world_size + dist.rank) % (1 << 31))
    torch.manual_seed(cfg.training.seed)
    torch.backends.cudnn.benchmark = cfg.training.cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    fp_optimizations = cfg.training.fp_optimizations
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if fp_optimizations == "amp-fp16" else torch.bfloat16

    total_train_steps = cfg.training.total_train_steps

    # Load dataset.
    logger0.info("Loading dataset...")

    dataset_cls = dataset_classes[cfg.dataset.name]
    del cfg.dataset.name

    dataset_train = dataset_cls(cfg.dataset, train=True)
    dataset_valid = dataset_cls(cfg.dataset, train=False)

    background_channels = dataset_train.background_channels()
    state_channels = dataset_train.state_channels()
    lead_time_steps = dataset_train.lead_time_steps

    sampler = InfiniteSampler(
        dataset=dataset_train,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=cfg.training.seed,
    )
    valid_sampler = InfiniteSampler(
        dataset=dataset_valid,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=cfg.training.seed,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=local_batch_size,
        num_workers=cfg.training.num_data_workers,
        sampler=sampler,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    valid_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=local_batch_size,
        num_workers=cfg.training.num_data_workers,
        sampler=valid_sampler,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    dataset_iterator = iter(data_loader)
    valid_dataset_iterator = iter(valid_data_loader)

    # load pretrained regression net if training diffusion
    if "regression" in condition_list:
        regression_net = Module.from_checkpoint(cfg.model.regression_weights)
        if cfg.training.compile_model:
            regression_net = torch.compile(regression_net)
        regression_net = regression_net.eval().requires_grad_(False).to(device)
    else:
        regression_net = None

    invariant_array = dataset_train.get_invariants()
    if invariant_array is not None:
        invariant_tensor = torch.from_numpy(invariant_array).to(device)
        invariant_tensor = invariant_tensor.unsqueeze(0)
        invariant_tensor = invariant_tensor.repeat(local_batch_size, 1, 1, 1)
    else:
        invariant_tensor = None

    # Construct network
    logger0.info("Constructing network...")
    num_condition_channels = {
        "state": len(state_channels),
        "background": len(background_channels),
        "regression": len(state_channels),
        "invariant": 0 if invariant_tensor is None else invariant_tensor.shape[1],
    }
    num_condition_channels = sum(num_condition_channels[c] for c in condition_list)

    logger0.info(f"model conditions {condition_list}")
    logger0.info(f"background_channels {background_channels}")
    logger0.info(f"state_channels {state_channels}")
    logger0.info(f"num_condition_channels {num_condition_channels}")

    net = get_preconditioned_architecture(
        name=net_name,
        img_resolution=dataset_train.image_shape(),
        target_channels=len(state_channels),
        conditional_channels=num_condition_channels,
        spatial_embedding=cfg.model.spatial_pos_embed,
        attn_resolutions=list(cfg.model.attn_resolutions),
        lead_time_steps=lead_time_steps,
        amp_mode=enable_amp,
        **cfg.model.get("hyperparameters", {}),
    )

    net.train().requires_grad_(True).to(device)

    # Setup optimizer.
    logger0.info("Setting up optimizer...")
    if cfg.training.loss == "regression":
        loss_fn = regression_loss_fn
    elif cfg.training.loss == "edm":
        loss_fn = EDMLoss(P_mean=cfg.model.P_mean)
    if cfg.training.compile_model:
        loss_fn = torch.compile(loss_fn)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.training.lr)
    augment_pipe = None
    ddp = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[device], broadcast_buffers=False
    )

    # Resume training from previous snapshot.
    ckpt_path = os.path.join(cfg.training.rundir, f"checkpoints_{net_name}")
    logger0.info(f'Trying to resume training state from "{ckpt_path}"...')
    total_steps = load_checkpoint(
        path=ckpt_path,
        models=net,
        optimizer=optimizer,
        epoch=None
        if cfg.training.resume_checkpoint == "latest"
        else cfg.training.resume_checkpoint,
    )

    if total_steps == 0:
        logger0.info(f"No resumable training state found.")
        init_weights = cfg.training.initial_weights
        if init_weights is None:
            logger0.info(f"Starting training from scratch...")
        else:
            logger0.info(f"Starting training from weights saved in {init_weights}...")
            if init_weights.endswith(".mdlus"):
                net.load(init_weights)
            elif init_weights.endswith(".pt"):
                model_dict = torch.load(init_weights, map_location=net.device)
                net.load_state(model_dict, strict=True)

    # Train.
    logger0.info(
        f"Training up to {total_train_steps} steps starting from step {total_steps}..."
    )
    wandb_logs = {}
    done = total_steps >= total_train_steps

    train_start = time.time()
    avg_train_loss = 0
    train_steps = 0
    valid_time = -1  # set as placeholders until validation is actually done
    val_loss = -1
    while not done:
        # Compute & accumulate gradients.
        optimizer.zero_grad(set_to_none=True)

        for _ in range(num_accumulation_rounds):
            # Format input batch
            batch = next(dataset_iterator)
            (background, state, lead_time_label) = unpack_batch(batch, device)

            with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                (condition, target, reg_out) = build_network_condition_and_target(
                    background,
                    state,
                    invariant_tensor,
                    lead_time_label=lead_time_label,
                    regression_net=regression_net,
                    condition_list=condition_list,
                    regression_condition_list=cfg.model.regression_conditions,
                )
                loss = loss_fn(
                    net=ddp,
                    images=target,
                    condition=condition,
                    augment_pipe=augment_pipe,
                    lead_time_label=lead_time_label,
                )

            if log_to_wandb:
                channelwise_loss = loss.mean(dim=(0, 2, 3))
                channelwise_loss_dict = {
                    f"ChLoss/{ch}": channelwise_loss[i].item()
                    for (i, ch) in enumerate(state_channels)
                }
                wandb_logs["channelwise_loss"] = channelwise_loss_dict

            loss_value = loss.sum() / len(state_channels)
            loss_value.backward()

        if cfg.training.clip_grad_norm > 0:
            clip_grad_norm_(net.parameters(), cfg.training.clip_grad_norm)

        # Update weights.
        for g in optimizer.param_groups:
            g["lr"] = cfg.training.lr * min(
                total_steps / max(cfg.training.lr_rampup_steps, 1e-8), 1
            )
            if log_to_wandb:
                wandb_logs["lr"] = g["lr"]
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )

        optimizer.step()

        if dist.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)

        avg_train_loss += loss.mean().cpu().item()
        train_steps += 1
        if log_to_wandb:
            wandb_logs["loss"] = loss.mean().cpu().item() / train_steps

        # Perform maintenance tasks once per tick.
        total_steps += 1
        done = total_steps >= total_train_steps

        # Perform validation step
        if total_steps % cfg.training.validation_freq == 0:
            valid_start = time.time()
            batch = next(valid_dataset_iterator)

            with torch.no_grad():
                (background, state, lead_time_label) = unpack_batch(batch, device)

                with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                    (condition, target, reg_out) = build_network_condition_and_target(
                        background,
                        state,
                        invariant_tensor,
                        lead_time_label=lead_time_label,
                        regression_net=regression_net,
                        condition_list=condition_list,
                        regression_condition_list=cfg.model.regression_conditions,
                    )

                    # evaluate validation loss
                    loss_kwargs = (
                        {"return_model_outputs": True}
                        if net_name == "regression"
                        else {}
                    )
                    valid_loss = loss_fn(
                        net=net,
                        images=target,
                        condition=condition,
                        augment_pipe=augment_pipe,
                        lead_time_label=lead_time_label,
                        **loss_kwargs,
                    )

                    if net_name == "diffusion":
                        output_images = diffusion_model_forward(
                            net,
                            condition,
                            state[1].shape,
                            sampler_args=dict(cfg.sampler.args),
                            lead_time_label=lead_time_label,
                        )
                        if "regression" in condition_list:
                            output_images += reg_out
                        del reg_out
                    else:
                        (
                            valid_loss,
                            output_images,
                        ) = valid_loss  # in this case, loss_fn returns a tuple
                        if log_to_wandb:
                            channelwise_valid_loss = valid_loss.mean(dim=[0, 2, 3])
                            channelwise_valid_loss_dict = {
                                f"ChLoss_valid/{state_channels[i]}": channelwise_valid_loss[
                                    i
                                ].item()
                                for i in range(state_channels)
                            }
                            wandb_logs["channelwise_valid_loss"] = (
                                channelwise_valid_loss_dict
                            )

                if dist.world_size > 1:
                    torch.distributed.barrier()
                    torch.distributed.all_reduce(
                        valid_loss, op=torch.distributed.ReduceOp.AVG
                    )
                val_loss = valid_loss.mean().cpu().item()
                if log_to_wandb:
                    wandb_logs["valid_loss"] = val_loss

            # Save plots locally (and optionally to wandb)
            if dist.rank == 0:
                for i in range(output_images.shape[0]):
                    image = output_images[i].cpu().numpy()
                    fields = cfg.training.validation_plot_variables

                    # Compute spectral metrics
                    figs, spec_ratios = ps1d_plots(
                        output_images[i], state[1][i], fields, state_channels
                    )

                    for f_ in fields:
                        f_index = state_channels.index(f_)
                        image_dir = os.path.join(cfg.training.rundir, "images", f_)
                        os.makedirs(image_dir, exist_ok=True)

                        generated = image[f_index]
                        truth = state[1][i, f_index].cpu().numpy()

                        fig = validation_plot(generated, truth, f_)
                        fig.savefig(
                            os.path.join(image_dir, f"{total_steps}_{i}_{f_}.png")
                        )

                        specfig = "PS1D_" + f_
                        figs[specfig].savefig(
                            os.path.join(image_dir, f"{total_steps}_{i}_{f_}_spec.png")
                        )
                        if log_to_wandb:
                            # Save plots as wandb Images
                            for figname, plot in figs.items():
                                wandb_logs[figname] = wandb.Image(plot)
                            wandb_logs.update({f"generated_{f_}": wandb.Image(fig)})

                    plt.close("all")

                if log_to_wandb:
                    wandb_logs.update(spec_ratios)
                    wandb.log(wandb_logs, step=total_steps)

            valid_time = time.time() - valid_start

        # Print training stats
        current_time = time.time()
        if total_steps % cfg.training.print_progress_freq == 0:
            fields = []
            fields += [f"steps {total_steps:<5d}"]
            fields += [f"samples {total_steps * batch_size}"]
            fields += [f"tot_time {current_time - start_time: .2f}"]
            fields += [f"step_time {(current_time - train_start) / train_steps: .2f}"]
            fields += [f"valid_time {valid_time: .2f}"]
            fields += [
                f"cpumem {psutil.Process(os.getpid()).memory_info().rss / 2**30:<6.2f}"
            ]
            fields += [
                f"gpumem {torch.cuda.max_memory_allocated(device) / 2**30:<6.2f}"
            ]
            fields += [f"train_loss {avg_train_loss / train_steps:<6.3f}"]
            fields += [f"val_loss {val_loss:<6.3f}"]
            logger0.info(" ".join(fields))

            # Reset counters
            train_steps = 0
            train_start = time.time()
            avg_train_loss = 0
            torch.cuda.reset_peak_memory_stats()

        # Save full dump of the training state.
        if (
            (done or total_steps % cfg.training.checkpoint_freq == 0)
            and total_steps != 0
            and dist.rank == 0
        ):
            save_checkpoint(
                path=os.path.join(cfg.training.rundir, f"checkpoints_{net_name}"),
                models=net,
                optimizer=optimizer,
                epoch=total_steps,
            )

    # Done.
    torch.distributed.barrier()
    logger0.info("\nExiting...")
