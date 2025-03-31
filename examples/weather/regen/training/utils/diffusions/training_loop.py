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
import copy
import json
from utils.YParams import YParams
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from utils.diffusions.generate import edm_sampler
import utils.diffusions.networks
import utils.diffusions.losses
from utils.data_loader_hrrr_era5 import get_dataset, worker_init
import matplotlib.pyplot as plt
import wandb
from utils.spectrum import compute_ps1d

# ----------------------------------------------------------------------------


def downscaling_training_loop(
    run_dir=".",  # Output directory.
    optimizer_kwargs={},  # Options for optimizer.
    seed=0,  # Global random seed.
    ema_halflife_kimg=500,  # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg=10000,  # Learning rate ramp-up duration.
    loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick=50,  # Interval of progress prints.
    snapshot_ticks=50,  # How often to save network snapshots, None = disable.
    state_dump_ticks=500,  # How often to dump training state, None = disable.
    resume_pkl=None,  # Start from the given network snapshot, None = random initialization.
    resume_state_dump=None,  # Start from the given training state, None = reset training state.
    resume_kimg=0,  # Start from the given training progress.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    device=torch.device("cuda"),
    config_file=None,
    config_name=None,
    log_to_wandb=False,
):
    params = YParams(config_file, config_name)
    batch_size = params.batch_size
    local_batch_size = batch_size // dist.get_world_size()
    img_per_tick = params.img_per_tick
    use_regression_net = params.use_regression_net
    residual = params.residual
    log_scale_residual = params.log_scale_residual
    previous_step_conditioning = params.previous_step_conditioning
    tendency_normalization = params.tendency_normalization

    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    if resume_state_dump is not None:
        print("Resuming from state dump:", resume_state_dump)
        print("Resuming from kimg:", resume_kimg)
        print("Resuming from pkl:", resume_pkl)

    # Load dataset.
    dist.print0("Loading dataset...")
    # hard code this name

    total_kimg = params.total_kimg

    dataset_obj = get_dataset(params, train=True)
    # hrrr_channels = dataset_obj.hrrr_channels.values.tolist()
    base_hrrr_channels, kept_hrrr_channels = dataset_obj._get_hrrr_channel_names()

    # hrrr_channels = hrrr_channels[:-1] #remove the last channel vil. TODO: fix this in the dataset
    hrrr_channels = kept_hrrr_channels
    diffusion_channels = params.diffusion_channels
    if diffusion_channels == "all":
        diffusion_channels = hrrr_channels
    input_channels = params.input_channels
    diffusion_channel_indices = [
        hrrr_channels.index(channel) for channel in diffusion_channels
    ]
    dist.print0("diffusion_channel_indices", diffusion_channel_indices)
    if input_channels == "all":
        input_channel_indices = [
            hrrr_channels.index(channel) for channel in hrrr_channels
        ]
        input_channels = hrrr_channels
    else:
        input_channel_indices = [
            hrrr_channels.index(channel) for channel in input_channels
        ]

    sampler = misc.InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_obj,
        batch_size=local_batch_size,
        num_workers=params.num_data_workers,
        sampler=sampler,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    dataset_iterator = iter(data_loader)

    # Construct network.
    dist.print0("Constructing network...")
    resolution = params.crop_size if params.crop_size is not None else 512
    target_channels = len(diffusion_channels)

    label_dim = 0

    dist.print0("hrrr_channels", kept_hrrr_channels)
    dist.print0("target_channels for diffusion", target_channels)

    net = utils.diffusions.networks.get_preconditioned_architecture(
        name="ddpmpp-cwb-v0",
        resolution=resolution,
        target_channels=target_channels,
        conditional_channels=0,
        label_dim=0,
        spatial_embedding=params.spatial_pos_embed,
        attn_resolutions=params.attn_resolutions,
    )

    # Setup optimizer.
    loss_fn = utils.diffusions.losses.EDMLoss(P_mean=params.P_mean)
    assert net.sigma_min < net.sigma_max
    net.train().requires_grad_(True).to(device)

    optimizer = dnnlib.util.construct_class_by_name(
        params=net.parameters(), **optimizer_kwargs
    )  # subclass of torch.optim.Optimizer
    augment_pipe = None
    ddp = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[device], broadcast_buffers=False
    )
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=net, require_all=False
        )
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=ema, require_all=False
        )
        del data  # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device("cpu"))
        misc.copy_params_and_buffers(
            src_module=data["net"], dst_module=net, require_all=True
        )
        optimizer.load_state_dict(data["optimizer_state"])
        del data  # conserve memory

    # Train.
    dist.print0(f"Training for {total_kimg} kimg...")
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    wandb_logs = {}

    def transform(x):
        return utils.img_utils.image_to_crops(x, resolution, resolution)

    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        batch = next(dataset_iterator)
        hrrr = batch["hrrr"].to(device).to(torch.float32)
        hrrr = hrrr[:, diffusion_channel_indices, :, :]

        loss = loss_fn(net=ddp, x=hrrr, condition=None, augment_pipe=augment_pipe)
        channelwise_loss = loss.mean(dim=(0, 2, 3))
        channelwise_loss_dict = {
            f"ChLoss/{diffusion_channels[i]}": channelwise_loss[i].item()
            for i in range(target_channels)
        }
        training_stats.report("Loss/loss", loss)
        loss_value = loss.sum() / target_channels
        if log_to_wandb:
            wandb_logs["loss"] = loss_value.item()
            wandb_logs["channelwise_loss"] = channelwise_loss_dict

        loss_value.backward()

        if params.clip_grad_norm is not None:

            torch.nn.utils.clip_grad_norm_(net.parameters(), params.clip_grad_norm)

        # Update weights.
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )
            if log_to_wandb:
                wandb_logs["lr"] = g["lr"]
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        effective_batch_size = (batch_size // local_batch_size) * hrrr.shape[0]
        ema_beta = 0.5 ** (effective_batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += effective_batch_size
        done = cur_nimg >= total_kimg * 1000
        if (
            (not done)
            and (cur_tick != 0)
            and (cur_nimg < tick_start_nimg + img_per_tick)
        ):
            continue

        # make inference
        if cur_tick % params.validate_every == 0:
            with torch.no_grad():
                n = 1
                hrrr = batch["hrrr"]
                hrrr = hrrr.to(torch.float32).to(device)

                with torch.no_grad():
                    latents = torch.randn_like(
                        hrrr[0:n, diffusion_channel_indices, :, :]
                    )
                    output_images = edm_sampler(net, latents=latents, condition=None)

                hrrr = hrrr[0:n, diffusion_channel_indices, :, :]

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"
        ]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"
        ]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(" ".join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0("Aborting...")

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value  # conserve memory
            if dist.get_rank() == 0:
                with open(
                    os.path.join(run_dir, f"network-snapshot-{cur_nimg//1000:06d}.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(data, f)

            del data  # conserve memory

        if cur_tick % params.validate_every == 0:

            if log_to_wandb:
                if dist.get_rank() == 0:
                    print("logging to wandb")
                    wandb.log(wandb_logs, step=cur_nimg)

            if dist.get_rank() == 0:
                # TODO: improve the image saving and run_dir setup for thread safe image saving from all ranks

                for i in range(output_images.shape[0]):
                    image = output_images[i].cpu().numpy()
                    hrrr_channels = dataset_obj.hrrr_channels
                    fields = ["10u", "10v", "tp"]

                    # Compute spectral metrics
                    figs, spec_ratios = compute_ps1d(
                        output_images[i], hrrr[i], fields, diffusion_channels
                    )
                    if log_to_wandb:
                        wandb.log(spec_ratios, step=cur_nimg)
                        for figname, fig in figs.items():
                            wandb.log({figname: wandb.Image(fig)}, step=cur_nimg)

                    for f_ in fields:
                        f_index = diffusion_channels.index(f_)
                        image_dir = os.path.join(run_dir, "images", f_)
                        generated = image[f_index]
                        truth = hrrr[i, f_index].cpu().numpy()

                        fig, (a, b) = plt.subplots(1, 2)
                        im = a.imshow(generated)
                        a.set_title("generated, {}.png".format(f_))
                        plt.colorbar(im, fraction=0.046, pad=0.04)
                        im = b.imshow(truth)
                        b.set_title("truth")
                        plt.colorbar(im, fraction=0.046, pad=0.04)
                        os.makedirs(image_dir, exist_ok=True)
                        plt.savefig(os.path.join(image_dir, f"{cur_tick}_{i}_{f_}.png"))
                        plt.close("all")

                        specfig = "PS1D_" + f_
                        figs[specfig].savefig(
                            os.path.join(image_dir, f"{cur_tick}{i}{f_}_spec.png")
                        )
                        plt.close(figs[specfig])

                        # log the images to wandb
                        if log_to_wandb:
                            # log fig to wandb
                            wandb.log({f"generated_{f_}": fig}, step=cur_nimg)

        # Save full dump of the training state.
        if (
            (state_dump_ticks is not None)
            and (done or cur_tick % state_dump_ticks == 0)
            and cur_tick != 0
            and dist.get_rank() == 0
        ):
            torch.save(
                dict(net=net, optimizer_state=optimizer.state_dict()),
                os.path.join(run_dir, f"training-state-{cur_nimg//1000:06d}.pt"),
            )

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
            stats_jsonl.write(
                json.dumps(
                    dict(
                        training_stats.default_collector.as_dict(),
                        timestamp=time.time(),
                    )
                )
                + "\n"
            )
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0("Exiting...")


def training_loop(**c):
    params = YParams(c["config_file"], c["config_name"])
    if params.task == "downscale":
        return downscaling_training_loop(**c)
    else:  # default is forecasting
        return forecasting_training_loop(**c)
