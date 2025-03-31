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

import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import torch._dynamo
import nvtx
import numpy as np
import netCDF4 as nc
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo import Module
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from einops import rearrange
from torch.distributed import gather


from hydra.utils import to_absolute_path
from physicsnemo.utils.generative import deterministic_sampler, stochastic_sampler
from physicsnemo.utils.corrdiff import (
    NetCDFWriter,
    get_time_from_range,
    regression_step,
    diffusion_step,
)


from helpers.generate_helpers import (
    get_dataset_and_sampler,
    save_images,
)
from helpers.train_helpers import set_patch_shape


@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # Initialize logger
    logger = PythonLogger("generate")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate.log")

    # Handle the batch size
    seeds = list(np.arange(cfg.generation.num_ensembles))
    num_batches = (
        (len(seeds) - 1) // (cfg.generation.seed_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

    # Synchronize
    if dist.world_size > 1:
        torch.distributed.barrier()

    # Parse the inference input times
    if cfg.generation.times_range and cfg.generation.times:
        raise ValueError("Either times_range or times must be provided, but not both")
    if cfg.generation.times_range:
        times = get_time_from_range(cfg.generation.times_range)
    else:
        times = cfg.generation.times

    # Create dataset object
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    if "has_lead_time" in cfg.generation:
        has_lead_time = cfg.generation["has_lead_time"]
    else:
        has_lead_time = False
    dataset, sampler = get_dataset_and_sampler(
        dataset_cfg=dataset_cfg, times=times, has_lead_time=has_lead_time
    )
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())

    # Parse the patch shape
    if hasattr(cfg.generation, "patch_shape_x"):  # TODO better config handling
        patch_shape_x = cfg.generation.patch_shape_x
    else:
        patch_shape_x = None
    if hasattr(cfg.generation, "patch_shape_y"):
        patch_shape_y = cfg.generation.patch_shape_y
    else:
        patch_shape_y = None
    patch_shape = (patch_shape_y, patch_shape_x)
    img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if patch_shape != img_shape:
        logger0.info("Patch-based training enabled")
    else:
        logger0.info("Patch-based training disabled")

    # Parse the inference mode
    if cfg.generation.inference_mode == "regression":
        load_net_reg, load_net_res = True, False
    elif cfg.generation.inference_mode == "diffusion":
        load_net_reg, load_net_res = False, True
    elif cfg.generation.inference_mode == "all":
        load_net_reg, load_net_res = True, True
    else:
        raise ValueError(f"Invalid inference mode {cfg.generation.inference_mode}")

    # Load diffusion network, move to device, change precision
    if load_net_res:
        res_ckpt_filename = cfg.generation.io.res_ckpt_filename
        logger0.info(f'Loading residual network from "{res_ckpt_filename}"...')
        net_res = Module.from_checkpoint(to_absolute_path(res_ckpt_filename))
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
        if cfg.generation.perf.force_fp16:
            net_res.use_fp16 = True
    else:
        net_res = None

    # load regression network, move to device, change precision
    if load_net_reg:
        reg_ckpt_filename = cfg.generation.io.reg_ckpt_filename
        logger0.info(f'Loading network from "{reg_ckpt_filename}"...')
        net_reg = Module.from_checkpoint(to_absolute_path(reg_ckpt_filename))
        net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
        if cfg.generation.perf.force_fp16:
            net_reg.use_fp16 = True
    else:
        net_reg = None

    # Reset since we are using a different mode.
    if cfg.generation.perf.use_torch_compile:
        torch._dynamo.reset()
        # Only compile residual network
        # Overhead of compiling regression network outweights any benefits
        if net_res:
            net_res = torch.compile(net_res, mode="reduce-overhead")

    # Partially instantiate the sampler based on the configs
    if cfg.sampler.type == "deterministic":
        if cfg.generation.hr_mean_conditioning:
            raise NotImplementedError(
                "High-res mean conditioning is not yet implemented for the deterministic sampler"
            )
        sampler_fn = partial(
            deterministic_sampler,
            num_steps=cfg.sampler.num_steps,
            # num_ensembles=cfg.generation.num_ensembles,
            solver=cfg.sampler.solver,
        )
    elif cfg.sampler.type == "stochastic":
        sampler_fn = partial(
            stochastic_sampler,
            img_shape=img_shape[1],
            patch_shape=patch_shape[1],
            boundary_pix=cfg.sampler.boundary_pix,
            overlap_pix=cfg.sampler.overlap_pix,
        )
    else:
        raise ValueError(f"Unknown sampling method {cfg.sampling.type}")

    # Main generation definition
    def generate_fn():
        img_shape_y, img_shape_x = img_shape
        with nvtx.annotate("generate_fn", color="green"):
            if cfg.generation.sample_res == "full":
                image_lr_patch = image_lr
            else:
                torch.cuda.nvtx.range_push("rearrange")
                image_lr_patch = rearrange(
                    image_lr,
                    "b c (h1 h) (w1 w) -> (b h1 w1) c h w",
                    h1=img_shape_y // patch_shape[0],
                    w1=img_shape_x // patch_shape[1],
                )
                torch.cuda.nvtx.range_pop()
            image_lr_patch = image_lr_patch.to(memory_format=torch.channels_last)

            if net_reg:
                with nvtx.annotate("regression_model", color="yellow"):
                    image_reg = regression_step(
                        net=net_reg,
                        img_lr=image_lr_patch,
                        latents_shape=(
                            cfg.generation.seed_batch_size,
                            img_out_channels,
                            img_shape[0],
                            img_shape[1],
                        ),
                        lead_time_label=lead_time_label,
                    )
            if net_res:
                if cfg.generation.hr_mean_conditioning:
                    mean_hr = image_reg[0:1]
                else:
                    mean_hr = None
                with nvtx.annotate("diffusion model", color="purple"):
                    image_res = diffusion_step(
                        net=net_res,
                        sampler_fn=sampler_fn,
                        seed_batch_size=cfg.generation.seed_batch_size,
                        img_shape=img_shape,
                        img_out_channels=img_out_channels,
                        rank_batches=rank_batches,
                        img_lr=image_lr_patch.expand(
                            cfg.generation.seed_batch_size, -1, -1, -1
                        ).to(memory_format=torch.channels_last),
                        rank=dist.rank,
                        device=device,
                        hr_mean=mean_hr,
                        lead_time_label=lead_time_label,
                    )
            if cfg.generation.inference_mode == "regression":
                image_out = image_reg
            elif cfg.generation.inference_mode == "diffusion":
                image_out = image_res
            else:
                image_out = image_reg + image_res

            if cfg.generation.sample_res != "full":
                image_out = rearrange(
                    image_out,
                    "(b h1 w1) c h w -> b c (h1 h) (w1 w)",
                    h1=img_shape_y // patch_shape[0],
                    w1=img_shape_x // patch_shape[1],
                )
            # Gather tensors on rank 0
            if dist.world_size > 1:
                if dist.rank == 0:
                    gathered_tensors = [
                        torch.zeros_like(
                            image_out, dtype=image_out.dtype, device=image_out.device
                        )
                        for _ in range(dist.world_size)
                    ]
                else:
                    gathered_tensors = None

                torch.distributed.barrier()
                gather(
                    image_out,
                    gather_list=gathered_tensors if dist.rank == 0 else None,
                    dst=0,
                )

                if dist.rank == 0:
                    return torch.cat(gathered_tensors)
                else:
                    return None
            else:
                return image_out

    # generate images
    output_path = getattr(cfg.generation.io, "output_filename", "corrdiff_output.nc")
    logger0.info(f"Generating images, saving results to {output_path}...")
    batch_size = 1
    warmup_steps = min(len(times) - 1, 2)
    # Generates model predictions from the input data using the specified
    # `generate_fn`, and save the predictions to the provided NetCDF file. It iterates
    # through the dataset using a data loader, computes predictions, and saves them along
    # with associated metadata.
    if dist.rank == 0:
        f = nc.Dataset(output_path, "w")
        # add attributes
        f.cfg = str(cfg)

    with torch.cuda.profiler.profile():
        with torch.autograd.profiler.emit_nvtx():

            data_loader = torch.utils.data.DataLoader(
                dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True
            )
            time_index = -1
            if dist.rank == 0:
                writer = NetCDFWriter(
                    f,
                    lat=dataset.latitude(),
                    lon=dataset.longitude(),
                    input_channels=dataset.input_channels(),
                    output_channels=dataset.output_channels(),
                    has_lead_time=has_lead_time,
                )

                # Initialize threadpool for writers
                writer_executor = ThreadPoolExecutor(
                    max_workers=cfg.generation.perf.num_writer_workers
                )
                writer_threads = []

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            times = dataset.time()
            for image_tar, image_lr, index, *lead_time_label in iter(data_loader):
                time_index += 1
                if dist.rank == 0:
                    logger0.info(f"starting index: {time_index}")

                if time_index == warmup_steps:
                    start.record()

                # continue
                if lead_time_label:
                    lead_time_label = lead_time_label[0].to(dist.device).contiguous()
                else:
                    lead_time_label = None
                image_lr = (
                    image_lr.to(device=device)
                    .to(torch.float32)
                    .to(memory_format=torch.channels_last)
                )
                image_tar = image_tar.to(device=device).to(torch.float32)
                image_out = generate_fn()
                if dist.rank == 0:
                    batch_size = image_out.shape[0]
                    # write out data in a seperate thread so we don't hold up inferencing
                    writer_threads.append(
                        writer_executor.submit(
                            save_images,
                            writer,
                            dataset,
                            list(times),
                            image_out.cpu(),
                            image_tar.cpu(),
                            image_lr.cpu(),
                            time_index,
                            index[0],
                            has_lead_time,
                        )
                    )
            end.record()
            end.synchronize()
            elapsed_time = start.elapsed_time(end) / 1000.0  # Convert ms to s
            timed_steps = time_index + 1 - warmup_steps
            if dist.rank == 0:
                average_time_per_batch_element = elapsed_time / timed_steps / batch_size
                logger.info(
                    f"Total time to run {timed_steps} steps and {batch_size} members = {elapsed_time} s"
                )
                logger.info(
                    f"Average time per batch element = {average_time_per_batch_element} s"
                )

            # make sure all the workers are done writing
            if dist.rank == 0:
                for thread in list(writer_threads):
                    thread.result()
                    writer_threads.remove(thread)
                writer_executor.shutdown()

    if dist.rank == 0:
        f.close()
    logger0.info("Generation Completed.")


if __name__ == "__main__":
    main()
