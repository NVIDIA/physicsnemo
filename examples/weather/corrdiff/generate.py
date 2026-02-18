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

import contextlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path
import torch
import torch._dynamo
from torch.distributed import gather
import numpy as np
import nvtx
import netCDF4 as nc
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.experimental.models.diffusion.preconditioning import (
    tEDMPrecondSuperRes,
)
from physicsnemo.diffusion.multi_diffusion import GridPatching2D
from physicsnemo import Module
from physicsnemo.diffusion.samplers import (
    deterministic_sampler,
    stochastic_sampler,
)
from physicsnemo.diffusion.generate import regression_step, diffusion_step

from helpers.generate_helpers import (
    get_dataset_and_sampler,
    save_images,
    NetCDFWriter,
    get_time_from_range,
)
from helpers.train_helpers import set_patch_shape
from datasets.dataset import register_dataset


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

    # Register dataset (if custom dataset)
    register_dataset(cfg.dataset.type)
    logger0.info(f"Using dataset: {cfg.dataset.type}")

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
    if cfg.generation.patching:
        patch_shape_x = cfg.generation.patch_shape_x
        patch_shape_y = cfg.generation.patch_shape_y
    else:
        patch_shape_x, patch_shape_y = None, None
    patch_shape = (patch_shape_y, patch_shape_x)
    use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if use_patching:
        patching = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            boundary_pix=cfg.generation.boundary_pix,
            overlap_pix=cfg.generation.overlap_pix,
        )
        logger0.info("Patch-based training enabled")
    else:
        patching = None
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
        net_res = Module.from_checkpoint(
            to_absolute_path(res_ckpt_filename),
            override_args={
                "use_apex_gn": getattr(cfg.generation.perf, "use_apex_gn", False)
            },
        )
        net_res.profile_mode = getattr(cfg.generation.perf, "profile_mode", False)
        net_res.use_fp16 = getattr(cfg.generation.perf, "use_fp16", False)
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)

        # Disable AMP for inference (even if model is trained with AMP)
        if hasattr(net_res, "amp_mode"):
            net_res.amp_mode = False
    else:
        net_res = None

    # load regression network, move to device, change precision
    if load_net_reg:
        reg_ckpt_filename = cfg.generation.io.reg_ckpt_filename
        logger0.info(f'Loading network from "{reg_ckpt_filename}"...')
        net_reg = Module.from_checkpoint(
            to_absolute_path(reg_ckpt_filename),
            override_args={
                "use_apex_gn": getattr(cfg.generation.perf, "use_apex_gn", False)
            },
        )
        net_reg.profile_mode = getattr(cfg.generation.perf, "profile_mode", False)
        net_reg.use_fp16 = getattr(cfg.generation.perf, "use_fp16", False)
        net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)

        # Disable AMP for inference (even if model is trained with AMP)
        if hasattr(net_reg, "amp_mode"):
            net_reg.amp_mode = False
    else:
        net_reg = None

    # Reset since we are using a different mode.
    if cfg.generation.perf.use_torch_compile:
        torch._dynamo.config.cache_size_limit = 264
        torch._dynamo.reset()
        if net_res:
            net_res = torch.compile(net_res)
        if net_reg:
            net_reg = torch.compile(net_reg)

    # Partially instantiate the sampler based on the configs
    if cfg.sampler.type == "deterministic":
        sampler_fn = partial(
            deterministic_sampler,
            num_steps=cfg.sampler.num_steps,
            # num_ensembles=cfg.generation.num_ensembles,
            solver=cfg.sampler.solver,
            patching=patching,
        )
    elif cfg.sampler.type == "stochastic":
        sampler_fn = partial(
            stochastic_sampler, patching=patching, num_steps=cfg.sampler.num_steps
        )
    else:
        raise ValueError(f"Unknown sampling method {cfg.sampling.type}")

    # Parse the distribution type
    distribution = getattr(cfg.generation, "distribution", None)
    student_t_nu = getattr(cfg.generation, "student_t_nu", None)
    if distribution is not None and not cfg.generation.inference_mode in [
        "diffusion",
        "all",
    ]:
        raise ValueError(
            f"cfg.generation.distribution should only be specified for "
            f"inference mode 'diffusion' or 'all', but got {cfg.generation.inference_mode}."
        )
    if distribution not in ["normal", "student_t", None]:
        raise ValueError(f"Invalid distribution: {distribution}.")
    if distribution == "student_t":
        if student_t_nu is None:
            raise ValueError(
                "student_t_nu must be provided in cfg.generation.student_t_nu for student_t distribution"
            )
        elif student_t_nu <= 2:
            raise ValueError(f"Expected nu > 2, but got {student_t_nu}.")
        if net_res and not isinstance(net_res, tEDMPrecondSuperRes):
            logger0.warning(
                f"Student-t distribution sampling is supposed to be used with "
                f"tEDMPrecondSuperRes model, but got {type(net_res)}."
            )
    elif isinstance(net_res, tEDMPrecondSuperRes):
        logger0.warning(
            f"tEDMPrecondSuperRes model is supposed to be used with student-t "
            f"distribution, but got {distribution}."
        )

    # Parse P_mean and P_std
    P_mean = getattr(cfg.generation, "P_mean", None)
    P_std = getattr(cfg.generation, "P_std", None)

    # Main generation definition
    def generate_fn():
        with nvtx.annotate("generate_fn", color="green"):
            diffusion_step_kwargs = {}
            if distribution is not None:
                diffusion_step_kwargs["distribution"] = distribution
            if student_t_nu is not None:
                diffusion_step_kwargs["nu"] = student_t_nu
            if P_mean is not None:
                diffusion_step_kwargs["P_mean"] = P_mean
            if P_std is not None:
                diffusion_step_kwargs["P_std"] = P_std

            # (1, C, H, W)
            img_lr = image_lr.to(memory_format=torch.channels_last)

            if net_reg:
                with nvtx.annotate("regression_model", color="yellow"):
                    image_reg = regression_step(
                        net=net_reg,
                        img_lr=img_lr,
                        latents_shape=(
                            sum(map(len, rank_batches)),
                            img_out_channels,
                            img_shape[0],
                            img_shape[1],
                        ),  # (batch_size, C, H, W)
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
                        img_shape=img_shape,
                        img_out_channels=img_out_channels,
                        rank_batches=rank_batches,
                        img_lr=img_lr.expand(
                            cfg.generation.seed_batch_size, -1, -1, -1
                        ).to(memory_format=torch.channels_last),
                        rank=dist.rank,
                        device=device,
                        mean_hr=mean_hr,
                        lead_time_label=lead_time_label,
                        **diffusion_step_kwargs,
                    )
            if cfg.generation.inference_mode == "regression":
                image_out = image_reg
            elif cfg.generation.inference_mode == "diffusion":
                image_out = image_res
            else:
                image_out = image_reg + image_res

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
        return

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

    torch_cuda_profiler = (
        torch.cuda.profiler.profile()
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    torch_nvtx_profiler = (
        torch.autograd.profiler.emit_nvtx()
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with torch_cuda_profiler:
        with torch_nvtx_profiler:
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

                if cfg.generation.perf.io_synchronous:
                    writer_executor = ThreadPoolExecutor(
                        max_workers=cfg.generation.perf.num_writer_workers
                    )
                    writer_threads = []

            # Create timer objects only if CUDA is available
            use_cuda_timing = torch.cuda.is_available()
            if use_cuda_timing:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
            else:
                # Dummy no-op functions for CPU case
                class DummyEvent:
                    def record(self):
                        pass

                    def synchronize(self):
                        pass

                    def elapsed_time(self, _):
                        return 0

                start = end = DummyEvent()

            times = dataset.time()
            for dataset_index, (image_tar, image_lr, *lead_time_label) in zip(
                sampler,
                iter(data_loader),
            ):
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
                    if cfg.generation.perf.io_synchronous:
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
                                dataset_index,
                            )
                        )
                    else:
                        save_images(
                            writer,
                            dataset,
                            list(times),
                            image_out.cpu(),
                            image_tar.cpu(),
                            image_lr.cpu(),
                            time_index,
                            dataset_index,
                        )
            end.record()
            end.synchronize()
            elapsed_time = (
                start.elapsed_time(end) / 1000.0 if use_cuda_timing else 0
            )  # Convert ms to s
            timed_steps = time_index + 1 - warmup_steps
            if dist.rank == 0 and use_cuda_timing:
                average_time_per_batch_element = elapsed_time / timed_steps / batch_size
                logger.info(
                    f"Total time to run {timed_steps} steps and {batch_size} members = {elapsed_time} s"
                )
                logger.info(
                    f"Average time per batch element = {average_time_per_batch_element} s"
                )

            # make sure all the workers are done writing
            if dist.rank == 0 and cfg.generation.perf.io_synchronous:
                for thread in list(writer_threads):
                    thread.result()
                    writer_threads.remove(thread)
                writer_executor.shutdown()

    if dist.rank == 0:
        f.close()
    logger0.info("Generation Completed.")


if __name__ == "__main__":
    main()
