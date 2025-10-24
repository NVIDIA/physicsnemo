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
# limitations under the License.anguage governing permissions and
# limitations under the License.

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os

os.environ["TORCHELASTIC_ENABLE_FILE_TIMER"] = "1"  # TODO is this needed?

import json
import re

import hydra
import torch
from omegaconf import DictConfig
from training_loop import training_loop
from physicsnemo.utils.diffusion.utils import EasyDict, construct_class_by_name

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

try:
    from apex.optimizers import FusedAdam

    apex_imported = True
except ImportError:
    apex_imported = False

from omegaconf import OmegaConf
import argparse


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """

    # Initialize distributed manager.
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize logger.
    logger = PythonLogger("main")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging()

    # TODO add mlflow/wandb logging

    # Initialize config dict.
    c = EasyDict()
    c.dataset = cfg.dataset
    if cfg.arch == "dfsr":
        print("Training diffusion model for fluid data super-resolution.")
        dataset_class_name = "dataset.KolmogorovFlowDataset"
    else:
        dataset_class_name = "dataset.ImageFolderDataset"
    c.dataset_kwargs = EasyDict(
        class_name=dataset_class_name,
        path=cfg.data,
        use_labels=cfg.cond,
        xflip=cfg.xflip,
        cache=cfg.cache,
    )
    c.data_loader_kwargs = EasyDict(
        pin_memory=True, num_workers=cfg.workers, prefetch_factor=2
    )
    c.network_kwargs = EasyDict()
    c.loss_kwargs = EasyDict()
    c.optimizer_kwargs = EasyDict(
        class_name="apex.optimizers.FusedAdam"
        if apex_imported and cfg.fused_adam
        else "torch.optim.Adam",
        lr=cfg.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
    )
    dataset_name = cfg.dataset

    # Validate dataset options.
    try:
        dataset_obj = construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = (
            dataset_obj.resolution
        )  # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj)  # be explicit about dataset size
        if cfg.cond and not dataset_obj.has_labels:
            raise ValueError("cond=True requires labels specified in dataset.json")
        del dataset_obj  # conserve memory
    except IOError as err:
        raise ValueError(f"data: {err}")

    # Network architecture.
    # if cfg.arch == 'ddpmpp-cwb-v2':
    #     c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')  #, attn_resolutions=[28]
    #     c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[1,2,2,4,4,8], attn_resolutions=[14])   #era5-cwb, larger run, 448x448

    # elif cfg.arch == 'ddpmpp-cwb-v1':
    #     c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')  #, attn_resolutions=[28]
    #     c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[1,2,2,4,4], attn_resolutions=[28])   #era5-cwb, 448x448

    # elif cfg.arch == 'ddpmpp-cwb-v0-regression':
    #     c.network_kwargs.update(model_type='SongUNet', embedding_type='zero', encoder_type='standard', decoder_type='standard')  #, attn_resolutions=[28]
    #     c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[1,2,2,2,2], attn_resolutions=[28])   #era5-cwb, 448x448

    # elif cfg.arch == 'ddpmpp-cwb-v0':
    #     c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')  #, attn_resolutions=[28]
    #     c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[1,2,2,2,2], attn_resolutions=[28])   #era5-cwb, 448x448

    # elif cfg.arch == 'ddpmpp-cifar':
    #     c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')  #, attn_resolutions=[28]
    #     c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])    #cifar-10, 32x32

    # elif cfg.arch == 'ncsnpp':
    #     c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
    #     c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])

    if cfg.arch == "ddpmpp":
        c.network_kwargs.update(
            model_type="SongUNet",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
        )
        c.network_kwargs.update(
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
        )
    elif cfg.arch == "ncsnpp":
        c.network_kwargs.update(
            model_type="SongUNet",
            embedding_type="fourier",
            encoder_type="residual",
            decoder_type="standard",
        )
        c.network_kwargs.update(
            channel_mult_noise=2,
            resample_filter=[1, 3, 3, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
        )
    elif cfg.arch == "dfsr":  # two model types for fluid data super-resolution
        c.network_kwargs.update(
            model_type="SongUNet",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
        )
        c.network_kwargs.update(
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=64,
            channel_mult=[1, 1, 1, 2],
        )
    else:
        assert cfg.arch == "adm"
        c.network_kwargs.update(
            model_type="DhariwalUNet", model_channels=192, channel_mult=[1, 2, 3, 4]
        )

    # Preconditioning & loss function.
    if cfg.precond == "vp":
        c.network_kwargs.class_name = "physicsnemo.models.diffusion.VPPrecond"
        c.loss_kwargs.class_name = "physicsnemo.metrics.diffusion.VPLoss"
    elif cfg.precond == "ve":
        c.network_kwargs.class_name = "physicsnemo.models.diffusion.VEPrecond"
        c.loss_kwargs.class_name = "physicsnemo.metrics.diffusion.VELoss"
    elif cfg.precond == "edm":
        c.network_kwargs.class_name = "physicsnemo.models.diffusion.EDMPrecond"
        c.loss_kwargs.class_name = "physicsnemo.metrics.diffusion.EDMLoss"
    # elif cfg.precond == 'unetregression':
    #     c.network_kwargs.class_name = 'training.networks.UNet'
    #     c.loss_kwargs.class_name = 'training.loss.RegressionLoss'
    # elif cfg.precond == 'mixture':
    #     c.network_kwargs.class_name = 'training.networks.EDMPrecond'
    #     c.loss_kwargs.class_name = 'training.loss.MixtureLoss'
    # elif cfg.precond == 'resloss':
    #     c.network_kwargs.class_name = 'training.networks.EDMPrecond'
    #     c.loss_kwargs.class_name = 'training.loss.ResLoss'
    elif cfg.precond == "dfsr":
        # Configure model for fluid data super-resolution
        c.network_kwargs.class_name = "physicsnemo.models.diffusion.VEPrecond_dfsr"
        c.loss_kwargs.class_name = "physicsnemo.metrics.diffusion.VELoss_dfsr"
    elif cfg.precond == "dfsr_cond":
        # Configure model for physics-informed conditional fluid data super-resolution
        c.network_kwargs.class_name = "physicsnemo.models.diffusion.VEPrecond_dfsr_cond"
        c.loss_kwargs.class_name = "physicsnemo.metrics.diffusion.VELoss_dfsr"

    # Network options.
    if cfg.cbase is not None:
        c.network_kwargs.model_channels = cfg.cbase
    if cfg.cres is not None:
        c.network_kwargs.channel_mult = cfg.cres
    if cfg.augment:
        raise NotImplementedError("Augmentation is not implemented")
    c.network_kwargs.update(dropout=cfg.dropout, use_fp16=cfg.fp16)

    # Training options.
    c.total_kimg = max(int(cfg.duration * 1000), 1)
    c.ema_halflife_kimg = int(cfg.ema * 1000)
    c.update(batch_size=cfg.batch, batch_gpu=cfg.batch_gpu)
    c.update(loss_scaling=cfg.ls, cudnn_benchmark=cfg.bench)
    c.update(kimg_per_tick=cfg.tick, snapshot_ticks=cfg.snap, state_dump_ticks=cfg.dump)

    # Random seed.
    if cfg.seed is not None:
        c.seed = cfg.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=dist.device)
        if dist.distributed:
            torch.distributed.broadcast(seed, src=0)  # TODO check if this fails
        c.seed = int(seed)

    # check if resume.txt exists
    resume_path = os.path.join(cfg.outdir, "resume.txt")
    if os.path.exists(resume_path):
        with open(resume_path, "r") as f:
            cfg.resume = f.read()
            f.close()

    logger0.info(f"cfg.resume: {cfg.resume}")

    # Transfer learning and resume.
    if cfg.transfer is not None:
        if cfg.resume is not None:
            raise ValueError("transfer and resume cannot be specified at the same time")
        c.resume_pkl = cfg.transfer
        c.ema_rampup_ratio = None
    elif cfg.resume is not None:  # TODO replace prints with PhysicsNeMo logger
        print("gets into elif cfg.resume is not None ...")
        match = re.fullmatch(r"training-state-(\d+).pt", os.path.basename(cfg.resume))
        print("match", match)
        print("match.group(1)", match.group(1))
        c.resume_pkl = os.path.join(
            os.path.dirname(cfg.resume), f"network-snapshot-{match.group(1)}.pkl"
        )
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = cfg.resume
        logger0.info(f"c.resume_pkl: {c.resume_pkl}")
        logger0.info(f"c.resume_kimg: {c.resume_kimg}")
        logger0.info(f"c.resume_state_dump: {c.resume_state_dump}")

    # Description string.
    cond_str = "cond" if c.dataset_kwargs.use_labels else "uncond"
    dtype_str = "fp16" if c.network_kwargs.use_fp16 else "fp32"
    desc = f"{dataset_name:s}-{cond_str:s}-{cfg.arch:s}-{cfg.precond:s}-gpus{dist.world_size:d}-batch{c.batch_size:d}-{dtype_str:s}"
    if cfg.desc is not None:
        desc += f"-{cfg.desc}"

    c.run_dir = cfg.outdir

    # # Weather data
    # c.data_type = cfg.data_type
    # c.data_config = cfg.data_config
    # c.task = cfg.task

    # Print options.  # TODO replace prints with PhysicsNeMo logger
    logger0.info("Training options:")
    logger0.info(json.dumps(c, indent=2))
    logger0.info(f"Output directory:        {c.run_dir}")
    logger0.info(f"Dataset path:            {c.dataset_kwargs.path}")
    logger0.info(f"Class-conditional:       {c.dataset_kwargs.use_labels}")
    logger0.info(f"Network architecture:    {cfg.arch}")
    logger0.info(f"Preconditioning & loss:  {cfg.precond}")
    logger0.info(f"Number of GPUs:          {dist.world_size}")
    logger0.info(f"Batch size:              {c.batch_size}")
    logger0.info(f"Mixed-precision:         {c.network_kwargs.use_fp16}")

    # Dry run?
    if cfg.dry_run:
        logger0.info("Dry run; exiting.")
        return

    # Create output directory.
    logger0.info("Creating output directory...")
    if dist.rank == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, "training_options.json"), "wt") as f:
            json.dump(c, f, indent=2)
        # utils.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop(**c, dist=dist, logger0=logger0)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
