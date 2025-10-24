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

import torch
import os
import hydra
import numpy as np
import matplotlib.pyplot as plt

from torch.nn.parallel import DistributedDataParallel
from omegaconf import DictConfig, OmegaConf

from physicsnemo.models.pangu import Pangu
from physicsnemo.datapipes.climate import ERA5HDF5Datapipe
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad

from physicsnemo.launch.logging import LaunchLogger, PythonLogger
from physicsnemo.launch.logging.mlflow import initialize_mlflow
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

try:
    from apex import optimizers
except:
    raise ImportError(
        "Pangu-Weather training requires apex package for optimizer."
        + "See https://github.com/nvidia/apex for install details."
    )


def loss_func(x, y):
    return torch.nn.functional.l1_loss(x, y)


@torch.no_grad()
def validation_step(
    eval_step, pangu_model, datapipe, surface_mask, channels=[0, 1], epoch=0
):
    loss_epoch = 0
    num_examples = 0  # Number of validation examples
    # Dealing with DDP wrapper
    if hasattr(pangu_model, "module"):
        pangu_model = pangu_model.module
    pangu_model.eval()
    for i, data in enumerate(datapipe):
        invar_surface = data[0]["invar"].detach()[:, :4, :, :]
        invar_upper_air = (
            data[0]["invar"]
            .detach()[:, 4:, :, :]
            .reshape(
                (
                    data[0]["invar"].shape[0],
                    5,
                    -1,
                    data[0]["invar"].shape[2],
                    data[0]["invar"].shape[3],
                )
            )
        )
        outvar_surface = data[0]["outvar"].cpu().detach()[:, :, :4, :, :]
        outvar_upper_air = (
            data[0]["outvar"]
            .cpu()
            .detach()[:, :, 4:, :, :]
            .reshape(
                (
                    data[0]["outvar"].shape[0],
                    data[0]["outvar"].shape[1],
                    5,
                    -1,
                    data[0]["outvar"].shape[3],
                    data[0]["outvar"].shape[4],
                )
            )
        )
        predvar_surface = torch.zeros_like(outvar_surface)
        predvar_upper_air = torch.zeros_like(outvar_upper_air)

        for t in range(outvar_surface.shape[1]):
            output_surface, output_upper_air = eval_step(
                pangu_model, invar_surface, surface_mask, invar_upper_air
            )
            invar_surface.copy_(output_surface)
            invar_upper_air.copy_(output_upper_air)
            predvar_surface[:, t] = output_surface.detach().cpu()
            predvar_upper_air[:, t] = output_upper_air.detach().cpu()

        num_elements_surface = torch.prod(torch.Tensor(list(predvar_surface.shape[1:])))
        num_elements_upper_air = torch.prod(
            torch.Tensor(list(predvar_upper_air.shape[1:]))
        )
        loss_epoch += (
            torch.sum(torch.pow(predvar_surface - outvar_surface, 2))
            + torch.sum(torch.pow(predvar_upper_air - outvar_upper_air, 2))
        ) / (num_elements_surface + num_elements_upper_air)

        num_examples += predvar_surface.shape[0]

    pangu_model.train()
    return loss_epoch / num_examples


@hydra.main(version_base="1.2", config_path="conf", config_name="config_lite")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers
    initialize_mlflow(
        experiment_name=cfg.experiment_name,
        experiment_desc=cfg.experiment_desc,
        run_name="Pangu-lite-trainng",
        run_desc=cfg.experiment_desc,
        user_name="PhysicsNeMo User",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)  # PhysicsNeMo launch logger
    logger = PythonLogger("main")  # General python logger

    number_channels_pangu = 4 + 5 * 13
    datapipe = ERA5HDF5Datapipe(
        data_dir=cfg.train.data_dir,
        stats_dir=cfg.train.stats_dir,
        channels=[i for i in range(number_channels_pangu)],
        num_samples_per_year=cfg.train.num_samples_per_year,
        batch_size=cfg.train.batch_size,
        patch_size=OmegaConf.to_object(cfg.train.patch_size),
        num_workers=cfg.train.num_workers,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )
    logger.success(f"Loaded datapipe of size {len(datapipe)}")

    mask_dir = cfg.mask_dir
    if cfg.get("mask_dtype", "float32") == "float32":
        mask_dtype = np.float32
    elif cfg.get("mask_dtype", "float32") == "float16":
        mask_dtype = np.float16
    else:
        mask_dtype = np.float32
    land_mask = torch.from_numpy(
        np.load(os.path.join(mask_dir, "land_mask.npy")).astype(mask_dtype)
    )
    soil_type = torch.from_numpy(
        np.load(os.path.join(mask_dir, "soil_type.npy")).astype(mask_dtype)
    )
    topography = torch.from_numpy(
        np.load(os.path.join(mask_dir, "topography.npy")).astype(mask_dtype)
    )
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to(
        dist.device
    )
    logger.success(f"Loaded suface constant mask from {mask_dir}")

    if dist.rank == 0:
        logger.file_logging()
        validation_datapipe = ERA5HDF5Datapipe(
            data_dir=cfg.val.data_dir,
            stats_dir=cfg.val.stats_dir,
            channels=[i for i in range(number_channels_pangu)],
            num_steps=1,
            num_samples_per_year=cfg.val.num_samples_per_year,
            batch_size=cfg.val.batch_size,
            patch_size=OmegaConf.to_object(cfg.val.patch_size),
            device=dist.device,
            num_workers=cfg.val.num_workers,
            shuffle=False,
        )
        logger.success(f"Loaded validaton datapipe of size {len(validation_datapipe)}")

    pangu_model = Pangu(
        img_size=OmegaConf.to_object(cfg.pangu.img_size),
        patch_size=OmegaConf.to_object(cfg.pangu.patch_size),
        embed_dim=cfg.pangu.embed_dim,
        num_heads=OmegaConf.to_object(cfg.pangu.num_heads),
        window_size=OmegaConf.to_object(cfg.pangu.window_size),
    ).to(dist.device)

    # Distributed learning
    if dist.world_size > 1:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            pangu_model = DistributedDataParallel(
                pangu_model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # Initialize optimizer and scheduler
    optimizer = optimizers.FusedAdam(
        pangu_model.parameters(), betas=(0.9, 0.999), lr=0.0005, weight_decay=0.000003
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Attempt to load latest checkpoint if one exists
    loaded_epoch = load_checkpoint(
        "./checkpoints",
        models=pangu_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=dist.device,
    )

    @StaticCaptureEvaluateNoGrad(model=pangu_model, logger=logger, use_graphs=False)
    def eval_step_forward(my_model, invar_surface, surface_mask, invar_upper_air):
        invar = my_model.prepare_input(invar_surface, surface_mask, invar_upper_air)
        return my_model(invar)

    @StaticCaptureTraining(model=pangu_model, optim=optimizer, logger=logger)
    def train_step_forward(
        my_model,
        invar_surface,
        surface_mask,
        invar_upper_air,
        outvar_surface,
        outvar_upper_air,
    ):
        # Multi-step prediction
        loss = 0
        # Multi-step not supported
        for t in range(outvar_surface.shape[1]):
            invar = my_model.prepare_input(invar_surface, surface_mask, invar_upper_air)
            outpred_surface, outpred_upper_air = my_model(invar)
            invar_surface = outpred_surface
            invar_upper_air = outpred_upper_air
            loss += loss_func(outpred_surface, outvar_surface[:, t]) * 0.25 + loss_func(
                outpred_upper_air, outvar_upper_air[:, t]
            )
        return loss

    # Main training loop
    max_epoch = cfg.max_epoch
    for epoch in range(max(1, loaded_epoch + 1), max_epoch + 1):
        # Wrap epoch in launch logger for console / WandB logs
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(datapipe), epoch_alert_freq=10
        ) as log:
            # === Training step ===
            for j, data in enumerate(datapipe):
                invar_surface = data[0]["invar"][:, :4, :, :]
                invar_upper_air = data[0]["invar"][:, 4:, :, :].reshape(
                    (
                        data[0]["invar"].shape[0],
                        5,
                        -1,
                        data[0]["invar"].shape[2],
                        data[0]["invar"].shape[3],
                    )
                )
                outvar_surface = data[0]["outvar"][:, :, :4, :, :]
                outvar_upper_air = data[0]["outvar"][:, :, 4:, :, :].reshape(
                    (
                        data[0]["outvar"].shape[0],
                        data[0]["outvar"].shape[1],
                        5,
                        -1,
                        data[0]["outvar"].shape[3],
                        data[0]["outvar"].shape[4],
                    )
                )
                loss = train_step_forward(
                    pangu_model,
                    invar_surface,
                    surface_mask,
                    invar_upper_air,
                    outvar_surface,
                    outvar_upper_air,
                )

                log.log_minibatch({"loss": loss.detach()})
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        if dist.rank == 0:
            # Wrap validation in launch logger for console / WandB logs
            with LaunchLogger("valid", epoch=epoch) as log:
                # === Validation step ===
                error = validation_step(
                    eval_step_forward,
                    pangu_model,
                    validation_datapipe,
                    surface_mask,
                    epoch=epoch,
                )
                log.log_epoch({"Validation error": error})

        if dist.world_size > 1:
            torch.distributed.barrier()

        scheduler.step()

        if (epoch % 5 == 0 or epoch == 1) and dist.rank == 0:
            # Use PhysicsNeMo Launch checkpoint
            save_checkpoint(
                "./checkpoints",
                models=pangu_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )

    if dist.rank == 0:
        logger.info("Finished training!")


if __name__ == "__main__":
    main()
