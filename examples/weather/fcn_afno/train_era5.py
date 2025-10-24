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
import hydra
from hydra.utils import to_absolute_path
import wandb
import matplotlib.pyplot as plt

from torch.nn.parallel import DistributedDataParallel
from omegaconf import DictConfig

from physicsnemo.models.afno import AFNO
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
        "FCN training requires apex package for optimizer."
        + "See https://github.com/nvidia/apex for install details."
    )


def loss_func(x, y, p=2.0):
    yv = y.reshape(x.size()[0], -1)
    xv = x.reshape(x.size()[0], -1)
    diff_norms = torch.linalg.norm(xv - yv, ord=p, dim=1)
    y_norms = torch.linalg.norm(yv, ord=p, dim=1)

    return torch.mean(diff_norms / y_norms)


@torch.no_grad()
def validation_step(
    eval_step,
    fcn_model,
    datapipe,
    channels=[
        0,
    ],
    epoch=0,
):
    loss_epoch = 0
    num_examples = 0  # Number of validation examples
    # Dealing with DDP wrapper
    if hasattr(fcn_model, "module"):
        fcn_model = fcn_model.module
    fcn_model.eval()
    for i, data in enumerate(datapipe):
        invar = data[0]["invar"].detach()
        outvar = data[0]["outvar"].cpu().detach()
        predvar = torch.zeros_like(outvar)

        for t in range(outvar.shape[1]):
            output = eval_step(fcn_model, invar)
            invar.copy_(output)
            predvar[:, t] = output.detach().cpu()

        num_elements = torch.prod(torch.Tensor(list(predvar.shape[1:])))
        loss_epoch += torch.sum(torch.pow(predvar - outvar, 2)) / num_elements
        num_examples += predvar.shape[0]

        # Plotting
        if i == 0:
            predvar = predvar.numpy()
            outvar = outvar.numpy()
            for chan in channels:
                plt.close("all")
                fig, ax = plt.subplots(
                    3, predvar.shape[1], figsize=(15, predvar.shape[0] * 5)
                )
                for t in range(outvar.shape[1]):
                    ax[0, t].imshow(predvar[0, t, chan])
                    ax[1, t].imshow(outvar[0, t, chan])
                    ax[2, t].imshow(predvar[0, t, chan] - outvar[0, t, chan])

                fig.savefig(f"era5_validation_channel{chan}_epoch{epoch}.png")

    fcn_model.train()
    return loss_epoch / num_examples


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers
    # initialize_wandb(
    #     project="PhysicsNeMo-Launch-Dev",
    #     entity="PhysicsNeMo",
    #     name="FourCastNet-Training",
    #     group="FCN-DDP-Group",
    # )
    initialize_mlflow(
        experiment_name="PhysicsNeMo-Launch-Dev",
        experiment_desc="PhysicsNeMo launch development",
        run_name="FCN-Training",
        run_desc="FCN ERA5 Training",
        user_name="PhysicsNeMo User",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=cfg.use_mlflow)  # PhysicsNeMo launch logger
    logger = PythonLogger("main")  # General python logger

    datapipe = ERA5HDF5Datapipe(
        data_dir=to_absolute_path(cfg.train_dir),
        stats_dir=to_absolute_path(cfg.stats_dir),
        channels=cfg.channels,
        num_steps=cfg.num_steps_train,
        num_samples_per_year=cfg.num_samples_per_year_train,
        batch_size=cfg.batch_size_train,
        patch_size=(8, 8),
        num_workers=cfg.num_workers_train,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )
    logger.success(f"Loaded datapipe of size {len(datapipe)}")
    if dist.rank == 0:
        logger.file_logging()
        validation_datapipe = ERA5HDF5Datapipe(
            data_dir=to_absolute_path(cfg.validation_dir),
            stats_dir=to_absolute_path(cfg.stats_dir),
            channels=cfg.channels,
            num_steps=cfg.num_steps_validation,
            num_samples_per_year=cfg.num_samples_per_year_validation,
            batch_size=cfg.batch_size_validation,
            patch_size=(8, 8),
            device=dist.device,
            num_workers=cfg.num_workers_validation,
            shuffle=False,
        )
        logger.success(f"Loaded validation datapipe of size {len(validation_datapipe)}")

    fcn_model = AFNO(
        inp_shape=[720, 1440],
        in_channels=len(cfg.channels),
        out_channels=len(cfg.channels),
        patch_size=[8, 8],
        embed_dim=768,
        depth=12,
        num_blocks=8,
    ).to(dist.device)

    if dist.rank == 0 and wandb.run is not None:
        wandb.watch(
            fcn_model, log="all", log_freq=1000, log_graph=(True)
        )  # currently does not work with scripted modules. This will be fixed in the next release of W&B SDK.
    # Distributed learning
    if dist.world_size > 1:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            fcn_model = DistributedDataParallel(
                fcn_model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # Initialize optimizer and scheduler
    optimizer = optimizers.FusedAdam(
        fcn_model.parameters(), betas=(0.9, 0.999), lr=0.0005, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    # Attempt to load latest checkpoint if one exists
    loaded_epoch = load_checkpoint(
        to_absolute_path(cfg.ckpt_path),
        models=fcn_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=dist.device,
    )

    @StaticCaptureEvaluateNoGrad(model=fcn_model, logger=logger, use_graphs=False)
    def eval_step_forward(my_model, invar):
        return my_model(invar)

    @StaticCaptureTraining(model=fcn_model, optim=optimizer, logger=logger)
    def train_step_forward(my_model, invar, outvar):
        # Multi-step prediction
        loss = 0
        for t in range(outvar.shape[1]):
            outpred = my_model(invar)
            invar = outpred
            loss += loss_func(outpred, outvar[:, t])
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
                invar = data[0]["invar"]
                outvar = data[0]["outvar"]
                loss = train_step_forward(fcn_model, invar, outvar)

                log.log_minibatch({"loss": loss.detach()})
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        if dist.rank == 0:
            # Wrap validation in launch logger for console / WandB logs
            with LaunchLogger("valid", epoch=epoch) as log:
                # === Validation step ===
                error = validation_step(
                    eval_step_forward, fcn_model, validation_datapipe, epoch=epoch
                )
                log.log_epoch({"Validation error": error})

        if dist.world_size > 1:
            torch.distributed.barrier()

        scheduler.step()

        if (epoch % 5 == 0 or epoch == 1) and dist.rank == 0:
            # Use PhysicsNeMo Launch checkpoint
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=fcn_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )

    if dist.rank == 0:
        logger.info("Finished training!")


if __name__ == "__main__":
    main()
