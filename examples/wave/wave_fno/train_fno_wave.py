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

import hydra
from omegaconf import DictConfig
from math import ceil

from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler

from physicsnemo.models.fno import FNO
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, LaunchLogger

from wave_data import WaveDataLoader
from validator import WaveValidator


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def wave_trainer(cfg: DictConfig) -> None:
    """Training for the 2D wave equation benchmark problem.

    This training script demonstrates how to set up a data-driven model for a 2D
    wave equation using Fourier Neural Operators (FNO) and acts as a benchmark for
    hyperbolic PDE operator learning. Training data is generated on the fly via
    leapfrog finite-difference integration. The model learns to map an initial
    wavefield u(x, y, 0) to the solution u(x, y, T) at a specified target time.
    """
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="wave_fno")
    log.file_logging()
    LaunchLogger.initialize()  # PhysicsNeMo launch logger

    # define model, loss, optimiser, scheduler, data loader
    model = FNO(
        in_channels=cfg.arch.fno.in_channels,
        out_channels=cfg.arch.decoder.out_features,
        decoder_layers=cfg.arch.decoder.layers,
        decoder_layer_size=cfg.arch.decoder.layer_size,
        dimension=cfg.arch.fno.dimension,
        latent_channels=cfg.arch.fno.latent_channels,
        num_fno_layers=cfg.arch.fno.fno_layers,
        num_fno_modes=cfg.arch.fno.fno_modes,
        padding=cfg.arch.fno.padding,
    ).to(dist.device)
    loss_fun = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step
    )

    dataloader = WaveDataLoader(
        resolution=cfg.training.resolution,
        batch_size=cfg.training.batch_size,
        wave_speed=cfg.wave.speed,
        target_time=cfg.wave.target_time,
        nr_modes=cfg.wave.nr_modes,
        cfl=cfg.wave.cfl,
        device=dist.device,
    )
    validator = WaveValidator(loss_fun=MSELoss(reduction="mean"))

    ckpt_args = {
        "path": f"./checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_pseudo_epoch = load_checkpoint(device=dist.device, **ckpt_args)

    # calculate steps per pseudo epoch
    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    )
    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)
    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_pseudo_epoch,
        "epoch_alert_freq": 1,
    }
    if cfg.training.pseudo_epoch_sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased pseudo_epoch_sample_size to multiple of "
            f"batch size: {steps_per_pseudo_epoch * cfg.training.batch_size}"
        )
    if cfg.validation.sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased validation sample size to multiple of "
            f"batch size: {validation_iters * cfg.training.batch_size}"
        )

    # define forward passes for training and inference
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    def forward_train(invars, target):
        pred = model(invars)
        loss = loss_fun(pred, target)
        return loss

    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        return model(invars)

    if loaded_pseudo_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch + 1}.")

    for pseudo_epoch in range(
        max(1, loaded_pseudo_epoch + 1), cfg.training.max_pseudo_epochs + 1
    ):
        # Wrap epoch in launch logger for console / MLFlow logs
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            for _, batch in zip(range(steps_per_pseudo_epoch), dataloader):
                loss = forward_train(batch["initial"], batch["target"])
                logger.log_minibatch({"loss": loss.detach()})
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # save checkpoint
        if pseudo_epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

        # validation step
        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_loss = 0.0
                for _, batch in zip(range(validation_iters), dataloader):
                    val_loss = validator.compare(
                        batch["initial"],
                        batch["target"],
                        forward_eval(batch["initial"]),
                        pseudo_epoch,
                        logger,
                    )
                    total_loss += val_loss
                logger.log_epoch(
                    {"Validation error": total_loss / validation_iters}
                )

        # update learning rate
        if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
            scheduler.step()

    save_checkpoint(**ckpt_args, epoch=cfg.training.max_pseudo_epochs)
    log.success("Training completed *yay*")


if __name__ == "__main__":
    wave_trainer()
