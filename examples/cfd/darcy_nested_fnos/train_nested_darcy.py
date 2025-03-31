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
import glob
import hydra
from typing import Tuple
from omegaconf import DictConfig
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.models.fno import FNO
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
    LaunchLogger,
)
from physicsnemo.launch.logging.mlflow import initialize_mlflow
from utils import NestedDarcyDataset, GridValidator


def InitializeLoggers(cfg: DictConfig) -> Tuple[DistributedManager, PythonLogger]:
    """Class containing most important objects

    In this class the infrastructure for training is set.

    Parameters
    ----------
    cfg : DictConfig
        config file parameters

    Returns
    -------
    Tuple[DistributedManager, PythonLogger]
    """
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere
    logger = PythonLogger(name="darcy_nested_fno")

    assert hasattr(cfg, "model"), logger.error(
        f"define which model to train: $ python {__file__.split(os.sep)[-1]} +model=<model_name>"
    )
    logger.info(f"training model {cfg.model}")

    # initialize monitoring
    initialize_mlflow(
        experiment_name=f"Nested FNO, model: {cfg.model}",
        experiment_desc=f"training model {cfg.model} for nested FNOs",
        run_name=f"Nested FNO training, model: {cfg.model}",
        run_desc=f"training model {cfg.model} for nested FNOs",
        user_name="Gretchen Ross",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)  # PhysicsNeMo launch logger

    return dist, RankZeroLoggingWrapper(logger, dist)


class SetUpInfrastructure:
    """Class containing most important objects

    In this class the infrastructure for training is set.

    Parameters
    ----------
    cfg : DictConfig
        config file parameters
    dist : DistributedManager
        persistent class instance for storing parallel environment information
    logger : PythonLogger
        logger for command line output
    """

    def __init__(
        self, cfg: DictConfig, dist: DistributedManager, logger: PythonLogger
    ) -> None:
        # define model, loss, optimiser, scheduler, data loader
        level = int(cfg.model[-1])
        model_cfg = cfg.arch[cfg.model]
        loss_fun = MSELoss(reduction="mean")
        norm = {
            "permeability": (
                cfg.normaliser.permeability.mean,
                cfg.normaliser.permeability.std,
            ),
            "darcy": (cfg.normaliser.darcy.mean, cfg.normaliser.darcy.std),
        }

        self.training_set = NestedDarcyDataset(
            mode="train",
            data_path=cfg.training.training_set,
            model_name=cfg.model,
            norm=norm,
            log=logger,
        )
        self.valid_set = NestedDarcyDataset(
            mode="train",
            data_path=cfg.validation.validation_set,
            model_name=cfg.model,
            norm=norm,
            log=logger,
        )

        logger.log(
            f"Training set contains {len(self.training_set)} samples, "
            + f"validation set contains {len(self.valid_set)} samples."
        )

        train_sampler = DistributedSampler(
            self.training_set,
            num_replicas=dist.world_size,
            rank=dist.local_rank,
            shuffle=True,
            drop_last=False,
        )

        valid_sampler = DistributedSampler(
            self.valid_set,
            num_replicas=dist.world_size,
            rank=dist.local_rank,
            shuffle=True,
            drop_last=False,
        )

        self.train_loader = DataLoader(
            self.training_set,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            sampler=train_sampler,
        )
        self.valid_loader = DataLoader(
            self.valid_set,
            batch_size=cfg.validation.batch_size,
            shuffle=False,
            sampler=valid_sampler,
        )
        self.validator = GridValidator(loss_fun=loss_fun, norm=norm)
        self.model = FNO(
            in_channels=model_cfg.fno.in_channels,
            out_channels=model_cfg.decoder.out_features,
            decoder_layers=model_cfg.decoder.layers,
            decoder_layer_size=model_cfg.decoder.layer_size,
            dimension=model_cfg.fno.dimension,
            latent_channels=model_cfg.fno.latent_channels,
            num_fno_layers=model_cfg.fno.fno_layers,
            num_fno_modes=model_cfg.fno.fno_modes,
            padding=model_cfg.fno.padding,
        ).to(dist.device)

        # distributed data parallel for multi-node training
        if dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )

        self.optimizer = Adam(self.model.parameters(), lr=cfg.scheduler.initial_lr)
        self.scheduler = lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step
        )
        self.log_args = {
            "name_space": "train",
            "num_mini_batch": len(self.train_loader),
            "epoch_alert_freq": 1,
        }
        self.ckpt_args = {
            "path": f"./checkpoints/all/{cfg.model}",
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "models": self.model,
        }
        self.bst_ckpt_args = {
            "path": f"./checkpoints/best/{cfg.model}",
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "models": self.model,
        }

        # define forward for training and inference
        @StaticCaptureTraining(
            model=self.model,
            optim=self.optimizer,
            logger=logger,
            use_amp=False,
            use_graphs=False,
        )
        def _forward_train(invars, target):
            pred = self.model(invars)
            loss = loss_fun(pred, target)
            return loss

        @StaticCaptureEvaluateNoGrad(
            model=self.model, logger=logger, use_amp=False, use_graphs=False
        )
        def _forward_eval(invars):
            return self.model(invars)

        self.forward_train = _forward_train
        self.forward_eval = _forward_eval


def TrainModel(cfg: DictConfig, base: SetUpInfrastructure, loaded_epoch: int) -> None:
    """Training Loop

    Parameters
    ----------
    cfg : DictConfig
        config file parameters
    base : SetUpInfrastructure
        important objects
    loaded_epoch : int
        epoch from which training is restarted, ==0 if starting from scratch
    """

    min_valid_loss = 9.0e9
    for epoch in range(max(1, loaded_epoch + 1), cfg.training.max_epochs + 1):
        # Wrap epoch in launch logger for console / MLFlow logs
        with LaunchLogger(**base.log_args, epoch=epoch) as log:
            for batch in base.train_loader:
                loss = base.forward_train(batch["permeability"], batch["darcy"])
                log.log_minibatch({"loss": loss.detach()})
            log.log_epoch({"Learning Rate": base.optimizer.param_groups[0]["lr"]})

        # validation
        if (
            epoch % cfg.validation.validation_epochs == 0
            or epoch % cfg.training.rec_results_freq == 0
            or epoch == cfg.training.max_epochs
        ):
            with LaunchLogger("valid", epoch=epoch) as log:
                total_loss = 0.0
                for batch in base.valid_loader:
                    loss = base.validator.compare(
                        batch["permeability"],
                        batch["darcy"],
                        base.forward_eval(batch["permeability"]),
                        epoch,
                        log,
                    )
                    total_loss += loss * batch["darcy"].shape[0] / len(base.valid_set)
                log.log_epoch({"Validation error": total_loss})

        # save checkpoint
        if (
            epoch % cfg.training.rec_results_freq == 0
            or epoch == cfg.training.max_epochs
        ):
            save_checkpoint(**base.ckpt_args, epoch=epoch)
            if (
                total_loss < min_valid_loss
            ):  # save seperately if best checkpoint thus far
                min_valid_loss = total_loss
                for ckpt in glob.glob(base.bst_ckpt_args["path"] + "/*.pt"):
                    os.remove(ckpt)
                save_checkpoint(**base.bst_ckpt_args, epoch=epoch)

        # update learning rate
        if epoch % cfg.scheduler.decay_epochs == 0:
            base.scheduler.step()


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def nested_darcy_trainer(cfg: DictConfig) -> None:
    """Training for the 2D nested Darcy flow problem.

    This training script demonstrates how to set up a data-driven model for a nested 2D Darcy flow
    using nested Fourier Neural Operators (nFNO, https://arxiv.org/abs/2210.17051). nFNOs are
    basically a concatenation of individual FNO models. Individual FNOs can be trained independently
    and in any order. The order only gets important for fine tuning (tba) and inference.
    """

    # initialize loggers
    dist, logger = InitializeLoggers(cfg)

    # set up infrastructure
    base = SetUpInfrastructure(cfg, dist, logger)

    # catch restart in case checkpoint exists
    loaded_epoch = load_checkpoint(**base.ckpt_args, device=dist.device)
    if loaded_epoch == 0:
        logger.success("Training started...")
    else:
        logger.warning(f"Resuming training from epoch {loaded_epoch+1}.")

    # train model
    TrainModel(cfg, base, loaded_epoch)
    logger.success("Training completed *yay*")


if __name__ == "__main__":
    nested_darcy_trainer()
