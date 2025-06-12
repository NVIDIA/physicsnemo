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
from omegaconf import DictConfig
from math import ceil

from torch.nn import MSELoss
from utils.testloss import TestLoss
from torch.optim import Adam, lr_scheduler, AdamW

from physicsnemo.models.transolver import Transolver
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, LaunchLogger
from physicsnemo.launch.logging.mlflow import initialize_mlflow

from darcy_datapipe_fix import Darcy2D_fix
from validator_fix import GridValidator


class UnitTransformer:
    """Unit transformer class for normalizing and denormalizing data."""

    def __init__(self, X):
        self.mean = X.mean(dim=(0, 1), keepdim=True)
        self.std = X.std(dim=(0, 1), keepdim=True) + 1e-8

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def encode(self, x):
        x = (x - self.mean) / (self.std)
        return x

    def decode(self, x):
        return x * self.std + self.mean

    def transform(self, X, inverse=True, component="all"):
        if component == "all" or "all-reduce":
            if inverse:
                orig_shape = X.shape
                return (X * (self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X - self.mean) / self.std
        else:
            if inverse:
                orig_shape = X.shape
                return (
                    X * (self.std[:, component] - 1e-8) + self.mean[:, component]
                ).view(orig_shape)
            else:
                return (X - self.mean[:, component]) / self.std[:, component]


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


@hydra.main(version_base="1.3", config_path=".", config_name="config_fix.yaml")
def darcy_trainer(cfg: DictConfig) -> None:
    """Training for the 2D Darcy flow benchmark problem."""
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="darcy_transolver")
    log.file_logging()
    initialize_mlflow(
        experiment_name=f"Darcy_Transolver",
        experiment_desc=f"training a Transformer-based PDE solver for the Darcy problem",
        run_name=f"Darcy Transolver training",
        run_desc=f"training Transolver for Darcy",
        user_name="Haixu Wu, Huakun Luo, Haowen Wang",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)  # PhysicsNeMo launch logger

    # define model, loss, optimiser, scheduler, data loader
    model = Transolver(
        space_dim=cfg.model.space_dim,
        n_layers=cfg.model.n_layers,
        n_hidden=cfg.model.n_hidden,
        dropout=cfg.model.dropout,
        n_head=cfg.model.n_head,
        Time_Input=cfg.model.Time_Input,
        act=cfg.model.act,
        mlp_ratio=cfg.model.mlp_ratio,
        fun_dim=cfg.model.fun_dim,
        out_dim=cfg.model.out_dim,
        slice_num=cfg.model.slice_num,
        ref=cfg.model.ref,
        unified_pos=cfg.model.unified_pos,
        H=cfg.training.resolution,
        W=cfg.training.resolution,
    ).to(dist.device)
    count_parameters(model)
    loss_fun = TestLoss(size_average=False)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.scheduler.initial_lr,
        weight_decay=cfg.scheduler.weight_decay,
    )
    # scheduler = lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step
    # )

    norm_vars = cfg.normaliser
    normaliser = {
        "permeability": (norm_vars.permeability.mean, norm_vars.permeability.std_dev),
        "darcy": (norm_vars.darcy.mean, norm_vars.darcy.std_dev),
    }
    # train_dataloader = Darcy2D_fix(
    #     resolution=cfg.training.resolution,
    #     batch_size=cfg.training.batch_size,
    #     normaliser=normaliser,
    #     train_path="/data/fno/piececonst_r421_N1024_smooth1.mat",
    #     is_test=False,
    # )
    train_dataloader = Darcy2D_fix(
        resolution=cfg.training.resolution,
        batch_size=cfg.training.batch_size,
        normaliser=normaliser,
        train_path="/data/fno/piececonst_r421_N1024_smooth1.mat",
        is_test=False,
    )
    # calculate steps per pseudo epoch
    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.scheduler.initial_lr,
        steps_per_epoch=steps_per_pseudo_epoch,
        epochs=cfg.training.max_pseudo_epochs,
    )

    x_normalizer, y_normalizer = train_dataloader.__get_normalizer__()

    test_dataloader = Darcy2D_fix(
        resolution=cfg.training.resolution,
        batch_size=cfg.training.batch_size,
        normaliser=normaliser,
        train_path="/data/fno/piececonst_r421_N1024_smooth2.mat",
        is_test=True,
        x_normalizer=x_normalizer,
    )

    validator = GridValidator(loss_fun=TestLoss(size_average=False), norm=y_normalizer)

    ckpt_args = {
        "path": f"./checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_pseudo_epoch = load_checkpoint(device=dist.device, **ckpt_args)

    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)
    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_pseudo_epoch,
        "epoch_alert_freq": 1,
    }
    if cfg.training.pseudo_epoch_sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased pseudo_epoch_sample_size to multiple of \
                      batch size: {steps_per_pseudo_epoch*cfg.training.batch_size}"
        )
    if cfg.validation.sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased validation sample size to multiple of \
                      batch size: {validation_iters*cfg.training.batch_size}"
        )

    # define forward passes for training and inference
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    def forward_train(pos, x, y, y_normalizer):
        pred = model(pos, fx=x.unsqueeze(-1)).squeeze(-1)
        pred = y_normalizer.decode(pred)
        loss = loss_fun(pred, y)
        return loss

    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(pos, x, y, y_normalizer):
        pred = model(pos, fx=x.unsqueeze(-1)).squeeze(-1)
        return y_normalizer.decode(pred)

    if loaded_pseudo_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch+1}.")

    for pseudo_epoch in range(
        max(1, loaded_pseudo_epoch + 1), cfg.training.max_pseudo_epochs + 1
    ):
        # Wrap epoch in launch logger for console / MLFlow logs
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            for _, batch in zip(range(steps_per_pseudo_epoch), train_dataloader):
                loss = forward_train(*batch, y_normalizer)
                logger.log_minibatch({"loss": loss.detach() / cfg.training.batch_size})
                scheduler.step()
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # save checkpoint
        if pseudo_epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

        # validation step
        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_loss = 0.0
                for _, batch in zip(range(validation_iters), test_dataloader):
                    val_loss = validator.compare(
                        batch[2],
                        forward_eval(*batch, y_normalizer),
                        pseudo_epoch,
                        logger,
                    )
                    total_loss += val_loss
                logger.log_epoch(
                    {
                        "Validation error": total_loss
                        / (validation_iters * cfg.training.batch_size)
                    }
                )

        # update learning rate
        # if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:

    save_checkpoint(**ckpt_args, epoch=cfg.training.max_pseudo_epochs)
    log.success("Training completed *yay*")


if __name__ == "__main__":
    darcy_trainer()
