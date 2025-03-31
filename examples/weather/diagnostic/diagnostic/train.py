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

from typing import Callable, Iterable, Sequence, Type, Union
import warnings

import torch
from torch import Tensor

try:
    from apex.optimizers import FusedAdam
except ImportError:
    warnings.warn("Apex is not installed, defaulting to PyTorch optimizers.")

from physicsnemo import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import LaunchLogger, PythonLogger
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad


class Trainer:
    """Training loop for diagnostic models."""

    def __init__(
        self,
        model: Module,
        dist_manager: DistributedManager,
        loss: Callable,
        train_datapipe: Sequence,
        valid_datapipe: Sequence,
        input_output_from_batch_data: Union[Callable, None] = None,
        optimizer: Union[Type[torch.optim.Optimizer], None] = None,
        optimizer_params: Union[dict, None] = None,
        scheduler: Union[Type[torch.optim.lr_scheduler.LRScheduler], None] = None,
        scheduler_params: Union[dict, None] = None,
        max_epoch: int = 1,
        load_epoch: Union[int, str, None] = None,
        checkpoint_every: int = 1,
        checkpoint_dir: Union[str, None] = None,
        validation_callbacks: Iterable[Callable] = (),
    ):
        self.model = model
        self.dist_manager = dist_manager
        self.loss = loss
        self.train_datapipe = train_datapipe
        self.valid_datapipe = valid_datapipe
        self.max_epoch = max_epoch
        if input_output_from_batch_data is None:
            input_output_from_batch_data = lambda x: x
        self.input_output_from_batch_data = input_output_from_batch_data
        self.optimizer = self._setup_optimizer(
            opt_cls=optimizer, opt_params=optimizer_params
        )
        self.lr_scheduler = self._setup_lr_scheduler(
            scheduler_cls=scheduler, scheduler_params=scheduler_params
        )
        self.validation_callbacks = list(validation_callbacks)
        self.device = self.dist_manager.device
        self.logger = PythonLogger()

        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir
        self.epoch = 1
        if load_epoch is not None:
            epoch = None if load_epoch == "latest" else load_epoch
            self.load_checkpoint(epoch=epoch)

        # wrap capture here instead of using decorator so it'll still be wrapped if
        # overridden by a subclass
        self.train_step_forward = StaticCaptureTraining(
            model=self.model,
            optim=self.optimizer,
            logger=self.logger,
            use_graphs=False,  # for some reason use_graphs=True causes a crash
        )(self.train_step_forward)

        self.eval_step = StaticCaptureEvaluateNoGrad(
            model=self.model, logger=self.logger, use_graphs=False
        )(self.eval_step)

    def eval_step(self, invar: Tensor) -> Tensor:
        """Perform one step of model evaluation."""
        return self.model(invar)

    def train_step_forward(self, invar: Tensor, outvar_true: Tensor) -> Tensor:
        """Train model on one batch."""
        outvar_pred = self.model(invar)
        return self.loss(outvar_pred, outvar_true)

    def fit(self):
        """Main function for training loop."""
        for self.epoch in range(self.epoch, self.max_epoch + 1):
            self.train_on_epoch()

        if self.dist_manager.rank == 0:
            self.logger.info("Finished training!")

    def train_on_epoch(self):
        """Train for one epoch."""
        with LaunchLogger(
            "train",
            epoch=self.epoch,
            num_mini_batch=len(self.train_datapipe),
            epoch_alert_freq=10,
        ) as log:
            for batch in self.train_datapipe:
                loss = self.train_step_forward(
                    *self.input_output_from_batch_data(batch)
                )
                log.log_minibatch({"loss": loss.detach()})

            log.log_epoch({"Learning Rate": self.optimizer.param_groups[0]["lr"]})

        # Validation
        if self.dist_manager.rank == 0:
            with LaunchLogger("valid", epoch=self.epoch) as log:
                error = self.validate_on_epoch()
                log.log_epoch({"Validation error": error})

        if self.dist_manager.world_size > 1:
            torch.distributed.barrier()

        self.lr_scheduler.step()

        checkpoint_epoch = (self.checkpoint_dir is not None) and (
            (self.epoch % self.checkpoint_every == 0) or (self.epoch == self.max_epoch)
        )
        if checkpoint_epoch and self.dist_manager.rank == 0:
            # Save PhysicsNeMo Launch checkpoint
            self.save_checkpoint()

    @torch.no_grad()
    def validate_on_epoch(self) -> Tensor:
        """Return average loss over one validation epoch."""
        loss_epoch = 0
        num_examples = 0  # Number of validation examples
        # Dealing with DDP wrapper
        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model

        try:
            model.eval()
            for (i, batch) in enumerate(self.valid_datapipe):
                (invar, outvar_true) = self.input_output_from_batch_data(batch)
                invar = invar.detach()
                outvar_true = outvar_true.detach()
                outvar_pred = self.eval_step(invar)

                loss_epoch += self.loss(outvar_pred, outvar_true)
                num_examples += 1

                for callback in self.validation_callbacks:
                    callback(outvar_true, outvar_pred, epoch=self.epoch, batch_idx=i)
        finally:  # restore train state even if exception occurs
            model.train()
        return loss_epoch / num_examples

    def _setup_optimizer(self, opt_cls=None, opt_params=None):
        """Initialize optimizer."""
        opt_kwargs = {"lr": 0.0005}
        if opt_params is not None:
            opt_kwargs.update(opt_params)

        if opt_cls is None:
            try:
                opt_cls = FusedAdam
            except NameError:  # in case we don't have apex
                opt_cls = torch.optim.AdamW

        return opt_cls(self.model.parameters(), **opt_kwargs)

    def _setup_lr_scheduler(self, scheduler_cls=None, scheduler_params=None):
        """Initialize learning rate scheduler."""
        scheduler_kwargs = {}
        if scheduler_cls is None:
            scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR
            scheduler_kwargs["T_max"] = self.max_epoch
        if scheduler_params is not None:
            scheduler_kwargs.update(scheduler_params)

        return scheduler_cls(self.optimizer, **scheduler_kwargs)

    def load_checkpoint(self, epoch: Union[int, None] = None) -> int:
        """Load training state from checkpoint.

        Parameters
        ----------
        epoch: int or None, optional
            The epoch for which the state is loaded. If None, will load the
            latest epoch.
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be set in order to load checkpoints.")
        self.epoch = load_checkpoint(
            self.checkpoint_dir,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            device=self.device,
            epoch=epoch,
        )
        return self.epoch

    def save_checkpoint(self):
        """Save training state from checkpoint."""
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be set in order to save checkpoints.")
        save_checkpoint(
            self.checkpoint_dir,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            epoch=self.epoch,
        )
