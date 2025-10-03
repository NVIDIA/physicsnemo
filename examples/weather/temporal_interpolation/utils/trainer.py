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

from collections.abc import Callable, Sequence
from typing import Any, Literal
import warnings

from physicsnemo import Module
from physicsnemo.datapipes.climate.climate import ClimateDatapipe
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.launch.logging import LaunchLogger, PythonLogger
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
import torch

try:
    from apex.optimizers import FusedAdam
except ImportError:
    warnings.warn("Apex is not installed, defaulting to PyTorch optimizers.")


class Trainer:
    """Training loop.

    Args:
        model: model to train
        dist_manager: initialized DistributedManager
        loss: loss function
        train_datapipe: ClimateDatapipe providing training data
        valid_datapipe: ClimateDatapipe providing validation data
        samples_per_epoch: number of samples to draw from the datapipe per 'epoch'
        input_output_from_batch_data: function that converts datapipe outputs to training
            batches, if not provided will try to use outputs as-is
        optimizer: optimizer class used for training, when None will setup
            apex.optimizers.FusedAdam if available, otherwise PyTorch Adam
        optimizer_params: dict of parameters (e.g. learning rate) to pass to optimizer
        scheduler: learning rate scheduler class, when None will setup CosineAnnealingLR
        scheduler_params: dict of parameters to pass to LR scheduler
        max_epoch: the last training epoch
        load_epoch: which epoch to load; one of:
            - "latest" (to continue from latest checkpoint in checkpoint_dir)
            - int (to continue from the specified epoch)
            - None (to start from scratch)
        checkpoint_dir: the directory where checkpoints are saved
        validation_callbacks: optional callables to execute on validation, signature
            callback(outvar_true, outvar_pred, epoch=epoch, batch_idx=batch_idx)
    """

    def __init__(
        self,
        model: Module,
        dist_manager: DistributedManager,
        loss: torch.nn.Module,
        train_datapipe: ClimateDatapipe,
        valid_datapipe: ClimateDatapipe,
        samples_per_epoch: int,
        input_output_from_batch_data: Callable = lambda x: x,
        optimizer: type[torch.optim.Optimizer] | None = None,
        optimizer_params: dict[str, Any] | None = None,
        scheduler: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        scheduler_params: dict[str, Any] | None = None,
        max_epoch: int = 1,
        load_epoch: int | Literal["latest"] = None,
        checkpoint_every: int = 1,
        checkpoint_dir: str | None = None,
        validation_callbacks: Sequence[Callable] = (),
    ):
        self.model = model
        self.dist_manager = dist_manager
        self.loss = loss
        self.train_datapipe = train_datapipe
        self.valid_datapipe = valid_datapipe
        self.max_epoch = max_epoch
        self.input_output_from_batch_data = input_output_from_batch_data
        self.optimizer = self.setup_optimizer(
            model, opt_cls=optimizer, opt_params=optimizer_params
        )
        self.lr_scheduler = self.setup_lr_scheduler(
            self.optimizer, scheduler_cls=scheduler, scheduler_params=scheduler_params
        )
        self.validation_callbacks = validation_callbacks
        self.device = self.dist_manager.device
        self.logger = PythonLogger()

        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir
        self.epoch = 1
        if load_epoch is not None:
            epoch = None if load_epoch == "latest" else load_epoch
            self.load_checkpoint(epoch=epoch)
            self.epoch += 1

        # wrap capture here instead of using decorator so it'll still be wrapped if
        # overridden by a subclass
        self.train_step_forward = StaticCaptureTraining(
            model=self.model,
            optim=self.optimizer,
            logger=self.logger,
            use_graphs=False,  # use_graphs=True seems crash prone
        )(self.train_step_forward)

        self.eval_step = StaticCaptureEvaluateNoGrad(
            model=self.model, logger=self.logger, use_graphs=False
        )(self.eval_step)

        self.train_iter = self._train_iter()
        self.local_batches_per_epoch = samples_per_epoch // (
            train_datapipe.world_size * train_datapipe.batch_size
        )

    def eval_step(self, invar: tuple) -> torch.Tensor:
        """Evaluate model for one step.

        Args:
            invar: The inputs to the model, packed into a tuple.

        Returns:
            The output of the model.
        """
        return self.model(*invar)

    def train_step_forward(
        self, invar: tuple, outvar_true: torch.Tensor
    ) -> torch.Tensor:
        """Training step.

        Args:
            invar: model inputs packed into a tuple
            outvar_true: correct output value
        Returns:
            Model loss on the given data.
        """
        outvar_pred = self.model(*invar)
        return self.loss(outvar_pred, outvar_true)

    def fit(self):
        """Main function for training loop."""
        for self.epoch in range(self.epoch, self.max_epoch + 1):
            self.train_on_epoch()

        if self.dist_manager.rank == 0:
            self.logger.info("Finished training!")

    def _train_iter(self):
        """Iterate training items."""
        while True:
            yield from self.train_datapipe

    def _iter_n_train_batches(self, n):
        """Iterate over n training items."""
        for _ in range(n):
            yield next(self.train_iter)

    def train_on_epoch(self):
        """Train for one epoch."""
        with LaunchLogger(
            "train",
            epoch=self.epoch,
            num_mini_batch=self.local_batches_per_epoch,
            epoch_alert_freq=10,
        ) as log:
            for batch in self._iter_n_train_batches(self.local_batches_per_epoch):
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
            # Save Modulus Launch checkpoint
            self.save_checkpoint()

    @torch.no_grad()
    def validate_on_epoch(self) -> torch.Tensor:
        """Compute loss and metrics over one validation epoch.

        Returns:
            Validation loss as a tensor.
        """
        loss_epoch = 0
        num_examples = 0  # Number of validation examples
        # Dealing with DDP wrapper
        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model

        try:
            model.eval()
            for i, batch in enumerate(self.valid_datapipe):
                (invar, outvar_true) = self.input_output_from_batch_data(batch)
                invar = tuple(v.detach() for v in invar)
                outvar_true = outvar_true.detach()
                outvar_pred = self.eval_step(invar)

                loss_epoch += self.loss(outvar_pred, outvar_true)
                num_examples += 1

                for callback in self.validation_callbacks:
                    callback(outvar_true, outvar_pred, epoch=self.epoch, batch_idx=i)
        finally:  # restore train state even if exception occurs
            model.train()
        return loss_epoch / num_examples

    def setup_optimizer(
        self,
        model: torch.nn.Module,
        opt_cls: type[torch.optim.Optimizer] | None = None,
        opt_params: dict | None = None,
    ) -> torch.optim.Optimizer:
        """Setup optimizer.

        Args:
            model: model that optimizer is applied to
            opt_cls: optimizer class; when None will setup
                apex.optimizers.FusedAdam if available, otherwise PyTorch Adam
            opt_params: dict of parameters (e.g. learning rate) to pass to optimizer

        Returns:
            Initialized optimizer.
        """

        opt_kwargs = {"lr": 0.0005}
        if opt_params is not None:
            opt_kwargs.update(opt_params)

        if opt_cls is None:
            try:
                opt_cls = FusedAdam
            except NameError:  # in case we don't have apex
                opt_cls = torch.optim.Adam

        return opt_cls(model.parameters(), **opt_kwargs)

    def setup_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        scheduler_params: dict[str, Any] | None = None,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Setup learning rate scheduler.

        Args:
            optimizer: optimizer to which the scheduling is applied
            scheduler_cls: scheduler class; when None will setup
                apex.optimizers.FusedAdam if available, otherwise PyTorch Adam
            scheduler_params: dict of parameters to pass to scheduler

        Returns:
            Initialized optimizer.
        """

        scheduler_kwargs = {}
        if scheduler_cls is None:
            scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR
            scheduler_kwargs["T_max"] = self.max_epoch
        if scheduler_params is not None:
            scheduler_kwargs.update(scheduler_params)

        return scheduler_cls(optimizer, **scheduler_kwargs)

    def load_checkpoint(self, epoch: int | None = None) -> int:
        """Try to load model state from a checkpoint. Do nothing if a checkpoint
        is not found in self.checkpoint_dir.

        Args:
            epoch: The number of epoch to load. When None, the latest epoch is loaded.
        Returns:
            The epoch of the loaded checkpoint, or 0 if no checkpoint was found.
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
        """Save current model state as a checkpoint."""
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be set in order to save checkpoints.")
        save_checkpoint(
            self.checkpoint_dir,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            epoch=self.epoch,
        )
