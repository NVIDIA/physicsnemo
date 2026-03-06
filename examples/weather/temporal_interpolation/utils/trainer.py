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

from collections.abc import Callable, Sequence
from typing import Any, Literal
import time

import torch
import wandb

from physicsnemo import Module
from physicsnemo.datapipes.climate.climate import ClimateDatapipe
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.utils.logging import LaunchLogger, PythonLogger
from physicsnemo.utils import load_checkpoint, save_checkpoint


class Trainer:
    """Training loop.

    Parameters
    ----------
    model : Module
        Model to train.
    dist_manager : DistributedManager
        Initialized DistributedManager.
    loss : torch.nn.Module
        Loss function.
    train_datapipe : ClimateDatapipe
        ClimateDatapipe providing training data.
    valid_datapipe : ClimateDatapipe
        ClimateDatapipe providing validation data.
    samples_per_epoch : int
        Number of samples to draw from the datapipe per 'epoch'.
    optimizer : torch.optim.Optimizer
        Optimizer used for training.
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Learning rate scheduler.
    input_output_from_batch_data : Callable, optional
        Function that converts datapipe outputs to training batches.
        If not provided, will try to use outputs as-is.
    max_epoch : int, optional
        The last training epoch.
    load_epoch : int, "latest", or None, optional
        Which epoch to load. Options:
        - "latest": continue from latest checkpoint in checkpoint_dir
        - int: continue from the specified epoch
        - None: start from scratch
    checkpoint_every : int, optional
        Save checkpoint every N epochs.
    checkpoint_dir : str or None, optional
        The directory where checkpoints are saved.
    validation_callbacks : Sequence[Callable], optional
        Optional callables to execute on validation. Signature:
        callback(outvar_true, outvar_pred, epoch=epoch, batch_idx=batch_idx).
    use_wandb : bool, optional
        When True, log metrics to Weights & Biases.
    """

    def __init__(
        self,
        model: Module,
        dist_manager: DistributedManager,
        loss: torch.nn.Module,
        train_datapipe: ClimateDatapipe,
        valid_datapipe: ClimateDatapipe,
        samples_per_epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        input_output_from_batch_data: Callable = lambda x: x,
        max_epoch: int = 1,
        load_epoch: int | Literal["latest"] | None = "latest",
        checkpoint_every: int = 1,
        checkpoint_dir: str | None = None,
        validation_callbacks: Sequence[Callable] = (),
        use_wandb: bool = False,
    ):
        self.model = model
        self.dist_manager = dist_manager
        self.loss = loss
        self.train_datapipe = train_datapipe
        self.train_iterator = iter(self.train_datapipe)
        self.valid_datapipe = valid_datapipe
        self.max_epoch = max_epoch
        self.input_output_from_batch_data = input_output_from_batch_data
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.validation_callbacks = validation_callbacks
        self.device = self.dist_manager.device
        self.logger = PythonLogger()
        self.use_wandb = use_wandb

        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir
        self.epoch = 1
        self.total_samples_trained = 0
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
        )(self._train_step_forward)

        self.eval_step = StaticCaptureEvaluateNoGrad(
            model=self.model, logger=self.logger, use_graphs=False
        )(self._eval_step)

        self.local_batches_per_epoch = samples_per_epoch // (
            train_datapipe.world_size * train_datapipe.batch_size
        )

    def _eval_step(self, invar: tuple) -> torch.Tensor:
        """Evaluate model for one step.

        Parameters
        ----------
        invar : tuple
            The inputs to the model, packed into a tuple.

        Returns
        -------
        torch.Tensor
            The output of the model.
        """
        return self.model(*invar)

    def _train_step_forward(
        self, invar: tuple, outvar_true: torch.Tensor
    ) -> torch.Tensor:
        """Training step.

        Parameters
        ----------
        invar : tuple
            Model inputs packed into a tuple.
        outvar_true : torch.Tensor
            Correct output value.

        Returns
        -------
        torch.Tensor
            Model loss on the given data.
        """
        outvar_pred = self.model(*invar)
        return self.loss(outvar_pred, outvar_true)

    def fit(self):
        """Main function for training loop."""
        # Log initial learning rate to wandb
        use_wandb_log = self.use_wandb and self.dist_manager.rank == 0
        if use_wandb_log:
            current_lr = self.optimizer.param_groups[0]["lr"]
            wandb.log({"lr": current_lr, "epoch": self.epoch - 1})

        for self.epoch in range(self.epoch, self.max_epoch + 1):
            epoch_loss = 0.0
            epoch_samples = 0
            time_start = time.time()

            with LaunchLogger(
                "train",
                epoch=self.epoch,
                num_mini_batch=self.local_batches_per_epoch,
                epoch_alert_freq=10,
            ) as log:
                for _ in range(self.local_batches_per_epoch):
                    try:
                        batch = next(self.train_iterator)
                    except StopIteration:
                        self.train_iterator = iter(self.train_datapipe)
                        batch = next(self.train_iterator)
                    loss = self.train_step_forward(
                        *self.input_output_from_batch_data(batch)
                    )
                    log.log_minibatch({"loss": loss.detach()})

                    # Track loss for epoch average
                    batch_size = self.train_datapipe.batch_size
                    epoch_loss += loss.item() * batch_size
                    epoch_samples += batch_size

                    # Log batch-level metrics to wandb
                    if use_wandb_log:
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        wandb.log({"batch_loss": loss.item(), "lr": current_lr})

                log.log_epoch({"Learning Rate": self.optimizer.param_groups[0]["lr"]})

            # Compute epoch statistics
            time_end = time.time()
            mean_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
            self.total_samples_trained += epoch_samples

            # Log epoch-level metrics to wandb
            if use_wandb_log:
                current_lr = self.optimizer.param_groups[0]["lr"]
                metrics = {
                    "epoch": self.epoch,
                    "mean_loss": mean_loss,
                    "time_per_epoch": time_end - time_start,
                    "lr": current_lr,
                    "total_samples_trained": self.total_samples_trained,
                    "epoch_samples": epoch_samples,
                }
                wandb.log(metrics)

            # Validation
            if self.dist_manager.rank == 0:
                with LaunchLogger("valid", epoch=self.epoch) as log:
                    error = self.validate_on_epoch()
                    log.log_epoch({"Validation error": error})

                    # Log validation metrics to wandb
                    if use_wandb_log:
                        val_loss = error.item() if torch.is_tensor(error) else error
                        val_metrics = {
                            "val_loss": val_loss,
                            "epoch": self.epoch,
                            "total_samples_trained": (self.total_samples_trained),
                        }
                        wandb.log(val_metrics)

            if self.dist_manager.world_size > 1:
                torch.distributed.barrier()

            self.lr_scheduler.step()

            checkpoint_epoch = (self.checkpoint_dir is not None) and (
                (self.epoch % self.checkpoint_every == 0)
                or (self.epoch == self.max_epoch)
            )
            if checkpoint_epoch and self.dist_manager.rank == 0:
                # Save Modulus Launch checkpoint
                self.save_checkpoint()

        if self.dist_manager.rank == 0:
            self.logger.info("Finished training!")

    @torch.no_grad()
    def validate_on_epoch(self) -> torch.Tensor:
        """Compute loss and metrics over one validation epoch.

        Returns
        -------
        torch.Tensor
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

    def load_checkpoint(self, epoch: int | None = None) -> int:
        """Try to load model state from a checkpoint.

        Do nothing if a checkpoint is not found in self.checkpoint_dir.

        Parameters
        ----------
        epoch : int or None, optional
            The number of epoch to load. When None, the latest epoch is loaded.

        Returns
        -------
        int
            The epoch of the loaded checkpoint, or 0 if no checkpoint was found.
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be set in order to load checkpoints.")
        metadata = {}
        self.epoch = load_checkpoint(
            self.checkpoint_dir,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            device=self.device,
            epoch=epoch,
            metadata_dict=metadata,
        )
        self.total_samples_trained = metadata.get("total_samples_trained", 0)
        return self.epoch

    def save_checkpoint(self):
        """Save current model state as a checkpoint."""
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be set in order to save checkpoints.")
        metadata = {"total_samples_trained": self.total_samples_trained}
        if self.use_wandb and wandb.run is not None:
            metadata["wandb_id"] = wandb.run.id
        save_checkpoint(
            self.checkpoint_dir,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            epoch=self.epoch,
            metadata=metadata,
        )
