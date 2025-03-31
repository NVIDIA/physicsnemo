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

#!/usr/bin/env python3
import gc
import os
import threading

import torch

# distributed stuff
from physicsnemo.distributed import DistributedManager
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# custom
from utils import write_checkpoint

from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper


class Trainer:
    """
    A class for DLWP model training
    """

    def __init__(
        self,
        model: torch.nn.Module,  # Specify... (import)
        data_module: torch.nn.Module,  # Specify... (import)
        criterion: torch.nn.Module,  # Specify... (import)
        optimizer: torch.nn.Module,  # Specify... (import)
        lr_scheduler: torch.nn.Module,  # Specify... (import)
        max_epochs: int = 500,
        early_stopping_patience: int = None,
        amp_mode: str = "none",
        graph_mode: str = "none",
        device: torch.device = torch.device("cpu"),
        output_dir: str = "/outputs/",
        max_norm: float = None,
    ):
        """
        Constructor.

        Parameters:
        model: torch.nn.Module
            The model to train
        data_module: torch.nn.Module
            The Pytorch module used for dataloading
        criterion: torch.nn.Module
            The PyTorch loss module to use
        optimizer: torch.nn.Module
            The PyTorch optimizer module to use
        lr_scheduler: torch.nn.Module
            The PyTorch learning rate scheduler module to use
        max_epochs: int, optional
            The maximum number of epochs to train for
        early_stopping_patience: int, optional
        amp_mode: str, optional
            amp mode to use, valid options ["fp16", "bfloat16"]
        graph_mode: str, optional
            Where to use cudagraphs for training, valid options ["train", "train_eval"]
        device: torch.device, optional
            Device on which to run training on, can be any available torch.device
        output_dir: str, optional
            Where to store results
        max_norm: float, optional
            Maximum norm to use for training
        """
        self.device = device
        self.amp_enable = False if (amp_mode == "none") else True
        self.amp_dtype = torch.float16 if (amp_mode == "fp16") else torch.bfloat16
        self.output_variables = data_module.output_variables
        self.early_stopping_patience = early_stopping_patience
        self.max_norm = max_norm

        self.model = model.to(device=self.device)

        self.dist = DistributedManager()

        # Initialize logger.
        self.logger = PythonLogger(name="training_loop")  # General python logger
        self.logger0 = RankZeroLoggingWrapper(self.logger, self.dist)
        self.logger.file_logging(file_name=f".logs/training_loop_{self.dist.rank}.log")

        self.dataloader_train, self.sampler_train = data_module.train_dataloader(
            num_shards=self.dist.world_size, shard_id=self.dist.rank
        )
        self.dataloader_valid, self.sampler_valid = data_module.val_dataloader(
            num_shards=self.dist.world_size, shard_id=self.dist.rank
        )
        self.output_dir_tb = os.path.join(output_dir, "tensorboard")

        # set the other parameters
        self.optimizer = optimizer
        # Set up criterion, pass metadata
        self.criterion = criterion.to(device=self.device)
        try:
            self.criterion.setup(self)
        except AttributeError:
            raise NotImplementedError(
                'Attribute error encountered in call to criterio.setup(). \
                Could be that criterion is not compatable with custom loss dlwp training. See \
                "physicsnemo/metrics/climate/healpix_loss.py" for proper criterion implementation examples.'
            )

        # opportunity for custom loss classes to get everything in order
        self.lr_scheduler = lr_scheduler
        self.max_epochs = max_epochs

        # add gradient scaler
        self.gscaler = amp.GradScaler(
            enabled=(self.amp_enable and self.amp_dtype == torch.float16)
        )

        # use distributed data parallel if requested:
        self.print_to_screen = True
        self.train_graph = None
        self.eval_graph = None

        # for status bars
        self.print_to_screen = self.dist.rank == 0

        if self.dist.device.type == "cuda":
            capture_stream = torch.cuda.Stream()
            if torch.distributed.is_initialized():
                with torch.cuda.stream(capture_stream):
                    self.model = DDP(
                        self.model,
                        device_ids=[self.device.index],
                        output_device=[self.device.index],
                        broadcast_buffers=True,
                        find_unused_parameters=False,
                        gradient_as_bucket_view=True,
                    )
                    capture_stream.synchronize()

            # capture graph if requested
            if graph_mode in ["train", "train_eval"]:
                self.logger0.info("Capturing model for training ...")
                # get the shapes
                inp, tar = next(iter(self.dataloader_train))

                self._train_capture(capture_stream, [x.shape for x in inp], tar.shape)

                if graph_mode == "train_eval":
                    self.logger0.info("Capturing model for validation ...")
                    self._eval_capture(capture_stream)

        # Set up tensorboard summary_writer or try 'weights and biases'
        # Initialize tensorbaord to track scalars
        if self.dist.rank == 0:
            self.writer = SummaryWriter(log_dir=self.output_dir_tb)

    def _train_capture(
        self, capture_stream, inp_shapes, tar_shape, num_warmup_steps=20
    ):
        # perform graph capture of the model
        self.static_inp = [
            torch.zeros(x_shape, dtype=torch.float32, device=self.device)
            for x_shape in inp_shapes
        ]
        self.static_tar = torch.zeros(
            tar_shape, dtype=torch.float32, device=self.device
        )

        self.model.train()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(num_warmup_steps):
                self.model.zero_grad(set_to_none=True)

                # FW
                with amp.autocast(enabled=self.amp_enable, dtype=self.amp_dtype):
                    self.static_gen_train = self.model.forward(self.static_inp)

                    self.static_loss_train = self.criterion(
                        self.static_gen_train, self.static_tar
                    )

                # BW
                self.gscaler.scale(self.static_loss_train).backward()

            # sync here
            capture_stream.synchronize()

            gc.collect()
            torch.cuda.empty_cache()

            # create graph
            self.train_graph = torch.cuda.CUDAGraph()

            # zero grads before capture:
            self.model.zero_grad(set_to_none=True)

            # start capture
            with torch.cuda.graph(self.train_graph):
                # FW
                with amp.autocast(enabled=self.amp_enable, dtype=self.amp_dtype):
                    # self.static_gen_train = self.model(self.static_inp)
                    self.static_gen_train = self.model.forward(self.static_inp)

                    self.static_loss_train = self.criterion(
                        self.static_gen_train, self.static_tar
                    )

                # BW
                self.gscaler.scale(self.static_loss_train).backward()

        # wait for capture to finish
        torch.cuda.current_stream().wait_stream(capture_stream)

    def _eval_capture(self, capture_stream, num_warmup_steps=20):
        self.model.eval()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            with torch.no_grad():
                for _ in range(num_warmup_steps):
                    # FW
                    with amp.autocast(enabled=self.amp_enable, dtype=self.amp_dtype):
                        # self.static_gen_eval = self.model(self.static_inp)
                        self.static_gen_eval = self.model.forward(self.static_inp)

                        self.static_loss_eval = self.criterion(
                            self.static_gen_eval, self.static_tar
                        )
                        # False flag for average channels ensures criterion will keep variable loss separated
                        self.static_losses_eval = self.criterion(
                            self.static_gen_eval,
                            self.static_tar,
                            average_channels=False,
                        )

            # sync here
            capture_stream.synchronize()

            gc.collect()
            torch.cuda.empty_cache()

            # create graph
            self.eval_graph = torch.cuda.CUDAGraph()

            # start capture:
            with torch.cuda.graph(self.eval_graph, pool=self.train_graph.pool()):
                # FW
                with torch.no_grad():
                    with amp.autocast(enabled=self.amp_enable, dtype=self.amp_dtype):
                        # self.static_gen_eval = self.model(self.static_inp)
                        self.static_gen_eval = self.model.forward(self.static_inp)

                        self.static_loss_eval = self.criterion(
                            self.static_gen_eval, self.static_tar
                        )
                        # False flag for average channels ensures criterion will keep variable loss separated
                        self.static_losses_eval = self.criterion(
                            self.static_gen_eval,
                            self.static_tar,
                            average_channels=False,
                        )

        # wait for capture to finish
        torch.cuda.current_stream().wait_stream(capture_stream)

    def fit(
        self,
        epoch: int = 0,
        validation_error: torch.Tensor = torch.inf,
        iteration: int = 0,
        epochs_since_improved: int = 0,
    ):
        """
        Perform training by iterating over all epochs

        Parameters
        epoch: int, optional
            Current epoch number
        validation_error: torch.Tensor, optional
            Current best validation error
        iteration: int, optional
            Current iteration number
        epochs_since_improved: int, optional
            Number of epochs that have seen improvement in validation error
        """
        best_validation_error = validation_error
        for epoch in range(epoch, self.max_epochs):
            torch.cuda.nvtx.range_push(f"training epoch {epoch}")

            if self.sampler_train is not None:
                self.sampler_train.set_epoch(epoch)

            # Train: iterate over all training samples
            training_step = 0
            self.model.train()
            for inputs, target in (
                pbar := tqdm(self.dataloader_train, disable=(not self.print_to_screen))
            ):
                pbar.set_description(f"Training   epoch {epoch}/{self.max_epochs}")

                # Trach epoch in tensorboard
                if self.dist.rank == 0:
                    self.writer.add_scalar(
                        tag="epoch", scalar_value=epoch, global_step=iteration
                    )

                torch.cuda.nvtx.range_push(f"training step {training_step}")

                inputs = [x.to(device=self.device) for x in inputs]
                target = target.to(device=self.device)

                # do optimizer step
                if self.train_graph is not None:
                    # copy data into entry nodes
                    for idx, inp in enumerate(inputs):
                        self.static_inp[idx].copy_(inp)

                    self.static_tar.copy_(target)

                    # replay
                    self.train_graph.replay()

                    # extract loss
                    output = self.static_gen_train
                    train_loss = self.static_loss_train
                else:
                    # zero grads
                    self.model.zero_grad(set_to_none=True)

                    if self.amp_enable:
                        with amp.autocast(
                            enabled=self.amp_enable, dtype=self.amp_dtype
                        ):
                            output = self.model(inputs)
                            train_loss = self.criterion(output, target)
                    else:
                        output = self.model(inputs)
                        train_loss = self.criterion(output, target)

                    self.gscaler.scale(train_loss).backward()

                # Gradient clipping
                self.gscaler.unscale_(self.optimizer)
                try:
                    curr_lr = (
                        self.optimizer.param_groups[-1]["lr"]
                        if self.lr_scheduler is None
                        else self.lr_scheduler.get_last_lr()[0]
                    )
                except (
                    AttributeError
                ):  # try loop required since LearnOnPlateau has no "get_last_lr" attribute
                    curr_lr = (
                        self.optimizer.param_groups[-1]["lr"]
                        if self.lr_scheduler is None
                        else self.optimizer.param_groups[0]["lr"]
                    )
                # check that max norm was not given to trainer
                if self.max_norm is None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), curr_lr)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_norm
                    )

                # Optimizer step
                self.gscaler.step(self.optimizer)
                self.gscaler.update()

                pbar.set_postfix({"Loss": train_loss.item()})

                torch.cuda.nvtx.range_pop()

                if self.dist.rank == 0:
                    self.writer.add_scalar(
                        tag="loss", scalar_value=train_loss, global_step=iteration
                    )
                iteration += 1
                training_step += 1

            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(f"validation epoch {epoch}")

            # Validate (without gradients)
            if self.sampler_valid is not None:
                self.sampler_valid.set_epoch(epoch)

            self.model.eval()
            with torch.no_grad():
                validation_stats = torch.zeros(
                    (2 + len(self.output_variables)),
                    dtype=torch.float32,
                    device=self.device,
                )
                for inputs, target in (
                    pbar := tqdm(
                        self.dataloader_valid, disable=(not self.print_to_screen)
                    )
                ):
                    pbar.set_description(f"Validation epoch {epoch}/{self.max_epochs}")
                    inputs = [x.to(device=self.device) for x in inputs]
                    target = target.to(device=self.device)
                    bsize = float(target.shape[0])

                    # do eval step
                    if self.eval_graph is not None:
                        # copy data into entry nodes
                        for idx, inp in enumerate(inputs):
                            self.static_inp[idx].copy_(inp)
                        self.static_tar.copy_(target)

                        # replay graph
                        self.eval_graph.replay()

                        # increase the loss
                        validation_stats[0] += self.static_loss_eval * bsize

                        # Same for the per-variable loss
                        for v_idx, v_name in enumerate(self.output_variables):
                            validation_stats[1 + v_idx] += (
                                self.static_losses_eval[v_idx] * bsize
                            )
                    else:
                        if self.amp_enable:
                            with amp.autocast(
                                enabled=self.amp_enable, dtype=self.amp_dtype
                            ):
                                output = self.model(inputs)
                                validation_stats[0] += (
                                    self.criterion(prediction=output, target=target)
                                    * bsize
                                )
                                # save per variable loss
                                eval_losses = self.criterion(
                                    output, target, average_channels=False
                                )
                                for v_idx, v_name in enumerate(self.output_variables):
                                    validation_stats[1 + v_idx] += (
                                        eval_losses[v_idx] * bsize
                                    )
                        else:
                            output = self.model(inputs)
                            validation_stats[0] += (
                                self.criterion(prediction=output, target=target) * bsize
                            )
                            eval_losses = self.criterion(
                                output, target, average_channels=False
                            )
                            for v_idx, v_name in enumerate(self.output_variables):
                                validation_stats[1 + v_idx] += (
                                    eval_losses[v_idx] * bsize
                                )

                    pbar.set_postfix(
                        {"Loss": (validation_stats[0] / validation_stats[-1]).item()}
                    )

                    # increment sample counter
                    validation_stats[-1] += bsize

                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(validation_stats)

                validation_error = (validation_stats[0] / validation_stats[-1]).item()

                # Record error per variable
                validation_errors = []
                for v_idx, v_name in enumerate(self.output_variables):
                    validation_errors.append(
                        (validation_stats[1 + v_idx] / validation_stats[-1]).item()
                    )

                # Track validation improvement to later check early stopping criterion
                if validation_error < best_validation_error:
                    best_validation_error = validation_error
                    epochs_since_improved = 0
                else:
                    epochs_since_improved += 1

            torch.cuda.nvtx.range_pop()

            # Logging and checkpoint saving
            if self.dist.rank == 0:
                if self.lr_scheduler is not None:
                    self.writer.add_scalar(
                        tag="learning_rate",
                        scalar_value=self.optimizer.param_groups[0]["lr"],
                        global_step=iteration,
                    )
                self.writer.add_scalar(
                    tag="val_loss", scalar_value=validation_error, global_step=iteration
                )

                # Per-variable loss
                for v_idx, v_name in enumerate(self.output_variables):
                    self.writer.add_scalar(
                        tag=f"val_loss/{v_name}",
                        scalar_value=validation_errors[v_idx],
                        global_step=iteration,
                    )

                # Write model checkpoint to file, using a separate thread
                self.logger0.info("Writing checkpoint")
                thread = threading.Thread(
                    target=write_checkpoint,
                    args=(
                        self.model.module
                        if torch.distributed.is_initialized()
                        else self.model,
                        self.optimizer,
                        self.lr_scheduler,
                        epoch,
                        iteration,
                        validation_error,
                        epochs_since_improved,
                        self.output_dir_tb,
                    ),
                )
                thread.start()

            # Update learning rate
            try:
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            except TypeError:  # Plateau Learning rate requires val loss
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(validation_error)

            # Check early stopping criterium
            if (
                self.early_stopping_patience is not None
                and epochs_since_improved >= self.early_stopping_patience
            ):
                self.logger0.info(
                    f"Hit early stopping criterium by not improving the validation error for {epochs_since_improved}"
                    " epochs. Finishing training."
                )
                break

        # Wrap up
        if self.dist.rank == 0:
            try:
                thread.join()
            except UnboundLocalError:
                pass
            self.writer.flush()
            self.writer.close()
