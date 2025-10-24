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
import time

import fsspec
import hydra
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import zarr
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as torch_optimizers
from tqdm import tqdm

from utils import get_filesystem

# Add eval to OmegaConf TODO: Remove when OmegaConf is updated
OmegaConf.register_new_resolver("eval", eval)

try:
    from apex import optimizers as apex_optimizers
except:
    raise ImportError(
        "training requires apex package for optimizer."
        + "See https://github.com/nvidia/apex for install details."
    )

from physicsnemo import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import (
    LaunchLogger,
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.launch.logging.mlflow import initialize_mlflow
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils import StaticCaptureEvaluateNoGrad, StaticCaptureTraining

from seq_zarr_datapipe import SeqZarrDatapipe
from model_packages import save_inference_model_package


def batch_normalized_mse(pred: Tensor, target: Tensor) -> Tensor:
    """Calculates batch-wise normalized mse error between two tensors."""

    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    diff_norms = torch.linalg.norm(pred_flat - target_flat, ord=2.0, dim=1)
    target_norms = torch.linalg.norm(target_flat, ord=2.0, dim=1)

    error = diff_norms / target_norms
    return torch.mean(error)


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function for unified weather model training.
    """

    # Resolve config so that all values are concrete
    OmegaConf.resolve(cfg)

    # Initialize distributed environment for training
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers
    initialize_mlflow(
        experiment_name=cfg.experiment_name,
        experiment_desc=cfg.experiment_desc,
        run_name=f"{cfg.model.name}-trainng",
        run_desc=cfg.experiment_desc,
        user_name="PhysicsNeMo User",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)  # PhysicsNeMo launch logger
    logger = PythonLogger("main")  # General python logger

    # Initialize model
    model = Module.instantiate(
        {
            "__name__": cfg.model.name,
            "__args__": {
                k: tuple(v) if isinstance(v, ListConfig) else v
                for k, v in cfg.model.args.items()
            },  # TODO: maybe mobe this conversion to resolver?
        }
    )
    model = model.to(dist.device)

    # Distributed learning
    if dist.world_size > 1:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # Initialize optimizer (TODO: unify optimizer handling)
    if cfg.training.optimizer.framework == "apex":
        optimizers = apex_optimizers
    elif cfg.training.optimizer.framework == "torch":
        optimizers = torch_optimizers
    OptimizerClass = getattr(optimizers, cfg.training.optimizer.name)
    optimizer = OptimizerClass(model.parameters(), **cfg.training.optimizer.args)

    # Normalizer (TODO: Maybe wrap this into model)
    predicted_batch_norm = nn.BatchNorm2d(
        cfg.curated_dataset.nr_predicted_variables, momentum=None, affine=False
    ).to(dist.device)
    unpredicted_batch_norm = nn.BatchNorm2d(
        cfg.curated_dataset.nr_unpredicted_variables, momentum=None, affine=False
    ).to(dist.device)

    def normalize_variables(variables, batch_norm):
        shape = variables.shape
        variables = variables.flatten(0, 1)
        variables = batch_norm(variables)
        variables = variables.view(shape)
        return variables

    # Attempt to load latest checkpoint if one exists
    if dist.world_size > 1:
        torch.distributed.barrier()
    loaded_epoch = load_checkpoint(
        "./checkpoints",
        models=[model, predicted_batch_norm, unpredicted_batch_norm],
        optimizer=optimizer,
        scheduler=None,
        device=dist.device,
    )

    # Initialize filesytem (TODO: Add multiple filesystem support)
    fs = get_filesystem(
        cfg.filesystem.type,
        cfg.filesystem.key,
        cfg.filesystem.endpoint_url,
        cfg.filesystem.region_name,
    )

    # Get filesystem mapper for datasets
    train_dataset_mapper = fs.get_mapper(cfg.curated_dataset.train_dataset_filename)
    val_dataset_mapper = fs.get_mapper(cfg.curated_dataset.val_dataset_filename)

    # Initialize validation datapipe
    val_datapipe = SeqZarrDatapipe(
        file_mapping=val_dataset_mapper,
        variables=["time", "predicted", "unpredicted"],
        batch_size=cfg.validation.batch_size,
        num_steps=cfg.validation.num_steps + cfg.training.nr_input_steps,
        shuffle=False,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
        batch=cfg.datapipe.batch,
        parallel=cfg.datapipe.parallel,
        num_threads=cfg.datapipe.num_threads,
        prefetch_queue_depth=cfg.datapipe.prefetch_queue_depth,
        py_num_workers=cfg.datapipe.py_num_workers,
        py_start_method=cfg.datapipe.py_start_method,
    )

    # Unroll network
    def unroll(
        model, predicted_variables, unpredicted_variables, nr_input_steps, cpu=False
    ):
        # Get number of steps to unroll
        steps = unpredicted_variables.shape[1] - nr_input_steps

        # Create first input
        unpred_i = unpredicted_variables[:, :nr_input_steps]
        pred_i = predicted_variables[:, :nr_input_steps]

        # Unroll
        model_predicted = []
        for i in range(steps):
            # Get prediction for the first step
            model_pred_i = model(torch.cat([pred_i, unpred_i], dim=2).flatten(1, 2))
            model_predicted.append(model_pred_i)

            # Create new inputs
            unpred_i = unpredicted_variables[:, i : nr_input_steps + i]
            pred_i = torch.cat([pred_i[:, 1:], model_pred_i.unsqueeze(1)], dim=1)

        # Stack predictions
        model_predicted = torch.stack(model_predicted, dim=1)

        return model_predicted

    # Evaluation forward pass
    @StaticCaptureEvaluateNoGrad(model=model, logger=logger, use_graphs=False)
    def eval_forward(model, predicted_variables, unpredicted_variables, nr_input_steps):
        # Forward pass
        net_predicted_variables = unroll(
            model, predicted_variables, unpredicted_variables, nr_input_steps
        )

        # Get l2 loss
        num_elements = torch.prod(torch.Tensor(list(net_predicted_variables.shape[1:])))
        loss = (
            torch.sum(
                torch.pow(
                    net_predicted_variables - predicted_variables[:, nr_input_steps:], 2
                )
            )
            / num_elements
        )

        return loss, net_predicted_variables, predicted_variables[:, nr_input_steps:]

    # Training forward pass
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=logger, use_amp=cfg.training.amp_supported
    )  # TODO: remove amp supported config after SFNO fixed
    def train_step_forward(
        model, predicted_variables, unpredicted_variables, nr_input_steps
    ):
        # Forward pass
        net_predicted_variables = unroll(
            model, predicted_variables, unpredicted_variables, nr_input_steps
        )

        # Compute loss
        loss = batch_normalized_mse(
            net_predicted_variables, predicted_variables[:, nr_input_steps:]
        )

        return loss

    # Main training loop
    global_epoch = 0
    for stage in cfg.training.stages:
        # Skip if loaded epoch is greater than current stage
        if loaded_epoch > global_epoch:
            # Check if current stage needs to be run
            if loaded_epoch >= global_epoch + stage.num_epochs:
                # Skip stage
                global_epoch += stage.num_epochs
                continue
            # Otherwise, run stage for remaining epochs
            else:
                num_epochs = stage.num_epochs - (loaded_epoch - global_epoch)
        else:
            num_epochs = stage.num_epochs

        # Create new datapipe
        train_datapipe = SeqZarrDatapipe(
            file_mapping=train_dataset_mapper,
            variables=["time", "predicted", "unpredicted"],
            batch_size=stage.batch_size,
            num_steps=stage.unroll_steps + cfg.training.nr_input_steps,
            shuffle=True,
            device=dist.device,
            process_rank=dist.rank,
            world_size=dist.world_size,
            batch=cfg.datapipe.batch,
            parallel=cfg.datapipe.parallel,
            num_threads=cfg.datapipe.num_threads,
            prefetch_queue_depth=cfg.datapipe.prefetch_queue_depth,
            py_num_workers=cfg.datapipe.py_num_workers,
            py_start_method=cfg.datapipe.py_start_method,
        )

        # Initialize scheduler
        SchedulerClass = getattr(torch.optim.lr_scheduler, stage.lr_scheduler_name)
        scheduler = SchedulerClass(optimizer, **stage.args)

        # Set scheduler to current step
        scheduler.step(stage.num_epochs - num_epochs)

        # Get current step for checking if max iterations is reached
        current_step = len(train_datapipe) * (stage.num_epochs - num_epochs)

        # Run number of epochs
        for epoch in range(num_epochs):
            # Wrap epoch in launch logger for console / WandB logs
            with LaunchLogger(
                "train",
                epoch=epoch,
                num_mini_batch=len(train_datapipe),
                epoch_alert_freq=10,
            ) as log:
                # Track memory throughput
                tic = time.time()
                nr_bytes = 0

                # Training loop
                for j, data in tqdm(enumerate(train_datapipe)):
                    # Check if ran max iterations for stage
                    if current_step >= stage.max_iterations:
                        break

                    # Get predicted and unpredicted variables
                    predicted_variables = data[0]["predicted"]
                    unpredicted_variables = data[0]["unpredicted"]

                    # Normalize variables
                    predicted_variables = normalize_variables(
                        predicted_variables, predicted_batch_norm
                    )
                    unpredicted_variables = normalize_variables(
                        unpredicted_variables, unpredicted_batch_norm
                    )

                    # Log memory throughput
                    nr_bytes += (
                        predicted_variables.element_size()
                        * predicted_variables.nelement()
                    )
                    nr_bytes += (
                        unpredicted_variables.element_size()
                        * unpredicted_variables.nelement()
                    )

                    # Perform training step
                    loss = train_step_forward(
                        model,
                        predicted_variables,
                        unpredicted_variables,
                        cfg.training.nr_input_steps,
                    )
                    log.log_minibatch({"loss": loss.detach()})

                    # Increment current step
                    current_step += 1

                # Step scheduler (each step is an epoch)
                scheduler.step()

                # Log learning rate
                log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

                # Log memory throughput
                log.log_epoch({"GB/s": nr_bytes / (time.time() - tic) / 1e9})

            # Perform validation
            if dist.rank == 0:
                # Wrap validation in launch logger for console / WandB logs
                with LaunchLogger("valid", epoch=epoch) as log:
                    # Switch to eval mode
                    model.eval()

                    # Validation loop
                    loss_epoch = 0.0
                    num_examples = 0
                    for i, data in enumerate(val_datapipe):
                        # Get predicted and unpredicted variables
                        predicted_variables = data[0]["predicted"]
                        unpredicted_variables = data[0]["unpredicted"]

                        # Normalize variables
                        predicted_variables = normalize_variables(
                            predicted_variables, predicted_batch_norm
                        )
                        unpredicted_variables = normalize_variables(
                            unpredicted_variables, unpredicted_batch_norm
                        )

                        # Perform validation step and compute loss
                        (
                            loss,
                            net_predicted_variables,
                            predicted_variables,
                        ) = eval_forward(
                            model,
                            predicted_variables,
                            unpredicted_variables,
                            cfg.training.nr_input_steps,
                        )
                        loss_epoch += loss.detach().cpu().numpy()
                        num_examples += predicted_variables.shape[0]

                        # Plot validation on first batch
                        if i == 0:
                            net_predicted_variables = (
                                net_predicted_variables.cpu().numpy()
                            )
                            predicted_variables = predicted_variables.cpu().numpy()
                            for chan in range(net_predicted_variables.shape[2]):
                                plt.close("all")
                                fig, ax = plt.subplots(
                                    3,
                                    net_predicted_variables.shape[1],
                                    figsize=(15, net_predicted_variables.shape[0] * 5),
                                )
                                for t in range(net_predicted_variables.shape[1]):
                                    ax[0, t].set_title(
                                        "Network prediction, Step {}".format(t)
                                    )
                                    ax[1, t].set_title(
                                        "Ground truth, Step {}".format(t)
                                    )
                                    ax[2, t].set_title("Difference, Step {}".format(t))
                                    ax[0, t].imshow(net_predicted_variables[0, t, chan])
                                    ax[1, t].imshow(predicted_variables[0, t, chan])
                                    ax[2, t].imshow(
                                        net_predicted_variables[0, t, chan]
                                        - predicted_variables[0, t, chan]
                                    )

                                fig.savefig(
                                    f"forcast_validation_channel{chan}_epoch{epoch}.png"
                                )

                    # Log validation loss
                    log.log_epoch({"Validation error": loss_epoch / num_examples})

                    # Switch back to train mode
                    model.train()

            # Sync after each epoch
            if dist.world_size > 1:
                torch.distributed.barrier()

            # Save checkpoint
            if (epoch % 5 == 0 or epoch == 1) and dist.rank == 0:
                # Use PhysicsNeMo Launch checkpoint
                save_checkpoint(
                    "./checkpoints",
                    models=[model, predicted_batch_norm, unpredicted_batch_norm],
                    optimizer=optimizer,
                    scheduler=None,
                    epoch=epoch,
                )

                # Save model package
                logger.info("Saving model card")
                save_inference_model_package(
                    model,
                    cfg,
                    predicted_variable_normalizer=predicted_batch_norm,
                    unpredicted_variable_normalizer=unpredicted_batch_norm,
                    latitude=zarr.open(cfg.dataset.dataset_filename, mode="r")[
                        "latitude"
                    ],
                    longitude=zarr.open(cfg.dataset.dataset_filename, mode="r")[
                        "longitude"
                    ],
                    save_path="./model_package_{}".format(cfg.experiment_name),
                    readme="This is a model card for the global weather model.",
                )

        # Finish training
        if dist.rank == 0:
            # Save model card
            logger.info("Finished training!")


if __name__ == "__main__":
    main()
