"""
SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

=====================================================================
                        BATCH TRAINING UTILITIES
=====================================================================

This module provides high-level training utilities for batch reservoir
forward-modeling surrogates. It contains the end-to-end training loop with
metric logging, learning-rate scheduling, model selection, and periodic
artifact saving (models and checkpoints).

Key Features:
- Multi-target training (pressure/saturation/gas/oil/Peaceman) in batch mode
- Per-target optimizer and scheduler stepping per mini-batch
- MLflow metric logging and console progress logging via LaunchLogger
- Periodic persistence of best models and checkpoints
- Support for distributed (DDP) execution

Usage:
    from forward.utils.batch.training_function import run_training_loop

@Author : Clement Etienam
"""

# 🔥 Torch & PhyNeMo
import torch
import copy

from forward.machine_extract import (
    save_model_to_buffer,
)

from physicsnemo.launch.logging import (
    LaunchLogger,
)


def run_training_loop(
    dist,
    logger,
    cfg,
    mlflow,
    use_epoch,
    output_variables,
    surrogate_pressure,
    surrogate_gas,
    surrogate_saturation,
    surrogate_oil,
    surrogate_peacemann,
    labelled_loader_train,
    labelled_loader_testt,
    composite_model,
    input_keys,
    input_keys_peacemann,
    output_keys_pressure,
    output_keys_gas,
    output_keys_saturation,
    output_keys_oil,
    output_keys_peacemann,
    training_step,
    validation_step,
    training_step_metrics,
    val_step_metrics,
    steppi,
    UO,
    BO,
    UW,
    BW,
    DZ,
    RE,
    max_inn_fcnx,
    max_out_fcnx,
    params,
    p_bub,
    p_atm,
    CFO,
    Relperm,
    SWI,
    SWR,
    SWOW,
    SWOG,
    params1_swow,
    params2_swow,
    params1_swog,
    params2_swog,
    N_pr,
    lenwels,
    neededM,
    nx,
    ny,
    nz,
    target_min,
    target_max,
    minK,
    maxK,
    minP,
    maxP,
    pde_method,
    max_inn_fcn,
    max_out_fcn,
    scheduler_pressure,
    scheduler_saturation,
    scheduler_oil,
    scheduler_gas,
    scheduler_peacemann,
    optimizer_pressure,
    optimizer_saturation,
    optimizer_oil,
    optimizer_gas,
    optimizer_peacemann,
    best_pressure,
    best_gas,
    best_saturation,
    best_oil,
    best_peacemann,
):
    """Run the batch training/validation loop with logging and persistence.

    Executes training over ``cfg.training.max_steps`` epochs, stepping the
    appropriate optimizers/schedulers per mini-batch and logging metrics to
    MLflow/console. Tracks the best-performing models (by training loss), and
    periodically saves both full model files and lightweight checkpoints.

    Parameters
    ----------
    dist : Any
        Distributed context exposing ``rank``, ``device``, etc.
    logger : logging.Logger
        Logger for console/file messages.
    cfg : DictConfig
        Hydra configuration with model/training settings.
    mlflow : module
        Active MLflow client for metric logging.
    use_epoch : int
        Last restored epoch; training resumes from ``max(1, use_epoch+1)``.
    output_variables : list[str]
        Targets being trained (e.g., ["PRESSURE", "SWAT", ...]).
    surrogate_* : torch.nn.Module
        Model instances for each target (present only if target is enabled).
    labelled_loader_train, labelled_loader_testt : torch.utils.data.DataLoader
        Training and validation data loaders.
    composite_model : torch.nn.Module
        Model wrapper that routes inputs to per-target surrogates.
    input_keys, input_keys_peacemann : list[str]
        Input tensors expected by the composite model (grid and Peaceman).
    output_keys_* : list[str]
        Per-target output keys dictating targets extracted from each batch.
    training_step, validation_step : Callable
        Functions that compute one training/validation step and update metrics.
    training_step_metrics, val_step_metrics : dict
        Mutable dicts populated by the step functions with per-target losses.
    steppi : int
        Number of time steps per sample for grid targets.
    UO, BO, UW, BW, DZ, RE, max_inn_fcnx, max_out_fcnx, params, p_bub, p_atm, CFO,
    Relperm, SWI, SWR, SWOW, SWOG, params1_swow, params2_swow, params1_swog, params2_swog,
    N_pr, lenwels, neededM, nx, ny, nz, target_min, target_max, minK, maxK, minP, maxP,
    pde_method, max_inn_fcn, max_out_fcn : Various
        Physical and scaling parameters forwarded to the step functions.
    scheduler_* : torch.optim.lr_scheduler._LRScheduler
        Learning-rate schedulers per target.
    optimizer_* : torch.optim.Optimizer
        Optimizers per target and peaceman model.
    best_* : torch.nn.Module
        Best-so-far copies of models; updated when training loss improves.

    Returns
    -------
    None
        Operates for side effects: training, logging, and saving artifacts.
    """
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                     START PHYNEMO SOLVER                        |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        if mlflow.active_run() is None:
            logger.info("[MLflow] Ensuring an active run before logging...")
            mlflow.start_run()
            logger.info("[MLflow] Active run confirmed.")
    if cfg.custom.model_Distributed == 1:
        torch.distributed.barrier()
    training_loss, validation_loss = 0, 0
    validation_loss_history, training_loss_history = [], []
    max_epoch = cfg.training.max_steps

    for epoch in range(max(1, use_epoch + 1), max_epoch + 1):
        if "PRESSURE" in output_variables:
            surrogate_pressure.train()
        if "SGAS" in output_variables:
            surrogate_gas.train()
        if "SWAT" in output_variables:
            surrogate_saturation.train()
        if "SOIL" in output_variables:
            surrogate_oil.train()
        surrogate_peacemann.train()
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(labelled_loader_train),
            epoch_alert_freq=1,
        ) as log:
            # === Training step ===
            total_losst = 0
            num_batchest = 0
            total_losspt = 0
            total_losswt = 0
            total_lossot = 0
            total_lossgt = 0
            total_losspet = 0
            d3 = 0
            d4 = 0
            d6 = 0
            d7 = 0
            for data in labelled_loader_train:
                inputin = {key: data[key] for key in input_keys}
                inputin_p = {key: data[key] for key in input_keys_peacemann}
                TARGETS = {}
                if "PRESSURE" in output_variables:
                    target_pressure = {key: data[key] for key in output_keys_pressure}
                    TARGETS["PRESSURE"] = target_pressure
                if "SGAS" in output_variables:
                    target_gas = {key: data[key] for key in output_keys_gas}
                    TARGETS["GAS"] = target_gas
                if "SWAT" in output_variables:
                    target_saturation = {
                        key: data[key] for key in output_keys_saturation
                    }
                    TARGETS["SATURATION"] = target_saturation
                if "SOIL" in output_variables:
                    target_oil = {key: data[key] for key in output_keys_oil}
                    TARGETS["OIL"] = target_oil
                target_peacemann = {key: data[key] for key in output_keys_peacemann}
                TARGETS["PEACEMANN"] = target_peacemann
                loss = training_step(
                    composite_model,
                    inputin,
                    inputin_p,
                    TARGETS,
                    cfg,
                    dist.device,
                    output_keys_saturation,
                    steppi,
                    output_variables,
                    training_step_metrics,
                    UO,
                    BO,
                    UW,
                    BW,
                    DZ,
                    RE,
                    max_inn_fcnx,
                    max_out_fcnx,
                    params,
                    p_bub,
                    p_atm,
                    CFO,
                    Relperm,
                    SWI,
                    SWR,
                    SWOW,
                    SWOG,
                    params1_swow,
                    params2_swow,
                    params1_swog,
                    params2_swog,
                    N_pr,
                    lenwels,
                    neededM,
                    nx,
                    ny,
                    nz,
                    target_min,
                    target_max,
                    minK,
                    maxK,
                    minP,
                    maxP,
                    pde_method,
                    max_inn_fcn,
                    max_out_fcn,
                    epoch,
                )
                if "PRESSURE" in output_variables:
                    total_losspt += training_step_metrics.get("pressure_loss", None)
                if "SWAT" in output_variables:
                    total_losswt += training_step_metrics.get("water_loss", None)
                if "SOIL" in output_variables:
                    total_lossot += training_step_metrics.get("oil_loss", None)
                if "SGAS" in output_variables:
                    total_lossgt += training_step_metrics.get("gas_loss", None)
                total_losspet += training_step_metrics.get("peacemann_loss", None)
                num_batchest += 1
                if cfg.custom.fno_type == "PINO":
                    d3 += training_step_metrics.get("pressured", None)
                    d4 += training_step_metrics.get("saturationd", None)
                    d6 += training_step_metrics.get("gasd", None)
                    d7 += training_step_metrics.get("peacemanned", None)
                total_losst += loss.item()
                if "PRESSURE" in output_variables:
                    scheduler_pressure.step()
                if "SWAT" in output_variables:
                    scheduler_saturation.step()
                if "SOIL" in output_variables:
                    scheduler_oil.step()
                if "SGAS" in output_variables:
                    scheduler_gas.step()
                scheduler_peacemann.step()
                del loss
                del TARGETS, inputin, inputin_p
            loss_train = total_losst / num_batchest
            if "PRESSURE" in output_variables:
                pressure_loss = total_losspt / num_batchest
            if "SWAT" in output_variables:
                water_loss = total_losswt / num_batchest
            if "SOIL" in output_variables:
                oil_loss = total_lossot / num_batchest
            if "SGAS" in output_variables:
                gas_loss = total_lossgt / num_batchest
            peacemann_loss = total_losspet / num_batchest
            if cfg.custom.fno_type == "PINO":
                f_pressure2 = d3 / num_batchest
                f_water2 = d4 / num_batchest
                f_gas2 = d6 / num_batchest
                f_peacemann2 = d7 / num_batchest
            current_training_loss = loss_train
            training_loss_history.append(current_training_loss)
            if (epoch % 100 == 0 or epoch == 1) and dist.rank == 0:
                mlflow.log_metric("training_loss", loss_train, step=epoch)
                if "PRESSURE" in output_variables:
                    mlflow.log_metric(
                        "training_data_pressure_loss", pressure_loss, step=epoch
                    )
                if "SWAT" in output_variables:
                    mlflow.log_metric(
                        "training_data_water_loss", water_loss, step=epoch
                    )
                if "SOIL" in output_variables:
                    mlflow.log_metric("training_data_oil_loss", oil_loss, step=epoch)
                if "SGAS" in output_variables:
                    mlflow.log_metric("training_data_gas_loss", gas_loss, step=epoch)
                mlflow.log_metric(
                    "training_data_peacemann_loss", peacemann_loss, step=epoch
                )
                if cfg.custom.fno_type == "PINO":
                    mlflow.log_metric(
                        "training_physics_pressure_loss", f_pressure2, step=epoch
                    )
                    mlflow.log_metric(
                        "training_physics_water_loss", f_water2, step=epoch
                    )
                    # mlflow.log_metric("training_closed_form_loss", loss_pde3, step=epoch)
                    mlflow.log_metric("training_physics_gas_loss", f_gas2, step=epoch)
                    mlflow.log_metric(
                        "training_physics_peacemann_loss", f_peacemann2, step=epoch
                    )
                logger.info(f"[MLflow] Logged training metrics for epoch {epoch}")
                log_data = {}
                if "PRESSURE" in output_variables:
                    log_data["training_data_pressure_loss"] = pressure_loss
                if "SWAT" in output_variables:
                    log_data["training_data_water_loss"] = water_loss
                if "SOIL" in output_variables:
                    log_data["training_data_oil_loss"] = oil_loss
                if "SGAS" in output_variables:
                    log_data["training_data_gas_loss"] = gas_loss
                log_data["training_loss"] = loss_train
                log_data["training_data_peacemann_loss"] = peacemann_loss
                if cfg.custom.fno_type == "PINO":
                    log_data.update(
                        {
                            "training_physics_pressure_loss": f_pressure2,
                            "training_physics_water_loss": f_water2,
                            "training_physics_gas_loss": f_gas2,
                            "training_physics_peacemann_loss": f_peacemann2,
                        }
                    )
                log.log_epoch(log_data)
                if training_loss < current_training_loss:
                    log.log_epoch(
                        {
                            "Loss increased by": abs(
                                training_loss - current_training_loss
                            )
                        }
                    )
                elif training_loss > current_training_loss:
                    log.log_epoch(
                        {
                            "Loss decreased by ": abs(
                                training_loss - current_training_loss
                            )
                        }
                    )
                else:
                    log.log_epoch({"No change in loss ": 0})

                log.log_epoch(
                    {
                        "Learning Rate - pressure": optimizer_pressure.param_groups[0][
                            "lr"
                        ],
                        "Learning Rate - saturation": optimizer_saturation.param_groups[
                            0
                        ]["lr"],
                        "Learning Rate - peacemann": optimizer_peacemann.param_groups[
                            0
                        ]["lr"],
                    }
                )
        if dist.rank == 0:
            with LaunchLogger("validation", epoch=epoch) as log:
                total_loss = 0
                num_batches = 0
                total_lossp = 0
                total_lossw = 0
                total_losso = 0
                total_lossg = 0
                total_losspe = 0
                for data in labelled_loader_testt:
                    inputin = {key: data[key] for key in input_keys}
                    inputin_p = {key: data[key] for key in input_keys_peacemann}
                    TARGETS = {}
                    if "PRESSURE" in output_variables:
                        target_pressure = {
                            key: data[key] for key in output_keys_pressure
                        }
                        TARGETS["PRESSURE"] = target_pressure
                    if "SGAS" in output_variables:
                        target_gas = {key: data[key] for key in output_keys_gas}
                        TARGETS["GAS"] = target_gas
                    if "SWAT" in output_variables:
                        target_saturation = {
                            key: data[key] for key in output_keys_saturation
                        }
                        TARGETS["SATURATION"] = target_saturation
                    if "SOIL" in output_variables:
                        target_oil = {key: data[key] for key in output_keys_oil}
                        TARGETS["OIL"] = target_oil
                    target_peacemann = {key: data[key] for key in output_keys_peacemann}
                    TARGETS["PEACEMANN"] = target_peacemann
                    batch_loss = validation_step(
                        composite_model,
                        inputin,
                        inputin_p,
                        TARGETS,
                        cfg,
                        dist.device,
                        output_keys_saturation,
                        steppi,
                        output_variables,
                        val_step_metrics,
                    )
                    total_loss += batch_loss.item()
                    if "PRESSURE" in output_variables:
                        total_lossp += val_step_metrics.get("pressure_loss", None)
                    if "SWAT" in output_variables:
                        total_lossw += val_step_metrics.get("water_loss", None)
                    if "SOIL" in output_variables:
                        total_losso += val_step_metrics.get("oil_loss", None)
                    if "SGAS" in output_variables:
                        total_lossg += val_step_metrics.get("gas_loss", None)
                    total_losspe += val_step_metrics.get("peacemann_loss", None)
                    num_batches += 1
                loss_test = total_loss / num_batches
                if "PRESSURE" in output_variables:
                    pressure_loss = total_lossp / num_batches
                if "SWAT" in output_variables:
                    water_loss = total_lossw / num_batches
                if "SOIL" in output_variables:
                    oil_loss += total_losso / num_batches
                if "SGAS" in output_variables:
                    gas_loss += total_lossg / num_batches
                peacemann_loss += total_losspe / num_batches
                current_validation_loss = loss_test
                validation_loss_history.append(current_validation_loss)
                if epoch % 100 == 0 or epoch == 1:
                    mlflow.log_metric("Validation_loss", loss_test, step=epoch)
                    if "PRESSURE" in output_variables:
                        mlflow.log_metric(
                            "Validation_data_pressure_loss", pressure_loss, step=epoch
                        )
                    if "SWAT" in output_variables:
                        mlflow.log_metric(
                            "Validation_data_water_loss", water_loss, step=epoch
                        )
                    if "SOIL" in output_variables:
                        mlflow.log_metric(
                            "Validation_data_oil_loss", oil_loss, step=epoch
                        )
                    if "SGAS" in output_variables:
                        mlflow.log_metric(
                            "Validation_data_gas_loss", gas_loss, step=epoch
                        )
                    mlflow.log_metric(
                        "Validation_data_peacemann_loss", peacemann_loss, step=epoch
                    )
                    logger.info(f"[MLflow] Logged validation metrics for epoch {epoch}")
                    validation_log_data = {}
                    if "PRESSURE" in output_variables:
                        validation_log_data["Validation_data_pressure_loss"] = (
                            pressure_loss
                        )
                    if "SWAT" in output_variables:
                        validation_log_data["Validation_data_water_loss"] = water_loss
                    if "SOIL" in output_variables:
                        validation_log_data["Validation_data_oil_loss"] = oil_loss
                    if "SGAS" in output_variables:
                        validation_log_data["Validation_data_gas_loss"] = gas_loss
                    validation_log_data["Validation_loss"] = loss_test
                    validation_log_data["Validation_data_peacemann_loss"] = (
                        peacemann_loss
                    )
                    log.log_epoch(validation_log_data)
                    if validation_loss < current_validation_loss:
                        log.log_epoch(
                            {
                                "Test loss increased by ": abs(
                                    validation_loss - current_validation_loss
                                )
                            }
                        )
                    elif validation_loss > current_validation_loss:
                        log.log_epoch(
                            {
                                "Test loss decreased by": abs(
                                    validation_loss - current_validation_loss
                                )
                            }
                        )
                    else:
                        log.log_epoch({"No change in loss": 0})
                training_loss = current_training_loss
                validation_loss = current_validation_loss
                if epoch == 1:
                    best_cost = training_loss
                else:
                    pass
                forward_model_log = {}
                if best_cost > current_training_loss:
                    forward_model_log["model saved"] = 1
                    forward_model_log["current_best_cost"] = best_cost
                    forward_model_log["current_epoch_cost"] = current_training_loss
                    best_cost = current_training_loss
                    del (
                        best_pressure,
                        best_saturation,
                        best_gas,
                        best_peacemann,
                        best_oil,
                    )
                    best_pressure = copy.deepcopy(surrogate_pressure)
                    best_gas = copy.deepcopy(surrogate_gas)
                    best_peacemann = copy.deepcopy(surrogate_peacemann)
                    best_saturation = copy.deepcopy(surrogate_saturation)
                    best_oil = copy.deepcopy(surrogate_oil)
                else:
                    forward_model_log["model NOT saved"] = 0
                    forward_model_log["current_best_cost"] = best_cost
                    forward_model_log["current_epoch_cost"] = current_training_loss
                log.log_epoch(forward_model_log)
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.distributed.barrier()
        if (epoch % 500 == 0 or epoch == 1) and dist.rank == 0:
            logger.info(f"🔥 Saving all models at epoch {epoch}...")
            if "PRESSURE" in output_variables:
                if cfg.custom.fno_type == "PINO":
                    save_model_to_buffer(
                        best_pressure,
                        "../MODELS/PINO/checkpoints_pressure/pino_pressure_forward_model.pth",
                    )
                else:
                    save_model_to_buffer(
                        best_pressure,
                        "../MODELS/FNO/checkpoints_pressure/fno_pressure_forward_model.pth",
                    )
            if "SGAS" in output_variables:
                if cfg.custom.fno_type == "PINO":
                    save_model_to_buffer(
                        best_gas,
                        "../MODELS/PINO/checkpoints_gas/pino_gas_forward_model.pth",
                    )
                else:
                    save_model_to_buffer(
                        best_gas,
                        "../MODELS/FNO/checkpoints_gas/fno_gas_forward_model.pth",
                    )
            if "SWAT" in output_variables:
                if cfg.custom.fno_type == "PINO":
                    save_model_to_buffer(
                        best_saturation,
                        "../MODELS/PINO/checkpoints_saturation/pino_saturation_forward_model.pth",
                    )
                else:
                    save_model_to_buffer(
                        best_saturation,
                        "../MODELS/FNO/checkpoints_saturation/fno_saturation_forward_model.pth",
                    )
            if "SOIL" in output_variables:
                if cfg.custom.fno_type == "PINO":
                    save_model_to_buffer(
                        best_oil,
                        "../MODELS/PINO/checkpoints_oil/pino_oil_forward_model.pth",
                    )
                else:
                    save_model_to_buffer(
                        best_oil,
                        "../MODELS/FNO/checkpoints_oil/fno_oil_forward_model.pth",
                    )
            if cfg.custom.fno_type == "PINO":
                save_model_to_buffer(
                    best_peacemann,
                    "../MODELS/PINO/checkpoints_peacemann/pino_peacemann_forward_model.pth",
                )
            else:
                save_model_to_buffer(
                    best_peacemann,
                    "../MODELS/FNO/checkpoints_peacemann/fno_peacemann_forward_model.pth",
                )
            if cfg.custom.fno_type == "FNO":
                if "PRESSURE" in output_variables:
                    torch.save(
                        {
                            "surrogate_pressure_state_dict": surrogate_pressure.state_dict(),
                            "optimizer_state_dict": optimizer_pressure.state_dict(),
                            "scheduler_state_dict": scheduler_pressure.state_dict(),
                            "epoch": epoch,  # Save the epoch in the checkpoint data as well
                        },
                        "../MODELS/FNO/checkpoints_pressure/checkpoint.pth",
                    )  # Attach epoch to filename
                if "SGAS" in output_variables:
                    torch.save(
                        {
                            "surrogate_gas_state_dict": surrogate_gas.state_dict(),
                            "optimizer_state_dict": optimizer_gas.state_dict(),
                            "scheduler_state_dict": scheduler_gas.state_dict(),
                            "epoch": epoch,  # Save the epoch in the checkpoint data as well
                        },
                        "../MODELS/FNO/checkpoints_gas/checkpoint.pth",
                    )  # Attach epoch to filename
                if "SWAT" in output_variables:
                    torch.save(
                        {
                            "surrogate_saturation_state_dict": surrogate_saturation.state_dict(),
                            "optimizer_state_dict": optimizer_saturation.state_dict(),
                            "scheduler_state_dict": scheduler_saturation.state_dict(),
                            "epoch": epoch,  # Save the epoch in the checkpoint data as well
                        },
                        "../MODELS/FNO/checkpoints_saturation/checkpoint.pth",
                    )  # Attach epoch to filename
                if "SOIL" in output_variables:
                    torch.save(
                        {
                            "surrogate_oil_state_dict": surrogate_oil.state_dict(),
                            "optimizer_state_dict": optimizer_oil.state_dict(),
                            "scheduler_state_dict": scheduler_oil.state_dict(),
                            "epoch": epoch,  # Save the epoch in the checkpoint data as well
                        },
                        "../MODELS/FNO/checkpoints_oil/checkpoint.pth",
                    )  # Attach epoch to filename
                torch.save(
                    {
                        "surrogate_peacemann_state_dict": surrogate_peacemann.state_dict(),
                        "optimizer_state_dict": optimizer_peacemann.state_dict(),
                        "scheduler_state_dict": scheduler_peacemann.state_dict(),
                        "epoch": epoch,  # Save the epoch in the checkpoint data as well
                    },
                    "../MODELS/FNO/checkpoints_peacemann/checkpoint.pth",
                )  # Attach epoch to filename
            else:
                if "PRESSURE" in output_variables:
                    torch.save(
                        {
                            "surrogate_pressure_state_dict": surrogate_pressure.state_dict(),
                            "optimizer_state_dict": optimizer_pressure.state_dict(),
                            "scheduler_state_dict": scheduler_pressure.state_dict(),
                            "epoch": epoch,  # Save the epoch in the checkpoint data as well
                        },
                        "../MODELS/PINO/checkpoints_pressure/checkpoint.pth",
                    )  # Attach epoch to filename
                if "SGAS" in output_variables:
                    torch.save(
                        {
                            "surrogate_gas_state_dict": surrogate_gas.state_dict(),
                            "optimizer_state_dict": optimizer_gas.state_dict(),
                            "scheduler_state_dict": scheduler_gas.state_dict(),
                            "epoch": epoch,  # Save the epoch in the checkpoint data as well
                        },
                        "../MODELS/PINO/checkpoints_gas/checkpoint.pth",
                    )  # Attach epoch to filename
                if "SWAT" in output_variables:
                    torch.save(
                        {
                            "surrogate_saturation_state_dict": surrogate_saturation.state_dict(),
                            "optimizer_state_dict": optimizer_saturation.state_dict(),
                            "scheduler_state_dict": scheduler_saturation.state_dict(),
                            "epoch": epoch,  # Save the epoch in the checkpoint data as well
                        },
                        "../MODELS/PINO/checkpoints_saturation/checkpoint.pth",
                    )  # Attach epoch to filename
                if "SOIL" in output_variables:
                    torch.save(
                        {
                            "surrogate_oil_state_dict": surrogate_oil.state_dict(),
                            "optimizer_state_dict": optimizer_oil.state_dict(),
                            "scheduler_state_dict": scheduler_saturation.state_dict(),
                            "epoch": epoch,  # Save the epoch in the checkpoint data as well
                        },
                        "../MODELS/PINO/checkpoints_oil/checkpoint.pth",
                    )  # Attach epoch to filename
                torch.save(
                    {
                        "surrogate_peacemann_state_dict": surrogate_peacemann.state_dict(),
                        "optimizer_state_dict": optimizer_peacemann.state_dict(),
                        "scheduler_state_dict": scheduler_peacemann.state_dict(),
                        "epoch": epoch,  # Save the epoch in the checkpoint data as well
                    },
                    "../MODELS/PINO/checkpoints_peacemann/checkpoint.pth",
                )  # Attach epoch to filename
