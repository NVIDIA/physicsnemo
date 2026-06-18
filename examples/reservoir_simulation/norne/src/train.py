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

"""
=====================================================================
 NVIDIA PHYSICSNEMO SURROGATE RESERVOIR SIMULATION FORWARD MODELLING
 (SEQUENTIAL PROCESSING VERSION)
=====================================================================

This module implements sequential processing for reservoir simulation forward modelling
using NVIDIA PhysicsNeMo. It provides a machine learning framework for predicting
reservoir behavior using neural networks with sequential processing capabilities.

Key Features:
- Sequential processing for reservoir simulations
- Multi-GPU support for distributed training
- Neural network models for pressure and saturation prediction
- Comprehensive model evaluation and visualization
- Integration with MLflow for experiment tracking

Usage:
    python Forward_problem.py --config-path=conf --config-name=DECK_CONFIG

Inputs:
    - Configuration file with model parameters
    - Training data from reservoir simulations
    - Test data for model evaluation

Outputs:
    - Trained neural network models
    - Prediction results with evaluation metrics
    - Visualization plots for model performance

@Author : Clement Etienam
"""

import os
import sys
import getpass
import copy
import time
import pickle
import logging
import warnings
import multiprocessing
from datetime import timedelta
from pathlib import Path

import gzip
import scipy.io as sio
import numpy as np
from omegaconf import DictConfig
from cpuinfo import get_cpu_info
from filelock import FileLock
import torch
import hydra
from hydra.utils import to_absolute_path
import mlflow
import mlflow.tracking

from physicsnemo.launch.logging import (
    LaunchLogger,
    PythonLogger,
)
from physicsnemo.launch.logging.mlflow import initialize_mlflow
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureEvaluateNoGrad

from forward.binaries_extract import (
    Black_oil_seq,
    train_polynomial_models,
)

from forward.gradients_extract import (
    loss_func,
    combined_loss,
    loss_func_physics,
    Black_oil_peacemann,
)

from forward.machine_extract import (
    InitializeLoggers,
    check_and_remove_dirs,
)
from forward.simulator import (
    simulation_data_types,
)
from forward.utils.sequential.seq_misc_operation_1 import load_and_setup_training_data
from data_extract.opm_extract_rates import read_compdats2
from compare.sequential.misc_gather import read_compdats
from forward.gradients_extract import clip_and_convert_to_float32
from forward.utils.sequential.training_function import run_training_loop
from forward.utils.sequential.training_config import (
    DataLoaders,
    ModelKeys,
    NormParams,
    Optimizers,
    PhysicsParams,
    Schedulers,
    SurrogateModels,
    TrainingState,
)

torch.cuda.empty_cache()


def is_available() -> bool:
    """Check if NVIDIA GPU is available using native Python methods."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def setup_logging() -> logging.Logger:
    """Configure and return the main logger with green INFO console output."""
    logger = logging.getLogger("forward problem")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on re-entry

    formatter = logging.Formatter(" %(asctime)s - %(levelname)s - %(message)s")

    # File handler — plain, no colors (keeps log file clean)
    f_handler = logging.FileHandler(filename="read_vectors.log", mode="w")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    # Console handler — colored
    class _ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: "\033[0;36m",  # cyan
            logging.INFO: "\033[0;32m",  # green
            logging.WARNING: "\033[1;33m",  # yellow
            logging.ERROR: "\033[0;31m",  # red
            logging.CRITICAL: "\033[1;31m",  # bold red
        }
        RESET = "\033[0m"

        def format(self, record):
            msg = super().format(record)
            if sys.stdout.isatty():
                color = self.COLORS.get(record.levelno, "")
                return f"{color}{msg}{self.RESET}"
            return msg

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(
        _ColorFormatter(" %(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(c_handler)

    return logger


def initialize_environment() -> tuple[bool, logging.Logger]:
    """Initialize the environment and return GPU availability and logger."""
    logger = setup_logging()

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Log PyTorch and CUDA information
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")

    # Check GPU availability
    gpu_available = is_available()
    if gpu_available:
        logger.info("GPU Available with CUDA")
    else:
        logger.info("No GPU Available")

    # Log CPU information
    cpu_info = get_cpu_info()
    logger.info("CPU Info:")
    for key, value in cpu_info.items():
        logger.info(f"\t{key}: {value}")

    warnings.filterwarnings("ignore")
    return gpu_available, logger


def training_step(
    model,
    inputin,
    inputin_p,
    TARGETS,
    cfg,
    device,
    input_keys,
    output_keys_saturation,
    steppi,
    output_variables,
    training_step_metrics,
    neededM,
    neededMx,
    epoch,
    physics,
    norm,
):
    """Compute the combined training loss for one step across all output variables.

    Supports standard forward pass and unroll BPTT with optional truncated K-window.
    Accumulates pressure, water, oil, gas and Peacemann well losses.

    Parameters
    ----------
    model : torch.nn.Module
        Neural operator model (FNO or Transolver) to train.
    inputin : dict
        Dictionary of dynamic input tensors keyed by input variable name.
    inputin_p : dict
        Dictionary containing the static pressure input tensor under key ``'X'``.
    TARGETS : dict
        Dictionary of ground-truth output tensors keyed by output variable name.
    cfg : omegaconf.DictConfig
        Hydra configuration object with training, custom, and model settings.
    device : torch.device
        Device on which tensors and the model reside.
    input_keys : list of str
        Names of the model's input channels.
    output_keys_saturation : list of str
        Names of the saturation output channels (water, oil, gas).
    steppi : int
        Number of sequential timesteps in the simulation window.
    output_variables : list of str
        All output variable names expected from the model.
    training_step_metrics : dict
        Mutable metrics accumulator dict updated in-place with per-variable losses.
    neededM : numpy.ndarray
        Peacemann well model reference data for loss computation.
    neededMx : numpy.ndarray
        Additional Peacemann reference data for extended well loss computation.
    epoch : int
        Current training epoch number (used for BPTT window scheduling).
    physics : PhysicsParams
        Bundled physical constants, grid dimensions, and relative-permeability
        tables referenced by the black-oil PDE residuals.
    norm : NormParams
        Bundled normalisation bounds (target range, permeability, pressure,
        time, and rate scalers) used inside the physics residuals.

    Returns
    -------
    torch.Tensor
        Scalar combined training loss for the current step.
    """
    # Import helper functions locally to avoid circular imports
    from utils.array_utils import (
        Make_correct,
    )
    from compare.sequential.misc_forward_enact import (
        process_data,
        get_dyna,
    )

    # Unpack dataclasses into the local names used by the function body.
    nx = physics.nx
    ny = physics.ny
    nz = physics.nz
    chunk_size = nz
    N_pr = physics.N_pr
    lenwels = physics.lenwels
    UO, BO, UW, BW = physics.UO, physics.BO, physics.UW, physics.BW
    DZ, RE = physics.DZ, physics.RE
    p_bub, p_atm, CFO = physics.p_bub, physics.p_atm, physics.CFO
    SWI, SWR = physics.SWI, physics.SWR
    SWOW, SWOG = physics.SWOW, physics.SWOG
    Relperm = physics.Relperm
    params = physics.params
    params1_swow, params2_swow = physics.params1_swow, physics.params2_swow
    params1_swog, params2_swog = physics.params1_swog, physics.params2_swog
    pde_method = physics.pde_method
    unique_entries = physics.unique_entries
    time_physics = physics.time_physics

    target_min, target_max = norm.target_min, norm.target_max
    minK, maxK = norm.minK, norm.maxK
    minP, maxP = norm.minP, norm.maxP
    maxT, maxQ, maxQw, maxQg = norm.maxT, norm.maxQ, norm.maxQw, norm.maxQg
    max_inn_fcn, max_out_fcn = norm.max_inn_fcn, norm.max_out_fcn
    max_inn_fcnx, max_out_fcnx = norm.max_inn_fcnx, norm.max_out_fcnx

    # n_cells for combined_loss rescaling — auto-adjusts to grid size
    n_cells = nz * nx * ny

    # Prepare input tensors
    if cfg.custom.unroll == "TRUE":
        cfg.training.max_steps = 1500
    input_tensor_p = inputin_p["X"]

    # Initialize accumulators
    loss = 0
    metrics_accumulator = {
        f"{var}_loss": 0.0 for var in ["pressure", "water", "oil", "gas", "peacemann"]
    }
    metrics_accumulator["peacemanned"] = 0.0

    if cfg.custom.fno_type == "PINO":
        pino_metrics = {
            "pressured": 0.0,
            "saturationd": 0.0,
            "gasd": 0.0,
            "peacemanned": 0.0,
        }

    # ---- K-step truncated BPTT config ----
    if cfg.custom.unroll == "TRUE":
        K = getattr(cfg.custom, "K_unroll", steppi)
        if K < 1:
            K = 1
        if steppi < K:
            K = steppi
        loss_value = 0.0

    all_pressure = []
    all_water = []
    all_oil = []
    all_gas = []

    if cfg.custom.unroll == "TRUE":
        if cfg.custom.unroll_cost == "AUTO":
            predictions_prev = None
            loss_window = 0.0

            for x in range(steppi):
                # --------- build per-timestep inputs ----------
                if x == 0:
                    inputin_t = {}
                    for k, v in inputin.items():
                        if isinstance(v, torch.Tensor) and v.dim() == 5:
                            inputin_t[k] = v[:, x : x + 1, ...]
                        else:
                            inputin_t[k] = v
                else:
                    inputin_t = {
                        "perm": inputin["perm"][:, x : x + 1, ...],
                        "poro": inputin["poro"][:, x : x + 1, ...],
                        "pini": predictions_prev["pressure"],
                        "sini": predictions_prev["water"],
                        "sgini": predictions_prev["gas"],
                        "soini": predictions_prev["oil"],
                        "fault": inputin["fault"][:, x : x + 1, ...],
                        "Q": inputin["Q"][:, x : x + 1, ...],
                        "Qg": inputin["Qg"][:, x : x + 1, ...],
                        "Qw": inputin["Qw"][:, x : x + 1, ...],
                        "dt": inputin["dt"][:, x : x + 1, ...],
                        "t": inputin["t"][:, x : x + 1, ...],
                    }

                # --------- build model input ----------
                if cfg.custom.model_type == "FNO":
                    tensors_ar = [
                        value
                        for value in inputin_t.values()
                        if isinstance(value, torch.Tensor)
                    ]
                    input_temp = torch.cat(tensors_ar, dim=1)
                else:
                    vars_for_cat = []
                    for key in input_keys:
                        t = inputin_t[key]
                        t = t.unsqueeze(-1)
                        vars_for_cat.append(t)
                    input_temp = torch.cat(vars_for_cat, dim=-1)

                target_chunks = {}
                step_loss = 0.0

                # --------- targets ----------
                if "PRESSURE" in output_variables:
                    target_chunks["pressure"] = {
                        "pressure": TARGETS["PRESSURE"]["pressure"][:, x : x + 1, ...]
                    }
                if "SWAT" in output_variables:
                    target_chunks["saturation"] = {
                        "water_sat": TARGETS["SATURATION"]["water_sat"][
                            :, x : x + 1, ...
                        ]
                    }
                if "SOIL" in output_variables:
                    target_chunks["oil"] = {
                        "oil_sat": TARGETS["OIL"]["oil_sat"][:, x : x + 1, ...]
                    }
                if "SGAS" in output_variables:
                    target_chunks["gas"] = {
                        "gas_sat": TARGETS["GAS"]["gas_sat"][:, x : x + 1, ...]
                    }

                # --------- model predictions ----------
                predictions = {}
                if "PRESSURE" in output_variables:
                    predictions["pressure"] = model(input_temp, mode="pressure")[
                        "pressure"
                    ]
                if "SGAS" in output_variables:
                    predictions["gas"] = model(input_temp, mode="gas")["gas"]
                if "SWAT" in output_variables:
                    predictions["water"] = model(input_temp, mode="saturation")[
                        "saturation"
                    ]
                if "SOIL" in output_variables:
                    predictions["oil"] = model(input_temp, mode="oil")["oil"]

                # --------- supervised losses (combined Sobolev + mean L^p, all heads) ----------
                if "PRESSURE" in output_variables:
                    pressure_loss = combined_loss(
                        target_chunks["pressure"]["pressure"],
                        predictions["pressure"],
                        weight=cfg.loss.weights.pressure,
                        n_cells=n_cells,
                        alpha=1.0,
                        beta=1.0,
                        p=2.0,
                    )
                    step_loss = step_loss + pressure_loss
                    metrics_accumulator["pressure_loss"] += pressure_loss.item()

                if "SWAT" in output_variables:
                    water_loss = combined_loss(
                        target_chunks["saturation"]["water_sat"],
                        predictions["water"],
                        weight=cfg.loss.weights.water_sat,
                        n_cells=n_cells,
                        alpha=1.0,
                        beta=1.0,
                        p=2.0,
                    )
                    step_loss = step_loss + water_loss
                    metrics_accumulator["water_loss"] += water_loss.item()

                if "SOIL" in output_variables:
                    oil_loss = combined_loss(
                        target_chunks["oil"]["oil_sat"],
                        predictions["oil"],
                        weight=cfg.loss.weights.oil_sat,
                        n_cells=n_cells,
                        alpha=1.0,
                        beta=1.0,
                        p=2.0,
                    )
                    step_loss = step_loss + oil_loss
                    metrics_accumulator["oil_loss"] += oil_loss.item()

                if "SGAS" in output_variables:
                    gas_loss = combined_loss(
                        target_chunks["gas"]["gas_sat"],
                        predictions["gas"],
                        weight=cfg.loss.weights.gas_sat,
                        n_cells=n_cells,
                        alpha=1.0,
                        beta=1.0,
                        p=2.0,
                    )
                    step_loss = step_loss + gas_loss
                    metrics_accumulator["gas_loss"] += gas_loss.item()

                # --------- PINO physics loss ----------
                if (
                    cfg.custom.fno_type == "PINO"
                    and epoch % max(1, int(0.01 * cfg.training.max_steps)) == 0
                ):
                    input_varr = {
                        **dict(inputin_t.items()),
                        "pressure": predictions.get("pressure"),
                        "water_sat": predictions.get("water"),
                        "gas_sat": predictions.get("gas"),
                        "oil_sat": predictions.get("oil"),
                    }

                    evaluate = Black_oil_seq(
                        input_varr,
                        neededM,
                        SWI,
                        SWR,
                        UW,
                        BW,
                        UO,
                        BO,
                        nx,
                        ny,
                        chunk_size,
                        SWOW,
                        SWOG,
                        target_min,
                        target_max,
                        minK,
                        maxK,
                        minP,
                        maxP,
                        p_bub,
                        p_atm,
                        CFO,
                        Relperm,
                        params,
                        pde_method,
                        RE,
                        max_inn_fcn,
                        max_out_fcn,
                        DZ,
                        device,
                        params1_swow,
                        params2_swow,
                        params1_swog,
                        params2_swog,
                        maxQw,
                        maxQg,
                        maxQ,
                        maxT,
                    )

                    f_pressure2 = loss_func_physics(
                        evaluate["pressured"], cfg.loss.weights.pressured
                    )
                    f_water2 = loss_func_physics(
                        evaluate["saturationd"], cfg.loss.weights.saturationd
                    )
                    f_gas2 = loss_func_physics(evaluate["gasd"], cfg.loss.weights.gasd)

                    step_loss = step_loss + f_pressure2 + f_water2 + f_gas2
                    pino_metrics["pressured"] += f_pressure2.item()
                    pino_metrics["saturationd"] += f_water2.item()
                    pino_metrics["gasd"] += f_gas2.item()

                # --------- nan / inf guard ----------
                if torch.isnan(step_loss) or torch.isinf(step_loss):
                    predictions_prev = {k: v.detach() for k, v in predictions.items()}
                    all_pressure.append(predictions_prev["pressure"])
                    all_water.append(predictions_prev["water"])
                    all_oil.append(predictions_prev["oil"])
                    all_gas.append(predictions_prev["gas"])
                    loss_window = torch.tensor(0.0, device=device)
                    continue

                # --------- accumulate into K-window and backward ----------
                loss_window = loss_window + step_loss
                is_window_end = ((x + 1) % K == 0) or (x == steppi - 1)

                if is_window_end:
                    (loss_window).backward()
                    loss_value += loss_window.detach().item()
                    loss_window = 0.0
                    predictions_prev = {k: v.detach() for k, v in predictions.items()}
                else:
                    predictions_prev = predictions.copy()

                all_pressure.append(predictions_prev["pressure"])
                all_water.append(predictions_prev["water"])
                all_oil.append(predictions_prev["oil"])
                all_gas.append(predictions_prev["gas"])

            if not isinstance(maxP, torch.Tensor):
                maxP = torch.as_tensor(maxP, dtype=torch.float32, device=device)
            if not isinstance(maxK, torch.Tensor):
                maxK = torch.as_tensor(maxK, dtype=torch.float32, device=device)
            pressure_all = Make_correct(torch.cat(all_pressure, dim=1)) * maxP
            water_all = Make_correct(torch.cat(all_water, dim=1))
            oil_all = Make_correct(torch.cat(all_oil, dim=1))
            gas_all = Make_correct(torch.cat(all_gas, dim=1))
            perm_all = Make_correct(inputin["perm"][:, 0:1, ...]) * maxK

        else:
            # ---- unroll_cost != "AUTO": K-window BPTT with AR logic ----
            predictions_prev = None
            loss_window = 0.0
            loss_autoregressive = 0.0

            for x in range(steppi):
                inputin_t = {}
                for k, v in inputin.items():
                    if isinstance(v, torch.Tensor) and v.dim() == 5:
                        inputin_t[k] = v[:, x : x + 1, ...]
                    else:
                        inputin_t[k] = v

                if cfg.custom.model_type == "FNO":
                    tensors = [
                        value
                        for value in inputin_t.values()
                        if isinstance(value, torch.Tensor)
                    ]
                    input_temp = torch.cat(tensors, dim=1)
                else:
                    vars_for_cat = []
                    for key in input_keys:
                        t = inputin_t[key]
                        t = t.unsqueeze(-1)
                        vars_for_cat.append(t)
                    input_temp = torch.cat(vars_for_cat, dim=-1)

                nz = input_temp.shape[2]
                chunk_size = nz
                num_chunks = 1

                target_chunks = {}
                step_loss = 0.0

                if "PRESSURE" in output_variables:
                    target_chunks["pressure"] = {
                        "pressure": TARGETS["PRESSURE"]["pressure"][:, x : x + 1, ...]
                    }
                if "SWAT" in output_variables:
                    target_chunks["saturation"] = {
                        "water_sat": TARGETS["SATURATION"]["water_sat"][
                            :, x : x + 1, ...
                        ]
                    }
                if "SOIL" in output_variables:
                    target_chunks["oil"] = {
                        "oil_sat": TARGETS["OIL"]["oil_sat"][:, x : x + 1, ...]
                    }
                if "SGAS" in output_variables:
                    target_chunks["gas"] = {
                        "gas_sat": TARGETS["GAS"]["gas_sat"][:, x : x + 1, ...]
                    }

                predictions = {}
                if "PRESSURE" in output_variables:
                    predictions["pressure"] = model(input_temp, mode="pressure")[
                        "pressure"
                    ]
                if "SGAS" in output_variables:
                    predictions["gas"] = model(input_temp, mode="gas")["gas"]
                if "SWAT" in output_variables:
                    predictions["water"] = model(input_temp, mode="saturation")[
                        "saturation"
                    ]
                if "SOIL" in output_variables:
                    predictions["oil"] = model(input_temp, mode="oil")["oil"]

                # Supervised losses (combined, all heads)
                if "PRESSURE" in output_variables:
                    pressure_loss = combined_loss(
                        target_chunks["pressure"]["pressure"],
                        predictions["pressure"],
                        weight=cfg.loss.weights.pressure,
                        n_cells=n_cells,
                        alpha=1.0,
                        beta=1.0,
                        p=2.0,
                    )
                    step_loss = step_loss + pressure_loss
                    metrics_accumulator["pressure_loss"] += pressure_loss.item()

                if "SWAT" in output_variables:
                    water_loss = combined_loss(
                        target_chunks["saturation"]["water_sat"],
                        predictions["water"],
                        weight=cfg.loss.weights.water_sat,
                        n_cells=n_cells,
                        alpha=1.0,
                        beta=1.0,
                        p=2.0,
                    )
                    step_loss = step_loss + water_loss
                    metrics_accumulator["water_loss"] += water_loss.item()

                if "SOIL" in output_variables:
                    oil_loss = combined_loss(
                        target_chunks["oil"]["oil_sat"],
                        predictions["oil"],
                        weight=cfg.loss.weights.oil_sat,
                        n_cells=n_cells,
                        alpha=1.0,
                        beta=1.0,
                        p=2.0,
                    )
                    step_loss = step_loss + oil_loss
                    metrics_accumulator["oil_loss"] += oil_loss.item()

                if "SGAS" in output_variables:
                    gas_loss = combined_loss(
                        target_chunks["gas"]["gas_sat"],
                        predictions["gas"],
                        weight=cfg.loss.weights.gas_sat,
                        n_cells=n_cells,
                        alpha=1.0,
                        beta=1.0,
                        p=2.0,
                    )
                    step_loss = step_loss + gas_loss
                    metrics_accumulator["gas_loss"] += gas_loss.item()

                # PINO physics loss
                if (
                    cfg.custom.fno_type == "PINO"
                    and epoch % max(1, int(0.01 * cfg.training.max_steps)) == 0
                ):
                    input_varr = {
                        **dict(inputin_t.items()),
                        "pressure": predictions.get("pressure"),
                        "water_sat": predictions.get("water"),
                        "gas_sat": predictions.get("gas"),
                        "oil_sat": predictions.get("oil"),
                    }

                    evaluate = Black_oil_seq(
                        input_varr,
                        neededM,
                        SWI,
                        SWR,
                        UW,
                        BW,
                        UO,
                        BO,
                        nx,
                        ny,
                        chunk_size,
                        SWOW,
                        SWOG,
                        target_min,
                        target_max,
                        minK,
                        maxK,
                        minP,
                        maxP,
                        p_bub,
                        p_atm,
                        CFO,
                        Relperm,
                        params,
                        pde_method,
                        RE,
                        max_inn_fcn,
                        max_out_fcn,
                        DZ,
                        device,
                        params1_swow,
                        params2_swow,
                        params1_swog,
                        params2_swog,
                        maxQw,
                        maxQg,
                        maxQ,
                        maxT,
                    )

                    f_pressure2 = loss_func_physics(
                        evaluate["pressured"], cfg.loss.weights.pressured
                    )
                    f_water2 = loss_func_physics(
                        evaluate["saturationd"], cfg.loss.weights.saturationd
                    )
                    f_gas2 = loss_func_physics(evaluate["gasd"], cfg.loss.weights.gasd)

                    step_loss = step_loss + f_pressure2 + f_water2 + f_gas2
                    pino_metrics["pressured"] += f_pressure2.item()
                    pino_metrics["saturationd"] += f_water2.item()
                    pino_metrics["gasd"] += f_gas2.item()

                # Autoregressive loss
                if x > 0 and predictions_prev is not None:
                    input_autoregressive = {
                        "perm": inputin["perm"][:, x : x + 1, ...],
                        "poro": inputin["poro"][:, x : x + 1, ...],
                        "pini": predictions_prev["pressure"],
                        "sini": predictions_prev["water"],
                        "sgini": predictions_prev["gas"],
                        "soini": predictions_prev["oil"],
                        "fault": inputin["fault"][:, x : x + 1, ...],
                        "Q": inputin["Q"][:, x : x + 1, ...],
                        "Qg": inputin["Qg"][:, x : x + 1, ...],
                        "Qw": inputin["Qw"][:, x : x + 1, ...],
                        "dt": inputin["dt"][:, x : x + 1, ...],
                        "t": inputin["t"][:, x : x + 1, ...],
                    }

                    tensors_ar = [
                        value
                        for value in input_autoregressive.values()
                        if isinstance(value, torch.Tensor)
                    ]
                    input_tensor_ar = torch.cat(tensors_ar, dim=1)

                    predictions_ar = {}
                    if "PRESSURE" in output_variables:
                        predictions_ar["pressure"] = model(
                            input_tensor_ar, mode="pressure"
                        )["pressure"]
                    if "SGAS" in output_variables:
                        predictions_ar["gas"] = model(input_tensor_ar, mode="gas")[
                            "gas"
                        ]
                    if "SWAT" in output_variables:
                        predictions_ar["water"] = model(
                            input_tensor_ar, mode="saturation"
                        )["saturation"]
                    if "SOIL" in output_variables:
                        predictions_ar["oil"] = model(input_tensor_ar, mode="oil")[
                            "oil"
                        ]

                    predictions = predictions_ar

                    autoregressive_timestep_loss = 0
                    if "PRESSURE" in output_variables:
                        pressure_loss_ar = combined_loss(
                            target_chunks["pressure"]["pressure"],
                            predictions["pressure"],
                            weight=cfg.loss.weights.pressure,
                            n_cells=n_cells,
                            alpha=1.0,
                            beta=1.0,
                            p=2.0,
                        )
                        autoregressive_timestep_loss += pressure_loss_ar
                        metrics_accumulator["pressure_loss"] += pressure_loss_ar.item()

                    if "SWAT" in output_variables:
                        water_loss_ar = combined_loss(
                            target_chunks["saturation"]["water_sat"],
                            predictions["water"],
                            weight=cfg.loss.weights.water_sat,
                            n_cells=n_cells,
                            alpha=1.0,
                            beta=1.0,
                            p=2.0,
                        )
                        autoregressive_timestep_loss += water_loss_ar
                        metrics_accumulator["water_loss"] += water_loss_ar.item()

                    if "SOIL" in output_variables:
                        oil_loss_ar = combined_loss(
                            target_chunks["oil"]["oil_sat"],
                            predictions["oil"],
                            weight=cfg.loss.weights.oil_sat,
                            n_cells=n_cells,
                            alpha=1.0,
                            beta=1.0,
                            p=2.0,
                        )
                        autoregressive_timestep_loss += oil_loss_ar
                        metrics_accumulator["oil_loss"] += oil_loss_ar.item()

                    if "SGAS" in output_variables:
                        gas_loss_ar = combined_loss(
                            target_chunks["gas"]["gas_sat"],
                            predictions["gas"],
                            weight=cfg.loss.weights.gas_sat,
                            n_cells=n_cells,
                            alpha=1.0,
                            beta=1.0,
                            p=2.0,
                        )
                        autoregressive_timestep_loss += gas_loss_ar
                        metrics_accumulator["gas_loss"] += gas_loss_ar.item()

                    loss_autoregressive += autoregressive_timestep_loss
                    step_loss = step_loss + (
                        autoregressive_timestep_loss
                        * cfg.loss.weights.get("autoregressive_weight", 0.1)
                    )

                # nan / inf guard
                if torch.isnan(step_loss) or torch.isinf(step_loss):
                    predictions_prev = {k: v.detach() for k, v in predictions.items()}
                    all_pressure.append(predictions_prev["pressure"])
                    all_water.append(predictions_prev["water"])
                    all_oil.append(predictions_prev["oil"])
                    all_gas.append(predictions_prev["gas"])
                    loss_window = torch.tensor(0.0, device=device)
                    continue

                loss_window = loss_window + step_loss
                is_window_end = ((x + 1) % K == 0) or (x == steppi - 1)

                if is_window_end:
                    (loss_window).backward()
                    loss_value += loss_window.detach().item()
                    loss_window = 0.0
                    predictions_prev = {k: v.detach() for k, v in predictions.items()}
                else:
                    predictions_prev = predictions.copy()

                all_pressure.append(predictions_prev["pressure"])
                all_water.append(predictions_prev["water"])
                all_oil.append(predictions_prev["oil"])
                all_gas.append(predictions_prev["gas"])

            if not isinstance(maxP, torch.Tensor):
                maxP = torch.as_tensor(maxP, dtype=torch.float32, device=device)
            if not isinstance(maxK, torch.Tensor):
                maxK = torch.as_tensor(maxK, dtype=torch.float32, device=device)
            pressure_all = Make_correct(torch.cat(all_pressure, dim=1)) * maxP
            water_all = Make_correct(torch.cat(all_water, dim=1))
            oil_all = Make_correct(torch.cat(all_oil, dim=1))
            gas_all = Make_correct(torch.cat(all_gas, dim=1))
            perm_all = Make_correct(inputin["perm"][:, 0:1, ...]) * maxK

    else:
        # ------------------ non-unroll branch ------------------
        if cfg.custom.model_type == "FNO":
            tensors = [
                value for value in inputin.values() if isinstance(value, torch.Tensor)
            ]
            input_tensor = torch.cat(tensors, dim=1)
        else:
            vars_for_cat = []
            for key in input_keys:
                t = inputin[key]
                t = t.unsqueeze(-1)
                vars_for_cat.append(t)
            input_tensor = torch.cat(vars_for_cat, dim=-1)

        nz = input_tensor.shape[2]
        chunk_size = nz
        num_chunks = 1
        input_temp = input_tensor
        target_chunks = {}

        if "PRESSURE" in output_variables:
            target_chunks["pressure"] = {"pressure": TARGETS["PRESSURE"]["pressure"]}
        if "SWAT" in output_variables:
            target_chunks["saturation"] = {
                "water_sat": TARGETS["SATURATION"]["water_sat"]
            }
        if "SOIL" in output_variables:
            target_chunks["oil"] = {"oil_sat": TARGETS["OIL"]["oil_sat"]}
        if "SGAS" in output_variables:
            target_chunks["gas"] = {"gas_sat": TARGETS["GAS"]["gas_sat"]}

        predictions = {}
        if "PRESSURE" in output_variables:
            predictions["pressure"] = model(input_temp, mode="pressure")["pressure"]
        if "SGAS" in output_variables:
            predictions["gas"] = model(input_temp, mode="gas")["gas"]
        if "SWAT" in output_variables:
            predictions["water"] = model(input_temp, mode="saturation")["saturation"]
        if "SOIL" in output_variables:
            predictions["oil"] = model(input_temp, mode="oil")["oil"]

        if "PRESSURE" in output_variables:
            pressure_loss = combined_loss(
                target_chunks["pressure"]["pressure"],
                predictions["pressure"],
                weight=cfg.loss.weights.pressure,
                n_cells=n_cells,
                alpha=1.0,
                beta=1.0,
                p=2.0,
            )
            loss += pressure_loss
            metrics_accumulator["pressure_loss"] += pressure_loss.item()

        if "SWAT" in output_variables:
            water_loss = combined_loss(
                target_chunks["saturation"]["water_sat"],
                predictions["water"],
                weight=cfg.loss.weights.water_sat,
                n_cells=n_cells,
                alpha=1.0,
                beta=1.0,
                p=2.0,
            )
            loss += water_loss
            metrics_accumulator["water_loss"] += water_loss.item()

        if "SOIL" in output_variables:
            oil_loss = combined_loss(
                target_chunks["oil"]["oil_sat"],
                predictions["oil"],
                weight=cfg.loss.weights.oil_sat,
                n_cells=n_cells,
                alpha=1.0,
                beta=1.0,
                p=2.0,
            )
            loss += oil_loss
            metrics_accumulator["oil_loss"] += oil_loss.item()

        if "SGAS" in output_variables:
            gas_loss = combined_loss(
                target_chunks["gas"]["gas_sat"],
                predictions["gas"],
                weight=cfg.loss.weights.gas_sat,
                n_cells=n_cells,
                alpha=1.0,
                beta=1.0,
                p=2.0,
            )
            loss += gas_loss
            metrics_accumulator["gas_loss"] += gas_loss.item()

        # PINO physics loss
        if (
            cfg.custom.fno_type == "PINO"
            and epoch % max(1, int(0.01 * cfg.training.max_steps)) == 0
        ):
            input_varr = {
                **dict(inputin.items()),
                "pressure": predictions.get("pressure"),
                "water_sat": predictions.get("water"),
                "gas_sat": predictions.get("gas"),
                "oil_sat": predictions.get("oil"),
            }

            evaluate = Black_oil_seq(
                input_varr,
                neededM,
                SWI,
                SWR,
                UW,
                BW,
                UO,
                BO,
                nx,
                ny,
                chunk_size,
                SWOW,
                SWOG,
                target_min,
                target_max,
                minK,
                maxK,
                minP,
                maxP,
                p_bub,
                p_atm,
                CFO,
                Relperm,
                params,
                pde_method,
                RE,
                max_inn_fcn,
                max_out_fcn,
                DZ,
                device,
                params1_swow,
                params2_swow,
                params1_swog,
                params2_swog,
                maxQw,
                maxQg,
                maxQ,
                maxT,
            )

            f_pressure2 = loss_func_physics(
                evaluate["pressured"], cfg.loss.weights.pressured
            )
            f_water2 = loss_func_physics(
                evaluate["saturationd"], cfg.loss.weights.saturationd
            )
            f_gas2 = loss_func_physics(evaluate["gasd"], cfg.loss.weights.gasd)

            loss += f_pressure2 + f_water2 + f_gas2
            pino_metrics["pressured"] += f_pressure2.item()
            pino_metrics["saturationd"] += f_water2.item()
            pino_metrics["gasd"] += f_gas2.item()

    # ---- scale / aggregate loss for logging ----
    if cfg.custom.unroll == "TRUE":
        loss = loss_value / steppi

    outputs_p = model(input_tensor_p, mode="peacemann")
    peacemann_pred = outputs_p["peacemann"]
    target_peacemann = {
        "Y": TARGETS.get("PEACEMANN", {}).get("Y", torch.zeros_like(peacemann_pred))
    }

    # ---- Peacemann head ----
    if cfg.custom.unroll == "TRUE":
        peacemann_loss_true = loss_func(
            peacemann_pred,
            target_peacemann["Y"],
            "peaceman",
            cfg.loss.weights.Y,
            p=2.0,
        )

        def to_torch(arr):
            return torch.from_numpy(arr).to(
                device, dtype=torch.float32, non_blocking=True
            )

        def _to_np(x):
            return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

        perm_all_np = _to_np(perm_all)
        pressure_all_np = _to_np(pressure_all)
        gas_all_np = _to_np(gas_all)
        water_all_np = _to_np(water_all)
        oil_all_np = _to_np(oil_all)
        time_physics_np = _to_np(time_physics)

        innn = np.zeros(
            (perm_all_np.shape[0], steppi, (N_pr * 4) + 2), dtype=np.float32
        )
        well_indices = process_data(unique_entries)

        for i in range(perm_all_np.shape[0]):
            permuse = perm_all_np[i, 0, :, :, :]
            mean_big = []
            for indices_list in well_indices.values():
                values = [
                    permuse[i_idx, j_idx, k_idx]
                    if k_idx == l_idx
                    else permuse[i_idx, j_idx, k_idx : l_idx + 1]
                    for i_idx, j_idx, k_idx, l_idx in indices_list
                ]
                mean_big.append(np.mean(values))
            permxx = np.tile(mean_big, (steppi, 1))
            presure_use = pressure_all_np[i, :, :, :, :]
            gas_use = gas_all_np[i, :, :, :, :]
            water_use = water_all_np[i, :, :, :, :]
            oil_use = oil_all_np[i, :, :, :, :]
            Time_usee = time_physics_np[0, :, :, :, :]
            a3 = get_dyna(steppi, well_indices, water_use)
            a2 = get_dyna(steppi, well_indices, gas_use)
            a5 = get_dyna(steppi, well_indices, oil_use)
            a1 = np.zeros((steppi, 1))
            a4 = np.zeros((steppi, 1))
            for k in range(steppi):
                uniep = presure_use[k, :, :, :]
                permuse = uniep
                a1[k, 0] = np.mean(permuse)
                a4[k, 0] = Time_usee[k, :, :, :][0, 0, 0]
            inn1 = np.hstack((permxx, a1, a5, a2, a3, a4))
            innn[i, :, :] = inn1

        inn_t = to_torch(innn).transpose(1, 2).contiguous()

        if not isinstance(max_inn_fcn, torch.Tensor):
            max_inn_fcn = torch.as_tensor(
                max_inn_fcn, dtype=torch.float32, device=device
            )
        inn = inn_t / max_inn_fcn

        peacemann_pred_prev = model(inn, mode="peacemann")["peacemann"]
        peacemann_loss_prev = loss_func(
            peacemann_pred_prev,
            target_peacemann["Y"],
            "peaceman",
            cfg.loss.weights.Y,
            p=2.0,
        )
        W1, W2 = 0.5, 0.5
        metrics_accumulator["peacemann_loss"] = (
            W1 * peacemann_loss_true.item() + W2 * peacemann_loss_prev.item()
        )
        peacemann_loss = W1 * peacemann_loss_true + W2 * peacemann_loss_prev
    else:
        peacemann_loss = loss_func(
            peacemann_pred,
            target_peacemann["Y"],
            "peaceman",
            cfg.loss.weights.Y,
            p=2.0,
        )
        metrics_accumulator["peacemann_loss"] = peacemann_loss.item()

    if cfg.custom.unroll == "TRUE":
        peacemann_loss.backward()
        loss += peacemann_loss.item()
    else:
        loss += peacemann_loss

    # Peacemann physics
    if (
        cfg.custom.fno_type == "PINO"
        and epoch % max(1, int(0.01 * cfg.training.max_steps)) == 0
    ):
        inputs1 = {**inputin_p, "Y": peacemann_pred}
        evaluate = Black_oil_peacemann(
            inputs1,
            UO,
            BO,
            UW,
            BW,
            DZ,
            RE,
            device,
            max_inn_fcnx,
            max_out_fcnx,
            params,
            p_bub,
            p_atm,
            steppi,
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
        )

        f_peacemann2 = loss_func_physics(
            evaluate["peacemanned"], cfg.loss.weights.peacemanned
        )
        pino_metrics["peacemanned"] = f_peacemann2.item()

        if cfg.custom.unroll == "TRUE":
            f_peacemann2.backward()
            loss += f_peacemann2.item()
        else:
            loss += f_peacemann2

    # Average metrics
    denom = steppi if cfg.custom.unroll == "TRUE" else 1

    for key in metrics_accumulator:
        training_step_metrics[key] = metrics_accumulator[key] / (
            denom if key != "peacemann_loss" else 1
        )

    if cfg.custom.fno_type == "PINO":
        for key in pino_metrics:
            training_step_metrics[key] = pino_metrics[key] / (
                num_chunks if key != "peacemanned" else 1
            )

    if cfg.custom.unroll == "FALSE":
        loss.backward()
        loss = loss.detach().item()

    return loss


def _validation_step_impl(
    model,
    inputin,
    inputin_p,
    TARGETS,
    cfg,
    device,
    input_keys,
    output_keys_saturation,
    steppi,
    output_variables,
    neededM,
    neededMxt,
    val_step_metrics,
    physics,
    norm,
):
    # ``physics`` and ``norm`` are accepted for signature parity with
    # ``training_step`` (so the same call site can dispatch to either), but the
    # validation pass computes only the data-fit loss and does not need any of
    # the PDE-residual constants, so we deliberately do not unpack them here.

    """Compute the combined validation loss over all output variables.

    Mirror of ``training_step`` operating under ``torch.no_grad()``; accumulates
    pressure, water, oil, gas and Peacemann losses without gradient computation.
    Uses the same ``combined_loss`` (relative H^1 + mean L^p) as training so
    train/val curves are directly comparable.

    Parameters
    ----------
    model : torch.nn.Module
        Neural operator model (FNO or Transolver) in evaluation mode.
    inputin : dict
        Dictionary of dynamic input tensors keyed by input variable name.
    inputin_p : dict
        Dictionary containing the static pressure input tensor under key ``'X'``.
    TARGETS : dict
        Dictionary of ground-truth output tensors keyed by output variable name.
    cfg : omegaconf.DictConfig
        Hydra configuration object with training, custom, and model settings.
    device : torch.device
        Device on which tensors and the model reside.
    input_keys : list of str
        Names of the model's input channels.
    output_keys_saturation : list of str
        Names of the saturation output channels (water, oil, gas).
    steppi : int
        Number of sequential timesteps in the simulation window.
    output_variables : list of str
        All output variable names expected from the model.
    neededM : numpy.ndarray
        Peacemann well model reference data for loss computation.
    neededMxt : numpy.ndarray
        Additional Peacemann reference data for extended well loss computation.
    val_step_metrics : dict
        Mutable metrics accumulator dict updated in-place with per-variable losses.
    physics : PhysicsParams
        Bundled physical constants and grid dimensions. Used here only to read
        grid dims for the combined_loss n_cells rescaling.
    norm : NormParams
        Bundled normalisation bounds. Accepted for signature parity with
        ``training_step`` but not consumed in the validation pass.

    Returns
    -------
    torch.Tensor
        Scalar combined validation loss for the current step.
    """
    # Read grid dims from physics for n_cells rescaling. norm is unused in the
    # validation pass (no PDE residuals) but kept for call-site parity.
    nx = physics.nx
    ny = physics.ny
    nz = physics.nz
    n_cells = nz * nx * ny
    del norm

    # Prepare input tensors
    if cfg.custom.unroll == "TRUE":
        cfg.training.max_steps = 1500
    input_tensor_p = inputin_p["X"]

    # Initialize accumulators
    loss = 0
    metrics_accumulator = {
        f"{var}_loss": 0.0 for var in ["pressure", "water", "oil", "gas", "peacemann"]
    }

    if cfg.custom.unroll == "TRUE":
        for x in range(steppi):
            inputin_t = {}
            for k, v in inputin.items():
                if isinstance(v, torch.Tensor) and v.dim() == 5:
                    # v: (B, T, nz, nx, ny) -> take timestep x and keep dim=1
                    inputin_t[k] = v[:, x : x + 1, ...]
                else:
                    # static or already right shape
                    inputin_t[k] = v

            if cfg.custom.model_type == "FNO":
                tensors = [
                    value
                    for value in inputin_t.values()
                    if isinstance(value, torch.Tensor)
                ]
                input_tensor = torch.cat(tensors, dim=1)
            else:
                # === KEY FIX: build input_temp with channels LAST (B, steppi, nz, nx, ny, C) ===
                vars_for_cat = []
                for key in input_keys:
                    t = inputin_t[key]  # (B, steppi, nz, nx, ny)
                    t = t.unsqueeze(-1)  # (B, steppi, nz, nx, ny, 1)
                    vars_for_cat.append(t)

                # input_temp: (B, steppi, nz, nx, ny, C)
                input_tensor = torch.cat(vars_for_cat, dim=-1)

            # Extract chunks
            input_temp = input_tensor
            target_chunks = {}

            # Extract target chunks
            if "PRESSURE" in output_variables:
                target_chunks["pressure"] = {
                    "pressure": TARGETS["PRESSURE"]["pressure"][:, x : x + 1, ...]
                }
            if "SWAT" in output_variables:
                target_chunks["saturation"] = {
                    "water_sat": TARGETS["SATURATION"]["water_sat"][:, x : x + 1, ...]
                }
            if "SOIL" in output_variables:
                target_chunks["oil"] = {
                    "oil_sat": TARGETS["OIL"]["oil_sat"][:, x : x + 1, ...]
                }
            if "SGAS" in output_variables:
                target_chunks["gas"] = {
                    "gas_sat": TARGETS["GAS"]["gas_sat"][:, x : x + 1, ...]
                }

            # Model predictions
            predictions = {}
            if "PRESSURE" in output_variables:
                predictions["pressure"] = model(input_temp, mode="pressure")["pressure"]
            if "SGAS" in output_variables:
                predictions["gas"] = model(input_temp, mode="gas")["gas"]
            if "SWAT" in output_variables:
                predictions["water"] = model(input_temp, mode="saturation")[
                    "saturation"
                ]
            if "SOIL" in output_variables:
                predictions["oil"] = model(input_temp, mode="oil")["oil"]

            # Compute losses (combined Sobolev + mean L^p, all heads)
            if "PRESSURE" in output_variables:
                pressure_loss = combined_loss(
                    target_chunks["pressure"]["pressure"],
                    predictions["pressure"],
                    weight=cfg.loss.weights.pressure,
                    n_cells=n_cells,
                    alpha=1.0,
                    beta=1.0,
                    p=2.0,
                )
                loss += pressure_loss
                metrics_accumulator["pressure_loss"] += pressure_loss.item()

            if "SWAT" in output_variables:
                water_loss = combined_loss(
                    target_chunks["saturation"]["water_sat"],
                    predictions["water"],
                    weight=cfg.loss.weights.water_sat,
                    n_cells=n_cells,
                    alpha=1.0,
                    beta=1.0,
                    p=2.0,
                )
                loss += water_loss
                metrics_accumulator["water_loss"] += water_loss.item()

            if "SOIL" in output_variables:
                oil_loss = combined_loss(
                    target_chunks["oil"]["oil_sat"],
                    predictions["oil"],
                    weight=cfg.loss.weights.oil_sat,
                    n_cells=n_cells,
                    alpha=1.0,
                    beta=1.0,
                    p=2.0,
                )
                loss += oil_loss
                metrics_accumulator["oil_loss"] += oil_loss.item()

            if "SGAS" in output_variables:
                gas_loss = combined_loss(
                    target_chunks["gas"]["gas_sat"],
                    predictions["gas"],
                    weight=cfg.loss.weights.gas_sat,
                    n_cells=n_cells,
                    alpha=1.0,
                    beta=1.0,
                    p=2.0,
                )
                loss += gas_loss
                metrics_accumulator["gas_loss"] += gas_loss.item()

    else:
        # ------------------ non-unroll branch ------------------
        if cfg.custom.model_type == "FNO":
            tensors = [
                value for value in inputin.values() if isinstance(value, torch.Tensor)
            ]
            input_tensor = torch.cat(tensors, dim=1)
        else:
            # === KEY FIX: build input_temp with channels LAST (B, steppi, nz, nx, ny, C) ===
            vars_for_cat = []
            for key in input_keys:
                t = inputin[key]  # (B, steppi, nz, nx, ny)
                t = t.unsqueeze(-1)  # (B, steppi, nz, nx, ny, 1)
                vars_for_cat.append(t)

            # input_temp: (B, steppi, nz, nx, ny, C)
            input_tensor = torch.cat(vars_for_cat, dim=-1)

        # Extract chunks
        input_temp = input_tensor
        target_chunks = {}

        # Extract target chunks
        if "PRESSURE" in output_variables:
            target_chunks["pressure"] = {"pressure": TARGETS["PRESSURE"]["pressure"]}
        if "SWAT" in output_variables:
            target_chunks["saturation"] = {
                "water_sat": TARGETS["SATURATION"]["water_sat"]
            }
        if "SOIL" in output_variables:
            target_chunks["oil"] = {"oil_sat": TARGETS["OIL"]["oil_sat"]}
        if "SGAS" in output_variables:
            target_chunks["gas"] = {"gas_sat": TARGETS["GAS"]["gas_sat"]}

        # Model predictions
        predictions = {}
        if "PRESSURE" in output_variables:
            predictions["pressure"] = model(input_temp, mode="pressure")["pressure"]
        if "SGAS" in output_variables:
            predictions["gas"] = model(input_temp, mode="gas")["gas"]
        if "SWAT" in output_variables:
            predictions["water"] = model(input_temp, mode="saturation")["saturation"]
        if "SOIL" in output_variables:
            predictions["oil"] = model(input_temp, mode="oil")["oil"]

        # Compute losses (combined Sobolev + mean L^p, all heads)
        if "PRESSURE" in output_variables:
            pressure_loss = combined_loss(
                target_chunks["pressure"]["pressure"],
                predictions["pressure"],
                weight=cfg.loss.weights.pressure,
                n_cells=n_cells,
                alpha=1.0,
                beta=1.0,
                p=2.0,
            )
            loss += pressure_loss
            metrics_accumulator["pressure_loss"] += pressure_loss.item()

        if "SWAT" in output_variables:
            water_loss = combined_loss(
                target_chunks["saturation"]["water_sat"],
                predictions["water"],
                weight=cfg.loss.weights.water_sat,
                n_cells=n_cells,
                alpha=1.0,
                beta=1.0,
                p=2.0,
            )
            loss += water_loss
            metrics_accumulator["water_loss"] += water_loss.item()

        if "SOIL" in output_variables:
            oil_loss = combined_loss(
                target_chunks["oil"]["oil_sat"],
                predictions["oil"],
                weight=cfg.loss.weights.oil_sat,
                n_cells=n_cells,
                alpha=1.0,
                beta=1.0,
                p=2.0,
            )
            loss += oil_loss
            metrics_accumulator["oil_loss"] += oil_loss.item()

        if "SGAS" in output_variables:
            gas_loss = combined_loss(
                target_chunks["gas"]["gas_sat"],
                predictions["gas"],
                weight=cfg.loss.weights.gas_sat,
                n_cells=n_cells,
                alpha=1.0,
                beta=1.0,
                p=2.0,
            )
            loss += gas_loss
            metrics_accumulator["gas_loss"] += gas_loss.item()

    if cfg.custom.unroll == "TRUE":
        loss = loss / steppi

    # ---- Peacemann head ----
    outputs_p = model(input_tensor_p, mode="peacemann")
    peacemann_pred = outputs_p["peacemann"]
    target_peacemann = {
        "Y": TARGETS.get("PEACEMANN", {}).get("Y", torch.zeros_like(peacemann_pred))
    }

    peacemann_loss = loss_func(
        peacemann_pred, target_peacemann["Y"], "peaceman", cfg.loss.weights.Y, p=2.0
    )
    loss += peacemann_loss
    metrics_accumulator["peacemann_loss"] = peacemann_loss.item()

    denom = steppi if cfg.custom.unroll == "TRUE" else 1

    for key in metrics_accumulator:
        val_step_metrics[key] = metrics_accumulator[key] / (
            denom if key != "peacemann_loss" else 1
        )

    return loss


@hydra.main(version_base="1.2", config_path="conf", config_name="DECK_CONFIG")
def main(cfg: DictConfig) -> None:
    """Main function for batch forward problem solving."""
    _gpu_available, logger = initialize_environment()

    (
        _type_dict,
        _ecl_extensions,
        _dynamic_props,
        _ecl_vectors,
        _static_props,
        _SUPPORTED_DATA_TYPES,
    ) = simulation_data_types()

    sequences = ["pressure", "saturation", "oil", "gas", "peacemann"]
    model_type = "FNO" if cfg.custom.fno_type == "FNO" else "PINO"
    model_paths = [f"../MODELS/{model_type}/checkpoints_{seq}" for seq in sequences]
    checkpoint_dir = "checkpoints"

    if cfg.custom.model_Distributed == 1:
        # InitializeLoggers handles mlruns/ teardown+setup itself — don't touch it here.
        dist, logger = InitializeLoggers(cfg)
        if dist.rank == 0:
            base_dirs = ["__pycache__/", "../RUNS", "outputs/"]
            directories_to_check = [checkpoint_dir, *base_dirs]
            if cfg.custom.reset_models == "Yes":
                directories_to_check.extend(model_paths)
                logger.warning(
                    "reset_models=Yes — trained model checkpoints will be deleted."
                )
            check_and_remove_dirs(
                directories_to_check, cfg.custom.file_response, logger
            )
            logger.info(
                "|-----------------------------------------------------------------|"
            )
    else:
        base_dirs = ["__pycache__/", "../RUNS", "outputs/"]
        if cfg.custom.reset_mlruns == "Yes":
            base_dirs.append("mlruns")
            logger.warning("reset_mlruns=Yes — MLflow run history will be deleted.")
        directories_to_check = [checkpoint_dir, *base_dirs]
        if cfg.custom.reset_models == "Yes":
            directories_to_check.extend(model_paths)
            logger.warning(
                "reset_models=Yes — trained model checkpoints will be deleted."
            )
        check_and_remove_dirs(directories_to_check, cfg.custom.file_response, logger)
        logger.info(
            "|-----------------------------------------------------------------|"
        )

        DistributedManager.initialize()
        dist = DistributedManager()
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(dist.rank)
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(dist.rank % torch.cuda.device_count())
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            device_id = dist.rank % gpu_count
            torch.cuda.set_device(device_id)
            logger.info(
                f"Process {dist.rank} is using GPU {device_id}: {torch.cuda.get_device_name(device_id)}"
            )
        else:
            logger.info(f"Process {dist.rank} is using CPU")

        initialize_mlflow(
            experiment_name="PhysicsNeMo-Reservoir Batch Modelling",
            experiment_desc="PhysicsNeMo launch development",
            run_name="Reservoir batch forward modelling",
            run_desc="Reservoir batch forward modelling training",
            user_name=getpass.getuser(),
            mode="offline",
        )
        logger = PythonLogger(name=" PhysicsNeMo Reservoir_Characterisation")
        LaunchLogger.initialize(use_mlflow=cfg.use_mlflow)
    device = dist.device
    # ... rest of main unchanged
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                PHYNEMO RESERVOIR CHARACTERISATION:              |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        if cfg.custom.model_Distributed == 1:
            logger.info(
                "|-----------------------------------------------------------------|"
            )
            logger.info(
                "|                     MULTI GPU USAGE MODEL :                     |"
            )
            logger.info(
                "|-----------------------------------------------------------------|"
            )
        else:
            logger.info(
                "|-----------------------------------------------------------------|"
            )
            logger.info(
                "|                     SINGLE GPU USAGE MODEL :                    |"
            )
            logger.info(
                "|-----------------------------------------------------------------|"
            )
    os.getcwd()
    Relperm = int(cfg.custom.Relperm)
    # interest = cfg.custom.interest
    pde_method = int(cfg.custom.pde_method)
    params = {
        "k_rwmax": torch.tensor(0.3),
        "k_romax": torch.tensor(0.9),
        "k_rgmax": torch.tensor(0.8),
        "n": torch.tensor(2.0),
        "p": torch.tensor(2.0),
        "q": torch.tensor(2.0),
        "m": torch.tensor(2.0),
        "Swi": torch.tensor(0.1),
        "Sor": torch.tensor(0.2),
    }
    if cfg.custom.interest == "Yes":
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        folders_to_create = ["../RUNS", "../data"]
        if dist.rank == 0:
            if os.path.isfile(to_absolute_path("../data/conversions.mat")):
                os.remove(to_absolute_path("../data/conversions.mat"))
            for folder in folders_to_create:
                absolute_path = to_absolute_path(folder)
                lock_path = (
                    absolute_path + ".lock"
                )  # Use a lock file for synchronization
                with FileLock(lock_path):  # Only one process will create the directory
                    if Path(absolute_path).exists():
                        logger.info(f"Directory already exists: {absolute_path}")
                    else:
                        os.makedirs(absolute_path, exist_ok=True)
                        logger.info(f"Created directory: {absolute_path}")
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    if cfg.custom.model_type == "FNO":
        if cfg.custom.fno_type == "FNO":
            folders_to_create = [
                "../MODELS/FNO/checkpoints_saturation",
                "../MODELS/FNO/checkpoints_oil",
                "../MODELS/FNO/checkpoints_pressure",
                "../MODELS/FNO/checkpoints_gas",
                "../MODELS/FNO/checkpoints_peacemann",
            ]
        else:
            folders_to_create = [
                "../MODELS/PINO/checkpoints_saturation",
                "../MODELS/PINO/checkpoints_oil",
                "../MODELS/PINO/checkpoints_pressure",
                "../MODELS/PINO/checkpoints_gas",
                "../MODELS/PINO/checkpoints_peacemann",
            ]
    else:
        if cfg.custom.fno_type == "FNO":
            folders_to_create = [
                "../MODELS/TRANSOLVER/checkpoints_saturation",
                "../MODELS/TRANSOLVER/checkpoints_oil",
                "../MODELS/TRANSOLVER/checkpoints_pressure",
                "../MODELS/TRANSOLVER/checkpoints_gas",
                "../MODELS/TRANSOLVER/checkpoints_peacemann",
            ]
        else:
            folders_to_create = [
                "../MODELS/PI-TRANSOLVER/checkpoints_saturation",
                "../MODELS/PI-TRANSOLVER/checkpoints_oil",
                "../MODELS/PI-TRANSOLVER/checkpoints_pressure",
                "../MODELS/PI-TRANSOLVER/checkpoints_gas",
                "../MODELS/PI-TRANSOLVER/checkpoints_peacemann",
            ]
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        for folder in folders_to_create:
            absolute_path = to_absolute_path(folder)
            lock_path = absolute_path + ".lock"  # Use a lock file for synchronization
            with FileLock(lock_path):  # Only one process will create the directory
                if Path(absolute_path).exists():
                    logger.info(f"Directory already exists: {absolute_path}")
                else:
                    os.makedirs(absolute_path, exist_ok=True)
                    logger.info(f"Created directory: {absolute_path}")
        logger.info(
            "|-----------------------------------------------------------------|"
        )

    nx = cfg.custom.PROPS.nx
    ny = cfg.custom.PROPS.ny
    nz = cfg.custom.PROPS.nz
    file_path = to_absolute_path("../data/conversions.mat")
    file_exists = os.path.isfile(file_path)
    if file_exists:
        mat = sio.loadmat(file_path)
        steppi = int(mat["steppi"])
        # steppi_indices = mat["steppi_indices"].flatten()
        N_ens = int(mat["N_ens"])
    else:
        steppi = cfg.custom.steppi
        # steppi_indices = np.linspace(1, 164, steppi, dtype=int)
        N_ens = cfg.custom.ntrain
    logger.info(f"Rank {dist.rank}: steppi = {steppi}, N_ens = {N_ens}")
    # oldfolder2 = os.getcwd()
    sourc_dir = cfg.custom.file_location
    source_dir = to_absolute_path(sourc_dir)  # ('../simulator_data')
    effective = np.genfromtxt(Path(source_dir) / "actnum.out", dtype="float")
    effective_i = np.reshape(effective, (nx, ny, nz), "F")
    SWOW = torch.tensor(np.array(np.vstack(cfg.custom.WELLSPECS.SWOW), dtype=float)).to(
        device
    )
    SWOG = torch.tensor(np.array(np.vstack(cfg.custom.WELLSPECS.SWOG), dtype=float)).to(
        device
    )
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                 Learning the interpolation machines    :        |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    params1_swow, params2_swow = train_polynomial_models(SWOW, device)
    params1_swog, params2_swog = train_polynomial_models(SWOG, device)
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|      Converged  Learning the interpolation machines    :        |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    # Extract fluid properties
    BO = float(cfg.custom.PROPS.BO)
    BW = float(cfg.custom.PROPS.BW)
    UW = float(cfg.custom.PROPS.UW)
    UO = float(cfg.custom.PROPS.UO)
    SWI = float(cfg.custom.PROPS.SWI)
    SWR = float(cfg.custom.PROPS.SWR)
    SGINI = float(cfg.custom.PROPS.SG1)
    CFO = float(cfg.custom.PROPS.CFO)
    p_atm = float(cfg.custom.PROPS.PATM)
    p_bub = float(cfg.custom.PROPS.PB)
    SO1 = float(cfg.custom.PROPS.SO1)
    # Extract bounds
    DZ = torch.tensor(100).to(device)
    RE = torch.tensor(0.2 * 100).to(device)
    Truee1 = np.genfromtxt(Path(source_dir) / "rossmary.GRDECL", dtype="float")
    active_grid_flat = np.reshape(Truee1.T, (nx, ny, nz), "F")
    active_grid_flat = np.reshape(active_grid_flat, (-1, 1), "F")
    active_grid_flat = active_grid_flat * effective.reshape(-1, 1)
    if dist.rank == 0:
        navail = multiprocessing.cpu_count()
        logger.info(f"Available CPU cores: {navail}")
    njobs = max(1, multiprocessing.cpu_count() // 5)  # Ensure at least 1 core is used
    if dist.rank == 0:
        logger.info(f"Using {njobs} cores for parallel processing.")
    sourc_dir = cfg.custom.file_location
    source_dir = to_absolute_path(sourc_dir)  # ('../simulator_data')

    gas_injectors, producers, injectors = read_compdats2(
        to_absolute_path(cfg.custom.COMPLETIONS_DATA),
        to_absolute_path(cfg.custom.SUMMARY_DATA),
    )  # filename
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                         PRINT WELLS                           : |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info("gas injectors wells")
        logger.info(f"Gas injectors: {gas_injectors}")
        logger.info("producer well")
        logger.info(f"Producers: {producers}")
        logger.info("water injector well")
        logger.info(f"Injectors: {injectors}")
    well_measurements = cfg.custom.well_measurements
    lenwels = len(well_measurements)
    input_variables = cfg.custom.input_properties
    output_variables = cfg.custom.output_properties
    N_pr = len(producers)  # Number of producers
    well_names = [entry[-1] for entry in producers]  # Producer well names
    well_namesg = [entry[-1] for entry in gas_injectors]  # gas injectors well names
    well_namesw = [entry[-1] for entry in injectors]  # water injectors well names
    compdat_data = read_compdats(cfg.custom.COMPLETIONS_DATA, well_names)
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                         PRINT WELL NAMES                      : |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info("producer well names")
        logger.info(f"Producer well names: {well_names}")
        logger.info("gas injectors well names")
        logger.info(f"Gas injector well names: {well_namesg}")
        logger.info("water injector well names")
        logger.info(f"Water injector well names: {well_namesw}")
    try:
        with gzip.open(to_absolute_path("../data/static.pkl.gz"), "rb") as f2:
            mat = pickle.load(f2)

        with gzip.open(to_absolute_path("../data/time_train.pkl.gz"), "rb") as f1:
            time_physics = pickle.load(f1)

    except (pickle.PickleError, EOFError, FileNotFoundError) as e:
        logger.error(f"Error loading pickle file: {e}")
        raise
    X_data1 = mat
    for key, value in X_data1.items():
        if dist.rank == 0:
            logger.info(f"For key '{key}':")
            logger.info(f"\tContains inf: {np.isinf(value).any()}")
            logger.info(f"\tContains -inf: {np.isinf(-value).any()}")
            logger.info(f"\tContains NaN: {np.isnan(value).any()}")
            logger.info(
                "|-----------------------------------------------------------------|"
            )
    perm_ensemble = X_data1["ensemble"]
    poro_ensemble = X_data1["ensemblep"]
    perm_ensemble = clip_and_convert_to_float32(perm_ensemble)
    poro_ensemble = clip_and_convert_to_float32(poro_ensemble)
    SWI = torch.from_numpy(np.array(SWI)).to(device)
    SGINI = torch.from_numpy(np.array(SGINI)).to(device)
    SO1 = torch.from_numpy(np.array(SO1)).to(device)
    SWR = torch.from_numpy(np.array(SWR)).to(device)
    UW = torch.from_numpy(np.array(UW)).to(device)
    BW = torch.from_numpy(np.array(BW)).to(device)
    UO = torch.from_numpy(np.array(UO)).to(device)
    BO = torch.from_numpy(np.array(BO)).to(device)
    p_bub = torch.from_numpy(np.array(p_bub)).to(device)
    p_atm = torch.from_numpy(np.array(p_atm)).to(device)
    CFO = torch.from_numpy(np.array(CFO)).to(device)
    mat = sio.loadmat(to_absolute_path("../data/conversions.mat"))
    minK = mat["minK"]
    maxK = mat["maxK"]
    maxT = mat["maxT"]
    minP = mat["minP"]
    maxP = mat["maxP"]
    maxQw = mat["maxQW"]
    maxQg = mat["maxQg"]
    maxQ = mat["maxQ"]
    max_inn_fcn = mat["max_inn_fcn"]
    max_out_fcn = mat["max_out_fcn"]
    target_min = 0.01
    target_max = 1
    max_inn_fcnx = torch.from_numpy(max_inn_fcn).to(device)
    max_out_fcnx = torch.from_numpy(max_out_fcn).to(device)

    training_setup = load_and_setup_training_data(
        input_variables,
        output_variables,
        cfg,
        dist,
        N_ens,
        nx,
        ny,
        nz,
        steppi,
        maxP,
        N_pr,
        lenwels,
        effective_i,
    )
    labelled_loader_train = training_setup["labelled_loader_train"]
    labelled_loader_testt = training_setup["labelled_loader_testt"]
    labelled_loader_trainp = training_setup["labelled_loader_trainp"]
    labelled_loader_testtp = training_setup["labelled_loader_testtp"]
    composite_model = training_setup["composite_model"]
    training_setup["combined_optimizer"]
    input_keys = training_setup["input_keys"]
    input_keys_peacemann = training_setup["input_keys_peacemann"]
    output_keys_peacemann = training_setup["output_keys_peacemann"]
    output_keys_pressure = training_setup["output_keys_pressure"]
    output_keys_saturation = training_setup["output_keys_saturation"]
    output_keys_gas = training_setup["output_keys_gas"]
    output_keys_oil = training_setup["output_keys_oil"]
    use_epoch = training_setup["use_epoch"]
    neededM = training_setup["neededM"]
    neededMx = training_setup["neededMx"]
    neededMxt = training_setup["neededMxt"]
    MODELS = training_setup["MODELS"]
    MODELS_C = training_setup["MODELS_C"]
    SCHEDULER = training_setup["SCHEDULER"]

    if "PRESSURE" in output_variables:
        surrogate_pressure = MODELS["PRESSURE"]
        optimizer_pressure = MODELS_C["pressure"]
        scheduler_pressure = SCHEDULER["PRESSURE"]
    if "SGAS" in output_variables:
        surrogate_gas = MODELS["SGAS"]
        optimizer_gas = MODELS_C["gas"]
        scheduler_gas = SCHEDULER["SGAS"]
    if "SWAT" in output_variables:
        surrogate_saturation = MODELS["SATURATION"]
        optimizer_saturation = MODELS_C["saturation"]
        scheduler_saturation = SCHEDULER["SATURATION"]
    if "SOIL" in output_variables:
        surrogate_oil = MODELS["SOIL"]
        optimizer_oil = MODELS_C["oil"]
        scheduler_oil = SCHEDULER["SOIL"]
    surrogate_peacemann = MODELS["PEACEMANN"]
    optimizer_peacemann = MODELS_C["peacemann"]
    scheduler_peacemann = SCHEDULER["PEACEMANN"]

    validation_step = StaticCaptureEvaluateNoGrad(
        model=composite_model, logger=logger, use_amp=False, use_graphs=True
    )(_validation_step_impl)

    training_step_metrics = {}
    val_step_metrics = {}

    if "PRESSURE" in output_variables:
        best_pressure = copy.deepcopy(surrogate_pressure)
    if "SGAS" in output_variables:
        best_gas = copy.deepcopy(surrogate_gas)
    best_peacemann = copy.deepcopy(surrogate_peacemann)
    if "SWAT" in output_variables:
        best_saturation = copy.deepcopy(surrogate_saturation)
    if "SOIL" in output_variables:
        best_oil = copy.deepcopy(surrogate_oil)
    start_time = time.time()
    run_training_loop(
        dist=dist,
        logger=logger,
        cfg=cfg,
        mlflow=mlflow,
        use_epoch=use_epoch,
        pde_method=pde_method,
        models=SurrogateModels(
            composite_model=composite_model,
            surrogate_pressure=surrogate_pressure,
            surrogate_gas=surrogate_gas,
            surrogate_saturation=surrogate_saturation,
            surrogate_oil=surrogate_oil,
            surrogate_peacemann=surrogate_peacemann,
            best_pressure=best_pressure,
            best_gas=best_gas,
            best_saturation=best_saturation,
            best_oil=best_oil,
            best_peacemann=best_peacemann,
        ),
        loaders=DataLoaders(
            labelled_loader_train=labelled_loader_train,
            labelled_loader_trainp=labelled_loader_trainp,
            labelled_loader_testt=labelled_loader_testt,
            labelled_loader_testtp=labelled_loader_testtp,
        ),
        keys=ModelKeys(
            output_variables=output_variables,
            input_keys=input_keys,
            input_keys_peacemann=input_keys_peacemann,
            output_keys_pressure=output_keys_pressure,
            output_keys_gas=output_keys_gas,
            output_keys_saturation=output_keys_saturation,
            output_keys_oil=output_keys_oil,
            output_keys_peacemann=output_keys_peacemann,
        ),
        physics=PhysicsParams(
            nx=nx,
            ny=ny,
            nz=nz,
            steppi=steppi,
            N_pr=N_pr,
            lenwels=lenwels,
            neededM=neededM,
            neededMx=neededMx,
            neededMxt=neededMxt,
            UO=UO,
            BO=BO,
            UW=UW,
            BW=BW,
            DZ=DZ,
            RE=RE,
            params=params,
            p_bub=p_bub,
            p_atm=p_atm,
            CFO=CFO,
            Relperm=Relperm,
            SWI=SWI,
            SWR=SWR,
            SWOW=SWOW,
            SWOG=SWOG,
            params1_swow=params1_swow,
            params2_swow=params2_swow,
            params1_swog=params1_swog,
            params2_swog=params2_swog,
            pde_method=pde_method,
            unique_entries=compdat_data,
            time_physics=time_physics,
        ),
        norm=NormParams(
            max_inn_fcnx=max_inn_fcnx,
            max_out_fcnx=max_out_fcnx,
            max_inn_fcn=max_inn_fcn,
            max_out_fcn=max_out_fcn,
            target_min=target_min,
            target_max=target_max,
            minK=minK,
            maxK=maxK,
            minP=minP,
            maxP=maxP,
            maxT=maxT,
            maxQ=maxQ,
            maxQw=maxQw,
            maxQg=maxQg,
        ),
        optimizers=Optimizers(
            optimizer_pressure=optimizer_pressure,
            optimizer_saturation=optimizer_saturation,
            optimizer_oil=optimizer_oil,
            optimizer_gas=optimizer_gas,
            optimizer_peacemann=optimizer_peacemann,
        ),
        schedulers=Schedulers(
            scheduler_pressure=scheduler_pressure,
            scheduler_saturation=scheduler_saturation,
            scheduler_oil=scheduler_oil,
            scheduler_gas=scheduler_gas,
            scheduler_peacemann=scheduler_peacemann,
        ),
        state=TrainingState(
            training_step=training_step,
            validation_step=validation_step,
            training_step_metrics=training_step_metrics,
            val_step_metrics=val_step_metrics,
        ),
    )

    if dist.rank == 0:
        mlflow.end_run()
        text = "  Training Converged   "
        logger.info(text)
        logger.info("")
        elapsed_time_secs2 = time.time() - start_time
        msg = f"Reservoir Modelling training with Nvidia PhysicsNeMo took: {timedelta(seconds=round(elapsed_time_secs2))} secs (Wall clock time)"
        logger.info(msg)
        logger.info("")


if __name__ == "__main__":
    main()
