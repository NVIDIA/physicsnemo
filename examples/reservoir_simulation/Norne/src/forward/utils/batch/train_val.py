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
                        BATCH TRAINING AND VALIDATION
=====================================================================

This module provides training and validation functions for batch processing
in reservoir simulation forward modeling. It includes functions for model
training, validation, and performance evaluation.

Key Features:
- Training step implementation
- Validation step implementation
- Model performance evaluation
- Loss computation and optimization

Usage:
    from forward.utils.batch.train_val import (
        training_step,
        validation_step,
        setup_training_functions
    )

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import logging
import warnings
from typing import Dict, Tuple, List, Any, Optional, Union

# 🔧 Third-party Libraries
from cpuinfo import get_cpu_info

# 🔥 Torch & PhyNeMo
import torch
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad

# 📦 Local Modules
from imresize import imresize
from forward.gradients_extract import (
    dx,
    dy,
    dz,
    compute_differential,
    compute_second_differential,
    compute_gradient_3d,
    compute_second_order_gradient_3d,
    convert_back,
    rmsee,
    extra_loss,
    kmeans,
    compute_boundary_mask,
    compute_hamming_distance,
    process_tensor_sat,
    process_tensor,
    loss_func,
    loss_func_physics,
)
from forward.binaries_extract import (
    Black_oil2,
    Black_oil,
    interpolate_pytorch_gpu,
    linear_interp2D,
    replace_with_mean,
    process_and_print,
    normalize_tensors_adjusted,
    process_task,
    safe_mean_std,
)
from forward.machine_extract import (
    FNOModel,
    create_fno_model,
    run_gnn_model,
    create_spatial_coords,
    create_edge_features,
    create_spatial_graph,
    CompositeModel,
    CompositeOptimizer,
    create_gnn_model,
    GNNModel,
    safe_rmtree,
    save_model_to_buffer,
    write_buffers_to_disk,
    InitializeLoggers,
    are_models_equal,
    remove_ddp,
    broadcast_read,
    check_and_remove_dirs,
    on_rm_error,
)
from forward.simulator import (
    simulation_data_types,
    byte2str,
    get_world_size,
    str2byte,
    filter_fakes,
    EclArray,
    EclBinaryParser,
    is_valid_vector,
    get_shape,
    fast_gaussian,
    NorneInitialEnsemble,
    gaussian_with_variable_parameters,
    add_gnoise,
    adjust_variable_within_bounds,
    read_until_line,
    Reinvent,
    Add_marker,
    loss_compute_abs,
)


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# 🖥️ Detect GPU
def is_available() -> bool:
    """Check if NVIDIA GPU is available using subprocess."""
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


# Initialize environment
gpu_available = is_available()
use_gpu = 0 if gpu_available else 1

# 🖥️ Display CPU Info
logger = setup_logging()
cpu_info = get_cpu_info()
logger.info("CPU Info:")
for k, v in cpu_info.items():
    logger.info(f"\t{k}: {v}")

# 🚨 Suppress Warnings
warnings.filterwarnings("ignore")

# ====== DEFINE THESE VARIABLES FIRST ======
# You need to initialize these before using them in decorators
composite_model = None
combined_optimizer = None
logger = None
# ==========================================


@StaticCaptureTraining(
    model=composite_model,
    optim=combined_optimizer,
    logger=logger,
    use_amp=False,
    use_graphs=True,
)
def training_step(
    model,
    inputin,
    inputin_p,
    TARGETS,
    cfg,
    device,
    output_keys_saturation,
    steppi,
    output_variables,
    training_step_metrics,
    # Add missing parameters:
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
):
    tensors = [value for value in inputin.values() if isinstance(value, torch.Tensor)]
    input_tensor = torch.cat(tensors, dim=1)
    input_tensor_p = inputin_p["X"]  # Ensure it's a tensor, not a list
    if "PRESSURE" in output_variables:
        target_pressure = TARGETS["PRESSURE"]
        outputs = model(input_tensor, mode="pressure")
        pressure_pred = outputs["pressure"]
    if "SGAS" in output_variables:
        target_gas = TARGETS["GAS"]
        outputsg = model(input_tensor, mode="gas")
        gas_pred = outputsg["gas"]
    outputs_p = model(input_tensor_p, mode="peacemann")
    peacemann_pred = outputs_p["peacemann"]
    if "SWAT" in output_variables:
        target_saturation = TARGETS["SATURATION"]
        outputs = model(input_tensor, mode="saturation")
        water_pred = outputs["saturation"]
    if "SOIL" in output_variables:
        target_oil = TARGETS["OIL"]
        outputs = model(input_tensor, mode="oil")
        oil_pred = outputs["oil"]
    loss = 0
    if "PRESSURE" in output_variables:
        pressure_loss = loss_func(
            pressure_pred,
            target_pressure["pressure"],
            "eliptical",
            cfg.loss.weights.pressure,
            p=2.0,
        ) / len(inputin["perm"])
        loss += pressure_loss
    peacemann_loss = loss_func(
        peacemann_pred, target_peacemann["Y"], "peaceman", cfg.loss.weights.Y, p=2.0
    ) / len(inputin["perm"])
    loss += peacemann_loss
    if "SWAT" in output_variables:
        water_loss = loss_func(
            water_pred,
            target_saturation["water_sat"],
            "hyperbolic",
            cfg.loss.weights.water_sat,
            p=2.0,
        ) / len(inputin["perm"])
        loss += water_loss
    if "SOIL" in output_variables:
        oil_loss = loss_func(
            oil_pred,
            target_oil["oil_sat"],
            "hyperbolic",
            cfg.loss.weights.oil_sat,
            p=2.0,
        ) / len(inputin["perm"])
        loss += oil_loss
    if "SGAS" in output_variables:
        gass_loss = loss_func(
            gas_pred,
            target_gas["gas_sat"],
            "hyperbolic",
            cfg.loss.weights.gas_sat,
            p=2.0,
        ) / len(inputin["perm"])
        loss += gass_loss
    inputs1 = {
        **inputin_p,
        "Y": peacemann_pred,
    }
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
    ) / len(inputin["perm"])
    input_varr = {
        **inputin,
        "pressure": pressure_pred,
        "water_sat": water_pred,
        "gas_sat": gas_pred,
        "oil_sat": oil_pred,
    }
    loss += f_peacemann2  # + loss_pde3
    if "PRESSURE" in output_variables:
        training_step_metrics["pressure_loss"] = pressure_loss.item()
    if "SWAT" in output_variables:
        training_step_metrics["water_loss"] = water_loss.item()
    if "SOIL" in output_variables:
        training_step_metrics["oil_loss"] = oil_loss.item()
    if "SGAS" in output_variables:
        training_step_metrics["gas_loss"] = gass_loss.item()
    training_step_metrics["peacemann_loss"] = peacemann_loss.item()
    if cfg.custom.fno_type == "PINO":
        evaluate = Black_oil(
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
            nz,
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
        )
        f_pressure2 = loss_func_physics(
            evaluate["pressured"], cfg.loss.weights.pressured
        ) / len(inputin["perm"])
        f_water2 = loss_func_physics(
            evaluate["saturationd"], cfg.loss.weights.saturationd
        ) / len(inputin["perm"])
        f_gas2 = loss_func_physics(evaluate["gasd"], cfg.loss.weights.gasd) / len(
            inputin["perm"]
        )
        loss += f_pressure2 + f_water2 + f_gas2
        training_step_metrics["pressured"] = f_pressure2.item()
        training_step_metrics["saturationd"] = f_water2.item()
        training_step_metrics["gasd"] = f_gas2.item()
        training_step_metrics["peacemanned"] = f_peacemann2.item()
    return loss


@StaticCaptureEvaluateNoGrad(
    model=composite_model, logger=logger, use_amp=False, use_graphs=True
)
def validation_step(
    model,
    inputin,
    inputin_p,
    TARGETS,
    cfg,
    device,
    output_keys_saturation,
    steppi,
    output_variables,
    val_step_metrics,
):
    tensors = [value for value in inputin.values() if isinstance(value, torch.Tensor)]
    input_tensor = torch.cat(tensors, dim=1)
    input_tensor_p = inputin_p["X"]  # Ensure it's a tensor, not a list
    if "PRESSURE" in output_variables:
        target_pressure = TARGETS["PRESSURE"]
        outputs = model(input_tensor, mode="pressure")
        pressure_pred = outputs["pressure"]
    if "SGAS" in output_variables:
        target_gas = TARGETS["GAS"]
        outputsg = model(input_tensor, mode="gas")
        gas_pred = outputsg["gas"]
    outputs_p = model(input_tensor_p, mode="peacemann")
    peacemann_pred = outputs_p["peacemann"]

    # Predict water and oil saturation based on model type
    if "SWAT" in output_variables:
        target_saturation = TARGETS["SATURATION"]
        outputs = model(input_tensor, mode="saturation")
        water_pred = outputs["saturation"]
    if "SOIL" in output_variables:
        target_oil = TARGETS["OIL"]
        outputs = model(input_tensor, mode="oil")
        oil_pred = outputs["oil"]
    loss = 0
    if "PRESSURE" in output_variables:
        pressure_loss = loss_func(
            pressure_pred,
            target_pressure["pressure"],
            "eliptical",
            cfg.loss.weights.pressure,
            p=2.0,
        ) / len(inputin["perm"])
        loss = loss + pressure_loss
    peacemann_loss = loss_func(
        peacemann_pred, target_peacemann["Y"], "peaceman", cfg.loss.weights.Y, p=2.0
    ) / len(inputin["perm"])
    loss = loss + peacemann_loss
    if "SWAT" in output_variables:
        water_loss = loss_func(
            water_pred,
            target_saturation["water_sat"],
            "hyperbolic",
            cfg.loss.weights.water_sat,
            p=2.0,
        ) / len(inputin["perm"])
        loss = loss + water_loss
    if "SOIL" in output_variables:
        oil_loss = loss_func(
            oil_pred,
            target_oil["oil_sat"],
            "hyperbolic",
            cfg.loss.weights.oil_sat,
            p=2.0,
        ) / len(inputin["perm"])
        loss = loss + oil_loss
    if "SGAS" in output_variables:
        gass_loss = loss_func(
            gas_pred,
            target_gas["gas_sat"],
            "hyperbolic",
            cfg.loss.weights.oil_sat,
            p=2.0,
        ) / len(inputin["perm"])
        loss = loss + gass_loss
    if "PRESSURE" in output_variables:
        val_step_metrics["pressure_loss"] = pressure_loss.item()
    if "SWAT" in output_variables:
        val_step_metrics["water_loss"] = water_loss.item()
    if "SGAS" in output_variables:
        val_step_metrics["gas_loss"] = gass_loss.item()
    if "SOIL" in output_variables:
        val_step_metrics["oil_loss"] = oil_loss.item()
    val_step_metrics["peacemann_loss"] = peacemann_loss.item()
    return loss
