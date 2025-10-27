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
                    SEQUENTIAL MISC OPERATIONS
=====================================================================

This module provides sequential processing utilities for reservoir simulation
forward modeling. It includes functions for data loading, dataset creation,
and distributed training setup for sequential processing.

Key Features:
- Sequential data loading and processing
- Distributed training setup
- Dataset creation and management
- GPU detection and configuration

Usage:
    from forward.utils.sequential.seq_misc_operation_1 import (
        load_and_setup_training_data,
        create_dataset,
        setup_distributed_training
    )

@Author : Clement Etienam
"""

# 🛠 Standard Library
import pickle
import logging
import warnings
import gzip

# 🔧 Third-party Libraries
import numpy as np
import numpy.linalg
import numpy.matlib
from cpuinfo import get_cpu_info

# 🔥 Torch & PhyNeMo
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from hydra.utils import to_absolute_path
from physicsnemo.launch.utils import load_checkpoint

# 📊 MLFlow & Logging
import wandb

# 📦 Local Modules
from forward.machine_extract import (
    create_fno_model,
    CompositeModel,
    CompositeOptimizer,
)
from forward.gradients_extract import (
    clip_and_convert_to_float32,
    clip_and_convert_to_float3,
    replace_nans_and_infs,
    Labelledset,
)


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Forward problem")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
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


def create_param_groupss(model, weight_decay):
    """Create parameter groups for AdamW with selective weight decay.

    Splits parameters into decayed and non-decayed sets following best
    practices: biases, norm statistics, and embeddings are excluded from
    weight decay; higher-dimensional weights are decayed.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters will be grouped.
    weight_decay : float
        Weight decay value to apply to decayed parameters.

    Returns
    -------
    list[dict]
        Parameter groups compatible with AdamW optimizers.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Parameters that should NOT have weight decay:
        # - 1D parameters (biases, etc.)
        # - Normalization layers (batch norm, layer norm, etc.)
        # - Embedding layers
        if (
            param.ndim < 2
            or "bias" in name.lower()
            or "norm" in name.lower()
            or "bn" in name.lower()
            or "ln" in name.lower()
            or "embed" in name.lower()
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def load_and_setup_training_data(
    # Configuration
    input_variables,
    output_variables,
    cfg,
    dist,
    # Model parameters
    N_ens,
    nx,
    ny,
    nz,
    steppi,
    maxP,
    # Well configuration
    N_pr,
    lenwels,
    # Additional parameters
    effective_i,
):
    """Prepare datasets, dataloaders, models, and optimizers for training.

    Loads sequential training/test data, cleans and normalizes arrays,
    materializes input/output tensors per variable, sets up distributed
    samplers and data loaders, creates required FNO/PINO models (including
    the Peaceman mapping), optionally wraps them with DDP, configures
    optimizers and LR schedulers, restores from checkpoints when available,
    and returns a dictionary bundling all training artifacts.

    See the batch counterpart for a similar API; this function differs mainly
    in tensor layout/time-handling to suit sequential processing.

    Returns
    -------
    dict
        Aggregated training setup including tensors, dataloaders, models,
        optimizers, schedulers, I/O keys, and helper tensors (e.g., dt, flows).
    """
    device = dist.device
    if dist.rank == 0:
        logger.info(
            "---------------------------------------------------------------------"
        )
        logger.info("Load simulated labelled training data")
    with gzip.open(to_absolute_path("../data/data_train.pkl.gz"), "rb") as f2:
        mat = pickle.load(f2)
    X_data1 = mat
    for key, value in X_data1.items():
        if dist.rank == 0:
            logger.info(
                "---------------------------------------------------------------------"
            )
            logger.info(f"For key '{key}':")
            logger.info(f"\tContains inf: {np.isinf(value).any()}")
            logger.info(f"\tContains -inf: {np.isinf(-value).any()}")
            logger.info(f"\tContains NaN: {np.isnan(value).any()}")
    for key in X_data1.keys():
        X_data1[key][np.isnan(X_data1[key])] = 0.0
        X_data1[key][np.isinf(X_data1[key])] = 0.0
        X_data1[key] = clip_and_convert_to_float32(X_data1[key])
    unique_times = np.unique(X_data1["Time"])
    logger.info(f"Min time: {np.min(X_data1['Time'])}")
    logger.info(f"Max time: {np.max(X_data1['Time'])}")
    logger.info(f"Sorted unique time values: {unique_times}")
    time_values = np.array(
        [8.0, 252.0, 475.0, 637.0, 846.0, 1039.0, 1334.0, 1576.0, 1851.0, 2105.0]
    )

    # Compute safe normalizers
    def _safe_max(arr, default=1.0):
        try:
            m = float(np.nanmax(arr))
            return m if m and np.isfinite(m) and m > 0 else default
        except Exception:
            return default

    maxQ = _safe_max(X_data1.get("Q", 1.0))
    maxQw = _safe_max(X_data1.get("Qw", 1.0))
    maxQg = _safe_max(X_data1.get("Qg", 1.0))
    maxT = _safe_max(time_values)
    if "PERM" in input_variables:
        cPerm = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Permeability
    if "PORO" in input_variables:
        cPhi = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Porosity
    if "PINI" in input_variables:
        cPini = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Initial pressure
    if "SINI" in input_variables:
        cSini = np.zeros(
            (N_ens, 1, nz, nx, ny), dtype=np.float32
        )  # Initial water saturation
    if "FAULT" in input_variables:
        cfault = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Fault
    if "SGINI" in input_variables:
        cSgini = 1e-3 * np.ones(
            (N_ens, 1, nz, nx, ny), dtype=np.float32
        )  # Initial gas saturation
        cSgini = clip_and_convert_to_float3(cSgini)
    if "SOINI" in input_variables:
        cSoini = cfg.custom.PROPS.SO1 * np.ones(
            (N_ens, 1, nz, nx, ny), dtype=np.float32
        )  # Initial gas saturation
        cSoini = clip_and_convert_to_float3(cSoini)
    cQ = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)
    cQw = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)
    cQg = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)
    cQo = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)
    cTime = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)
    cactnum = np.zeros((1, 1, nz, nx, ny), dtype=np.float32)
    if "PRESSURE" in output_variables:
        cPress = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)  # Pressure
    if "SWAT" in output_variables:
        cSat = np.zeros(
            (N_ens, steppi, nz, nx, ny), dtype=np.float32
        )  # Water saturation
    if "SGAS" in output_variables:
        cSatg = np.zeros(
            (N_ens, steppi, nz, nx, ny), dtype=np.float32
        )  # gas saturation
    if "SOIL" in output_variables:
        cSato = np.zeros(
            (N_ens, steppi, nz, nx, ny), dtype=np.float32
        )  # oil saturation
    X_data1["Q"][X_data1["Q"] <= 0] = 0
    X_data1["Qw"][X_data1["Qw"] <= 0] = 0
    X_data1["Qg"][X_data1["Qg"] <= 0] = 0
    X_data1["Qo"][X_data1["Qo"] <= 0] = 0
    for kk in range(N_ens):
        for mv in range(steppi):
            for i in range(nz):
                X_data1["Q"][kk, mv, :, :, i] = np.where(
                    X_data1["Q"][kk, mv, :, :, i] < 0, 0, X_data1["Q"][kk, mv, :, :, i]
                )
                X_data1["Qw"][kk, mv, :, :, i] = np.where(
                    X_data1["Qw"][kk, mv, :, :, i] < 0,
                    0,
                    X_data1["Qw"][kk, mv, :, :, i],
                )
                X_data1["Qg"][kk, mv, :, :, i] = np.where(
                    X_data1["Qg"][kk, mv, :, :, i] < 0,
                    0,
                    X_data1["Qg"][kk, mv, :, :, i],
                )
                X_data1["Qo"][kk, mv, :, :, i] = np.where(
                    X_data1["Qo"][kk, mv, :, :, i] < 0,
                    0,
                    X_data1["Qo"][kk, mv, :, :, i],
                )
                cQ[kk, mv, i, :, :] = X_data1["Q"][kk, mv, :, :, i]
                cQw[kk, mv, i, :, :] = X_data1["Qw"][kk, mv, :, :, i]
                cQg[kk, mv, i, :, :] = X_data1["Qg"][kk, mv, :, :, i]
                cQo[kk, mv, i, :, :] = X_data1["Qo"][kk, mv, :, :, i]
            cactnum[0, 0, i, :, :] = X_data1["actnum"][0, 0, :, :, i]
            cTime[kk, :, :, :, :] = time_values[mv] * np.ones(
                (steppi, nz, nx, ny), dtype=np.float32
            )
    neededM = {
        "Q": torch.from_numpy(cQ[0:1, :, :, :, :]).to(device, torch.float32),
        "Qw": torch.from_numpy(cQw[0:1, :, :, :, :]).to(device, dtype=torch.float32),
        "Qg": torch.from_numpy(cQg[0:1, :, :, :, :]).to(device, dtype=torch.float32),
        "Qo": torch.from_numpy(cQo[0:1, :, :, :, :]).to(device, dtype=torch.float32),
        "actnum": torch.from_numpy(cactnum[0:1, :, :, :, :]).to(
            device, dtype=torch.float32
        ),
        "Time": torch.from_numpy(cTime[0:1, :, :, :, :]).to(
            device, dtype=torch.float32
        ),
    }
    for key in neededM:
        neededM[key] = replace_nans_and_infs(neededM[key])
    cQ = cQ / maxQ
    cQw = cQw / maxQw
    cQg = cQg / maxQg
    Timebefore = np.zeros((steppi))
    Timebefore[1:] = time_values[:-1]
    dT = time_values - Timebefore
    cdT = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)
    for kk in range(N_ens):
        for tt in range(steppi):
            cdT[kk, tt, :, :, :] = dT[tt] * np.ones((nz, nx, ny), dtype=np.float32)
    for kk in range(N_ens):
        for i in range(nz):
            if "permeability" in X_data1:
                cPerm[kk, 0, i, :, :] = clip_and_convert_to_float3(
                    X_data1["permeability"][kk, 0, :, :, i]
                )
            if "Fault" in X_data1:
                cfault[kk, 0, i, :, :] = clip_and_convert_to_float3(
                    X_data1["Fault"][kk, 0, :, :, i]
                )
            if "porosity" in X_data1:
                cPhi[kk, 0, i, :, :] = clip_and_convert_to_float3(
                    X_data1["porosity"][kk, 0, :, :, i]
                )
            if "Pini" in X_data1:
                cPini[kk, 0, i, :, :] = (
                    clip_and_convert_to_float3(X_data1["Pini"][kk, 0, :, :, i]) / maxP
                )
            if "Sini" in X_data1:
                cSini[kk, 0, i, :, :] = clip_and_convert_to_float3(
                    X_data1["Sini"][kk, 0, :, :, i]
                )
        for mv in range(steppi):
            for i in range(nz):
                if "Pressure" in X_data1:
                    cPress[kk, mv, i, :, :] = clip_and_convert_to_float3(
                        X_data1["Pressure"][kk, mv, :, :, i]
                    )
                if "Water_saturation" in X_data1:
                    cSat[kk, mv, i, :, :] = clip_and_convert_to_float3(
                        X_data1["Water_saturation"][kk, mv, :, :, i]
                    )
                if "Gas_saturation" in X_data1:
                    cSatg[kk, mv, i, :, :] = clip_and_convert_to_float3(
                        X_data1["Gas_saturation"][kk, mv, :, :, i]
                    )
                if "Oil_saturation" in X_data1:
                    cSato[kk, mv, i, :, :] = clip_and_convert_to_float3(
                        X_data1["Oil_saturation"][kk, mv, :, :, i]
                    )
    effec_abbi = np.zeros((1, 1, nz, nx, ny), dtype=np.float32)
    for i in range(nz):
        effec_abbi[:, :, i, :, :] = effective_i[:, :, i]  # Reorder dimensions
    cQ[cQ == 0] = 0.1
    cQw[cQw == 0] = 0.1
    cQg[cQg == 0] = 0.1
    cQo[cQo == 0] = 0.1
    cdT = cdT / maxT
    neededMx = {
        "Q": torch.from_numpy(cQ[0:1, :, :, :, :]).to(device, torch.float32),
        "Qw": torch.from_numpy(cQw[0:1, :, :, :, :]).to(device, dtype=torch.float32),
        "Qg": torch.from_numpy(cQg[0:1, :, :, :, :]).to(device, dtype=torch.float32),
        "Qo": torch.from_numpy(cQo[0:1, :, :, :, :]).to(device, dtype=torch.float32),
        "dt": torch.from_numpy(cdT[0:1, :, :, :, :]).to(device, dtype=torch.float32),
    }
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15 = [
        [] for _ in range(15)
    ]
    for yy in range(N_ens):
        temp1 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # permeability
        temp2 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # porosity
        temp3 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # Initial pressure
        temp4 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # Initial water sat
        temp5 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # Initial gas sat
        temp6 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # Initial oil sat
        temp7 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # Fault
        temp8 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # QG
        temp9 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # Qw
        temp10 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # dT
        temp11 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # pressure
        temp12 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # water
        temp13 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # gas
        temp14 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # oil
        temp15 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # QG
        for kk in range(steppi):
            temp1[kk, :, :, :] = cPerm[yy, 0, :, :, :]
            temp2[kk, :, :, :] = cPhi[yy, 0, :, :, :]
            temp7[kk, :, :, :] = cfault[yy, 0, :, :, :]
            temp8[kk, :, :, :] = cQg[yy, kk, :, :, :]
            temp9[kk, :, :, :] = cQw[yy, kk, :, :, :]
            temp10[kk, :, :, :] = cdT[yy, kk, :, :, :]
            temp11[kk, :, :, :] = cPress[yy, kk, :, :, :]
            temp12[kk, :, :, :] = cSat[yy, kk, :, :, :]
            temp13[kk, :, :, :] = cSatg[yy, kk, :, :, :]
            temp14[kk, :, :, :] = cSato[yy, kk, :, :, :]
            temp15[kk, :, :, :] = cQ[yy, kk, :, :, :]
            if kk == 0:
                temp3[kk, :, :, :] = cPini[yy, 0, :, :, :]
                temp4[kk, :, :, :] = cSini[yy, 0, :, :, :]
                temp5[kk, :, :, :] = cSgini[yy, 0, :, :, :]
                temp6[kk, :, :, :] = cSoini[yy, 0, :, :, :]
            else:
                temp3[kk, :, :, :] = cPress[yy, kk - 1, :, :, :]
                temp4[kk, :, :, :] = cSat[yy, kk - 1, :, :, :]
                temp5[kk, :, :, :] = cSatg[yy, kk - 1, :, :, :]
                temp6[kk, :, :, :] = cSato[yy, kk - 1, :, :, :]
        p1.append(temp1)
        p2.append(temp2)
        p3.append(temp3)
        p4.append(temp4)
        p5.append(temp5)
        p6.append(temp6)
        p7.append(temp7)
        p8.append(temp8)
        p9.append(temp9)
        p10.append(temp10)
        p11.append(temp11)
        p12.append(temp12)
        p13.append(temp13)
        p14.append(temp14)
        p15.append(temp15)
    cPerm = np.concatenate(p1, axis=0)[:, None, :, :, :]
    cPhi = np.concatenate(p2, axis=0)[:, None, :, :, :]
    cPini = np.concatenate(p3, axis=0)[:, None, :, :, :]
    cSini = np.concatenate(p4, axis=0)[:, None, :, :, :]
    cSgini = np.concatenate(p5, axis=0)[:, None, :, :, :]
    cSoini = np.concatenate(p6, axis=0)[:, None, :, :, :]
    cfault = np.concatenate(p7, axis=0)[:, None, :, :, :]
    cQg = np.concatenate(p8, axis=0)[:, None, :, :, :]
    cQ = np.concatenate(p15, axis=0)[:, None, :, :, :]
    cQw = np.concatenate(p9, axis=0)[:, None, :, :, :]
    cdT = np.concatenate(p10, axis=0)[:, None, :, :, :]
    cPress = np.concatenate(p11, axis=0)[:, None, :, :, :]
    cSat = np.concatenate(p12, axis=0)[:, None, :, :, :]
    cSatg = np.concatenate(p13, axis=0)[:, None, :, :, :]
    cSato = np.concatenate(p14, axis=0)[:, None, :, :, :]
    data = {}
    if "permeability" in X_data1:
        data["perm"] = cPerm
    if "porosity" in X_data1:
        data["poro"] = cPhi
    if "Pini" in X_data1:
        data["pini"] = cPini
    if "Sini" in X_data1:
        data["sini"] = cSini
    if "SGAS" in output_variables:
        data["sgini"] = cSgini
    if "SOIL" in output_variables:
        data["soini"] = cSoini
    if "Q" in X_data1:
        data["Q"] = cQ
    if "Fault" in X_data1:
        data["fault"] = cfault / 100
    if "Qg" in X_data1:
        data["Qg"] = cQg  # [:, 0:1, :, :, :]
    if "Qw" in X_data1:
        data["Qw"] = cQw  # [:, 0:1, :, :, :]
    if "Time" in X_data1:
        data["dt"] = cdT  # [:, 0:1, :, :, :]
    if "Pressure" in X_data1:
        data["pressure"] = cPress  # * effec_abbi
    if "Water_saturation" in X_data1:
        data["water_sat"] = cSat  # * effec_abbi
    if "Gas_saturation" in X_data1:
        data["gas_sat"] = cSatg  # * effec_abbi
    if "Oil_saturation" in X_data1:
        data["oil_sat"] = cSato
    if dist.rank == 0:
        logger.info(
            "---------------------------------------------------------------------"
        )
        logger.info("Load simulated labelled training data for peacemann")
    with gzip.open(
        to_absolute_path("../data/data_train_peaceman.pkl.gz"), "rb"
    ) as f:
        mat = pickle.load(f)
    X_data2 = mat
    data2 = X_data2
    data2n = {key: value.transpose(0, 2, 1) for key, value in data2.items()}
    for key in data2n:
        data2n[key][data2n[key] <= 0] = 0
    if dist.rank == 0:
        logger.info(
            "---------------------------------------------------------------------"
        )
        logger.info("Load simulated labelled test data from .gz file")
    with gzip.open(to_absolute_path("../data/data_test.pkl.gz"), "rb") as f:
        mat = pickle.load(f)
    X_data1t = mat
    for key, value in X_data1t.items():
        logger.info(
            "---------------------------------------------------------------------"
        )
        logger.info(f"For key '{key}':")
        logger.info(f"\tContains inf: {np.isinf(value).any()}")
        logger.info(f"\tContains -inf: {np.isinf(-value).any()}")
        logger.info(f"\tContains NaN: {np.isnan(value).any()}")
    unique_times = np.unique(X_data1t["Time"])
    logger.info(f"Min time: {np.min(X_data1t['Time'])}")
    logger.info(f"Max time: {np.max(X_data1t['Time'])}")
    logger.info(f"Sorted unique time values: {unique_times}")
    for key in X_data1t.keys():
        X_data1t[key][np.isnan(X_data1t[key])] = 0
        X_data1t[key][np.isinf(X_data1t[key])] = 0
        X_data1t[key] = clip_and_convert_to_float32(X_data1t[key])
    if "PERM" in input_variables:
        cPerm = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Permeability
    if "PORO" in input_variables:
        cPhi = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Porosity
    if "PINI" in input_variables:
        cPini = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Initial pressure
    if "SINI" in input_variables:
        cSini = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)
    if "FAULT" in input_variables:
        cfault = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Fault
    if "SGINI" in input_variables:
        cSgini = 1e-3 * np.ones(
            (N_ens, 1, nz, nx, ny), dtype=np.float32
        )  # Initial gas saturation
        cSgini = clip_and_convert_to_float3(cSgini)
    if "SOINI" in input_variables:
        cSoini = cfg.custom.PROPS.SO1 * np.ones(
            (N_ens, 1, nz, nx, ny), dtype=np.float32
        )  # Initial gas saturation
        cSoini = clip_and_convert_to_float3(cSoini)
    cQ = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)
    cQw = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)
    cQg = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)
    cQo = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)
    cTime = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)
    if "PRESSURE" in output_variables:
        cPress = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)  # Pressure
    if "SWAT" in output_variables:
        cSat = np.zeros(
            (N_ens, steppi, nz, nx, ny), dtype=np.float32
        )  # Water saturation
    if "SGAS" in output_variables:
        cSatg = np.zeros(
            (N_ens, steppi, nz, nx, ny), dtype=np.float32
        )  # gas saturation
    if "SOIL" in output_variables:
        cSato = np.zeros(
            (N_ens, steppi, nz, nx, ny), dtype=np.float32
        )  # oil saturation
    for kk in range(N_ens):
        for i in range(nz):
            if "permeability" in X_data1t:
                cPerm[kk, 0, i, :, :] = X_data1t["permeability"][kk, 0, :, :, i]
            if "Fault" in X_data1t:
                cfault[kk, 0, i, :, :] = X_data1t["Fault"][kk, 0, :, :, i]
            if "porosity" in X_data1t:
                cPhi[kk, 0, i, :, :] = X_data1t["porosity"][kk, 0, :, :, i]
            if "Pini" in X_data1t:
                cPini[kk, 0, i, :, :] = X_data1t["Pini"][kk, 0, :, :, i] / maxP
            if "Sini" in X_data1t:
                cSini[kk, 0, i, :, :] = X_data1t["Sini"][kk, 0, :, :, i]
        for mv in range(steppi):
            for i in range(nz):
                if "Pressure" in X_data1t:
                    cPress[kk, mv, i, :, :] = X_data1t["Pressure"][kk, mv, :, :, i]
                if "Water_saturation" in X_data1t:
                    cSat[kk, mv, i, :, :] = X_data1t["Water_saturation"][
                        kk, mv, :, :, i
                    ]
                if "Gas_saturation" in X_data1t:
                    cSatg[kk, mv, i, :, :] = X_data1t["Gas_saturation"][kk, mv, :, :, i]
                if "Oil_saturation" in X_data1t:
                    cSato[kk, mv, i, :, :] = X_data1t["Oil_saturation"][kk, mv, :, :, i]
                cTime[kk, :, :, :, :] = time_values[mv] * np.ones(
                    (steppi, nz, nx, ny), dtype=np.float32
                )
    X_data1t["Q"][X_data1t["Q"] <= 0] = 0
    X_data1t["Qw"][X_data1t["Qw"] <= 0] = 0
    X_data1t["Qg"][X_data1t["Qg"] <= 0] = 0
    X_data1t["Qo"][X_data1t["Qo"] <= 0] = 0
    for kk in range(N_ens):
        for mv in range(steppi):
            for i in range(nz):
                X_data1t["Q"][kk, mv, :, :, i] = np.where(
                    X_data1t["Q"][kk, mv, :, :, i] < 0,
                    0,
                    X_data1t["Q"][kk, mv, :, :, i],
                )
                X_data1t["Qw"][kk, mv, :, :, i] = np.where(
                    X_data1t["Qw"][kk, mv, :, :, i] < 0,
                    0,
                    X_data1t["Qw"][kk, mv, :, :, i],
                )
                X_data1t["Qg"][kk, mv, :, :, i] = np.where(
                    X_data1t["Qg"][kk, mv, :, :, i] < 0,
                    0,
                    X_data1t["Qg"][kk, mv, :, :, i],
                )
                X_data1t["Qo"][kk, mv, :, :, i] = np.where(
                    X_data1t["Qo"][kk, mv, :, :, i] < 0,
                    0,
                    X_data1t["Qo"][kk, mv, :, :, i],
                )
                cQ[kk, mv, i, :, :] = X_data1t["Q"][kk, mv, :, :, i]
                cQw[kk, mv, i, :, :] = X_data1t["Qw"][kk, mv, :, :, i]
                cQg[kk, mv, i, :, :] = X_data1t["Qg"][kk, mv, :, :, i]
                cQo[kk, mv, i, :, :] = X_data1t["Qo"][kk, mv, :, :, i]
    cQ = cQ / maxQ
    cQw = cQw / maxQw
    cQg = cQg / maxQg
    # Removed unused variable Tact
    Timebefore = np.zeros((steppi))
    Timebefore[1:] = time_values[:-1]
    dT = time_values - Timebefore
    cdT = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)
    for kk in range(N_ens):
        for tt in range(steppi):
            cdT[kk, tt, :, :, :] = dT[tt] * np.ones((nz, nx, ny), dtype=np.float32)
    cQ[cQ == 0] = 0.1
    cQw[cQw == 0] = 0.1
    cQg[cQg == 0] = 0.1
    cQo[cQo == 0] = 0.1
    cdT = cdT / maxT
    neededMxt = {
        "Q": torch.from_numpy(cQ[0:1, :, :, :, :]).to(device, torch.float32),
        "Qw": torch.from_numpy(cQw[0:1, :, :, :, :]).to(device, dtype=torch.float32),
        "Qg": torch.from_numpy(cQg[0:1, :, :, :, :]).to(device, dtype=torch.float32),
        "Qo": torch.from_numpy(cQo[0:1, :, :, :, :]).to(device, dtype=torch.float32),
        "dt": torch.from_numpy(cdT[0:1, :, :, :, :]).to(device, dtype=torch.float32),
    }
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15 = [
        [] for _ in range(15)
    ]
    for yy in range(N_ens):
        temp1 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # permeability
        temp2 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # porosity
        temp3 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # Initial pressure
        temp4 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # Initial water sat
        temp5 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # Initial gas sat
        temp6 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # Initial oil sat
        temp7 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # Fault
        temp8 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # QG
        temp9 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # Qw
        temp10 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # dT
        temp11 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # pressure
        temp12 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # water
        temp13 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # gas
        temp14 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # oil
        temp15 = np.zeros((steppi, nz, nx, ny), dtype=np.float32)  # Qw
        for kk in range(steppi):
            temp1[kk, :, :, :] = cPerm[yy, 0, :, :, :]
            temp2[kk, :, :, :] = cPhi[yy, 0, :, :, :]
            temp7[kk, :, :, :] = cfault[yy, 0, :, :, :]
            temp8[kk, :, :, :] = cQg[yy, kk, :, :, :]
            temp9[kk, :, :, :] = cQw[yy, kk, :, :, :]
            temp10[kk, :, :, :] = cdT[yy, kk, :, :, :]
            temp11[kk, :, :, :] = cPress[yy, kk, :, :, :]
            temp12[kk, :, :, :] = cSat[yy, kk, :, :, :]
            temp13[kk, :, :, :] = cSatg[yy, kk, :, :, :]
            temp14[kk, :, :, :] = cSato[yy, kk, :, :, :]
            temp15[kk, :, :, :] = cQ[yy, kk, :, :, :]
            if kk == 0:
                temp3[kk, :, :, :] = cPini[yy, 0, :, :, :]
                temp4[kk, :, :, :] = cSini[yy, 0, :, :, :]
                temp5[kk, :, :, :] = cSgini[yy, 0, :, :, :]
                temp6[kk, :, :, :] = cSoini[yy, 0, :, :, :]
            else:
                temp3[kk, :, :, :] = cPress[yy, kk - 1, :, :, :]
                temp4[kk, :, :, :] = cSat[yy, kk - 1, :, :, :]
                temp5[kk, :, :, :] = cSatg[yy, kk - 1, :, :, :]
                temp6[kk, :, :, :] = cSato[yy, kk - 1, :, :, :]
        p1.append(temp1)
        p2.append(temp2)
        p3.append(temp3)
        p4.append(temp4)
        p5.append(temp5)
        p6.append(temp6)
        p7.append(temp7)
        p8.append(temp8)
        p9.append(temp9)
        p10.append(temp10)
        p11.append(temp11)
        p12.append(temp12)
        p13.append(temp13)
        p14.append(temp14)
        p15.append(temp15)
    cPerm = np.concatenate(p1, axis=0)[:, None, :, :, :]
    cPhi = np.concatenate(p2, axis=0)[:, None, :, :, :]
    cPini = np.concatenate(p3, axis=0)[:, None, :, :, :]
    cSini = np.concatenate(p4, axis=0)[:, None, :, :, :]
    cSgini = np.concatenate(p5, axis=0)[:, None, :, :, :]
    cSoini = np.concatenate(p6, axis=0)[:, None, :, :, :]
    cfault = np.concatenate(p7, axis=0)[:, None, :, :, :]
    cQg = np.concatenate(p8, axis=0)[:, None, :, :, :]
    cQ = np.concatenate(p15, axis=0)[:, None, :, :, :]
    cQw = np.concatenate(p9, axis=0)[:, None, :, :, :]
    cdT = np.concatenate(p10, axis=0)[:, None, :, :, :]
    cPress = np.concatenate(p11, axis=0)[:, None, :, :, :]
    cSat = np.concatenate(p12, axis=0)[:, None, :, :, :]
    cSatg = np.concatenate(p13, axis=0)[:, None, :, :, :]
    cSato = np.concatenate(p14, axis=0)[:, None, :, :, :]
    data_test = {}
    if "permeability" in X_data1t:
        data_test["perm"] = cPerm  # * effec_abbi
    if "porosity" in X_data1t:
        data_test["poro"] = cPhi  # * effec_abbi
    if "Pini" in X_data1t:
        data_test["pini"] = cPini  # * effec_abbi
    if "Sini" in X_data1t:
        data_test["sini"] = cSini  # * effec_abbi
    if "SGAS" in output_variables:
        data_test["sgini"] = cSgini
    if "SOIL" in output_variables:
        data_test["soini"] = cSoini
    if "Fault" in X_data1t:
        data_test["fault"] = cfault / 100  # * effec_abbi
    if "Q" in X_data1t:
        data_test["Q"] = cQ
    if "Qg" in X_data1t:
        data_test["Qg"] = cQg  # [:, 0:1, :, :, :]
    if "Qw" in X_data1t:
        data_test["Qw"] = cQw  # [:, 0:1, :, :, :]
    if "Time" in X_data1t:
        data_test["dt"] = cdT  # [:, 0:1, :, :, :]
    if "Pressure" in X_data1t:
        data_test["pressure"] = cPress  # * effec_abbi
    if "Water_saturation" in X_data1t:
        data_test["water_sat"] = cSat  # * effec_abbi
    if "Gas_saturation" in X_data1t:
        data_test["gas_sat"] = cSatg  # * effec_abbi
    if "Oil_saturation" in X_data1t:
        data_test["oil_sat"] = cSato
    if dist.rank == 0:
        logger.info(
            "---------------------------------------------------------------------"
        )
        logger.info("Load simulated labelled test data for peacemann modelling")
    with gzip.open(to_absolute_path("../data/data_test_peaceman.pkl.gz"), "rb") as f:
        mat = pickle.load(f)
    X_data2t = mat
    data2_test = X_data2t
    data2n_test = {key: value.transpose(0, 2, 1) for key, value in data2_test.items()}
    for key in data2n_test:
        data2n_test[key][data2n_test[key] <= 0] = 0
    input_keys = []
    if "PERM" in input_variables:
        input_keys.append("perm")
    if "PORO" in input_variables:
        input_keys.append("poro")
    if "PINI" in input_variables:
        input_keys.append("pini")
    if "SINI" in input_variables:
        input_keys.append("sini")
    if "SGINI" in input_variables:
        input_keys.append("sgini")
    if "SOINI" in input_variables:
        input_keys.append("soini")
    if "FAULT" in input_variables:
        input_keys.append("fault")
    # input_variables2 = input_variables
    if "WTIR" in cfg.custom.input_properties2:
        input_keys.append("Q")
    if "WGIR" in cfg.custom.input_properties2:
        input_keys.append("Qg")
    if "WWIR" in cfg.custom.input_properties2:
        input_keys.append("Qw")
    if "DELTA_TIME" in cfg.custom.input_properties2:
        input_keys.append("dt")
    input_keys_peacemann = []
    input_keys_peacemann.append("X")
    output_keys_peacemann = []
    output_keys_peacemann.append("Y")
    output_keys_pressure = []
    if "PRESSURE" in output_variables:
        output_keys_pressure.append("pressure")
    output_keys_saturation = []
    output_keys_oil = []
    output_keys_gas = []
    if "SWAT" in output_variables:
        output_keys_saturation.append("water_sat")
    if "SGAS" in output_variables:
        output_keys_gas.append("gas_sat")
    if "SOIL" in output_variables:
        output_keys_oil.append("oil_sat")
    logger.info("--------------------------------------------------------------")
    for key in data.keys():
        data[key][np.isnan(data[key])] = np.min(data[key])  # Convert NaN to 0
        data[key][np.isinf(data[key])] = np.min(data[key])  # Convert infinity to 0
        data[key] = clip_and_convert_to_float32(data[key])
        data[key] = np.clip(data[key], 0, None)  # Clip negative values to 0
    for key, value in data.items():
        if dist.rank == 0:
            logger.info(f"For key in Training Data '{key}':")
            logger.info(f"\tContains inf: {np.isinf(value).any()}")
            logger.info(f"\tContains -inf: {np.isinf(-value).any()}")
            logger.info(f"\tContains NaN: {np.isnan(value).any()}")
            logger.info(f"\tSize = : {value.shape}")
            logger.info(f"\tMin value: {np.min(value)}")
            logger.info(f"\tMax value: {np.max(value)}")
            logger.info(
                "--------------------------------------------------------------"
            )
    for key in data_test.keys():
        data_test[key][np.isnan(data_test[key])] = np.min(
            data_test[key]
        )  # Convert NaN to 0
        data_test[key][np.isinf(data_test[key])] = np.min(
            data_test[key]
        )  # Convert infinity to 0
        data_test[key] = clip_and_convert_to_float32(data_test[key])
        data_test[key] = np.clip(data_test[key], 0, None)
    for key, value in data_test.items():
        if dist.rank == 0:
            logger.info(f"For key in Test Data '{key}':")
            logger.info(f"\tContains inf: {np.isinf(value).any()}")
            logger.info(f"\tContains -inf: {np.isinf(-value).any()}")
            logger.info(f"\tContains NaN: {np.isnan(value).any()}")
            logger.info(f"\tSize = : {value.shape}")
            logger.info(f"\tMin value: {np.min(value)}")
            logger.info(f"\tMax value: {np.max(value)}")
            logger.info(
                "--------------------------------------------------------------"
            )
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                     DATASET AND DATALOADER                      |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    if cfg.custom.model_Distributed == 2:
        if cfg.custom.model_saturation == "FNO":
            batch_sizee = cfg.batch_size.grid_fno
        else:
            batch_sizee = cfg.batch_size.grid_gnn
    else:
        if cfg.custom.model_saturation == "FNO":
            temp = cfg.batch_size.grid_fno
            num_ranks = dist.world_size
            if dist.rank == 0:
                logger.info(f"Number of GPU ranks in use: {num_ranks}")
            temp = temp / num_ranks
            batch_sizee = int(temp)
            if batch_sizee < 1:
                batch_sizee = 1
        else:
            batch_sizee = cfg.batch_size.grid_gnn
    dataset_train = Labelledset(data, dist.device)
    train_sampler = DistributedSampler(
        dataset_train,
        num_replicas=dist.world_size,
        rank=dist.local_rank,
        shuffle=True,
        drop_last=False,
    )
    labelled_loader_train = DataLoader(
        dataset_train,  # Pass the dataset here
        batch_size=batch_sizee,
        shuffle=False,  # No need for shuffle when using a sampler
        sampler=train_sampler,  # Pass the sampler here
    )
    datasetp_train = Labelledset(data2n, dist.device)
    trainp_sampler = DistributedSampler(
        datasetp_train,
        num_replicas=dist.world_size,
        rank=dist.local_rank,
        shuffle=True,
        drop_last=False,
    )
    labelled_loader_trainp = DataLoader(
        datasetp_train,  # Pass the dataset here
        batch_size=batch_sizee,
        shuffle=False,  # No need for shuffle when using a sampler
        sampler=trainp_sampler,  # Pass the sampler here
    )
    dataset_testt = Labelledset(data_test, dist.device)
    test_sampler = DistributedSampler(
        dataset_testt,
        num_replicas=dist.world_size,
        rank=dist.local_rank,
        shuffle=True,
        drop_last=False,
    )
    labelled_loader_testt = DataLoader(
        dataset_testt,  # Pass the dataset here
        batch_size=cfg.batch_size.test,
        shuffle=False,  # No need for shuffle when using a sampler
        sampler=test_sampler,  # Pass the sampler here
    )
    datasetp_testt = Labelledset(data2n_test, dist.device)
    testp_sampler = DistributedSampler(
        datasetp_testt,
        num_replicas=dist.world_size,
        rank=dist.local_rank,
        shuffle=True,
        drop_last=False,
    )
    labelled_loader_testtp = DataLoader(
        datasetp_testt,  # Pass the dataset here
        batch_size=cfg.batch_size.test,
        shuffle=False,  # No need for shuffle when using a sampler
        sampler=testp_sampler,  # Pass the sampler here
    )
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|               SET UP SURROGATE ARCHITECTURE                     |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    if "PRESSURE" in output_variables:
        surrogate_pressure = create_fno_model(
            len(input_keys),
            1,
            len(output_keys_pressure),
            dist.device,
            num_fno_modes=16,
            latent_channels=32,
            decoder_layer_size=32,
            padding=22,
            decoder_layers=4,
            dimension=3,
        )
    surrogate_peacemann = create_fno_model(
        2 + (4 * N_pr),
        lenwels * N_pr,
        len(output_keys_peacemann),
        dist.device,
        num_fno_modes=13,
        latent_channels=64,
        decoder_layer_size=32,
        padding=20,
        num_fno_layers=5,
        decoder_layers=4,
        dimension=1,
    )
    if "SGAS" in output_variables:
        surrogate_gas = create_fno_model(
            len(input_keys),
            1,
            len(output_keys_gas),
            dist.device,
            num_fno_modes=16,
            latent_channels=32,
            decoder_layer_size=32,
            padding=22,
            decoder_layers=4,
            dimension=3,
        )
    if "SWAT" in output_variables:
        surrogate_saturation = create_fno_model(
            len(input_keys),
            1,
            len(output_keys_saturation),
            dist.device,
            num_fno_modes=16,
            latent_channels=32,
            decoder_layer_size=32,
            padding=22,
            decoder_layers=4,
            dimension=3,
        )

    if "SOIL" in output_variables:
        surrogate_oil = create_fno_model(
            len(input_keys),
            1,
            len(output_keys_oil),
            dist.device,
            num_fno_modes=16,
            latent_channels=32,
            decoder_layer_size=32,
            padding=22,
            decoder_layers=4,
            dimension=3,
        )
    if cfg.custom.model_saturation == "FNO":
        if cfg.custom.fno_type == "FNO":
            if dist.rank == 0:
                logger.info(
                    "|-----------------------------------------------------------------|"
                )
                logger.info(
                    "|   PRESSURE MODEL = FNO   SATUARATION MODEL = FNO   :            |"
                )
                logger.info(
                    "|-----------------------------------------------------------------|"
                )
        else:
            if dist.rank == 0:
                logger.info(
                    "|-----------------------------------------------------------------|"
                )
                logger.info(
                    "|   PRESSURE MODEL = PINO   SATUARATION MODEL = PINO   :            |"
                )
                logger.info(
                    "|-----------------------------------------------------------------|"
                )
    if dist.rank == 0 and wandb.run is not None:
        if "PRESSURE" in output_variables:
            wandb.watch(surrogate_pressure, log="all", log_freq=1000, log_graph=(True))
        if "SWAT" in output_variables:
            wandb.watch(
                surrogate_saturation, log="all", log_freq=1000, log_graph=(True)
            )
        if "SOIL" in output_variables:
            wandb.watch(surrogate_oil, log="all", log_freq=1000, log_graph=(True))
        wandb.watch(surrogate_peacemann, log="all", log_freq=1000, log_graph=(True))
        if "SGAS" in output_variables:
            wandb.watch(surrogate_gas, log="all", log_freq=1000, log_graph=(True))
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                   SETUP DISTRIBUTED LEARNING                    |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    if dist.world_size > 1:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            if "PRESSURE" in output_variables:
                surrogate_pressure = DistributedDataParallel(
                    surrogate_pressure,
                    device_ids=[dist.local_rank],
                    output_device=dist.device,
                    broadcast_buffers=dist.broadcast_buffers,
                    find_unused_parameters=dist.find_unused_parameters,
                )
            if "SWAT" in output_variables:
                surrogate_saturation = DistributedDataParallel(
                    surrogate_saturation,
                    device_ids=[dist.local_rank],
                    output_device=dist.device,
                    broadcast_buffers=dist.broadcast_buffers,
                    find_unused_parameters=dist.find_unused_parameters,
                )
            if "SOIL" in output_variables:
                surrogate_oil = DistributedDataParallel(
                    surrogate_oil,
                    device_ids=[dist.local_rank],
                    output_device=dist.device,
                    broadcast_buffers=dist.broadcast_buffers,
                    find_unused_parameters=dist.find_unused_parameters,
                )
            if "SGAS" in output_variables:
                surrogate_gas = DistributedDataParallel(
                    surrogate_gas,
                    device_ids=[dist.local_rank],
                    output_device=dist.device,
                    broadcast_buffers=dist.broadcast_buffers,
                    find_unused_parameters=dist.find_unused_parameters,
                )
            surrogate_peacemann = DistributedDataParallel(
                surrogate_peacemann,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
    if cfg.custom.model_Distributed == 2:
        lr = cfg.optimizer.lr
        weight_decay = cfg.optimizer.weight_decay
        # is_distributed = False
    else:
        num_ranks = dist.world_size
        if dist.rank == 0:
            logger.info(f"Number of GPU ranks in use: {num_ranks}")
        lr = cfg.optimizer.lr * 2# num_ranks
        weight_decay = cfg.optimizer.weight_decay
        # is_distributed = True
    optimizer_config = {
        "lr": lr,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }

    if "PRESSURE" in output_variables:
        optimizer_pressure = torch.optim.AdamW(
            create_param_groupss(surrogate_pressure, weight_decay), **optimizer_config
        )

        scheduler_pressure = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_pressure, gamma=cfg.optimizer.gamma
        )
    if "SGAS" in output_variables:
        optimizer_gas = torch.optim.AdamW(
            create_param_groupss(surrogate_gas, weight_decay), **optimizer_config
        )
        scheduler_gas = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_gas, gamma=cfg.optimizer.gamma
        )
    if "SWAT" in output_variables:
        optimizer_saturation = torch.optim.AdamW(
            create_param_groupss(surrogate_saturation, weight_decay), **optimizer_config
        )
        scheduler_saturation = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_saturation, gamma=cfg.optimizer.gamma
        )

    if "SOIL" in output_variables:
        optimizer_oil = torch.optim.AdamW(
            create_param_groupss(surrogate_oil, weight_decay), **optimizer_config
        )
        scheduler_oil = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_oil, gamma=cfg.optimizer.gamma
        )
    optimizer_peacemann = torch.optim.Adam(
        surrogate_peacemann.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    scheduler_peacemann = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_peacemann, gamma=cfg.optimizer.gamma
    )
    MODELS_C = {}
    if "PRESSURE" in output_variables:
        MODELS_C["pressure"] = optimizer_pressure
    if "SWAT" in output_variables:
        MODELS_C["saturation"] = optimizer_saturation
    if "SOIL" in output_variables:
        MODELS_C["oil"] = optimizer_oil
    MODELS_C["peacemann"] = optimizer_peacemann
    if "SGAS" in output_variables:
        MODELS_C["gas"] = optimizer_gas
    combined_optimizer = CompositeOptimizer(MODELS_C)
    if cfg.custom.fno_type == "FNO":
        # Attempt to load latest checkpoint if one exists
        if "PRESSURE" in output_variables:
            loaded_epoch_pressure = load_checkpoint(
                to_absolute_path("../MODELS/FNO/checkpoints_pressure_seq"),
                models=surrogate_pressure,
                optimizer=optimizer_pressure,
                scheduler=scheduler_pressure,
                device=dist.device,
            )
            use_epoch = loaded_epoch_pressure
        if "SGAS" in output_variables:
            loaded_epoch_gas = load_checkpoint(
                to_absolute_path("../MODELS/FNO/checkpoints_gas_seq"),
                models=surrogate_gas,
                optimizer=optimizer_gas,
                scheduler=scheduler_gas,
                device=dist.device,
            )
            use_epoch = loaded_epoch_gas
        if "SWAT" in output_variables:
            loaded_epoch_saturation = load_checkpoint(
                to_absolute_path("../MODELS/FNO/checkpoints_saturation_seq"),
                models=surrogate_saturation,
                optimizer=optimizer_saturation,
                scheduler=scheduler_saturation,
                device=dist.device,
            )
            use_epoch = loaded_epoch_saturation
        if "SOIL" in output_variables:
            loaded_epoch_oil = load_checkpoint(
                to_absolute_path("../MODELS/FNO/checkpoints_oil_seq"),
                models=surrogate_oil,
                optimizer=optimizer_oil,
                scheduler=scheduler_oil,
                device=dist.device,
            )
            use_epoch = loaded_epoch_oil
        loaded_epoch_peacemann = load_checkpoint(
            to_absolute_path("../MODELS/FNO/checkpoints_peacemann_seq"),
            models=surrogate_peacemann,
            optimizer=optimizer_peacemann,
            scheduler=scheduler_peacemann,
            device=dist.device,
        )
        use_epoch = loaded_epoch_peacemann
    else:
        # Attempt to load latest checkpoint if one exists
        if "PRESSURE" in output_variables:
            loaded_epoch_pressure = load_checkpoint(
                to_absolute_path("../MODELS/PINO/checkpoints_pressure_seq"),
                models=surrogate_pressure,
                optimizer=optimizer_pressure,
                scheduler=scheduler_pressure,
                device=dist.device,
            )
            use_epoch = loaded_epoch_pressure
        if "SGAS" in output_variables:
            loaded_epoch_gas = load_checkpoint(
                to_absolute_path("../MODELS/PINO/checkpoints_gas_seq"),
                models=surrogate_gas,
                optimizer=optimizer_gas,
                scheduler=scheduler_gas,
                device=dist.device,
            )
            use_epoch = loaded_epoch_gas
        if "SWAT" in output_variables:
            loaded_epoch_saturation = load_checkpoint(
                to_absolute_path("../MODELS/PINO/checkpoints_saturation_seq"),
                models=surrogate_saturation,
                optimizer=optimizer_saturation,
                scheduler=scheduler_saturation,
                device=dist.device,
            )
            use_epoch = loaded_epoch_saturation
        if "SOIL" in output_variables:
            loaded_epoch_oil = load_checkpoint(
                to_absolute_path("../MODELS/PINO/checkpoints_oil_seq"),
                models=surrogate_oil,
                optimizer=optimizer_oil,
                scheduler=scheduler_oil,
                device=dist.device,
            )
            use_epoch = loaded_epoch_oil
        loaded_epoch_peacemann = load_checkpoint(
            to_absolute_path("../MODELS/PINO/checkpoints_peacemann_seq"),
            models=surrogate_peacemann,
            optimizer=optimizer_peacemann,
            scheduler=scheduler_peacemann,
            device=dist.device,
        )
        use_epoch = loaded_epoch_peacemann
    MODELS = {}
    SCHEDULER = {}
    if "PRESSURE" in output_variables:
        MODELS["PRESSURE"] = surrogate_pressure
        SCHEDULER["PRESSURE"] = scheduler_pressure
    if "SGAS" in output_variables:
        MODELS["SGAS"] = surrogate_gas
        SCHEDULER["SGAS"] = scheduler_pressure
    if "SWAT" in output_variables:
        MODELS["SATURATION"] = surrogate_saturation
        SCHEDULER["SATURATION"] = scheduler_saturation
    if "SOIL" in output_variables:
        MODELS["SOIL"] = surrogate_oil
        SCHEDULER["SOIL"] = scheduler_oil
    MODELS["PEACEMANN"] = surrogate_peacemann
    SCHEDULER["PEACEMANN"] = scheduler_peacemann
    composite_model = CompositeModel(MODELS, output_variables)

    training_setup = {
        "data_train": data,  # Training data dictionary
        "data_test": data_test,  # Test data dictionary
        "labelled_loader_train": labelled_loader_train,  # Training dataloader
        "labelled_loader_trainp": labelled_loader_trainp,  # Training dataloader
        "labelled_loader_testt": labelled_loader_testt,  # Test dataloader
        "labelled_loader_testtp": labelled_loader_testtp,  # Test dataloader
        "models": MODELS,  # Dictionary of FNO/PINO models
        "composite_model": composite_model,  # Combined model
        "combined_optimizer": combined_optimizer,  # Combined optimizer
        "input_keys": input_keys,  # Input variable keys
        "input_keys_peacemann": input_keys_peacemann,  # Input variable keys
        "output_keys_peacemann": output_keys_peacemann,  # Input variable keys
        "output_keys_pressure": output_keys_pressure,  # Pressure output keys
        "output_keys_saturation": output_keys_saturation,  # Saturation output keys
        "output_keys_gas": output_keys_gas,  # Gas output keys
        "output_keys_oil": output_keys_oil,  # Oil output keys
        "MODELS": MODELS,  # MODELS
        "MODELS_C": MODELS_C,  # OPTIMIZERS
        "SCHEDULER": SCHEDULER,  # OPTIMIZERS
        "neededMx": neededMx,
        "neededMxt": neededMxt,
        "use_epoch": use_epoch,  # OPTIMIZERS
        "neededM": neededM,  # Additional needed tensors
    }
    return training_setup
