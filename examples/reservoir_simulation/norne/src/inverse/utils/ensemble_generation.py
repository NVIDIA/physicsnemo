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
                    ENSEMBLE GENERATION UTILITIES MODULE
=====================================================================

This module provides ensemble generation utilities for inverse problems
in reservoir simulation. It includes model loading, ensemble creation,
and data processing utilities.

This version is rank-aware for multi-GPU execution under torchrun:
- All numpy/torch RNG calls are seeded so every rank computes identical
  ensemble indices and noise perturbations (no ensemble divergence).
- Plotting and disk writes are guarded with `if dist.rank == 0`.
- A barrier follows any rank-0-only directory deletion to prevent races.

@Author : Clement Etienam
"""

import os
import sys
import random
import pickle
import time
import gzip
import shutil
import yaml
import logging
from math import sqrt

import numpy as np
import numpy.matlib
import pandas as pd
import torch
import torch.distributed as torchdist
from hydra.utils import to_absolute_path

from inverse.inversion_operation_ensemble import (
    clip_ensemble_params,
)


from inverse.inversion_operation_gather import (
    plot_rsm_singleT,
)

from inverse.inversion_operation_misc import (
    read_until_line,
    add_gnoise,
)
from utils.ensemble_utils import fast_gaussian
from utils.model_utils import (
    create_fno_model,
    create_transolver_model,
    load_modell,
)
from inverse.utils.inverse_config import (
    EnsembleRuntime,
    EnsembleSetup,
    GridConfig,
    InversionParams,
    PermBounds,
    PriorEnsembles,
    TimeArrays,
    WellConfig,
)


def _is_dist_active():
    """Return True if torch.distributed has been initialised."""
    return torchdist.is_available() and torchdist.is_initialized()


def _barrier():
    """Distributed barrier that no-ops in single-GPU mode."""
    if _is_dist_active():
        torchdist.barrier()


def _safe_seed(cfg, dist):
    """Seed numpy and Python RNGs identically on every rank.

    This is critical for multi-GPU runs — without a shared seed, each rank
    would draw different random ensemble indices and the inversion would
    diverge across ranks.
    """
    seed = int(getattr(cfg.custom, "seed", 12345))
    np.random.seed(seed)
    random.seed(seed)
    # torch.manual_seed not strictly required here but keeps things consistent
    torch.manual_seed(seed)


def setup_logging() -> logging.Logger:
    """Configure and return the main logger with green INFO console output.

    Note: this is called at module import. The file handler writes to a
    single `read_vectors.log`. Under torchrun multiple ranks will all open
    this file; if you see interleaved log lines, switch to a per-rank
    filename (e.g. `read_vectors_rank{rank}.log`) — but normally only
    rank 0's logger is actively used so this is fine in practice.
    """
    logger = logging.getLogger("inverse problem")
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


logger = setup_logging()


def Get_Time(nx, ny, nz, steppi, steppi_indices, N):
    """Return tiled time volume and shape helpers for dataset construction."""
    with gzip.open(to_absolute_path("../data/data_train.pkl.gz"), "rb") as f2:
        mat = pickle.load(f2)
    X_data1 = mat
    Time = X_data1["Time"]
    np_array2 = np.zeros(Time.shape[1])
    for mm in range(Time.shape[1]):
        np_array2[mm] = Time[0, mm, 0, 0, 0]
    Timee = []
    for _k in range(N):
        check = np.ones((nx, ny, nz), dtype=np.float16)
        unie = []
        for zz in range(len(np_array2)):
            aa = np_array2[zz] * check
            unie.append(aa)
        Time = np.stack(unie, axis=0)
        Timee.append(Time)
    return np.stack(Timee, axis=0)


def historydata(timestep, steppi, steppi_indices, N_pr):
    """Load and assemble historical RSM slices by category for a NORNE deck."""
    file_path = "../simulator_data/Flow.xlsx"
    df = pd.read_excel(file_path, skiprows=1)
    data_array = df.to_numpy()[:10, 1:]
    WOIL1 = data_array[:, :N_pr]
    WWATER1 = data_array[:, N_pr : 2 * N_pr]
    WGAS1 = data_array[:, 2 * N_pr : 3 * N_pr]
    DATA = {"OIL": WOIL1, "WATER": WWATER1, "GAS": WGAS1}
    DATA2 = WGAS1.reshape(-1, 1, order="F")
    new = np.hstack([WOIL1, WWATER1, WGAS1])
    return DATA, DATA2, new


def scale_array(arr):
    """Scale array magnitude to ~3 digits and return scaled array and factor.

    bool1=1 → scaled down, bool1=2 → scaled up
    """
    max_val = np.max(np.abs(arr))
    if max_val == 0:
        return arr, 1, 1  # No scaling needed for an array of zeroes
    num_digits_before_decimal = int(np.floor(np.log10(max_val))) + 1
    if num_digits_before_decimal >= 3:
        scaling_factor = 10 ** (num_digits_before_decimal - 3)
    else:
        scaling_factor = 10 ** (3 - num_digits_before_decimal)
    if num_digits_before_decimal >= 3:
        scaled_arr = arr / scaling_factor
        bool1 = 1
    else:
        scaled_arr = arr * scaling_factor
        bool1 = 2
    return scaled_arr, scaling_factor, bool1


def get_keep_mask(data, threshold=1e-4, reshape_fortran=True):
    """Return boolean mask: True = keep (value > threshold)."""
    is_torch = isinstance(data, torch.Tensor)

    if is_torch:
        if reshape_fortran:
            flat = data.permute(*reversed(range(data.ndim))).contiguous().reshape(-1)
        else:
            flat = data.reshape(-1)
        return flat > threshold
    else:
        order = "F" if reshape_fortran else "C"
        return np.reshape(data, -1, order) > threshold


def setup_models_and_data(
    input_variables: list,
    output_variables: list,
    runtime: "EnsembleRuntime",
    grid: "GridConfig",
    well: "WellConfig",
    priors: "PriorEnsembles",
    Ne: int,
    minK,
    maxK,
):
    """Initialise surrogates, load data, and prepare loaders for training.

    Parameters
    ----------
    input_variables : list[str]
        Active input property names (e.g. ``["PERM", "PORO", "FAULT"]``).
    output_variables : list[str]
        Active output property names (e.g. ``["PRESSURE", "SWAT", "SGAS"]``).
    runtime : EnsembleRuntime
        Hydra config, distributed manager, device, oldfolder, DEFAULT toggle,
        ``excel`` flag, and the mutable ``TEMPLATEFILE`` dict.
    grid : GridConfig
        Grid dimensions and time-step indexing.
    well : WellConfig
        Producer/injector counts, names, and completion data.
    priors : PriorEnsembles
        Prior PERM / PORO / FAULT ensemble arrays.
    Ne : int
        Number of ensemble members to construct.
    minK, maxK : float
        Permeability normalisation bounds used for clipping and rescaling.

    Returns
    -------
    tuple
        Models, configuration, and processed data needed by the inverse
        pipeline (see legacy implementation for the precise tuple shape).
    """
    cfg = runtime.cfg
    dist = runtime.dist
    device = runtime.device
    oldfolder = runtime.oldfolder
    DEFAULT = runtime.DEFAULT
    excel = runtime.excel
    TEMPLATEFILE = runtime.TEMPLATEFILE

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    steppi = grid.steppi
    steppi_indices = grid.steppi_indices

    N_pr = well.N_pr
    lenwels = well.lenwels
    well_names = well.well_names

    perm_ensembley = priors.perm

    # Seed RNGs identically across ranks BEFORE any np.random call below.
    _safe_seed(cfg, dist)

    input_variables2 = cfg.custom.input_properties2
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
    if "WTIR" in input_variables2:
        input_keys.append("Q")
    if "WGIR" in input_variables2:
        input_keys.append("Qg")
    if "WWIR" in input_variables2:
        input_keys.append("Qw")
    if "DELTA_TIME" in input_variables2:
        input_keys.append("dt")
        input_keys.append("t")
    output_keys_peacemann = ["Y"]
    output_keys_pressure = []
    if "PRESSURE" in output_variables:
        output_keys_pressure.append("pressure")
    output_keys_gas = []
    if "SGAS" in output_variables:
        output_keys_gas.append("gas_sat")
    output_keys_saturation = []
    if "SWAT" in output_variables:
        output_keys_saturation.append("water_sat")
    output_keys_oil = []
    if "SOIL" in output_variables:
        output_keys_oil.append("oil_sat")

    if cfg.custom.model_type == "FNO":
        if "PRESSURE" in output_variables:
            fno_supervised_pressure = create_fno_model(
                len(input_keys),
                1,
                len(output_keys_pressure),
                device,
            )
        if "SGAS" in output_variables:
            fno_supervised_gas = create_fno_model(
                len(input_keys),
                1,
                len(output_keys_gas),
                device,
            )
        fno_supervised_peacemann = create_fno_model(
            2 + (4 * N_pr),
            lenwels * N_pr,
            len(output_keys_peacemann),
            dist.device,
            num_fno_modes=13,
            decoder_layers=1,
            padding=20,
            num_fno_layers=5,
            dimension=1,
        )
        if "SWAT" in output_variables:
            fno_supervised_saturation = create_fno_model(
                len(input_keys),
                1,
                len(output_keys_saturation),
                device,
            )
        if "SOIL" in output_variables:
            fno_supervised_oil = create_fno_model(
                len(input_keys),
                1,
                len(output_keys_oil),
                device,
            )
    else:
        if "PRESSURE" in output_variables:
            fno_supervised_pressure = create_transolver_model(
                functional_dim=len(input_keys),
                out_dim=len(output_keys_pressure),
                device=device,
                n_layers=4,
                n_hidden=24,
                n_head=12,
                structured_shape=(nx, ny),
                use_te=True,
            )
        if "SGAS" in output_variables:
            fno_supervised_gas = create_transolver_model(
                functional_dim=len(input_keys),
                out_dim=len(output_keys_gas),
                device=device,
                n_layers=4,
                n_hidden=24,
                n_head=12,
                structured_shape=(nx, ny),
                use_te=True,
            )
        fno_supervised_peacemann = create_fno_model(
            2 + (4 * N_pr),
            lenwels * N_pr,
            len(output_keys_peacemann),
            dist.device,
            num_fno_modes=13,
            decoder_layers=1,
            padding=20,
            num_fno_layers=5,
            dimension=1,
        )
        if "SWAT" in output_variables:
            fno_supervised_saturation = create_transolver_model(
                functional_dim=len(input_keys),
                out_dim=len(output_keys_saturation),
                device=device,
                n_layers=4,
                n_hidden=24,
                n_head=12,
                structured_shape=(nx, ny),
                use_te=True,
            )
        if "SOIL" in output_variables:
            fno_supervised_oil = create_transolver_model(
                functional_dim=len(input_keys),
                out_dim=len(output_keys_oil),
                device=device,
                n_layers=4,
                n_hidden=24,
                n_head=12,
                structured_shape=(nx, ny),
                use_te=True,
            )

    # excel = 2
    base_paths = {
        "pressure": "./checkpoints_pressure",
        "gas": "./checkpoints_gas",
        "peacemann": "./checkpoints_peacemann",
        "saturation": "./checkpoints_saturation",
        "oil": "./checkpoints_oil",
    }

    if cfg.custom.model_type == "FNO":
        if cfg.custom.fno_type == "FNO":
            os.chdir("../MODELS/FNO")
            if dist.rank == 0:
                logger.info(
                    "|-----------------------------------------------------------------|"
                )
                logger.info(
                    "|                     FNO MODEL LEARNING    :                     |"
                )
                logger.info(
                    "|-----------------------------------------------------------------|"
                )
                logger.info(
                    "|-------------------------------------------------------------------------|"
                )
                logger.info(
                    "|   PRESSURE MODEL = FNO;   SATUARATION MODEL = FNO; PEACEMAN MODEL = FNO |"
                )
                logger.info(
                    "|-------------------------------------------------------------------------|"
                )
            models = {}
            if "PRESSURE" in output_variables:
                if dist.rank == 0:
                    logger.info("🟢 Loading Surrogate Model for Pressure")
                model_path = (
                    os.path.join(
                        base_paths["pressure"], "fno_pressure_forward_model.pth"
                    )
                    if excel == 1
                    else os.path.join(base_paths["pressure"], "checkpoint.pth")
                )
                fno_supervised_pressure = load_modell(
                    fno_supervised_pressure,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "PRESSURE",
                )
                models["pressure"] = fno_supervised_pressure
            if "SGAS" in output_variables:
                if dist.rank == 0:
                    logger.info("🟠 Loading Surrogate Model for Gas")
                model_path = (
                    os.path.join(base_paths["gas"], "fno_gas_forward_model.pth")
                    if excel == 1
                    else os.path.join(base_paths["gas"], "checkpoint.pth")
                )
                fno_supervised_gas = load_modell(
                    fno_supervised_gas,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "SGAS",
                )
                models["gas"] = fno_supervised_gas
            if dist.rank == 0:
                logger.info("🔵 Loading Surrogate Model for Peacemann")
            model_path = (
                os.path.join(base_paths["peacemann"], "fno_peacemann_forward_model.pth")
                if excel == 1
                else os.path.join(base_paths["peacemann"], "checkpoint.pth")
            )
            fno_supervised_peacemann = load_modell(
                fno_supervised_peacemann,
                model_path,
                cfg.custom.model_Distributed,
                device,
                excel,
                "PEACEMANN",
            )
            models["peacemann"] = fno_supervised_peacemann
            if "SWAT" in output_variables:
                if dist.rank == 0:
                    logger.info("🟣 Loading Surrogate Model for Saturation")
                model_path = (
                    os.path.join(
                        base_paths["saturation"], "fno_saturation_forward_model.pth"
                    )
                    if excel == 1
                    else os.path.join(base_paths["saturation"], "checkpoint.pth")
                )
                fno_supervised_saturation = load_modell(
                    fno_supervised_saturation,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "SWAT",
                )
                models["saturation"] = fno_supervised_saturation
            if "SOIL" in output_variables:
                if dist.rank == 0:
                    logger.info("🟣 Loading Surrogate Model for oil")
                model_path = (
                    os.path.join(base_paths["oil"], "fno_oil_forward_model.pth")
                    if excel == 1
                    else os.path.join(base_paths["oil"], "checkpoint.pth")
                )
                fno_supervised_oil = load_modell(
                    fno_supervised_oil,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "SOIL",
                )
                models["oil"] = fno_supervised_oil
        else:
            os.chdir("../MODELS/PINO")
            if dist.rank == 0:
                logger.info(
                    "|-----------------------------------------------------------------|"
                )
                logger.info(
                    "|                     PINO MODEL LEARNING    :                     |"
                )
                logger.info(
                    "|-----------------------------------------------------------------|"
                )
            models = {}
            if "PRESSURE" in output_variables:
                if dist.rank == 0:
                    logger.info("🟢 Loading Surrogate Model for Pressure")
                model_path = (
                    os.path.join(
                        base_paths["pressure"], "pino_pressure_forward_model.pth"
                    )
                    if excel == 1
                    else os.path.join(base_paths["pressure"], "checkpoint.pth")
                )
                fno_supervised_pressure = load_modell(
                    fno_supervised_pressure,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "PRESSURE",
                )
                models["pressure"] = fno_supervised_pressure
            if "SGAS" in output_variables:
                if dist.rank == 0:
                    logger.info("🟠 Loading Surrogate Model for Gas")
                model_path = (
                    os.path.join(base_paths["gas"], "pino_gas_forward_model.pth")
                    if excel == 1
                    else os.path.join(base_paths["gas"], "checkpoint.pth")
                )
                fno_supervised_gas = load_modell(
                    fno_supervised_gas,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "SGAS",
                )
                models["gas"] = fno_supervised_gas
            if dist.rank == 0:
                logger.info("🔵 Loading Surrogate Model for Peacemann")
            model_path = (
                os.path.join(
                    base_paths["peacemann"], "pino_peacemann_forward_model.pth"
                )
                if excel == 1
                else os.path.join(base_paths["peacemann"], "checkpoint.pth")
            )
            fno_supervised_peacemann = load_modell(
                fno_supervised_peacemann,
                model_path,
                cfg.custom.model_Distributed,
                device,
                excel,
                "PEACEMANN",
            )
            models["peacemann"] = fno_supervised_peacemann
            if "SWAT" in output_variables:
                if dist.rank == 0:
                    logger.info("🟣 Loading Surrogate Model for Saturation")
                model_path = (
                    os.path.join(
                        base_paths["saturation"], "pino_saturation_forward_model.pth"
                    )
                    if excel == 1
                    else os.path.join(base_paths["saturation"], "checkpoint.pth")
                )
                fno_supervised_saturation = load_modell(
                    fno_supervised_saturation,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "SWAT",
                )
                models["saturation"] = fno_supervised_saturation
            if "SOIL" in output_variables:
                if dist.rank == 0:
                    logger.info("🟣 Loading Surrogate Model for oil")
                model_path = (
                    os.path.join(base_paths["oil"], "pino_oil_forward_model.pth")
                    if excel == 1
                    else os.path.join(base_paths["oil"], "checkpoint.pth")
                )
                fno_supervised_oil = load_modell(
                    fno_supervised_oil,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "SOIL",
                )
                models["oil"] = fno_supervised_oil
    else:
        if cfg.custom.fno_type == "FNO":
            os.chdir("../MODELS/TRANSOLVER")
            if dist.rank == 0:
                logger.info(
                    "|-----------------------------------------------------------------|"
                )
                logger.info(
                    "|                     TRANSOLVER MODEL LEARNING    :              |"
                )
                logger.info(
                    "|-----------------------------------------------------------------|"
                )
            models = {}
            if "PRESSURE" in output_variables:
                if dist.rank == 0:
                    logger.info("🟢 Loading Surrogate Model for Pressure")
                model_path = (
                    os.path.join(
                        base_paths["pressure"], "transolver_pressure_forward_model.pth"
                    )
                    if excel == 1
                    else os.path.join(base_paths["pressure"], "checkpoint.pth")
                )
                fno_supervised_pressure = load_modell(
                    fno_supervised_pressure,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "PRESSURE",
                )
                models["pressure"] = fno_supervised_pressure
            if "SGAS" in output_variables:
                if dist.rank == 0:
                    logger.info("🟠 Loading Surrogate Model for Gas")
                model_path = (
                    os.path.join(base_paths["gas"], "transolver_gas_forward_model.pth")
                    if excel == 1
                    else os.path.join(base_paths["gas"], "checkpoint.pth")
                )
                fno_supervised_gas = load_modell(
                    fno_supervised_gas,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "SGAS",
                )
                models["gas"] = fno_supervised_gas
            if dist.rank == 0:
                logger.info("🔵 Loading Surrogate Model for Peacemann")
            model_path = (
                os.path.join(base_paths["peacemann"], "fno_peacemann_forward_model.pth")
                if excel == 1
                else os.path.join(base_paths["peacemann"], "checkpoint.pth")
            )
            fno_supervised_peacemann = load_modell(
                fno_supervised_peacemann,
                model_path,
                cfg.custom.model_Distributed,
                device,
                excel,
                "PEACEMANN",
            )
            models["peacemann"] = fno_supervised_peacemann
            if "SWAT" in output_variables:
                if dist.rank == 0:
                    logger.info("🟣 Loading Surrogate Model for Saturation")
                model_path = (
                    os.path.join(
                        base_paths["saturation"],
                        "transolver_saturation_forward_model.pth",
                    )
                    if excel == 1
                    else os.path.join(base_paths["saturation"], "checkpoint.pth")
                )
                fno_supervised_saturation = load_modell(
                    fno_supervised_saturation,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "SWAT",
                )
                models["saturation"] = fno_supervised_saturation
            if "SOIL" in output_variables:
                if dist.rank == 0:
                    logger.info("🟣 Loading Surrogate Model for oil")
                model_path = (
                    os.path.join(base_paths["oil"], "transolver_oil_forward_model.pth")
                    if excel == 1
                    else os.path.join(base_paths["oil"], "checkpoint.pth")
                )
                fno_supervised_oil = load_modell(
                    fno_supervised_oil,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "SOIL",
                )
                models["oil"] = fno_supervised_oil
        else:
            os.chdir("../MODELS/PI-TRANSOLVER")
            if dist.rank == 0:
                logger.info(
                    "|-----------------------------------------------------------------|"
                )
                logger.info(
                    "|                     PI-TRANSOLVER MODEL LEARNING    :           |"
                )
                logger.info(
                    "|-----------------------------------------------------------------|"
                )
            models = {}
            if "PRESSURE" in output_variables:
                if dist.rank == 0:
                    logger.info("🟢 Loading Surrogate Model for Pressure")
                model_path = (
                    os.path.join(
                        base_paths["pressure"],
                        "pi-transolver_pressure_forward_model.pth",
                    )
                    if excel == 1
                    else os.path.join(base_paths["pressure"], "checkpoint.pth")
                )
                fno_supervised_pressure = load_modell(
                    fno_supervised_pressure,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "PRESSURE",
                )
                models["pressure"] = fno_supervised_pressure
            if "SGAS" in output_variables:
                if dist.rank == 0:
                    logger.info("🟠 Loading Surrogate Model for Gas")
                model_path = (
                    os.path.join(
                        base_paths["gas"], "pi-transolver_gas_forward_model.pth"
                    )
                    if excel == 1
                    else os.path.join(base_paths["gas"], "checkpoint.pth")
                )
                fno_supervised_gas = load_modell(
                    fno_supervised_gas,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "SGAS",
                )
                models["gas"] = fno_supervised_gas
            if dist.rank == 0:
                logger.info("🔵 Loading Surrogate Model for Peacemann")
            model_path = (
                os.path.join(
                    base_paths["peacemann"], "pino_peacemann_forward_model.pth"
                )
                if excel == 1
                else os.path.join(base_paths["peacemann"], "checkpoint.pth")
            )
            fno_supervised_peacemann = load_modell(
                fno_supervised_peacemann,
                model_path,
                cfg.custom.model_Distributed,
                device,
                excel,
                "PEACEMANN",
            )
            models["peacemann"] = fno_supervised_peacemann
            if "SWAT" in output_variables:
                if dist.rank == 0:
                    logger.info("🟣 Loading Surrogate Model for Saturation")
                model_path = (
                    os.path.join(
                        base_paths["saturation"],
                        "pi-transolver_saturation_forward_model.pth",
                    )
                    if excel == 1
                    else os.path.join(base_paths["saturation"], "checkpoint.pth")
                )
                fno_supervised_saturation = load_modell(
                    fno_supervised_saturation,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "SWAT",
                )
                models["saturation"] = fno_supervised_saturation
            if "SOIL" in output_variables:
                if dist.rank == 0:
                    logger.info("🟣 Loading Surrogate Model for oil")
                model_path = (
                    os.path.join(
                        base_paths["oil"], "pi-transolver_oil_forward_model.pth"
                    )
                    if excel == 1
                    else os.path.join(base_paths["oil"], "checkpoint.pth")
                )
                fno_supervised_oil = load_modell(
                    fno_supervised_oil,
                    model_path,
                    cfg.custom.model_Distributed,
                    device,
                    excel,
                    "SOIL",
                )
                models["oil"] = fno_supervised_oil
    os.chdir(oldfolder)

    Trainmoe = cfg.custom.INVERSE_PROBLEM.Train_Moe
    if Trainmoe == "MoE":
        TEMPLATEFILE["Peaceman modelling inference"] = (
            "Inference peacemann = Mixture of Experts"
        )
    else:
        TEMPLATEFILE["Peaceman modelling inference"] = "Inference peacemann = FNO"
    if Trainmoe == "MoE":
        if dist.rank == 0:
            logger.info(
                "------------------------------------------------------L-----------"
            )
            logger.info(
                "Using Cluster Classify Regress (CCR) for peacemann model          "
            )
            logger.info("References for CCR include: ")
            logger.info(
                "(1): David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park,\n\
Kody J. H. Law, and Clement Etienam. Cluster, classify, regress:A general\n\
method for learning discontinuous functions.Foundations of Data Science,\n\
1(2639-8001-2019-4-491):491, 2019.\n"
            )
            logger.info(
                "(2): Clement Etienam, Kody Law, Sara Wade. Ultra-fast Deep Mixtures of\n\
Gaussian Process Experts. arXiv preprint arXiv:2006.13309, 2020.\n"
            )
            logger.info(
                "-----------------------------------------------------------------------"
            )
        pred_type = 1
    else:
        pred_type = 1
    degg = 3
    rho = 1.05
    aay1 = minK
    bby1 = maxK
    Low_K1, High_K1 = aay1, bby1
    perm_high = maxK
    perm_low = minK
    High_P, Low_P = 0.5, 0.05
    poro_high = High_P
    poro_low = Low_P
    High_K, Low_K, High_P, Low_P = perm_high, perm_low, poro_high, poro_low

    # Recreate RESULTS/HM_RESULTS — rank 0 only, then barrier
    if dist.rank == 0:
        target = to_absolute_path("../RESULTS/HM_RESULTS")
        if os.path.exists(target):
            shutil.rmtree(target)
        os.makedirs(target, exist_ok=True)
    _barrier()

    if DEFAULT == "Yes":
        BASSE = "Percentage of data value"
        if dist.rank == 0:
            logger.info(
                "Covarance data noise matrix using percentage of measured value"
            )
    else:
        BASSE = cfg.custom.INVERSE_PROBLEM.CD_matrix
    if BASSE == "Percentage of data value":
        TEMPLATEFILE["Covariance matrix generation"] = (
            "Covariance noise matrix generation = data percentage\n"
        )
    else:
        TEMPLATEFILE["Covariance matrix generation"] = (
            "Covariance noise matrix generation = constant value\n"
        )

    os.chdir(to_absolute_path(cfg.custom.file_location))
    Time = Get_Time(nx, ny, nz, steppi, steppi_indices, 1)
    Time_unie1 = np.zeros(steppi)
    for i in range(steppi):
        Time_unie1[i] = Time[0, i, 0, 0, 0]
    os.chdir(oldfolder)
    os.chdir(cfg.custom.file_location)
    timestep = np.genfromtxt(to_absolute_path("../simulator_data/timestep.out"))
    timestep = timestep.astype(int)
    os.chdir(oldfolder)

    if dist.rank == 0:
        logger.info("Read Historical data")

    # Re-seed before drawing indii so all ranks pick the same value.
    _safe_seed(cfg, dist)
    indii = np.random.randint(1, cfg.custom.ntrain)

    _, _True_data1, True_mat = historydata(timestep, steppi, steppi_indices, N_pr)
    True_mat[True_mat <= 0] = 0

    if dist.rank == 0:
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
        # plot_rsm_singleT(True_mat[:, :N_pr],          Time_unie1, N_pr, well_names, "WOPR")
        # plot_rsm_singleT(True_mat[:, N_pr:2*N_pr],    Time_unie1, N_pr, well_names, "WWPR")
        plot_rsm_singleT(
            True_mat[:, 2 * N_pr : 3 * N_pr], Time_unie1, N_pr, well_names, "WGPR"
        )
        os.chdir(oldfolder)
    _barrier()

    True_K = perm_ensembley[:, indii]
    # quant_big = {}
    # for k in range(lenwels):
    # quantt = {}
    # ajes, bjes, cjes = scale_array(True_mat[:, k * N_pr : (k + 1) * N_pr])
    # quantt["value"] = ajes
    # quantt["scale"] = bjes
    # quantt["boolean"] = cjes
    # quant_big[f"K_{k}"] = quantt
    True_data = np.reshape(_True_data1, (-1, 1), "F")
    # rows_to_remove = np.where(True_data <= 1e-4)[0]

    # rows_to_remove, keep_mask, True_data = find_small_rows(_True_data1, threshold=1e-4)
    mask = get_keep_mask(_True_data1, threshold=1e-4)
    True_data = True_data[mask]
    # sdoperat = np.std(True_data, axis=0).reshape(1, -1)
    # menoperat = np.mean(True_data, axis=0).reshape(1, -1)
    True_dataTI = True_data.reshape(-1, 1)
    Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)
    return (
        models,
        TEMPLATEFILE,
        True_data,
        True_mat,
        True_dataTI,
        mask,
        Time_unie1,
        timestep,
        indii,
        Low_K1,
        High_K1,
        Low_K,
        High_K,
        Low_P,
        High_P,
        pred_type,
        degg,
        rho,
        Trainmoe,
        BASSE,
        Time,
        True_K,
    )


def NorneInitialEnsemble(nx, ny, nz, ensembleSize=100, randomNumber=1.2345e5):
    """Create NORNE-shaped initial ensemble from random seeds."""
    np.random.seed(int(randomNumber))
    N = ensembleSize
    norne = NorneGeostat(nx, ny, nz)
    A = norne["actnum"]
    D = norne["dim"]
    N_F = D[0] * D[1] * D[2]
    M = [norne["poroMean"], norne["permxLogMean"], 0.6]
    S = [norne["poroStd"], norne["permxStd"], norne["ntgStd"]]
    A_L = [A[i : i + D[1] * D[2]] for i in range(0, len(A), D[1] * D[2])]
    A_L = np.array(A_L)
    M_MF = 0.6
    S_MF = norne["multfltStd"]
    C = [norne["poroRange"], norne["permxRange"], norne["ntgRange"]]
    C_S = 2
    R1 = norne["poroPermxCorr"]
    ensembleperm = np.zeros((N_F, N))
    ensemblefault = np.zeros((53, N))
    ensembleporo = np.zeros((N_F, N))
    indices = np.where(A == 1)
    for i in range(N):
        A_MZ = A_L[:, [0, 7, 10, 11, 14, 17]]
        A_MZ = A_MZ.flatten()
        X = M_MF + S_MF * np.random.randn(53)
        ensemblefault[:, i] = X
        C = np.array(C)
        X1 = gaussian_with_variable_parameters(D, np.zeros(N_F), 1, C[0], C_S)[0]
        X1 = X1.reshape(-1, 1)
        ensembleporo[indices, i] = (M[0] + S[0] * X1[indices]).ravel()
        X2 = gaussian_with_variable_parameters(D, np.zeros(N_F), 1, C[1], C_S)[0]
        X2 = X2.reshape(-1, 1)
        X = R1 * X1 + np.sqrt(1 - R1**2) * X2
        indices = np.where(A == 1)
        ensembleperm[indices, i] = np.exp((M[1] + S[1] * X[indices]).ravel())
    return ensembleperm, ensembleporo, ensemblefault


def gaussian_with_variable_parameters(
    field_dim, mean_value, sdev, mean_corr_length, std_corr_length
):
    """Sample Gaussian field with variable variance/correlation across layers."""
    corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)
    if len(field_dim) < 3:
        x = mean_value + fast_gaussian(field_dim, sdev, corr_length)
    else:
        layer_dim = np.prod(field_dim[:2])
        x = np.copy(mean_value)
        if np.isscalar(sdev):
            for i in range(field_dim[2]):
                idx_range = slice(i * layer_dim, (i + 1) * layer_dim)
                x[idx_range] = mean_value[idx_range] + fast_gaussian(
                    field_dim[:2], sdev, corr_length
                )
                corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)
        else:
            for i in range(field_dim[2]):
                idx_range = slice(i * layer_dim, (i + 1) * layer_dim)
                x[idx_range] = mean_value[idx_range] + fast_gaussian(
                    field_dim[:2], sdev[idx_range], corr_length
                )
                corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)
    return x, corr_length


def NorneGeostat(nx, ny, nz):
    """Compute NORNE geostatistics (means/stds and correlations per layer)."""
    norne = {}
    dim = np.array([nx, ny, nz])
    ldim = dim[0] * dim[1]
    norne["dim"] = dim
    act = read_until_line("../simulator_data/ACTNUM_0704.prop")
    act = act.T
    act = np.reshape(act, (-1,), "F")
    norne["actnum"] = act
    meanv = np.zeros(dim[2])
    stdv = np.zeros(dim[2])
    file_path = "../simulator_data/porosity.dat"
    p = read_until_line(file_path)
    p = p[act != 0]
    for nr in range(int(dim[2])):
        index_start = ldim * nr
        index_end = ldim * (nr + 1)
        values_range_start = int(np.sum(act[:index_start]))
        values_range_end = int(np.sum(act[:index_end]))
        values = p[values_range_start:values_range_end]
        meanv[nr] = np.mean(values)
        stdv[nr] = np.std(values)
    norne["poroMean"] = p
    norne["poroLayerMean"] = meanv
    norne["poroLayerStd"] = stdv
    norne["poroStd"] = 0.05
    norne["poroLB"] = 0.1
    norne["poroUB"] = 0.4
    norne["poroRange"] = 26
    k = read_until_line("../simulator_data/permx.dat")
    k = np.log(k)
    k = k[act != 0]
    meanv = np.zeros(dim[2])
    stdv = np.zeros(dim[2])
    for nr in range(int(dim[2])):
        index_start = ldim * nr
        index_end = ldim * (nr + 1)
        values_range_start = int(np.sum(act[:index_start]))
        values_range_end = int(np.sum(act[:index_end]))
        values = k[values_range_start:values_range_end]
        meanv[nr] = np.mean(values)
        stdv[nr] = np.std(values)
    norne["permxLogMean"] = k
    norne["permxLayerLnMean"] = meanv
    norne["permxLayerStd"] = stdv
    norne["permxStd"] = 1
    norne["permxLB"] = 0.1
    norne["permxUB"] = 10
    norne["permxRange"] = 26
    corr_with_next_layer = np.zeros(dim[2] - 1)
    for nr in range(dim[2] - 1):
        index_start = ldim * nr
        index_end = ldim * (nr + 1)
        index2_start = ldim * (nr + 1)
        index2_end = ldim * (nr + 2)
        act_layer1 = act[index_start:index_end]
        act_layer2 = act[index2_start:index2_end]
        active = act_layer1 * act_layer2
        values1_range_start = int(np.sum(act[:index_start]))
        values1_range_end = int(np.sum(act[:index_end]))
        values1 = np.concatenate(
            (
                k[values1_range_start:values1_range_end],
                p[values1_range_start:values1_range_end],
            )
        )
        values2_range_start = int(np.sum(act[:index2_start]))
        values2_range_end = int(np.sum(act[:index2_end]))
        values2 = np.concatenate(
            (
                k[values2_range_start:values2_range_end],
                p[values2_range_start:values2_range_end],
            )
        )
        v1 = np.concatenate((act_layer1, act_layer1))
        v1[v1 == 1] = values1.flatten()
        v2 = np.concatenate((act_layer2, act_layer2))
        v2[v2 == 1] = values2.flatten()
        active_full = np.concatenate((active, active))
        co = np.corrcoef(v1[active_full == 1], v2[active_full == 1])
        corr_with_next_layer[nr] = co[0, 1]
    norne["corr_with_next_layer"] = corr_with_next_layer.T
    norne["poroPermxCorr"] = 0.7
    norne["poroNtgCorr"] = 0.6
    norne["ntgStd"] = 0.1
    norne["ntgLB"] = 0.01
    norne["ntgUB"] = 1
    norne["ntgRange"] = 26
    norne["krwMean"] = 1.15
    norne["krwLB"] = 0.8
    norne["krwUB"] = 1.5
    norne["krgMean"] = 0.9
    norne["krgLB"] = 0.8
    norne["krgUB"] = 1
    norne["owcMean"] = np.array([2692.0, 2585.5, 2618.0, 2400.0, 2693.3])
    norne["owcLB"] = norne["owcMean"] - 10
    norne["owcUB"] = norne["owcMean"] + 10
    norne["multregtLogMean"] = np.log10(np.array([0.0008, 0.1, 0.05]))
    norne["multregtStd"] = 0.5
    norne["multregtLB"] = -5
    norne["multregtUB"] = 0
    z_means = [-2, -1.3, -2, -2, -2, -2]
    z_stds = [0.5, 0.5, 0.5, 0.5, 1, 1]
    for i, (mean_, std_) in enumerate(zip(z_means, z_stds, strict=False), start=1):
        norne[f"z{i}Mean"] = mean_
        norne[f"z{i}Std"] = std_
    norne["zLB"] = -4
    norne["zUB"] = 0
    norne["multzRange"] = 26
    norne["multfltStd"] = 0.5
    norne["multfltLB"] = -5
    norne["multfltUB"] = 2
    return norne


def compute_rowwise_scaling(True_data: np.ndarray) -> np.ndarray:
    """
    Build a (Nop, 1) scaling vector that brings each value into roughly 1-digit range
    (i.e. magnitude in [1, 10)). Works in both directions:

    Examples:
        60000  → scale = 0.0001   → scaled = 6.0    ✓
        500    → scale = 0.01     → scaled = 5.0    ✓
        50     → scale = 0.1      → scaled = 5.0    ✓
        5      → scale = 1.0      → scaled = 5.0    ✓  (already 1-digit)
        0.5    → scale = 10.0     → scaled = 5.0    ✓
        0.001  → scale = 1000.0   → scaled = 1.0    ✓
        0.0    → scale = 1.0      → scaled = 0.0    ✓  (guard, untouched)
    """
    Nop = True_data.shape[0]
    scale_vec = np.ones((Nop, 1), dtype=np.float64)

    for i in range(Nop):
        val = abs(float(True_data[i, 0]))
        if val == 0.0:
            continue
        # Bring val into [1, 10): find the exponent needed
        exp = np.floor(np.log10(val))
        scale_vec[i, 0] = 10.0 ** (-exp)

    return scale_vec  # (Nop, 1)


def generate_ensemble(
    cfg,
    dist,
    device,
    oldfolder: str,
    TEMPLATEFILE: dict,
    gpu_available: bool,
    grid: "GridConfig",
    perm: "PermBounds",
    ens: "EnsembleSetup",
    well: "WellConfig",
    time_arr: "TimeArrays",
    inversion: "InversionParams",
) -> tuple:
    """Top-level ensemble generation pipeline orchestrating all steps.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing all user-specified settings.
    dist : DistributedManager
        PhysicsNeMo distributed-training context; used for rank-gated logging.
    device : torch.device
        Compute device (CUDA or CPU) for tensor operations.
    oldfolder : str
        Working directory to restore after file I/O operations.
    TEMPLATEFILE : dict
        Mutable metadata dict written to the YAML summary at the end of the
        run (keys are human-readable labels, values are settings).
    gpu_available : bool
        ``True`` when at least one CUDA-capable GPU is detected.
    grid : GridConfig
        Grid dimensions (nx, ny, nz) and time-step indexing (steppi,
        steppi_indices).
    perm : PermBounds
        Prior bounds for permeability (High_K, Low_K, High_K1, Low_K1) and
        porosity (High_P, Low_P).
    ens : EnsembleSetup
        Ensemble size (Ne, N_ens), active-cell masks (effec), quantile arrays
        (quant_big, rows_to_remove), and deletion indices (indii).
    well : WellConfig
        Well identifiers (well_names), producer count (N_pr), and well-channel
        count (lenwels).
    time_arr : TimeArrays
        Time arrays used by the data-extraction helpers (timestep, Time_unie1).
    inversion : InversionParams
        Algorithmic settings including noise_level applied to synthetic
        observations.

    Returns
    -------
    tuple
        (ensemble, ensemblep, ensemblef, ini_ensemble, ini_ensemblep,
        ini_ensemblefault, True_data, True_mat, dt, Nop, CDd,
        perturbations, start_time)
    """
    Ne = ens.Ne
    N_ens = ens.N_ens
    effec = ens.effec
    indii = ens.indii
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    steppi = grid.steppi
    steppi_indices = grid.steppi_indices
    High_K, Low_K = perm.High_K, perm.Low_K
    High_K1, Low_K1 = perm.High_K1, perm.Low_K1
    High_P, Low_P = perm.High_P, perm.Low_P
    well_names = well.well_names
    N_pr = well.N_pr
    timestep = time_arr.timestep
    Time_unie1 = time_arr.Time_unie1
    noise_level = inversion.noise_level

    # Seed identically across ranks
    _safe_seed(cfg, dist)

    if dist.rank == 0:
        logger.info("****************************************************************")
        logger.info("                     Generating ensemble                        ")
        logger.info("****************************************************************")

    if Ne == int(cfg.custom.ntrain):
        with gzip.open(to_absolute_path("../data/static.pkl.gz"), "rb") as f2:
            mat = pickle.load(f2)
        X_data1 = mat
        for key, value in X_data1.items():
            if dist.rank == 0:
                logger.info(
                    "****************************************************************"
                )
                logger.info(f"For key '{key}':")
                logger.info("\tContains inf: %s", np.isinf(value).any())
                logger.info("\tContains -inf: %s", np.isinf(-value).any())
                logger.info("\tContains NaN: %s", np.isnan(value).any())
                logger.info(
                    "****************************************************************"
                )
        perm_ensemble = X_data1["ensemble"]
        poro_ensemble = X_data1["ensemblep"]
        fault_ensemblep = X_data1["ensemblefault"]
        perm_ensemble = np.delete(perm_ensemble, indii, axis=1)
        poro_ensemble = np.delete(poro_ensemble, indii, axis=1)
        fault_ensemblep = np.delete(fault_ensemblep, indii, axis=1)
        Neuse = 1
        permf, porof, faultf = NorneInitialEnsemble(
            nx, ny, nz, ensembleSize=Neuse, randomNumber=1.2345e5
        )
        ini_ensemble = np.hstack((permf, perm_ensemble))
        ini_ensemblep = np.hstack((porof, poro_ensemble))
        ini_ensemblefault = np.hstack((faultf, fault_ensemblep))
        outt = {"PERM": ini_ensemble, "PORO": ini_ensemblep}
        outt = clip_ensemble_params(
            outt, nx, ny, nz, N_ens, High_K, Low_K, High_P, Low_P, effec
        )
        ini_ensemble = outt["PERM"]
        ini_ensemblep = outt["PORO"]
        os.chdir(to_absolute_path(cfg.custom.file_location))
        Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)
        Time_unie = np.zeros(steppi)
        for i in range(steppi):
            Time_unie[i] = Time[0, i, 0, 0, 0]
        _, _True_data1, True_mat = historydata(timestep, steppi, steppi_indices, N_pr)
        True_mat[True_mat <= 0] = 0
        if dist.rank == 0:
            os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
            # plot_rsm_singleT(True_mat[:, :N_pr],         Time_unie1, N_pr, well_names, "WOPR")
            # plot_rsm_singleT(True_mat[:, N_pr:2*N_pr],   Time_unie1, N_pr, well_names, "WWPR")
            plot_rsm_singleT(
                True_mat[:, 2 * N_pr : 3 * N_pr], Time_unie1, N_pr, well_names, "WGPR"
            )
            os.chdir(oldfolder)
        _barrier()
        True_data = np.reshape(_True_data1, (-1, 1), "F")
        mask = get_keep_mask(True_data, threshold=1e-4)
        True_data = np.reshape(_True_data1, (-1, 1), "F")[mask]
        Nop = True_data.shape[0]
        os.chdir(oldfolder)
        dt = Time_unie

    if (Ne > int(cfg.custom.ntrain)) and (Ne < 5000):
        os.chdir(to_absolute_path(cfg.custom.file_location))
        Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)
        Time_unie = np.zeros(steppi)
        for i in range(steppi):
            Time_unie[i] = Time[0, i, 0, 0, 0]
        os.chdir(oldfolder)
        dt = Time_unie
        with gzip.open(to_absolute_path("../data/static.pkl.gz"), "rb") as f2:
            mat = pickle.load(f2)
        X_data1 = mat
        for key, value in X_data1.items():
            if dist.rank == 0:
                logger.info(
                    "****************************************************************"
                )
                logger.info(f"For key '{key}':")
                logger.info("\tContains inf: %s", np.isinf(value).any())
                logger.info("\tContains -inf: %s", np.isinf(-value).any())
                logger.info("\tContains NaN: %s", np.isnan(value).any())
                logger.info(
                    "****************************************************************"
                )
        perm_ensemble = X_data1["ensemble"]
        poro_ensemble = X_data1["ensemblep"]
        fault_ensemblep = X_data1["ensemblefault"]
        perm_ensemble = np.delete(perm_ensemble, indii, axis=1)
        poro_ensemble = np.delete(poro_ensemble, indii, axis=1)
        fault_ensemblep = np.delete(fault_ensemblep, indii, axis=1)
        _, _True_data1, True_mat = historydata(timestep, steppi, steppi_indices, N_pr)
        True_mat[True_mat <= 0] = 0
        if dist.rank == 0:
            os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
            # plot_rsm_singleT(True_mat[:, :N_pr],         Time_unie1, N_pr, well_names, "WOPR")
            # plot_rsm_singleT(True_mat[:, N_pr:2*N_pr],   Time_unie1, N_pr, well_names, "WWPR")
            plot_rsm_singleT(
                True_mat[:, 2 * N_pr : 3 * N_pr], Time_unie1, N_pr, well_names, "WGPR"
            )
            os.chdir(oldfolder)
        _barrier()
        True_data = np.reshape(_True_data1, (-1, 1), "F")
        mask = get_keep_mask(True_data, threshold=1e-4)
        True_data = np.reshape(_True_data1, (-1, 1), "F")[mask]
        Neuse = int(Ne - int(cfg.custom.ntrain)) + 1
        permf, porof, faultf = NorneInitialEnsemble(
            nx, ny, nz, ensembleSize=Neuse, randomNumber=1.2345e5
        )
        ini_ensemble = np.hstack((permf, perm_ensemble))
        ini_ensemblep = np.hstack((porof, poro_ensemble))
        ini_ensemblefault = np.hstack((faultf, fault_ensemblep))

    if Ne < int(cfg.custom.ntrain):
        # Re-seed before draws so all ranks pick same indices
        _safe_seed(cfg, dist)
        indices = np.random.choice(Ne, size=Ne, replace=False)
        with gzip.open(to_absolute_path("../data/static.pkl.gz"), "rb") as f2:
            mat = pickle.load(f2)
        X_data1 = mat
        for key, value in X_data1.items():
            if dist.rank == 0:
                logger.info(
                    "****************************************************************"
                )
                logger.info(f"For key '{key}':")
                logger.info("\tContains inf: %s", np.isinf(value).any())
                logger.info("\tContains -inf: %s", np.isinf(-value).any())
                logger.info("\tContains NaN: %s", np.isnan(value).any())
                logger.info(
                    "****************************************************************"
                )
        ini_ensemble = X_data1["ensemble"]
        ini_ensemblep = X_data1["ensemblep"]
        ini_ensemblefault = X_data1["ensemblefault"]
        ini_ensemble = np.delete(ini_ensemble, indii, axis=1)
        ini_ensemblep = np.delete(ini_ensemblep, indii, axis=1)
        ini_ensemblefault = np.delete(ini_ensemblefault, indii, axis=1)
        ini_ensemble = ini_ensemble[:, indices]
        ini_ensemblep = ini_ensemblep[:, indices]
        ini_ensemblefault = ini_ensemblefault[:, indices]
        os.chdir(to_absolute_path(cfg.custom.file_location))
        Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)
        Time_unie = np.zeros(steppi)
        for i in range(steppi):
            Time_unie[i] = Time[0, i, 0, 0, 0]
        _, _True_data1, True_mat = historydata(timestep, steppi, steppi_indices, N_pr)
        True_mat[True_mat <= 0] = 0
        if dist.rank == 0:
            os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
            # plot_rsm_singleT(True_mat[:, :N_pr],         Time_unie1, N_pr, well_names, "WOPR")
            # plot_rsm_singleT(True_mat[:, N_pr:2*N_pr],   Time_unie1, N_pr, well_names, "WWPR")
            plot_rsm_singleT(
                True_mat[:, 2 * N_pr : 3 * N_pr], Time_unie1, N_pr, well_names, "WGPR"
            )
            os.chdir(oldfolder)
        _barrier()
        True_data = np.reshape(_True_data1, (-1, 1), "F")
        mask = get_keep_mask(True_data, threshold=1e-4)
        True_data = np.reshape(_True_data1, (-1, 1), "F")[mask]
        Nop = True_data.shape[0]
        os.chdir(oldfolder)
        dt = Time_unie

    TEMPLATEFILE["Ensemble size"] = Ne
    if dist.rank == 0:
        logger.info(
            "----------------------------------------------------------------------"
        )
        logger.info(
            "              History Matching Operational conditions                 "
        )
        logger.info(
            "----------------------------------------------------------------------"
        )
        for key, value in TEMPLATEFILE.items():
            logger.info(f"{key}: {value}")

    yaml_filename = to_absolute_path(
        "../RESULTS/HM_RESULTS/History_Matching_Template_file.yaml"
    )
    if dist.rank == 0:
        with open(yaml_filename, "w") as yaml_file:
            yaml.dump(TEMPLATEFILE, yaml_file)
    _barrier()

    start_time = time.time()
    if dist.rank == 0:
        print("----------------------------------------------------------------------")
        print(
            "----------------Starting the History matching with - ",
            str(Ne) + " Ensemble members  ",
        )
        print("****************************************************************")
    os.chdir(oldfolder)

    ensemble = ini_ensemble
    ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
    ensemblep = ini_ensemblep
    ensemblef = ini_ensemblefault
    outt = {"PERM": ensemble, "PORO": ensemblep}
    outt = clip_ensemble_params(
        outt, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P, effec
    )
    ensemble = outt["PERM"]
    ensemblep = outt["PORO"]

    Nop = True_data.shape[0]
    ax = np.zeros((Nop, 1))

    scale_mat = compute_rowwise_scaling(np.reshape(True_data, (-1, 1), "F"))
    True_data_temp = np.reshape(True_data, (-1, 1), "F") * scale_mat
    for iq in range(Nop):
        if (True_data_temp[iq, :] > 0) and (True_data_temp[iq, :] <= 1e10):
            ax[iq, :] = sqrt(noise_level * True_data_temp[iq, :])
        else:
            ax[iq, :] = 1
    R = ax**2
    R = torch.as_tensor(R, dtype=torch.float32).to(device)
    CDd = torch.diag(R.view(-1))
    Cini = CDd.clone().to(device)

    # Sample perturbations deterministically on CPU so every rank produces
    # bit-identical noise. CUDA RNG is non-deterministic across different
    # physical devices even with the same seed, so we sample on CPU using
    # an explicit generator object, then move the result to numpy.
    seed = int(getattr(cfg.custom, "seed", 12345))

    Cini_cpu = Cini.detach().cpu().to(torch.float32)
    mean_cpu = torch.zeros(Nop, dtype=torch.float32)

    # Cholesky factor of the covariance — Cini = L L^T
    L = torch.linalg.cholesky(Cini_cpu)  # (Nop, Nop)

    # Standard normal samples with explicit generator (deterministic)
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(Ne, Nop, generator=g_cpu)  # (Ne, Nop)

    # Transform to N(mean, Cini): each row of z @ L.T is one sample
    perturbations = (mean_cpu + z @ L.T).T  # (Nop, Ne)
    perturbations = perturbations.detach().cpu().numpy()
    return (
        ensemble,
        ensemblep,
        ensemblef,
        ini_ensemble,
        ini_ensemblep,
        ini_ensemblefault,
        True_data,
        True_mat,
        dt,
        Nop,
        CDd,
        perturbations,
        start_time,
    )
