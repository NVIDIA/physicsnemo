"""
SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
                    PERCENTILE ENSEMBLE UTILITIES MODULE
=====================================================================

This module provides percentile ensemble utilities for inverse problems
in reservoir simulation. It includes percentile analysis, visualization,
and statistical processing utilities.

Key Features:
- Percentile model analysis
- Statistical visualization
- Ensemble comparison utilities
- Data processing tools

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import shutil
import logging

# 🔧 Third-party Libraries
import numpy as np
import numpy.matlib
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
import joblib

# 📦 Local Modules
from inverse.inversion_operation_surrogate import (
    Forward_model_ensemble,
)
from inverse.inversion_operation_gather import (
    Plot_RSM_percentile,
)
from inverse.inversion_operation_misc import (
    Plot_PhyNeMo,
)
from inverse.inversion_operation_ensemble import (
    ensemble_pytorch,
)


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Inverse problem")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logging()


def plot_percentile_models(
    # Ensemble data
    ensembleout,
    ensembleoutf1,
    # Model parameters
    nx,
    ny,
    nz,
    effective,
    oldfolder,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    minQ,
    maxQ,
    minQw,
    maxQw,
    minQg,
    maxQg,
    steppi,
    device,
    steppi_indices,
    input_variables,
    cfg,
    # Forward model parameters
    models,
    min_inn_fcn,
    max_inn_fcn,
    min_out_fcn,
    max_out_fcn,
    Time,
    effectiveuse,
    Trainmoe,
    num_cores,
    pred_type,
    degg,
    experts,
    min_out_fcn2,
    max_out_fcn2,
    min_inn_fcn2,
    max_inn_fcn2,
    producers,
    compdat_data,
    output_variables,
    quant_big,
    N_pr,
    lenwels,
    effective_abi,
    rows_to_remove,
    # True data and models
    True_mat,
    Time_unie1,
    well_names,
    True_K,
    base_k,
    # Control models from previous results
    X_data1,
    # Additional parameters
    dist,
    # Well configuration
    N_injw,
    N_injg,
    injectors,
    gass,
):
    if dist.rank == 0:
        print("****************************************************************")
        print("          PLOT P10,P50,P90 RESERVOIR UQ MODELS                   ")
        print("****************************************************************")
        if not os.path.exists(to_absolute_path("../RESULTS/HM_RESULTS/PERCENTILE")):
            os.makedirs(
                to_absolute_path("../RESULTS/HM_RESULTS/PERCENTILE"), exist_ok=True
            )
        else:
            shutil.rmtree(to_absolute_path("../RESULTS/HM_RESULTS/PERCENTILE"))
            os.makedirs(
                to_absolute_path("../RESULTS/HM_RESULTS/PERCENTILE"), exist_ok=True
            )
    ensemblepy = ensemble_pytorch(
        ensembleout,
        nx,
        ny,
        nz,
        ensembleout["PERM"].shape[1],
        effective,
        oldfolder,
        target_min,
        target_max,
        minK,
        maxK,
        minT,
        maxT,
        minP,
        maxP,
        minQ,
        maxQ,
        minQw,
        maxQw,
        minQg,
        maxQg,
        steppi,
        device,
        steppi_indices,
        input_variables,
        cfg,
    )
    os.chdir(oldfolder)
    # mazw = 0  # Dont smooth the presure field
    simout = Forward_model_ensemble(
        ensembleoutf1.shape[1],
        ensemblepy,
        steppi,
        min_inn_fcn,
        max_inn_fcn,
        target_min,
        target_max,
        minK,
        maxK,
        minT,
        maxT,
        minP,
        maxP,
        models,
        device,
        min_out_fcn,
        max_out_fcn,
        Time,
        effectiveuse,
        Trainmoe,
        num_cores,
        pred_type,
        oldfolder,
        degg,
        experts,
        min_out_fcn2,
        max_out_fcn2,
        min_inn_fcn2,
        max_inn_fcn2,
        producers,
        compdat_data,
        output_variables,
        quant_big,
        cfg,
        N_pr,
        lenwels,
        effective_abi,
        rows_to_remove,
        nx,
        ny,
        nz,
    )
    yzout = simout["ouut_p"]
    if "PRESSURE" in output_variables:
        pressure_percentile = simout["PRESSURE"]
    if "SWAT" in output_variables:
        water_percentile = simout["SWAT"]
    if "SOIL" in output_variables:
        oil_percentile = simout["SOIL"]
    if "SGAS" in output_variables:
        gas_percentile = simout["SGAS"]
    os.chdir("../RESULTS/HM_RESULTS/PERCENTILE")
    if dist.rank == 0:
        Plot_RSM_percentile(yzout, True_mat, Time_unie1, N_pr, well_names)
    X_data11 = {
        "PERM_Reali": ensembleout["PERM"],
        "FAULT_Reali": ensembleout["FAULT"],
        "PORO_Reali": ensembleout["PORO"],
        "Simulated_data_plots": yzout,
        "Pressures": pressure_percentile,
        "Water_saturation": water_percentile,
        "Oil_saturation": oil_percentile,
        "Gas_saturation": gas_percentile,
    }
    if dist.rank == 0:
        joblib.dump(
            X_data1,
            to_absolute_path(
                "../RESULTS/HM_RESULTS/Posterior_Ensembles_percentile.joblib"
            ),
            compress=3,
        )
        f_3 = plt.figure(figsize=(20, 20), dpi=200)
        look = ((np.reshape(True_K, (nx, ny, nz), "F")) * effectiveuse)[:, :, ::-1]
        ax1 = f_3.add_subplot(3, 3, 1, projection="3d")
        Plot_PhyNeMo(
            ax1,
            nx,
            ny,
            nz,
            look,
            N_injw,
            N_pr,
            N_injg,
            "True model",
            injectors,
            producers,
            gass,
        )
        look = ((np.reshape(base_k, (nx, ny, nz), "F")) * effectiveuse)[:, :, ::-1]
        ax1 = f_3.add_subplot(3, 3, 2, projection="3d")
        Plot_PhyNeMo(
            ax1,
            nx,
            ny,
            nz,
            look,
            N_injw,
            N_pr,
            N_injg,
            "Prior",
            injectors,
            producers,
            gass,
        )
        look = ((np.reshape(X_data1["P10_Perm"], (nx, ny, nz), "F")) * effectiveuse)[
            :, :, ::-1
        ]
        ax1 = f_3.add_subplot(3, 3, 3, projection="3d")
        Plot_PhyNeMo(
            ax1,
            nx,
            ny,
            nz,
            look,
            N_injw,
            N_pr,
            N_injg,
            "P10",
            injectors,
            producers,
            gass,
        )
        look = ((np.reshape(X_data1["P50_Perm"], (nx, ny, nz), "F")) * effectiveuse)[
            :, :, ::-1
        ]
        ax1 = f_3.add_subplot(3, 3, 4, projection="3d")
        Plot_PhyNeMo(
            ax1,
            nx,
            ny,
            nz,
            look,
            N_injw,
            N_pr,
            N_injg,
            "P50",
            injectors,
            producers,
            gass,
        )
        look = ((np.reshape(X_data1["P90_Perm"], (nx, ny, nz), "F")) * effectiveuse)[
            :, :, ::-1
        ]
        ax1 = f_3.add_subplot(3, 3, 5, projection="3d")
        Plot_PhyNeMo(
            ax1,
            nx,
            ny,
            nz,
            look,
            N_injw,
            N_pr,
            N_injg,
            "P90",
            injectors,
            producers,
            gass,
        )
        look = (
            (np.reshape(X_data1["yes_best"]["PERM"], (nx, ny, nz), "F")) * effectiveuse
        )[:, :, ::-1]
        ax1 = f_3.add_subplot(3, 3, 6, projection="3d")
        Plot_PhyNeMo(
            ax1,
            nx,
            ny,
            nz,
            look,
            N_injw,
            N_pr,
            N_injg,
            "cumm-best",
            injectors,
            producers,
            gass,
        )
        look = (
            (np.reshape(X_data1["yes_mean"]["PERM"], (nx, ny, nz), "F")) * effectiveuse
        )[:, :, ::-1]
        ax1 = f_3.add_subplot(3, 3, 7, projection="3d")
        Plot_PhyNeMo(
            ax1,
            nx,
            ny,
            nz,
            look,
            N_injw,
            N_pr,
            N_injg,
            "cumm-mean",
            injectors,
            producers,
            gass,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        tita = "Reservoir Models permeability Fields"
        plt.suptitle(tita, fontsize=16)
        plt.savefig("Reservoir_models.png")
        plt.clf()
        plt.close()

    return X_data11
