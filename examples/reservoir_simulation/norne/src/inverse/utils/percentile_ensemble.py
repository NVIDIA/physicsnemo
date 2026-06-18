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
                    PERCENTILE ENSEMBLE UTILITIES MODULE
=====================================================================

Generates P10/P50/P90 percentile reservoir model plots and saves the
posterior ensemble to disk. This version is rank-aware for multi-GPU
execution under torchrun:

- Forward model call participates from every rank (collective op).
- Plotting and disk writes are guarded with `dist.rank == 0`.
- Directory rmtree+makedirs is followed by a barrier so non-zero ranks
  don't race ahead into a directory that rank 0 is still recreating.

KNOWN ISSUE: the call to ``Forward_model_ensemble`` below passes
``quant_big`` and ``rows_to_remove`` as extra positional arguments.
These are NOT accepted by the canonical Forward_model_ensemble signature.
Either this code path is currently dead, or the signature has drifted.
Verify before running.

@Author : Clement Etienam
"""

import os
import sys
import shutil
import logging

import numpy as np
import torch.distributed as torchdist
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
import joblib

from utils.ccr_utils import (
    Forward_model_ensemble,
)
from utils.io_utils import (
    Plot_PhysicsNeMo,
)
from inverse.inversion_operation_gather import (
    plot_rsm_percentile,
)
from inverse.inversion_operation_ensemble import (
    ensemble_pytorch,
)
from inverse.utils.inverse_config import (
    GridConfig,
    NormBounds,
    EnsembleSetup,
    WellConfig,
    SurrogateConfig,
    TimeArrays,
    FlowArrays,
)


def _is_dist_active():
    """Return True if torch.distributed has been initialised."""
    return torchdist.is_available() and torchdist.is_initialized()


def _barrier():
    """Distributed barrier that no-ops in single-GPU mode."""
    if _is_dist_active():
        torchdist.barrier()


def _recreate_dir(path, dist):
    """Rank 0 deletes (if exists) and recreates `path`; all ranks barrier."""
    if dist.rank == 0:
        abs_path = to_absolute_path(path)
        if os.path.exists(abs_path):
            shutil.rmtree(abs_path)
        os.makedirs(abs_path, exist_ok=True)
    _barrier()


def setup_logging() -> logging.Logger:
    """Configure and return the main logger with green INFO console output."""
    logger = logging.getLogger("inverse problem")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on re-entry

    formatter = logging.Formatter(" %(asctime)s - %(levelname)s - %(message)s")

    f_handler = logging.FileHandler(filename="read_vectors.log", mode="w")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    class _ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: "\033[0;36m",
            logging.INFO: "\033[0;32m",
            logging.WARNING: "\033[1;33m",
            logging.ERROR: "\033[0;31m",
            logging.CRITICAL: "\033[1;31m",
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


def plot_percentile_models(
    cfg,
    dist,
    device,
    oldfolder: str,
    input_variables: list,
    output_variables: list,
    ensembleout,
    ensembleoutf1,
    base_k,
    X_data1,
    True_mat,
    True_K,
    grid: "GridConfig",
    norm: "NormBounds",
    ens: "EnsembleSetup",
    well: "WellConfig",
    surrogate: "SurrogateConfig",
    time_arr: "TimeArrays",
    flow: "FlowArrays",
):
    """Run the forward model on P10/P50/P90 ensemble members and generate percentile plots.

    Selects percentile ensemble realisations, runs the surrogate forward model,
    and saves well-response comparison plots and 3D permeability visualisations.

    Parameters
    ----------
    ensembleout : dict
        Dictionary of posterior ensemble arrays keyed by variable name (e.g. ``'PERM'``).
    ensembleoutf1 : dict
        Dictionary of prior ensemble arrays keyed by variable name.
    nx : int
        Number of grid cells in the x-direction.
    ny : int
        Number of grid cells in the y-direction.
    nz : int
        Number of grid cells in the z-direction.
    effective : numpy.ndarray
        Active cell mask array of shape (nx * ny * nz,).
    oldfolder : str
        Original working directory to restore after file operations.
    target_min : float
        Minimum value used for output de-normalisation.
    target_max : float
        Maximum value used for output de-normalisation.
    minK : float
        Minimum permeability value for de-normalisation.
    maxK : float
        Maximum permeability value for de-normalisation.
    minT : float
        Minimum transmissibility value for de-normalisation.
    maxT : float
        Maximum transmissibility value for de-normalisation.
    minP : float
        Minimum pressure value for de-normalisation.
    maxP : float
        Maximum pressure value for de-normalisation.
    minQ : float
        Minimum oil rate value for de-normalisation.
    maxQ : float
        Maximum oil rate value for de-normalisation.
    minQw : float
        Minimum water rate value for de-normalisation.
    maxQw : float
        Maximum water rate value for de-normalisation.
    minQg : float
        Minimum gas rate value for de-normalisation.
    maxQg : float
        Maximum gas rate value for de-normalisation.
    steppi : int
        Number of sequential timesteps in the simulation window.
    device : torch.device
        Device for neural-operator inference.
    steppi_indices : numpy.ndarray
        Integer indices selecting the active timesteps from the full sequence.
    input_variables : list of str
        Names of the model input channels.
    cfg : omegaconf.DictConfig
        Hydra configuration object with model and path settings.
    models : dict
        Dictionary of loaded surrogate model objects keyed by variable name.
    min_inn_fcn : float
        Global minimum of input features for normalisation.
    max_inn_fcn : float
        Global maximum of input features for normalisation.
    min_out_fcn : float
        Global minimum of output features for de-normalisation.
    max_out_fcn : float
        Global maximum of output features for de-normalisation.
    Time : numpy.ndarray
        Simulation time points array of shape (steppi,).
    active_cells_ensemble : numpy.ndarray
        Boolean active-cell mask replicated for the ensemble.
    Trainmoe : int
        Flag indicating whether MOE CCR surrogate is used (1) or neural operator (0).
    num_cores : int
        Number of CPU cores for parallel MOE CCR inference.
    pred_type : int
        Prediction variant selector for CCR inference.
    degg : int
        Polynomial degree for polynomial CCR experts.
    experts : int
        Expert type selector for CCR models.
    min_out_fcn2 : float
        Secondary output minimum for additional de-normalisation.
    max_out_fcn2 : float
        Secondary output maximum for additional de-normalisation.
    min_inn_fcn2 : float
        Secondary input minimum for additional normalisation.
    max_inn_fcn2 : float
        Secondary input maximum for additional normalisation.
    producers : list
        Well producer identifiers used for Peacemann well modelling.
    compdat_data : dict
        Completion data dictionary for Peacemann well model configuration.
    output_variables : list of str
        Names of all output variables returned by the forward model.
    quant_big : numpy.ndarray
        Quantile reference data for ensemble statistics.
    N_pr : int
        Number of producer wells in the model.
    lenwels : int
        Total number of wells (producers + injectors).
    active_mask_3d : numpy.ndarray
        3D boolean active-cell mask of shape (nx, ny, nz).
    rows_to_remove : list of int
        Row indices to exclude when reshaping well output arrays.
    True_mat : numpy.ndarray
        Observed well response matrix used as ground truth for plotting.
    Time_unie1 : numpy.ndarray
        Time axis array for the observed data.
    well_names : list of str
        Names of all wells for plot labelling.
    True_K : numpy.ndarray
        True (reference) permeability field for comparison plots.
    base_k : numpy.ndarray
        Baseline permeability realisation for comparison.
    X_data1 : numpy.ndarray
        Control input data from a previous inversion result.
    dist : physicsnemo.distributed.DistributedManager
        Distributed manager used to gate rank-0-only operations.
    N_injw : int
        Number of water injector wells.
    N_injg : int
        Number of gas injector wells.
    injectors : list
        Water injector well identifiers.
    gas_injectors : list
        Gas injector well identifiers.

    Returns
    -------
    None
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    steppi = grid.steppi
    steppi_indices = grid.steppi_indices
    effective = ens.effective
    active_cells_ensemble = ens.active_cells_ensemble
    active_mask_3d = ens.active_mask_3d
    target_min = norm.target_min
    target_max = norm.target_max
    minK, maxK = norm.minK, norm.maxK
    minT, maxT = norm.minT, norm.maxT
    minP, maxP = norm.minP, norm.maxP
    minQ, maxQ = norm.minQ, norm.maxQ
    minQw, maxQw = norm.minQw, norm.maxQw
    minQg, maxQg = norm.minQg, norm.maxQg
    min_inn_fcn, max_inn_fcn = norm.min_inn_fcn, norm.max_inn_fcn
    min_out_fcn, max_out_fcn = norm.min_out_fcn, norm.max_out_fcn
    min_inn_fcn2, max_inn_fcn2 = norm.min_inn_fcn2, norm.max_inn_fcn2
    min_out_fcn2, max_out_fcn2 = norm.min_out_fcn2, norm.max_out_fcn2
    models = surrogate.models
    Trainmoe = surrogate.Trainmoe
    pred_type = surrogate.pred_type
    degg = surrogate.degg
    experts = surrogate.experts
    num_cores = surrogate.num_cores
    producers = well.producers
    injectors = well.injectors
    gas_injectors = well.gas_injectors
    well_names = well.well_names
    N_pr = well.N_pr
    N_injw = well.N_injw
    N_injg = well.N_injg
    lenwels = well.lenwels
    compdat_data = well.compdat_data
    Time = time_arr.Time
    Time_unie1 = time_arr.Time_unie1
    awater = flow.awater
    agas = flow.agas
    aoil = flow.aoil
    aqq = flow.aqq

    if dist.rank == 0:
        print("****************************************************************")
        print("          PLOT P10,P50,P90 RESERVOIR UQ MODELS                   ")
        print("****************************************************************")

    # Recreate PERCENTILE dir on rank 0, barrier so others wait
    _recreate_dir("../RESULTS/HM_RESULTS/PERCENTILE", dist)

    # ── prepare inputs (cheap; OK on every rank) ─────────────────────────────
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

    # ── forward model (collective — every rank participates) ─────────────────
    simout = Forward_model_ensemble(
        ensembleoutf1.shape[1]
        if hasattr(ensembleoutf1, "shape")
        else ensembleout["PERM"].shape[1],
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
        active_cells_ensemble,
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
        well_names,
        cfg,
        N_pr,
        lenwels,
        active_mask_3d,
        awater,
        agas,
        aoil,
        aqq,
        nx,
        ny,
        nz,
        minQ,
        maxQ,
        minQw,
        maxQw,
        minQg,
        maxQg,
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

    # ── well-response percentile plots (rank 0) ──────────────────────────────
    if dist.rank == 0:
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/PERCENTILE"))
        # plot_rsm_percentile(yzout[:, :, :N_pr],
        # True_mat[:, :N_pr],         Time_unie1, N_pr, well_names, "WOPR")
        # plot_rsm_percentile(yzout[:, :, N_pr:2*N_pr],
        # True_mat[:, N_pr:2*N_pr],   Time_unie1, N_pr, well_names, "WWPR")
        plot_rsm_percentile(
            yzout[:, :, 2 * N_pr : 3 * N_pr],
            True_mat[:, 2 * N_pr : 3 * N_pr],
            Time_unie1,
            N_pr,
            well_names,
            "WGPR",
        )
        os.chdir(oldfolder)
    # _barrier()

    # ── package posterior dict ───────────────────────────────────────────────
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

    # ── save artefact + 3D perm field plot (rank 0) ──────────────────────────
    if dist.rank == 0:
        joblib.dump(
            X_data11,
            to_absolute_path(
                "../RESULTS/HM_RESULTS/PERCENTILE/Posterior_Ensembles_percentile.joblib"
            ),
            compress=3,
        )

        f_3 = plt.figure(figsize=(20, 20), dpi=200)
        look = ((np.reshape(True_K, (nx, ny, nz), "F")) * active_cells_ensemble)[
            :, :, ::-1
        ]
        ax1 = f_3.add_subplot(3, 3, 1, projection="3d")
        Plot_PhysicsNeMo(
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
            gas_injectors,
        )

        look = ((np.reshape(base_k, (nx, ny, nz), "F")) * active_cells_ensemble)[
            :, :, ::-1
        ]
        ax1 = f_3.add_subplot(3, 3, 2, projection="3d")
        Plot_PhysicsNeMo(
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
            gas_injectors,
        )

        look = (
            (np.reshape(X_data1["P10_Perm"], (nx, ny, nz), "F")) * active_cells_ensemble
        )[:, :, ::-1]
        ax1 = f_3.add_subplot(3, 3, 3, projection="3d")
        Plot_PhysicsNeMo(
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
            gas_injectors,
        )

        look = (
            (np.reshape(X_data1["P50_Perm"], (nx, ny, nz), "F")) * active_cells_ensemble
        )[:, :, ::-1]
        ax1 = f_3.add_subplot(3, 3, 4, projection="3d")
        Plot_PhysicsNeMo(
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
            gas_injectors,
        )

        look = (
            (np.reshape(X_data1["P90_Perm"], (nx, ny, nz), "F")) * active_cells_ensemble
        )[:, :, ::-1]
        ax1 = f_3.add_subplot(3, 3, 5, projection="3d")
        Plot_PhysicsNeMo(
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
            gas_injectors,
        )

        look = (
            (np.reshape(X_data1["yes_best"]["PERM"], (nx, ny, nz), "F"))
            * active_cells_ensemble
        )[:, :, ::-1]
        ax1 = f_3.add_subplot(3, 3, 6, projection="3d")
        Plot_PhysicsNeMo(
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
            gas_injectors,
        )

        look = (
            (np.reshape(X_data1["yes_mean"]["PERM"], (nx, ny, nz), "F"))
            * active_cells_ensemble
        )[:, :, ::-1]
        ax1 = f_3.add_subplot(3, 3, 7, projection="3d")
        Plot_PhysicsNeMo(
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
            gas_injectors,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        tita = "Reservoir Models permeability Fields"
        plt.suptitle(tita, fontsize=16)
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/PERCENTILE"))
        plt.savefig("Reservoir_models.png")
        plt.clf()
        plt.close()
        os.chdir(oldfolder)
    # _barrier()

    return X_data11
