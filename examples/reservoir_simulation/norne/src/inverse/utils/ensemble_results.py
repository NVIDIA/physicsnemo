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
                    ENSEMBLE RESULTS UTILITIES MODULE
=====================================================================

This module provides ensemble results processing utilities for inverse
problems in reservoir simulation. It includes result analysis,
visualization, and data processing utilities.

This version is rank-aware for multi-GPU execution under torchrun:
- Forward model calls participate from every rank (collective op).
- Plotting, file writes, and rmtree/makedirs are guarded on rank 0
  with barriers afterwards so other ranks don't race ahead.
- The 5 forward passes still run on all ranks (correct).

@Author : Clement Etienam
"""

import os
import sys
import re
import time
import pickle
import gzip
import shutil
import logging

import numpy as np
import numpy.matlib
import torch.distributed as torchdist
from hydra.utils import to_absolute_path
from PIL import Image
from joblib import Parallel, delayed

from utils.ccr_utils import (
    Forward_model_ensemble,
)
from utils.ensemble_utils import (
    ProgressBar,
    ShowBar,
)
from inverse.inversion_operation_misc import (
    process_step,
)
from inverse.inversion_operation_ensemble import (
    clip_ensemble_params,
    compute_data_mismatch,
    ensemble_pytorch,
)
from inverse.inversion_operation_gather import (
    plot_rsm,
    plot_rsm_percentile_model,
    write_rsm,
    Plot_petrophysical,
    plot_rsm_single,
    Plot_mean,
    Plot_Histogram_now,
)
from inverse.utils.inverse_config import (
    GridConfig,
    NormBounds,
    PermBounds,
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


def process_final_results(
    cfg,
    dist,
    device,
    oldfolder: str,
    input_variables: list,
    output_variables: list,
    ensemble_bestK,
    ensemble_meanK,
    ensemble_bestP,
    ensemble_meanP,
    ensemble_bestf,
    ensemble_meanf,
    ensemble,
    ensemblep,
    ensemblef,
    use_k,
    use_p,
    use_f,
    chm,
    dt,
    cc_ini,
    mean_cost,
    best_cost,
    ini_ensemble,
    ensemble_dict,
    yes_best,
    ensemble_best,
    yes_mean,
    ensemble_mean,
    all_ensemble,
    True_mat,
    True_data,
    True_K,
    grid: "GridConfig",
    norm: "NormBounds",
    perm: "PermBounds",
    ens: "EnsembleSetup",
    well: "WellConfig",
    surrogate: "SurrogateConfig",
    time_arr: "TimeArrays",
    flow: "FlowArrays",
):
    """Aggregate, plot, and persist ensemble results and diagnostics.

    Consumes best/mean ensembles and configuration to generate RSM plots,
    progress bars, and final artefacts for downstream analysis.

    Parameters
    ----------
    input_variables : list[str]
        Active input property names (e.g. ``["PERM", "PORO", "FAULT"]``).
    output_variables : list[str]
        Active output property names (e.g. ``["PRESSURE", "SWAT", "SGAS"]``).
    ensemble_bestK, ensemble_meanK : list[np.ndarray]
        Per-iteration best/mean permeability snapshots collected during HM.
    ensemble_bestP, ensemble_meanP : list[np.ndarray]
        Per-iteration best/mean porosity snapshots.
    ensemble_bestf, ensemble_meanf : list[np.ndarray]
        Per-iteration best/mean fault multiplier snapshots.
    ensemble : np.ndarray
        Current permeability ensemble, shape ``(n_cells, N_ens)``.
    ensemblep : np.ndarray
        Current porosity ensemble, shape ``(n_cells, N_ens)``.
    ensemblef : np.ndarray
        Current fault ensemble, shape ``(n_cells, N_ens)``.
    use_k, use_p, use_f : np.ndarray
        Best single-member arrays for PERM, PORO, and FAULT.
    chm : int
        Column index of the best ensemble member.
    Ne : int
        Total number of ensemble members used for HM.
    nx, ny, nz : int
        Grid dimensions (cells in x, y, z).
    N_ens : int
        Ensemble size (may differ from Ne in final processing).
    High_K1, Low_K1 : float
        Physical bounds (log-scale) for permeability clipping.
    High_P, Low_P : float
        Physical bounds for porosity clipping.
    effec : np.ndarray
        Active-cell indicator array, shape ``(n_cells,)``.
    effective : np.ndarray
        Active-cell mask used by the forward surrogate.
    oldfolder : str
        Working directory to restore after result writing.
    target_min, target_max : float
        Global normalisation range applied to inputs.
    minK, maxK : float
        Permeability normalisation bounds.
    minT, maxT : float
        Time normalisation bounds.
    minP, maxP : float
        Pressure normalisation bounds.
    minQ, maxQ : float
        Total rate normalisation bounds.
    minQw, maxQw : float
        Water rate normalisation bounds.
    minQg, maxQg : float
        Gas rate normalisation bounds.
    steppi : int
        Number of simulation time steps.
    device : torch.device
        Compute device for surrogate inference.
    steppi_indices : np.ndarray
        Zero-based indices of the selected output time steps.
    cfg : DictConfig
        Hydra configuration object.
    models : dict
        Surrogate model dict keyed by output type.
    min_inn_fcn, max_inn_fcn : np.ndarray
        Input normalisation statistics (min/max per channel).
    min_out_fcn, max_out_fcn : np.ndarray
        Output normalisation statistics for primary surrogates.
    Time : np.ndarray
        Simulation time vector, shape ``(steppi,)``.
    active_cells_ensemble : np.ndarray
        Active-cell mask broadcast to ensemble size ``(n_cells, N_ens)``.
    Trainmoe : str
        Label identifying the Peacemann surrogate type (e.g. ``"MoE"``).
    num_cores : int
        Number of CPU cores for parallel ensemble forward passes.
    pred_type : int
        Prediction mode flag passed to the Peacemann CCR model.
    degg : int
        Polynomial degree used by the Peacemann CCR model.
    experts : int
        Number of expert clusters in the Peacemann CCR model.
    min_out_fcn2, max_out_fcn2 : np.ndarray
        Output normalisation statistics for the Peacemann surrogate.
    min_inn_fcn2, max_inn_fcn2 : np.ndarray
        Input normalisation statistics for the Peacemann surrogate.
    producers : list
        Producer well metadata list (name, i, j, k, …).
    compdat_data : list
        COMPDAT completion entries for all wells.
    quant_big : dict
        Per-well scaling metadata ``{"K_0": {"value", "scale", "boolean"}, …}``.
    N_pr : int
        Number of producer wells.
    lenwels : int
        Total number of well measurement channels.
    active_mask_3d : np.ndarray
        3-D active-cell mask, shape ``(nx, ny, nz)``.
    rows_to_remove : np.ndarray
        Row indices where observed data are below the dead-well threshold.
    True_mat : np.ndarray
        Observed production matrix, shape ``(steppi, lenwels * N_pr)``.
    True_data : np.ndarray
        Scaled observed data vector, shape ``(n_obs, 1)``.
    Time_unie1 : np.ndarray
        Time axis for RSM plots (years or days).
    well_names : list[str]
        Producer well name strings for plot labels.
    dt : float
        Simulation time-step size.
    dist : DistributedManager
        PhysicsNeMo distributed context (rank / world_size).
    cc_ini : float
        Initial cost (RMSE) before assimilation.
    mean_cost, best_cost : list[float]
        Per-iteration ensemble-mean and best-member RMSE histories.
    ini_ensemble : np.ndarray
        Original prior ensemble before any updates.
    N_injw, N_injg : int
        Number of water and gas injector wells.
    injectors : list
        Water injector well metadata list.
    gas_injectors : list
        Gas injector well metadata list.
    True_K : np.ndarray
        Reference (true) permeability field, shape ``(n_cells, 1)``.
    yes_best, yes_mean : dict
        Single best/mean member arrays keyed by property name.
    ensemble_best, ensemble_mean : dict
        Full stacked best/mean ensemble arrays keyed by property name.
    all_ensemble : dict
        Complete final ensemble arrays keyed by property name.
    ensemble_dict : dict
        Ensemble dict passed to ``clip_ensemble_params`` for physical bounding.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    steppi = grid.steppi
    steppi_indices = grid.steppi_indices
    Ne = ens.Ne
    N_ens = ens.N_ens
    effec = ens.effec
    effective = ens.effective
    active_cells_ensemble = ens.active_cells_ensemble
    active_mask_3d = ens.active_mask_3d
    rows_to_remove = ens.rows_to_remove
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
    High_K1, Low_K1 = perm.High_K1, perm.Low_K1
    High_P, Low_P = perm.High_P, perm.Low_P
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

    # ── stack per-iteration ensembles into matrices ──────────────────────────
    if "PERM" in input_variables:
        ensemble_bestK = np.hstack(ensemble_bestK)
        ensemble_meanK = np.hstack(ensemble_meanK)
        ensemble_best["PERM"] = np.hstack(ensemble_bestK)
        yes_best["PERM"] = ensemble_bestK[:, chm].reshape(-1, 1)
        ensemble_mean["PERM"] = np.hstack(ensemble_meanK)
        yes_mean["PERM"] = ensemble_meanK[:, chm].reshape(-1, 1)
        all_ensemble["PERM"] = use_k
        ensemble_dict["PERM"] = ensemble
    if "PORO" in input_variables:
        ensemble_bestP = np.hstack(ensemble_bestP)
        ensemble_meanP = np.hstack(ensemble_meanP)
        ensemble_best["PORO"] = np.hstack(ensemble_bestP)
        yes_best["PORO"] = ensemble_bestP[:, chm].reshape(-1, 1)
        ensemble_mean["PORO"] = np.hstack(ensemble_meanP)
        yes_mean["PORO"] = ensemble_meanP[:, chm].reshape(-1, 1)
        all_ensemble["PORO"] = use_p
        ensemble_dict["PORO"] = ensemblep
    if "FAULT" in input_variables:
        ensemble_bestf = np.hstack(ensemble_bestf)
        ensemble_meanf = np.hstack(ensemble_meanf)
        ensemble_best["FAULT"] = np.hstack(ensemble_bestf)
        yes_best["FAULT"] = ensemble_bestf[:, chm].reshape(-1, 1)
        ensemble_mean["FAULT"] = np.hstack(ensemble_meanf)
        yes_mean["FAULT"] = ensemble_meanf[:, chm].reshape(-1, 1)
        use_f = np.clip(use_f, 0, 1)
        all_ensemble["FAULT"] = use_f
        ensemble_dict["FAULT"] = ensemblef

    ensemble = ensemble_dict
    if "PERM" in input_variables or "PORO" in input_variables:
        ensemble_dict = clip_ensemble_params(
            ensemble_dict, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P, effec
        )
    ensemble = ensemble_dict
    if "PERM" in input_variables or "PORO" in input_variables:
        all_ensemble = clip_ensemble_params(
            all_ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P, effec
        )
    if "FAULT" in input_variables:
        all_ensemble["FAULT"] = np.clip(all_ensemble["FAULT"], 0, 1)

    meann = {}
    if "PERM" in input_variables:
        meann["PERM"] = np.reshape(np.mean(ensemble["PERM"], axis=1), (-1, 1), "F")
    if "PORO" in input_variables:
        meann["PORO"] = np.reshape(np.mean(ensemble["PORO"], axis=1), (-1, 1), "F")
    if "FAULT" in input_variables:
        meann["FAULT"] = np.reshape(np.mean(ensemble["FAULT"], axis=1), (-1, 1), "F")
    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")

    controljj = {}
    if "PERM" in input_variables:
        controljj["PERM"] = np.reshape(meann["PERM"], (-1, 1), "F")
    if "PORO" in input_variables:
        controljj["PORO"] = np.reshape(meann["PORO"], (-1, 1), "F")
    if "FAULT" in input_variables:
        controljj["FAULT"] = np.reshape(meann["FAULT"], (-1, 1), "F")

    # ── posterior ensemble forward pass (collective) ─────────────────────────
    ensemblepy = ensemble_pytorch(
        ensemble,
        nx,
        ny,
        nz,
        ensemble["PERM"].shape[1],
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
    simout = Forward_model_ensemble(
        ensemble["PERM"].shape[1],
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
    simDatafinal = simout["sim"][rows_to_remove]
    predMatrix = simout["ouut_p"]
    if "PRESSURE" in output_variables:
        pressure_ensemble = simout["PRESSURE"]
    if "SWAT" in output_variables:
        water_ensemble = simout["SWAT"]
    if "SOIL" in output_variables:
        oil_ensemble = simout["SOIL"]
    if "SGAS" in output_variables:
        gas_ensemble = simout["SGAS"]

    # ── all_ensemble forward pass (collective) ───────────────────────────────
    ensemblepya = ensemble_pytorch(
        all_ensemble,
        nx,
        ny,
        nz,
        all_ensemble["PERM"].shape[1],
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
    simout = Forward_model_ensemble(
        all_ensemble["PERM"].shape[1],
        ensemblepya,
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
    simDatafinala = simout["sim"][rows_to_remove]
    predMatrixa = simout["ouut_p"]
    if "PRESSURE" in output_variables:
        pressure_ensemblea = simout["PRESSURE"]
    if "SWAT" in output_variables:
        water_ensemblea = simout["SWAT"]
    if "SOIL" in output_variables:
        oil_ensemblea = simout["SOIL"]
    if "SGAS" in output_variables:
        gas_ensemblea = simout["SGAS"]

    # ── posterior plots (rank 0) ─────────────────────────────────────────────
    if dist.rank == 0:
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
        # plot_rsm(predMatrixa[:, :, :N_pr],          True_mat[:, :N_pr],
        # "POSTERIOR_ENSEMBLE_WOPR", Ne, Time_unie1, N_pr, well_names, "WOPR")
        # plot_rsm(predMatrixa[:, :, N_pr:2 * N_pr],  True_mat[:, N_pr:2 * N_pr],
        # "POSTERIOR_ENSEMBLE_WWPR", Ne, Time_unie1, N_pr, well_names, "WWPR")
        plot_rsm(
            predMatrixa[:, :, 2 * N_pr : 3 * N_pr],
            True_mat[:, 2 * N_pr : 3 * N_pr],
            "POSTERIOR_ENSEMBLE_WGPR",
            Ne,
            Time_unie1,
            N_pr,
            well_names,
            "WGPR",
        )
        os.chdir(oldfolder)
    _barrier()

    # ── data mismatch summary ────────────────────────────────────────────────
    _aa, _bb, cc = compute_data_mismatch(simDatafinal, True_data)
    muv = np.argmin(cc)

    controlbest = {}
    if "PERM" in input_variables:
        controlbest["PERM"] = np.reshape(ensemble["PERM"][:, muv], (-1, 1), "F")
    if "PORO" in input_variables:
        controlbest["PORO"] = np.reshape(ensemble["PORO"][:, muv], (-1, 1), "F")
    if "FAULT" in input_variables:
        controlbest["FAULT"] = np.reshape(ensemble["FAULT"][:, muv], (-1, 1), "F")
    controlbest2 = {}
    if "PERM" in input_variables:
        controlbest2["PERM"] = controljj["PERM"]
    if "PORO" in input_variables:
        controlbest2["PORO"] = controljj["PORO"]
    if "FAULT" in input_variables:
        controlbest2["FAULT"] = controljj["FAULT"]
    muvbad = np.argmax(cc)
    controlbad = {}
    if "PERM" in input_variables:
        controlbad["PERM"] = np.reshape(ensemble["PERM"][:, muvbad], (-1, 1), "F")
    if "PORO" in input_variables:
        controlbad["PORO"] = np.reshape(ensemble["PORO"][:, muvbad], (-1, 1), "F")
    if "FAULT" in input_variables:
        controlbad["FAULT"] = np.reshape(ensemble["FAULT"][:, muvbad], (-1, 1), "F")

    if dist.rank == 0:
        Plot_Histogram_now(Ne, cc_ini, cc, mean_cost, best_cost, oldfolder)

    # ── ADAPT_REKI section ───────────────────────────────────────────────────
    _recreate_dir("../RESULTS/HM_RESULTS/ADAPT_REKI", dist)
    if dist.rank == 0:
        logger.info(
            "**********************************************************************"
        )
        logger.info(
            "                   ANALYSIS FOR MLE RESERVOIR_MODEL                    "
        )
        logger.info(
            "***********************************************************************"
        )

    # MLE forward pass (collective)
    ensemblepy = ensemble_pytorch(
        controlbest,
        nx,
        ny,
        nz,
        controlbest["PERM"].shape[1],
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
    simout = Forward_model_ensemble(
        controlbest["PERM"].shape[1],
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
    yycheck = simout["ouut_p"]  # [rows_to_remove]
    if "PRESSURE" in output_variables:
        pree = simout["PRESSURE"]
    if "SWAT" in output_variables:
        wats = simout["SWAT"]
    if "SOIL" in output_variables:
        oilss = simout["SOIL"]
    if "SGAS" in output_variables:
        gasss = simout["SGAS"]

    if dist.rank == 0:
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/ADAPT_REKI"))
        # plot_rsm_single(np.squeeze(yycheck[:, :, :N_pr],         axis=0), Time_unie1, N_pr, well_names, "WOPR")
        # plot_rsm_single(np.squeeze(yycheck[:, :, N_pr:2*N_pr],   axis=0), Time_unie1, N_pr, well_names, "WWPR")
        plot_rsm_single(
            np.squeeze(yycheck[:, :, 2 * N_pr : 3 * N_pr], axis=0),
            Time_unie1,
            N_pr,
            well_names,
            "WGPR",
        )

        Plot_petrophysical(
            controlbest["PERM"],
            controlbest["PORO"],
            nx,
            ny,
            nz,
            Low_K1,
            High_K1,
            active_cells_ensemble,
            N_injw,
            N_pr,
            N_injg,
            injectors,
            producers,
            gas_injectors,
            Low_P,
            High_P,
        )
        os.chdir(oldfolder)

    # Save MLE artefacts on rank 0
    X_data1 = {}
    if "PERM" in input_variables:
        X_data1["PERM"] = controlbest["PERM"]
    if "PORO" in input_variables:
        X_data1["PORO"] = controlbest["PORO"]
    if "FAULT" in input_variables:
        X_data1["FAULT"] = controlbest["FAULT"]
    if "PRESSURE" in output_variables:
        X_data1["PRESSURE"] = simout["PRESSURE"]
    if "SWAT" in output_variables:
        X_data1["SWAT"] = simout["SWAT"]
    if "SOIL" in output_variables:
        X_data1["SOIL"] = simout["SOIL"]
    if "SGAS" in output_variables:
        X_data1["SGAS"] = simout["SGAS"]
    X_data1["Simulated_data_plots"] = yycheck

    if dist.rank == 0:
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/ADAPT_REKI"))
        with gzip.open("RESERVOIR_MODEL.pkl.gz", "wb") as f1:
            pickle.dump(X_data1, f1)
        os.chdir(oldfolder)
    # _barrier()

    Time_vector = np.zeros(steppi)
    for kk in range(steppi):
        Time_vector[kk] = dt[kk]

    folderrin = os.path.join(oldfolder, "..", "RESULTS", "HM_RESULTS", "ADAPT_REKI")

    if dist.rank == 0:
        Parallel(n_jobs=num_cores, backend="loky", verbose=10)(
            delayed(process_step)(
                kk,
                steppi,
                dt,
                pree,
                active_cells_ensemble,
                wats,
                oilss,
                gasss,
                nx,
                ny,
                nz,
                N_injw,
                N_pr,
                N_injg,
                injectors,
                producers,
                gas_injectors,
                to_absolute_path(folderrin),
                oldfolder,
            )
            for kk in range(steppi)
        )
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, steppi - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)

    import glob as _glob_mod

    if dist.rank == 0:
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/ADAPT_REKI"))
        frames = []
        imgs = sorted(
            _glob_mod.glob("*Dynamic*"),
            key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)],
        )
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)
        if frames:
            frames[0].save(
                "Evolution.gif",
                format="GIF",
                append_images=frames[1:],
                save_all=True,
                duration=500,
                loop=0,
            )
        for f3 in _glob_mod.glob("*Dynamic*"):
            os.remove(f3)
        write_rsm(
            yycheck[0, :, : lenwels * N_pr],
            Time_vector,
            "PhysicsNeMo",
            well_names,
            N_pr,
        )
        # plot_rsm_percentile_model(np.squeeze(yycheck[:, :, :N_pr],         axis=0),
        # True_mat[:, :N_pr],         Time_unie1, N_pr, well_names, "WOPR")
        # plot_rsm_percentile_model(np.squeeze(yycheck[:, :, N_pr:2*N_pr],   axis=0),
        # True_mat[:, N_pr:2*N_pr],   Time_unie1, N_pr, well_names, "WWPR")
        plot_rsm_percentile_model(
            np.squeeze(yycheck[:, :, 2 * N_pr : 3 * N_pr], axis=0),
            True_mat[:, 2 * N_pr : 3 * N_pr],
            Time_unie1,
            N_pr,
            well_names,
            "WGPR",
        )
        os.chdir(oldfolder)
    # _barrier()

    yycheck = simout["sim"][rows_to_remove]
    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    if dist.rank == 0:
        logger.info("RMSE OF MLE_RESERVOIR_MODEL   =  %s", str(cc))

    if dist.rank == 0:
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
        Plot_mean(
            controlbest["PERM"],
            yes_mean["PERM"],
            meanini,
            nx,
            ny,
            nz,
            Low_K1,
            High_K1,
            True_K,
            active_cells_ensemble,
            injectors,
            producers,
            gas_injectors,
            N_injw,
            N_pr,
            N_injg,
        )
        os.chdir(oldfolder)
    # _barrier()

    # ── BEST_RESERVOIR_MODEL section ─────────────────────────────────────────
    if dist.rank == 0:
        logger.info(
            "**********************************************************************"
        )
        logger.info(
            "                ANALYSIS FOR BEST_RESERVOIR_MODEL                      "
        )
        logger.info(
            "***********************************************************************"
        )
    _recreate_dir("../RESULTS/HM_RESULTS/BEST_RESERVOIR_MODEL", dist)

    ensemblepy = ensemble_pytorch(
        yes_best,
        nx,
        ny,
        nz,
        controlbest["PERM"].shape[1],
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
    simout = Forward_model_ensemble(
        controlbest["PERM"].shape[1],
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
    yycheck = simout["ouut_p"]
    if "PRESSURE" in output_variables:
        preebest = simout["PRESSURE"]
    if "SWAT" in output_variables:
        watsbest = simout["SWAT"]
    if "SOIL" in output_variables:
        oilssbest = simout["SOIL"]
    if "SGAS" in output_variables:
        gasbest = simout["SGAS"]

    if dist.rank == 0:
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/BEST_RESERVOIR_MODEL"))
        # plot_rsm_single(np.squeeze(yycheck[:, :, :N_pr],         axis=0), Time_unie1, N_pr, well_names, "WOPR")
        # plot_rsm_single(np.squeeze(yycheck[:, :, N_pr:2*N_pr],   axis=0), Time_unie1, N_pr, well_names, "WWPR")
        plot_rsm_single(
            np.squeeze(yycheck[:, :, 2 * N_pr : 3 * N_pr], axis=0),
            Time_unie1,
            N_pr,
            well_names,
            "WGPR",
        )
        Plot_petrophysical(
            yes_best["PERM"],
            yes_best["PORO"],
            nx,
            ny,
            nz,
            Low_K1,
            High_K1,
            active_cells_ensemble,
            N_injw,
            N_pr,
            N_injg,
            injectors,
            producers,
            gas_injectors,
            Low_P,
            High_P,
        )
        os.chdir(oldfolder)

    X_data1 = {}
    if "PERM" in input_variables:
        X_data1["PERM"] = yes_best["PERM"]
    if "PORO" in input_variables:
        X_data1["PORO"] = yes_best["PORO"]
    if "FAULT" in input_variables:
        X_data1["FAULT"] = yes_best["FAULT"]
    if "PRESSURE" in output_variables:
        X_data1["PRESSURE"] = simout["PRESSURE"]
    if "SWAT" in output_variables:
        X_data1["SWAT"] = simout["SWAT"]
    if "SOIL" in output_variables:
        X_data1["SOIL"] = simout["SOIL"]
    if "SGAS" in output_variables:
        X_data1["SGAS"] = simout["SGAS"]
    X_data1["Simulated_data_plots"] = yycheck

    if dist.rank == 0:
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/BEST_RESERVOIR_MODEL"))
        with gzip.open("BEST_RESERVOIR_MODEL.pkl.gz", "wb") as f1:
            pickle.dump(X_data1, f1)
        os.chdir(oldfolder)
    # _barrier()

    folderrin = os.path.join(
        oldfolder, "..", "RESULTS", "HM_RESULTS", "BEST_RESERVOIR_MODEL"
    )

    if dist.rank == 0:
        Parallel(n_jobs=num_cores, backend="loky", verbose=10)(
            delayed(process_step)(
                kk,
                steppi,
                dt,
                preebest,
                active_cells_ensemble,
                watsbest,
                oilssbest,
                gasbest,
                nx,
                ny,
                nz,
                N_injw,
                N_pr,
                N_injg,
                injectors,
                producers,
                gas_injectors,
                to_absolute_path(folderrin),
                oldfolder,
            )
            for kk in range(steppi)
        )
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, steppi - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)

        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/BEST_RESERVOIR_MODEL"))
        frames = []
        imgs = sorted(
            _glob_mod.glob("*Dynamic*"),
            key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)],
        )
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)
        if frames:
            frames[0].save(
                "Evolution.gif",
                format="GIF",
                append_images=frames[1:],
                save_all=True,
                duration=500,
                loop=0,
            )
        for f3 in _glob_mod.glob("*Dynamic*"):
            os.remove(f3)
        write_rsm(
            yycheck[0, :, : lenwels * N_pr],
            Time_vector,
            "PhysicsNeMo",
            well_names,
            N_pr,
        )
        # plot_rsm_percentile_model(np.squeeze(yycheck[:, :, :N_pr],         axis=0),
        # True_mat[:, :N_pr],         Time_unie1, N_pr, well_names, "WOPR")
        # plot_rsm_percentile_model(np.squeeze(yycheck[:, :, N_pr:2*N_pr],   axis=0),
        # True_mat[:, N_pr:2*N_pr],   Time_unie1, N_pr, well_names, "WWPR")
        plot_rsm_percentile_model(
            np.squeeze(yycheck[:, :, 2 * N_pr : 3 * N_pr], axis=0),
            True_mat[:, 2 * N_pr : 3 * N_pr],
            Time_unie1,
            N_pr,
            well_names,
            "WGPR",
        )
        os.chdir(oldfolder)
    # _barrier()

    # yycheck = yycheck[0, :, : lenwels * N_pr]
    # usesim = np.reshape(yycheck, (-1, 1), "F")
    yycheck = simout["sim"][rows_to_remove]
    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    if dist.rank == 0:
        logger.info("RMSE OF BEST RESERVOIR MODEL  =  %s", str(cc))

    # ── MEAN_RESERVOIR_MODEL section ─────────────────────────────────────────
    if dist.rank == 0:
        logger.info(
            "**********************************************************************"
        )
        logger.info(
            "              ANALYSIS FOR MEAN_RESERVOIR_MODEL                       "
        )
        logger.info(
            "**********************************************************************"
        )
    _recreate_dir("../RESULTS/HM_RESULTS/MEAN_RESERVOIR_MODEL", dist)

    ensemblepy = ensemble_pytorch(
        yes_mean,
        nx,
        ny,
        nz,
        yes_mean["PERM"].shape[1],
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
    simout = Forward_model_ensemble(
        controlbest2["PERM"].shape[1],
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
    yycheck = simout["ouut_p"]
    if "PRESSURE" in output_variables:
        preebest = simout["PRESSURE"]
    if "SWAT" in output_variables:
        watsbest = simout["SWAT"]
    if "SOIL" in output_variables:
        oilssbest = simout["SOIL"]
    if "SGAS" in output_variables:
        gasbest = simout["SGAS"]

    if dist.rank == 0:
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/MEAN_RESERVOIR_MODEL"))
        # plot_rsm_single(np.squeeze(yycheck[:, :, :N_pr],         axis=0), Time_unie1, N_pr, well_names, "WOPR")
        # plot_rsm_single(np.squeeze(yycheck[:, :, N_pr:2*N_pr],   axis=0), Time_unie1, N_pr, well_names, "WWPR")
        plot_rsm_single(
            np.squeeze(yycheck[:, :, 2 * N_pr : 3 * N_pr], axis=0),
            Time_unie1,
            N_pr,
            well_names,
            "WGPR",
        )
        Plot_petrophysical(
            yes_mean["PERM"],
            yes_mean["PORO"],
            nx,
            ny,
            nz,
            Low_K1,
            High_K1,
            active_cells_ensemble,
            N_injw,
            N_pr,
            N_injg,
            injectors,
            producers,
            gas_injectors,
            Low_P,
            High_P,
        )
        os.chdir(oldfolder)

    X_data1 = {}
    if "PERM" in input_variables:
        X_data1["PERM"] = yes_mean["PERM"]
    if "PORO" in input_variables:
        X_data1["PORO"] = yes_mean["PORO"]
    if "FAULT" in input_variables:
        X_data1["FAULT"] = yes_mean["FAULT"]
    if "PRESSURE" in output_variables:
        X_data1["PRESSURE"] = simout["PRESSURE"]
    if "SWAT" in output_variables:
        X_data1["SWAT"] = simout["SWAT"]
    if "SOIL" in output_variables:
        X_data1["SOIL"] = simout["SOIL"]
    if "SGAS" in output_variables:
        X_data1["SGAS"] = simout["SGAS"]
    X_data1["Simulated_data_plots"] = yycheck

    if dist.rank == 0:
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/MEAN_RESERVOIR_MODEL"))
        with gzip.open("MEAN_RESERVOIR_MODEL.pkl.gz", "wb") as f1:
            pickle.dump(X_data1, f1)
        os.chdir(oldfolder)
    # _barrier()

    folderrin = os.path.join(
        oldfolder, "..", "RESULTS", "HM_RESULTS", "MEAN_RESERVOIR_MODEL"
    )

    if dist.rank == 0:
        Parallel(n_jobs=num_cores, backend="loky", verbose=10)(
            delayed(process_step)(
                kk,
                steppi,
                dt,
                preebest,
                active_cells_ensemble,
                watsbest,
                oilssbest,
                gasbest,
                nx,
                ny,
                nz,
                N_injw,
                N_pr,
                N_injg,
                injectors,
                producers,
                gas_injectors,
                to_absolute_path(folderrin),
                oldfolder,
            )
            for kk in range(steppi)
        )
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, steppi - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)

        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/MEAN_RESERVOIR_MODEL"))
        frames = []
        imgs = sorted(
            _glob_mod.glob("*Dynamic*"),
            key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)],
        )
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)
        if frames:
            frames[0].save(
                "Evolution.gif",
                format="GIF",
                append_images=frames[1:],
                save_all=True,
                duration=500,
                loop=0,
            )
        for f3 in _glob_mod.glob("*Dynamic*"):
            os.remove(f3)
        write_rsm(
            yycheck[0, :, : lenwels * N_pr],
            Time_vector,
            "PhysicsNeMo",
            well_names,
            N_pr,
        )
        # plot_rsm_percentile_model(np.squeeze(yycheck[:, :, :N_pr],         axis=0),
        # True_mat[:, :N_pr],         Time_unie1, N_pr, well_names, "WOPR")
        # plot_rsm_percentile_model(np.squeeze(yycheck[:, :, N_pr:2*N_pr],   axis=0),
        # True_mat[:, N_pr:2*N_pr],   Time_unie1, N_pr, well_names, "WWPR")
        plot_rsm_percentile_model(
            np.squeeze(yycheck[:, :, 2 * N_pr : 3 * N_pr], axis=0),
            True_mat[:, 2 * N_pr : 3 * N_pr],
            Time_unie1,
            N_pr,
            well_names,
            "WGPR",
        )
        os.chdir(oldfolder)
    # _barrier()

    # yycheck = yycheck[0, :, : lenwels * N_pr]
    # usesim = np.reshape(yycheck, (-1, 1), "F")
    yycheck = simout["sim"][rows_to_remove]
    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    if dist.rank == 0:
        logger.info("RMSE of MAP RESERVOIR MODEL  =  %s", str(cc))
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
        os.chdir(oldfolder)

    return {
        "PERM_Reali": ensemble["PERM"],
        "FAULT_Reali": ensemble["FAULT"],
        "PORO_Reali": ensemble["PORO"],
        "P10_Perm": controlbest["PERM"],
        "P50_Perm": controljj["PERM"],
        "P90_Perm": controlbad["PERM"],
        "P10_Poro": controlbest["PORO"],
        "P50_Poro": controljj["PORO"],
        "P90_Poro": controlbad["PORO"],
        "P10_Fault": controlbest["FAULT"],
        "P50_Fault": controljj["FAULT"],
        "P90_Fault": controlbad["FAULT"],
        "Simulated_data": simDatafinal,
        "Simulated_data_plots": predMatrix,
        "Pressures": pressure_ensemble,
        "Water_saturation": water_ensemble,
        "Oil_saturation": oil_ensemble,
        "Gas_saturation": gas_ensemble,
        "Simulated_data_best_ensemble": simDatafinala,
        "Simulated_data_plots_best_ensemble": predMatrixa,
        "Pressures_best_ensemble": pressure_ensemblea,
        "Water_saturation_best_ensemble": water_ensemblea,
        "Oil_saturation_best_ensemble": oil_ensemblea,
        "Gas_saturation_best_ensemble": gas_ensemblea,
        "ensemble_best": ensemble_best,
        "yes_best": yes_best,
        "ensemble_mean": ensemble_mean,
        "yes_mean": yes_mean,
        "all_ensemble": all_ensemble,
        "ensemble_dict": ensemble_dict,
    }
