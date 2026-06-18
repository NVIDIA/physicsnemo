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
                    SEQUENTIAL UTILITIES - CORE FUNCTIONS
=====================================================================

This module provides core utilities for sequential FVM surrogate model
comparisons. It includes functions for environment setup, data processing,
and analysis.

Key Features:
- Environment initialization and setup
- Data processing and analysis utilities
- Performance monitoring and logging
- Result comparison and analysis

Usage:
    from compare.sequential.utils.misc_utils import (
        compare_and_analyze_results,
        setup_logging,
        initialize_environment
    )

@Author : Clement Etienam
"""

import os
import time
import math
import glob
import shutil
import logging
import warnings
from datetime import timedelta
import re
from sklearn.metrics import r2_score as r2

import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from PIL import Image

from hydra.utils import to_absolute_path
from compare.sequential.misc_plotting_utils import (
    simulation_data_types,
    Get_Time,
)

from compare.sequential.misc_operations import (
    ProgressBar,
    ShowBar,
    plot_rsm_percentile,
)

from compare.sequential.misc_model import (
    process_step,
)

from compare.sequential.misc_forward import (
    write_rsm,
)

from utils.ccr_utils import (
    Forward_model_ensemble,
)

from compare.sequential.utils.compare_config import (
    CompareFields,
    CompareFlow,
    CompareGrid,
    CompareNorms,
    CompareRuntime,
    CompareSurrogate,
    CompareTiming,
    CompareWellResults,
    CompareWells,
)


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logging()
warnings.filterwarnings("ignore")


# Sorting function for row-major order
def sort_key(path):
    """Extract (row, col) sort key from a PNG filename for row-major ordering.

    Parameters
    ----------
    path : str
        File path whose basename contains a pattern like ``_row_col.png``.

    Returns
    -------
    tuple of int
        ``(row_index, col_index)`` parsed from the filename, or
        ``(inf, inf)`` if the pattern is not found.
    """
    # Extract row and column indices from the filename using regex
    match = re.search(r"_(\d+)_(\d+)\.png", path)  # Match pattern like "_row_col.png"
    if match:
        row_index, col_index = int(match.group(1)), int(match.group(2))
        # Sort by row first, then by column (row-major order)
        return (row_index, col_index)
    return float("inf"), float("inf")  # Handle unexpected filenames


(
    type_dict,
    ecl_extensions,
    dynamic_props,
    ecl_vectors,
    static_props,
    SUPPORTED_DATA_TYPES,
) = simulation_data_types()


# =============================================================================
# Helper — execution-time bar chart (flow vs physicsnemo)
# =============================================================================
def _plot_time_comparison(physicsnemo_time, flow_time, save_dir, oldfolder):
    """Bar-chart comparing flow simulator vs PhysicsNeMo surrogate execution time."""
    if physicsnemo_time < flow_time:
        slower_time, faster_time = physicsnemo_time, flow_time
        slower, faster = "Nvidia physicsnemo Surrogate", "flow Reservoir simulator"
        speedup = math.ceil(flow_time / physicsnemo_time)
    else:
        slower_time, faster_time = flow_time, physicsnemo_time
        slower, faster = "flow Reservoir simulator", "Nvidia physicsnemo Surrogate"
        speedup = math.ceil(physicsnemo_time / flow_time)

    # Bars in fixed order so colors stay meaningful
    tasks = ["Flow simulator", "PhysicsNeMo"]
    times = [flow_time, physicsnemo_time]
    colors = ["#2C3E50", "#27AE60"]  # dark slate vs NVIDIA green

    os.chdir(to_absolute_path(save_dir))

    # ── style ────────────────────────────────────────────────────────────────
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.4,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "figure.dpi": 150,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    bars = ax.bar(
        tasks,
        times,
        color=colors,
        width=0.55,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )

    # ── value labels above bars ──────────────────────────────────────────────
    y_max = max(times)
    for bar, v in zip(bars, times, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.02,
            f"{v:,.2f} s",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#2C3E50",
        )

    # ── speedup callout box ──────────────────────────────────────────────────
    ax.text(
        0.5,
        0.92,
        f"Speedup:  {speedup}×",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#1A5276",
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="#FDF2E9",
            edgecolor="#E59866",
            linewidth=1.8,
        ),
        zorder=5,
    )

    # ── axes & title ─────────────────────────────────────────────────────────
    ax.set_ylabel("Time (seconds)", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(
        "Execution Time — PhysicsNeMo Surrogate vs Flow Simulator",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylim(0, y_max * 1.30)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(
        axis="y", linestyle="--", linewidth=0.6, alpha=0.5, color="#AAAAAA", zorder=0
    )
    ax.spines[["top", "right"]].set_visible(False)

    # Bold tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    # ── faster/slower legend ─────────────────────────────────────────────────
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(
            facecolor=colors[0],
            edgecolor="white",
            label=f"Flow simulator   ({flow_time:,.2f} s)",
        ),
        Patch(
            facecolor=colors[1],
            edgecolor="white",
            label=f"PhysicsNeMo      ({physicsnemo_time:,.2f} s)",
        ),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        fontsize=10,
    )
    for text in leg.get_texts():
        text.set_fontweight("bold")

    plt.savefig(
        "Compare_time.png",
        bbox_inches="tight",
        dpi=200,
        facecolor="white",
        edgecolor="none",
    )
    plt.clf()
    plt.close()

    os.chdir(to_absolute_path(oldfolder))

    msg = (
        f"{slower} execution took: {slower_time} seconds\n"
        f"{faster} execution took: {faster_time} seconds\n"
        f"Speedup =  {speedup}X  "
    )
    print(msg)
    return speedup


def _select_compare_folder(cfg, active="fno"):
    """Return paths and prepare only the active destination directory.

    Parameters
    ----------
    active : {"fno", "fno_ccr"}
        Which destination should be wiped+recreated this call. The other
        is left alone so previous-stage outputs survive.
    """
    base = "../RESULTS/FORWARD_RESULTS/RESULTS/COMPARE_RESULTS"
    if cfg.custom.model_type == "FNO":
        sub = "PINO" if cfg.custom.fno_type == "PINO" else "FNO"
    else:
        sub = "PI-TRANSOLVER" if cfg.custom.fno_type == "PINO" else "TRANSOLVER"

    folderr = f"{base}/{sub}/PEACEMANN_FNO"
    folderr2 = f"{base}/{sub}/PEACEMANN_FNO_CCR"
    source = to_absolute_path(f"{base}/{sub}/PEACEMANN_CCR")
    dest = to_absolute_path(folderr)
    dest2 = to_absolute_path(folderr2)

    target = dest if active == "fno" else dest2
    if os.path.exists(target):
        shutil.rmtree(target)
    os.makedirs(target, exist_ok=True)

    return folderr, folderr2, source, dest, dest2


# =============================================================================
# Helper — four-curve plot per variable (truth / CCR / FNO / FNO+CCR)
# =============================================================================
def plot_rsm_three_curves(
    CCR_pred,
    FNO_pred,
    FNOCCR_pred,
    True_mat,
    timezz,
    well_names,
    N_wells,
    ope_f,
    cfg,
):
    """
    Plot CCR, FNO, FNO+CCR hybrid surrogates, and numerical truth for one variable.

    Parameters
    ----------
    CCR_pred    : ndarray (T, N_wells)  CCR surrogate predictions
    FNO_pred    : ndarray (T, N_wells)  FNO surrogate predictions
    FNOCCR_pred : ndarray (T, N_wells)  FNO+CCR hybrid surrogate predictions
    True_mat    : ndarray (T, N_wells)  numerical truth
    timezz      : ndarray (T,)          time in days
    well_names  : list[str]             well names length N_wells
    N_wells     : int                   number of wells
    ope_f       : str                   WOPR | WWPR | WGPR
    cfg         : Hydra config          used for cfg.custom.fno_type
    """
    # ── label logic ───────────────────────────────────────────────────────────
    is_pino = cfg.custom.fno_type == "FNO"
    lbl_ccr = "PINO — CCR" if is_pino else "FNO — CCR"
    lbl_fno = "PINO — FNO" if is_pino else "FNO — FNO"
    lbl_hybrid = "PINO — FNO+CCR" if is_pino else "FNO — FNO+CCR"

    # ── palette & style ───────────────────────────────────────────────────────
    CLR_TRUE = "#C0392B"  # deep red       — numerical
    CLR_CCR = "#2471A3"  # steel blue     — CCR surrogate
    CLR_FNO = "#D68910"  # amber orange   — FNO surrogate
    CLR_HYBRID = "#1E8449"  # forest green   — FNO+CCR hybrid

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.4,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "legend.framealpha": 0.88,
            "legend.edgecolor": "#CCCCCC",
            "figure.dpi": 150,
        }
    )

    # ── config per variable ───────────────────────────────────────────────────
    vcfg = {
        "WOPR": {
            "ylabel": r"$Q_{oil}\ (bbl/day)$",
            "suptitle": r"Oil Production Rate — $Q_{oil}$",
            "fname": "WOPR.png",
        },
        "WWPR": {
            "ylabel": r"$Q_{water}\ (bbl/day)$",
            "suptitle": r"Water Production Rate — $Q_{water}$",
            "fname": "WWPR.png",
        },
        "WGPR": {
            "ylabel": r"$Q_{gas}\ (scf/day)$",
            "suptitle": r"Gas Production Rate — $Q_{gas}$",
            "fname": "WGPR.png",
        },
    }
    vc = vcfg.get(ope_f, vcfg["WOPR"])

    # ── layout ────────────────────────────────────────────────────────────────
    ncols = min(N_wells, 5)
    nrows = int(np.ceil(N_wells / ncols))
    fig = plt.figure(figsize=(7.5 * ncols, 5.8 * nrows), constrained_layout=True)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.40, wspace=0.32)

    r2_ccr_all, r2_fno_all, r2_hybrid_all = [], [], []

    for k in range(N_wells):
        row, col = divmod(k, ncols)
        ax = fig.add_subplot(gs[row, col])

        y_true = True_mat[:, k]
        y_ccr = CCR_pred[:, k]
        y_fno = FNO_pred[:, k]
        y_hybrid = FNOCCR_pred[:, k]

        r2_ccr = r2(y_true, y_ccr) * 100.0
        r2_fno = r2(y_true, y_fno) * 100.0
        r2_hybrid = r2(y_true, y_hybrid) * 100.0
        r2_ccr_all.append(r2_ccr)
        r2_fno_all.append(r2_fno)
        r2_hybrid_all.append(r2_hybrid)

        # subtle fill between hybrid (the recommended one) and truth
        ax.fill_between(
            timezz, y_true, y_hybrid, alpha=0.10, color=CLR_HYBRID, label="_nolegend_"
        )

        # four curves
        ax.plot(
            timezz, y_true, color=CLR_TRUE, lw=2.4, label="Numerical (truth)", zorder=5
        )
        ax.plot(
            timezz,
            y_ccr,
            color=CLR_CCR,
            lw=2.1,
            linestyle="--",
            label=lbl_ccr,
            zorder=4,
        )
        ax.plot(
            timezz, y_fno, color=CLR_FNO, lw=2.1, linestyle=":", label=lbl_fno, zorder=4
        )
        ax.plot(
            timezz,
            y_hybrid,
            color=CLR_HYBRID,
            lw=2.3,
            linestyle="-.",
            label=lbl_hybrid,
            zorder=6,
        )

        # R² annotation — three lines, one per surrogate
        annot = (
            f"$R^2$ {lbl_ccr.split('—')[1].strip()} = {r2_ccr:.1f}%\n"
            f"$R^2$ {lbl_fno.split('—')[1].strip()} = {r2_fno:.1f}%\n"
            f"$R^2$ {lbl_hybrid.split('—')[1].strip()} = {r2_hybrid:.1f}%"
        )
        ax.annotate(
            annot,
            xy=(0.97, 0.05),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#1A5276",
            linespacing=1.55,
            bbox=dict(
                boxstyle="round,pad=0.40",
                facecolor="white",
                edgecolor="#AED6F1",
                linewidth=1.2,
                alpha=0.92,
            ),
        )

        # axes styling
        ax.set_xlabel("Time  (days)", fontsize=12, fontweight="bold", labelpad=6)
        ax.set_ylabel(vc["ylabel"], fontsize=12, fontweight="bold", labelpad=6)
        ax.set_title(well_names[k], fontsize=13, fontweight="bold", pad=8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(which="both", length=4)
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.45, color="#AAAAAA")
        ax.spines[["top", "right"]].set_visible(False)

        leg = ax.legend(
            loc="upper right", frameon=True, handlelength=2.2, borderpad=0.65
        )
        for text in leg.get_texts():
            text.set_fontweight("bold")

    # hide unused subplots
    for extra in range(N_wells, nrows * ncols):
        row, col = divmod(extra, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    # ── global title with overall R² ─────────────────────────────────────────
    r2_ccr_mean = float(np.mean(r2_ccr_all))
    r2_fno_mean = float(np.mean(r2_fno_all))
    r2_hybrid_mean = float(np.mean(r2_hybrid_all))

    fig.suptitle(
        f"{vc['suptitle']}\n"
        f"Mean $R^2$:  {lbl_ccr} = {r2_ccr_mean:.1f}%   |   "
        f"{lbl_fno} = {r2_fno_mean:.1f}%   |   "
        f"{lbl_hybrid} = {r2_hybrid_mean:.1f}%",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.savefig(
        vc["fname"], bbox_inches="tight", dpi=200, facecolor="white", edgecolor="none"
    )
    plt.clf()
    plt.close()

    print(
        f"  Saved {vc['fname']:<18} | "
        f"Mean R²  {lbl_ccr} = {r2_ccr_mean:.2f}%   "
        f"{lbl_fno} = {r2_fno_mean:.2f}%   "
        f"{lbl_hybrid} = {r2_hybrid_mean:.2f}%"
    )


# =============================================================================
# Wrapper — three plots (WOPR, WWPR, WGPR), each with four curves
# =============================================================================
def plot_all_comparisons(
    CCRhard_wopr,
    FNOpred_wopr,
    FNO_ccr_pred_wopr,
    Truedata_wopr,
    CCRhard_wwpr,
    FNOpred_wwpr,
    FNO_ccr_pred_wwpr,
    Truedata_wwpr,
    CCRhard_wgpr,
    FNOpred_wgpr,
    FNO_ccr_pred_wgpr,
    Truedata_wgpr,
    Time_vector,
    well_names,
    N_pr,
    cfg,
):
    """Wrapper — calls plot_rsm_three_curves for WOPR / WWPR / WGPR."""
    plot_rsm_three_curves(
        CCRhard_wopr,
        FNOpred_wopr,
        FNO_ccr_pred_wopr,
        Truedata_wopr,
        Time_vector,
        well_names,
        N_pr,
        "WOPR",
        cfg,
    )
    plot_rsm_three_curves(
        CCRhard_wwpr,
        FNOpred_wwpr,
        FNO_ccr_pred_wwpr,
        Truedata_wwpr,
        Time_vector,
        well_names,
        N_pr,
        "WWPR",
        cfg,
    )
    plot_rsm_three_curves(
        CCRhard_wgpr,
        FNOpred_wgpr,
        FNO_ccr_pred_wgpr,
        Truedata_wgpr,
        Time_vector,
        well_names,
        N_pr,
        "WGPR",
        cfg,
    )


# =============================================================================
# Main analysis routine
# =============================================================================
def compare_and_analyze_results(
    timing: "CompareTiming",
    grid: "CompareGrid",
    fields: "CompareFields",
    wells: "CompareWells",
    well_results: "CompareWellResults",
    runtime: "CompareRuntime",
    norms: "CompareNorms",
    surrogate: "CompareSurrogate",
    flow: "CompareFlow",
):
    """Run the full surrogate-vs-simulator comparison and analysis pipeline.

    Generates timing plots, per-timestep R²/L² accuracy panels, three-curve
    well comparisons, animated GIFs, and RMSE histograms.

    Parameters
    ----------
    physicsnemo_time : float
        Wall-clock time (seconds) for the surrogate forward pass.
    flow_time : float
        Wall-clock time (seconds) for the OPM Flow simulator run.
    nx : int
        Number of grid cells in the x-direction.
    ny : int
        Number of grid cells in the y-direction.
    nz : int
        Number of grid cells in the z-direction.
    steppi : int
        Number of sequential timesteps in the simulation window.
    steppi_indices : numpy.ndarray
        Integer indices selecting the active timesteps from the full sequence.
    Ne : int
        Number of ensemble members.
    pressure : numpy.ndarray
        Surrogate-predicted pressure field of shape (n_cells, steppi, Ne).
    pressure_true : numpy.ndarray
        Simulator pressure field of shape (n_cells, steppi, Ne).
    Swater : numpy.ndarray
        Surrogate-predicted water saturation field.
    Swater_true : numpy.ndarray
        Simulator water saturation field.
    Soil : numpy.ndarray
        Surrogate-predicted oil saturation field.
    Soil_true : numpy.ndarray
        Simulator oil saturation field.
    Sgas : numpy.ndarray
        Surrogate-predicted gas saturation field.
    Sgas_true : numpy.ndarray
        Simulator gas saturation field.
    ouut_peacemann : numpy.ndarray
        Surrogate well-rate predictions (oil, water, gas concatenated columns).
    out_fcn_true : numpy.ndarray
        Simulator well-rate ground truth (oil, water, gas concatenated columns).
    cfg : omegaconf.DictConfig
        Hydra configuration object with model and path settings.
    device : torch.device
        Device for any neural-operator inference calls.
    num_cores : int
        Number of CPU cores for parallel per-timestep processing.
    oldfolder : str
        Original working directory path to restore after file operations.
    folderr : str
        Output folder path where comparison plots are saved.
    N_injw : int
        Number of water injector wells.
    N_pr : int
        Number of producer wells.
    N_injg : int
        Number of gas injector wells.
    injectors : list
        Water injector well identifiers.
    producers : list
        Producer well identifiers.
    gas_injectors : list
        Gas injector well identifiers.
    well_names : list of str
        Names of all wells for plot labelling.
    inn : numpy.ndarray
        Input feature matrix used for the second forward-model timing run.
    min_inn_fcn : float
        Global minimum of input features for normalisation.
    max_inn_fcn : float
        Global maximum of input features for normalisation.
    target_min : float
        Minimum value for output de-normalisation.
    target_max : float
        Maximum value for output de-normalisation.
    minK : float
        Minimum permeability for de-normalisation.
    maxK : float
        Maximum permeability for de-normalisation.
    minT : float
        Minimum transmissibility for de-normalisation.
    maxT : float
        Maximum transmissibility for de-normalisation.
    minP : float
        Minimum pressure for de-normalisation.
    maxP : float
        Maximum pressure for de-normalisation.
    models : dict
        Dictionary of loaded surrogate model objects keyed by variable name.
    min_out_fcn : float
        Global minimum of output features for de-normalisation.
    max_out_fcn : float
        Global maximum of output features for de-normalisation.
    Time : numpy.ndarray
        Simulation time array of shape (1, steppi, nz, nx, ny).
    active_mask_3d : numpy.ndarray
        3D boolean active-cell mask of shape (nx, ny, nz).
    degg : int
        Polynomial degree for polynomial CCR expert models.
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
    compdat_data : dict
        Completion data dictionary for Peacemann well model configuration.
    output_variables : list of str
        Names of all output variables returned by the forward model.
    well_measurements : numpy.ndarray
        Observed well measurement data used for three-curve comparison.
    active_cells_ensemble : numpy.ndarray
        Boolean active-cell mask replicated for the ensemble.
    columns : list
        Column indices or labels for well output selection.
    lenwels : int
        Total number of wells (producers + injectors).
    awater : numpy.ndarray
        Water-rate scaling coefficient array.
    agas : numpy.ndarray
        Gas-rate scaling coefficient array.
    aoil : numpy.ndarray
        Oil-rate scaling coefficient array.
    aqq : numpy.ndarray
        Combined-rate scaling coefficient array.
    minQ : float
        Minimum oil rate for de-normalisation.
    maxQ : float
        Maximum oil rate for de-normalisation.
    minQw : float
        Minimum water rate for de-normalisation.
    maxQw : float
        Maximum water rate for de-normalisation.
    minQg : float
        Minimum gas rate for de-normalisation.
    maxQg : float
        Maximum gas rate for de-normalisation.

    Returns
    -------
    None
    """
    # ── 0. Unpack dataclasses into the local names used by the function body ─────
    physicsnemo_time = timing.physicsnemo_time
    flow_time = timing.flow_time

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    steppi = grid.steppi
    steppi_indices = grid.steppi_indices
    Ne = grid.Ne

    pressure, pressure_true = fields.pressure_pred, fields.pressure_true
    Swater, Swater_true = fields.water_pred, fields.water_true
    Soil, Soil_true = fields.oil_pred, fields.oil_true
    Sgas, Sgas_true = fields.gas_pred, fields.gas_true

    N_pr, N_injw, N_injg = wells.N_pr, wells.N_injw, wells.N_injg
    lenwels = wells.lenwels
    injectors = wells.injectors
    producers = wells.producers
    gas_injectors = wells.gas_injectors
    well_names = wells.well_names
    # ``columns`` is part of the bundle so callers can preserve it, but the
    # current function body does not consume it.
    _columns = wells.columns
    compdat_data = wells.compdat_data

    ouut_peacemann = well_results.ouut_peacemann
    out_fcn_true = well_results.out_fcn_true

    cfg = runtime.cfg
    device = runtime.device
    num_cores = runtime.num_cores
    oldfolder = runtime.oldfolder
    folderr = runtime.folderr
    output_variables = runtime.output_variables
    well_measurements = runtime.well_measurements

    min_inn_fcn, max_inn_fcn = norms.min_inn_fcn, norms.max_inn_fcn
    min_out_fcn, max_out_fcn = norms.min_out_fcn, norms.max_out_fcn
    min_inn_fcn2, max_inn_fcn2 = norms.min_inn_fcn2, norms.max_inn_fcn2
    min_out_fcn2, max_out_fcn2 = norms.min_out_fcn2, norms.max_out_fcn2
    target_min, target_max = norms.target_min, norms.target_max
    minK, maxK = norms.minK, norms.maxK
    minT, maxT = norms.minT, norms.maxT
    minP, maxP = norms.minP, norms.maxP
    minQ, maxQ = norms.minQ, norms.maxQ
    minQw, maxQw = norms.minQw, norms.maxQw
    minQg, maxQg = norms.minQg, norms.maxQg

    models = surrogate.models
    degg = surrogate.degg
    experts = surrogate.experts
    inn = surrogate.inn

    active_cells_ensemble = flow.active_cells_ensemble
    active_mask_3d = flow.active_mask_3d
    awater = flow.awater
    agas = flow.agas
    aoil = flow.aoil
    aqq = flow.aqq
    # ``Time`` is rebuilt below from the simulator data; keep the unpacked value
    # available so call sites that pass it pre-computed don't drop it silently.
    _Time_in = flow.Time

    # ── 1. First timing comparison (caller-supplied flow_time vs physicsnemo_time) ─
    _plot_time_comparison(physicsnemo_time, flow_time, folderr, oldfolder)

    # ── 2. Build Time vector ─────────────────────────────────────────────────────
    os.chdir(to_absolute_path("../simulator_data"))
    Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)
    Time_unie = np.zeros(steppi)
    for i in range(steppi):
        Time_unie[i] = Time[0, i, 0, 0, 0]
    os.chdir(to_absolute_path(oldfolder))
    dt = Time_unie
    Time_vector = Time_unie

    # ── 3. Per-step pressure / saturation accuracy (R² + L²) ─────────────────────
    Accuracy_presure = np.zeros((steppi, 2))
    Accuracy_oil = np.zeros((steppi, 2))
    Accuracy_water = np.zeros((steppi, 2))
    Accuracy_gas = np.zeros((steppi, 2))

    results = Parallel(n_jobs=num_cores, backend="loky", verbose=10)(
        delayed(process_step)(
            kk,
            steppi,
            dt,
            pressure,
            active_cells_ensemble,
            pressure_true,
            Swater,
            Swater_true,
            Soil,
            Soil_true,
            Sgas,
            Sgas_true,
            nx,
            ny,
            nz,
            N_injw,
            N_pr,
            N_injg,
            injectors,
            producers,
            gas_injectors,
            folderr,
            oldfolder,
            Accuracy_presure,
            Accuracy_oil,
            Accuracy_water,
            Accuracy_gas,
        )
        for kk in range(steppi)
    )
    os.chdir(to_absolute_path(oldfolder))

    progressBar = "\rPlotting Progress: " + ProgressBar(
        steppi - 1, steppi - 1, steppi - 1
    )
    ShowBar(progressBar)
    time.sleep(1)

    for kk, (R2p, L2p, R2w, L2w, R2o, L2o, R2g, L2g) in enumerate(results):
        Accuracy_presure[kk, 0] = R2p
        Accuracy_presure[kk, 1] = L2p
        Accuracy_water[kk, 0] = R2w
        Accuracy_water[kk, 1] = L2w
        Accuracy_oil[kk, 0] = R2o
        Accuracy_oil[kk, 1] = L2o
        Accuracy_gas[kk, 0] = R2g
        Accuracy_gas[kk, 1] = L2g

    # ── 4. R²/L² 2x4 panel (pressure, water, oil, gas) ───────────────────────────
    os.chdir(to_absolute_path(folderr))
    # ── style ────────────────────────────────────────────────────────────────
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.4,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
        }
    )

    # Accent colors per quantity (consistent across R²/L² rows)
    PANEL_CLR = {
        "Pressure": "#C0392B",  # deep red
        "water saturation": "#2471A3",  # steel blue
        "oil saturation": "#27AE60",  # green
        "gas saturation": "#D68910",  # amber
    }

    fig4, axes = plt.subplots(
        2,
        4,
        figsize=(18, 9),
        constrained_layout=True,
    )
    fig4.patch.set_facecolor("white")

    panels = [
        # row, col, title,                series,                  ylabel,    metric
        (0, 0, "Pressure", Accuracy_presure[:, 0], "R²  ", "R²"),
        (0, 1, "water saturation", Accuracy_water[:, 0], "R²  ", "R²"),
        (0, 2, "oil saturation", Accuracy_oil[:, 0], "R²  ", "R²"),
        (0, 3, "gas saturation", Accuracy_gas[:, 0], "R²  ", "R²"),
        (1, 0, "Pressure", Accuracy_presure[:, 1], "L²  ", "L²"),
        (1, 1, "water saturation", Accuracy_water[:, 1], "L²  ", "L²"),
        (1, 2, "oil saturation", Accuracy_oil[:, 1], "L²  ", "L²"),
        (1, 3, "gas saturation", Accuracy_gas[:, 1], "L²  ", "L²"),
    ]

    for r, c, title, series, ylab, metric in panels:
        ax = axes[r, c]
        clr = PANEL_CLR[title]

        ax.plot(
            Time_vector,
            series,
            color=clr,
            linewidth=2.0,
            marker="o",
            markersize=6,
            markerfacecolor=clr,
            markeredgecolor="white",
            markeredgewidth=1.2,
            label=metric,
            zorder=4,
        )

        # Mean line for quick visual reference
        mean_val = float(np.nanmean(series))
        ax.axhline(
            mean_val,
            color=clr,
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
            zorder=3,
        )
        ax.annotate(
            f"mean = {mean_val:.2f}",
            xy=(0.97, 0.06),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=clr,
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor="white",
                edgecolor=clr,
                linewidth=1.0,
                alpha=0.92,
            ),
        )

        ax.set_title(title.title(), fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Time (days)", fontsize=11, fontweight="bold", labelpad=4)
        ax.set_ylabel(ylab, fontsize=11, fontweight="bold", labelpad=4)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45, color="#AAAAAA")
        ax.spines[["top", "right"]].set_visible(False)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    # ── row super-titles ─────────────────────────────────────────────────────
    fig4.text(
        0.5,
        1.02,
        "PhysicsNeMo vs Numerical (GPU) — Accuracy over Time",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#1A5276",
    )
    fig4.text(
        0.5,
        0.97,
        r"Top: $R^{2}$      |     Bottom: $L^{2}$ ",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#566573",
    )

    plt.savefig(
        "R2L2.png",
        bbox_inches="tight",
        dpi=200,
        facecolor="white",
        edgecolor="none",
    )
    plt.clf()
    plt.close()

    # ── 5. Animated GIF of dynamic snapshots ─────────────────────────────────────
    print("Now - Creating GIF")
    imgs = sorted(
        glob.glob("*Dynamic*"),
        key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)],
    )
    frames = [Image.open(i) for i in imgs]
    frames[0].save(
        "Evolution.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=500,
        loop=0,
    )

    # ── 6. CSV writes & first percentile plot (CCR baseline) ────────────────────
    print("Saving prediction in CSV file")
    write_rsm(ouut_peacemann[0, :, :], Time_vector, "PhysicsNeMo", well_names, N_pr)
    write_rsm(out_fcn_true[0, :, :], Time_vector, "Flow", well_names, N_pr)
    CCRhard = ouut_peacemann[0, :, :]
    Truedata = out_fcn_true[0, :, :]
    print("Plotting well responses and accuracies")

    # ── 7. Slice CCR predictions per phase (single block of 3*N_pr cols) ────────
    CCRhard_wopr = CCRhard[:, :N_pr]
    CCRhard_wwpr = CCRhard[:, N_pr : 2 * N_pr]
    CCRhard_wgpr = CCRhard[:, 2 * N_pr : 3 * N_pr]
    Truedata_wopr = Truedata[:, :N_pr]
    Truedata_wwpr = Truedata[:, N_pr : 2 * N_pr]
    Truedata_wgpr = Truedata[:, 2 * N_pr : 3 * N_pr]

    plot_rsm_percentile(
        CCRhard_wopr, Truedata_wopr, Time_vector, well_names, N_pr, "WOPR"
    )
    plot_rsm_percentile(
        CCRhard_wwpr, Truedata_wwpr, Time_vector, well_names, N_pr, "WWPR"
    )
    plot_rsm_percentile(
        CCRhard_wgpr, Truedata_wgpr, Time_vector, well_names, N_pr, "WGPR"
    )

    # ── 8. Switch to FNO comparison folder, copy GIF/R2L2 from CCR ──────────────
    os.chdir(to_absolute_path(oldfolder))
    Trainmoe = "FNO"
    pred_type = 1
    logger.info(
        "----------------------------------------------------------------------"
    )
    logger.info("Using FNO for peacemann model           ")
    # folderr1,_, source_directory, destination_directory1,_ = _select_compare_folder(cfg)
    folderr1, _, source_directory, destination_directory1, _ = _select_compare_folder(
        cfg, active="fno"
    )

    for fname in ("Evolution.gif", "R2L2.png"):
        src = to_absolute_path(os.path.join(source_directory, fname))
        dst = to_absolute_path(os.path.join(destination_directory1, fname))
        shutil.copy(src, dst)

    # ── 9. Re-run forward model with FNO Peacemann head ─────────────────────────
    start_time_plots2 = time.time()
    active_mask_3d_yes = active_mask_3d
    simout = Forward_model_ensemble(
        Ne,
        inn,
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
        active_mask_3d,
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
        well_measurements,
        cfg,
        N_pr,
        lenwels,
        active_mask_3d_yes,
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
    elapsed_time_secs2 = time.time() - start_time_plots2
    msg = f"Reservoir simulation with NVIDIA PhysicsNeMo (FNO)  took: {timedelta(seconds=round(elapsed_time_secs2))} secs (Wall clock time)"
    print(msg)

    if "PRESSURE" in output_variables:
        pressure = simout["PRESSURE"]
    if "SWAT" in output_variables:
        Swater = simout["SWAT"]
    if "SOIL" in output_variables:
        Soil = simout["SOIL"]
    if "SGAS" in output_variables:
        Sgas = simout["SGAS"]
    ouut_fno = simout["ouut_p"]
    physicsnemo_time = elapsed_time_secs2

    # ── 10. Second timing comparison (FNO physicsnemo_time vs same flow_time) ────
    _plot_time_comparison(physicsnemo_time, flow_time, folderr1, oldfolder)

    # ── 11. Rebuild Time vector (matches original behaviour) ─────────────────────
    os.chdir(to_absolute_path("../simulator_data"))
    Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)
    Time_unie = np.zeros(steppi)
    for i in range(steppi):
        Time_unie[i] = Time[0, i, 0, 0, 0]
    os.chdir(oldfolder)
    dt = Time_unie

    print("Plotting outputs")
    os.chdir(to_absolute_path(folderr1))
    Time_vector = np.zeros(steppi)
    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Time_vector[kk] = dt[kk]
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    # ── 12. CSV writes & second percentile plot (FNO) ────────────────────────────
    print("Saving prediction in CSV file")
    write_rsm(ouut_fno[0, :, :], Time_vector, "PhysicsNeMo", well_names, N_pr)
    write_rsm(out_fcn_true[0, :, :], Time_vector, "Flow", well_names, N_pr)
    print("Plotting well responses and accuracies")
    # plot_rsm_percentile(ouut_peacemann[0, :, :], out_fcn_true[0, :, :],
    # Time_vector, well_names, N_pr)

    FNOpred = ouut_fno[0, :, :]
    FNOpred_wopr = FNOpred[:, :N_pr]
    FNOpred_wwpr = FNOpred[:, N_pr : 2 * N_pr]
    FNOpred_wgpr = FNOpred[:, 2 * N_pr : 3 * N_pr]

    plot_rsm_percentile(
        FNOpred_wopr, Truedata_wopr, Time_vector, well_names, N_pr, "WOPR"
    )
    plot_rsm_percentile(
        FNOpred_wwpr, Truedata_wwpr, Time_vector, well_names, N_pr, "WWPR"
    )
    plot_rsm_percentile(
        FNOpred_wgpr, Truedata_wgpr, Time_vector, well_names, N_pr, "WGPR"
    )
    os.chdir(oldfolder)
    # ── 13. Three-curve comparison plots (truth / CCR / FNO) for WOPR/WWPR/WGPR ──

    Trainmoe = "BOTH"
    logger.info("Using FNO and CCR for peacemann model           ")
    # _, folderr, source_directory,_, destination_directory = _select_compare_folder(cfg)
    _, folderr, source_directory, _, destination_directory = _select_compare_folder(
        cfg, active="fno_ccr"
    )

    for fname in ("Evolution.gif", "R2L2.png"):
        src = to_absolute_path(os.path.join(source_directory, fname))
        dst = to_absolute_path(os.path.join(destination_directory, fname))
        shutil.copy(src, dst)

    # ── 9. Re-run forward model with FNO Peacemann head ─────────────────────────
    start_time_plots3 = time.time()
    active_mask_3d_yes = active_mask_3d
    simout = Forward_model_ensemble(
        Ne,
        inn,
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
        active_mask_3d,
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
        well_measurements,
        cfg,
        N_pr,
        lenwels,
        active_mask_3d_yes,
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
    elapsed_time_secs2 = time.time() - start_time_plots3
    msg = f"Reservoir simulation with NVIDIA PhysicsNeMo (FNO + CCR)  took: {timedelta(seconds=round(elapsed_time_secs2))} secs (Wall clock time)"
    print(msg)

    if "PRESSURE" in output_variables:
        pressure = simout["PRESSURE"]
    if "SWAT" in output_variables:
        Swater = simout["SWAT"]
    if "SOIL" in output_variables:
        Soil = simout["SOIL"]
    if "SGAS" in output_variables:
        Sgas = simout["SGAS"]
    ouut_fno_ccr = simout["ouut_p"]
    physicsnemo_time2 = elapsed_time_secs2

    # ── 10. Second timing comparison (FNO physicsnemo_time vs same flow_time) ────
    _plot_time_comparison(physicsnemo_time2, flow_time, folderr, oldfolder)

    # ── 11. Rebuild Time vector (matches original behaviour) ─────────────────────
    os.chdir(to_absolute_path("../simulator_data"))
    Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)
    Time_unie = np.zeros(steppi)
    for i in range(steppi):
        Time_unie[i] = Time[0, i, 0, 0, 0]
    os.chdir(oldfolder)
    dt = Time_unie

    os.chdir(to_absolute_path(folderr))
    Time_vector = np.zeros(steppi)
    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Time_vector[kk] = dt[kk]
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    # ── 12. CSV writes & second percentile plot (FNO) ────────────────────────────
    print("Saving prediction in CSV file")
    write_rsm(ouut_fno_ccr[0, :, :], Time_vector, "PhysicsNeMo", well_names, N_pr)
    write_rsm(out_fcn_true[0, :, :], Time_vector, "Flow", well_names, N_pr)
    print("Plotting well responses and accuracies")
    # plot_rsm_percentile(ouut_peacemann[0, :, :], out_fcn_true[0, :, :],
    # Time_vector, well_names, N_pr)

    FNO_ccr_pred = ouut_fno_ccr[0, :, :]
    FNO_ccr_pred_wopr = FNO_ccr_pred[:, :N_pr]
    FNO_ccr_pred_wwpr = FNO_ccr_pred[:, N_pr : 2 * N_pr]
    FNO_ccr_pred_wgpr = FNO_ccr_pred[:, 2 * N_pr : 3 * N_pr]

    plot_rsm_percentile(
        FNO_ccr_pred_wopr, Truedata_wopr, Time_vector, well_names, N_pr, "WOPR"
    )
    plot_rsm_percentile(
        FNO_ccr_pred_wwpr, Truedata_wwpr, Time_vector, well_names, N_pr, "WWPR"
    )
    plot_rsm_percentile(
        FNO_ccr_pred_wgpr, Truedata_wgpr, Time_vector, well_names, N_pr, "WGPR"
    )

    # ── 13. Three-curve comparison plots (truth / CCR / FNO) for WOPR/WWPR/WGPR ──
    os.chdir(oldfolder)

    os.chdir(to_absolute_path("../RESULTS/FORWARD_RESULTS/RESULTS/COMPARE_RESULTS"))

    plot_all_comparisons(
        CCRhard_wopr,
        FNOpred_wopr,
        FNO_ccr_pred_wopr,
        Truedata_wopr,
        CCRhard_wwpr,
        FNOpred_wwpr,
        FNO_ccr_pred_wwpr,
        Truedata_wwpr,
        CCRhard_wgpr,
        FNOpred_wgpr,
        FNO_ccr_pred_wgpr,
        Truedata_wgpr,
        Time_vector,
        well_names,
        N_pr,
        cfg,
    )

    # ── 14. RMSE summary + histogram (CCR vs FNO vs FNO+CCR hybrid) ──────────
    def compute_rmse(pred, true):
        """Compute normalised RMSE between predicted and true arrays.

        Parameters
        ----------
        pred : numpy.ndarray
            Predicted values; reshaped column-major to (-1, 1).
        true : numpy.ndarray
            Ground-truth values; reshaped column-major to (-1, 1).

        Returns
        -------
        float
            Normalised root-mean-square error (Frobenius norm divided by n_elements).
        """
        pred = np.reshape(pred, (-1, 1), "F")
        true = np.reshape(true, (-1, 1), "F")
        return ((np.sum((pred - true) ** 2)) ** 0.5) / true.shape[0]

    rmse = {
        "wopr": {
            "CCR": compute_rmse(CCRhard_wopr, Truedata_wopr),
            "FNO": compute_rmse(FNOpred_wopr, Truedata_wopr),
            "FNO+CCR": compute_rmse(FNO_ccr_pred_wopr, Truedata_wopr),
        },
        "wwpr": {
            "CCR": compute_rmse(CCRhard_wwpr, Truedata_wwpr),
            "FNO": compute_rmse(FNOpred_wwpr, Truedata_wwpr),
            "FNO+CCR": compute_rmse(FNO_ccr_pred_wwpr, Truedata_wwpr),
        },
        "wgpr": {
            "CCR": compute_rmse(CCRhard_wgpr, Truedata_wgpr),
            "FNO": compute_rmse(FNOpred_wgpr, Truedata_wgpr),
            "FNO+CCR": compute_rmse(FNO_ccr_pred_wgpr, Truedata_wgpr),
        },
    }

    avg_ccr = np.mean([rmse[t]["CCR"] for t in rmse])
    avg_fno = np.mean([rmse[t]["FNO"] for t in rmse])
    avg_hybrid = np.mean([rmse[t]["FNO+CCR"] for t in rmse])

    avg_scores = {"CCR": avg_ccr, "FNO": avg_fno, "FNO+CCR": avg_hybrid}
    best = min(avg_scores, key=avg_scores.get)
    best_avg = avg_scores[best]

    print("=" * 70)
    print(
        f"  WOPR — CCR: {rmse['wopr']['CCR']:.4f}   "
        f"FNO: {rmse['wopr']['FNO']:.4f}   "
        f"FNO+CCR: {rmse['wopr']['FNO+CCR']:.4f}"
    )
    print(
        f"  WWPR — CCR: {rmse['wwpr']['CCR']:.4f}   "
        f"FNO: {rmse['wwpr']['FNO']:.4f}   "
        f"FNO+CCR: {rmse['wwpr']['FNO+CCR']:.4f}"
    )
    print(
        f"  WGPR — CCR: {rmse['wgpr']['CCR']:.4f}   "
        f"FNO: {rmse['wgpr']['FNO']:.4f}   "
        f"FNO+CCR: {rmse['wgpr']['FNO+CCR']:.4f}"
    )
    print("-" * 70)
    print(
        f"  Avg RMSE — CCR: {avg_ccr:.4f}   "
        f"FNO: {avg_fno:.4f}   "
        f"FNO+CCR: {avg_hybrid:.4f}"
    )
    print(f"  Recommended model : Neural Operator — {best}")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    tags = ["wopr", "wwpr", "wgpr"]
    titles = [
        "Oil Production Rate (WOPR)",
        "Water Production Rate (WWPR)",
        "Gas Production Rate (WGPR)",
    ]
    colors = ["#2471A3", "#D68910", "#1E8449"]  # blue, orange, green
    labels = [
        "Neural Operator — CCR",
        "Neural Operator — FNO",
        "Neural Operator — FNO+CCR",
    ]
    keys = ["CCR", "FNO", "FNO+CCR"]

    for ax, tag, title in zip(axes.flatten(), tags, titles, strict=False):
        values = [rmse[tag][k] for k in keys]
        bars = ax.bar(
            labels, values, color=colors, width=0.55, edgecolor="white", linewidth=0.8
        )

        for bar, v in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        best_idx = int(np.argmin(values))
        best_bar = bars[best_idx]
        ax.text(
            best_bar.get_x() + best_bar.get_width() / 2,
            best_bar.get_height() + max(values) * 0.08,
            "★ best",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1E8449",
            fontweight="bold",
        )

        ax.set_title(title, fontweight="bold", fontsize=11, pad=10)
        ax.set_ylabel("RMSE", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_ylim(0, max(values) * 1.30)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, color="#AAAAAA")
        ax.legend(
            bars,
            labels,
            loc="upper right",
            fontsize=8.5,
            framealpha=0.88,
            title="Model",
            title_fontsize=9,
        )
        for text in ax.get_legend().get_texts():
            text.set_fontweight("bold")

    fig.suptitle(
        f"Surrogate Model RMSE Comparison\n"
        f"Best overall: Neural Operator — {best}  "
        f"(avg RMSE = {best_avg:.4f})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        "Histogram.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.clf()
    plt.close()
    os.chdir(oldfolder)
