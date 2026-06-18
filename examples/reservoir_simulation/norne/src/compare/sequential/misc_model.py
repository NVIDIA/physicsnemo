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
                    SEQUENTIAL MODEL UTILITIES
=====================================================================

This module provides model utilities for sequential FVM surrogate model
comparisons. It includes functions for model operations, visualization,
and analysis.

Key Features:
- FNO model operations and visualization
- Model performance metrics
- Data processing and transformation
- Visualization utilities

Usage:
    from compare.sequential.misc_model import (
        Plot_PhysicsNeMo,
        compute_metrics,
        process_step,
        run_gnn_model
    )

@Author : Clement Etienam
"""

import os
import time
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib import cm

from compare.sequential.misc_operations import (
    ProgressBar,
    ShowBar,
)


def Plot_PhysicsNeMo(
    ax,
    nx,
    ny,
    nz,
    Truee,
    N_injw,
    N_pr,
    N_injg,
    varii,
    injectors,
    producers,
    gas_injectors,
):
    """
    Render a 3-D voxel field on a Matplotlib 3-D axes with well annotations and a colourbar.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Existing 3-D axes to draw into.
    nx : int
        Grid dimension in the X direction.
    ny : int
        Grid dimension in the Y direction.
    nz : int
        Grid dimension in the Z direction.
    Truee : np.ndarray
        Flattened 1-D field values of length nx*ny*nz.
    N_injw : int
        Number of water injector wells.
    N_pr : int
        Number of producer wells.
    N_injg : int
        Number of gas injector wells.
    varii : str
        Label key controlling colourbar text and title (e.g. 'perm', 'pressure PhysicsNeMo').
    injectors : list of tuple
        Water injector info; each tuple provides (x, y, ..., well_name).
    producers : list of tuple
        Producer info; each tuple provides (x, y, ..., well_name).
    gas_injectors : list of tuple
        Gas injector info; each tuple provides (x, y, ..., well_name).

    Returns
    -------
    None
    """
    # ── field reshape & normalise ─────────────────────────────────────────────
    Pressz = np.reshape(Truee, (nx, ny, nz), "F")
    maxii = max(Pressz.ravel())
    minii = min(Pressz.ravel())
    Pressz = Pressz / (maxii + 1e-12)

    colors = plt.cm.jet(Pressz)
    colors[np.isnan(Pressz), :3] = 1
    norm = mpl.colors.Normalize(vmin=minii, vmax=maxii)

    # ── voxel render ──────────────────────────────────────────────────────────
    ax.voxels(Pressz, facecolors=colors, alpha=0.52, edgecolor="none", shade=True)

    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])

    # ── axis labels (large, bold) ─────────────────────────────────────────────
    AXIS_FS = 16  # axis label fontsize
    TICK_FS = 13  # tick label fontsize  (hidden here but kept for reference)
    WELL_FS = 11  # well name annotation
    LEG_FS = 12  # legend

    ax.set_xlabel("X", fontsize=AXIS_FS, fontweight="bold", labelpad=8)
    ax.set_ylabel("Y", fontsize=AXIS_FS, fontweight="bold", labelpad=8)
    ax.set_zlabel("Z", fontsize=AXIS_FS, fontweight="bold", labelpad=8)

    # ── axis limits & appearance ──────────────────────────────────────────────
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)

    ax.grid(False)
    ax.set_box_aspect([nx, ny, nz])
    ax.set_proj_type("ortho")

    # hide tick labels but keep pane structure
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["tick"]["inward_factor"] = 0
        axis._axinfo["tick"]["outward_factor"] = 0.4

    ax.view_init(elev=30, azim=45)

    WELL_FS = 9  # was 11 — smaller helps with crowding

    # Helper to stagger label heights when wells are close together
    def _label_offset(idx, base_z, n_levels=4, step=2.5):
        """Cycle through n_levels heights so adjacent labels don't overlap."""
        return base_z + (idx % n_levels) * step

    # ── water injector lines & labels ────────────────────────────────────────
    for mm in range(N_injw):
        usethis = injectors[mm]
        xloc = int(np.asarray(usethis[0]).flat[0])
        yloc = int(np.asarray(usethis[1]).flat[0])
        discrip = str(usethis[-1])
        z_stick = (nz * 2) + 7
        z_label = _label_offset(mm, z_stick + 0.5)
        ax.plot(
            [xloc, xloc],
            [yloc, yloc],
            [0, z_stick],
            color="#1565C0",
            linewidth=1.6,
            zorder=5,
        )
        ax.text(
            xloc,
            yloc,
            z_label,
            discrip,
            color="#0D47A1",
            fontsize=WELL_FS,
            fontweight="bold",
            ha="center",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
            ),
        )

    # ── producer lines & labels ──────────────────────────────────────────────
    for mm in range(N_pr):
        usethis = producers[mm]
        xloc = int(np.asarray(usethis[0]).flat[0])
        yloc = int(np.asarray(usethis[1]).flat[0])
        discrip = str(usethis[-1])
        z_stick = (nz * 2) + 5
        z_label = _label_offset(mm, z_stick + 0.5)
        ax.plot(
            [xloc, xloc],
            [yloc, yloc],
            [0, z_stick],
            color="#2E7D32",
            linewidth=1.6,
            zorder=5,
        )
        ax.text(
            xloc,
            yloc,
            z_label,
            discrip,
            color="#1B5E20",
            fontsize=WELL_FS,
            fontweight="bold",
            ha="center",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
            ),
        )

    # ── gas injector lines & labels ──────────────────────────────────────────
    for mm in range(N_injg):
        usethis = gas_injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        z_stick = (nz * 2) + 5
        z_label = _label_offset(mm, z_stick + 0.5)
        ax.plot(
            [xloc, xloc],
            [yloc, yloc],
            [0, z_stick],
            color="#C62828",
            linewidth=1.6,
            zorder=5,
        )
        ax.text(
            xloc,
            yloc,
            z_label,
            discrip,
            color="#B71C1C",
            fontsize=WELL_FS,
            fontweight="bold",
            ha="center",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
            ),
        )

    blue_line = mlines.Line2D(
        [], [], color="blue", linewidth=2.2, label="water injector"
    )
    green_line = mlines.Line2D(
        [], [], color="green", linewidth=2.2, label="oil/water/gas producer"
    )
    red_line = mlines.Line2D([], [], color="red", linewidth=2.2, label="gas injectors")

    leg = ax.legend(
        handles=[blue_line, green_line, red_line],
        loc="lower left",
        fontsize=LEG_FS,
        framealpha=0.88,
        edgecolor="#CCCCCC",
    )

    for text in leg.get_texts():
        text.set_fontweight("bold")

    # ── colourbar ─────────────────────────────────────────────────────────────
    cbar = plt.colorbar(m, ax=ax, orientation="horizontal", shrink=0.52, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FS)

    if varii == "perm":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title(
            "Permeability Field with well locations", fontsize=12, weight="bold"
        )
    elif varii == "water PhysicsNeMo":
        cbar.set_label("water saturation", fontsize=12)
        ax.set_title("water saturation -PhysicsNeMo", fontsize=12, weight="bold")
    elif varii == "water Numerical":
        cbar.set_label("water saturation", fontsize=12)
        ax.set_title("water saturation - Numerical(Flow)", fontsize=12, weight="bold")
    elif varii == "water diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "water saturation - (Numerical(Flow) -PhysicsNeMo))",
            fontsize=12,
            weight="bold",
        )
    elif varii == "oil PhysicsNeMo":
        cbar.set_label("Oil saturation", fontsize=12)
        ax.set_title("Oil saturation -PhysicsNeMo", fontsize=12, weight="bold")
    elif varii == "oil Numerical":
        cbar.set_label("Oil saturation", fontsize=12)
        ax.set_title("Oil saturation - Numerical(Flow)", fontsize=12, weight="bold")
    elif varii == "oil diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "oil saturation - (Numerical(Flow) -PhysicsNeMo))",
            fontsize=12,
            weight="bold",
        )
    elif varii == "gas PhysicsNeMo":
        cbar.set_label("Gas saturation", fontsize=12)
        ax.set_title("Gas saturation -PhysicsNeMo", fontsize=12, weight="bold")
    elif varii == "gas Numerical":
        cbar.set_label("Gas saturation", fontsize=12)
        ax.set_title("Gas saturation - Numerical(Flow)", fontsize=12, weight="bold")
    elif varii == "gas diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "gas saturation - (Numerical(Flow) -PhysicsNeMo))",
            fontsize=12,
            weight="bold",
        )
    elif varii == "pressure PhysicsNeMo":
        cbar.set_label("pressure", fontsize=12)
        ax.set_title("Pressure -PhysicsNeMo", fontsize=12, weight="bold")
    elif varii == "pressure Numerical":
        cbar.set_label("pressure", fontsize=12)
        ax.set_title("Pressure -Numerical(Flow)", fontsize=12, weight="bold")
    elif varii == "pressure diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "Pressure - (Numerical(Flow) -PhysicsNeMo))", fontsize=12, weight="bold"
        )
    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=12)
        ax.set_title("Porosity Field", fontsize=12, weight="bold")
    if varii == "P10":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title("P10 Reservoir Model", fontsize=12, weight="bold")
    if varii == "P50":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title("P50 Reservoir Model", fontsize=12, weight="bold")
    if varii == "P90":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title("P90 Reservoir Model", fontsize=12, weight="bold")
    if varii == "True model":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title("True Reservoir Model", fontsize=12, weight="bold")
    if varii == "Prior":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title("initial Reservoir Model", fontsize=12, weight="bold")
    if varii == "cumm-mean":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title("Cummulative mean Reservoir Model", fontsize=12, weight="bold")
    if varii == "cumm-best":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title("Cummulative best Reservoir Model", fontsize=12, weight="bold")
    cbar.mappable.set_clim(minii, maxii)


def compute_metrics(y_true, y_pred):
    """
    Compute R-squared and relative L2 accuracy between true and predicted arrays.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values (arbitrary shape, will be compared element-wise).
    y_pred : np.ndarray
        Predicted values with the same shape as y_true.

    Returns
    -------
    R2 : float
        Coefficient of determination (1 - RSS/TSS).
    L2_accuracy : float
        Relative L2 accuracy (1 - norm(residual) / norm(total)).
    """
    y_true_mean = np.mean(y_true)
    TSS = np.sum((y_true - y_true_mean) ** 2)
    RSS = np.sum((y_true - y_pred) ** 2)

    R2 = 1 - (RSS / TSS)
    L2_accuracy = 1 - np.sqrt(RSS) / np.sqrt(TSS)
    return R2, L2_accuracy


def process_step(
    kk,
    steppi,
    dt,
    pressure,
    active_mask,
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
    fol,
    fol1,
    Accuracy_presure,
    Accuracy_oil,
    Accuracy_water,
    Accuracy_gas,
):
    """
    Plot surrogate vs. numerical 3-D fields for a single time step and record accuracy metrics.

    Parameters
    ----------
    kk : int
        Current time-step index (1-based).
    steppi : int
        Total number of time steps.
    dt : np.ndarray
        Array of time values in days, indexed by kk.
    pressure : np.ndarray
        Surrogate pressure field of shape (1, steppi, nx, ny, nz).
    active_mask : np.ndarray
        Boolean or float mask of shape (nx, ny, nz) for active cells.
    pressure_true : np.ndarray
        Numerical (ground-truth) pressure field of shape (1, steppi, nx, ny, nz).
    Swater : np.ndarray
        Surrogate water saturation field of shape (1, steppi, nx, ny, nz).
    Swater_true : np.ndarray
        Numerical water saturation field of shape (1, steppi, nx, ny, nz).
    Soil : np.ndarray
        Surrogate oil saturation field of shape (1, steppi, nx, ny, nz).
    Soil_true : np.ndarray
        Numerical oil saturation field of shape (1, steppi, nx, ny, nz).
    Sgas : np.ndarray
        Surrogate gas saturation field of shape (1, steppi, nx, ny, nz).
    Sgas_true : np.ndarray
        Numerical gas saturation field of shape (1, steppi, nx, ny, nz).
    nx : int
        Grid dimension in the X direction.
    ny : int
        Grid dimension in the Y direction.
    nz : int
        Grid dimension in the Z direction.
    N_injw : int
        Number of water injector wells.
    N_pr : int
        Number of producer wells.
    N_injg : int
        Number of gas injector wells.
    injectors : list of tuple
        Water injector location/name tuples.
    producers : list of tuple
        Producer location/name tuples.
    gas_injectors : list of tuple
        Gas injector location/name tuples.
    fol : str
        Output directory for saving the figure.
    fol1 : str
        Secondary directory (currently unreachable after return).
    Accuracy_presure : np.ndarray
        Array of shape (steppi, 2) accumulating (R2, L2) for pressure.
    Accuracy_oil : np.ndarray
        Array of shape (steppi, 2) accumulating (R2, L2) for oil saturation.
    Accuracy_water : np.ndarray
        Array of shape (steppi, 2) accumulating (R2, L2) for water saturation.
    Accuracy_gas : np.ndarray
        Array of shape (steppi, 2) accumulating (R2, L2) for gas saturation.

    Returns
    -------
    R2p : float
        R-squared for pressure at this time step.
    L2p : float
        L2 accuracy for pressure at this time step.
    R2w : float
        R-squared for water saturation at this time step.
    L2w : float
        L2 accuracy for water saturation at this time step.
    R2o : float
        R-squared for oil saturation at this time step.
    L2o : float
        L2 accuracy for oil saturation at this time step.
    R2g : float
        R-squared for gas saturation at this time step.
    L2g : float
        L2 accuracy for gas saturation at this time step.
    """
    os.chdir(fol)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk - 1, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    current_time = dt[kk]
    # Time_vector[kk] = current_time

    f_3 = plt.figure(figsize=(20, 20), dpi=200)

    look = ((pressure[0, kk, :, :, :]) * active_mask)[:, :, ::-1]

    lookf = ((pressure_true[0, kk, :, :, :]) * active_mask)[:, :, ::-1]
    # lookf = lookf * pini_alt
    diff1 = ((abs(look - lookf)) * active_mask)[:, :, ::-1]

    ax1 = f_3.add_subplot(4, 3, 1, projection="3d")
    Plot_PhysicsNeMo(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "pressure PhysicsNeMo",
        injectors,
        producers,
        gas_injectors,
    )
    ax2 = f_3.add_subplot(4, 3, 2, projection="3d")
    Plot_PhysicsNeMo(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "pressure Numerical",
        injectors,
        producers,
        gas_injectors,
    )
    ax3 = f_3.add_subplot(4, 3, 3, projection="3d")
    Plot_PhysicsNeMo(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "pressure diff",
        injectors,
        producers,
        gas_injectors,
    )
    R2p, L2p = compute_metrics(look.ravel(), lookf.ravel())
    Accuracy_presure[kk, 0] = R2p
    Accuracy_presure[kk, 1] = L2p

    look = ((Swater[0, kk, :, :, :]) * active_mask)[:, :, ::-1]
    lookf = ((Swater_true[0, kk, :, :, :]) * active_mask)[:, :, ::-1]
    diff1 = ((abs(look - lookf)) * active_mask)[:, :, ::-1]
    ax1 = f_3.add_subplot(4, 3, 4, projection="3d")
    Plot_PhysicsNeMo(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "water PhysicsNeMo",
        injectors,
        producers,
        gas_injectors,
    )
    ax2 = f_3.add_subplot(4, 3, 5, projection="3d")
    Plot_PhysicsNeMo(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "water Numerical",
        injectors,
        producers,
        gas_injectors,
    )
    ax3 = f_3.add_subplot(4, 3, 6, projection="3d")
    Plot_PhysicsNeMo(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "water diff",
        injectors,
        producers,
        gas_injectors,
    )
    R2w, L2w = compute_metrics(look.ravel(), lookf.ravel())
    Accuracy_water[kk, 0] = R2w
    Accuracy_water[kk, 1] = L2w

    look = Soil[0, kk, :, :, :]
    look = (look)[:, :, ::-1]
    lookf = Soil_true[0, kk, :, :, :]
    lookf = (lookf)[:, :, ::-1]
    diff1 = (abs(look - lookf))[:, :, ::-1]
    ax1 = f_3.add_subplot(4, 3, 7, projection="3d")
    Plot_PhysicsNeMo(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "oil PhysicsNeMo",
        injectors,
        producers,
        gas_injectors,
    )
    ax2 = f_3.add_subplot(4, 3, 8, projection="3d")
    Plot_PhysicsNeMo(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "oil Numerical",
        injectors,
        producers,
        gas_injectors,
    )
    ax3 = f_3.add_subplot(4, 3, 9, projection="3d")
    Plot_PhysicsNeMo(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "oil diff",
        injectors,
        producers,
        gas_injectors,
    )
    R2o, L2o = compute_metrics(look.ravel(), lookf.ravel())
    Accuracy_oil[kk, 0] = R2o
    Accuracy_oil[kk, 1] = L2o
    look = ((Sgas[0, kk, :, :, :]) * active_mask)[:, :, ::-1]
    lookf = ((Sgas_true[0, kk, :, :, :]) * active_mask)[:, :, ::-1]
    diff1 = (abs(look - lookf))[:, :, ::-1]
    ax1 = f_3.add_subplot(4, 3, 10, projection="3d")
    Plot_PhysicsNeMo(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "gas PhysicsNeMo",
        injectors,
        producers,
        gas_injectors,
    )
    ax2 = f_3.add_subplot(4, 3, 11, projection="3d")
    Plot_PhysicsNeMo(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "gas Numerical",
        injectors,
        producers,
        gas_injectors,
    )
    ax3 = f_3.add_subplot(4, 3, 12, projection="3d")
    Plot_PhysicsNeMo(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "gas diff",
        injectors,
        producers,
        gas_injectors,
    )
    R2g, L2g = compute_metrics(look.ravel(), lookf.ravel())
    Accuracy_gas[kk, 0] = R2g
    Accuracy_gas[kk, 1] = L2g
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(current_time) + " days"
    plt.suptitle(tita, fontsize=16)
    # plt.savefig('Dynamic' + str(int(kk)))
    plt.savefig("Dynamic" + str(int(kk)))
    plt.clf()
    plt.close()
    return R2p, L2p, R2w, L2w, R2o, L2o, R2g, L2g
