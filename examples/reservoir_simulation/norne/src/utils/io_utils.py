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
I/O, system, and miscellaneous utilities shared across all sub-modules.

@Author : Clement Etienam
"""

# Standard Library
import subprocess
import logging
import warnings

# Third-party Libraries
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib import cm
from shutil import rmtree

# Third-party: optional at import time (torch may not always be present)
import torch
from cpuinfo import get_cpu_info

# Local Modules
from utils.logging_utils import setup_logging

# ---------------------------------------------------------------------------
# GPU / system check
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Check if NVIDIA GPU is available using native Python methods."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


# ---------------------------------------------------------------------------
# Environment initialisation
# ---------------------------------------------------------------------------


def initialize_environment() -> tuple[bool, logging.Logger]:
    """Initialize the environment and return GPU availability and logger."""
    logger = setup_logging("io_utils")

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


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------


def read_yaml(fname):
    """Read Yaml file into a dict of parameters"""
    logger = setup_logging(__name__)
    logger.info(f"Read simulation cfg from {fname}...")
    with open(fname) as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger = setup_logging(__name__)
            logger.error(exc)
        return data


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Plot utility
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Folder utility
# ---------------------------------------------------------------------------


def Remove_folder(N_ens, straa):
    """Delete ``N_ens`` numbered ensemble folders ``straa + str(jj)``.

    Parameters
    ----------
    N_ens : int
        Number of ensemble folders to remove.
    straa : str
        Common path prefix; folder names are ``straa + str(jj)``.

    Returns
    -------
    None
    """
    for jj in range(N_ens):
        folderr = straa + str(jj)
        rmtree(folderr)
