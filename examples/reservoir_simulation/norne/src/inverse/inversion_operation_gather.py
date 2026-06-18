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
@Author : Clement Etienam
"""

import os
import sys
import pickle
import logging
import gzip
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy

import torch
from hydra.utils import to_absolute_path
from sklearn.metrics import r2_score as r2

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from imresize import imresize
from inverse.inversion_operation_misc import (
    add_gnoise,
    NorneGeostat,
)


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


def Plot_2D(
    XX,
    YY,
    plt,
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
    """Render a 2D pcolormesh of a field averaged over Z-layers with well markers.

    Parameters
    ----------
    XX : np.ndarray
        2D meshgrid of X coordinates, shape (nx, ny).
    YY : np.ndarray
        2D meshgrid of Y coordinates, shape (nx, ny).
    plt : module
        Matplotlib pyplot module for the current axes context.
    nx : int
        Number of grid cells in X direction.
    ny : int
        Number of grid cells in Y direction.
    nz : int
        Number of grid cells in Z direction.
    Truee : np.ndarray
        Field array, shape (nx*ny*nz,) or (nx, ny, nz).
    N_injw : int
        Number of water injector wells.
    N_pr : int
        Number of producer wells.
    N_injg : int
        Number of gas injector wells.
    varii : str
        Variable identifier (e.g. 'perm', 'porosity', 'water PhysicsNeMo').
    injectors : list
        Water injector well coordinates and labels.
    producers : list
        Producer well coordinates and labels.
    gas_injectors : list
        Gas injector well coordinates and labels.

    Returns
    -------
    None
        Plots directly onto the current Matplotlib axes.
    """
    avg_2d = (
        np.mean(Truee, axis=2) if Truee.ndim == 3 else np.reshape(Truee, (nx, ny), "F")
    )
    maxii = max(avg_2d.ravel())
    minii = min(avg_2d.ravel())
    avg_2d[avg_2d == 0] = np.nan  # Convert zeros to NaNs
    plt.pcolormesh(XX.T, YY.T, avg_2d, cmap="jet")
    cbar = plt.colorbar()
    if varii == "perm":
        cbar.set_label("Log K(mD)", fontsize=11)
        plt.title("Permeability Field with well locations", fontsize=11, weight="bold")
    elif varii == "water PhysicsNeMo":
        cbar.set_label("water saturation", fontsize=11)
        plt.title("water saturation -PhysicsNeMo", fontsize=11, weight="bold")
    elif varii == "water FLOW":
        cbar.set_label("water saturation", fontsize=11)
        plt.title("water saturation - FLOW", fontsize=11, weight="bold")
    elif varii == "water diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("water saturation - (FLOW -PhysicsNeMo)", fontsize=11, weight="bold")
    elif varii == "oil PhysicsNeMo":
        cbar.set_label("Oil saturation", fontsize=11)
        plt.title("Oil saturation -PhysicsNeMo", fontsize=11, weight="bold")

    elif varii == "oil FLOW":
        cbar.set_label("Oil saturation", fontsize=11)
        plt.title("Oil saturation - Flow", fontsize=11, weight="bold")
    elif varii == "oil diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("oil saturation - (FLOW -PhysicsNeMo)", fontsize=11, weight="bold")
    elif varii == "gas PhysicsNeMo":
        cbar.set_label("Gas saturation", fontsize=11)
        plt.title("Gas saturation -PhysicsNeMo", fontsize=11, weight="bold")
    elif varii == "gas FLOW":
        cbar.set_label("Gas saturation", fontsize=11)
        plt.title("Gas saturation -FLOW", fontsize=11, weight="bold")
    elif varii == "gas diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("gas saturation - (FLOW -PhysicsNeMo)", fontsize=11, weight="bold")
    elif varii == "pressure PhysicsNeMo":
        cbar.set_label("pressure", fontsize=11)
        plt.title("Pressure -PhysicsNeMo", fontsize=11, weight="bold")
    elif varii == "pressure FLOW":
        cbar.set_label("pressure", fontsize=11)
        plt.title("Pressure -FLOW", fontsize=11, weight="bold")
    elif varii == "pressure diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("Pressure - (FLOW -PhysicsNeMo)", fontsize=11, weight="bold")
    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=11)
        plt.title("Porosity Field", fontsize=11, weight="bold")
    cbar.mappable.set_clim(minii, maxii)
    plt.ylabel("Y", fontsize=11)
    plt.xlabel("X", fontsize=11)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    Add_marker2(plt, XX, YY, injectors, producers, gas_injectors)


def Plot_2DD(
    XX,
    YY,
    plt,
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
    """Render a 2D pcolormesh of a reshaped field with colorbar and well markers.

    Parameters
    ----------
    XX : np.ndarray
        2D meshgrid of X coordinates, shape (nx, ny).
    YY : np.ndarray
        2D meshgrid of Y coordinates, shape (nx, ny).
    plt : module
        Matplotlib pyplot module for the current axes context.
    nx : int
        Number of grid cells in X direction.
    ny : int
        Number of grid cells in Y direction.
    nz : int
        Number of grid cells in Z direction.
    Truee : np.ndarray
        Field array, shape (nx*ny*nz,) or (nx, ny, nz).
    N_injw : int
        Number of water injector wells.
    N_pr : int
        Number of producer wells.
    N_injg : int
        Number of gas injector wells.
    varii : str
        Variable identifier controlling colorbar label.
    injectors : list
        Water injector well coordinates and labels.
    producers : list
        Producer well coordinates and labels.
    gas_injectors : list
        Gas injector well coordinates and labels.

    Returns
    -------
    None
        Plots directly onto the current Matplotlib axes.
    """
    if Truee.ndim == 3:
        Pressz = np.reshape(Truee, (nx, ny, nz), "F")
        avg_2d = np.mean(Pressz, axis=2)
    else:
        Pressz = np.reshape(Truee, (nx, ny), "F")
        avg_2d = Pressz
    maxii = max(Pressz.ravel())
    minii = min(Pressz.ravel())
    avg_2d[avg_2d == 0] = np.nan  # Convert zeros to NaNs
    plt.pcolormesh(XX.T, YY.T, avg_2d, cmap="jet")
    cbar = plt.colorbar()
    if varii == "perm":
        cbar.set_label("Log K(mD)", fontsize=11)
    elif varii == "water PhysicsNeMo":
        cbar.set_label("water saturation", fontsize=11)
    elif varii == "oil PhysicsNeMo":
        cbar.set_label("Oil saturation", fontsize=11)
    elif varii == "gas PhysicsNeMo":
        cbar.set_label("Gas saturation", fontsize=11)
    elif varii == "pressure PhysicsNeMo":
        cbar.set_label("pressure", fontsize=11)
    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=11)
    cbar.mappable.set_clim(minii, maxii)
    plt.ylabel("Y", fontsize=11)
    plt.xlabel("X", fontsize=11)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    Add_marker2(plt, XX, YY, injectors, producers, gas_injectors)


def Add_marker2(plt, XX, YY, injectors, producers, gas_injectors):
    """Overlay large well markers (size=200) on a 2D pcolormesh plot.

    Parameters
    ----------
    plt : module
        Matplotlib pyplot module for the current axes context.
    XX : np.ndarray
        2D meshgrid of X coordinates, shape (nx, ny).
    YY : np.ndarray
        2D meshgrid of Y coordinates, shape (nx, ny).
    injectors : list
        Water injector well entries; each entry provides (x, y, ..., label).
    producers : list
        Producer well entries; each entry provides (x, y, ..., label).
    gas_injectors : list
        Gas injector well entries; each entry provides (x, y, ..., label).

    Returns
    -------
    None
        Adds scatter markers and text annotations directly to the current axes.
    """
    n_inj = len(injectors)  # Number of water injectors
    n_prod = len(producers)  # Number of producers
    n_injg = len(gas_injectors)  # Number of gas injectors
    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=200,
            marker="v",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )
    for mm in range(n_injg):
        usethis = gas_injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=200,
            marker="v",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )
    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=200,
            marker="^",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )


def Add_marker3(plt, XX, YY, injectors, producers, gas_injectors):
    """Overlay small well markers (size=100) on a 2D pcolormesh plot.

    Parameters
    ----------
    plt : module
        Matplotlib pyplot module for the current axes context.
    XX : np.ndarray
        2D meshgrid of X coordinates, shape (nx, ny).
    YY : np.ndarray
        2D meshgrid of Y coordinates, shape (nx, ny).
    injectors : list
        Water injector well entries; each entry provides (x, y, ..., label).
    producers : list
        Producer well entries; each entry provides (x, y, ..., label).
    gas_injectors : list
        Gas injector well entries; each entry provides (x, y, ..., label).

    Returns
    -------
    None
        Adds scatter markers and text annotations directly to the current axes.
    """
    n_inj = len(injectors)  # Number of injectors
    n_prod = len(producers)  # Number of producers
    n_injg = len(gas_injectors)  # Number of gas injectors
    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=100,
            marker="v",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9,
        )
    for mm in range(n_injg):
        usethis = gas_injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=100,
            marker="v",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9,
        )
    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=100,
            marker="^",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9,
        )


def Plot_mean(
    permbest,
    permmean,
    iniperm,
    nx,
    ny,
    nz,
    Low_K,
    High_K,
    True_perm,
    active_cells_ensemble,
    injectors,
    producers,
    gas_injectors,
    N_injw,
    N_pr,
    N_injg,
):
    """Plot a 2x2 grid comparing MAP, best, initial, and true permeability fields.

    Parameters
    ----------
    permbest : np.ndarray
        Best ensemble permeability realisation, shape (nx*ny*nz, 1) or flat.
    permmean : np.ndarray
        MAP (mean) ensemble permeability, shape (nx*ny*nz, 1) or flat.
    iniperm : np.ndarray
        Initial permeability ensemble, shape (nx*ny*nz, 1) or flat.
    nx : int
        Number of grid cells in X direction.
    ny : int
        Number of grid cells in Y direction.
    nz : int
        Number of grid cells in Z direction.
    Low_K : float
        Lower colorbar clim bound for permeability.
    High_K : float
        Upper colorbar clim bound for permeability.
    True_perm : np.ndarray
        True permeability field, shape (nx*ny*nz, 1) or flat.
    active_cells_ensemble : np.ndarray
        Active cell mask array, shape (nx, ny, nz).
    injectors : list
        Water injector well coordinates and labels.
    producers : list
        Producer well coordinates and labels.
    gas_injectors : list
        Gas injector well coordinates and labels.
    N_injw : int
        Number of water injector wells.
    N_pr : int
        Number of producer wells.
    N_injg : int
        Number of gas injector wells.

    Returns
    -------
    None
        Saves 'Comparison.png' to disk.
    """
    Low_Ka = Low_K
    High_Ka = High_K
    permmean = np.mean(
        np.reshape(permmean, (nx, ny, nz), "F") * active_cells_ensemble, axis=2
    )
    permbest = np.mean(
        np.reshape(permbest, (nx, ny, nz), "F") * active_cells_ensemble, axis=2
    )
    iniperm = np.mean(
        np.reshape(iniperm, (nx, ny, nz), "F") * active_cells_ensemble, axis=2
    )
    True_perm = np.mean(
        np.reshape(True_perm, (nx, ny, nz), "F") * active_cells_ensemble, axis=2
    )
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.figure(figsize=(30, 30))
    plt.subplot(2, 2, 1)
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        permmean,
        N_injw,
        N_pr,
        N_injg,
        "perm",
        injectors,
        producers,
        gas_injectors,
    )
    plt.title(" MAP", fontsize=15)
    plt.clim(Low_Ka, High_Ka)
    plt.subplot(2, 2, 2)
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        permbest,
        N_injw,
        N_pr,
        N_injg,
        "perm",
        injectors,
        producers,
        gas_injectors,
    )
    plt.title(" Best", fontsize=15)
    plt.clim(Low_Ka, High_Ka)
    plt.subplot(2, 2, 3)
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        iniperm,
        N_injw,
        N_pr,
        N_injg,
        "perm",
        injectors,
        producers,
        gas_injectors,
    )
    plt.title(" initial", fontsize=15)
    plt.clim(Low_Ka, High_Ka)
    plt.subplot(2, 2, 4)
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        True_perm,
        N_injw,
        N_pr,
        N_injg,
        "perm",
        injectors,
        producers,
        gas_injectors,
    )
    plt.title(" True", fontsize=15)
    plt.clim(Low_Ka, High_Ka)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Permeability comparison", fontsize=25)
    plt.savefig("Comparison.png")
    plt.close()
    plt.clf()


def Plot_petrophysical(
    permmean,
    poroo,
    nx,
    ny,
    nz,
    Low_K,
    High_K,
    active_cells_ensemble,
    N_injw,
    N_pr,
    N_injg,
    injectors,
    producers,
    gas_injectors,
    Low_P,
    High_P,
):
    """Plot raw and NL-means-smoothed permeability and porosity side by side.

    Parameters
    ----------
    permmean : np.ndarray
        Mean ensemble permeability field, shape (nx*ny*nz,) or (nx, ny, nz).
    poroo : np.ndarray
        Mean ensemble porosity field, shape (nx*ny*nz,) or (nx, ny, nz).
    nx : int
        Number of grid cells in X direction.
    ny : int
        Number of grid cells in Y direction.
    nz : int
        Number of grid cells in Z direction.
    Low_K : float
        Lower colorbar clim bound for permeability.
    High_K : float
        Upper colorbar clim bound for permeability.
    active_cells_ensemble : np.ndarray
        Active cell mask array, shape (nx, ny, nz).
    N_injw : int
        Number of water injector wells.
    N_pr : int
        Number of producer wells.
    N_injg : int
        Number of gas injector wells.
    injectors : list
        Water injector well coordinates and labels.
    producers : list
        Producer well coordinates and labels.
    gas_injectors : list
        Gas injector well coordinates and labels.
    Low_P : float
        Lower colorbar clim bound for porosity.
    High_P : float
        Upper colorbar clim bound for porosity.

    Returns
    -------
    None
        Saves 'Petro_Recon.png' to disk.
    """
    Low_Ka = Low_K
    High_Ka = High_K
    permmean = np.reshape(permmean, (nx, ny, nz), "F")
    poroo = np.reshape(poroo, (nx, ny, nz), "F")
    from skimage.restoration import denoise_nl_means, estimate_sigma

    temp_K = permmean
    temp_phi = poroo
    timk = 5
    for _n in range(timk):
        sigma_est1 = np.mean(estimate_sigma(temp_K))
        sigma_est2 = np.mean(estimate_sigma(temp_phi))
        patch_kw = dict(
            patch_size=5,  # 5x5 patches
            patch_distance=6,
        )
        temp_K = denoise_nl_means(
            temp_K, h=0.8 * sigma_est1, fast_mode=True, **patch_kw
        )
        temp_phi = denoise_nl_means(
            temp_phi, h=0.8 * sigma_est2, fast_mode=True, **patch_kw
        )
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.figure(figsize=(30, 30))
    permmean = np.mean(
        np.reshape(permmean, (nx, ny, nz), "F") * active_cells_ensemble, axis=2
    )
    temp_K = np.mean(
        np.reshape(temp_K, (nx, ny, nz), "F") * active_cells_ensemble, axis=2
    )
    poroo = np.mean(
        np.reshape(poroo, (nx, ny, nz), "F") * active_cells_ensemble, axis=2
    )
    temp_phi = np.mean(
        np.reshape(temp_phi, (nx, ny, nz), "F") * active_cells_ensemble, axis=2
    )
    plt.subplot(2, 2, 1)
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        permmean,
        N_injw,
        N_pr,
        N_injg,
        "perm",
        injectors,
        producers,
        gas_injectors,
    )
    plt.title(" Permeability", fontsize=15)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers, gas_injectors)
    plt.subplot(2, 2, 2)
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        temp_K,
        N_injw,
        N_pr,
        N_injg,
        "perm",
        injectors,
        producers,
        gas_injectors,
    )
    plt.title("Smoothed - Permeability", fontsize=15)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers, gas_injectors)
    plt.subplot(2, 2, 3)
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        poroo,
        N_injw,
        N_pr,
        N_injg,
        "poro",
        injectors,
        producers,
        gas_injectors,
    )
    plt.title("Porosity", fontsize=15)
    plt.clim(Low_P, High_P)
    Add_marker2(plt, XX, YY, injectors, producers, gas_injectors)
    plt.subplot(2, 2, 4)
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        temp_phi,
        N_injw,
        N_pr,
        N_injg,
        "poro",
        injectors,
        producers,
        gas_injectors,
    )
    plt.title("Smoothed Porosity", fontsize=15)
    plt.clim(Low_P, High_P)
    Add_marker2(plt, XX, YY, injectors, producers, gas_injectors)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Petrophysical Reconstruction", fontsize=25)
    plt.savefig("Petro_Recon.png")
    plt.close()
    plt.clf()


def Getporosity_ensemble(ini_ensemble, machine_map, N_ens):
    """Map a permeability ensemble to porosity using a trained regressor.

    Parameters
    ----------
    ini_ensemble : np.ndarray
        Permeability ensemble, shape (n_cells, N_ens).
    machine_map : object
        Trained scikit-learn-compatible regressor with a `predict` method.
    N_ens : int
        Number of ensemble members.

    Returns
    -------
    np.ndarray
        Predicted porosity ensemble, shape (n_cells, N_ens).
    """
    ini_ensemblep = []
    for ja in range(N_ens):
        usek = np.reshape(ini_ensemble[:, ja], (-1, 1), "F")
        ini_ensemblep.append(usek)
    ini_ensemble = np.vstack(ini_ensemblep)
    ini_ensemblep = machine_map.predict(ini_ensemble)
    ini_ensemblee = np.split(ini_ensemblep, N_ens, axis=0)
    ini_ensemble = []
    for ky in range(N_ens):
        aa = ini_ensemblee[ky]
        aa = np.reshape(aa, (-1, 1), "F")
        ini_ensemble.append(aa)
    return np.hstack(ini_ensemble)


def plot_rsm_percentile(pertoutt, True_mat, timezz, N_pr, well_names, ope_f):
    """
    Plot P10/P50/P90/aREKI-best/aREKI-MAP/Base curves vs truth for one variable.
    Also computes and logs RMSE for each model and saves an elegant histogram.

    Parameters
    ----------
    pertoutt  : list of 6 arrays each (T, N_pr)
                [P10, P50, P90, arekibest, amean, base]
    True_mat  : ndarray (T, N_pr)  truth
    timezz    : ndarray (T,)       time in days
    N_pr      : int                number of wells
    well_names: list[str]          well names
    ope_f     : str                WOPR | WWPR | WBHP | WBHP_PROD | WWCT_CUM
    """

    P10 = pertoutt[0]
    P50 = pertoutt[1]
    P90 = pertoutt[2]
    arekibest = pertoutt[3]
    amean = pertoutt[4]
    base = pertoutt[5]

    # ── palette & style ───────────────────────────────────────────────────────
    CURVES = [
        (P10, "#2471A3", "-", 2.2, "P10 model"),
        (P50, "#1ABC9C", "-", 2.2, "P50 model"),
        (P90, "#27AE60", "-", 2.2, "P90 model"),
        (arekibest, "#2C3E50", "--", 2.0, "aREKI cum-best"),
        (amean, "#8E44AD", "--", 2.0, "aREKI cum-MAP"),
        (base, "#D68910", ":", 2.0, "Base case"),
    ]
    CLR_TRUE = "#C0392B"

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
            "legend.fontsize": 9,
            "legend.framealpha": 0.88,
            "legend.edgecolor": "#CCCCCC",
            "figure.dpi": 150,
        }
    )

    # ── per-variable config ───────────────────────────────────────────────────
    # FIX: WWPR was saving to WOPR.png — corrected
    vcfg = {
        "WOPR": {
            "ylabel": r"$Q_{oil}\ (bbl/day)$",
            "suptitle": r"Oil Production Rate — $Q_{oil}$",
            "fname": "WOPR.png",
            "fname2": "HISTOGRAM_WOPR.png",
        },
        "WWPR": {
            "ylabel": r"$Q_{water}\ (bbl/day)$",
            "suptitle": r"Water Production Rate — $Q_{water}$",
            "fname": "WWPR.png",
            "fname2": "HISTOGRAM_WWPR.png",
        },  # FIX
        "WGPR": {
            "ylabel": r"$Q_{gas}\ (scf/day)$",
            "suptitle": r"Gas Production Rate — $Q_{gas}$",
            "fname": "WGPR.png",
            "fname2": "HISTOGRAM_WGPR.png",
        },
    }

    if ope_f not in vcfg:
        raise ValueError(f"Unknown ope_f: {ope_f!r}. Choose from {list(vcfg.keys())}")

    vc = vcfg[ope_f]

    # ── R² helper ─────────────────────────────────────────────────────────────
    # def r2(y_true, y_pred):
    # ss_res = np.sum((y_true - y_pred) ** 2)
    # ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # return (1.0 - ss_res / (ss_tot + 1e-12)) * 100.0

    # ── RMSE helper ───────────────────────────────────────────────────────────
    def rmse(pred, true):
        """Compute normalised RMSE between `pred` and `true` arrays.

        Parameters
        ----------
        pred : np.ndarray
            Predicted values array (any shape).
        true : np.ndarray
            True reference values array (any shape).

        Returns
        -------
        float
            Root mean squared error divided by the number of elements.
        """
        p = np.reshape(pred, (-1, 1), "F")
        t = np.reshape(true, (-1, 1), "F")
        return float(((np.sum((p - t) ** 2)) ** 0.5) / t.shape[0])

    # ── multi-curve subplot figure ────────────────────────────────────────────
    ncols = min(N_pr, 5)
    nrows = int(np.ceil(N_pr / ncols))
    fig = plt.figure(figsize=(8.0 * ncols, 6.0 * nrows), constrained_layout=True)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)

    for k in range(N_pr):
        row, col = divmod(k, ncols)
        ax = fig.add_subplot(gs[row, col])

        # P10-P90 shaded envelope
        ax.fill_between(
            timezz,
            P10[:, k],
            P90[:, k],
            alpha=0.14,
            color="#AED6F1",
            label="_nolegend_",
        )

        # truth (thick, topmost)
        ax.plot(
            timezz, True_mat[:, k], color=CLR_TRUE, lw=2.8, label="True model", zorder=6
        )

        # six model curves
        for arr, clr, ls, lw, lbl in CURVES:
            ax.plot(
                timezz, arr[:, k], color=clr, lw=lw, linestyle=ls, label=lbl, zorder=4
            )

        # R² annotation for best model (arekibest)
        r2_best = r2(True_mat[:, k], arekibest[:, k]) * 100.0
        ax.annotate(
            f"aREKI best $R^2$ = {r2_best:.1f}%",
            xy=(0.97, 0.05),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#1A5276",
            bbox=dict(
                boxstyle="round,pad=0.38",
                facecolor="white",
                edgecolor="#AED6F1",
                linewidth=1.2,
                alpha=0.92,
            ),
        )

        ax.set_xlabel("Time  (days)", fontsize=13, fontweight="bold", labelpad=5)
        ax.set_ylabel(vc["ylabel"], fontsize=13, fontweight="bold", labelpad=5)
        ax.set_title(well_names[k], fontsize=14, fontweight="bold", pad=8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(which="both", length=4)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, color="#AAAAAA")
        ax.spines[["top", "right"]].set_visible(False)

        leg = ax.legend(
            loc="upper right",
            frameon=True,
            handlelength=2.0,
            borderpad=0.6,
            ncol=2,
            fontsize=8.5,
        )
        for text in leg.get_texts():
            text.set_fontweight("bold")

    for extra in range(N_pr, nrows * ncols):
        row, col = divmod(extra, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    fig.suptitle(vc["suptitle"], fontsize=16, fontweight="bold", y=1.01)
    plt.savefig(
        vc["fname"], bbox_inches="tight", dpi=200, facecolor="white", edgecolor="none"
    )
    plt.clf()
    plt.close()
    print(f"  Saved {vc['fname']}")

    # ── RMSE computation & logging ────────────────────────────────────────────
    True_data = True_mat

    cc10 = rmse(P10, True_data)
    cc50 = rmse(P50, True_data)
    cc90 = rmse(P90, True_data)
    cc99 = rmse(arekibest, True_data)
    ccmean = rmse(amean, True_data)
    ccbase = rmse(base, True_data)

    logger.info("=" * 58)
    logger.info(f"  RMSE — {ope_f}")
    logger.info("-" * 58)
    logger.info(f"  P10            : {cc10:.6f}")
    logger.info(f"  P50            : {cc50:.6f}")
    logger.info(f"  P90            : {cc90:.6f}")
    logger.info(f"  aREKI cum-best : {cc99:.6f}")
    logger.info(f"  aREKI cum-MAP  : {ccmean:.6f}")
    logger.info(f"  Base case      : {ccbase:.6f}")

    values = [cc10, cc50, cc90, cc99, ccmean, ccbase]
    model_names = ["P10", "P50", "P90", "aREKI\nbest", "aREKI\nMAP", "Base\ncase"]
    best_idx = int(np.argmin(values))
    best_model = model_names[best_idx].replace("\n", " ")
    logger.info("-" * 58)
    logger.info(f"  Min RMSE = {values[best_idx]:.6f}  →  Best: {best_model}")
    logger.info("=" * 58)

    # ── elegant histogram ─────────────────────────────────────────────────────
    bar_colors = ["#2471A3", "#1ABC9C", "#27AE60", "#2C3E50", "#8E44AD", "#D68910"]

    fig2, ax2 = plt.subplots(figsize=(11, 6), constrained_layout=True)
    fig2.patch.set_facecolor("white")

    bars = ax2.bar(
        model_names,
        values,
        color=bar_colors,
        width=0.52,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    # value labels on each bar
    y_max = max(values)
    for bar, v in zip(bars, values, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.015,
            f"{v:.5f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#2C3E50",
        )

    # star + "best" label on the lowest bar
    best_bar = bars[best_idx]
    ax2.text(
        best_bar.get_x() + best_bar.get_width() / 2,
        best_bar.get_height() + y_max * 0.065,
        "★  best",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#1E8449",
    )

    ax2.set_xlabel("Reservoir model", fontsize=14, fontweight="bold", labelpad=8)
    ax2.set_ylabel("RMSE", fontsize=14, fontweight="bold", labelpad=8)
    ax2.set_title(
        f"RMSE comparison — {ope_f}\nBest: {best_model}  (RMSE = {values[best_idx]:.5f})",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax2.set_ylim(0, y_max * 1.30)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    ax2.grid(
        axis="y", linestyle="--", linewidth=0.6, alpha=0.5, color="#AAAAAA", zorder=0
    )
    ax2.spines[["top", "right"]].set_visible(False)

    # colour legend
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(
            facecolor=bar_colors[i],
            edgecolor="white",
            label=model_names[i].replace("\n", " "),
        )
        for i in range(len(model_names))
    ]
    leg2 = ax2.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=10,
        framealpha=0.88,
        edgecolor="#CCCCCC",
        title="Models",
        title_fontsize=10,
    )
    for text in leg2.get_texts():
        text.set_fontweight("bold")

    plt.savefig(
        vc["fname2"], bbox_inches="tight", dpi=200, facecolor="white", edgecolor="none"
    )
    plt.clf()
    plt.close()
    print(f"  Saved Histogram  |  Best: {best_model}  RMSE = {values[best_idx]:.5f}")


def plot_rsm_percentile_model(pertoutt, True_mat, timezz, N_pr, well_names, ope_f):
    """
    Plot history-matched model vs true model for a single variable type.

    Parameters
    ----------
    pertoutt  : ndarray (T, N_pr)  history-matched predictions
    True_mat  : ndarray (T, N_pr)  true model
    timezz    : ndarray (T,)       time in days
    N_pr      : int                number of wells
    well_names: list[str]          well names
    ope_f     : str                WOPR | WWPR | WBHP | WBHP_PROD | WWCT_CUM
    """

    P10 = pertoutt

    # ── palette & style ───────────────────────────────────────────────────────
    CLR_TRUE = "#C0392B"  # deep red      — true model
    CLR_PRED = "#1A5276"  # dark navy     — history matched
    CLR_FILL = "#AED6F1"  # light blue fill

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
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
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
            "fname": "WOPR_MODEL.png",
        },
        "WWPR": {
            "ylabel": r"$Q_{water}\ (bbl/day)$",
            "suptitle": r"Water Production Rate — $Q_{water}$",
            "fname": "WWPR_MODEL.png",
        },
        "WGPR": {
            "ylabel": r"$Q_{gas}\ (scf/day)$",
            "suptitle": r"Gas Production Rate — $Q_{gas}$",
            "fname": "WGPR_MODEL.png",
        },
    }

    if ope_f not in vcfg:
        raise ValueError(f"Unknown ope_f: {ope_f!r}. Choose from {list(vcfg.keys())}")

    vc = vcfg[ope_f]

    # ── R² helper ─────────────────────────────────────────────────────────────
    # def r2(y_true, y_pred):
    # ss_res = np.sum((y_true - y_pred) ** 2)
    # ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # return (1.0 - ss_res / (ss_tot + 1e-12)) * 100.0

    # ── layout ────────────────────────────────────────────────────────────────
    ncols = min(N_pr, 5)
    nrows = int(np.ceil(N_pr / ncols))
    fig = plt.figure(figsize=(7.5 * ncols, 5.8 * nrows), constrained_layout=True)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)

    r2_vals = []

    for k in range(N_pr):
        row, col = divmod(k, ncols)
        ax = fig.add_subplot(gs[row, col])

        y_true = True_mat[:, k]
        y_pred = P10[:, k]
        r2_val = r2(y_true, y_pred) * 100.0
        r2_vals.append(r2_val)

        # fill between curves
        ax.fill_between(
            timezz, y_true, y_pred, alpha=0.15, color=CLR_FILL, label="_nolegend_"
        )

        # curves
        ax.plot(timezz, y_true, color=CLR_TRUE, lw=2.4, label="True model", zorder=4)
        ax.plot(
            timezz,
            y_pred,
            color=CLR_PRED,
            lw=2.2,
            linestyle="--",
            label="History matched model",
            zorder=5,
        )

        # R² annotation
        ax.annotate(
            f"$R^2$ = {r2_val:.1f}%",
            xy=(0.97, 0.05),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#1A5276",
            bbox=dict(
                boxstyle="round,pad=0.38",
                facecolor="white",
                edgecolor="#AED6F1",
                linewidth=1.2,
                alpha=0.92,
            ),
        )

        # axes styling
        ax.set_xlabel("Time  (days)", fontsize=13, fontweight="bold", labelpad=6)
        ax.set_ylabel(vc["ylabel"], fontsize=13, fontweight="bold", labelpad=6)
        ax.set_title(well_names[k], fontsize=14, fontweight="bold", pad=8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(which="both", length=4)
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.45, color="#AAAAAA")
        ax.spines[["top", "right"]].set_visible(False)

        leg = ax.legend(
            loc="upper right", frameon=True, handlelength=2.0, borderpad=0.65
        )
        for text in leg.get_texts():
            text.set_fontweight("bold")

    # hide unused subplots
    for extra in range(N_pr, nrows * ncols):
        row, col = divmod(extra, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    # ── global title with mean R² ─────────────────────────────────────────────
    mean_r2 = float(np.mean(r2_vals))
    fig.suptitle(
        f"{vc['suptitle']}\n"
        r"Mean $R^2$ = " + f"{mean_r2:.1f}%",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    plt.savefig(
        vc["fname"], bbox_inches="tight", dpi=200, facecolor="white", edgecolor="none"
    )
    plt.clf()
    plt.close()
    print(f"  Saved {vc['fname']}  |  Mean R² = {mean_r2:.2f}%")


def plot_rsm_singleT(True_mat, timezz, N_pr, well_names, ope_f):
    """
    Plot a single numerical history curve per well for one variable type.

    Parameters
    ----------
    True_mat  : ndarray (1, T, N_pr) or (T, N_pr)  numerical truth
    timezz    : ndarray (T,)                         time in days
    N_pr      : int                                  number of wells
    well_names: list[str]                            well names
    ope_f     : str                                  WOPR | WWPR | WBHP | WBHP_PROD | WWCT_CUM
    """

    # handle both (1, T, N) and (T, N) input shapes
    if True_mat.ndim == 3:
        True_mat = True_mat[0]

    # ── style ─────────────────────────────────────────────────────────────────
    CLR = "#C0392B"  # deep red

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
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.framealpha": 0.88,
            "legend.edgecolor": "#CCCCCC",
            "figure.dpi": 150,
        }
    )

    # ── variable config ───────────────────────────────────────────────────────
    vcfg = {
        "WOPR": {
            "ylabel": r"$Q_{oil}\ (bbl/day)$",
            "suptitle": r"Oil Production Rate — $Q_{oil}$",
            "fname": "WOPR_HISTORY.png",
        },
        "WWPR": {
            "ylabel": r"$Q_{water}\ (bbl/day)$",
            "suptitle": r"Water Production Rate — $Q_{water}$",
            "fname": "WWPR_HISTORY.png",
        },
        "WGPR": {
            "ylabel": r"$Q_{gas}\ (scf/day)$",
            "suptitle": r"Gas Production Rate — $Q_{gas}$",
            "fname": "WGPR_HISTORY.png",
        },
    }

    if ope_f not in vcfg:
        raise ValueError(f"Unknown ope_f: {ope_f!r}. Choose from {list(vcfg.keys())}")

    vc = vcfg[ope_f]

    # ── layout — dynamic grid adapts to any N_pr ──────────────────────────────
    ncols = min(N_pr, 5)
    nrows = int(np.ceil(N_pr / ncols))
    fig = plt.figure(figsize=(7.5 * ncols, 5.8 * nrows), constrained_layout=True)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)

    for k in range(N_pr):
        row, col = divmod(k, ncols)
        ax = fig.add_subplot(gs[row, col])

        # area fill under curve for visual weight
        ax.fill_between(
            timezz, 0, True_mat[:, k], alpha=0.12, color=CLR, label="_nolegend_"
        )

        ax.plot(
            timezz,
            True_mat[:, k],
            color=CLR,
            lw=2.4,
            label="Numerical (Flow)",
            zorder=4,
        )

        ax.set_xlabel("Time  (days)", fontsize=13, fontweight="bold", labelpad=6)
        ax.set_ylabel(vc["ylabel"], fontsize=13, fontweight="bold", labelpad=6)
        ax.set_title(well_names[k], fontsize=14, fontweight="bold", pad=8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(which="both", length=4)
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.45, color="#AAAAAA")
        ax.spines[["top", "right"]].set_visible(False)

        leg = ax.legend(
            loc="upper right", frameon=True, handlelength=2.0, borderpad=0.65
        )
        for text in leg.get_texts():
            text.set_fontweight("bold")

    # hide any unused subplot slots
    for extra in range(N_pr, nrows * ncols):
        row, col = divmod(extra, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    fig.suptitle(vc["suptitle"], fontsize=16, fontweight="bold", y=1.01)

    plt.savefig(
        vc["fname"], bbox_inches="tight", dpi=200, facecolor="white", edgecolor="none"
    )
    plt.clf()
    plt.close()
    # print(f"  Saved {vc['fname']}")


def plot_rsm_single(True_mat, timezz, N_pr, well_names, ope_f):
    """
    Plot a single numerical curve per well for one variable type.

    Parameters
    ----------
    True_mat  : ndarray (1, T, N_pr) or (T, N_pr)  numerical truth
    timezz    : ndarray (T,)                         time in days
    N_pr      : int                                  number of wells
    well_names: list[str]                            well names
    ope_f     : str                                  WOPR | WWPR | WBHP | WBHP_PROD | WWCT_CUM
    """

    # handle both (1, T, N) and (T, N) input shapes
    if True_mat.ndim == 3:
        True_mat = True_mat[0]

    # ── style ─────────────────────────────────────────────────────────────────
    CLR = "#C0392B"  # deep red

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
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.framealpha": 0.88,
            "legend.edgecolor": "#CCCCCC",
            "figure.dpi": 150,
        }
    )

    # ── config per variable ───────────────────────────────────────────────────
    # FIX: WWPR was saving to WOPR_SINGLE.png — corrected fname
    vcfg = {
        "WOPR": {
            "ylabel": r"$Q_{oil}\ (bbl/day)$",
            "suptitle": r"Oil Production Rate — $Q_{oil}$",
            "fname": "WOPR_SINGLE.png",
        },
        "WWPR": {
            "ylabel": r"$Q_{water}\ (bbl/day)$",
            "suptitle": r"Water Production Rate — $Q_{water}$",
            "fname": "WWPR_SINGLE.png",
        },
        "WGPR": {
            "ylabel": r"$Q_{gas}\ (scf/day)$",
            "suptitle": r"Gas Production Rate — $Q_{gas}$",
            "fname": "WGPR_SINGLE.png",
        },
    }
    # vc = vcfg.get(ope_f, vcfg["WWCT_CUM"])
    vc = vcfg[ope_f]

    # ── layout ────────────────────────────────────────────────────────────────
    ncols = min(N_pr, 5)
    nrows = int(np.ceil(N_pr / ncols))
    fig = plt.figure(figsize=(7.5 * ncols, 5.8 * nrows), constrained_layout=True)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.40, wspace=0.32)

    for k in range(N_pr):
        row, col = divmod(k, ncols)
        ax = fig.add_subplot(gs[row, col])

        ax.fill_between(
            timezz, 0, True_mat[:, k], alpha=0.12, color=CLR, label="_nolegend_"
        )
        ax.plot(
            timezz,
            True_mat[:, k],
            color=CLR,
            lw=2.4,
            label="Numerical (Flow)",
            zorder=4,
        )

        ax.set_xlabel("Time  (days)", fontsize=13, fontweight="bold", labelpad=6)
        ax.set_ylabel(vc["ylabel"], fontsize=13, fontweight="bold", labelpad=6)
        ax.set_title(well_names[k], fontsize=14, fontweight="bold", pad=8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(which="both", length=4)
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.45, color="#AAAAAA")
        ax.spines[["top", "right"]].set_visible(False)

        leg = ax.legend(
            loc="upper right", frameon=True, handlelength=2.0, borderpad=0.65
        )
        for text in leg.get_texts():
            text.set_fontweight("bold")

    # hide unused subplots
    for extra in range(N_pr, nrows * ncols):
        row, col = divmod(extra, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    fig.suptitle(vc["suptitle"], fontsize=16, fontweight="bold", y=1.01)

    plt.savefig(
        vc["fname"], bbox_inches="tight", dpi=200, facecolor="white", edgecolor="none"
    )
    plt.clf()
    plt.close()
    print(f"  Saved {vc['fname']}")


def initialize_arrays(Nt, Ne, num_variables):
    """Allocate a zero-filled data cube for ensemble time-series storage.

    Parameters
    ----------
    Nt : int
        Number of time steps.
    Ne : int
        Number of ensemble members.
    num_variables : int
        Number of variables (e.g. wells or outputs).

    Returns
    -------
    np.ndarray
        Zero array of shape (Nt, Ne, num_variables).
    """
    return np.zeros((Nt, Ne, num_variables))


def plot_variable(timezz, pred_data, true_data, Ne, title, ylabel, ax, vline_x=None):
    """
    Plot Ne ensemble realisations plus the true model curve on a single axis.

    Parameters
    ----------
    timezz    : (T,)       time vector in days
    pred_data : (T, Ne)    ensemble predictions
    true_data : (T,)       truth / reference curve
    Ne        : int        number of ensemble members
    title     : str        subplot title
    ylabel    : str        y-axis label
    ax        : Axes       matplotlib axes to draw on
    vline_x   : float|None optional vertical dashed line (e.g. end of history)
    """
    from matplotlib.ticker import MaxNLocator

    CLR_ENS = "#5DADE2"  # light steel blue — ensemble
    CLR_TRUE = "#C0392B"  # deep red         — truth
    CLR_VLIN = "#2C3E50"  # dark slate       — vertical marker

    # ensemble realisations (thin, semi-transparent)
    for j in range(Ne):
        ax.plot(
            timezz,
            pred_data[:, j],
            color=CLR_ENS,
            lw=1.2,
            alpha=0.55,
            label="Realisations" if j == 0 else "_nolegend_",
            zorder=2,
        )

    # ensemble envelope (P10-P90 shading)
    p10 = np.percentile(pred_data, 10, axis=1)
    p90 = np.percentile(pred_data, 90, axis=1)
    ax.fill_between(
        timezz, p10, p90, alpha=0.18, color=CLR_ENS, label="_nolegend_", zorder=1
    )

    # truth curve
    ax.plot(timezz, true_data, color=CLR_TRUE, lw=2.6, label="True model", zorder=4)

    # optional vertical line
    if vline_x is not None:
        ax.axvline(
            x=vline_x,
            color=CLR_VLIN,
            linestyle="--",
            linewidth=1.4,
            alpha=0.75,
            label="_nolegend_",
        )

    # axes styling
    ax.set_xlabel("Time  (days)", fontsize=13, fontweight="bold", labelpad=5)
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", labelpad=5)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(which="both", length=4, labelsize=11)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, color="#AAAAAA")
    ax.spines[["top", "right"]].set_visible(False)

    # deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles, strict=False))
    leg = ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        frameon=True,
        handlelength=2.0,
        borderpad=0.65,
        fontsize=10,
        framealpha=0.88,
        edgecolor="#CCCCCC",
    )
    for text in leg.get_texts():
        text.set_fontweight("bold")


def plot_rsm_inner(
    timezz,
    pred_matrix,
    true_mat,
    Ne,
    variable_names,
    ylabels,
    file_name,
    Namesz,
    suptitle="",
    vline_x=None,
):
    """
    Build a multi-panel figure with one subplot per well/variable.

    Parameters
    ----------
    timezz         : (T,)
    pred_matrix    : list of Ne arrays, each (T, n_vars)
    true_mat       : (T, n_vars)
    Ne             : int
    variable_names : list[str]   subplot titles (well names)
    ylabels        : list[str]   y-axis labels per variable
    file_name      : str         output filename stem
    Namesz         : str         appended to filename
    suptitle       : str         figure-level super title
    vline_x        : float|None  optional vertical dashed line
    """

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
            "figure.dpi": 150,
        }
    )

    n_vars = len(variable_names)
    Nt = pred_matrix[0].shape[0]

    # build data cube (T, Ne, n_vars)
    data_matrix = initialize_arrays(Nt, Ne, n_vars)
    for i in range(Ne):
        for vi in range(n_vars):
            data_matrix[:, i, vi] = pred_matrix[i][:, vi]

    ncols = min(n_vars, 5)
    nrows = int(np.ceil(n_vars / ncols))
    fig = plt.figure(figsize=(7.5 * ncols, 5.8 * nrows), constrained_layout=True)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)

    for vi, name in enumerate(variable_names):
        row, col = divmod(vi, ncols)
        ax = fig.add_subplot(gs[row, col])
        plot_variable(
            timezz,
            data_matrix[:, :, vi],
            true_mat[:, vi],
            Ne,
            name,
            ylabels[vi],
            ax,
            vline_x=vline_x,
        )

    # hide unused subplots
    for extra in range(n_vars, nrows * ncols):
        row, col = divmod(extra, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.01)

    fname = f"{file_name}_{Namesz}.png"
    plt.savefig(
        fname, bbox_inches="tight", dpi=200, facecolor="white", edgecolor="none"
    )
    plt.clf()
    plt.close()
    # print(f"  Saved {fname}")


def plot_rsm(
    predMatrix, True_mat, Namesz, Ne, timezz, N_pr, well_names, ope_f, vline_x=None
):
    """
    Top-level dispatcher — call for each well-response variable.

    Parameters
    ----------
    predMatrix : list of Ne arrays each (T, n_vars)
    True_mat   : (T, n_vars)
    Namesz     : str    label appended to output filename
    Ne         : int    number of ensemble members
    timezz     : (T,)  time in days
    N_pr       : int   number of producers (or injectors for WBHP)
    well_names : list[str]
    ope_f      : str   WOPR | WWPR | WBHP | WBHP_PROD | WWCT_CUM
    vline_x    : float optional vertical dashed line (end of history)
    """
    variable_names = well_names

    # ── variable config ───────────────────────────────────────────────────────
    vcfg = {
        "WOPR": {
            "ylabels": [r"$Q_{oil}\ (bbl/day)$"] * N_pr,
            "fname": "WOPR_OIL",
            "suptitle": r"Oil Production Rate — $Q_{oil}$",
        },
        "WWPR": {
            "ylabels": [r"$Q_{water}\ (bbl/day)$"] * N_pr,
            "fname": "WWPR_WATER",
            "suptitle": r"Water Production Rate — $Q_{water}$",
        },
        "WGPR": {
            "ylabels": [r"$Q_{gas}\ (scf/day)$"] * N_pr,
            "fname": "WGPR_WATER",
            "suptitle": r"Gas Production Rate — $Q_{gas}$",
        },
    }

    if ope_f not in vcfg:
        raise ValueError(f"Unknown ope_f: {ope_f!r}. Choose from {list(vcfg.keys())}")

    vc = vcfg[ope_f]
    plot_rsm_inner(
        timezz,
        predMatrix,
        True_mat,
        Ne,
        variable_names,
        vc["ylabels"],
        vc["fname"],
        Namesz,
        suptitle=vc["suptitle"],
        vline_x=vline_x,
    )


def De_correlate_ensemble(nx, ny, nz, Ne, High_K, Low_K, Yet):
    """Decorrelate a GAN ensemble via truncated SVD and return Ne realisations.

    Parameters
    ----------
    nx : int
        Number of grid cells in X direction.
    ny : int
        Number of grid cells in Y direction.
    nz : int
        Number of grid cells in Z direction.
    Ne : int
        Number of desired ensemble members to return.
    High_K : float
        Upper permeability bound applied after decorrelation.
    Low_K : float
        Lower permeability bound applied after decorrelation.
    Yet : int
        If 0, convert tensors to NumPy; otherwise keep as tensors.

    Returns
    -------
    np.ndarray
        Decorrelated permeability ensemble, shape (n_cells, Ne).
    """
    filename = "../data/Ganensemble.pkl.gz"

    with gzip.open(filename, "rb") as f2:
        try:
            mat1 = pickle.load(f2)
        except (pickle.PickleError, EOFError, FileNotFoundError) as e:
            logger.error(f"Error loading pickle file: {e}")
            raise
    mat = mat1["permeability"]
    ini_ensemblef = mat
    ini_ensemblef = torch.as_tensor(ini_ensemblef, dtype=torch.float32)
    beta = int(
        torch.ceil(
            torch.tensor(ini_ensemblef.shape[0] / Ne, dtype=torch.float32)
        ).item()
    )
    U, S1, V = torch.linalg.svd(ini_ensemblef, full_matrices=True)
    v = V[:, :Ne]
    U1 = U.T
    u = U1[:, :Ne]
    S11 = S1[:Ne]
    s = S11[:]
    S = (1 / ((beta) ** (0.5))) * s
    X = (v * S).dot(u.T)
    if Yet == 0:
        X = X.detach().cpu().numpy()
        ini_ensemblef = ini_ensemblef.detach().cpu().numpy()
    else:
        pass
    X[Low_K >= X] = Low_K
    X[High_K <= X] = High_K
    return X[:, :Ne]


def whiten(X, method="zca"):
    """Whiten data matrix X using the specified whitening method.

    Parameters
    ----------
    X : np.ndarray
        Input data array; rows are samples, columns are features.
    method : str, optional
        Whitening method: 'zca', 'pca', 'cholesky', 'zca_cor', or 'pca_cor'.

    Returns
    -------
    np.ndarray
        Whitened data with the same number of samples as X.

    Raises
    ------
    Exception
        If an unsupported whitening method is specified.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None
    if method in ["zca", "pca", "cholesky"]:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == "zca":
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method == "pca":
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == "cholesky":
            W = np.linalg.cholesky(
                np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))
            ).T
    elif method in ["zca_cor", "pca_cor"]:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(
            np.dot(
                np.linalg.solve(V_sqrt, Sigma),
                np.linalg.solve(V_sqrt, np.eye(V_sqrt.shape[0])),
            ),
            np.linalg.solve(V_sqrt, np.eye(V_sqrt.shape[0])),
        )
        G, Theta, _ = np.linalg.svd(P)
        if method == "zca_cor":
            W = np.dot(
                np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)),
                np.linalg.solve(V_sqrt, np.eye(V_sqrt.shape[0])),
            )
        elif method == "pca_cor":
            W = np.dot(
                np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T),
                np.linalg.solve(V_sqrt, np.eye(V_sqrt.shape[0])),
            )
    else:
        raise Exception("Whitening method not found.")
    return np.dot(X_centered, W.T)


def write_rsm(data, Time, Name, well_names, N_pr):
    """Write production rates and time to a multi-header Excel workbook.

    Parameters
    ----------
    data : np.ndarray
        Production rate data array with shape (n_timesteps, n_columns).
    Time : np.ndarray
        Time vector in days, shape (n_timesteps,).
    Name : str
        Output file name stem (without extension).
    well_names : list
        List of well name strings used as column headers.
    N_pr : int
        Number of producer wells; controls column width in the sheet.

    Returns
    -------
    None
        Writes '<Name>.xlsx' to the current working directory.
    """
    groups = ["WOPR(bbl/day)", "WWPR(bbl/day)", "WGPR(scf/day)"]
    columns = well_names  # ['L1', 'L2', 'L3', 'LU1', 'LU2',
    headers = pd.MultiIndex.from_product([groups, columns])
    df = pd.DataFrame(data, columns=headers)
    df.insert(0, "Time(days)", Time)
    with pd.ExcelWriter(Name + ".xlsx", engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Sheet1")
        writer.sheets["Sheet1"] = worksheet
        header_format = workbook.add_format({"bold": True, "align": "center"})
        worksheet.write(0, 0, "Time(days)", header_format)
        col = 1
        for sub_col in columns * len(groups):
            worksheet.write(1, col, sub_col)
            col += 1
        col = 1
        for group in groups:
            end_col = col + len(columns) - 1
            worksheet.merge_range(0, col, 0, end_col, group, header_format)
            col = end_col + 1
        for row_num, row_data in enumerate(df.values):
            worksheet.write_row(row_num + 2, 0, row_data)
        time_data = np.arange(1, 101)
        for row_num, time_val in enumerate(time_data):
            worksheet.write(row_num + 2, 0, time_val)
        worksheet.set_column(0, 0, 12)  # 'Time(days)' column
        worksheet.set_column(1, col, N_pr)  # Data columns


def to_numpy(x):
    """Convert a torch.Tensor to a NumPy array, or return x unchanged.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Input value to convert.

    Returns
    -------
    np.ndarray
        Detached CPU NumPy array, or original value if not a Tensor.
    """
    if isinstance(x, torch.Tensor):
        return (
            x.detach().cpu().numpy()
        )  # Ensure it's detached, moved to CPU, and converted
    return x  # If it's already a NumPy array or another type, return as is


def Plot_Histogram_now(N, E12, E11, mean_cost, best_cost, oldfolder):
    """
    Plot initial/final RMSE per realisation and cost evolution over iterations.

    Parameters
    ----------
    N         : int            number of ensemble realisations
    E11       : array (N,)     initial RMSE per realisation  (pre-assimilation)
    E12       : array (N,)     final   RMSE per realisation  (post-assimilation)
    mean_cost : list/array     mean ensemble cost per iteration
    best_cost : list/array     best ensemble cost per iteration
    oldfolder : str            working directory to return to after saving
    """
    N = int(to_numpy(N))
    E11 = to_numpy(E11).reshape(N)
    E12 = to_numpy(E12).reshape(N)
    mean_cost = np.vstack(to_numpy(mean_cost)).reshape(-1)
    best_cost = np.vstack(to_numpy(best_cost)).reshape(-1)

    reali = np.arange(1, N + 1)
    timezz = np.arange(1, mean_cost.shape[0] + 1)

    # ── colours ───────────────────────────────────────────────────────────────
    CLR_INIT = "#5DADE2"
    CLR_FINAL = "#1ABC9C"
    CLR_SCATTER = "#2C3E50"
    CLR_MEAN = "#27AE60"
    CLR_BEST = "#2471A3"
    CLR_IMPROV = "#E74C3C"

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
            "legend.fontsize": 11,
            "legend.framealpha": 0.88,
            "legend.edgecolor": "#CCCCCC",
            "figure.dpi": 150,
        }
    )

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── helper: bar + scatter panel — independent y-axis per panel ───────────
    def bar_panel(ax, values, color, title):
        """Each panel scales to its own data — no shared ymax."""
        ax.bar(
            reali,
            values,
            color=color,
            width=0.75,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            alpha=0.88,
        )
        ax.scatter(reali, values, s=20, color=CLR_SCATTER, zorder=5)

        mean_val = np.mean(values)
        ax.axhline(
            mean_val,
            color=CLR_SCATTER,
            linestyle="--",
            linewidth=1.4,
            alpha=0.75,
            label=f"Mean = {mean_val:.4f}",
        )

        # annotate mean value on the line
        ax.text(
            N * 0.02,
            mean_val * 1.03,
            f"μ = {mean_val:.4f}",
            fontsize=10,
            fontweight="bold",
            color=CLR_SCATTER,
            va="bottom",
        )

        ax.set_xlabel("Realisation", fontsize=13, fontweight="bold", labelpad=6)
        ax.set_ylabel("RMSE", fontsize=13, fontweight="bold", labelpad=6)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlim(0.5, N + 0.5)
        ax.set_ylim(bottom=0, top=values.max() * 1.15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(which="both", length=4, labelsize=11)
        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.5,
            alpha=0.45,
            color="#AAAAAA",
            zorder=0,
        )
        ax.spines[["top", "right"]].set_visible(False)

        leg = ax.legend(loc="upper right", frameon=True, borderpad=0.65, fontsize=11)
        for t in leg.get_texts():
            t.set_fontweight("bold")

    # ── Panel 1: initial RMSE ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bar_panel(ax1, E11, CLR_INIT, "Initial cost function  (pre-assimilation)")

    # ── Panel 2: final RMSE ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bar_panel(ax2, E12, CLR_FINAL, "Final cost function  (post-assimilation)")

    # improvement annotation
    mean_init = np.mean(E11)
    mean_final = np.mean(E12)
    pct_improv = (mean_init - mean_final) / (mean_init + 1e-12) * 100.0
    sign = "↓" if pct_improv > 0 else "↑"
    ax2.annotate(
        f"{sign} {abs(pct_improv):.1f}% mean {'improvement' if pct_improv > 0 else 'degradation'}",
        xy=(0.50, 0.97),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=CLR_IMPROV if pct_improv > 0 else "#8E44AD",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=CLR_IMPROV,
            linewidth=1.2,
            alpha=0.92,
        ),
    )

    # ── Panel 3: cost evolution ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])

    ax3.fill_between(
        timezz, mean_cost, best_cost, alpha=0.15, color=CLR_MEAN, label="_nolegend_"
    )
    ax3.plot(
        timezz,
        mean_cost,
        color=CLR_MEAN,
        lw=2.6,
        marker="o",
        markersize=5,
        label="Mean ensemble cost",
        zorder=4,
    )
    ax3.plot(
        timezz,
        best_cost,
        color=CLR_BEST,
        lw=2.6,
        marker="s",
        markersize=5,
        linestyle="--",
        label="Best ensemble cost",
        zorder=5,
    )

    # annotate every point with its value
    for i, (mc, bc) in enumerate(zip(mean_cost, best_cost, strict=False)):
        ax3.text(
            timezz[i],
            mc + 0.02 * mean_cost.max(),
            f"{mc:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=CLR_MEAN,
        )
        ax3.text(
            timezz[i],
            bc - 0.05 * mean_cost.max(),
            f"{bc:.3f}",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=CLR_BEST,
        )

    ax3.set_xlabel("Iteration", fontsize=13, fontweight="bold", labelpad=6)
    ax3.set_ylabel("Cost function value", fontsize=13, fontweight="bold", labelpad=6)
    ax3.set_title("Cost function evolution", fontsize=14, fontweight="bold", pad=10)
    ax3.set_xlim(timezz[0] - 0.3, timezz[-1] + 0.3)
    ax3.set_ylim(bottom=0, top=mean_cost.max() * 1.15)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax3.yaxis.set_major_locator(MaxNLocator(6))
    ax3.tick_params(which="both", length=4, labelsize=11)
    ax3.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, color="#AAAAAA")
    ax3.spines[["top", "right"]].set_visible(False)

    leg3 = ax3.legend(
        loc="upper right", frameon=True, handlelength=2.4, borderpad=0.65, fontsize=12
    )
    for t in leg3.get_texts():
        t.set_fontweight("bold")

    # ── super title ───────────────────────────────────────────────────────────
    sign_label = "reduction" if pct_improv > 0 else "increase"
    fig.suptitle(
        f"Ensemble History Matching — Cost Function Summary\n"
        f"N = {N} realisations  |  "
        f"Initial mean RMSE = {mean_init:.4f}  →  "
        f"Final mean RMSE = {mean_final:.4f}  |  "
        f"{abs(pct_improv):.1f}% {sign_label}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # ── save ──────────────────────────────────────────────────────────────────
    os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
    plt.savefig(
        "Cost_Function.png",
        bbox_inches="tight",
        dpi=200,
        facecolor="white",
        edgecolor="none",
    )
    os.chdir(oldfolder)
    plt.clf()
    plt.close()
    print(
        f"  Saved Cost_Function.png  |  "
        f"Initial μ={mean_init:.4f}  Final μ={mean_final:.4f}  "
        f"({abs(pct_improv):.1f}% {sign_label})"
    )


def inverse_to_pytorch(Ne, val, nx, ny, nz, device, make_up):
    """Resize ensemble columns to padded spatial dims and return a GPU tensor.

    Parameters
    ----------
    Ne : int
        Number of ensemble members.
    val : np.ndarray
        Ensemble array, shape (nx*ny*nz, Ne).
    nx : int
        Number of grid cells in X direction.
    ny : int
        Number of grid cells in Y direction.
    nz : int
        Number of grid cells in Z direction.
    device : torch.device
        Target compute device (CPU or CUDA).
    make_up : int
        Number of extra pixels added to each spatial dimension via imresize.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape (Ne, nz, nx+make_up, ny+make_up) on `device`.
    """
    X_unie = np.zeros((Ne, nz, nx + make_up, ny + make_up))
    for i in range(Ne):
        aa = np.zeros((nz, nx + make_up, ny + make_up))
        tempp = np.reshape(val[:, i], (nx, ny, nz), "F")
        for kk in range(int(nz)):
            newy = imresize(tempp[:, :, kk], output_shape=(nx + make_up, ny + make_up))
            aa[kk, :, :] = newy
        X_unie[i, :, :, :] = aa
    return torch.from_numpy(X_unie).to(device, dtype=torch.float32)


def pytorch_to_inverse(Ne, val, nx, ny, nz):
    """Resize padded spatial tensor back to original grid and flatten columns.

    Parameters
    ----------
    Ne : int
        Number of ensemble members.
    val : np.ndarray or torch.Tensor
        Array of shape (Ne, nz, padded_nx, padded_ny).
    nx : int
        Target number of grid cells in X direction.
    ny : int
        Target number of grid cells in Y direction.
    nz : int
        Number of grid cells in Z direction.

    Returns
    -------
    np.ndarray
        Ensemble array of shape (nx*ny*nz, Ne).
    """
    X_unie = np.zeros((nz * nx * ny, Ne))
    for i in range(Ne):
        tempp = val[i, :, :, :]
        aa = np.zeros((nx, ny, nz))
        for kk in range(nz):
            newy = imresize(tempp[kk, :, :], output_shape=(nx, ny))
            aa[:, :, kk] = newy
        X_unie[:, i] = np.reshape(aa, (-1,), "F")
    return X_unie


def Remove_True(enss, locc):
    """Extract and reshape the `locc`-th column of an ensemble matrix.

    Parameters
    ----------
    enss : np.ndarray
        Ensemble matrix of shape (n_cells, N_ens).
    locc : int
        1-based index of the column (realisation) to extract.

    Returns
    -------
    np.ndarray
        Column vector of shape (n_cells, 1) in Fortran order.
    """
    return np.reshape(enss[:, locc - 1], (-1, 1), "F")


def Trim_ensemble(enss, locc):
    """Delete the `locc`-th column from an ensemble matrix.

    Parameters
    ----------
    enss : np.ndarray
        Ensemble matrix of shape (n_cells, N_ens).
    locc : int
        1-based index of the column (realisation) to remove.

    Returns
    -------
    np.ndarray
        Ensemble matrix with one fewer column, shape (n_cells, N_ens-1).
    """
    return np.delete(enss, locc - 1, 1)  # delete last column


def adaptive_rho(True_dataa, simDatafinal, rho):
    """Adaptively inflate or deflate `rho` based on mean absolute innovation.

    Parameters
    ----------
    True_dataa : np.ndarray or torch.Tensor
        Observed data vector.
    simDatafinal : np.ndarray or torch.Tensor
        Simulated data ensemble mean or single realisation.
    rho : float
        Current inflation factor.

    Returns
    -------
    float
        Updated inflation factor after applying increase or decrease.
    """
    # Parameters for adaptive inflation
    increase_factor = 1.05
    decrease_factor = 0.95
    threshold = 0.1  # Example threshold, adjust as necessary
    if not isinstance(True_dataa, torch.Tensor):
        True_dataa = torch.as_tensor(True_dataa, dtype=torch.float32)
    if not isinstance(simDatafinal, torch.Tensor):
        simDatafinal = torch.as_tensor(simDatafinal, dtype=torch.float32)
    innovation = True_dataa - simDatafinal
    MSE = torch.mean(torch.abs(innovation))
    if threshold < MSE:
        rho *= increase_factor
    else:
        rho *= decrease_factor
    return rho


def extract_non_zeros(mat):
    """Return rows of `mat` where the first column is non-negative, plus indices.

    Parameters
    ----------
    mat : np.ndarray
        2D array from which to select rows.

    Returns
    -------
    np.ndarray
        Subset of `mat` rows satisfying the condition.
    np.ndarray
        Integer indices of the selected rows.
    """
    indices = np.where(mat[:, 0] >= 0)[0]
    return mat[indices, :], indices


def place_back(extracted, indices, shape):
    """Scatter `extracted` rows back into a -1-filled array at `indices`.

    Parameters
    ----------
    extracted : np.ndarray or torch.Tensor
        Rows to scatter back, shape (len(indices), n_cols).
    indices : np.ndarray
        Integer row positions in the output array.
    shape : tuple
        Shape of the output array (total_rows, n_cols).

    Returns
    -------
    np.ndarray
        CPU NumPy array of `shape` with -1 background and extracted values placed.
    """
    result = torch.ones(shape, dtype=torch.float32) * -1
    for i, index in enumerate(indices):
        result[index, :] = extracted[i]
    return result.cpu().numpy()


def Recover_imageV(x, Ne, nx, ny, nz, latent_dim, vae, High_K, mem):
    """Decode latent vectors through a VAE decoder to recover spatial fields.

    Parameters
    ----------
    x : np.ndarray
        Latent ensemble matrix, shape (latent_dim, Ne).
    Ne : int
        Number of ensemble members.
    nx : int
        Number of grid cells in X direction.
    ny : int
        Number of grid cells in Y direction.
    nz : int
        Number of grid cells in Z direction.
    latent_dim : int
        Dimensionality of the VAE latent space.
    vae : object
        Trained VAE object with a `decoder.predict` method.
    High_K : float
        Scaling factor; multiplied by decoded output when `mem == 1`.
    mem : int
        If 1, scale decoded output by `High_K`; else use scale of 1.

    Returns
    -------
    np.ndarray
        Decoded field ensemble, shape (nx*ny*nz, Ne).
    """
    X_unie = np.zeros((Ne, latent_dim))
    for i in range(Ne):
        X_unie[i, :] = np.reshape(x[:, i], (latent_dim,), "F")
    if mem == 1:
        decoded_imgs2 = (vae.decoder.predict(X_unie)) * High_K
    else:
        decoded_imgs2 = (vae.decoder.predict(X_unie)) * 1
    ouut = np.zeros((nx * ny * nz, Ne))
    for i in range(Ne):
        jj = decoded_imgs2[i]  # .reshape(nx,ny,nz)
        jj = np.reshape(jj, (-1, 1), "F")
        ouut[:, i] = np.ravel(jj)
    return ouut


def fast_gaussian(dimension, Sdev, Corr):
    """Generate a Gaussian random field via Cholesky decomposition of a Toeplitz matrix.

    Parameters
    ----------
    dimension : array-like
        Grid dimensions; length 1 for square or length 2 for (m, n).
    Sdev : float or np.ndarray
        Standard deviation(s) for the field; scalar or per-cell array.
    Corr : list
        Correlation lengths [corr_x] or [corr_x, corr_y].

    Returns
    -------
    np.ndarray
        Flattened Gaussian random field of size m*n.

    Raises
    ------
    ValueError
        If `dimension` has more than 2 elements or `Corr` has more than 2 elements.
    """
    dimension = np.array(dimension).flatten()
    m = dimension[0]
    if len(dimension) == 1:
        n = m
    elif len(dimension) == 2:
        n = dimension[1]
    else:
        raise ValueError(
            "FastGaussian: Wrong input, dimension should have length at most 2"
        )
    # If Sdev is provided element-wise, the variance is folded in below; otherwise
    # use the scalar Sdev as the variance (works through the Kronecker product).
    variance = 1 if np.max(np.size(Sdev)) > 1 else Sdev
    if len(Corr) == 1:
        Corr = np.array([Corr[0], Corr[0]])
    elif len(Corr) > 2:
        raise ValueError("FastGaussian: Wrong input, Corr should have length at most 2")
    dist = np.arange(0, m) / Corr[0]
    T = scipy.linalg.toeplitz(dist)
    T = variance * np.exp(-(T**2)) + 1e-10 * np.eye(m)
    cholT = np.linalg.cholesky(T)
    if Corr[0] == Corr[1] and n == m:
        cholT2 = cholT
    else:
        dist2 = np.arange(0, n) / Corr[1]
        T2 = scipy.linalg.toeplitz(dist2)
        T2 = variance * np.exp(-(T2**2)) + 1e-10 * np.eye(n)
        cholT2 = np.linalg.cholesky(T2)
    rng = np.random.default_rng()
    x = rng.standard_normal(m * n)
    x = np.dot(cholT.T, np.dot(x.reshape(m, n), cholT2))
    x = x.flatten()
    if np.max(np.size(Sdev)) > 1:
        if np.min(np.shape(Sdev)) == 1 and len(Sdev) == len(x):
            x = Sdev * x
        else:
            raise ValueError("FastGaussian: Inconsistent dimension of Sdev")
    return x


def NorneInitialEnsemble(nx, ny, nz, ensembleSize=100, randomNumber=1.2345e5):
    """Generate initial permeability, porosity, and fault multiplier ensembles.

    Parameters
    ----------
    nx : int
        Number of grid cells in X direction.
    ny : int
        Number of grid cells in Y direction.
    nz : int
        Number of grid cells in Z direction.
    ensembleSize : int, optional
        Number of ensemble members to generate, by default 100.
    randomNumber : float, optional
        Seed for the NumPy random generator, by default 1.2345e5.

    Returns
    -------
    ensembleperm : np.ndarray
        Permeability ensemble, shape (nx*ny*nz, ensembleSize).
    ensembleporo : np.ndarray
        Porosity ensemble, shape (nx*ny*nz, ensembleSize).
    ensemblefault : np.ndarray
        Fault multiplier ensemble, shape (53, ensembleSize).
    """
    rng = np.random.default_rng(int(randomNumber))
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
        A_MZ = A_L[:, [0, 7, 10, 11, 14, 17]]  # Adjusted indexing to 0-based
        A_MZ = A_MZ.flatten()
        X = M_MF + S_MF * rng.standard_normal(53)
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
    """Generate a Gaussian random field with a randomly perturbed correlation length.

    Parameters
    ----------
    field_dim : array-like
        Grid dimensions, e.g. (nx, ny) or (nx, ny, nz).
    mean_value : np.ndarray
        Background mean array of size prod(field_dim).
    sdev : float or np.ndarray
        Standard deviation(s); scalar or per-cell array.
    mean_corr_length : float
        Mean of the Gaussian correlation length.
    std_corr_length : float
        Standard deviation used to perturb the correlation length.

    Returns
    -------
    x : np.ndarray
        Realised random field of size prod(field_dim).
    corr_length : float
        Actual correlation length used for the last layer.
    """
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
                # Generate new correlation length for the next layer
                corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)
        else:
            for i in range(field_dim[2]):
                idx_range = slice(i * layer_dim, (i + 1) * layer_dim)
                x[idx_range] = mean_value[idx_range] + fast_gaussian(
                    field_dim[:2], sdev[idx_range], corr_length
                )
                # Generate new correlation length for the next layer
                corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)
    return x, corr_length
