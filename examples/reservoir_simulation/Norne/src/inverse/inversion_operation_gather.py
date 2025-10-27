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

@Author : Clement Etienam
"""

# Standard Libraries
import os
import pickle
import logging
import gzip
import warnings
import yaml
from collections import OrderedDict

# Numerical Computing
import numpy as np
import numpy.matlib
import pandas as pd
import scipy

# Machine Learning
import torch
from hydra.utils import to_absolute_path

# Visualization
import matplotlib.pyplot as plt

# 📦 Local Modules
from imresize import imresize
from inverse.inversion_operation_misc import (
    add_gnoise,
    NorneGeostat,
)


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Inverse problem")
    if not logger.handlers:
        f_handler = logging.FileHandler(filename="read_vectors.log", mode="w")
        formatter = logging.Formatter(" %(asctime)s - %(levelname)s - %(message)s")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
        logger.setLevel(logging.INFO)
    warnings.filterwarnings("ignore")
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
    gass,
):
    if Truee.ndim == 3:
        avg_2d = np.mean(Truee, axis=2)
    else:
        avg_2d = np.reshape(Truee, (nx, ny), "F")
    maxii = max(avg_2d.ravel())
    minii = min(avg_2d.ravel())
    avg_2d[avg_2d == 0] = np.nan  # Convert zeros to NaNs
    plt.pcolormesh(XX.T, YY.T, avg_2d, cmap="jet")
    cbar = plt.colorbar()
    if varii == "perm":
        cbar.set_label("Log K(mD)", fontsize=11)
        plt.title("Permeability Field with well locations", fontsize=11, weight="bold")
    elif varii == "water PhyNeMo":
        cbar.set_label("water saturation", fontsize=11)
        plt.title("water saturation -PhyNeMo", fontsize=11, weight="bold")
    elif varii == "water FLOW":
        cbar.set_label("water saturation", fontsize=11)
        plt.title("water saturation - FLOW", fontsize=11, weight="bold")
    elif varii == "water diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("water saturation - (FLOW -PhyNeMo)", fontsize=11, weight="bold")
    elif varii == "oil PhyNeMo":
        cbar.set_label("Oil saturation", fontsize=11)
        plt.title("Oil saturation -PhyNeMo", fontsize=11, weight="bold")

    elif varii == "oil FLOW":
        cbar.set_label("Oil saturation", fontsize=11)
        plt.title("Oil saturation - Flow", fontsize=11, weight="bold")
    elif varii == "oil diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("oil saturation - (FLOW -PhyNeMo)", fontsize=11, weight="bold")
    elif varii == "gas PhyNeMo":
        cbar.set_label("Gas saturation", fontsize=11)
        plt.title("Gas saturation -PhyNeMo", fontsize=11, weight="bold")
    elif varii == "gas FLOW":
        cbar.set_label("Gas saturation", fontsize=11)
        plt.title("Gas saturation -FLOW", fontsize=11, weight="bold")
    elif varii == "gas diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("gas saturation - (FLOW -PhyNeMo)", fontsize=11, weight="bold")
    elif varii == "pressure PhyNeMo":
        cbar.set_label("pressure", fontsize=11)
        plt.title("Pressure -PhyNeMo", fontsize=11, weight="bold")
    elif varii == "pressure FLOW":
        cbar.set_label("pressure", fontsize=11)
        plt.title("Pressure -FLOW", fontsize=11, weight="bold")
    elif varii == "pressure diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("Pressure - (FLOW -PhyNeMo)", fontsize=11, weight="bold")
    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=11)
        plt.title("Porosity Field", fontsize=11, weight="bold")
    cbar.mappable.set_clim(minii, maxii)
    plt.ylabel("Y", fontsize=11)
    plt.xlabel("X", fontsize=11)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    Add_marker2(plt, XX, YY, injectors, producers, gass)


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
    gass,
):
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
    elif varii == "water PhyNeMo":
        cbar.set_label("water saturation", fontsize=11)
    elif varii == "oil PhyNeMo":
        cbar.set_label("Oil saturation", fontsize=11)
    elif varii == "gas PhyNeMo":
        cbar.set_label("Gas saturation", fontsize=11)
    elif varii == "pressure PhyNeMo":
        cbar.set_label("pressure", fontsize=11)
    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=11)
    cbar.mappable.set_clim(minii, maxii)
    plt.ylabel("Y", fontsize=11)
    plt.xlabel("X", fontsize=11)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    Add_marker2(plt, XX, YY, injectors, producers, gass)


def Add_marker2(plt, XX, YY, injectors, producers, gass):
    n_inj = len(injectors)  # Number of injectors
    n_prod = len(producers)  # Number of producers
    n_injg = len(gass)  # Number of gas injectors
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


def Add_marker3(plt, XX, YY, injectors, producers, gass):
    n_inj = len(injectors)  # Number of injectors
    n_prod = len(producers)  # Number of producers
    n_injg = len(gass)  # Number of gas injectors
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
    effectiveuse,
    injectors,
    producers,
    gass,
    N_injw,
    N_pr,
    N_injg,
):
    Low_Ka = Low_K
    High_Ka = High_K
    permmean = np.mean(np.reshape(permmean, (nx, ny, nz), "F") * effectiveuse, axis=2)
    permbest = np.mean(np.reshape(permbest, (nx, ny, nz), "F") * effectiveuse, axis=2)
    iniperm = np.mean(np.reshape(iniperm, (nx, ny, nz), "F") * effectiveuse, axis=2)
    True_perm = np.mean(np.reshape(True_perm, (nx, ny, nz), "F") * effectiveuse, axis=2)
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
        gass,
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
        gass,
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
        gass,
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
        gass,
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
    effectiveuse,
    N_injw,
    N_pr,
    N_injg,
    injectors,
    producers,
    gass,
    Low_P,
    High_P,
):
    Low_Ka = Low_K
    High_Ka = High_K
    permmean = np.reshape(permmean, (nx, ny, nz), "F")
    poroo = np.reshape(poroo, (nx, ny, nz), "F")
    from skimage.restoration import denoise_nl_means, estimate_sigma

    temp_K = permmean
    temp_phi = poroo
    timk = 5
    for n in range(timk):
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
    permmean = np.mean(np.reshape(permmean, (nx, ny, nz), "F") * effectiveuse, axis=2)
    temp_K = np.mean(np.reshape(temp_K, (nx, ny, nz), "F") * effectiveuse, axis=2)
    poroo = np.mean(np.reshape(poroo, (nx, ny, nz), "F") * effectiveuse, axis=2)
    temp_phi = np.mean(np.reshape(temp_phi, (nx, ny, nz), "F") * effectiveuse, axis=2)
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
        gass,
    )
    plt.title(" Permeability", fontsize=15)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers, gass)
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
        gass,
    )
    plt.title("Smoothed - Permeability", fontsize=15)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers, gass)
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
        gass,
    )
    plt.title("Porosity", fontsize=15)
    plt.clim(Low_P, High_P)
    Add_marker2(plt, XX, YY, injectors, producers, gass)
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
        gass,
    )
    plt.title("Smoothed Porosity", fontsize=15)
    plt.clim(Low_P, High_P)
    Add_marker2(plt, XX, YY, injectors, producers, gass)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Petrophysical Reconstruction", fontsize=25)
    plt.savefig("Petro_Recon.png")
    plt.close()
    plt.clf()


def Getporosity_ensemble(ini_ensemble, machine_map, N_ens):
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
    ini_ensemble = np.hstack(ini_ensemble)
    return ini_ensemble


def Plot_RSM_percentile(pertoutt, True_mat, timezz, N_pr, well_names):
    columns = well_names  # ['L1', 'L2', 'L3', 'LU1', 'LU2',
    P10 = pertoutt[0]
    P50 = pertoutt[1]
    P90 = pertoutt[2]
    arekibest = pertoutt[3]
    amean = pertoutt[4]
    base = pertoutt[5]
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k], color="red", lw="2", label="model")
        plt.plot(timezz, P10[:, k], color="blue", lw="2", label="P10 Model")
        plt.plot(timezz, P50[:, k], color="c", lw="2", label="P50 Model")
        plt.plot(timezz, P90[:, k], color="green", lw="2", label="P90 Model")
        plt.plot(
            timezz, arekibest[:, k], color="k", lw="2", label="aREKI cum best Model"
        )
        plt.plot(
            timezz, amean[:, k], color="purple", lw="2", label="aREKI cum MAP Model"
        )
        plt.plot(timezz, base[:, k], color="orange", lw="2", label="Base case Model")
        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Oil.png")
    plt.clf()
    plt.close()
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + N_pr], color="red", lw="2", label="model")
        plt.plot(timezz, P10[:, k + N_pr], color="blue", lw="2", label="P10 Model")
        plt.plot(timezz, P50[:, k + N_pr], color="c", lw="2", label="P50 Model")
        plt.plot(timezz, P90[:, k + N_pr], color="green", lw="2", label="P90 Model")
        plt.plot(
            timezz,
            arekibest[:, k + N_pr],
            color="k",
            lw="2",
            label="aREKI cum best Model",
        )
        plt.plot(
            timezz,
            amean[:, k + N_pr],
            color="purple",
            lw="2",
            label="aREKI cum MAP Model",
        )
        plt.plot(
            timezz, base[:, k + N_pr], color="orange", lw="2", label="Base case Model"
        )
        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Water.png")
    plt.clf()
    plt.close()
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 2 * N_pr], color="red", lw="2", label="model")
        plt.plot(timezz, P10[:, k + 2 * N_pr], color="blue", lw="2", label="P10 Model")
        plt.plot(timezz, P50[:, k + 2 * N_pr], color="c", lw="2", label="P50 Model")
        plt.plot(timezz, P90[:, k + 2 * N_pr], color="green", lw="2", label="P90 Model")
        plt.plot(
            timezz,
            arekibest[:, k + 2 * N_pr],
            color="k",
            lw="2",
            label="aREKI cum best Model",
        )
        plt.plot(
            timezz,
            amean[:, k + 2 * N_pr],
            color="purple",
            lw="2",
            label="aREKI cum MAP Model",
        )
        plt.plot(
            timezz,
            base[:, k + 2 * N_pr],
            color="orange",
            lw="2",
            label="Base case Model",
        )
        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Gas.png")
    plt.clf()
    plt.close()
    True_data = np.reshape(True_mat, (-1, 1), "F")
    P10 = np.reshape(P10, (-1, 1), "F")
    cc10 = ((np.sum((((P10) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    logger.info("RMSE of P10 reservoir model  =  " + str(cc10))
    P50 = np.reshape(P50, (-1, 1), "F")
    cc50 = ((np.sum((((P50) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    logger.info("RMSE of P50 reservoir model  =  " + str(cc50))
    P90 = np.reshape(P90, (-1, 1), "F")
    cc90 = ((np.sum((((P90) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    logger.info("RMSE of P90 reservoir model  = : " + str(cc90))
    P99 = np.reshape(arekibest, (-1, 1), "F")
    cc99 = ((np.sum((((P99) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    logger.info("RMSE of cummulative best reservoir model  =  " + str(cc99))
    Pmean = np.reshape(amean, (-1, 1), "F")
    ccmean = ((np.sum((((Pmean) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    logger.info("RMSE of cummulative mean reservoir model  =  " + str(ccmean))
    Pbase = np.reshape(base, (-1, 1), "F")
    ccbase = ((np.sum((((Pbase) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    logger.info("RMSE of base case reservoir model  =  " + str(ccbase))
    plt.figure(figsize=(10, 10))
    values = [cc10, cc50, cc90, cc99, ccmean, ccbase]
    model_names = ["P10", "P50", "P90", "cumm-best", "cumm-mean", "Base case"]
    colors = ["blue", "c", "green", "k", "purple", "orange"]
    min_rmse_index = np.argmin(values)
    min_rmse = values[min_rmse_index]
    best_model = model_names[min_rmse_index]
    logger.info(f"The minimum RMSE value = {min_rmse}")
    logger.info(f"Recommended reservoir model = {best_model} reservoir model.")
    plt.bar(model_names, values, color=colors)
    plt.xlabel("Reservoir Models")
    plt.ylabel("RMSE")
    plt.title("Histogram of RMSE Values for Different Reservoir Models")
    plt.legend(model_names)
    plt.savefig("Histogram.png")
    plt.clf()
    plt.close()


def Plot_RSM_percentile_model(pertoutt, True_mat, timezz, N_pr, well_names):
    columns = well_names  # ['L1', 'L2', 'L3', 'LU1', 'LU2',
    P10 = pertoutt
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k], color="red", lw="2", label="model")
        plt.plot(timezz, P10[:, k], color="blue", lw="2", label="Surrogate")
        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Oil.png")
    plt.clf()
    plt.close()
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + N_pr], color="red", lw="2", label="model")
        plt.plot(timezz, P10[:, k + N_pr], color="blue", lw="2", label="P10 Model")
        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Water.png")
    plt.clf()
    plt.close()
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 2 * N_pr], color="red", lw="2", label="model")
        plt.plot(timezz, P10[:, k + 2 * N_pr], color="blue", lw="2", label="P10 Model")
        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Gas.png")
    plt.clf()
    plt.close()


def Plot_RSM_single(True_mat, timezz, N_pr, well_names):
    True_mat = True_mat[0]
    columns = well_names  # ['L1', 'L2', 'L3', 'LU1', 'LU2',
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k], color="red", lw="2", label="model")
        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Oil_single.png")
    plt.clf()
    plt.close()
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + N_pr], color="red", lw="2", label="model")
        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Water_single.png")
    plt.clf()
    plt.close()
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 2 * N_pr], color="red", lw="2", label="model")
        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Gas_single.png")
    plt.clf()
    plt.close()


def Plot_RSM_singleT(True_mat, timezz, N_pr, well_names):
    columns = well_names  # ['L1', 'L2', 'L3', 'LU1', 'LU2',
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k], color="red", lw="2", label="model")
        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Oil_singleT.png")
    plt.clf()
    plt.close()
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + N_pr], color="red", lw="2", label="model")

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Water_singleT.png")
    plt.clf()
    plt.close()
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 2 * N_pr], color="red", lw="2", label="model")
        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Gas_singleT.png")
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def initialize_arrays(Nt, Ne, num_variables):
    return np.zeros((Nt, Ne, num_variables))


def plot_variable(timezz, pred_data, true_data, Ne, title, ylabel, ax):
    for j in range(Ne):
        ax.plot(
            timezz,
            pred_data[:, j],
            color="grey",
            lw=2,
            label="Realisations" if j == 0 else "_nolegend_",
        )
    ax.plot(timezz, true_data, color="red", lw=2, label="True model")
    ax.axvline(x=1500, color="black", linestyle="--")
    ax.set_xlabel("Time (days)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


def plot_rsm(
    timezz, pred_matrix, true_mat, Ne, variable_names, ylabels, file_name, Namesz
):
    Nt = pred_matrix[0].shape[0]
    data_matrix = initialize_arrays(Nt, Ne, len(variable_names))
    for i in range(Ne):
        for var_index in range(len(variable_names)):
            data_matrix[:, i, var_index] = pred_matrix[i][:, var_index]
    plt.figure(figsize=(40, 40))
    for var_index, name in enumerate(variable_names):
        ax = plt.subplot((len(variable_names) + 2) // 3, 3, var_index + 1)
        plot_variable(
            timezz,
            data_matrix[:, :, var_index],
            true_mat[:, var_index],
            Ne,
            name,
            ylabels[var_index],
            ax,
        )
    plt.savefig(f"{file_name}_{Namesz}.png")
    plt.clf()
    plt.close()


def Plot_RSM(predMatrix, True_mat, Namesz, Ne, timezz, N_pr, well_names):
    variable_names = well_names  # ['L1', 'L2', 'L3', 'LU1',
    oil_ylabels = ["$Q_{oil}(bbl/day)$"] * N_pr
    water_ylabels = ["$Q_{water}(bbl/day)$"] * N_pr
    gas_ylabels = ["$Q_{gas}(scf/day)$"] * N_pr
    plot_rsm(
        timezz,
        predMatrix[:, :, :N_pr],
        True_mat[:, :N_pr],
        Ne,
        variable_names,
        oil_ylabels,
        "Oil",
        Namesz,
    )
    plot_rsm(
        timezz,
        predMatrix[:, :, N_pr : 2 * N_pr],
        True_mat[:, N_pr : 2 * N_pr],
        Ne,
        variable_names,
        water_ylabels,
        "Water",
        Namesz,
    )
    plot_rsm(
        timezz,
        predMatrix[:, :, 2 * N_pr : 3 * N_pr],
        True_mat[:, 2 * N_pr : 3 * N_pr],
        Ne,
        variable_names,
        gas_ylabels,
        "Gas",
        Namesz,
    )


def De_correlate_ensemble(nx, ny, nz, Ne, High_K, Low_K, Yet):
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
    X[X <= Low_K] = Low_K
    X[X >= High_K] = High_K
    return X[:, :Ne]


def whiten(X, method="zca"):
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


def write_RSM(data, Time, Name, well_names, N_pr):
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
    if isinstance(x, torch.Tensor):
        return (
            x.detach().cpu().numpy()
        )  # Ensure it's detached, moved to CPU, and converted
    return x  # If it's already a NumPy array or another type, return as is


def Plot_Histogram_now(N, E11, E12, mean_cost, best_cost, oldfolder):
    N = to_numpy(N)
    E11 = to_numpy(E11)
    E12 = to_numpy(E12)
    mean_cost = to_numpy(mean_cost)
    best_cost = to_numpy(best_cost)
    mean_cost = np.vstack(mean_cost).reshape(-1, 1)
    best_cost = np.vstack(best_cost).reshape(-1, 1)
    reali = np.arange(1, N + 1)
    timezz = np.arange(1, mean_cost.shape[0] + 1)
    plttotalerror = np.reshape(E11, (N))  # Initial
    plttotalerror2 = np.reshape(E12, (N))  # Final
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.bar(reali, plttotalerror, color="c")
    plt.xlabel("Realizations")
    plt.ylabel("RMSE value")
    plt.ylim(ymin=0)
    plt.title("Initial Cost function")
    plt.scatter(reali, plttotalerror, s=1, color="k")
    plt.xlabel("Realizations")
    plt.ylabel("RMSE value")
    plt.xlim([1, (N - 1)])
    plt.subplot(2, 2, 2)
    plt.bar(reali, plttotalerror2, color="c")
    plt.xlabel("Realizations")
    plt.ylabel("RMSE value")
    plt.ylim(ymin=0)
    plt.ylim(ymax=max(plttotalerror))
    plt.title("Final Cost function")
    plt.scatter(reali, plttotalerror2, s=1, color="k")
    plt.xlabel("Realizations")
    plt.ylabel("RMSE value")
    plt.xlim([1, (N - 1)])
    plt.subplot(2, 2, 3)
    plt.plot(timezz, mean_cost, color="green", lw="2", label="mean_model_cost")
    plt.plot(timezz, best_cost, color="blue", lw="2", label="best_model_cost")
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function Value")
    # plt.ylim((0,25000))
    plt.title("Cost Evolution")
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()
    os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
    plt.savefig("Cost_Function.png")
    os.chdir(oldfolder)
    plt.close()
    plt.clf()


def inverse_to_pytorch(Ne, val, nx, ny, nz, device, make_up):
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
    return np.reshape(enss[:, locc - 1], (-1, 1), "F")


def Trim_ensemble(enss, locc):
    return np.delete(enss, locc - 1, 1)  # delete last column


def read_yaml(fname):
    """Read Yaml file into a dict of parameters"""
    logger.info(f"Read simulation cfg from {fname}...")
    with open(fname, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            # print(data)
        except yaml.YAMLError as exc:
            print(exc)
        return data


def adaptive_rho(True_dataa, simDatafinal, rho):
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
    if MSE > threshold:
        rho *= increase_factor
    else:
        rho *= decrease_factor
    return rho


def extract_non_zeros(mat):
    indices = np.where(mat[:, 0] >= 0)[0]
    return mat[indices, :], indices


def place_back(extracted, indices, shape):
    result = torch.ones(shape, dtype=torch.float32) * -1
    for i, index in enumerate(indices):
        result[index, :] = extracted[i]
    return result.get()


def Recover_imageV(x, Ne, nx, ny, nz, latent_dim, vae, High_K, mem):
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
    if np.max(np.size(Sdev)) > 1:  # check input
        variance = 1
    else:
        variance = Sdev
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


def get_shape(t):
    shape = []
    while isinstance(t, tuple):
        shape.append(len(t))
        t = t[0]
    return tuple(shape)


def NorneInitialEnsemble(nx, ny, nz, ensembleSize=100, randomNumber=1.2345e5):
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
