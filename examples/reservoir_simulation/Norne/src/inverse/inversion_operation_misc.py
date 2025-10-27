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

import os
import re
import sys
import time
import logging
import warnings
from collections import OrderedDict
import numpy as np
import numpy.matlib
import pandas as pd
import scipy.ndimage.morphology as spndmo
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from physicsnemo.models.fno import FNO
from physicsnemo.models.module import Module
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib import cm
from gstools.random import MasterRNG
from gstools import SRF, Gaussian


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


def sort_key(s):
    return int(re.search(r"\d+", s).group())


def plot_and_save(
    kk,
    dt,
    pree,
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
    gass,
    effectiveuse,
    Time_vector,
):
    """Render 3D voxel plots for pressure/water/oil/gas and return figure.

    Parameters capture timestep index, field tensors, grid shape and well
    locations; returns the timestep index and the created Matplotlib figure.
    """
    current_time = dt[kk]
    Time_vector[kk] = current_time
    f_3 = plt.figure(figsize=(20, 20), dpi=200)
    look = (pree[0, kk, :, :, :]) * effectiveuse
    ax1 = f_3.add_subplot(2, 2, 1, projection="3d")
    Plot_PhyNeMo(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "pressure PhyNeMo",
        injectors,
        producers,
        gass,
    )
    look = (wats[0, kk, :, :, :]) * effectiveuse
    ax2 = f_3.add_subplot(2, 2, 2, projection="3d")
    Plot_PhyNeMo(
        ax2,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "water PhyNeMo",
        injectors,
        producers,
        gass,
    )
    look = oilss[0, kk, :, :, :]
    look = look * effectiveuse
    ax3 = f_3.add_subplot(2, 2, 3, projection="3d")
    Plot_PhyNeMo(
        ax3,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "oil PhyNeMo",
        injectors,
        producers,
        gass,
    )

    look = (gasss[0, kk, :, :, :]) * effectiveuse
    ax4 = f_3.add_subplot(2, 2, 4, projection="3d")
    Plot_PhyNeMo(
        ax4,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "gas PhyNeMo",
        injectors,
        producers,
        gass,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    tita = "Timestep --" + str(current_time) + " days"
    plt.suptitle(tita, fontsize=16)
    # Return the kk and figure
    return kk, f_3


def add_gnoise(Ytrue, SIGMA, SQ=None):
    """Add Gaussian noise with scalar, diagonal or full covariance.

    Returns the noisy vector and the square-root of covariance actually used.
    """
    try:
        if SQ is not None and SQ == 1:
            RTSIGMA = SIGMA
            if np.isscalar(SIGMA) or np.ndim(SIGMA) == 1:
                rng = np.random.default_rng()
                error = RTSIGMA * rng.standard_normal(1)
            else:
                rng = np.random.default_rng()
                error = RTSIGMA @ rng.standard_normal((RTSIGMA.shape[1], 1))
        else:
            if np.isscalar(SIGMA) or np.ndim(SIGMA) == 1:
                RTSIGMA = np.sqrt(SIGMA)
                rng = np.random.default_rng()
                error = RTSIGMA * rng.standard_normal(Ytrue.shape)
            else:
                try:
                    RTSIGMA = np.linalg.cholesky(SIGMA).T
                except np.linalg.LinAlgError:
                    logger.warning("Problem with Cholesky factorization")
                    RTSIGMA = np.sqrtm(SIGMA).real
                    logger.info("Finally - we got a square root!")
                rng = np.random.default_rng()
                error = RTSIGMA @ rng.standard_normal(Ytrue.shape)
        Y = Ytrue + error.flatten()
    except Exception as e:
        logger.error("Error in AddGnoise")
        raise e
    return Y, RTSIGMA


def adjust_variable_within_bounds(variable, lowerbound=None, upperbound=None):
    """Clamp values in `variable` to provided lower/upper bounds in-place.

    Returns the adjusted array and the number of modified entries.
    """
    if lowerbound is None and upperbound is None:
        raise ValueError("At least one of lowerbound or upperbound must be provided.")
    n = 0
    ne = variable.shape[1]
    if lowerbound is not None:
        if np.isscalar(lowerbound):
            n += np.sum(variable < lowerbound)
            variable[variable < lowerbound] = lowerbound
        else:
            lowerbound_repeated = np.tile(lowerbound.reshape(-1, 1), (1, ne))
            n += np.sum(variable < lowerbound_repeated)
            variable[variable < lowerbound_repeated] = lowerbound_repeated[
                variable < lowerbound_repeated
            ]
    if upperbound is not None:
        if np.isscalar(upperbound):
            n += np.sum(variable > upperbound)
            variable[variable > upperbound] = upperbound
        else:
            upperbound_repeated = np.tile(upperbound.reshape(-1, 1), (1, ne))
            n += np.sum(variable > upperbound_repeated)
            variable[variable > upperbound_repeated] = upperbound_repeated[
                variable > upperbound_repeated
            ]

    return variable, n


def initial_ensemble_gaussian(Nx, Ny, Nz, N, minn, maxx, minnp, maxxp):
    """Create Gaussian ensembles and map them to specified min/max ranges."""
    fensemble = np.zeros((Nx * Ny * Nz, N))
    ensemblep = np.zeros((Nx * Ny * Nz, N))
    x = np.arange(Nx)
    y = np.arange(Ny)
    z = np.arange(Nz)
    model = Gaussian(dim=3, var=5, len_scale=4)  # Variance and lenght scale
    srf = SRF(model)
    seed = MasterRNG(20170519)
    for k in range(N):
        aoutt = srf.structured([x, y, z], seed=seed())
        foo = np.reshape(aoutt, (-1, 1), "F")
        clfy = MinMaxScaler(feature_range=(minn, maxx))
        (clfy.fit(foo))
        fout = clfy.transform(foo)
        fensemble[:, k] = np.ravel(fout)
        clfy1 = MinMaxScaler(feature_range=(minnp, maxxp))
        (clfy1.fit(foo))
        fout1 = clfy1.transform(foo)
        ensemblep[:, k] = np.ravel(fout1)
    return fensemble, ensemblep


def read_until_line(file_path, sep=r"\s+", header=None):
    """Read numeric blocks following keywords until '/' line in deck include."""
    start_reading = False  # Flag to start reading after keyword
    data_lines = []
    keywords = ["ACTNUM", "PORO", "PERMX", "PERMY", "PERMZ"]
    with open(file_path, "r") as f:
        for line in f:
            if any(
                keyword in line for keyword in keywords
            ):  # Check if line contains any keyword
                start_reading = True
                continue  # Skip the keyword line itself
            if start_reading:
                if "/" in line:  # Stop reading when encountering '/'
                    break
                data_lines.append(line.strip())
    if not data_lines:
        raise ValueError("Error: No valid data found before '/'!")
    try:
        df = pd.DataFrame([list(map(float, row.split())) for row in data_lines])
        df = df.apply(
            pd.to_numeric, errors="coerce"
        )  # Handle possible errors in conversion
    except ValueError as e:
        raise ValueError(f"Error parsing data: {e}")
    return df.values


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
    for i, (mean_, std_) in enumerate(zip(z_means, z_stds), start=1):
        norne[f"z{i}Mean"] = mean_
        norne[f"z{i}Std"] = std_
    norne["zLB"] = -4
    norne["zUB"] = 0
    norne["multzRange"] = 26
    norne["multfltStd"] = 0.5
    norne["multfltLB"] = -5
    norne["multfltUB"] = 2
    return norne


def remove_rows(matrix, indices_to_remove):
    """Remove rows by index from 2D `matrix` and return the reduced array."""
    matrix = np.delete(matrix, indices_to_remove, axis=0)
    return matrix


def Localisation(c, nx, ny, nz, N, gass, producers, injectors):
    """Build Gaspari–Cohn localisation weights with well positions as centers."""
    A = np.zeros((nx, ny, nz))

    def set_well_locations(wells, array):
        for well in wells:
            i, j = well[0], well[1]
            array[i, j, :] = 1  # Set all z-layers at position (i, j) to 1
        return array

    A = set_well_locations(
        gass, set_well_locations(producers, set_well_locations(injectors, A))
    )

    logger.info(
        "      Calculate the Euclidean distance function to the 22 producer wells"
    )
    lf = np.reshape(A, (nx, ny, nz), "F")
    young = np.zeros((int(nx * ny * nz / nz), nz))
    for j in range(nz):
        sdf = lf[:, :, j]
        (usdf, IDX) = spndmo.distance_transform_edt(
            np.logical_not(sdf), return_indices=True
        )
        usdf = np.reshape(usdf, (int(nx * ny * nz / nz)), "F")
        young[:, j] = usdf

    sdfbig = np.reshape(young, (nx * ny * nz, 1), "F")
    sdfbig1 = abs(sdfbig)
    z = sdfbig1
    c0OIL1 = np.zeros((nx * ny * nz, 1))
    logger.info("      Computing the Gaspari-Cohn coefficent")
    for i in range(nx * ny * nz):
        if 0 <= z[i, :] or z[i, :] <= c:
            c0OIL1[i, :] = (
                -0.25 * (z[i, :] / c) ** 5
                + 0.5 * (z[i, :] / c) ** 4
                + 0.625 * (z[i, :] / c) ** 3
                - (5.0 / 3.0) * (z[i, :] / c) ** 2
                + 1
            )
        elif z < 2 * c:
            c0OIL1[i, :] = (
                (1.0 / 12.0) * (z[i, :] / c) ** 5
                - 0.5 * (z[i, :] / c) ** 4
                + 0.625 * (z[i, :] / c) ** 3
                + (5.0 / 3.0) * (z[i, :] / c) ** 2
                - 5 * (z[i, :] / c)
                + 4
                - (2.0 / 3.0) * (c / z[i, :])
            )
        elif c <= z[i, :] or z[i, :] <= 2 * c:
            c0OIL1[i, :] = -5 * (z[i, :] / c) + 4 - 0.667 * (c / z[i, :])
        else:
            c0OIL1[i, :] = 0
    c0OIL1[c0OIL1 < 0] = 0
    schur = c0OIL1
    Bsch = np.tile(schur, (1, N))
    yoboschur = np.ones((nx * ny * nz, N))
    yoboschur[: nx * ny * nz, :] = Bsch
    return yoboschur


def compute_tol(A):
    """Return SVD tolerance scaled by matrix size and infinity-norm."""
    max_dim = max(A.shape)  # Get the largest dimension of A
    eps_val = torch.finfo(A.dtype).eps  # Machine epsilon for A's data type
    tol = max_dim * eps_val * torch.linalg.norm(A, float("inf"))  # Compute tolerance
    return tol


def pinvmatt(A, tol=0):
    """Return (V, X, U) where X approximates A^{-1} using truncated SVD."""
    device = A.device
    U, S1, Vt = torch.linalg.svd(A, full_matrices=False)
    if tol == 0:
        tol = torch.max(
            A.size(0) * torch.finfo(S1.dtype).eps * torch.linalg.norm(S1, float("inf"))
        )
    r1 = torch.sum(S1 > tol).item()  # Don't add 1 here!
    U = U[:, :r1]
    Vt = Vt[:r1, :]
    S1 = S1[:r1]
    S_inv = torch.diag(1.0 / S1)  # Convert to diagonal matrix
    X = Vt.t() @ S_inv @ U.t()  # Correct multiplication order
    return Vt.t().to(device), X.to(device), U.to(device)


def Get_Kalman_Gain(
    Y, simDatafinal, CDd, alpha, device, pertubations, True_data, Ne, dist
):
    """Compute ensemble Kalman update term given data covariance and alpha."""
    if not isinstance(simDatafinal, torch.Tensor):
        simDatafinal = torch.as_tensor(simDatafinal, dtype=torch.float32, device=device)
    if not isinstance(CDd, torch.Tensor):
        CDd = torch.as_tensor(CDd, dtype=torch.float32, device=device)
    if not isinstance(Y, torch.Tensor):
        Y = torch.as_tensor(Y, dtype=torch.float32, device=device)
    sqrt_Ne_1 = torch.sqrt(torch.tensor(Ne - 1, dtype=torch.float32, device=device))
    M = torch.mean(simDatafinal, dim=1, keepdim=True)
    M2 = torch.mean(Y, dim=1, keepdim=True)
    S = simDatafinal - M
    yprime = Y - M2
    Cdd = S / sqrt_Ne_1
    Cydd = yprime / sqrt_Ne_1
    GDT = Cdd.t() @ torch.linalg.inv(CDd).sqrt()
    inv_CDd = torch.linalg.inv(CDd).sqrt()
    Cdd = GDT.t() @ GDT
    Cyd = Cydd @ GDT
    Usig, Sig, Vsig = torch.linalg.svd(
        Cdd + (alpha * torch.eye(CDd.shape[1], device=device)), full_matrices=False
    )
    Bsig = torch.cumsum(Sig, dim=0)
    threshold = Bsig[-1] * 0.9999  # Compute threshold value
    indices = torch.nonzero(Bsig >= threshold).squeeze()
    if indices.numel() > 0:  # Ensure indices is not empty before indexing
        if indices.dim() == 0:  # Handle scalar tensor case
            tol = Sig[indices.item()]  # Convert scalar tensor to index
        else:
            tol = Sig[
                indices[0].item()
            ]  # Get the first valid index  # Take the first element from the list
        if dist.rank == 0:
            logger.info(f"Using computed tolerance from singular values: {tol.item()}")
    else:
        logger.info("using default tolerance")
        default_tol = compute_tol(
            Cdd + (alpha * torch.eye(Cdd.shape[1], device=device))
        )
        tol = default_tol  # torch.tensor(1e-6, dtype=Sig.dtype, device=Sig.device)  # Fallback value
        if dist.rank == 0:
            logger.info(f"Using default tolerance: {tol.item()}")
        tol = compute_tol(Cdd + (alpha * torch.eye(Cdd.shape[1], device=device)))
    V, X, U = pinvmatt(Cdd + (alpha * torch.eye(CDd.shape[1], device=device)), tol)
    pertubations_cu = torch.as_tensor(pertubations, dtype=torch.float32, device=device)
    true_data_cu = torch.as_tensor(True_data, dtype=torch.float32, device=device)
    alpha_cu = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    tile_true_ne = true_data_cu.repeat(1, Ne).to(device)
    pertu_alpha = torch.sqrt(alpha_cu) * pertubations_cu
    factor_sum = (tile_true_ne + pertu_alpha) - simDatafinal
    update_term = Cyd @ X @ inv_CDd @ factor_sum
    torch.cuda.empty_cache()
    return update_term


def process_task(k, x, y, z, seed, minn, maxx, minnp, maxxp, var, len_scale):
    """Generate and min-max scale a gstools SRF realisation on a grid."""
    model = Gaussian(dim=3, var=var, len_scale=len_scale)
    srf = SRF(model)
    aoutt = srf.structured([x, y, z], seed=seed)
    foo = np.reshape(aoutt, (-1, 1), "F")
    clfy = MinMaxScaler(feature_range=(minn, maxx))
    clfy.fit(foo)
    fout = clfy.transform(foo)
    clfy1 = MinMaxScaler(feature_range=(minnp, maxxp))
    clfy1.fit(foo)
    fout1 = clfy1.transform(foo)
    return np.ravel(fout), np.ravel(fout1)


def ProgressBar(Total, Progress, BarLength=20, ProgressIcon="#", BarIcon="-"):
    """Return a textual progress bar string for console display."""
    try:
        if BarLength < 1:
            BarLength = 20
        Status = ""
        Progress = float(Progress) / float(Total)
        if Progress >= 1.0:
            Progress = 1
            Status = "\r\n"  # Going to the next line
        Block = int(round(BarLength * Progress))
        # Show this
        Bar = "[{}] {:.0f}% {}".format(
            ProgressIcon * Block + BarIcon * (BarLength - Block),
            round(Progress * 100, 0),
            Status,
        )
        return Bar
    except Exception:
        return "ERROR"


def ProgressBar2(Total, Progress):
    """Return percentage string (e.g., '42%') for a given progress ratio."""
    try:
        Progress = float(Progress) / float(Total)
        if Progress >= 1.0:
            Progress = 1
            return "100%"
        return "{:.0f}%".format(round(Progress * 100, 0))
    except Exception:
        logger.info("")
        return "ERROR"


def ShowBar(Bar):
    """Write a progress bar string to stdout without newline."""
    sys.stdout.write(Bar)
    sys.stdout.flush()


def Plot_PhyNeMo(
    ax, nx, ny, nz, Truee, N_injw, N_pr, N_injg, varii, injectors, producers, gass
):
    """Plot a 3D voxel field with injector/producer/gas well annotations."""
    Pressz = np.reshape(Truee, (nx, ny, nz), "F")
    avg_2d = np.mean(Pressz, axis=2)
    avg_2d[avg_2d == 0] = np.nan  # Convert zeros to NaNs
    maxii = max(Pressz.ravel())
    minii = min(Pressz.ravel())
    Pressz = Pressz / maxii
    masked_Pressz = Pressz
    colors = plt.cm.jet(masked_Pressz)
    colors[np.isnan(Pressz), :3] = 1  # set color to white for NaN values
    norm = mpl.colors.Normalize(vmin=minii, vmax=maxii)
    arr_3d = Pressz
    x, y, z = np.indices((arr_3d.shape))
    x = x + 0.5
    y = y + 0.5
    z = z + 0.5
    ax.voxels(arr_3d, facecolors=colors, alpha=0.5, edgecolor="none", shade=True)
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_xlim(0, arr_3d.shape[0])
    ax.set_ylim(0, arr_3d.shape[1])
    ax.set_zlim(0, arr_3d.shape[2])
    ax.grid(False)
    ax.set_box_aspect([nx, ny, nz])
    ax.set_proj_type("ortho")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis._axinfo["tick"]["inward_factor"] = 0
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.view_init(elev=30, azim=45)
    n_inj = N_injw  # Number of injectors
    n_prod = N_pr  # Number of producers
    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        line_dir = (0, 0, (nz * 2) + 7)
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 7], "blue", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="blue",
            weight="bold",
            fontsize=5,
        )
    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        line_dir = (0, 0, (nz * 2) + 5)
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 5], "g", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="g",
            weight="bold",
            fontsize=5,
        )
    for mm in range(N_injg):
        usethis = gass[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        line_dir = (0, 0, (nz * 2) + 5)
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "red", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="red",
            weight="bold",
            fontsize=5,
        )
    blue_line = mlines.Line2D([], [], color="blue", linewidth=2, label="water injector")
    green_line = mlines.Line2D(
        [], [], color="green", linewidth=2, label="oil/water/gas producer"
    )
    red_line = mlines.Line2D([], [], color="red", linewidth=2, label="gas injectors")
    ax.legend(handles=[blue_line, green_line, red_line], loc="lower left", fontsize=9)
    cbar = plt.colorbar(m, ax=ax, orientation="horizontal", shrink=0.5)
    if varii == "perm":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title(
            "Permeability Field with well locations", fontsize=12, weight="bold"
        )
    elif varii == "water PhyNeMo":
        cbar.set_label("water saturation", fontsize=12)
        ax.set_title("water saturation -PhyNeMo", fontsize=12, weight="bold")
    elif varii == "water Numerical":
        cbar.set_label("water saturation", fontsize=12)
        ax.set_title("water saturation - Numerical(Flow)", fontsize=12, weight="bold")
    elif varii == "water diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "water saturation - (Numerical(Flow) -PhyNeMo))", fontsize=12, weight="bold"
        )
    elif varii == "oil PhyNeMo":
        cbar.set_label("Oil saturation", fontsize=12)
        ax.set_title("Oil saturation -PhyNeMo", fontsize=12, weight="bold")
    elif varii == "oil Numerical":
        cbar.set_label("Oil saturation", fontsize=12)
        ax.set_title("Oil saturation - Numerical(Flow)", fontsize=12, weight="bold")
    elif varii == "oil diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "oil saturation - (Numerical(Flow) -PhyNeMo))", fontsize=12, weight="bold"
        )
    elif varii == "gas PhyNeMo":
        cbar.set_label("Gas saturation", fontsize=12)
        ax.set_title("Gas saturation -PhyNeMo", fontsize=12, weight="bold")
    elif varii == "gas Numerical":
        cbar.set_label("Gas saturation", fontsize=12)
        ax.set_title("Gas saturation - Numerical(Flow)", fontsize=12, weight="bold")
    elif varii == "gas diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "gas saturation - (Numerical(Flow) -PhyNeMo))", fontsize=12, weight="bold"
        )
    elif varii == "pressure PhyNeMo":
        cbar.set_label("pressure", fontsize=12)
        ax.set_title("Pressure -PhyNeMo", fontsize=12, weight="bold")
    elif varii == "pressure Numerical":
        cbar.set_label("pressure", fontsize=12)
        ax.set_title("Pressure -Numerical(Flow)", fontsize=12, weight="bold")
    elif varii == "pressure diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "Pressure - (Numerical(Flow) -PhyNeMo))", fontsize=12, weight="bold"
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


def process_step(
    kk,
    steppi,
    dt,
    pressure,
    effectiveuse,
    Swater,
    Soil,
    Sgas,
    nx,
    ny,
    nz,
    N_injw,
    N_pr,
    N_injg,
    injectors,
    producers,
    gass,
    fol,
    fol1,
):
    """Render and save per-timestep 3D plots for dynamic fields to disk."""
    os.chdir(fol)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk - 1, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    current_time = dt[kk]
    f_3 = plt.figure(figsize=(20, 20), dpi=200)
    look = (pressure[0, kk, :, :, :]) * effectiveuse  # [:, :, ::-1]
    ax1 = f_3.add_subplot(2, 2, 1, projection="3d")
    Plot_PhyNeMo(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "pressure PhyNeMo",
        injectors,
        producers,
        gass,
    )
    look = (Swater[0, kk, :, :, :]) * effectiveuse  # [:, :, ::-1]
    ax2 = f_3.add_subplot(2, 2, 2, projection="3d")
    Plot_PhyNeMo(
        ax2,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "water PhyNeMo",
        injectors,
        producers,
        gass,
    )
    looko = (Soil[0, kk, :, :, :]) * effectiveuse
    ax3 = f_3.add_subplot(2, 2, 3, projection="3d")
    Plot_PhyNeMo(
        ax3,
        nx,
        ny,
        nz,
        looko,
        N_injw,
        N_pr,
        N_injg,
        "oil PhyNeMo",
        injectors,
        producers,
        gass,
    )
    look = (Sgas[0, kk, :, :, :]) * effectiveuse  # [:, :, ::-1]
    ax4 = f_3.add_subplot(2, 2, 4, projection="3d")
    Plot_PhyNeMo(
        ax4,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "gas PhyNeMo",
        injectors,
        producers,
        gass,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    tita = "Timestep --" + str(current_time) + " days"
    plt.suptitle(tita, fontsize=16)
    plt.savefig("Dynamic" + str(int(kk)))
    plt.clf()
    plt.close()
    os.chdir(fol1)


def scale_array(arr):
    """Scale array magnitude to ~3 digits and return scaled array and factor."""
    max_val = np.max(np.abs(arr))
    if max_val == 0:
        return arr, 1  # No scaling needed for an array of zeroes
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


class VCAE3D(nn.Module):
    def __init__(self, latent_dim=600):
        super(VCAE3D, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )
        self.flatten_dim = None
        self.fc_mu = None
        self.fc_logvar = None
        self.decoder_input = None
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )
        self.orig_shape = None  # store original input shape

    def _compute_flatten_dim(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            return torch.prod(torch.tensor(x.shape[1:])).item(), x.shape[1:]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), *self.decoder_shape)
        x = self.decoder(x)
        # crop back to original shape
        d, h, w = self.orig_shape
        return x[:, :, :d, :h, :w]

    def forward(self, x):
        # save original shape
        self.orig_shape = x.shape[2:]  # (D,H,W)
        if self.flatten_dim is None:
            self.flatten_dim, self.decoder_shape = self._compute_flatten_dim(x)
            self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim).to(x.device)
            self.fc_logvar = nn.Linear(self.flatten_dim, self.latent_dim).to(x.device)
            self.decoder_input = nn.Linear(self.latent_dim, self.flatten_dim).to(
                x.device
            )
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """Return reconstruction + KL divergence loss for a 3D VAE."""
    recon_loss = nn.MSELoss()(recon_x, x)  # Can also use BCE Loss
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence


def Train_VCAE(
    lr,
    latent_dim,
    epochs,
    batch_size,
    device,
    dataset,
    nz,
    nx,
    ny,
    model,
    optimizer,
    scheduler,
):
    """Train a lightweight 3D VAE over single-channel grid volumes."""
    cQ = np.zeros((dataset.shape[1], 1, nz, nx, ny), dtype=np.float32)  # Pressure
    cPressini = np.zeros(
        (dataset.shape[1], 1, nx, ny, nz), dtype=np.float32
    )  # Pressure

    for k in range(dataset.shape[1]):
        use = np.reshape(dataset[:, k], (nx, ny, nz), "F")
        cPressini[k, 0, :, :, :] = use
        del use
    for i in range(nz):
        cQ[:, 0, i, :, :] = cPressini[:, 0, :, :, i]
    dataset = torch.from_numpy(cQ).to(device, torch.float32)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = vae_loss(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        logger.info(f"Epoch {epoch + 1}|{epochs}, Loss: {total_loss / len(dataloader)}")
    return model


def encode_values(inputt, nz, nx, ny, device, model):
    """Encode grid volumes via VAE encoder, returning latent vectors (T x D)."""
    cQ = np.zeros((inputt.shape[1], 1, nz, nx, ny), dtype=np.float32)  # Pressure
    cPressini = np.zeros((inputt.shape[1], 1, nx, ny, nz), dtype=np.float32)  # Pressure
    for k in range(inputt.shape[1]):
        use = np.reshape(inputt[:, k], (nx, ny, nz), "F")
        cPressini[k, 0, :, :, :] = use
        del use
    for i in range(nz):
        cQ[:, 0, i, :, :] = cPressini[:, 0, :, :, i]
    dataset = torch.from_numpy(cQ).to(device, torch.float32)
    with torch.no_grad():
        mu, logvar = model.encode(dataset)
        latent_vectors = model.reparameterize(mu, logvar)
    return latent_vectors.t()


def Make_correct(array):
    """Reorder 5D tensor from (B,C,Z,X,Y) to (B,C,X,Y,Z)."""
    new_array = np.zeros(
        (array.shape[0], array.shape[1], array.shape[3], array.shape[4], array.shape[2])
    )
    for kk in range(array.shape[0]):
        perm_big = np.zeros(
            (array.shape[1], array.shape[3], array.shape[4], array.shape[2])
        )
        for mum in range(array.shape[1]):
            mum1 = np.zeros((array.shape[3], array.shape[4], array.shape[2]))
            for i in range(array.shape[2]):
                mum1[:, :, i] = array[kk, :, :, :, :][mum, :, :, :][i, :, :]
            perm_big[mum, :, :, :] = mum1
        new_array[kk, :, :, :, :] = perm_big
    return new_array


def decode_values(inputt, device, nx, ny, nz, model):
    """Decode latent vectors via VAE and return flattened grid volumes."""
    with torch.no_grad():
        if isinstance(inputt, np.ndarray):
            inputt = torch.from_numpy(inputt).float().to(device)
        noise = inputt.t()
        generated_samples = model.decode(noise)
    genn_samples = Make_correct(generated_samples.detach().cpu().numpy())
    cQ = np.zeros((nx * ny * nz, genn_samples.shape[0]), dtype=np.float32)
    for k in range(genn_samples.shape[0]):
        use = np.reshape(genn_samples[k, 0, :, :, :], (-1, 1), "F")
        cQ[:, k] = use.ravel()
    return cQ


def load_modell(model, model_path, is_distributed, device, express, namee):
    """Load model weights from checkpoint; handle DDP 'module.' prefixes."""
    logger.info(f"🔄 Loading model from: {model_path}")
    if express == 1:
        state_dict = torch.load(model_path, map_location=device)
        if is_distributed == 1:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:] if k.startswith("module.") else k] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
    else:
        checkpoint = torch.load(model_path, map_location=device)
        if namee == "PRESSURE":
            state_dict = checkpoint["surrogate_pressure_state_dict"]
        if namee == "SWAT":
            state_dict = checkpoint["surrogate_saturation_state_dict"]

        if namee == "SOIL":
            state_dict = checkpoint["surrogate_oil_state_dict"]

        if namee == "SGAS":
            state_dict = checkpoint["surrogate_gas_state_dict"]
        if namee == "PEACEMANN":
            state_dict = checkpoint["surrogate_peacemann_state_dict"]
        # ✅ Handle Distributed Data Parallel (Remove `module.` prefix if needed)
        if is_distributed == 1:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:] if k.startswith("module.") else k] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
    return model


class FNOModel(Module):
    def __init__(
        self,
        input_dim,
        steppi,
        output_shape,
        device,
        num_layers=4,
        decoder_layers=1,
        decoder_layer_size=32,
        dimension=3,
        latent_channels=32,
        num_fno_layers=4,
        padding=8,
        num_fno_modes=16,
    ):
        super().__init__()
        self.fno = FNO(
            in_channels=input_dim,
            out_channels=output_shape * steppi,
            decoder_layers=decoder_layers,
            decoder_layer_size=decoder_layer_size,
            dimension=dimension,
            latent_channels=latent_channels,
            num_fno_layers=num_layers,
            padding=padding,
            num_fno_modes=num_fno_modes,
        ).to(torch.device(device))  # Explicit device conversion
        self.meta = type("", (), {})()  # Empty object
        self.meta.name = "fno_model"

    def forward(self, x):
        return self.fno(x)


def create_fno_model(
    input_dim,
    steppi,
    output_shape,
    device,
    num_layers=4,
    decoder_layers=1,
    decoder_layer_size=32,
    dimension=3,
    latent_channels=32,
    num_fno_layers=4,
    padding=8,
    num_fno_modes=16,
):
    """Factory for FNO-based surrogate with given IO dimensions and config."""
    if dimension not in [1, 2, 3]:
        raise ValueError(f"Invalid dimension: {dimension}. Must be 1, 2 or 3.")
    return FNOModel(
        input_dim=input_dim,
        steppi=steppi,
        output_shape=output_shape,
        device=device,
        num_layers=num_layers,
        decoder_layers=decoder_layers,
        decoder_layer_size=decoder_layer_size,
        dimension=dimension,
        latent_channels=latent_channels,
        num_fno_layers=num_fno_layers,
        padding=padding,
        num_fno_modes=num_fno_modes,
    )
