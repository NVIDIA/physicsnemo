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
import time
from collections import OrderedDict
import numpy as np
import pandas as pd
import scipy.ndimage.morphology as spndmo
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from gstools.random import MasterRNG
from gstools import SRF, Gaussian
import torch.distributed as torchdist


from utils.logging_utils import setup_logging
from utils.array_utils import Make_correct
from utils.ensemble_utils import (
    ProgressBar,
    ShowBar,
)
from utils.io_utils import Plot_PhysicsNeMo

logger = setup_logging("inverse problem")


def _is_dist_active():
    return torchdist.is_available() and torchdist.is_initialized()


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
    gas_injectors,
    active_cells_ensemble,
    Time_vector,
):
    """Render 3D voxel plots for pressure/water/oil/gas and return figure.

    Parameters capture timestep index, field tensors, grid shape and well
    locations; returns the timestep index and the created Matplotlib figure.
    """
    current_time = dt[kk]
    Time_vector[kk] = current_time
    f_3 = plt.figure(figsize=(20, 20), dpi=200)
    look = (pree[0, kk, :, :, :]) * active_cells_ensemble
    ax1 = f_3.add_subplot(2, 2, 1, projection="3d")
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
    look = (wats[0, kk, :, :, :]) * active_cells_ensemble
    ax2 = f_3.add_subplot(2, 2, 2, projection="3d")
    Plot_PhysicsNeMo(
        ax2,
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
    look = oilss[0, kk, :, :, :]
    look = look * active_cells_ensemble
    ax3 = f_3.add_subplot(2, 2, 3, projection="3d")
    Plot_PhysicsNeMo(
        ax3,
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

    look = (gasss[0, kk, :, :, :]) * active_cells_ensemble
    ax4 = f_3.add_subplot(2, 2, 4, projection="3d")
    Plot_PhysicsNeMo(
        ax4,
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
    with open(file_path) as f:
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
        raise ValueError(f"Error parsing data: {e}") from e
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


def remove_rows(matrix, indices_to_remove):
    """Remove rows by index from 2D `matrix` and return the reduced array."""
    return np.delete(matrix, indices_to_remove, axis=0)


def Localisation(c, nx, ny, nz, N, gas_injectors, producers, injectors):
    """Build Gaspari-Cohn localisation weights with well positions as centers."""
    A = np.zeros((nx, ny, nz))

    def set_well_locations(wells, array):
        for well in wells:
            i, j = well[0], well[1]
            array[i, j, :] = 1
        return array

    A = set_well_locations(
        gas_injectors, set_well_locations(producers, set_well_locations(injectors, A))
    )

    logger.info(
        "      Calculate the Euclidean distance function to the 22 producer wells"
    )
    lf = np.reshape(A, (nx, ny, nz), "F")
    young = np.zeros((int(nx * ny * nz / nz), nz))
    for j in range(nz):
        sdf = lf[:, :, j]
        (usdf, _IDX) = spndmo.distance_transform_edt(
            np.logical_not(sdf), return_indices=True
        )
        usdf = np.reshape(usdf, (int(nx * ny * nz / nz)), "F")
        young[:, j] = usdf

    sdfbig = np.reshape(young, (nx * ny * nz, 1), "F")
    z = np.abs(sdfbig).flatten()  # (nx*ny*nz,)
    r = z / float(c)  # normalised distance

    logger.info("      Computing the Gaspari-Cohn coefficient")

    # Vectorised Gaspari-Cohn — three regimes
    c0 = np.zeros_like(r)

    # Regime 1: r <= 1
    near = r <= 1.0
    rn = r[near]
    c0[near] = -0.25 * rn**5 + 0.5 * rn**4 + 0.625 * rn**3 - (5.0 / 3.0) * rn**2 + 1.0

    # Regime 2: 1 < r <= 2
    mid = (r > 1.0) & (r <= 2.0)
    rm = r[mid]
    # Guard against rm == 0 in the c/z term — but rm > 1 here so it's safe
    c0[mid] = (
        (1.0 / 12.0) * rm**5
        - 0.5 * rm**4
        + 0.625 * rm**3
        + (5.0 / 3.0) * rm**2
        - 5.0 * rm
        + 4.0
        - (2.0 / 3.0) * (1.0 / rm)
    )

    # Regime 3: r > 2 → already zero from initialisation

    # Numerical safety: tiny negatives from float arithmetic at boundaries
    c0[c0 < 0] = 0.0

    schur = c0.reshape(-1, 1)  # (nx*ny*nz, 1)
    Bsch = np.tile(schur, (1, N))  # (nx*ny*nz, N)
    yoboschur = np.ones((nx * ny * nz, N))
    yoboschur[: nx * ny * nz, :] = Bsch
    return yoboschur


def compute_tol(A):
    """Return SVD tolerance scaled by matrix size and infinity-norm."""
    max_dim = max(A.shape)  # Get the largest dimension of A
    eps_val = torch.finfo(A.dtype).eps  # Machine epsilon for A's data type
    return max_dim * eps_val * torch.linalg.norm(A, float("inf"))  # Compute tolerance


def pinvmatt(A, tol=0):
    """Return (V, X, U) where X approximates A^{-1} using truncated SVD."""
    device = A.device
    U, S1, Vt = torch.linalg.svd(A, full_matrices=False)
    if tol == 0:
        tol = torch.max(
            A.size(0) * torch.finfo(S1.dtype).eps * torch.linalg.norm(S1, float("inf"))
        )
    r1 = torch.sum(tol < S1).item()  # Don't add 1 here!
    U = U[:, :r1]
    Vt = Vt[:r1, :]
    S1 = S1[:r1]
    S_inv = torch.diag(1.0 / S1)  # Convert to diagonal matrix
    X = Vt.t() @ S_inv @ U.t()  # Correct multiplication order
    return Vt.t().to(device), X.to(device), U.to(device)


def Get_Kalman_Gain_EKI(
    Y,
    simDatafinal,
    CDd,
    alpha,
    device,
    pertubations,
    True_data,
    Ne,
    dist,
):
    """Compute ensemble Kalman update term given data covariance and alpha.

    Inputs are identical across ranks (they came from gathered/broadcast
    upstream data). The computation is deterministic, so every rank
    computes its own copy locally — no broadcast, no deadlock risk.
    """
    # --- Tensorize on every rank ---
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
    Cdd_anom = S / sqrt_Ne_1
    Cydd_anom = yprime / sqrt_Ne_1

    GDT = Cdd_anom.t() @ torch.linalg.inv(CDd).sqrt()
    inv_CDd = torch.linalg.inv(CDd).sqrt()
    Cdd = GDT.t() @ GDT
    Cyd = Cydd_anom @ GDT

    _Usig, Sig, _Vsig = torch.linalg.svd(
        Cdd + (alpha * torch.eye(CDd.shape[1], device=device)),
        full_matrices=False,
    )
    Bsig = torch.cumsum(Sig, dim=0)
    threshold = Bsig[-1] * 0.9999
    indices = torch.nonzero(Bsig >= threshold).squeeze()

    if indices.numel() > 0:
        tol = Sig[indices.item()] if indices.dim() == 0 else Sig[indices[0].item()]
        if dist.rank == 0:
            logger.info(f"Using computed tolerance from singular values: {tol.item()}")
    else:
        if dist.rank == 0:
            logger.info("using default tolerance")
        tol = compute_tol(Cdd + (alpha * torch.eye(Cdd.shape[1], device=device)))
        if dist.rank == 0:
            logger.info(f"Using default tolerance: {tol.item()}")

    _V, X, _U = pinvmatt(
        Cdd + (alpha * torch.eye(CDd.shape[1], device=device)),
        tol,
    )

    pertubations_cu = torch.as_tensor(pertubations, dtype=torch.float32, device=device)
    true_data_cu = torch.as_tensor(True_data, dtype=torch.float32, device=device)
    alpha_cu = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    tile_true_ne = true_data_cu.repeat(1, Ne).to(device)
    pertu_alpha = torch.sqrt(alpha_cu) * pertubations_cu
    factor_sum = (tile_true_ne + pertu_alpha) - simDatafinal

    update_term = Cyd @ X @ inv_CDd @ factor_sum
    update_term = torch.nan_to_num(update_term, nan=0.0, posinf=0.0, neginf=0.0)

    # if torch.cuda.is_available():
    # torch.cuda.empty_cache()

    return update_term


def Get_Kalman_Gain_ESMDA(
    Y,
    simDatafinal,
    CDd,
    alpha,
    device,
    perturbations,
    True_data,
    Ne,
    dist,
    jitter_rel=1e-6,
    svd_rtol=1e-8,
    debug=False,
):
    """Stable ensemble Kalman update:
      update_term = Cyd * (Cdd + alpha*I)^(-1) * (D - sim)

    Inputs are identical across ranks. Computation is deterministic,
    so every rank computes locally — no broadcast.
    """
    if not isinstance(simDatafinal, torch.Tensor):
        simDatafinal = torch.as_tensor(simDatafinal, dtype=torch.float32, device=device)
    if not isinstance(CDd, torch.Tensor):
        CDd = torch.as_tensor(CDd, dtype=torch.float32, device=device)
    if not isinstance(Y, torch.Tensor):
        Y = torch.as_tensor(Y, dtype=torch.float32, device=device)

    if Ne <= 1:
        raise ValueError(f"Ne must be > 1, got {Ne}")

    sqrt_Ne_1 = torch.sqrt(torch.tensor(Ne - 1, dtype=torch.float32, device=device))
    M = torch.mean(simDatafinal, dim=1, keepdim=True)
    My = torch.mean(Y, dim=1, keepdim=True)
    S = simDatafinal - M
    Yprime = Y - My

    Cdd_anom = S / sqrt_Ne_1
    Cydd_anom = Yprime / sqrt_Ne_1

    diag_mean = torch.mean(torch.diagonal(CDd)).abs()
    jitter = jitter_rel * (diag_mean + 1.0)
    CDd_j = CDd + jitter * torch.eye(CDd.shape[0], device=device, dtype=CDd.dtype)

    L = torch.linalg.cholesky(CDd_j)

    def solve_CDd(B):
        tmp = torch.linalg.solve_triangular(L, B, upper=False)
        return torch.linalg.solve_triangular(L.transpose(-1, -2), tmp, upper=True)

    W = solve_CDd(Cdd_anom)
    A = Cdd_anom.transpose(0, 1) @ W
    A = A + alpha * torch.eye(Ne, device=device, dtype=A.dtype)

    U, Svals, Vh = torch.linalg.svd(A, full_matrices=False)
    smax = torch.max(Svals)
    smin = svd_rtol * smax
    S_inv = 1.0 / torch.clamp(Svals, min=smin)
    A_inv = (Vh.transpose(-1, -2) * S_inv) @ U.transpose(-1, -2)

    pert = torch.as_tensor(perturbations, dtype=torch.float32, device=device)
    true_data = torch.as_tensor(True_data, dtype=torch.float32, device=device).reshape(
        -1, 1
    )
    tile_true = true_data.repeat(1, Ne)

    pertu_alpha = (
        torch.sqrt(torch.as_tensor(alpha, dtype=torch.float32, device=device)) * pert
    )
    factor_sum = (tile_true + pertu_alpha) - simDatafinal

    rhs = Cdd_anom.transpose(0, 1) @ solve_CDd(factor_sum)
    coeff = A_inv @ rhs

    update_term = Cydd_anom @ coeff
    update_term = torch.nan_to_num(update_term, nan=0.0, posinf=0.0, neginf=0.0)

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


def ProgressBar2(Total, Progress):
    """Return percentage string (e.g., '42%') for a given progress ratio."""
    try:
        Progress = float(Progress) / float(Total)
        if Progress >= 1.0:
            Progress = 1
            return "100%"
        return f"{round(Progress * 100, 0):.0f}%"
    except Exception:
        logger.info("")
        return "ERROR"


def process_step(
    kk,
    steppi,
    dt,
    pressure,
    active_cells_ensemble,
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
    gas_injectors,
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
    look = (pressure[0, kk, :, :, :]) * active_cells_ensemble  # [:, :, ::-1]
    ax1 = f_3.add_subplot(2, 2, 1, projection="3d")
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
    look = (Swater[0, kk, :, :, :]) * active_cells_ensemble  # [:, :, ::-1]
    ax2 = f_3.add_subplot(2, 2, 2, projection="3d")
    Plot_PhysicsNeMo(
        ax2,
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
    looko = (Soil[0, kk, :, :, :]) * active_cells_ensemble
    ax3 = f_3.add_subplot(2, 2, 3, projection="3d")
    Plot_PhysicsNeMo(
        ax3,
        nx,
        ny,
        nz,
        looko,
        N_injw,
        N_pr,
        N_injg,
        "oil PhysicsNeMo",
        injectors,
        producers,
        gas_injectors,
    )
    look = (Sgas[0, kk, :, :, :]) * active_cells_ensemble  # [:, :, ::-1]
    ax4 = f_3.add_subplot(2, 2, 4, projection="3d")
    Plot_PhysicsNeMo(
        ax4,
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
    """V C A E3 D."""

    def __init__(self, latent_dim=600):
        """Initialise 3D variational convolutional autoencoder architecture.

        Parameters
        ----------
        latent_dim : int, optional
            Dimensionality of the VAE latent space, by default 600.
        """
        super().__init__()
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
        """Compute the flattened encoder output size and spatial shape.

        Parameters
        ----------
        x : torch.Tensor
            Sample input tensor, shape (B, 1, D, H, W).

        Returns
        -------
        int
            Total number of elements after the encoder (excluding batch).
        tuple
            Spatial shape of encoder output (C, D', H', W').
        """
        with torch.no_grad():
            x = self.encoder(x)
            return torch.prod(torch.tensor(x.shape[1:])).item(), x.shape[1:]

    def reparameterize(self, mu, logvar):
        """Sample a latent vector using the reparameterisation trick.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution, shape (B, latent_dim).
        logvar : torch.Tensor
            Log-variance of the latent distribution, shape (B, latent_dim).

        Returns
        -------
        torch.Tensor
            Sampled latent vector, shape (B, latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """Encode an input volume to (mu, logvar) latent parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, 1, D, H, W).

        Returns
        -------
        mu : torch.Tensor
            Latent mean, shape (B, latent_dim).
        logvar : torch.Tensor
            Latent log-variance, shape (B, latent_dim).
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        """Decode a latent vector back to the original spatial volume shape.

        Parameters
        ----------
        z : torch.Tensor
            Latent vector, shape (B, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed volume cropped to original input shape (B, 1, D, H, W).
        """
        x = self.decoder_input(z)
        x = x.view(x.size(0), *self.decoder_shape)
        x = self.decoder(x)
        # crop back to original shape
        d, h, w = self.orig_shape
        return x[:, :, :d, :h, :w]

    def forward(self, x):
        """Run a full VAE forward pass: encode, reparameterise, decode.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, 1, D, H, W).

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            Reconstructed volume, latent mean, and latent log-variance.
        """
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


# Make_correct is imported from utils.array_utils above.


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
