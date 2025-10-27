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

# Parallel Computing

# Standard Libraries
import os
import gc
import pickle
import logging
import gzip
import re
import random
import warnings
from collections import OrderedDict

# Numerical Computing
import numpy as np
import numpy.matlib
import numpy.ma as ma

# import scipy.io as sio
import scipy.optimize.lbfgsb as lbfgsb
from scipy import interpolate
from scipy.fftpack import dct, idct

# Machine Learning
import torch
from torch.utils.data import DataLoader
from hydra.utils import to_absolute_path
from torch.utils.data import TensorDataset

# Visualization
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib import cm
import mpslib as mps
from glob import glob
from shutil import rmtree
from scipy.linalg import norm

# 📦 Local Modules
# Removed unused imports from inverse.inversion_operation_surrogate
from inverse.inversion_operation_gather import Add_marker2
# Removed unused imports from inverse.inversion_operation_misc


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        f_handler = logging.FileHandler(filename="read_vectors.log", mode="w")
        formatter = logging.Formatter(" %(asctime)s - %(levelname)s - %(message)s")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
        logger.setLevel(logging.INFO)
    warnings.filterwarnings("ignore")
    return logger


logger = setup_logging()


def linear_interp(x, xp, fp):
    contiguous_xp = xp.contiguous()
    left_indices = torch.clamp(
        torch.searchsorted(contiguous_xp, x) - 1, 0, len(contiguous_xp) - 2
    )
    denominators = contiguous_xp[left_indices + 1] - contiguous_xp[left_indices]
    close_to_zero = denominators.abs() < 1e-10
    denominators[close_to_zero] = 1.0  # or any non-zero value to avoid NaN

    interpolated_value = (
        ((fp[left_indices + 1] - fp[left_indices]) / denominators)
        * (x - contiguous_xp[left_indices])
    ) + fp[left_indices]
    return interpolated_value


def replace_nan_with_zero(tensor):
    nan_mask = torch.isnan(tensor)
    return tensor * (~nan_mask).float() + nan_mask.float() * 0.0


def intial_ensemble(Nx, Ny, Nz, N, permx):
    mps_obj = mps.mpslib()
    mps_obj = mps.mpslib(method="mps_snesim_tree")
    mps_obj.par["n_real"] = N
    k = permx  # permeability field TI
    kjenn = k
    mps_obj.ti = kjenn
    mps_obj.par["simulation_grid_size"] = (Ny, Nx, Nz)
    mps_obj.run_parallel()
    ensemble = mps_obj.sim
    ens = []
    for kk in range(N):
        temp = np.reshape(ensemble[kk], (-1, 1), "F")
        ens.append(temp)
    ensemble = np.hstack(ens)

    for f3 in glob("thread*"):
        rmtree(f3)
    for f4 in glob("*mps_snesim_tree_*"):
        os.remove(f4)
    for f4 in glob("*ti_thread_*"):
        os.remove(f4)
    return ensemble


def smoothn(
    y,
    nS0=10,
    axis=None,
    smoothOrder=2.0,
    sd=None,
    verbose=False,
    s0=None,
    z0=None,
    isrobust=False,
    W=None,
    s=None,
    MaxIter=100,
    TolZ=1e-3,
    weightstr="bisquare",
):
    if isinstance(y, ma.core.MaskedArray):  # masked array
        # is_masked = True
        mask = y.mask
        y = np.array(y)
        y[mask] = 0.0
        if W is not None and np.any(W):
            W = np.array(W)
            W[mask] = 0.0
        if sd is not None and np.any(sd):
            W = np.array(1.0 / sd**2)
            W[mask] = 0.0
            sd = None
        y[mask] = np.nan

    if sd is not None and np.any(sd):
        sd_ = np.array(sd)
        mask = sd > 0.0
        W = np.zeros_like(sd_)
        W[mask] = 1.0 / sd_[mask] ** 2
        sd = None
    if W is not None and np.any(W):
        W = W / W.max()
    sizy = y.shape
    if axis is None:
        axis = tuple(np.arange(y.ndim))
    noe = y.size  # number of elements
    if noe < 2:
        z = y
        exitflag = 0
        Wtot = 0
        return z, s, exitflag, Wtot
    if W is None or not np.any(W):
        W = np.ones(sizy)
    weightstr = weightstr.lower()
    IsFinite = np.array(np.isfinite(y)).astype(bool)
    nof = IsFinite.sum()  # number of finite elements
    W = W * IsFinite
    if any(W < 0):
        raise RuntimeError("smoothn:NegativeWeights", "Weights must all be >=0")
    else:
        pass
    isweighted = any(W != 1)
    isauto = not s
    try:
        from scipy.fftpack.realtransforms import dct, idct
    except Exception:
        z = y
        exitflag = -1
        Wtot = 0
        return z, s, exitflag, Wtot
    axis = tuple(np.array(axis).flatten())
    d = y.ndim
    Lambda = np.zeros(sizy)
    for i in axis:
        siz0 = np.ones((1, y.ndim))[0].astype(int)
        siz0[i] = sizy[i]
        Lambda = Lambda + (
            np.cos(np.pi * (np.arange(1, sizy[i] + 1) - 1.0) / sizy[i]).reshape(siz0)
        )
    Lambda = -2.0 * (len(axis) - Lambda)
    if not isauto:
        Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)
    N = sum(np.array(sizy) != 1)  # tensor rank of the y-array
    hMin = 1e-6
    hMax = 0.99
    try:
        sMinBnd = np.sqrt(
            (
                ((1 + np.sqrt(1 + 8 * hMax ** (2.0 / N))) / 4.0 / hMax ** (2.0 / N))
                ** 2
                - 1
            )
            / 16.0
        )
        sMaxBnd = np.sqrt(
            (
                ((1 + np.sqrt(1 + 8 * hMin ** (2.0 / N))) / 4.0 / hMin ** (2.0 / N))
                ** 2
                - 1
            )
            / 16.0
        )
    except Exception:
        sMinBnd = None
        sMaxBnd = None
    Wtot = W
    if isweighted:
        if z0 is not None:  # an initial guess (z0) has been provided
            z = z0
        else:
            z = y  # InitialGuess(y,IsFinite);
            z[~IsFinite] = 0.0
    else:
        z = np.zeros(sizy)
    z0 = z
    y[~IsFinite] = 0  # arbitrary values for missing y-data
    tol = 1.0
    RobustIterativeProcess = True
    RobustStep = 1
    nit = 0
    errp = 0.1
    RF = 1 + 0.75 * isweighted
    if isauto:
        try:
            xpost = np.array([(0.9 * np.log10(sMinBnd) + np.log10(sMaxBnd) * 0.1)])
        except Exception:
            xpost = np.array([100.0])
    else:
        xpost = np.array([np.log10(s)])
    while RobustIterativeProcess:
        aow = sum(Wtot) / noe  # 0 < aow <= 1
        while tol > TolZ and nit < MaxIter:
            if verbose:
                logger.info("tol %s nit %s", tol, nit)
            nit = nit + 1
            DCTy = dctND(Wtot * (y - z) + z, f=dct)
            if isauto and not np.remainder(np.log2(nit), 1):
                if not s0:
                    ss = np.arange(nS0) * (1.0 / (nS0 - 1.0)) * (
                        np.log10(sMaxBnd) - np.log10(sMinBnd)
                    ) + np.log10(sMinBnd)
                    g = np.zeros_like(ss)
                    for i, p in enumerate(ss):
                        g[i] = gcv(
                            p,
                            Lambda,
                            aow,
                            DCTy,
                            IsFinite,
                            Wtot,
                            y,
                            nof,
                            noe,
                            smoothOrder,
                        )
                    xpost = [ss[g == g.min()]]
                else:
                    xpost = [s0]
                xpost, f, d = lbfgsb.fmin_l_bfgs_b(
                    gcv,
                    xpost,
                    fprime=None,
                    factr=1e7,
                    approx_grad=True,
                    bounds=[(np.log10(sMinBnd), np.log10(sMaxBnd))],
                    args=(Lambda, aow, DCTy, IsFinite, Wtot, y, nof, noe, smoothOrder),
                )
            s = 10 ** xpost[0]
            s0 = xpost[0]
            Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)
            z = RF * dctND(Gamma * DCTy, f=idct) + (1 - RF) * z
            tol = isweighted * norm(z0 - z) / norm(z)
            z0 = z  # re-initialization
        exitflag = nit < MaxIter
        if isrobust:  # -- Robust Smoothing: iteratively re-weighted process
            h = np.sqrt(1 + 16.0 * s)
            h = np.sqrt(1 + h) / np.sqrt(2) / h
            h = h**N
            Wtot = W * RobustWeights(y - z, IsFinite, h, weightstr)
            isweighted = True
            tol = 1
            nit = 0
            RobustStep = RobustStep + 1
            RobustIterativeProcess = RobustStep < 3
        else:
            RobustIterativeProcess = False  # stop the whole process
    if isauto:
        if abs(np.log10(s) - np.log10(sMinBnd)) < errp:
            warning(
                "MATLAB:smoothn:SLowerBound",
                [
                    "s = %.3f " % (s)
                    + ": the lower bound for s "
                    + "has been reached. Put s as an input variable if required."
                ],
            )
        elif abs(np.log10(s) - np.log10(sMaxBnd)) < errp:
            warning(
                "MATLAB:smoothn:SUpperBound",
                [
                    "s = %.3f " % (s)
                    + ": the upper bound for s "
                    + "has been reached. Put s as an input variable if required."
                ],
            )
    return z, s, exitflag, Wtot


def warning(s1, s2):
    logger.info(s1)
    logger.info(s2[0])


def gcv(p, Lambda, aow, DCTy, IsFinite, Wtot, y, nof, noe, smoothOrder):
    s = 10**p
    Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)
    if aow > 0.9:  # aow = 1 means that all of the data are equally weighted
        RSS = norm(DCTy * (Gamma - 1.0)) ** 2
    else:
        yhat = dctND(Gamma * DCTy, f=idct)
        RSS = norm(np.sqrt(Wtot[IsFinite]) * (y[IsFinite] - yhat[IsFinite])) ** 2
    TrH = sum(Gamma)
    GCVscore = RSS / float(nof) / (1.0 - TrH / float(noe)) ** 2
    return GCVscore


def sort_key(s):
    return int(re.search(r"\d+", s).group())


def RobustWeights(r, mask_I, h, wstr):
    MAD = np.median(abs(r[mask_I] - np.median(r[mask_I])))  # median absolute deviation
    u = abs(r / (1.4826 * MAD) / np.sqrt(1 - h))  # studentized residuals
    if wstr == "cauchy":
        c = 2.385
        W = 1.0 / (1 + (u / c) ** 2)  # Cauchy weights
    elif wstr == "talworth":
        c = 2.795
        W = u < c  # Talworth weights
    else:
        c = 4.685
        W = (1 - (u / c) ** 2) ** 2.0 * ((u / c) < 1)  # bisquare weights
    W[np.isnan(W)] = 0
    return W


def InitialGuess(y, mask_valid):
    if any(~mask_valid):
        try:
            from scipy.ndimage import distance_transform_edt

            L = distance_transform_edt(1 - mask_valid)
            z = y
            z[~mask_valid] = y[L[~mask_valid]]
        except (ImportError, IndexError, ValueError):
            # Handle specific exceptions: ImportError if scipy not available,
            # IndexError/ValueError if array operations fail
            z = y
            z[~mask_valid] = np.mean(y[mask_valid])
    else:
        z = y
    z = dctND(z, f=dct)
    k = np.array(z.shape)
    m = np.ceil(k / 10) + 1
    d = []
    for i in np.arange(len(k)):
        d.append(np.arange(m[i], k[i]))
    d = np.array(d).astype(int)
    z[d] = 0.0
    z = dctND(z, f=idct)
    return z


def dctND(data, f=dct):
    nd = len(data.shape)
    if nd == 1:
        return f(data, norm="ortho", type=2)
    elif nd == 2:
        return f(f(data, norm="ortho", type=2).T, norm="ortho", type=2).T
    elif nd == 3:
        return f(
            f(f(data, norm="ortho", type=2, axis=0), norm="ortho", type=2, axis=1),
            norm="ortho",
            type=2,
            axis=2,
        )
    elif nd == 4:
        return f(
            f(
                f(f(data, norm="ortho", type=2, axis=0), norm="ortho", type=2, axis=1),
                norm="ortho",
                type=2,
                axis=2,
            ),
            norm="ortho",
            type=2,
            axis=3,
        )


def peaks(n):
    xp = np.arange(n)
    [x, y] = np.meshgrid(xp, xp)
    z = np.zeros_like(x).astype(float)
    for i in np.xrange(n / 5):
        x0 = random() * n
        y0 = random() * n
        sdx = random() * n / 4.0
        sdy = sdx
        c = random() * 2 - 1.0
        f = np.exp(
            -(((x - x0) / sdx) ** 2)
            - ((y - y0) / sdy) ** 2
            - ((x - x0) / sdx) * ((y - y0) / sdy) * c
        )
        f *= random()
        z += f
    return z


def Add_marker(plt, XX, YY, locc):
    for i in range(locc.shape[0]):
        a = locc[i, :]
        xloc = int(a[0])
        yloc = int(a[1])
        if a[2] == 2:
            plt.scatter(
                XX.T[xloc - 1, yloc - 1] + 0.5,
                YY.T[xloc - 1, yloc - 1] + 0.5,
                s=100,
                marker="^",
                color="white",
            )
        else:
            plt.scatter(
                XX.T[xloc - 1, yloc - 1] + 0.5,
                YY.T[xloc - 1, yloc - 1] + 0.5,
                s=100,
                marker="v",
                color="white",
            )


def Plot_PhyNeMo(
    ax, nx, ny, nz, Truee, N_injw, N_pr, N_injg, varii, injectors, producers, gass
):
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


def honour2(filcc, nx, ny, nz, N_ens, High_K, Low_K, High_P, Low_P, effec):
    High_K = High_K.item()
    Low_K = Low_K.item()
    filcc["PERM"] = np.clip(filcc["PERM"], Low_K, High_K)
    filcc["PORO"] = np.clip(filcc["PORO"], Low_P, High_P)
    return filcc


def funcGetDataMismatch(sim_data, measurement):
    is_torch = isinstance(sim_data, torch.Tensor)
    if not is_torch:
        sim_data = np.asarray(sim_data)
        measurement = np.asarray(measurement)
        reshape_fn = np.reshape
        sqrt_fn = np.sqrt
        sum_fn = np.sum
        mean_fn = np.mean
        std_fn = np.std
        zeros_fn = np.zeros
    else:
        measurement = measurement.reshape(-1, 1)
        reshape_fn = torch.reshape
        sqrt_fn = torch.sqrt
        sum_fn = torch.sum
        mean_fn = torch.mean
        std_fn = torch.std

        def zeros_fn(shape, **kwargs):
            return torch.zeros(shape, device=sim_data.device, dtype=sim_data.dtype)

    ne = sim_data.shape[1]
    obj_real = zeros_fn((ne, 1))
    for j in range(ne):
        noww = reshape_fn(sim_data[:, j], (-1, 1))
        obj_real[j] = sqrt_fn(sum_fn((noww - measurement) ** 2)) / measurement.shape[0]
    obj = mean_fn(obj_real).item()
    obj_std = std_fn(obj_real).item()
    return obj, obj_std, obj_real


def pinvmatt(A, tol=0):
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


class MinMaxScalerVectorized(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, tensor):
        tensor = torch.stack(tensor)
        a, b = self.feature_range
        dist = tensor.max(dim=0, keepdim=True)[0] - tensor.min(dim=0, keepdim=True)[0]
        dist[dist == 0.0] = 1.0
        scale = 1.0 / dist
        tensor.mul_(scale).sub_(tensor.min(dim=0, keepdim=True)[0])
        tensor.mul_(b - a).add_(a)
        return tensor


def load_data_numpy_2(inn, out, ndata, batch_size):
    x_data = inn
    y_data = out
    logger.info(f"xtrain_data: {x_data.shape}")
    logger.info(f"ytrain_data: {y_data.shape}")
    data_tuple = (torch.FloatTensor(x_data), torch.FloatTensor(y_data))
    data_loader = DataLoader(
        TensorDataset(*data_tuple), batch_size=batch_size, shuffle=True, drop_last=True
    )
    return data_loader


def rescale_linear(array, new_min, new_max):
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def rescale_linear_numpy_pytorch(array, new_min, new_max, minimum, maximum):
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def rescale_linear_pytorch_numpy(array, new_min, new_max, minimum, maximum):
    m = (maximum - minimum) / (new_max - new_min)
    b = minimum - m * new_min
    return m * array + b


def Equivalent_time(tim1, max_t1, tim2, max_t2):
    tk2 = tim1 / max_t1
    tc2 = np.arange(0.0, 1 + tk2, tk2)
    tc2[tc2 >= 1] = 1
    tc2 = tc2.reshape(-1, 1)  # reference scaled to 1
    tc2r = np.arange(0.0, max_t1 + tim1, tim1)
    tc2r[tc2r >= max_t1] = max_t1
    tc2r = tc2r.reshape(-1, 1)  # reference original
    func = interpolate.interp1d(tc2r.ravel(), tc2.ravel())
    tc2rr = np.arange(0.0, max_t2 + tim2, tim2)
    tc2rr[tc2rr >= max_t2] = max_t2
    tc2rr = tc2rr.reshape(-1, 1)  # reference original
    ynew = func(tc2rr.ravel())
    return ynew


be_verbose = False


def Get_Time(nx, ny, nz, steppi, steppi_indices, N):
    try:
        with gzip.open(to_absolute_path("../data/data_train.pkl.gz"), "rb") as f2:
            mat = pickle.load(f2)
    except (pickle.PickleError, EOFError, FileNotFoundError) as e:
        logger.error(f"Error loading pickle file: {e}")
        raise
    X_data1 = mat
    del mat
    gc.collect()
    # mat = sio.loadmat(to_absolute_path("../data/conversions.mat"))
    Time = X_data1["Time"]  # * mat ["maxT"]
    np_array2 = np.zeros((Time.shape[1]))
    for mm in range(Time.shape[1]):
        np_array2[mm] = Time[0, mm, 0, 0, 0]
    Timee = []
    for k in range(N):
        check = np.ones((nx, ny, nz), dtype=np.float16)
        unie = []
        for zz in range(len(np_array2)):
            aa = np_array2[zz] * check
            unie.append(aa)
        Time = np.stack(unie, axis=0)
        Timee.append(Time)
    Timee = np.stack(Timee, axis=0)
    return Timee


def Get_fault(filename):
    with open(filename, "r") as file:
        injector_gas = set()  # Set to collect gas injector well names
        start_collecting_welspecs = False
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith("--"):
                continue
            if "MULTFLT" in stripped_line:
                start_collecting_welspecs = True
                continue
            if start_collecting_welspecs and stripped_line.startswith("/"):
                start_collecting_welspecs = False
                continue
            if start_collecting_welspecs:
                parts = stripped_line.split()
                fault_name = parts[0].strip("'")
                injector_gas.add((fault_name))
    return sorted(injector_gas)


def read_faults(filename, well_names):
    with open(filename, "r") as file:
        start_collecting = False
        data = []  # List to collect all entries
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith("--"):
                continue
            if "FAULTS" in stripped_line:
                start_collecting = True
                continue
            if start_collecting and stripped_line.startswith("/"):
                start_collecting = False
                continue
            if start_collecting and stripped_line:
                parts = stripped_line.split()
                # Strip quotes from the well name
                well_name = parts[0].strip("'")
                # print("Parts found:", parts)
                if well_name in well_names:
                    # Extract and add the tuple of the required columns to the list
                    data.append(
                        (
                            well_name,
                            parts[1],
                            parts[2],
                            parts[3],
                            parts[4],
                            parts[5],
                            parts[6],
                        )
                    )
    return data


def assign_faults(well_indices, nx, ny, nz, well_amount, data):
    faultm = np.ones((nx, ny, nz), dtype=np.float16)
    unique_well_names = OrderedDict()
    for idx, tuple_entry in enumerate(well_indices):
        well_name = tuple_entry[0]
        if well_name not in unique_well_names:
            unique_well_names[well_name] = len(unique_well_names)
    well_value_map = {
        well_name: data[idx] for idx, well_name in enumerate(unique_well_names)
    }

    # Assign fault values to the grid based on well indices and mapped values
    for well_name, average_value in well_value_map.items():
        # Find all tuples corresponding to this well name to update faultm accordingly
        entries_for_well = [t for t in well_indices if t[0] == well_name]

        # Assign the average value to each specified index in faultm
        for _, i_idx, i1_idx, j_idx, j1_idx, k_idx, k1_idx in entries_for_well:
            faultm[
                int(i_idx) - 1 : int(i1_idx),
                int(j_idx) - 1 : int(j1_idx),
                int(k_idx) - 1 : int(k1_idx),
            ] = average_value

    return faultm


def Get_falt(source_dir, nx, ny, nz, floatz, N, filename_fault, FAULT_INCLUDE):
    Fault = np.ones((nx, ny, nz), dtype=np.float16)
    flt = []

    # Loop over each ensemble member
    for k in range(N):
        # Extract fault parameter values for the current ensemble member
        floatts = floatz[:, k]

        # Get the fault template from the specified file
        fault_temp = Get_fault(os.path.join(source_dir, FAULT_INCLUDE))

        # Read fault data from the specified file using the fault template
        fault_data = read_faults(filename_fault, fault_temp)  # OIl

        # Assign faults to the grid using the read fault data and parameter values
        Fault = assign_faults(fault_data, nx, ny, nz, fault_temp, floatts)

        # Append the generated fault data to the list
        flt.append(Fault)

    # Stack the list of fault data into a single numpy array and add an additional dimension
    flt = np.stack(flt, axis=0)[:, None, :, :, :]

    # Return the stacked fault data
    return np.stack(flt, axis=0)


def fit_operation(tensor, target_min, target_max, tensor_min, tensor_max):
    # Rescale between target min and target max
    rescaled_tensor = tensor / tensor_max
    return rescaled_tensor


def ensemble_pytorch(
    param,
    nx,
    ny,
    nz,
    Ne,
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
):
    if "FAULT" in input_variables:
        Fault = np.ones((nx, ny, nz), dtype=np.float16)
        flt = []
        for k in range(Ne):
            floatts = param["FAULT"][:, k]
            fault_temp = Get_fault(cfg.custom.FAULT_INCLUDEE)
            fault_data = read_faults(
                to_absolute_path(cfg.custom.FAULT_DATA), fault_temp
            )  # OIl
            Fault = assign_faults(fault_data, nx, ny, nz, fault_temp, floatts)
            flt.append(Fault)
        flt = np.stack(flt, axis=0)[:, None, :, :, :]
        faultz = np.stack(flt, axis=0)

    ini_ensembles = {}
    if "PERM" in input_variables:
        ini_ensembles["perm"] = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    if "PORO" in input_variables:
        ini_ensembles["poro"] = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    if "PINI" in input_variables:
        ini_ensembles["pini"] = cfg.custom.PROPS.initial_pressure * np.ones(
            (Ne, 1, nz, nx, ny), dtype=np.float32
        )  # * effea[None,None,:,:,:]
    if "SINI" in input_variables:
        ini_ensembles["sini"] = cfg.custom.PROPS.initial_water_saturation * np.ones(
            (Ne, 1, nz, nx, ny), dtype=np.float32
        )  # * effea[None,None,:,:,:]
    if "FAULT" in input_variables:
        ini_ensembles["fault"] = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)

    for kk in range(Ne):
        if "PERM" in input_variables:
            a = np.reshape(param["PERM"][:, kk], (nx, ny, nz), "F")  # * effective
        if "PORO" in input_variables:
            a1 = np.reshape(param["PORO"][:, kk], (nx, ny, nz), "F")  # * effective

        for my in range(nz):
            if "PERM" in input_variables:
                ini_ensembles["perm"][kk, 0, my, :, :] = a[:, :, my]  # Permeability
            if "PORO" in input_variables:
                ini_ensembles["poro"][kk, 0, my, :, :] = a1[:, :, my]  # Porosity
            if "FAULT" in input_variables:
                ini_ensembles["fault"][kk, 0, my, :, :] = faultz[
                    kk, 0, :, :, my
                ]  # fault

    # Initial_pressure
    if "PINI" in input_variables:
        ini_ensembles["pini"] = fit_operation(
            ini_ensembles["pini"], target_min, target_max, minP, maxP
        )

    # Permeability
    if "PERM" in input_variables:
        ini_ensembles["perm"] = fit_operation(
            ini_ensembles["perm"], target_min, target_max, minK, maxK
        )

    # Prepare the dictionary dynamically
    inn = {}
    if "PERM" in input_variables:
        inn["perm"] = torch.from_numpy(ini_ensembles["perm"]).to(
            device, dtype=torch.float32
        )
    if "PORO" in input_variables:
        inn["poro"] = torch.from_numpy(ini_ensembles["poro"]).to(
            device, dtype=torch.float32
        )
    if "PINI" in input_variables:
        inn["pini"] = torch.from_numpy(ini_ensembles["pini"]).to(
            device, dtype=torch.float32
        )
    if "SINI" in input_variables:
        inn["sini"] = torch.from_numpy(ini_ensembles["sini"]).to(
            device, dtype=torch.float32
        )
    if "FAULT" in input_variables:
        inn["fault"] = torch.from_numpy(ini_ensembles["fault"]).to(
            device, dtype=torch.float32
        )

    return inn


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
