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
                    SEQUENTIAL FORWARD PROCESSING UTILITIES
=====================================================================

This module provides forward processing utilities for sequential FVM
surrogate model comparisons. It includes functions for data processing,
simulation, and analysis.

Key Features:
- Data type definitions for simulation data
- Forward processing functions
- Data transformation and interpolation
- Simulation utilities

Usage:
    from compare.sequential.misc_forward import (
        Remove_folder,
        linear_interp,
        replace_nan_with_zero,
        interp_torch
    )

@Author : Clement Etienam
"""

import os
from shutil import rmtree
from copy import copy
from collections import OrderedDict

import numpy as np
import pandas as pd
import numpy.linalg as LA
from scipy.stats import rankdata, norm
from scipy import interpolate
import torch
import matplotlib.pyplot as plt

from utils.logging_utils import setup_logging
from utils.array_utils import linear_interp
from utils.path_utils import pushd


def split_matrix(matrix, sizee):
    """Split a matrix into equal sub-arrays along axis 0.

    Parameters
    ----------
    matrix : np.ndarray
        Input 2-D array to be split.
    sizee : int
        Number of equal sub-arrays to produce.

    Returns
    -------
    list of np.ndarray
        List of sub-arrays resulting from the split.
    """
    return np.split(matrix, sizee, axis=0)


type_dict = {
    b"INTE": "i",
    b"CHAR": "8s",
    b"REAL": "f",
    b"DOUB": "d",
    b"LOGI": "4s",
    b"MESS": "?",
}

ecl_extensions = [
    ".DATA",
    ".DBG",
    ".ECLEND",
    ".EGRID",
    ".FEGRID",
    ".FGRID",
    ".FINIT",
    ".FINSPEC",
    ".FRFT",
    ".FRSSPEC",
    ".FSMSPEC",
    ".FUNRST",
    ".FUNSMRY",
    ".GRID",
    ".INIT",
    ".INSPEC",
    ".MSG",
    ".PRT",
    ".RFT",
    ".RSM",
    ".RSSPEC",
    ".SMSPEC",
    ".UNRST",
    ".UNSMRY",
    ".dbprtx",
]

dynamic_props = [
    "SEQNUM",
    "PRESSURE",
    "SWAT",
    "SGAS",
    "SOIL",
    "RS",
    "RV",
    "RSSAT",
    "RVSAT",
    "STATES",
    "OWC",
    "OGC",
    "GWC",
    "EOWC",
    "EOGC",
    "OILAPI",
    "SDENO",
    "FIPOIL",
    "RFIPOIL",
    "FIPGAS",
    "RFIPGAS",
    "FIPWAT",
    "RFIPWAT",
    "SFIPOIL",
    "SFIPGAS",
    "SFIPWAT",
    "SFIPPLY",
    "RFIPPLY",
    "SFIPSAL",
    "RFIPSAL",
    "SFIPSOL",
    "SFIPGGI",
    "RFIPOIL",
    "RFIPGAS",
    "RFIPWAT",
    "RFIPSOL",
    "RFIPGGI",
    "OIL-POTN",
    "GAS-POTN",
    "WAT-POTN",
    "POLYMER",
    "PADS",
    "PLYTRRFA",
    "POLYMAX",
    "SALT",
    "TEMP",
    "XMF",
    "YMF",
    "ZMF",
    "SSOL",
    "PBUB",
    "PDEW",
    "SURFACT",
    "SURFADS",
    "SURFMAX",
    "SURFCNM",
    "SURFST",
    "GGI",
    "WAT-PRES",
    "GAS-PRES",
    "OIL-VISC",
    "WAT-VISC",
    "GAS-VISC",
    "OIL-DEN",
    "WAT-DEN",
    "GAS-DEN",
    "DRAINAGE",
    "DRAINMIN",
    "PCOW",
    "PCOG",
    "1OVERBO",
    "1OVERBW",
    "1OVERBG",
    "POT_CORR",
    "OILKR",
    "WATKR",
    "GASKR",
    "HYDH",
    "HYDHFW",
    "PORV",
    "RPORV",
    "FOAM",
    "FOAMADS",
    "FOAMMAX",
    "FOAMDCY",
    "FOAMCNM",
    "FOAM_HL",
    "FOAMMOB",
    "ALKALINE",
    "ALKADS",
    "ALKMAX",
    "STMALK",
    "SFADALK",
    "PLADALK",
    "PADMAX",
    "CATSURF",
    "CATROCK",
    "ESALSUR",
    "ESALPLY",
    "COALGAS",
    "COALSOLV",
    "GASSATC",
    "MLANG",
    "MLANGSLV",
    "SWMIN",
    "SWMAX",
    "ISTHW",
    "SOMAX",
    "ISTHG",
    "SGMIN",
    "SGMAX",
    "PRESROCC",
    "CNV_OIL",
    "CNV_WAT",
    "CNV_GAS",
    "CNV_PLY",
    "TRANEXX",
    "TRANEXY",
    "TRANEXZ",
    "EXCAVNUM",
    "CNV_SAL",
    "CNV_SOL",
    "CNV_GGI",
    "CNV_DPRE",
    "CNV_DWAT",
    "CNV_DGAS",
    "CNV_DPLY",
    "CNV_DSAL",
    "CNV_DSOL",
    "CNV_DGGI",
    "CONV_VBR",
    "CONV_PRU",
    "CONV_NEW",
    "FLOOILI+",
    "FLOOILJ+",
    "FLOOILK+",
    "FLOGASI+",
    "FLOGASJ+",
    "FLOGASK+",
    "FLOWATI+",
    "FLOWATJ+",
    "FLOWATK+",
    "FLROILI+",
    "FLROILJ+",
    "FLROILK+",
    "FLRGASI+",
    "FLRGASJ+",
    "FLRGASK+",
    "FLRWATI+",
    "FLRWATJ+",
    "FLRWATK+",
    "VOILI+",
    "VOILJ+",
    "VOILK+",
    "VGASI+",
    "VGASJ+",
    "VGASK+",
    "VWATI+",
    "VWATJ+",
    "VWATK+",
    "FLOOILN+",
    "FLOGASN+",
    "FLOWATN+",
    "FLOOILL+",
    "FLOGASL+",
    "FLOWATL+",
    "FLOOILA+",
    "FLOGASA+",
    "FLOWATA+",
    "FLROILN+",
    "FLRGASN+",
    "FLRWATN+",
    "FLROILL+",
    "FLRGASL+",
    "FLRWATL+",
    "FLROILA+",
    "FLRGASA+",
    "FLRWATA+",
]


ecl_vectors = [
    "COPR",
    "COPT",
    "CWFR",
    "CWIR",
    "CWPR",
    "CWPT",
    "FGIR",
    "FGIT",
    "FGLIR",
    "FGOR",
    "FGORH",
    "FGPR",
    "FGPT",
    "FLPR",
    "FLPT",
    "FMCTP",
    "FMWWO",
    "FMWWT",
    "FODEN",
    "FOE",
    "FOIP",
    "FOPR",
    "FOPRF",
    "FOPRH",
    "FOPRS",
    "FOPT",
    "FOPTH",
    "FPR",
    "FVIR",
    "FVIT",
    "FVPR",
    "FVPT",
    "FWCT",
    "FWCTH",
    "FWIP",
    "FWIR",
    "FWIT",
    "FWPR",
    "FWPT",
    "GGOR",
    "GGPR",
    "GGPT",
    "GOPR",
    "GOPT",
    "GVIR",
    "GVIT",
    "GVPR",
    "GVPT",
    "GWCT",
    "GWIR",
    "GWPR",
    "MSUMLINS",
    "RGPV",
    "RHPV",
    "ROE",
    "ROEW",
    "ROPV",
    "ROSAT",
    "RPR",
    "RRPV",
    "RWPV",
    "TCPU",
    "TIME",
    "WBHP",
    "WBHPH",
    "WBP",
    "WBP4",
    "WBP9",
    "WGIR",
    "WGIT",
    "WGLIR",
    "WGOR",
    "WGORH",
    "WGPR",
    "WGPRH",
    "WGPTH",
    "WLPR",
    "WLPRH",
    "WLPT",
    "WLPTH",
    "WMCON",
    "WMCTL",
    "WOPR",
    "WOPRH",
    "WOPT",
    "WOPTH",
    "WPI",
    "WTHP",
    "WTICIW1",
    "WTICIW2",
    "WTIRIW1",
    "WTIRIW2",
    "WTPCIW1",
    "WTPCIW2",
    "WTPRIW1",
    "WTPRIW2",
    "WWCT",
    "WWCTH",
    "WWIR",
    "WWIRH",
    "WWIT",
    "WWITH",
    "WWPR",
    "WWPRH",
    "WWPT",
    "WWPTH",
    "YEARS",
]

static_props = [
    "DEPTH",
    "DX",
    "DR",
    "DY",
    "DTHETA",
    "DZ",
    "PORO",
    "PERMX",
    "PERMR",
    "PERMI",
    "PERMY",
    "PERMTHT",
    "PERMJ",
    "PERMZ",
    "PERMK",
    "MULTX",
    "MULTR",
    "MULTI",
    "MULTY",
    "MULTTHT",
    "MULTJ",
    "MULTZ",
    "MULTK",
    "TRANX",
    "TRANR",
    "TRANI",
    "TRANY",
    "TRANTHT",
    "TRANJ",
    "TRANZ",
    "TRANK",
    "DIFFMX",
    "DIFFMR",
    "DIFFMI",
    "DIFFMY",
    "DIFFMTHT",
    "DIFFMJ",
    "DIFFMZ",
    "DIFFMK",
    "DIFFX",
    "DIFFR",
    "DIFFI",
    "DIFFY",
    "DIFFTHT",
    "DIFFJ",
    "DIFFZ",
    "DIFFK",
    "DIFFTX",
    "DIFFTR",
    "DIFFTI",
    "DIFFTY",
    "DIFFTTHT",
    "DIFFTJ",
    "DIFFTZ",
    "DIFFTK",
    "HEATTX",
    "HEATTR",
    "HEATTY",
    "HEATTTHT",
    "MLANGI",
    "GASSATC",
    "MLNGSLVI",
    "MLANG",
    "GASSATC",
    "MLANGSLV",
    "AQUIFERN",
    "DOMAINS",
    "ENDNUM",
    "EQLNUM",
    "FIPNUM",
    "FLUXNUM",
    "KRO",
    "KRORW",
    "KRW",
    "KRWR",
    "MINPVV",
    "MULTNUM",
    "MULTPV",
    "MULTX",
    "MULTX-",
    "MULTY",
    "MULTY-",
    "MULTZ",
    "MULTZ-",
    "NTG",
    "OPERNUM",
    "PCW",
    "PORV",
    "PVTNUM",
    "SATNUM",
    "SOWCR",
    "SWATINIT",
    "SWCR",
    "SWL",
    "SWLPC",
    "SWU",
    "TOPS",
    "TRANNNC",
]
SUPPORTED_DATA_TYPES = {
    "INTE": (4, "i", 1000),
    "REAL": (4, "f", 1000),
    "LOGI": (4, "i", 1000),
    "DOUB": (8, "d", 1000),
    "CHAR": (8, "8s", 105),
    "MESS": (8, "8s", 105),
    "C008": (8, "8s", 105),
}


def remove_folders(N_ens, base_name):
    """
    Remove multiple folders with error handling.

    Args:
        N_ens: Number of folders to remove
        base_name: Base name for folders (folders will be base_name0, base_name1, etc.)
    """
    logger = setup_logging(__name__)
    for j in range(N_ens):
        folder_path = base_name + str(j)
        try:
            if os.path.exists(folder_path):
                rmtree(folder_path)
                logger.info(f"Successfully removed {folder_path}")
            else:
                logger.info(f"Folder {folder_path} does not exist, skipping")
        except OSError as e:
            logger.info(f"Error removing {folder_path}: {e}")


# linear_interp is imported from utils.array_utils above.


def replace_nan_with_zero(tensor):
    """Replace all NaN values in a tensor with zero.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor potentially containing NaN values.

    Returns
    -------
    torch.Tensor
        Tensor with all NaN entries replaced by 0.0.
    """
    nan_mask = torch.isnan(tensor)
    return tensor * (~nan_mask).float() + nan_mask.float() * 0.0


def interp_torch(cuda, reference_matrix1, reference_matrix2, tensor1):
    """Interpolate a tensor using reference matrices via chunked linear interpolation.

    Parameters
    ----------
    cuda : torch.device
        Device identifier (e.g. ``torch.device('cuda')`` or ``'cpu'``).
    reference_matrix1 : torch.Tensor
        Known x-coordinates used as the interpolation grid.
    reference_matrix2 : torch.Tensor
        Known y-values corresponding to `reference_matrix1`.
    tensor1 : torch.Tensor
        Query tensor whose values are to be interpolated.

    Returns
    -------
    list of torch.Tensor
        Processed chunks with interpolated values.
    """
    chunk_size = 1
    chunks = torch.chunk(tensor1, chunks=chunk_size, dim=0)
    processed_chunks = []
    for start_idx in range(chunk_size):
        interpolated_chunk = linear_interp(
            chunks[start_idx], reference_matrix1, reference_matrix2
        )
        processed_chunks.append(interpolated_chunk)
    return processed_chunks


def write_rsm(data, Time, Name, well_names, N_pr):
    """Write reservoir simulation results to an Excel workbook.

    Parameters
    ----------
    data : np.ndarray
        2-D array of well production rates (rows = timesteps, cols = well/group).
    Time : np.ndarray
        1-D array of simulation time values in days.
    Name : str
        Output filename (without extension); saved as ``<Name>.xlsx``.
    well_names : list of str
        Names of producer wells forming the column headers.
    N_pr : int
        Number of producer wells; used to set column width.

    Returns
    -------
    None
        Writes the Excel file to disk; no return value.
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


# sort_key is imported from utils.array_utils above.


def interpolatebetween(xtrain, cdftrain, xnew):
    """Interpolate CDF values column-wise from training data to new query points.

    Parameters
    ----------
    xtrain : np.ndarray
        2-D array of training x-coordinates, shape ``(n_train, n_cols)``.
    cdftrain : np.ndarray
        2-D array of CDF values at training points, shape ``(n_train, n_cols)``.
    xnew : np.ndarray
        2-D array of query points, shape ``(n_new, n_cols)``.

    Returns
    -------
    np.ndarray
        Interpolated CDF values at query points, shape ``(n_new, n_cols)``.
    """
    numrows1 = len(xnew)
    numcols = len(xnew[0])
    norm_cdftest2 = np.zeros((numrows1, numcols))
    for i in range(numcols):
        f = interpolate.interp1d((xtrain[:, i]), cdftrain[:, i], kind="linear")
        cdftest = f(xnew[:, i])
        norm_cdftest2[:, i] = np.ravel(cdftest)
    return norm_cdftest2


def gaussianizeit(input1):
    """Transform each column of an array to standard normal scores via rank mapping.

    Parameters
    ----------
    input1 : np.ndarray
        2-D array of input values, shape ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        Array of the same shape with each column mapped to normal quantiles.
    """
    numrows1 = len(input1)
    numcols = len(input1[0])
    newbig = np.zeros((numrows1, numcols))
    for i in range(numcols):
        input11 = input1[:, i]
        newX = norm.ppf(rankdata(input11) / (len(input11) + 1))
        newbig[:, i] = newX.T
    return newbig


def best_fit(X, Y):
    """Compute the least-squares best-fit line through paired data points.

    Parameters
    ----------
    X : array-like of float
        Independent variable values.
    Y : array-like of float
        Dependent variable values corresponding to `X`.

    Returns
    -------
    a : float
        Y-intercept of the best-fit line.
    b : float
        Slope of the best-fit line.
    """
    xbar = sum(X) / len(X)
    ybar = sum(Y) / len(Y)
    n = len(X)  # or len(Y)
    numer = sum([xi * yi for xi, yi in zip(X, Y, strict=False)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
    close_to_zero = abs(denum) < 1e-10
    denum[close_to_zero] = 1.0
    b = numer / denum
    a = ybar - b * xbar
    logger = setup_logging(__name__)
    logger.info(f"best fit line:\ny = {a:.2f} + {b:.2f}x")
    return a, b


def performance_plot_cost(CCR, Trued, stringg, training_master, oldfolder):
    """Generate scatter plots comparing model predictions to true values and compute metrics.

    Parameters
    ----------
    CCR : np.ndarray
        2-D array of model predictions, shape ``(n_samples, n_outputs)``.
    Trued : np.ndarray
        2-D array of true target values, shape ``(n_samples, n_outputs)``.
    stringg : str
        Base filename (without extension) for the saved plot image.
    training_master : str
        Directory path where the plot image will be saved.
    oldfolder : str
        Original working directory to restore after saving.

    Returns
    -------
    CoDoverall : np.ndarray
        Mean coefficient of determination across all outputs.
    R2overall : np.ndarray
        Mean L2-based R2 score across all outputs.
    CoDview : np.ndarray
        Per-output coefficient of determination values.
    R2view : np.ndarray
        Per-output L2-based R2 values.
    """
    CoDview = np.zeros((1, Trued.shape[1]))
    R2view = np.zeros((1, Trued.shape[1]))
    plt.figure(figsize=(40, 40))
    for jj in range(Trued.shape[1]):
        logger = setup_logging(__name__)
        logger.info(" Compute L2 and R2 for the machine _" + str(jj + 1))
        operationanswer2 = np.reshape(CCR[:, jj], (-1, 1))
        outputtest2 = np.reshape(Trued[:, jj], (-1, 1))
        numrowstest = len(outputtest2)
        outputtest2 = np.reshape(outputtest2, (-1, 1))
        Lerrorsparse = (
            LA.norm(outputtest2 - operationanswer2) / LA.norm(outputtest2)
        ) ** 0.5
        L_22 = 1 - (Lerrorsparse**2)
        outputreq = np.zeros((numrowstest, 1))
        for i in range(numrowstest):
            outputreq[i, :] = outputtest2[i, :] - np.mean(outputtest2)
        CoDspa = 1 - (LA.norm(outputtest2 - operationanswer2) / LA.norm(outputreq))
        CoD2 = 1 - (1 - CoDspa) ** 2
        logger.info("")
        CoDview[:, jj] = CoD2
        R2view[:, jj] = L_22
        jk = jj + 1
        plt.subplot(9, 9, jk)
        palette = copy(plt.get_cmap("inferno_r"))
        palette.set_under("white")  # 1.0 represents not transparent
        palette.set_over("black")  # 1.0 represents not transparent
        vmin = min(np.ravel(outputtest2))
        vmax = max(np.ravel(outputtest2))
        sc = plt.scatter(
            np.ravel(operationanswer2),
            np.ravel(outputtest2),
            c=np.ravel(outputtest2),
            vmin=vmin,
            vmax=vmax,
            s=35,
            cmap=palette,
        )
        plt.colorbar(sc)
        plt.title("Energy_" + str(jj), fontsize=9)
        plt.ylabel("Machine", fontsize=9)
        plt.xlabel("True data", fontsize=9)
        a, b = best_fit(
            np.ravel(operationanswer2),
            np.ravel(outputtest2),
        )
        yfit = [a + b * xi for xi in np.ravel(operationanswer2)]
        plt.plot(np.ravel(operationanswer2), yfit, color="r")
        plt.annotate(
            f"R2= {CoD2:.3f}",
            (0.8, 0.2),
            xycoords="axes fraction",
            ha="center",
            va="center",
            size=9,
        )
    CoDoverall = (np.sum(CoDview, axis=1)) / Trued.shape[1]
    R2overall = (np.sum(R2view, axis=1)) / Trued.shape[1]
    with pushd(training_master):
        plt.savefig(f"{stringg}.jpg")
    return CoDoverall, R2overall, CoDview, R2view


def extract_measurements(ouut_p, well_measurements, N_pr):
    """Extract and flatten well measurement slices from a 3-D output array.

    Parameters
    ----------
    ouut_p : np.ndarray
        3-D array of model outputs, shape ``(N_ens, n_timesteps, n_total_cols)``.
    well_measurements : list of str
        Ordered list of measurement names (e.g. ``['WOPR', 'WWPR', 'WGPR']``).
    N_pr : int
        Number of producer wells; defines slice width per measurement type.

    Returns
    -------
    np.ndarray
        Stacked 2-D array of extracted measurements, shape ``(N_ens * n_flat, 1)``.
    """
    measurement_indices = {
        name: (i * N_pr, (i + 1) * N_pr) for i, name in enumerate(well_measurements)
    }
    sim = []
    for zz in range(ouut_p.shape[0]):
        extracted_data = []
        for measurement in well_measurements:
            if measurement in measurement_indices:
                start_idx, end_idx = measurement_indices[measurement]
                extracted_data.append(ouut_p[zz, :, start_idx:end_idx])
            else:
                raise ValueError(f"Unknown measurement: {measurement}")
        spit = np.hstack(extracted_data)
        spit = np.reshape(spit, (-1, 1), "F")  # Flatten in column-major order
        sim.append(spit)
    return np.vstack(sim)  # Use vstack instead of hstack


def convert_backs(rescaled_tensor, max_val, N_pr, lenwels):
    """Denormalize a tensor by multiplying each well-group slice by its max value.

    Parameters
    ----------
    rescaled_tensor : np.ndarray
        Normalized 3-D array, shape ``(N_ens, n_timesteps, lenwels * N_pr)``.
    max_val : np.ndarray
        2-D array of maximum values per well group, shape ``(N_ens, lenwels)``.
    N_pr : int
        Number of producer wells per well-group block.
    lenwels : int
        Number of well-measurement types (groups).

    Returns
    -------
    np.ndarray
        Denormalized array with same shape as `rescaled_tensor`.
    """
    C = []
    for k in range(lenwels):
        rescaled_tensorr = (
            rescaled_tensor[:, :, k * N_pr : (k + 1) * N_pr] * max_val[:, k]
        )
        C.append(rescaled_tensorr)
    return np.concatenate(C, axis=-1)


def convert_backsin(rescaled_tensor, max_val, N_pr):
    """Normalize a flat input tensor by dividing each segment by its column max.

    Parameters
    ----------
    rescaled_tensor : np.ndarray
        2-D input array containing concatenated well segments to normalize.
    max_val : np.ndarray
        2-D array of maximum values, shape ``(N_ens, n_segments)``.
    N_pr : int
        Number of producer wells; determines segment boundary sizes.

    Returns
    -------
    np.ndarray
        Concatenated normalized array with the same number of rows as input.
    """
    C = []
    Anow = rescaled_tensor[:, :N_pr]
    max_vall = max_val[:, 0]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    Anow = rescaled_tensor[:, N_pr : N_pr + 1]
    max_vall = max_val[:, 1]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    Anow = rescaled_tensor[:, N_pr + 1 : 2 * N_pr + 1]
    max_vall = max_val[:, 2]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    Anow = rescaled_tensor[:, 2 * N_pr + 1 : 3 * N_pr + 1]
    max_vall = max_val[:, 3]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    Anow = rescaled_tensor[:, 3 * N_pr + 1 : 4 * N_pr + 1]
    max_vall = max_val[:, 4]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    Anow = rescaled_tensor[:, 4 * N_pr + 1 : 4 * N_pr + 2]
    max_vall = max_val[:, 5]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    return np.concatenate(C, axis=-1)


def process_data(data):
    """Parse well location entries into a dict of zero-indexed grid index tuples.

    Parameters
    ----------
    data : list of tuple
        Each entry is ``(well_name, i, j, k_start, k_end)`` with 1-based indices.

    Returns
    -------
    dict
        Mapping from well name (str) to list of ``(i, j, k_start, k_end)`` tuples.
    """
    well_indices = {}
    for entry in data:
        if entry[0] not in well_indices:
            well_indices[entry[0]] = []
        well_indices[entry[0]].append(
            (int(entry[1]) - 1, int(entry[2]) - 1, int(entry[3]) - 1, int(entry[4]) - 1)
        )
    return well_indices


def get_dyna2(
    steppi, well_indices, well_indicesg, well_indiceso, swatuse, gasuse, oiluse, Q, Qg
):
    """Distribute injection/production rates into 4-D grid arrays for all phases.

    Parameters
    ----------
    steppi : int
        Number of simulation time steps.
    well_indices : list of tuple
        Well location entries for water injectors (name + grid indices).
    well_indicesg : list of tuple
        Well location entries for gas injectors (name + grid indices).
    well_indiceso : list of tuple
        Well location entries for oil producers (name + grid indices).
    swatuse : np.ndarray
        4-D water rate grid to be filled, shape ``(steppi, nz, nx, ny)``.
    gasuse : np.ndarray
        4-D gas rate grid to be filled, shape ``(steppi, nz, nx, ny)``.
    oiluse : np.ndarray
        4-D oil rate grid to be filled, shape ``(steppi, nz, nx, ny)``.
    Q : np.ndarray
        2-D water injection rates, shape ``(steppi, n_water_wells)``.
    Qg : np.ndarray
        2-D gas injection rates, shape ``(steppi, n_gas_wells)``.

    Returns
    -------
    swatuse : np.ndarray
        Updated water rate grid.
    gasuse : np.ndarray
        Updated gas rate grid.
    oiluse : np.ndarray
        Updated oil rate grid (producers marked -1).
    """
    unique_well_names = OrderedDict()
    for _idx, tuple_entry in enumerate(well_indices):
        well_name = tuple_entry[0]
        if well_name not in unique_well_names:
            # Assign new index as the length of current unique keys
            unique_well_names[well_name] = len(unique_well_names)
    well_name_to_index = {name: index for index, name in enumerate(unique_well_names)}
    for xx in range(steppi):
        for well_name, q_idx in well_name_to_index.items():
            entries_for_well = [t for t in well_indices if t[0] == well_name]
            total_value = Q[xx, q_idx]
            average_value = (
                total_value / len(entries_for_well) if entries_for_well else 0
            )
            for _, i_idx, j_idx, k_idx, l_idx in entries_for_well:
                if int(k_idx) - 1 == int(l_idx) - 1:
                    swatuse[xx, int(i_idx) - 1, int(j_idx) - 1, int(k_idx) - 1] = (
                        average_value
                    )
                else:
                    swatuse[
                        xx,
                        int(i_idx) - 1,
                        int(j_idx) - 1,
                        int(k_idx) - 1 : int(l_idx) - 1 + 1,
                    ] = average_value
    unique_well_namesg = OrderedDict()
    for _idx, tuple_entry in enumerate(well_indicesg):
        well_nameg = tuple_entry[0]
        if well_nameg not in unique_well_namesg:
            unique_well_namesg[well_nameg] = len(unique_well_namesg)
    well_name_to_indexg = {name: index for index, name in enumerate(unique_well_namesg)}
    for xx in range(steppi):
        for well_nameg, q_idxg in well_name_to_indexg.items():
            entries_for_wellg = [t for t in well_indicesg if t[0] == well_nameg]
            total_valueg = Qg[xx, q_idxg]
            average_valueg = (
                total_valueg / len(entries_for_wellg) if entries_for_wellg else 0
            )
            for _, i_idx, j_idx, k_idx, l_idx in entries_for_wellg:
                if int(k_idx) - 1 == int(l_idx) - 1:
                    gasuse[xx, int(i_idx) - 1, int(j_idx) - 1, int(k_idx) - 1] = (
                        average_valueg
                    )
                else:
                    gasuse[
                        xx,
                        int(i_idx) - 1,
                        int(j_idx) - 1,
                        int(k_idx) - 1 : int(l_idx) - 1 + 1,
                    ] = average_valueg
    unique_well_nameso = OrderedDict()
    for _idx, tuple_entry in enumerate(well_indiceso):
        well_nameo = tuple_entry[0]
        if well_nameo not in unique_well_nameso:
            unique_well_nameso[well_nameo] = len(unique_well_nameso)
    well_name_to_indexo = {name: index for index, name in enumerate(unique_well_nameso)}
    for xx in range(steppi):
        for well_nameo in well_name_to_indexo:
            entries_for_wello = [t for t in well_indiceso if t[0] == well_nameo]
            for _, i_idx, j_idx, k_idx, l_idx in entries_for_wello:
                if int(k_idx) - 1 == int(l_idx) - 1:
                    oiluse[xx, int(i_idx) - 1, int(j_idx) - 1, int(k_idx) - 1] = -1
                else:
                    oiluse[
                        xx,
                        int(i_idx) - 1,
                        int(j_idx) - 1,
                        int(k_idx) - 1 : int(l_idx) - 1 + 1,
                    ] = -1
    return swatuse, gasuse, oiluse
