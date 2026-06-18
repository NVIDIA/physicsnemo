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

# Standard Libraries
import os
import sys
import shlex
import logging
import subprocess
from scipy.linalg import norm


# Numerical Computing
import numpy as np
import scipy.linalg as sla
from scipy.fftpack import dct, idct

# Machine Learning
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# Visualization
from matplotlib.colors import LinearSegmentedColormap

# Image Processing
from skimage.transform import resize as rzz

# Optimization and Sampling
from pyDOE import lhs
from utils.array_utils import (
    find_first_numeric_row,
)
from utils.ecl_binary import EclBinaryParser

from utils.opm_utils import (
    parse_unrst,
    _parse_ech_bin,
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

colors = [
    (0, 0, 0),
    (0.3, 0.15, 0.75),
    (0.6, 0.2, 0.50),
    (1, 0.25, 0.15),
    (0.9, 0.5, 0),
    (0.9, 0.9, 0.5),
    (1, 1, 1),
]
n_bins = 7  # Discretizes the interpolation into bins
cmap_name = "my_list"
cmm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


# ProgressBar is imported from utils.ensemble_utils above.


def ProgressBar2(Total, Progress):
    """Return percentage string for a given progress ratio."""
    try:
        Progress = float(Progress) / float(Total)
        if Progress >= 1.0:
            Progress = 1
            return "100%"
        return f"{round(Progress * 100, 0):.0f}%"
    except Exception as e:
        logger.error(f"Error in ProgressBar2: {e}")
        return "ERROR"


# ShowBar is imported from utils.ensemble_utils above.


def load_data_numpy(inn, batch_size):
    """Wrap a single tensor array into a DataLoader for batching."""
    x_data = inn
    logger.info(f"x_data: {x_data.shape}")
    data_tuple = (torch.FloatTensor(x_data),)
    return DataLoader(
        TensorDataset(*data_tuple), batch_size=batch_size, shuffle=True, drop_last=True
    )


# Pkgen is imported from utils.ensemble_utils above.


def distrib(shape):
    """Return complex Gaussian white noise array of given shape."""
    rng = np.random.default_rng()
    a = rng.normal(loc=0, scale=1, size=shape)
    b = rng.normal(loc=0, scale=1, size=shape)
    return a + 1j * b


class RMS:
    """R M S."""

    def __init__(self, truth, ensemble):
        """Compute RMSE and ensemble spread (RMSD) against a truth vector.

        Parameters
        ----------
        truth : np.ndarray
            Reference (true) vector, shape (n,).
        ensemble : np.ndarray
            Ensemble matrix, shape (n, Ne), used to compute mean and deviations.
        """
        mean = ensemble.mean(axis=0)
        err = truth - mean
        dev = ensemble - mean
        self.rmse = norm(err)
        self.rmsd = norm(dev)

    def __str__(self):
        """Return a formatted string with RMSE and RMSD values.

        Returns
        -------
        str
            Human-readable 'X.XXXX (rmse),  X.XXXX (std)' string.
        """
        return f"{self.rmse:6.4f} (rmse),  {self.rmsd:6.4f} (std)"


def RMS_all(series, vs):
    """Log RMS metrics of all series except the reference key `vs`."""
    for k in series:
        if k != vs:
            logger.info(f"{k:8}: {RMS(series[vs], series[k])}")


def svd0(A):
    """Compute SVD with minimal shapes depending on aspect ratio."""
    M, N = A.shape
    if M > N:
        return sla.svd(A, full_matrices=True)
    return sla.svd(A, full_matrices=False)


def pad0(ss, N):
    """Zero-pad vector `ss` to length `N`."""
    """Pad ss with zeros so that len(ss)==N."""
    out = np.zeros(N)
    out[: len(ss)] = ss
    return out


def center(E, axis=0, rescale=False):
    """Center ensemble `E` along axis; optionally rescale by sqrt(N/(N-1))."""
    x = np.mean(E, axis=axis, keepdims=True)
    X = E - x
    if rescale:
        N = E.shape[axis]
        X *= np.sqrt(N / (N - 1))
    x = x.squeeze()
    return X, x


def mean0(E, axis=0, rescale=True):
    """Same as: center(E, rescale=True)[0]."""
    return center(E, axis=axis, rescale=rescale)[0]


def inflate_ens(E, factor):
    """Inflate ensemble anomalies by `factor`."""
    if factor == 1:
        return E
    X, x = center(E)
    return x + X * factor


def test_points_gen(n_test, nder, interval=(-1.0, 1.0), distrib="random", **kwargs):
    """Generate test points either uniformly random or via LHS."""
    if distrib.lower() == "random":
        rng = np.random.default_rng()
        return interval[0] + (interval[1] - interval[0]) * rng.random((n_test, nder))
    if distrib.lower() == "lhs":
        return (interval[1] - interval[0]) * lhs(
            nder, samples=n_test, **kwargs
        ) + interval[0]
    raise ValueError(f"Unknown distribution: {distrib}")


# Reinvent is imported from utils.ensemble_utils above.


# fit_operation and convert_back are imported from utils.array_utils above.

# scale_operation, replace_large_and_invalid_values, clean_dict_arrays,
# replace_nans_and_infs, clip_and_convert_to_float32, clip_and_convert_to_float3,
# Make_correct are imported from utils.array_utils above.


SUPPORTED_DATA_TYPES = {
    "INTE": (4, "i", 1000),
    "REAL": (4, "f", 1000),
    "LOGI": (4, "i", 1000),
    "DOUB": (8, "d", 1000),
    "CHAR": (8, "8s", 105),
    "MESS": (8, "8s", 105),
    "C008": (8, "8s", 105),
}


def remove_rows(matrix, indices_to_remove):
    """Remove rows by index from 2D `matrix` and return the reduced array."""
    return np.delete(matrix, indices_to_remove, axis=0)


def Reservoir_simulation(
    perm,
    poro,
    fault,
    string_operat2,
    nx,
    ny,
    nz,
    steppi_indices,
    dest_dir,
    oldfolder,
    producer_wells,
    cfg,
    quant_big,
    rows_to_remove,
    N_pr,
    lenwels,
    steppi,
):
    """Run deck with modified properties and collect states and rates."""
    os.chdir(dest_dir)
    filename1 = cfg.custom.PERMX_INCLUDE  # 'AD_PERM' + '.INC'
    filename2 = cfg.custom.PORO_INCLUDE
    np.savetxt(
        filename1,
        perm,
        fmt="%.4f",
        delimiter=" \t",
        newline="\n",
        header="PERMX",
        footer="/",
        comments="",
    )

    np.savetxt(
        filename2,
        poro,
        fmt="%.4f",
        delimiter=" \t",
        newline="\n",
        header="PORO",
        footer="/",
        comments="",
    )
    my_array = fault.ravel()
    my_array_index = 0
    filename3 = cfg.custom.FAULT_INCLUDE
    with open(filename3) as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.strip() == "MULTFLT":
            continue
        parts = line.split(" ")
        if len(parts) > 1 and parts[1].replace(".", "", 1).replace("/", "").isdigit():
            parts[1] = str(my_array[my_array_index])
            lines[i] = " ".join(parts)
            my_array_index += 1
    with open(filename3, "w") as file:
        file.writelines(lines)
    # Parse the command string into argv tokens so we can run without a shell.
    # This avoids the shell-injection surface that ``shell=True`` exposes when
    # ``string_operat2`` is built from configuration values.
    try:
        argv = shlex.split(string_operat2, posix=os.name != "nt")
        result = subprocess.run(
            argv, shell=False, capture_output=True, text=True, timeout=30, check=False
        )
        if result.returncode != 0:
            logger.warning(f"Command failed: {result.stderr}")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.error(f"Error running command: {e}")
    check = np.ones((nx, ny, nz), dtype=np.float32)
    filenamea = os.path.basename(cfg.custom["DECK"])
    filenameui = os.path.splitext(filenamea)[0]
    unsmry_file = filenameui
    parser = EclBinaryParser(unsmry_file)
    vectors = parser.read_vectors()
    Time = vectors["TIME"]
    start_row = find_first_numeric_row(Time)
    if start_row is not None:
        numeric_df = Time.iloc[start_row:]
        np_array2 = numeric_df.to_numpy()
    np_array2 = np_array2[steppi_indices - 1, :].ravel()
    unie = []
    for zz in range(steppi):
        aa = np_array2[zz] * check
        unie.append(aa)
    Time = np.stack(unie, axis=0)
    pressure = []
    swat = []
    sgas = []
    attrs = ("GRIDHEAD", "ACTNUM")
    egrid = _parse_ech_bin(filenameui + ".EGRID", attrs)
    nx, ny, nz = egrid["GRIDHEAD"][0][1:4]
    actnum = egrid["ACTNUM"][0]  # numpy array of size nx * ny * nz
    states = parse_unrst(filenameui + ".UNRST")
    pressuree = states["PRESSURE"]
    swatt = states["SWAT"]
    sgass = states["SGAS"]
    # soils = states["SOIL"]
    active_index_array = np.where(actnum == 1)[0]
    len_act_indx = len(active_index_array)
    filtered_pressure = pressuree
    filtered_swat = swatt
    filtered_sgas = sgass
    active_index_array = np.where(actnum == 1)[0]
    len_act_indx = len(active_index_array)
    for pr_slice, sw_slice, sg_slice in zip(
        filtered_pressure, filtered_swat, filtered_sgas, strict=False
    ):
        for state_var, all_slices in zip(
            [pr_slice, sw_slice, sg_slice], [pressure, swat, sgas], strict=False
        ):
            resize_state_var = np.zeros((nx * ny * nz, 1))
            resize_state_var[active_index_array] = rzz(
                state_var.reshape(-1, 1), (len_act_indx,), order=1, preserve_range=True
            )
            resize_state_var = np.reshape(resize_state_var, (nx, ny, nz), "F")
            all_slices.append(resize_state_var)
    sgas = np.stack(sgas, axis=0)
    pressure = np.stack(pressure, axis=0)
    swat = np.stack(swat, axis=0)
    # soil = np.stack(soil, axis=0)
    soil = abs(1 - (swat + sgas))
    sgas = sgas[steppi_indices - 1, :, :, :]
    swat = swat[steppi_indices - 1, :, :, :]
    pressure = pressure[steppi_indices - 1, :, :, :]
    soil = soil[steppi_indices - 1, :, :, :]
    unsmry_file = filenameui
    parser = EclBinaryParser(unsmry_file)
    vectors = parser.read_vectors()
    namez = ["WOPR", "WWPR", "WGPR"]
    all_arrays = []
    producer_well_names = [well[-1] for well in producer_wells]
    for namey in namez:
        dfaa = vectors[namey]
        filtered_columns = [
            coll
            for coll in dfaa.columns
            if any(well_namee in coll for well_namee in producer_well_names)
        ]
        filtered_df = dfaa[filtered_columns]
        filtered_df = filtered_df[producer_well_names]
        start_row = find_first_numeric_row(filtered_df)
        if start_row is not None:
            numeric_df = filtered_df.iloc[start_row:]
            result_array = numeric_df.to_numpy()
            logger.info(f"Numeric data from {namey} processed successfully.")
        else:
            logger.info(f"No numeric rows found in the DataFrame for {namey}.")
            result_array = None
        all_arrays.append(result_array)
    final_array = np.concatenate(all_arrays, axis=1)
    final_array[final_array <= 0] = 0
    out = final_array[steppi_indices - 1, :].astype(float)
    out[out <= 0] = 0
    scaled_obs_chunks = []
    for k in range(lenwels):
        quantt = quant_big[f"K_{k}"]
        # ajes = quantt["value"]
        if quantt["boolean"] == 1:
            kodsval = out[:, k * N_pr : (k + 1) * N_pr] / quantt["scale"]
        else:
            kodsval = out[:, k * N_pr : (k + 1) * N_pr] * quantt["scale"]
        scaled_obs_chunks.append(kodsval)
    spit = np.hstack(scaled_obs_chunks)
    spit = np.reshape(spit, (-1, 1), "F")
    spit = remove_rows(spit, rows_to_remove).reshape(-1, 1)
    use = np.reshape(spit, (-1, 1), "F")
    os.chdir(oldfolder)
    return pressure, swat, soil, sgas, out, use


def Get_new_K(Low_K, High_K, LogS1):
    """Linearly mix lower/upper bounds using LogS1 mask to produce K."""
    return (High_K * LogS1) + (1 - LogS1) * Low_K


def Get_weighting(simData, measurment):
    """Compute weights proportional to similarity between simulations and data."""
    ne = simData.shape[1]
    measurment = measurment.reshape(-1, 1)
    objReal = np.zeros((ne, 1))
    temp = np.zeros((ne, 1))
    for j in range(ne):
        a = np.sum(simData[:, j] - measurment) ** 2
        b = np.sum((simData[:, j]) ** 2) + np.sum((measurment) ** 2)
        weight = a / b
        temp[j] = weight
    tempbig = np.sum(temp)
    right = ne - tempbig

    for j in range(ne):
        a = np.sum(simData[:, j] - measurment) ** 2
        b = np.sum((simData[:, j]) ** 2) + np.sum((measurment) ** 2)
        objReal[j] = (1 - (a / b)) / right

    return objReal


def idct22(a, Ne, nx, ny, nz, size1, size2):
    """Inverse DCT per layer reconstructing Nx x Ny fields from coefficients."""
    ouut = np.zeros((nx * ny * nz, Ne))
    for ix in range(Ne):
        # i=0
        subbj = a[:, ix]
        subbj = np.reshape(subbj, (size1, size2, nz), "F")
        neww = np.zeros((nx, ny))
        outt = []
        for jg in range(nz):
            # j=0
            usee = subbj[:, :, jg]
            neww[:size1, :size2] = usee
            aa = idct(idct(neww.T, norm="ortho").T, norm="ortho")
            subbout = np.reshape(aa, (-1, 1), "F")
            outt.append(subbout)
        outt = np.vstack(outt)
        ouut[:, ix] = np.ravel(outt)
    return ouut


# Split_Matrix is imported from utils.array_utils above.


def shuffle(x, axis=0):
    """Shuffle an array along `axis` using a random generator."""
    n_axis = len(x.shape)
    t = np.arange(n_axis)
    t[0] = axis
    t[axis] = 0
    xt = np.transpose(x.copy(), t)
    rng = np.random.default_rng()
    rng.shuffle(xt)
    return np.transpose(xt, t)


def dct2(a):
    """2D discrete cosine transform with ortho normalisation."""
    return dct(dct(a.T, norm="ortho").T, norm="ortho")


def idct2(a):
    """2D inverse discrete cosine transform with ortho normalisation."""
    return idct(idct(a.T, norm="ortho").T, norm="ortho")


def dct22(a, Ne, nx, ny, nz, size1, size2):
    """DCT per layer to get compact coefficients across a 3D grid field."""
    ouut = np.zeros((size1 * size2 * nz, Ne))
    for i in range(Ne):
        origi = np.reshape(a[:, i], (nx, ny, nz), "F")
        outt = []
        for j in range(nz):
            mike = origi[:, :, j]
            dctco = dct(dct(mike.T, norm="ortho").T, norm="ortho")
            subb = dctco[:size1, :size2]
            subb = np.reshape(subb, (-1, 1), "F")
            outt.append(subb)
        outt = np.vstack(outt)
        ouut[:, i] = np.ravel(outt)
    return ouut
