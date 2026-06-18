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
OPM geostatistics and property extraction helpers.

This module focuses on reading Eclipse/Flow binary files (EGRID/UNRST),
decoding keyword sections, and preparing gridded static/dynamic properties
for downstream tasks such as surrogate training and inversion. It also
provides utilities to:

- Generate initial ensembles using Gaussian random fields (gstools) or
  multiple-point statistics via mpslib.
- Rescale and clip arrays for numerical stability and float32 safety.
- Build per-well fault masks based on FAULTS/MULTFLT deck sections.
- Convert between Python/CuPy/Torch representations via DLPack bridges.

Key functions
-------------
- intial_ensemble / initial_ensemble_gaussian: Build permeability/porosity
  ensembles.
- parse_egrid / parse_unrst: Extract ACTNUM/GRIDHEAD and state variables.
- get_all_properties: Aggregate grid states, well rates and optional fault volumes
  for requested output variables.
- save_files / copy_files / Run_simulator: Deck editing and execution helpers.
- scale_operation*, replace_large_and_invalid_values: Robust scaling & cleanup.

Notes
-----
- Functions expect the working directory to contain the Eclipse binary files
  referred to by the provided deck path.
- Some utilities assume NORNE layout conventions used elsewhere in the codebase.

@Author: Clement Etienam
"""

import os
import random
import shlex
import logging
import subprocess
import warnings
import shutil
import numpy as np
import numpy.linalg
import numpy.matlib
from cpuinfo import get_cpu_info
from scipy.fftpack import dct, idct
from sklearn.preprocessing import MinMaxScaler
from gstools.random import MasterRNG
from gstools import SRF, Gaussian
from utils.io_utils import is_available
from utils.path_utils import pushd
from utils.ensemble_utils import (
    dct_nd,
)

warnings.filterwarnings("ignore")


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
s = get_cpu_info()
logger.info("CPU Info:")
for k, v in s.items():
    logger.info(f"\t{k}: {v}")

yet = is_available()
if yet:
    logging.getLogger(__name__).info("GPU Available with CUDA")
    import cupy as cp

else:
    logging.getLogger(__name__).info("No GPU Available")
    import numpy as cp

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


# Geostatistics module
# intial_ensemble is imported from utils.ensemble_utils above.


def initial_ensemble_gaussian(Nx, Ny, Nz, N, minn, maxx, minnp, maxxp):
    """Generate Gaussian random-field ensembles and rescale to ranges.

    Returns two matrices for permeability-like and porosity-like ranges.
    """
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


# Pkgen, distrib, test_points_gen are imported from utils.ensemble_utils above.
# round_array_to_4dp is imported from utils.array_utils above.


def Getit(data, input_keys, output_keys, output_keys2):
    """Partition a data dict into three sub-dicts by key lists.

    Parameters
    ----------
    data : dict
        Source mapping of key -> array (e.g., an `.npz` archive).
    input_keys : list[str]
        Keys to place in the input variable dict.
    output_keys : list[str]
        Keys to place in the first output variable dict.
    output_keys2 : list[str]
        Keys to place in the second output variable dict.

    Returns
    -------
    tuple[dict, dict, dict]
        ``(invar, outvar, outvar2)`` each containing the requested arrays.
    """
    _ks = [k for k in data if not k.startswith("__")]
    invar, outvar, outvar2 = dict(), dict(), dict()
    for d, keys in [
        (invar, input_keys),
        (outvar, output_keys),
        (outvar2, output_keys2),
    ]:
        for k in keys:
            x = data[k]  # N, C, H, W
            d[k] = x
    del data
    return (invar, outvar, outvar2)


def Getit2(data, input_keys, output_keys):
    """Partition a data dict into two sub-dicts by key lists.

    Parameters
    ----------
    data : dict
        Source mapping of key -> array (e.g., an `.npz` archive).
    input_keys : list[str]
        Keys to place in the input variable dict.
    output_keys : list[str]
        Keys to place in the output variable dict.

    Returns
    -------
    tuple[dict, dict]
        ``(invar, outvar)`` each containing the requested arrays.
    """
    _ks = [k for k in data if not k.startswith("__")]
    invar, outvar = dict(), dict()
    for d, keys in [(invar, input_keys), (outvar, output_keys)]:
        for k in keys:
            x = data[k]  # N, C, H, W
            d[k] = x
    return (invar, outvar)


# smoothn, gcv, RobustWeights are imported from utils.ensemble_utils above.


def InitialGuess(y, mask_I):
    """Fill missing values and DCT-smooth for an initial solution estimate.

    Parameters
    ----------
    y : np.ndarray
        Data array possibly containing NaN/missing entries.
    mask_I : np.ndarray
        Boolean array; ``True`` where `y` is valid (finite).

    Returns
    -------
    np.ndarray
        Low-frequency approximation of `y` with missing values filled.
    """
    if any(~mask_I):
        try:
            from scipy.ndimage.morphology import distance_transform_edt

            L = distance_transform_edt(1 - mask_I)
            z = y
            z[~mask_I] = y[L[~mask_I]]
        except Exception:
            z = y
            z[~mask_I] = np.mean(y[mask_I])
    else:
        z = y
    z = dct_nd(z, f=dct)
    k = np.array(z.shape)
    m = np.ceil(k / 10) + 1
    d = [np.arange(m[i], k[i]) for i in range(len(k))]
    d = np.array(d).astype(int)
    z[d] = 0.0
    return dct_nd(z, f=idct)


# dct_nd is imported from utils.ensemble_utils above.


def peaks(n):
    """Generate a random 2-D Gaussian-peaks surface on an ``n x n`` grid.

    Parameters
    ----------
    n : int
        Side length of the square grid.

    Returns
    -------
    np.ndarray
        2-D float array of shape ``(n, n)`` with superposed Gaussian bumps.
    """
    xp = np.arange(n)
    [x, y] = np.meshgrid(xp, xp)
    z = np.zeros_like(x).astype(float)
    for _i in np.xrange(n / 5):
        x0 = random.random() * n
        y0 = random.random() * n
        sdx = random.random() * n / 4.0
        sdy = sdx
        c = random.random() * 2 - 1.0
        f = np.exp(
            -(((x - x0) / sdx) ** 2)
            - ((y - y0) / sdy) ** 2
            - ((x - x0) / sdx) * ((y - y0) / sdy) * c
        )
        f *= random.random()
        z += f
    return z


def simulator_to_python(a):
    """Convert a 3-D simulator array (nx, ny, nz) to a column vector.

    Transposes each z-layer and column-major-reshapes the result to match
    Python/CuPy conventions used elsewhere in the pipeline.

    Parameters
    ----------
    a : cp.ndarray
        3-D array of shape ``(nx, ny, nz)`` in simulator layout.

    Returns
    -------
    cp.ndarray
        Column vector of shape ``(nx * ny * nz, 1)``.
    """
    kk = a.shape[2]
    anew = []
    for i in range(kk):
        afirst = a[:, :, i]
        afirst = afirst.T
        afirst = cp.reshape(afirst, (-1, 1), "F")
        anew.append(afirst)
    return cp.vstack(anew)


def python_to_simulator(a, ny, nx, nz):
    """Reshape a column vector back to the simulator's (nx, ny, nz) layout.

    Inverse of `simulator_to_python`; transposes each z-layer after reshaping.

    Parameters
    ----------
    a : cp.ndarray
        1-D or column-vector array of length ``nx * ny * nz``.
    ny : int
        Grid dimension in the y-direction.
    nx : int
        Grid dimension in the x-direction.
    nz : int
        Grid dimension in the z-direction.

    Returns
    -------
    cp.ndarray
        Stacked 2-D array of shape ``(nz * nx, ny)`` in simulator layout.
    """
    a = cp.reshape(a, (-1, 1), "F")
    a = cp.reshape(a, (ny, nx, nz), "F")
    anew = []
    for i in range(nz):
        afirst = a[:, :, i]
        afirst = afirst.T
        anew.append(afirst)
    return cp.vstack(anew)


# rescale_linear, rescale_linear_numpy_pytorch, rescale_linear_pytorch_numpy
# are imported from utils.array_utils above.


# fit_operation is imported from utils.array_utils above.


def copy_files(source_dir, dest_dir):
    """Copy all files from `source_dir` to `dest_dir`, excluding output files.

    Parameters
    ----------
    source_dir : str
        Path to the source directory.
    dest_dir : str
        Path to the destination directory.

    Returns
    -------
    None
    """
    files = os.listdir(source_dir)
    exclude_files = {"sgsim.out", "sgsimporo.out"}
    for file in files:
        if file not in exclude_files:  # Skip excluded files
            src_path = os.path.join(source_dir, file)
            shutil.copy(src_path, dest_dir)


def save_files(
    perm, poro, perm2, dest_dir, oldfolder, FAULT_INCLUDE, PERMX_INCLUDE, PORO_INCLUDE
):
    """Write PERMX, PORO, and update MULTFLT deck include files in place."""
    del oldfolder  # kept for backwards compat; pushd handles cwd restoration
    with pushd(dest_dir):
        filename1 = PERMX_INCLUDE  #'permx' + '.dat'
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
        filename2 = PORO_INCLUDE  # 'porosity'+'.dat'
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
        my_array = perm2.ravel()
        my_array_index = 0
        with open(FAULT_INCLUDE) as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if line.strip() == "MULTFLT":
                continue
            parts = line.split(" ")
            if (
                len(parts) > 1
                and parts[1].replace(".", "", 1).replace("/", "").isdigit()
            ):
                parts[1] = str(my_array[my_array_index])
                lines[i] = " ".join(parts)
                my_array_index += 1
        with open(FAULT_INCLUDE, "w") as file:
            file.writelines(lines)


def Run_simulator(dest_dir, oldfolder, string_simulation2):
    """Execute the external simulator command in `dest_dir` and return."""
    del oldfolder  # kept for backwards compat; pushd handles cwd restoration
    with pushd(dest_dir):
        # Parse the command into argv tokens so we can run without a shell,
        # avoiding the shell-injection surface that ``os.system`` exposes
        # when the command string is built from configuration.
        subprocess.run(
            shlex.split(string_simulation2, posix=os.name != "nt"),
            shell=False,
            check=False,
        )


# convert_back is imported from utils.array_utils above.


# replace_nans_and_infs is imported from utils.array_utils above.


def scale_operation(tensor, target_min, target_max):
    """Normalise a NumPy array to ``[0, 1]`` by dividing by its maximum.

    NaN and Inf values are replaced with zero before scaling.

    Parameters
    ----------
    tensor : np.ndarray
        Input array to normalise (modified in-place for NaN/Inf).
    target_min : float
        Intended output lower bound (currently unused).
    target_max : float
        Intended output upper bound (currently unused).

    Returns
    -------
    tuple[float, float, np.ndarray]
        ``(min_val, max_val, rescaled_tensor)`` — original extrema and
        the normalised array.
    """
    _n = int(np.sum(np.isnan(tensor) | np.isinf(tensor)))
    if _n > 0:
        logger.warning("scale_operation: replacing %d NaN/Inf values with 0", _n)
    tensor[np.isnan(tensor)] = 0
    tensor[np.isinf(tensor)] = 0
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    rescaled_tensor = tensor / max_val
    return min_val, max_val, rescaled_tensor


def scale_operation_pressure(tensor, max_val):
    """Normalise a pressure array using a supplied maximum value.

    NaN and Inf values are replaced with zero before scaling.

    Parameters
    ----------
    tensor : np.ndarray
        Pressure array to normalise (modified in-place for NaN/Inf).
    max_val : float
        External maximum used as the divisor.

    Returns
    -------
    tuple[float, float, np.ndarray]
        ``(min_val, max_val, rescaled_tensor)`` — minimum of the cleaned
        array, the supplied `max_val`, and the normalised array.
    """
    _n = int(np.sum(np.isnan(tensor) | np.isinf(tensor)))
    if _n > 0:
        logger.warning(
            "scale_operation_pressure: replacing %d NaN/Inf values with 0", _n
        )
    tensor[np.isnan(tensor)] = 0
    tensor[np.isinf(tensor)] = 0
    rescaled_tensor = tensor / max_val
    return np.min(tensor), max_val, rescaled_tensor


def scale_operationS(tensor, lenwels, N_pr):
    """Normalise per-well blocks of a tensor independently.

    Each block of ``N_pr`` columns corresponding to one well is divided by
    its own maximum. NaN and Inf values are replaced with zero first.

    Parameters
    ----------
    tensor : np.ndarray
        Array of shape ``(samples, timesteps, lenwels * N_pr)`` to normalise.
    lenwels : int
        Number of wells; controls how `tensor` is split along the last axis.
    N_pr : int
        Number of production measures per well.

    Returns
    -------
    tuple[np.ndarray, list[float], list[float]]
        ``(get_it2, Cmax, Cmin)`` — concatenated normalised array and
        lists of per-well max/min values.
    """
    _n = int(np.sum(np.isnan(tensor) | np.isinf(tensor)))
    if _n > 0:
        logger.warning("scale_operationS: replacing %d NaN/Inf values with 0", _n)
    tensor[np.isnan(tensor)] = 0
    tensor[np.isinf(tensor)] = 0
    C = []
    Cmax = []
    Cmin = []
    for k in range(lenwels):
        Anow = tensor[:, :, k * N_pr : (k + 1) * N_pr]
        min_val = np.min(Anow)
        max_val = np.max(Anow)
        rescaled_tensor = Anow / max_val
        C.append(rescaled_tensor)
        Cmax.append(max_val)
        Cmin.append(min_val)
    get_it2 = np.concatenate(C, 2)
    return get_it2, Cmax, Cmin


def scale_operationSin(tensor, N_pr):
    """Normalise six fixed segments of a single-well injection tensor.

    Each of the six segments (defined by offsets relative to `N_pr`) is
    divided by its own maximum. NaN and Inf values are zeroed first.

    Parameters
    ----------
    tensor : np.ndarray
        Array containing concatenated injection data for one well.
    N_pr : int
        Length of the primary production segment.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(get_it2, Cmax, Cmin)`` — concatenated normalised segments and
        arrays of shape ``(1, 6)`` with per-segment max/min.
    """
    _n = int(np.sum(np.isnan(tensor) | np.isinf(tensor)))
    if _n > 0:
        logger.warning("scale_operationSin: replacing %d NaN/Inf values with 0", _n)
    tensor[np.isnan(tensor)] = 0
    tensor[np.isinf(tensor)] = 0
    C = []
    Cmax = np.zeros((1, 6))
    Cmin = np.zeros((1, 6))
    Anow = tensor[:, :, :N_pr]
    min_val = np.min(Anow)
    max_val = np.max(Anow)
    rescaled_tensor = Anow / max_val
    C.append(rescaled_tensor)
    Cmax[:, 0] = max_val
    Cmin[:, 0] = min_val
    Anow = tensor[:, :, N_pr : N_pr + 1]
    min_val = np.min(Anow)
    max_val = np.max(Anow)
    rescaled_tensor = Anow / max_val
    C.append(rescaled_tensor)
    Cmax[:, 1] = max_val
    Cmin[:, 1] = min_val
    Anow = tensor[:, :, N_pr + 1 : 2 * N_pr + 1]
    min_val = np.min(Anow)
    max_val = np.max(Anow)
    rescaled_tensor = Anow / max_val
    C.append(rescaled_tensor)
    Cmax[:, 2] = max_val
    Cmin[:, 2] = min_val
    Anow = tensor[:, :, 2 * N_pr + 1 : 3 * N_pr + 1]
    min_val = np.min(Anow)
    max_val = np.max(Anow)
    rescaled_tensor = Anow / max_val
    C.append(rescaled_tensor)
    Cmax[:, 3] = max_val
    Cmin[:, 3] = min_val
    Anow = tensor[:, :, 3 * N_pr + 1 : 4 * N_pr + 1]
    min_val = np.min(Anow)
    max_val = np.max(Anow)
    rescaled_tensor = Anow / max_val
    C.append(rescaled_tensor)
    Cmax[:, 4] = max_val
    Cmin[:, 4] = min_val
    Anow = tensor[:, :, 4 * N_pr + 1 : 4 * N_pr + 2]
    min_val = np.min(Anow)
    max_val = np.max(Anow)
    rescaled_tensor = Anow / max_val
    C.append(rescaled_tensor)
    Cmax[:, 5] = max_val
    Cmin[:, 5] = min_val
    get_it2 = np.concatenate(C, 2)
    return get_it2, Cmax, Cmin
