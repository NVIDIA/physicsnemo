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
- Geta_all: Aggregate grid states, well rates and optional fault volumes
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
import math
import random
import logging
import warnings
import re
import shutil
import fnmatch
import yaml
import numpy as np
import numpy.linalg
import numpy.matlib
from struct import unpack
from shutil import rmtree
from collections import OrderedDict
from cpuinfo import get_cpu_info
from scipy.stats import norm
from scipy.fftpack import dct, idct
from scipy.optimize import lbfgsb
from skimage.transform import resize as rzz
from sklearn.preprocessing import MinMaxScaler
from pyDOE import lhs
from gstools.random import MasterRNG
from gstools import SRF, Gaussian
import mpslib as mps  # Assuming this is a custom multiprocessing library
import torch
from compare.batch.misc_forward_utils import EclBinaryParser

torch.cuda.empty_cache()  # Clears unused memory


def is_available():
    """Check if NVIDIA GPU is available using `nvidia-smi`."""
    return os.system("nvidia-smi") == 0


# 🚨 Suppress Warnings
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
def intial_ensemble(Nx, Ny, Nz, N, permx):
    """Generate an initial ensemble via multiple-point statistics (mpslib).

    Parameters
    ----------
    Nx, Ny, Nz : int
        Grid dimensions (x, y, z).
    N : int
        Number of realisations.
    permx : np.ndarray
        Training image (TI) used for MPS realisations.

    Returns
    -------
    np.ndarray
        Matrix of shape `(Nx*Ny*Nz, N)` with column-wise realisations.
    """
    mps_obj = mps.mpslib()
    mps_obj = mps.mpslib(method="mps_snesim_tree")
    mps_obj.par["n_real"] = N
    k = permx
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
    from glob import glob

    for f3 in glob("thread*"):
        rmtree(f3)
    for f4 in glob("*mps_snesim_tree_*"):
        os.remove(f4)
    for f4 in glob("*ti_thread_*"):
        os.remove(f4)
    return ensemble


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


def Pkgen(n):
    def Pk(k):
        return np.power(k, -n)

    return Pk


def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b


def test_points_gen(n_test, nder, interval=(-1.0, 1.0), distrib="random", **kwargs):
    return {
        "random": lambda n_test, nder: (interval[1] - interval[0])
        * np.random.rand(n_test, nder)
        + interval[0],
        "lhs": lambda n_test, nder: (interval[1] - interval[0])
        * lhs(nder, samples=n_test, **kwargs)
        + interval[0],
    }[distrib.lower()](n_test, nder)


def round_array_to_4dp(arr):
    arr = np.asarray(arr)
    return np.around(arr, 4)


def Getit(data, input_keys, output_keys, output_keys2):
    _ks = [k for k in data.keys() if not k.startswith("__")]
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
    _ks = [k for k in data.keys() if not k.startswith("__")]
    invar, outvar = dict(), dict()
    for d, keys in [(invar, input_keys), (outvar, output_keys)]:
        for k in keys:
            x = data[k]  # N, C, H, W
            d[k] = x
    return (invar, outvar)


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
    if isinstance(y, np.ma.MaskedArray):  # masked array
        mask = y.mask
        y = np.array(y)
        y[mask] = 0.0
        if np.any(W is not None):
            W = np.array(W)
            W[mask] = 0.0
        if sd is not None:
            sd_arr = np.asarray(sd)
            W = np.array(1.0 / sd_arr**2)
            W[mask] = 0.0
            sd = None
        y[mask] = np.nan
    if sd is not None:
        sd_ = np.array(sd)
        mask = sd_ > 0.0
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
    # ---
    tol = 1.0
    RobustIterativeProcess = True
    RobustStep = 1
    nit = 0
    # errp = 0.1
    RF = 1 + 0.75 * isweighted
    if isauto:
        try:
            xpost = np.array([(0.9 * np.log10(sMinBnd) + np.log10(sMaxBnd) * 0.1)])
        except Exception:
            np.array([100.0])
    else:
        xpost = np.array([np.log10(s)])
    while RobustIterativeProcess:
        aow = sum(Wtot) / noe  # 0 < aow <= 1
        while tol > TolZ and nit < MaxIter:
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
            s = 0.0 if s is None else s
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
    return z, s, exitflag, Wtot


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


def InitialGuess(y, mask_I):
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
    z = dctND(z, f=dct)
    k = np.array(z.shape)
    m = np.ceil(k / 10) + 1
    d = []
    for i in np.xrange(len(k)):
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
    kk = a.shape[2]
    anew = []
    for i in range(kk):
        afirst = a[:, :, i]
        afirst = afirst.T
        afirst = cp.reshape(afirst, (-1, 1), "F")
        anew.append(afirst)
    return cp.vstack(anew)


def python_to_simulator(a, ny, nx, nz):
    a = cp.reshape(a, (-1, 1), "F")
    a = cp.reshape(a, (ny, nx, nz), "F")
    anew = []
    for i in range(nz):
        afirst = a[:, :, i]
        afirst = afirst.T
        anew.append(afirst)
    return cp.vstack(anew)


def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def rescale_linear_numpy_pytorch(array, new_min, new_max, minimum, maximum):
    """Rescale an arrary linearly."""
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def rescale_linear_pytorch_numpy(array, new_min, new_max, minimum, maximum):
    """Rescale an arrary linearly."""
    m = (maximum - minimum) / (new_max - new_min)
    b = minimum - m * new_min
    return m * array + b


def read_yaml(fname):
    """Read Yaml file into a dict of parameters"""
    with open(fname, "r") as stream:
        data = yaml.safe_load(stream)

        return data


def fit_operation(tensor, target_min, target_max, tensor_min, tensor_max):
    rescaled_tensor = tensor / tensor_max
    return rescaled_tensor


def parse_egrid(path_to_result):
    """Parse EGRID keywords and return sections for GRIDHEAD/ACTNUM."""
    egrid_path = path_to_result
    attrs = ("GRIDHEAD", "ACTNUM")
    egrid = _parse_ech_bin(egrid_path, attrs)
    return egrid


def parse_unrst(path_to_result):
    """Parse UNRST keywords and return sections for PRESSURE/SGAS/SWAT."""
    unrst_path = path_to_result
    attrs = ("PRESSURE", "SGAS", "SWAT")
    states = _parse_ech_bin(unrst_path, attrs)
    return states


def _check_and_fetch_type_info(data_type):
    try:
        return SUPPORTED_DATA_TYPES[data_type]
    except KeyError as exc:
        raise ValueError("Unknown datatype %s." % data_type) from exc


def _check_and_fetch_file(path, pattern, return_relative=False):
    found = []
    reg_expr = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    for f in os.listdir(path):
        if re.match(reg_expr, f):
            f_path = os.path.join(path, f)
            if return_relative:
                found.append(os.path.relpath(f_path, start=path))
            else:
                found.append(f_path)
    return found


def _parse_keywords(path, attrs=None):
    sections_counter = {} if attrs is None else {attr: 0 for attr in attrs}
    with open(path, "rb") as f:
        header = f.read(4)
        sections = dict()
        while True:
            try:
                section_name = (
                    unpack("8s", f.read(8))[0].decode("ascii").strip().upper()
                )
            except Exception:
                break
            n_elements = unpack(">i", f.read(4))[0]
            data_type = unpack("4s", f.read(4))[0].decode("ascii")
            f.read(8)
            element_size, fmt, element_skip = _check_and_fetch_type_info(data_type)
            f.seek(f.tell() - 24)
            binary_data = f.read(
                24
                + element_size * n_elements
                + 8 * (math.floor((n_elements - 1) / element_skip) + 1)
            )
            if (attrs is None) or (section_name in attrs):
                sections_counter[section_name] = (
                    sections_counter.get(section_name, 0) + 1
                )
                if section_name not in sections:
                    sections[section_name] = []
                section = (
                    n_elements,
                    data_type,
                    element_size,
                    fmt,
                    element_skip,
                    binary_data,
                )
                section = _fetch_keyword_data(section)
                sections[section_name].append(section)

    return header, sections


def _parse_ech_bin(path, attrs=None):
    if attrs is None:
        raise ValueError("Keyword attribute cannot be empty")
    if isinstance(attrs, str):
        attrs = [attrs]
    attrs = [attr.strip().upper() for attr in attrs]
    _, sections = _parse_keywords(path, attrs)
    return sections


def _fetch_keyword_data(section):
    n_elements, data_type, element_size, fmt, element_skip, binary_data = section
    n_skip = math.floor((n_elements - 1) / element_skip)
    skip_elements = 8 // element_size
    skip_elements_total = n_skip * skip_elements
    data_format = fmt * (n_elements + skip_elements_total)
    data_size = element_size * (n_elements + skip_elements_total)
    if data_type in ["INTE", "REAL", "LOGI", "DOUB"]:
        data_format = ">" + data_format
    decoded_section = list(unpack(data_format, binary_data[24 : 24 + data_size]))
    del_ind = np.repeat(np.arange(1, 1 + n_skip) * element_skip, skip_elements)
    del_ind += np.arange(len(del_ind))
    decoded_section = np.delete(decoded_section, del_ind)
    if data_type in ["CHAR", "C008"]:
        decoded_section = np.char.decode(decoded_section, encoding="ascii")
    return decoded_section


def assign_faults(well_indices, nx, ny, nz, well_amount, data):
    """Build a 3D fault mask volume from FAULTS indices and magnitudes.

    Parameters
    ----------
    well_indices : list[tuple]
        FAULTS tuples `(name, i, i2, j, j2, k, k2)` (1-based, deck order).
    nx, ny, nz : int
        Grid dimensions.
    well_amount : list[str]
        Ordered fault names defining the mapping of `data` positions.
    data : np.ndarray
        Magnitudes for each fault name.
    """
    faultm = np.ones((nx, ny, nz), dtype=np.float16)
    unique_well_names = OrderedDict()
    for idx, tuple_entry in enumerate(well_indices):
        well_name = tuple_entry[0]
        if well_name not in unique_well_names:
            unique_well_names[well_name] = len(unique_well_names)
    well_value_map = {
        well_name: data[idx] for idx, well_name in enumerate(unique_well_names)
    }
    for well_name, average_value in well_value_map.items():
        entries_for_well = [t for t in well_indices if t[0] == well_name]
        for _, i_idx, i1_idx, j_idx, j1_idx, k_idx, k1_idx in entries_for_well:
            faultm[
                int(i_idx) - 1 : int(i1_idx),
                int(j_idx) - 1 : int(j1_idx),
                int(k_idx) - 1 : int(k1_idx),
            ] = average_value
    return faultm


def Get_fault(filename):
    """Return sorted set of fault names defined under MULTFLT sections."""
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


def Get_falt(source_dir, nx, ny, nz, floatz, N, filename_fault, FAULT_INCLUDE):
    Fault = np.ones((nx, ny, nz), dtype=np.float16)
    flt = []
    for k in range(N):
        floatts = floatz[:, k]
        fault_temp = Get_fault(os.path.join(source_dir, FAULT_INCLUDE))
        fault_data = read_faults(filename_fault, fault_temp)  # OIl
        Fault = assign_faults(fault_data, nx, ny, nz, fault_temp, floatts)
        flt.append(Fault)
    flt = np.stack(flt, axis=0)[:, None, :, :, :]
    return np.stack(flt, axis=0)


def read_faults(filename, well_names):
    """Parse FAULTS section and return tuples limited to `well_names`."""
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
                well_name = parts[0].strip("'")
                if well_name in well_names:
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


def Geta_all(
    folder,
    nx,
    ny,
    nz,
    oldfolder,
    steppi,
    steppi_indices,
    filenameui,
    injectors,
    gass,
    filename,
    compdat_datag,
    compdat_dataw,
    compdat_data,
    input_variables,
    output_variables,
    filename_fault,
    FAULT_INCLUDE,
):
    """Collect grid states, well rates, and optional faults into dict.

    This is the main aggregator used by downstream pipelines to construct
    tensors for surrogate training and evaluation.
    """
    from data_extract.opm_extract_rates import (
        find_first_numeric_row,
        extract_qs,
        get_dyna2,
    )

    os.chdir(folder)
    Qg = np.zeros((steppi, nx, ny, nz))
    Qw = np.zeros((steppi, nx, ny, nz))
    Qo = np.zeros((steppi, nx, ny, nz))
    check = np.ones((nx, ny, nz), dtype=np.float32)
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
    attrs = ("GRIDHEAD", "ACTNUM")
    egrid = _parse_ech_bin("FULLNORNE2.EGRID", attrs)
    nx, ny, nz = egrid["GRIDHEAD"][0][1:4]
    actnum = egrid["ACTNUM"][0]  # numpy array of size nx * ny * nz
    states = parse_unrst(filenameui + ".UNRST")
    filtered_states = {var: states[var] for var in output_variables if var in states}
    resized_states = {var: [] for var in output_variables}
    active_index_array = np.where(actnum == 1)[0]
    len_act_indx = len(active_index_array)
    for slices in zip(*(filtered_states[var] for var in filtered_states)):
        for state_var, var_name in zip(slices, filtered_states):
            resize_state_var = np.zeros((nx * ny * nz, 1))
            resize_state_var[active_index_array] = rzz(
                state_var.reshape(-1, 1), (len_act_indx,), order=1, preserve_range=True
            )
            resize_state_var = np.reshape(resize_state_var, (nx, ny, nz), "F")
            resized_states[var_name].append(resize_state_var)
    for var in resized_states:
        if var in ["SGAS", "SWAT", "PRESSURE"]:
            resized_states[var] = np.stack(resized_states[var], axis=0)
    for var in resized_states:
        if var in ["SGAS", "SWAT", "PRESSURE"]:
            resized_states[var] = resized_states[var][1:, :, :, :]
    if "SOIL" in output_variables:
        resized_states["SOIL"] = abs(
            1 - (resized_states["SWAT"] + resized_states["SGAS"])
        )
        resized_states["SOIL"] = resized_states["SOIL"][steppi_indices - 1, :, :, :]
    for var in resized_states:
        if var != "SOIL":
            resized_states[var] = resized_states[var][steppi_indices - 1, :, :, :]
    results = {var: resized_states[var] for var in output_variables}
    sgas = results.get("SGAS")
    swat = results.get("SWAT")
    pressure = results.get("PRESSURE")
    soil = results.get("SOIL")
    seeg, seew = extract_qs(
        steppi, steppi_indices, filenameui, injectors, gass, filename
    )
    awater, agas, aoil = get_dyna2(
        steppi, compdat_dataw, compdat_datag, compdat_data, Qw, Qg, Qo, seew, seeg
    )
    if "FAULT" in input_variables:
        float_parameters = []
        file_path = FAULT_INCLUDE  # "multflt.dat"
        with open(file_path, "r") as file:
            for line in file:
                split_line = line.split()
                if len(split_line) >= 2:
                    try:
                        float_parameter = float(split_line[1])
                        float_parameters.append(float_parameter)
                    except ValueError:
                        pass
                else:
                    pass
        floatts = np.hstack(float_parameters)

        fault_temp = Get_fault(FAULT_INCLUDE)
        fault_data = read_faults(filename_fault, fault_temp)  # OIl
        Fault = assign_faults(fault_data, nx, ny, nz, fault_temp, floatts)
    os.chdir(oldfolder)
    return_values = {}
    if "PRESSURE" in output_variables:
        return_values["PRESSURE"] = pressure
    if "SWAT" in output_variables:
        return_values["SWAT"] = swat
    if "SGAS" in output_variables:
        return_values["SGAS"] = sgas
    if "SOIL" in output_variables:
        return_values["SOIL"] = soil
    return_values["Time"] = Time
    return_values["actnum"] = actnum
    return_values["QG"] = agas
    return_values["QW"] = awater
    return_values["QO"] = aoil
    if "FAULT" in input_variables:
        return_values["FAULT"] = Fault
    return return_values


def copy_files(source_dir, dest_dir):
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
    os.chdir(dest_dir)
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
    with open(FAULT_INCLUDE, "r") as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.strip() == "MULTFLT":
            continue
        else:
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
    os.chdir(oldfolder)


def Run_simulator(dest_dir, oldfolder, string_simulation2):
    """Execute the external simulator command in `dest_dir` and return."""
    os.chdir(dest_dir)
    os.system(string_simulation2)
    os.chdir(oldfolder)


def convert_back(rescaled_tensor, target_min, target_max, min_val, max_val):
    return rescaled_tensor * max_val


def replace_nans_and_infs(tensor, value=0.0):
    tensor[torch.isnan(tensor) | torch.isinf(tensor)] = value
    return tensor


def scale_operation(tensor, target_min, target_max):
    tensor[np.isnan(tensor)] = 0  # Replace NaN with 0
    tensor[np.isinf(tensor)] = 0  # Replace infinity with 0
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    rescaled_tensor = tensor / max_val
    return min_val, max_val, rescaled_tensor


def scale_operation_pressure(tensor, max_val):
    tensor[np.isnan(tensor)] = 0  # Replace NaN with 0
    tensor[np.isinf(tensor)] = 0  # Replace infinity with 0
    rescaled_tensor = tensor / max_val
    return np.min(tensor), max_val, rescaled_tensor


def scale_operationS(tensor, lenwels, N_pr):
    tensor[np.isnan(tensor)] = 0  # Replace NaN with 0
    tensor[np.isinf(tensor)] = 0  # Replace infinity with 0
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
    tensor[np.isnan(tensor)] = 0  # Replace NaN with 0
    tensor[np.isinf(tensor)] = 0  # Replace infinity with 0
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


def replace_large_and_invalid_values(arr, placeholder=0.0):
    threshold = np.finfo(np.float32).max
    invalid_indices = (np.isnan(arr)) | (np.isinf(arr)) | (np.abs(arr) > threshold)
    arr[invalid_indices] = placeholder
    return arr


def clean_dict_arrays(data_dict):
    for key in data_dict:
        data_dict[key] = replace_large_and_invalid_values(data_dict[key])
    return data_dict


def clip_and_convert_to_float32(array):
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min

    array_clipped = np.clip(array, min_float32, max_float32)
    # array_clipped = round_array_to_4dp(array_clipped)
    return array_clipped.astype(np.float32)
