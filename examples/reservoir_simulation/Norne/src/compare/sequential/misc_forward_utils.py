"""
SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

=====================================================================
                    SEQUENTIAL FORWARD UTILITIES - CORE FUNCTIONS
=====================================================================

This module provides core forward utilities for sequential FVM
surrogate model comparisons. It includes functions for data processing,
model operations, and analysis.

Key Features:
- Data type definitions for simulation data
- Forward processing and model operations
- Data transformation and analysis
- Machine learning utilities

Usage:
    from compare.sequential.misc_forward_utils import (
        setup_simulation_parameters,
        validate_input_data,
        process_ensemble_results
    )

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import re
import pickle
import logging
import warnings

# 🔧 Third-party Libraries
import numpy as np
import pandas as pd
import numpy.matlib
import torch
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.utils.cholesky import psd_safe_cholesky
from collections import namedtuple
from struct import unpack_from
from mmap import mmap
import numpy.ma as ma

# 📦 Local Modules
from hydra.utils import to_absolute_path


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Forward problem")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logging()
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


def byte2str(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(map(byte2str, x))
    else:
        return str(x)[2:-1].strip()


def str2byte(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(map(str2byte, x))
    else:
        return bytes(x.ljust(8).upper(), "utf-8")


def filter_fakes(filename, ext, loc, target_size, fmt=">f", excl=np.inf):
    with open(filename + ext, "rb") as f:
        n_sieved = 0  # Number of fake values, initially zero
        tol = 1e-32
        array = np.array([])
        lengths = [0, 1]
        while len(array) < target_size and lengths[-1] != lengths[-2]:
            f.seek(loc + 24)
            array = np.fromfile(f, dtype=fmt, count=target_size + n_sieved)
            if fmt == "S8":  # For read_vectors
                condition = np.array(
                    [False if _.startswith("\\") else True for _ in byte2str(array)]
                )
                array = array[condition]
            elif fmt == ">i" and ext == ".SMSPEC":
                array = array[
                    ((np.abs(array) >= tol) | (array == 0))
                    & (np.abs(array) != 4000)
                    & (np.abs(array) != 2980)
                ]
            else:
                array = array[
                    ((np.abs(array) >= tol) | (array == 0)) & (np.abs(array) < excl)
                ]
            n_sieved += target_size - len(array)
            lengths.append(len(array))
    return array


def Get_data_FFNN(
    oldfolder,
    N,
    pressure,
    Sgas,
    Swater,
    Soil,
    perm,
    Time,
    steppi,
    steppi_indices,
    N_pr,
    producer_wells,
    unique_entries,
    filenameui,
    well_measurements,
    lenwels,
):
    """Assemble FFNN inputs/targets from summary vectors and grid tensors.

    Parameters mirror the batch variant. Returns `(innn, ouut)` arrays for
    sequential pipelines.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Input and output arrays for FFNN training.
    """
    well_indices = process_data(unique_entries)
    ouut = np.zeros((N, steppi, lenwels * N_pr))
    innn = np.zeros((N, steppi, (4 * N_pr) + 2))
    producer_well_names = [well[-1] for well in producer_wells]
    for i in range(N):
        folder = to_absolute_path("../RUNS/Realisation" + str(i))
        os.chdir(folder)
        unsmry_file = filenameui
        parser = EclBinaryParser(unsmry_file)
        vectors = parser.read_vectors()
        namez = well_measurements  # ['WOPR', 'WWPR', 'WGPR']
        all_arrays = []
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
                logger = setup_logging()
                logger.info(f"Numeric data from {namey} processed successfully.")
            else:
                logger = setup_logging()
                logger.info(f"No numeric rows found in the DataFrame for {namey}.")
                result_array = None
            all_arrays.append(result_array)
        final_array = np.concatenate(all_arrays, axis=1)
        final_array[final_array <= 0] = 0
        out = final_array[steppi_indices - 1, :].astype(float)
        out[out <= 0] = 0
        ouut[i, :, :] = out
        permuse = perm[i, 0, :, :, :]
        presure_use = pressure[i, :, :, :, :]
        gas_use = Sgas[i, :, :, :, :]
        water_use = Swater[i, :, :, :, :]
        oil_use = Soil[i, :, :, :, :]
        Time_use = Time[i, :, :, :, :]
        mean_big = []
        for idx, indices_list in well_indices.items():
            values = [
                permuse[i_idx, j_idx, k_idx]
                if k_idx == l_idx
                else permuse[i_idx, j_idx, k_idx : l_idx + 1]
                for i_idx, j_idx, k_idx, l_idx in indices_list
            ]
            mean_big.append(np.mean(values))
        permxx = np.tile(mean_big, (steppi, 1))
        a3 = get_dyna(steppi, well_indices, water_use[steppi_indices - 1])
        a2 = get_dyna(steppi, well_indices, gas_use[steppi_indices - 1])
        a5 = get_dyna(steppi, well_indices, oil_use[steppi_indices - 1])
        a1 = np.zeros((steppi, 1))
        a4 = np.zeros((steppi, 1))
        for k in range(steppi):
            uniep = presure_use[k, :, :, :]
            permuse = uniep
            a1[k, 0] = np.mean(permuse)
            unietime = Time_use[k, :, :, :]
            permuse = unietime
            a4[k, 0] = permuse[0, 0, 0]
        inn1 = np.hstack((permxx, a1, a5, a2, a3, a4))
        innn[i, :, :] = inn1
        os.chdir(oldfolder)
    return innn, ouut


def convert_back(rescaled_tensor, target_min, target_max, min_val, max_val):
    return rescaled_tensor * max_val


def process_data(data):
    well_indices = {}
    for entry in data:
        if entry[0] not in well_indices:
            well_indices[entry[0]] = []
        well_indices[entry[0]].append(
            (int(entry[1]) - 1, int(entry[2]) - 1, int(entry[3]) - 1, int(entry[4]) - 1)
        )
    return well_indices


class EclArray(object):
    def __init__(self, filename, offset=None, keyword=None, with_fakes=True):
        self.filename = filename
        if (offset is None and keyword is None) or (
            offset is not None and keyword is not None
        ):
            raise ValueError("Either offset or keyword must be specified")
        # First read file into buffer
        with open(filename, "rb") as f:
            buff = mmap(f.fileno(), 0, access=1)
        if offset is None:
            offset = next(_.start() for _ in re.finditer(str2byte(keyword), buff))
        self.header = unpack_from(">8si4s", buff, offset)
        self.keyword, self.number, self.typ = self.header
        fmt = ">" + type_dict[self.typ]
        excl = np.inf
        if self.typ == b"INTE":
            excl = 2500
        elif (
            self.typ == b"CHAR"
        ):  # For read_vectors to filter fake keywords, wgnames, and units
            fmt = "S8"
        if with_fakes:
            self.array = filter_fakes(
                os.path.splitext(filename)[0],
                os.path.splitext(filename)[1],
                offset,
                self.number,
                fmt=fmt,
                excl=excl,
            )
        else:
            self.array = np.array(
                unpack_from(">" + self.number * type_dict[self.typ], buff, offset + 24)
            )


class EclBinaryParser(object):
    def __init__(self, filename):
        self.vectors_df = None  # this will take the shape of the vectors
        if (
            isinstance(os.path.splitext(filename), tuple)
            and os.path.splitext(filename)[1] in ecl_extensions
        ):
            self.filename = os.path.splitext(filename)[0]
        else:
            self.filename = filename

    def _read_all_arrays(self, ext, keyword, with_fakes):
        all_arrays = []
        with open("{}.{}".format(self.filename, ext), "rb") as f:
            buff = mmap(f.fileno(), 0, access=1)
        keyword_locs = [_.start() for _ in re.finditer(str2byte(keyword), buff)]
        for keyword_loc in keyword_locs:
            all_arrays.append(
                EclArray(
                    "{}.{}".format(self.filename, ext),
                    offset=keyword_loc,
                    with_fakes=with_fakes,
                ).array
            )
        return all_arrays

    def _read_all_names(self, ext):
        return self._read_all_arrays(ext, "NAME", False)

    def _read_all_types(self, ext):
        return self._read_all_arrays(ext, "TYPE", False)

    def _read_all_pointers(self, ext):
        return self._read_all_arrays(ext, "POINTER", False)

    def _get_static_pointers(self):
        static_names = self._read_all_names("INSPEC")
        static_pointers = self._read_all_pointers("INSPEC")
        for _, (names, pointers) in enumerate(zip(static_names, static_pointers)):
            df0 = pd.DataFrame(pointers, index=names, columns=[_])
            if _ == 0:
                df = df0
            else:
                df = df.join(df0, how="outer")
            df = df[~df.index.duplicated(keep="first")]
        df.fillna("-9999", inplace=True)
        df = df.astype("int32").T.max()
        return df

    def _get_dynamic_pointers(self):
        dynamic_names = self._read_all_names("RSSPEC")
        dynamic_pointers = self._read_all_pointers("RSSPEC")
        for _, (names, pointers) in enumerate(zip(dynamic_names, dynamic_pointers)):
            df0 = pd.DataFrame(pointers, index=names, columns=[_])
            if _ == 0:
                df = df0
            else:
                df = df.join(df0, how="outer")
            df = df[~df.index.duplicated(keep="first")]
        df.fillna("-9999", inplace=True)
        df = df.astype("int32")
        df.columns = self.get_seqnum_dates().index
        return df

    def _get_all_pointers(self):
        all_pointers = pd.concat(
            [self._get_static_pointers(), self._get_dynamic_pointers()]
        )
        all_pointers = all_pointers.fillna(method="ffill", axis=1).astype("int32").T
        all_pointers.columns = [byte2str(column) for column in all_pointers.columns]
        all_pointers = self.get_seqnum_dates().join(all_pointers)
        return all_pointers

    def get_dimens(self):
        with open(self.filename + ".RSSPEC", "rb") as f:
            rsspec = mmap(f.fileno(), 0, access=1)  # Read-only access
        Dimens = namedtuple("DIMENS", "ni, nj, nk")
        ni, nj, nk = unpack_from(">3i", rsspec, offset=60)
        return Dimens(ni, nj, nk)

    def is_dual(self):
        if (
            len(
                EclArray(
                    self.filename + ".INIT", keyword="LOGIHEAD", with_fakes=False
                ).array[14]
            )
            != 0
        ):
            return True
        else:
            return False

    def get_actnum(self):
        porv_array = EclArray(
            self.filename + ".INIT", keyword="PORV", with_fakes=True
        ).array
        return ma.masked_equal(porv_array, 0)

    def get_seqnum_dates(self, condensed=True):
        itimes = self._read_all_arrays("RSSPEC", "ITIME", False)
        columns = [
            "SEQNUM",
            "DAY",
            "MONTH",
            "YEAR",
            "MINISTEP",
            "IS_UNIFIED",
            "IS_FORMATTED",
            "IS_SAVE",
            "IS_GRID",
            "IS_INIT",
            "HOUR",
            "MINUTE",
            "MICROSECOND",
        ]
        df = pd.DataFrame(itimes, columns=columns).set_index("SEQNUM")
        if condensed:
            df["DATETIME"] = pd.to_datetime(
                df[["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "MICROSECOND"]],
                format="%Y-%m-%d %H:%M:%S:%f",
            )
            df = pd.DataFrame(df["DATETIME"])
        return df

    def read_prop_array(self, prop, date=None):
        warnings.filterwarnings("ignore")
        seqnum_dates = self.get_seqnum_dates()
        ni, nj, nk = self.get_dimens()
        if prop.upper() not in self._get_all_pointers().columns:
            raise ValueError("There is no {} property".format(prop))
        if date is None:
            date = seqnum_dates.iloc[0, -1]
        if date not in seqnum_dates["DATETIME"]:
            raise ValueError(
                "There is no {} date among available restart dates".format(date)
            )
        seqnum = seqnum_dates[seqnum_dates["DATETIME"] == date].index[0]
        if prop in static_props:
            df = pd.DataFrame(self._get_static_pointers())
            ext = ".INIT"
        else:
            df = self._get_dynamic_pointers()
            ext = ".UNRST"
        pointer = df.loc[str2byte(prop), seqnum] + 4
        if pointer > 0:
            prop_array = EclArray(
                self.filename + ext, offset=pointer, with_fakes=True
            ).array
            temp_array = self.get_actnum()
            temp_array[temp_array == 0] = np.nan
            temp_array[temp_array > 0] = prop_array
            return np.reshape(temp_array, (nk, nj, ni)).T
        else:
            logger = setup_logging()
            logger.info(
                "No {0} value at {1}. Assuming zero for plotting \
                  ".format(prop, date)
            )
            return np.zeros((nk, nj, ni)).T

    def read_prop_time(self, prop, i, j, k):
        dates = self._get_all_pointers()["DATETIME"]
        values = [
            self.read_prop_array(prop, date)[i - 1, j - 1, k - 1] for date in dates
        ]
        return pd.DataFrame(
            values, index=dates, columns=["{}@({}, {}, {})".format(prop, i, j, k)]
        )

    def read_vectors(self):
        smspec = self.filename + ".SMSPEC"
        nlist, ni, nj, nk = EclArray(smspec, keyword="DIMENS", with_fakes=False).array[
            :4
        ]
        logging.debug("nlist: {}, ni: {}, nj: {}, nk: {}".format(nlist, ni, nj, nk))
        keywords = byte2str(EclArray(smspec, keyword="KEYWORDS", with_fakes=True).array)
        logging.debug("keywords: {}".format(keywords))
        wgnames = byte2str(EclArray(smspec, keyword="WGNAMES", with_fakes=True).array)
        logging.debug("wgnames: {}".format(wgnames))
        nums = EclArray(smspec, keyword="NUMS", with_fakes=True).array
        logging.debug("nums: {}".format(nums))
        units = byte2str(EclArray(smspec, keyword="UNITS", with_fakes=True).array)
        logging.debug("units: {}".format(units))
        logging.debug("LENGTHS")
        logging.debug("-------")
        logging.debug("keywords: {}".format(len(keywords)))
        logging.debug("wgnames: {}".format(len(wgnames)))
        logging.debug("nums: {}".format(len(nums)))
        logging.debug("units: {}".format(len(units)))
        logging.debug("ZIPS")
        logging.debug("-------")
        for i in zip(keywords, wgnames, nums, units):
            logging.warning(i)
        new_nums = []
        for keyword, num in zip(keywords, nums):
            if (keyword.startswith("C") or keyword.startswith("B")) and num > 0:
                k = int((num - 1) / (ni * nj) - 0.00001) + 1
                j = int((num - (k - 1) * ni * nj) / ni - 0.00001) + 1
                i = num - (j - 1) * ni - (k - 1) * ni * nj
                num = str("({0}, {1}, {2})".format(i, j, k))
            else:
                num = str(num)
            new_nums.append(num)
        nums = new_nums
        logging.debug("NUMS CONVERTED")
        logging.debug("-------")
        for i in nums:
            logging.debug(i)
        params = self._read_all_arrays("UNSMRY", "PARAMS", True)
        logging.warning(params)
        headers = pd.MultiIndex.from_tuples(
            list(zip(*[keywords, wgnames, nums, units])),
            names=["Vector", "Well/Group", "Cell/Region", "Units"],
        )
        df = pd.DataFrame(params, columns=headers).sort_index(axis=1)
        df.index.name = "MINISTEP"
        self.vectors_df = df
        return df

    def get_vectors_shape(self):
        if self.vectors_df is not None:
            return self.vectors_df.shape
        else:
            return None

    def get_vector_names(self):
        if self.vectors_df is not None:
            return sorted(set(self.vectors_df.columns.get_level_values(0)))
        else:
            return None

    def get_vector_column(self, vector_name):
        if self.vectors_df is not None:
            vector = self.vectors_df[[vector_name]]  # get a vector
            vector_us = vector.unstack()  # unstack the multi index df
            vector_us_ri = vector_us.reset_index()  # reset the index
            ser = vector_us_ri[0]  # extract first column. it's a series
            # blank_index = [''] * len(ser)
            # ser.index = blank_index
            ser.reset_index(drop=True, inplace=True)
            ser.name = vector_name
            return pd.DataFrame(ser)  # convert to dataframe


def find_first_numeric_row(df):
    """Find the first row in the DataFrame where all data is numeric."""
    for i in range(len(df)):
        if df.iloc[i].apply(np.isreal).all():
            return i
    return None


def get_dyna(steppi, well_indices, swatuse):
    mean_big_all = []
    for xx in range(steppi):
        mean_big = []  # Collects mean values for this particular timestep
        for idx, list1 in well_indices.items():  # Direct access to lists via .items()
            temp_perm_values = [
                swatuse[xx, i_idx, j_idx, k_idx]
                if k_idx == l_idx
                else swatuse[xx, i_idx, j_idx, k_idx : l_idx + 1]
                for i_idx, j_idx, k_idx, l_idx in list1
            ]
            mean_all = np.mean(temp_perm_values)
            mean_big.append(mean_all)
        mean_big_all.append(mean_big)
    outt2 = np.array(mean_big_all)
    return outt2


def Get_data_FFNN1(
    folder,
    oldfolder,
    N,
    pressure,
    Sgas,
    Swater,
    Soil,
    perm,
    Time,
    steppi,
    steppi_indices,
    N_pr,
    producer_wells,
    unique_entries,
    filenameui,
    well_measurements,
    lenwels,
):
    well_indices = process_data(unique_entries)
    ouut = np.zeros((N, steppi, lenwels * N_pr))
    innn = np.zeros((N, steppi, (4 * N_pr) + 2))
    producer_well_names = [well[-1] for well in producer_wells]
    for i in range(N):
        os.chdir(folder)
        unsmry_file = filenameui
        parser = EclBinaryParser(unsmry_file)
        vectors = parser.read_vectors()
        namez = well_measurements  # ['WOPR', 'WWPR', 'WGPR']
        all_arrays = []
        for namey in namez:
            dfaa = vectors[namey]
            filtered_columns = [
                coll
                for coll in dfaa.columns
                if any(well_namee in coll for well_namee in producer_well_names)
            ]
            filtered_df = dfaa[filtered_columns]
            filtered_df = filtered_df[producer_well_names]
            # Extract numeric data and convert to numpy array
            start_row = find_first_numeric_row(filtered_df)
            if start_row is not None:
                numeric_df = filtered_df.iloc[start_row:]
                result_array = numeric_df.to_numpy()
                logger = setup_logging()
                logger.info(f"Numeric data from {namey} processed successfully.")
            else:
                logger = setup_logging()
                logger.info(f"No numeric rows found in the DataFrame for {namey}.")
                result_array = None
            all_arrays.append(result_array)
        final_array = np.concatenate(all_arrays, axis=1)
        final_array[final_array <= 0] = 0
        out = final_array[steppi_indices - 1, :].astype(float)
        out[out <= 0] = 0
        ouut[i, :, :] = out
        permuse = perm[i, 0, :, :, :]
        presure_use = pressure[i, :, :, :, :]
        gas_use = Sgas[i, :, :, :, :]
        water_use = Swater[i, :, :, :, :]
        oil_use = Soil[i, :, :, :, :]
        Time_use = Time[i, :, :, :, :]
        mean_big = []
        for idx, indices_list in well_indices.items():
            values = [
                permuse[i_idx, j_idx, k_idx]
                if k_idx == l_idx
                else permuse[i_idx, j_idx, k_idx : l_idx + 1]
                for i_idx, j_idx, k_idx, l_idx in indices_list
            ]
            mean_big.append(np.mean(values))
        permxx = np.tile(mean_big, (steppi, 1))
        a3 = get_dyna(steppi, well_indices, water_use)
        a2 = get_dyna(steppi, well_indices, gas_use)
        a5 = get_dyna(steppi, well_indices, oil_use)
        a1 = np.zeros((steppi, 1))
        a4 = np.zeros((steppi, 1))
        for k in range(steppi):
            uniep = presure_use[k, :, :, :]
            permuse = uniep
            a1[k, 0] = np.mean(permuse)
            unietime = Time_use[k, :, :, :]
            permuse = unietime
            a4[k, 0] = permuse[0, 0, 0]
        inn1 = np.hstack((permxx, a1, a5, a2, a3, a4))
        innn[i, :, :] = inn1
        os.chdir(oldfolder)
    return innn, ouut


def endit(i, testt, training_master, oldfolder, pred_type, degg, big, experts, device):
    logger = setup_logging()
    logger.info("")
    logger.info("Starting prediction from machine %d" % (i + 1))
    numcols = len(testt[0])
    izz = PREDICTION_CCR__MACHINE(
        i,
        big,
        testt,
        numcols,
        training_master,
        oldfolder,
        pred_type,
        degg,
        experts,
        device,
    )
    logger.info("")
    logger.info("Finished Prediction from machine %d" % (i + 1))
    return izz


def predict_machine(a0, model):
    ynew = model.predict(xgb.DMatrix(a0))
    return ynew


def predict_machine3(a0, deg, model, poly):
    predicted = model.predict(poly.fit_transform(a0))
    return predicted


class SparseGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_points):
        super(SparseGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        self.variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            self.variational_distribution,
            learn_inducing_locations=True,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _cholesky(self, A):
        """✅ Fix Cholesky decomposition issue with jitter"""
        jitter = 1e-5  # Small positive value
        eye = torch.eye(A.size(-1), device=A.device)
        return psd_safe_cholesky(A + jitter * eye)  # ✅ Safe Cholesky

    def _mean_cache(self):
        """✅ Uses safe Cholesky for covariance matrix"""
        train_train_covar = self.train_train_covar.evaluate_kernel()
        train_labels_offset = self.train_labels - self.train_mean
        jitter = 1e-5
        identity = torch.eye(
            train_train_covar.size(-1), device=train_train_covar.device
        )
        chol = self._cholesky(train_train_covar + jitter * identity)
        return torch.cholesky_solve(train_labels_offset, chol).squeeze(-1)


def fit_Gp(X, y, device, itery, percentage=50.0):
    X = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    X_clone = X.clone()
    with torch.no_grad():
        X_np = X_clone.cpu().numpy()  # Now safe to convert to NumPy
        num_inducing_points = max(
            int(X_np.shape[0] * (percentage / 100)), 1
        )  # Ensure at least one inducing point
        kmeans = MiniBatchKMeans(
            n_clusters=num_inducing_points, random_state=42, n_init="auto"
        )
        kmeans.fit(X_np)  # Uses clone, keeps autograd
        inducing_points = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=device
        )  # Move centroids to GPU
    likelihood = GaussianLikelihood().to(device)
    model = SparseGPModel(X, y, likelihood, inducing_points).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=2e-2, betas=(0.9, 0.999), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99998708)

    mll = VariationalELBO(likelihood, model, num_data=y.size(0))
    model.train()
    likelihood.train()
    for epoch in range(itery):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss = loss.mean()  # Ensure loss is a scalar
        loss.backward(retain_graph=True)  # Keep the graph intact
        optimizer.step()
        scheduler.step()
        del loss  # Free memory
        torch.cuda.empty_cache()
    return model


def PREDICTION_CCR__MACHINE(
    ii,
    nclusters,
    inputtest,
    numcols,
    training_master,
    oldfolder,
    pred_type,
    deg,
    experts,
    device,
):
    filenamex = "clfx_%d.asv" % ii
    filenamey = "clfy_%d.asv" % ii
    os.chdir(training_master)
    if experts == 1 or experts == 3:
        filename1 = "Classifier_%d.bin" % ii
        loaded_model = xgb.Booster({"nthread": 4})  # init model
        loaded_model.load_model(filename1)  # load data
    if experts == 2:
        filename1 = "Classifier_%d.pkl" % ii
        with open(filename1, "rb") as file:
            loaded_model = pickle.load(file)
    clfx = pickle.load(open(filenamex, "rb"))
    clfy = pickle.load(open(filenamey, "rb"))
    os.chdir(oldfolder)
    inputtest = clfx.transform(inputtest)
    if experts == 2:
        labelDA = loaded_model.predict(inputtest)
    else:
        labelDA = loaded_model.predict(xgb.DMatrix(inputtest))
        if nclusters == 2:
            labelDAX = 1 - labelDA
            labelDA = np.reshape(labelDA, (-1, 1))
            labelDAX = np.reshape(labelDAX, (-1, 1))
            labelDA = np.concatenate((labelDAX, labelDA), axis=1)
            labelDA = np.argmax(labelDA, axis=-1)
        else:
            labelDA = np.argmax(labelDA, axis=-1)
        labelDA = np.reshape(labelDA, (-1, 1), "F")
    numrowstest = len(inputtest)
    operationanswer = np.zeros((numrowstest, 1))
    labelDA = np.reshape(labelDA, (-1, 1), "F")
    for i in range(nclusters):
        logging.getLogger(__name__).info(
            "-- Predicting cluster: %s | %s", str(i + 1), str(nclusters)
        )
        if experts == 1:  # Polynomial regressor experts
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            filename2b = "polfeat_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            os.chdir(training_master)
            with open(filename2, "rb") as file:
                model0 = pickle.load(file)
            with open(filename2b, "rb") as filex:
                poly0 = pickle.load(filex)
            os.chdir(oldfolder)
            labelDA0 = (np.asarray(np.where(labelDA == i))).T
            #    ##----------------------##------------------------##
            a00 = inputtest[labelDA0[:, 0], :]
            a00 = np.reshape(a00, (-1, numcols), "F")
            if a00.shape[0] != 0:
                operationanswer[labelDA0[:, 0], :] = np.reshape(
                    predict_machine3(a00, deg, model0, poly0), (-1, 1)
                )
        elif experts == 2:
            labelDA0 = (np.asarray(np.where(labelDA == i))).T
            #    ##----------------------##------------------------##
            a00 = inputtest[labelDA0[:, 0], :]
            a00 = np.reshape(a00, (-1, numcols), "F")
            a00 = torch.tensor(a00, dtype=torch.float32).to(device)
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pth"
            os.chdir(training_master)
            checkpoint = torch.load(filename2, map_location=device)
            num_inducing_points = checkpoint[
                "variational_strategy.inducing_points"
            ].shape[0]
            input_dim = checkpoint["variational_strategy.inducing_points"].shape[1]
            # output_dim = 1  # Assuming a single output per sample
            train_x = torch.zeros(a00.shape[0], input_dim).to(device)
            train_y = torch.zeros(a00.shape[0], 1).to(device)
            train_y = train_y.squeeze(-1)
            likelihood = GaussianLikelihood().to(device)
            inducing_points = torch.zeros(num_inducing_points, input_dim).to(device)
            model = SparseGPModel(train_x, train_y, likelihood, inducing_points).to(
                device
            )
            model.load_state_dict(checkpoint, strict=False)  # ✅ Pass strict=False here
            os.chdir(oldfolder)
            model = model.to(device)
            model.eval()
            batch_size = 1  # Adjust based on memory availability
            predictions = []
            if a00.shape[0] != 0:
                with torch.no_grad():
                    for i in range(0, a00.shape[0], batch_size):
                        batch = a00[i : i + batch_size]  # Take a batch of inputs
                        prediction = model(batch)  # Forward pass
                        pred = prediction.mean.detach().cpu().numpy()
                        predictions.append(pred)  # Store batch predictions
                operationanswer[labelDA0[:, 0], :] = np.vstack(predictions)
            torch.cuda.empty_cache()  # Free unused GPU memory
        else:  # XGBoost experts
            loaded_modelr = xgb.Booster({"nthread": 4})  # init model
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".bin"
            os.chdir(training_master)
            loaded_modelr.load_model(filename2)  # load data
            os.chdir(oldfolder)
            labelDA0 = (np.asarray(np.where(labelDA == i))).T
            a00 = inputtest[labelDA0[:, 0], :]
            a00 = np.reshape(a00, (-1, numcols), "F")
            if a00.shape[0] != 0:
                operationanswer[labelDA0[:, 0], :] = np.reshape(
                    predict_machine(a00, loaded_modelr), (-1, 1)
                )
    operationanswer = clfy.inverse_transform(operationanswer)
    return operationanswer
