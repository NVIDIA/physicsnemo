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
import sys
import math
import logging
import re
import shutil
import warnings
from struct import unpack, unpack_from
from mmap import mmap
from collections import namedtuple
import subprocess
import fnmatch
from scipy.linalg import norm


# Numerical Computing
import numpy as np
import numpy.matlib
import numpy.ma as ma
import pandas as pd
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


# def simulation_data_types():
"""Return common Eclipse/Flow dictionaries for parsing keywords."""
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


def byte2str(x):
    """Convert bytes-like or nested collections to strings."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(map(byte2str, x))
    else:
        return str(x)[2:-1].strip()


def str2byte(x):
    """Pad and uppercase strings to 8-byte Eclipse keyword format."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(map(str2byte, x))
    else:
        return bytes(x.ljust(8).upper(), "utf-8")


def filter_fakes(filename, ext, loc, target_size, fmt=">f", excl=np.inf):
    """Read and filter arrays from Eclipse binary sections skipping fakes."""
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


class EclArray(object):
    def __init__(self, filename, offset=None, keyword=None, with_fakes=True):
        self.filename = filename
        if (offset is None and keyword is None) or (
            offset is not None and keyword is not None
        ):
            raise ValueError("Either offset or keyword must be specified")
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
            logger.info(
                "No {0} value at {1}. Assuming zero for plotting".format(prop, date)
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
            ser.reset_index(drop=True, inplace=True)
            ser.name = vector_name
            return pd.DataFrame(ser)  # convert to dataframe


def is_available():
    """Return 0 if nvidia-smi successful, else non-zero code."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        code = result.returncode
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        code = 1
    return code


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
    """Return percentage string for a given progress ratio."""
    try:
        Progress = float(Progress) / float(Total)
        if Progress >= 1.0:
            Progress = 1
            return "100%"
        return "{:.0f}%".format(round(Progress * 100, 0))
    except Exception as e:
        logger.error(f"Error in is_available: {e}")
        return "ERROR"


def ShowBar(Bar):
    """Write a progress bar string to stdout without newline."""
    sys.stdout.write(Bar)
    sys.stdout.flush()


def load_data_numpy(inn, batch_size):
    """Wrap a single tensor array into a DataLoader for batching."""
    x_data = inn
    logger.info(f"x_data: {x_data.shape}")
    data_tuple = (torch.FloatTensor(x_data),)
    data_loader = DataLoader(
        TensorDataset(*data_tuple), batch_size=batch_size, shuffle=True, drop_last=True
    )
    return data_loader


def Pkgen(n):
    """Return power-law function k -> k^{-n}."""

    def Pk(k):
        return np.power(k, -n)

    return Pk


def distrib(shape):
    """Return complex Gaussian white noise array of given shape."""
    rng = np.random.default_rng()
    a = rng.normal(loc=0, scale=1, size=shape)
    b = rng.normal(loc=0, scale=1, size=shape)
    return a + 1j * b


class RMS:
    def __init__(self, truth, ensemble):
        mean = ensemble.mean(axis=0)
        err = truth - mean
        dev = ensemble - mean
        self.rmse = norm(err)
        self.rmsd = norm(dev)

    def __str__(self):
        return "%6.4f (rmse),  %6.4f (std)" % (self.rmse, self.rmsd)


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
    elif distrib.lower() == "lhs":
        return (interval[1] - interval[0]) * lhs(
            nder, samples=n_test, **kwargs
        ) + interval[0]
    else:
        raise ValueError(f"Unknown distribution: {distrib}")


def Reinvent(matt):
    """Reorder 3D array axes from (Z,X,Y) to (X,Y,Z)."""
    nx, ny, nz = matt.shape[1], matt.shape[2], matt.shape[0]
    dess = np.zeros((nx, ny, nz))
    for i in range(nz):
        dess[:, :, i] = matt[i, :, :]
    return dess


def fit_operation(tensor, target_min, target_max, tensor_min, tensor_max):
    """Scale tensor linearly by provided min/max bounds."""
    rescaled_tensor = tensor / tensor_max
    return rescaled_tensor


def scale_operation(tensor, target_min, target_max):
    """Replace invalids and scale by max value returning (min, max, scaled)."""
    tensor[np.isnan(tensor)] = 0  # Replace NaN with 0
    tensor[np.isinf(tensor)] = 0  # Replace infinity with 0
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    rescaled_tensor = tensor / max_val
    return min_val, max_val, rescaled_tensor


threshold = np.finfo(np.float32).max


def replace_large_and_invalid_values(arr, placeholder=0.0):
    """Replace NaN/Inf/overflow values with placeholder."""
    invalid_indices = (np.isnan(arr)) | (np.isinf(arr)) | (np.abs(arr) > threshold)
    arr[invalid_indices] = placeholder
    return arr


def clean_dict_arrays(data_dict):
    """Apply replace_large_and_invalid_values to each dict entry in-place."""
    for key in data_dict:
        data_dict[key] = replace_large_and_invalid_values(data_dict[key])
    return data_dict


def convert_back(rescaled_tensor, target_min, target_max, min_val, max_val):
    """Undo normalisation using max value (simple inverse for scale_operation)."""
    return rescaled_tensor * max_val


def replace_nans_and_infs(tensor, value=0.0):
    """Replace NaNs/Infs in a torch tensor with a constant value."""
    tensor[torch.isnan(tensor) | torch.isinf(tensor)] = value
    return tensor


threshold = np.finfo(np.float32).max


def clip_and_convert_to_float32(array):
    """Clip to float32 range and return array.astype(np.float32)."""
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min
    array_clipped = np.clip(array, min_float32, max_float32)
    return array_clipped.astype(np.float32)


def clip_and_convert_to_float3(array):
    """Alias for float32 clipping for historical compatibility."""
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min
    array_clipped = np.clip(array, min_float32, max_float32)
    return array_clipped.astype(np.float32)


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


SUPPORTED_DATA_TYPES = {
    "INTE": (4, "i", 1000),
    "REAL": (4, "f", 1000),
    "LOGI": (4, "i", 1000),
    "DOUB": (8, "d", 1000),
    "CHAR": (8, "8s", 105),
    "MESS": (8, "8s", 105),
    "C008": (8, "8s", 105),
}


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
    """Return (elem_size, fmt, skip) tuple for a supported Eclipse datatype."""
    try:
        return SUPPORTED_DATA_TYPES[data_type]
    except KeyError as exc:
        raise ValueError("Unknown datatype %s." % data_type) from exc


def _check_and_fetch_file(path, pattern, return_relative=False):
    """Return files in `path` matching `pattern` regex (case-insensitive)."""
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
    """Parse binary file and collect sections for requested keyword `attrs`."""
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
    """Parse Eclipse binary file and return decoded sections for `attrs`."""
    if attrs is None:
        raise ValueError("Keyword attribute cannot be empty")
    if isinstance(attrs, str):
        attrs = [attrs]
    attrs = [attr.strip().upper() for attr in attrs]
    _, sections = _parse_keywords(path, attrs)
    return sections


def _fetch_keyword_data(section):
    """Decode a single keyword section accounting for skip padding."""
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


def copy_files(source_dir, dest_dir):
    """Copy all files from `source_dir` into `dest_dir`."""
    files = os.listdir(source_dir)
    for file in files:
        shutil.copy(os.path.join(source_dir, file), dest_dir)


def find_first_numeric_row(df):
    """Return first row index where all values are numeric, else None."""
    """Find the first row in the DataFrame where all data is numeric."""
    for i in range(len(df)):
        if df.iloc[i].apply(np.isreal).all():
            return i
    return None


def process_dataframe(name, producer_well_names, vectors):
    """Extract numeric data for a vector and TIME column from UNSMRY."""
    df = vectors[name]
    filtered_columns = [
        col
        for col in df.columns
        if any(well_name in col for well_name in producer_well_names)
    ]
    filtered_df = df[filtered_columns]
    start_row = find_first_numeric_row(filtered_df)
    if start_row is not None:
        numeric_df = filtered_df.iloc[start_row:]
        result_array = numeric_df.to_numpy()
        logger.info(f"Numeric data from {name} processed successfully.")
    else:
        logger.info(f"No numeric rows found in the DataFrame for {name}.")
        result_array = None
    Time = vectors["TIME"]
    start_row = find_first_numeric_row(Time)
    if start_row is not None:
        numeric_df = Time.iloc[start_row:]
        time_array = numeric_df.to_numpy()
        logger.info(f"Numeric data from {name} processed successfully.")
    else:
        logger.info(f"No numeric rows found in the DataFrame for {name}.")
        time_array = None
    return result_array, time_array


def remove_rows(matrix, indices_to_remove):
    """Remove rows by index from 2D `matrix` and return the reduced array."""
    matrix = np.delete(matrix, indices_to_remove, axis=0)
    return matrix


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
    with open(filename3, "r") as file:
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
    with open(filename3, "w") as file:
        file.writelines(lines)
    try:
        result = subprocess.run(
            string_operat2, shell=True, capture_output=True, text=True, timeout=30
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
        filtered_pressure, filtered_swat, filtered_sgas
    ):
        for state_var, all_slices in zip(
            [pr_slice, sw_slice, sg_slice], [pressure, swat, sgas]
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
    soil = abs(1 - abs(swat + sgas))
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
    jesuni = []
    for k in range(lenwels):
        quantt = quant_big[f"K_{k}"]
        # ajes = quantt["value"]
        if quantt["boolean"] == 1:
            kodsval = out[:, k * N_pr : (k + 1) * N_pr] / quantt["scale"]
        else:
            kodsval = out[:, k * N_pr : (k + 1) * N_pr] * quantt["scale"]
        jesuni.append(kodsval)
    spit = np.hstack(jesuni)
    spit = np.reshape(spit, (-1, 1), "F")
    spit = remove_rows(spit, rows_to_remove).reshape(-1, 1)
    use = np.reshape(spit, (-1, 1), "F")
    os.chdir(oldfolder)
    return pressure, swat, soil, sgas, out, use


def Get_new_K(Low_K, High_K, LogS1):
    """Linearly mix lower/upper bounds using LogS1 mask to produce K."""
    newK = (High_K * LogS1) + (1 - LogS1) * Low_K
    return newK


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


def Split_Matrix(matrix, sizee):
    """Split matrix along axis 0 into equally sized chunks."""
    x_split = np.split(matrix, sizee, axis=0)
    return x_split


def shuffle(x, axis=0):
    """Shuffle an array along `axis` using a random generator."""
    n_axis = len(x.shape)
    t = np.arange(n_axis)
    t[0] = axis
    t[axis] = 0
    xt = np.transpose(x.copy(), t)
    rng = np.random.default_rng()
    rng.shuffle(xt)
    shuffled_x = np.transpose(xt, t)
    return shuffled_x


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
