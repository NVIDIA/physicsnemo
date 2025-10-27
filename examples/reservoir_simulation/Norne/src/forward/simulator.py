"""
SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
                            SIMULATOR MODULE
=====================================================================

This module provides simulation capabilities for reservoir forward modeling.
It includes functions for data type definitions, binary file processing,
ensemble generation, and simulation utilities.

Key Features:
- Simulation data type definitions
- Binary file reading and processing
- Ensemble generation and manipulation
- Statistical utilities and noise generation
- Data validation and processing

Usage:
    from forward.simulator import (
        simulation_data_types,
        EclArray,
        EclBinaryParser,
        NorneInitialEnsemble,
        gaussian_with_variable_parameters,
        add_gnoise,
        adjust_variable_within_bounds
    )

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import logging
import warnings
import re
from struct import unpack_from
from collections import namedtuple
from mmap import mmap

# 🔧 Third-party Libraries
import yaml
import scipy
import numpy as np
import numpy.linalg
import numpy.matlib
import pandas as pd
from pyDOE import lhs
import numpy.ma as ma

# 🔥 Torch & PhyNeMo
import torch

# 📦 Local Modules


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


def simulation_data_types():
    """Return common Eclipse/Flow dictionaries for parsing keywords.

    Provides SUPPORTED_DATA_TYPES, type_dict, ecl_extensions, dynamic_props,
    ecl_vectors, and static_props used by parsing helpers in this module.
    """
    SUPPORTED_DATA_TYPES = {
        "INTE": (4, "i", 1000),
        "REAL": (4, "f", 1000),
        "LOGI": (4, "i", 1000),
        "DOUB": (8, "d", 1000),
        "CHAR": (8, "8s", 105),
        "MESS": (8, "8s", 105),
        "C008": (8, "8s", 105),
    }

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

    return (
        type_dict,
        ecl_extensions,
        dynamic_props,
        ecl_vectors,
        static_props,
        SUPPORTED_DATA_TYPES,
    )


(
    type_dict,
    ecl_extensions,
    dynamic_props,
    ecl_vectors,
    static_props,
    SUPPORTED_DATA_TYPES,
) = simulation_data_types()


def byte2str(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(map(byte2str, x))
    else:
        return str(x)[2:-1].strip()


def get_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1  # If not initialized, assume single GPU


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
            # Take the first date
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
                # Convert NUMs to strings for subsequent plotting
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


def is_valid_vector(vector_name):
    valid_vectors = ecl_vectors
    return vector_name in valid_vectors


def get_shape(t):
    shape = []
    while isinstance(t, tuple):
        shape.append(len(t))
        t = t[0]
    return tuple(shape)


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
    x = np.random.randn(m * n)
    x = np.dot(cholT.T, np.dot(x.reshape(m, n), cholT2))
    x = x.flatten()
    if np.max(np.size(Sdev)) > 1:
        if np.min(np.shape(Sdev)) == 1 and len(Sdev) == len(x):
            x = Sdev * x
        else:
            raise ValueError("FastGaussian: Inconsistent dimension of Sdev")
    return x


def NorneInitialEnsemble(nx, ny, nz, ensembleSize=100, randomNumber=1.2345e5):
    np.random.seed(int(randomNumber))
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
        X = M_MF + S_MF * np.random.randn(53)
        ensemblefault[:, i] = X
        C = np.array(C)
        X1 = gaussian_with_variable_parameters(D, np.zeros(N_F), 1, C[0], C_S)[0]
        X1 = X1.reshape(-1, 1)
        ensembleporo[indices, i] = (M[0] + S[0] * X1[indices]).ravel()
        X2 = gaussian_with_variable_parameters(D, np.zeros(N_F), 1, C[1], C_S)[0]
        X2 = X2.reshape(-1, 1)
        X = R1 * X1 + np.sqrt(1 - R1**2) * X2
        indices = np.where(A == 1)
        ensembleperm[indices, i] = (M[1] + S[1] * X[indices]).ravel()
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


def add_gnoise(Ytrue, SIGMA, SQ=None):
    try:
        if SQ is not None and SQ == 1:
            # Use SIGMA*SIGMA' as covariance matrix
            RTSIGMA = SIGMA
            if np.isscalar(SIGMA) or np.ndim(SIGMA) == 1:
                # SIGMA is a scalar or vector
                error = RTSIGMA * np.random.randn(1)
            else:
                error = RTSIGMA @ np.random.randn(RTSIGMA.shape[1], 1)
        else:
            # Use SIGMA as covariance matrix
            if np.isscalar(SIGMA) or np.ndim(SIGMA) == 1:
                # SIGMA is entered as a scalar or a vector
                RTSIGMA = np.sqrt(SIGMA)
                error = RTSIGMA * np.random.randn(*Ytrue.shape)
            else:
                # The matrix must be transposed.
                try:
                    RTSIGMA = np.linalg.cholesky(SIGMA).T
                except np.linalg.LinAlgError:
                    logger = setup_logging()
                    logger.warning("Problem with Cholesky factorization")
                    RTSIGMA = np.sqrtm(SIGMA).real
                    logger.info("Finally - we got a square root!")

                error = RTSIGMA @ np.random.randn(*Ytrue.shape)
        Y = Ytrue + error.flatten()
    except Exception as e:
        logger = setup_logging()
        logger.error("Error in AddGnoise")
        raise e
    return Y, RTSIGMA


def adjust_variable_within_bounds(variable, lowerbound=None, upperbound=None):
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


def read_until_line(file_path, sep=r"\s+", header=None):
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
                # Append cleaned line to data list
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


def Reinvent(matt):
    nx, ny, nz = matt.shape[1], matt.shape[2], matt.shape[0]
    dess = np.zeros((nx, ny, nz))
    for i in range(nz):
        dess[:, :, i] = matt[i, :, :]
    return dess


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


def loss_compute_abs(a, b):
    loss = torch.sum(torch.abs(a - b) / a.shape[0])
    return loss


def compute_metrics(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    TSS = np.sum((y_true - y_true_mean) ** 2)
    RSS = np.sum((y_true - y_pred) ** 2)
    R2 = 1 - (RSS / TSS)
    L2_accuracy = 1 - np.sqrt(RSS) / np.sqrt(TSS)
    return R2 * 100, L2_accuracy * 100


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


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def round_array_to_4dp(arr):
    try:
        # Convert input to a numpy array if it's not already
        arr = np.asarray(arr)
        return np.around(arr, 4)
    except Exception as e:
        logger = setup_logging()
        logger.error(f"An error occurred: {str(e)}")
        return None  # You can choose to return None or handle the error differently


def Getit(data, input_keys, output_keys, output_keys2):
    logger = setup_logging()
    _ks = [k for k in data.keys() if not k.startswith("__")]
    logger.info(f"\navaliable keys: {_ks}")
    invar, outvar, outvar2 = dict(), dict(), dict()
    for d, keys in [
        (invar, input_keys),
        (outvar, output_keys),
        (outvar2, output_keys2),
    ]:
        for k in keys:
            x = data[k]  # N, C, H, W
            logger.info(f"selected key: {k}, mean: {x.mean():.5e}, std: {x.std():.5e}")
            d[k] = x
    return (invar, outvar, outvar2)


def Getit2(data, input_keys, output_keys):
    logger = setup_logging()
    _ks = [k for k in data.keys() if not k.startswith("__")]
    logger.info(f"\navaliable keys: {_ks}")
    invar, outvar = dict(), dict()
    for d, keys in [(invar, input_keys), (outvar, output_keys)]:
        for k in keys:
            x = data[k]  # N, C, H, W
            logger.info(f"selected key: {k}, mean: {x.mean():.5e}, std: {x.std():.5e}")
            d[k] = x
    del data
    return (invar, outvar)


def calc_mu_g(p):
    # Average reservoir pressure
    mu_g = 3e-6 * p**2 + 1e-6 * p + 0.0133
    return mu_g


def calc_rs(p_bub, p, device):
    device1 = device
    rs_factor = torch.where(
        p < p_bub,
        torch.tensor(1.0).to(device1, torch.float32),
        torch.tensor(1e-6).to(device1, torch.float32),
    )
    rs = (178.11**2) / 5.615 * (torch.pow(p / p_bub, 1.3) * rs_factor + (1 - rs_factor))
    return rs


def calc_dp(p_bub, p_atm, p):
    dp = torch.where(p < p_bub, p_atm - p, p_atm - p_bub)
    return dp


def calc_bg(p_bub, p_atm, p):
    # P is average reservoir pressure
    b_g = torch.divide(1, torch.exp(1.7e-3 * calc_dp(p_bub, p_atm, p)))
    return b_g


def calc_bo(p_bub, p_atm, CFO, p):
    # p is average reservoir pressure
    exp_term1 = torch.where(p < p_bub, -8e-5 * (p_atm - p), -8e-5 * (p_atm - p_bub))
    exp_term2 = -CFO * torch.where(p < p_bub, torch.zeros_like(p), p - p_bub)
    b_o = torch.divide(1, torch.exp(exp_term1) * torch.exp(exp_term2))
    return b_o


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
    logger = setup_logging()
    logger.info(f"Read simulation plan from {fname}...")
    with open(fname, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            # logger.debug(data)
        except yaml.YAMLError as exc:
            logger.error(exc)
        return data


def fit_scale_abs(tensor, target_min, target_max, tensor_min, tensor_max):
    # Rescale between target min and target max
    rescaled_tensor = tensor / tensor_max
    return rescaled_tensor


def StoneIIModel(params, device, Sg, Sw):
    # device = params["device"]
    k_rwmax = params["k_rwmax"].to(device)
    k_romax = params["k_romax"].to(device)
    k_rgmax = params["k_rgmax"].to(device)
    n = params["n"].to(device)
    p = params["p"].to(device)
    q = params["q"].to(device)
    m = params["m"].to(device)
    Swi = params["Swi"].to(device)
    Sor = params["Sor"].to(device)
    denominator = 1 - Swi - Sor
    krw = k_rwmax * ((Sw - Swi) / denominator).pow(n)
    kro = (
        k_romax * (1 - (Sw - Swi) / denominator).pow(p) * (1 - Sg / denominator).pow(q)
    )
    krg = k_rgmax * (Sg / denominator).pow(m)
    return krw, kro, krg


def compute_peacemannoil(
    UO,
    BO,
    UW,
    BW,
    DZ,
    RE,
    device,
    max_inn_fcn,
    max_out_fcn,
    paramz,
    p_bub,
    p_atm,
    steppi,
    CFO,
    sgas,
    swater,
    pressure,
    permeability,
):
    qoil = torch.zeros_like(sgas).to(device)
    skin = 0
    rwell = 200
    pwf_producer = 100

    def process_location(i, j, k, l_index):
        pre1 = pressure[i, j, :, :, :]
        sg1 = sgas[i, j, :, k, l_index]
        sw1 = swater[i, j, :, k, l_index]
        krw, kro, krg = StoneIIModel(paramz, device, sg1, sw1)
        BO_val = calc_bo(p_bub, p_atm, CFO, pre1.mean())
        up = UO * BO_val
        perm1 = permeability[i, 0, :, k, l_index]
        down = 2 * torch.pi * perm1 * kro * DZ
        right = torch.log(RE / rwell) + skin
        J = down / (up * right)
        drawdown = pre1.mean() - pwf_producer
        qoil1 = torch.abs(-(drawdown * J))
        return -qoil1

    locations = [
        (14, 30),
        (9, 31),
        (13, 33),
        (8, 36),
        (8, 45),
        (9, 28),
        (9, 23),
        (21, 21),
        (13, 27),
        (18, 37),
        (18, 53),
        (15, 65),
        (24, 36),
        (18, 53),
        (11, 71),
        (17, 67),
        (12, 66),
        (37, 97),
        (6, 63),
        (14, 75),
        (12, 66),
        (10, 27),
    ]
    for m in range(sgas.shape[0]):
        for step in range(sgas.shape[1]):
            for location in locations:
                qoil[m, step, :, location[0], location[1]] = process_location(
                    m, step, *location
                )
    return qoil


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


def RelPerm(Sa, Sg, SWI, SWR, SWOW, SWOG):
    one_minus_swi_swr = 1 - (SWI + SWR)
    so = ((1 - (Sa + Sg)) - SWR) / one_minus_swi_swr
    sw = (Sa - SWI) / one_minus_swi_swr
    sg = Sg / one_minus_swi_swr
    KROW = linear_interp(Sa, SWOW[:, 0], SWOW[:, 1])
    KRW = linear_interp(Sa, SWOW[:, 0], SWOW[:, 2])
    KROG = linear_interp(Sg, SWOG[:, 0], SWOG[:, 1])
    KRG = linear_interp(Sg, SWOG[:, 0], SWOG[:, 2])
    KRO = ((KROW / (1 - sw)) * (KROG / (1 - sg))) * so
    return KRW, KRO, KRG


def NorneGeostat(nx, ny, nz):
    norne = {}

    dim = np.array([nx, ny, nz])
    ldim = dim[0] * dim[1]
    norne["dim"] = dim
    act = read_until_line("../simulator_data/ACTNUM_0704.prop")
    act = act.T
    act = np.reshape(act, (-1,), "F")
    norne["actnum"] = act

    # porosity
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

    # Correlation between layers

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

    # Correlation between porosity and permeability
    norne["poroPermxCorr"] = 0.7

    norne["poroNtgCorr"] = 0.6
    norne["ntgStd"] = 0.1
    norne["ntgLB"] = 0.01
    norne["ntgUB"] = 1
    norne["ntgRange"] = 26

    # rel-perm end-point scaling
    norne["krwMean"] = 1.15
    norne["krwLB"] = 0.8
    norne["krwUB"] = 1.5
    norne["krgMean"] = 0.9
    norne["krgLB"] = 0.8
    norne["krgUB"] = 1

    # oil-water contact
    norne["owcMean"] = np.array([2692.0, 2585.5, 2618.0, 2400.0, 2693.3])
    norne["owcLB"] = norne["owcMean"] - 10
    norne["owcUB"] = norne["owcMean"] + 10

    # region multipliers
    norne["multregtLogMean"] = np.log10(np.array([0.0008, 0.1, 0.05]))
    norne["multregtStd"] = 0.5
    norne["multregtLB"] = -5
    norne["multregtUB"] = 0

    # z-multipliers
    z_means = [-2, -1.3, -2, -2, -2, -2]
    z_stds = [0.5, 0.5, 0.5, 0.5, 1, 1]
    for i, (mean_, std_) in enumerate(zip(z_means, z_stds), start=1):
        norne[f"z{i}Mean"] = mean_
        norne[f"z{i}Std"] = std_
    norne["zLB"] = -4
    norne["zUB"] = 0
    norne["multzRange"] = 26
    # fault multipliers
    norne["multfltStd"] = 0.5
    norne["multfltLB"] = -5
    norne["multfltUB"] = 2
    return norne
