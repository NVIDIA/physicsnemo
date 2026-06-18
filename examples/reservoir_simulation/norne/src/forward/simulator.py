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

import os
import sys
import logging
import warnings
import re
from struct import unpack_from
from collections import namedtuple
from mmap import mmap

import numpy as np
import numpy.linalg
import pandas as pd
import numpy.ma as ma
import torch
from utils.array_utils import linear_interp
from utils.ensemble_utils import read_until_line


def setup_logging() -> logging.Logger:
    """Configure and return the main logger with green INFO console output."""
    logger = logging.getLogger("forward problem")
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
    """Decode byte strings (or collections thereof) to plain Python strings.

    Parameters
    ----------
    x : bytes, list, tuple, or np.ndarray
        Single byte string or a collection of byte strings to decode.

    Returns
    -------
    str or list of str
        Decoded string(s) with leading ``b'`` and trailing ``'`` stripped.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(map(byte2str, x))
    return str(x)[2:-1].strip()


def get_world_size():
    """Return the number of processes in the current distributed group.

    Returns
    -------
    int
        World size if distributed training is initialised, otherwise 1.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1  # If not initialized, assume single GPU


def str2byte(x):
    """Encode plain strings (or collections thereof) to 8-byte upper-case byte strings.

    Parameters
    ----------
    x : str, list, tuple, or np.ndarray
        Single string or collection of strings to encode.

    Returns
    -------
    bytes or list of bytes
        UTF-8 encoded, left-justified, 8-character upper-case byte string(s).
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(map(str2byte, x))
    return bytes(x.ljust(8).upper(), "utf-8")


def filter_fakes(filename, ext, loc, target_size, fmt=">f", excl=np.inf):
    """Read binary Eclipse data and remove padding/fake values to reach ``target_size``.

    Parameters
    ----------
    filename : str
        Base path of the Eclipse file (without extension).
    ext : str
        File extension to append (e.g., ``".UNSMRY"``).
    loc : int
        Byte offset of the keyword header in the file.
    target_size : int
        Number of valid values expected after filtering.
    fmt : str, optional
        NumPy/struct format string for reading (default ``">f"`` for big-endian float).
    excl : float, optional
        Absolute value threshold above which entries are excluded. Default is ``np.inf``.

    Returns
    -------
    np.ndarray
        Array of valid values with fake/padding entries removed.
    """
    with open(filename + ext, "rb") as f:
        n_sieved = 0  # Number of fake values, initially zero
        tol = 1e-32
        array = np.array([])
        lengths = [0, 1]
        while len(array) < target_size and lengths[-1] != lengths[-2]:
            f.seek(loc + 24)
            array = np.fromfile(f, dtype=fmt, count=target_size + n_sieved)
            if fmt == "S8":  # For read_vectors
                condition = np.array([not _.startswith("\\") for _ in byte2str(array)])
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


class EclArray:
    """Ecl Array."""

    def __init__(self, filename, offset=None, keyword=None, with_fakes=True):
        """Read one Eclipse binary array from *filename*.

        Parameters
        ----------
        filename : str
            Path to the Eclipse binary file (.INIT, .UNRST, etc.).
        offset : int or None
            Byte offset at which the array header starts. Mutually exclusive with *keyword*.
        keyword : str or None
            Eclipse keyword string (e.g. ``"PORV"``). Mutually exclusive with *offset*.
        with_fakes : bool
            Whether to include fake/padding entries returned by ``filter_fakes``.
        """
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


class EclBinaryParser:
    """Ecl Binary Parser."""

    def __init__(self, filename):
        """Initialise a parser for Eclipse binary restart/summary files.

        Parameters
        ----------
        filename : str
            Path to an Eclipse binary file, with or without extension.
        """
        self.vectors_df = None  # this will take the shape of the vectors
        if (
            isinstance(os.path.splitext(filename), tuple)
            and os.path.splitext(filename)[1] in ecl_extensions
        ):
            self.filename = os.path.splitext(filename)[0]
        else:
            self.filename = filename

    def _read_all_arrays(self, ext, keyword, with_fakes):
        """Read all binary arrays matching *keyword* from the file with extension *ext*.

        Parameters
        ----------
        ext : str
            File extension to read (e.g. ``"INSPEC"``, ``"RSSPEC"``).
        keyword : str
            Eclipse keyword to search for (e.g. ``"NAME"``, ``"POINTER"``).
        with_fakes : bool
            Passed through to ``EclArray`` to control padding-entry inclusion.

        Returns
        -------
        list[np.ndarray]
            One array per keyword occurrence found in the file.
        """
        with open(f"{self.filename}.{ext}", "rb") as f:
            buff = mmap(f.fileno(), 0, access=1)
        keyword_locs = [_.start() for _ in re.finditer(str2byte(keyword), buff)]
        return [
            EclArray(
                f"{self.filename}.{ext}",
                offset=keyword_loc,
                with_fakes=with_fakes,
            ).array
            for keyword_loc in keyword_locs
        ]

    def _read_all_names(self, ext):
        """Read all NAME arrays from the spec file with extension *ext*.

        Parameters
        ----------
        ext : str
            File extension (e.g. ``"INSPEC"`` or ``"RSSPEC"``).

        Returns
        -------
        list[np.ndarray]
            List of name arrays (dtype ``S8``), one per occurrence.
        """
        return self._read_all_arrays(ext, "NAME", False)

    def _read_all_types(self, ext):
        """Read all TYPE arrays from the spec file with extension *ext*.

        Parameters
        ----------
        ext : str
            File extension (e.g. ``"INSPEC"`` or ``"RSSPEC"``).

        Returns
        -------
        list[np.ndarray]
            List of type arrays, one per occurrence.
        """
        return self._read_all_arrays(ext, "TYPE", False)

    def _read_all_pointers(self, ext):
        """Read all POINTER arrays from the spec file with extension *ext*.

        Parameters
        ----------
        ext : str
            File extension (e.g. ``"INSPEC"`` or ``"RSSPEC"``).

        Returns
        -------
        list[np.ndarray]
            List of pointer arrays (int32), one per occurrence.
        """
        return self._read_all_arrays(ext, "POINTER", False)

    def _get_static_pointers(self):
        """Build a DataFrame of static keyword byte-offsets from the INSPEC file.

        Returns
        -------
        pd.Series
            Maximum pointer value per static keyword, indexed by keyword name.
        """
        static_names = self._read_all_names("INSPEC")
        static_pointers = self._read_all_pointers("INSPEC")
        df = None
        for i, (names, pointers) in enumerate(
            zip(static_names, static_pointers, strict=False)
        ):
            df0 = pd.DataFrame(pointers, index=names, columns=[i])
            df = df0 if df is None else df.join(df0, how="outer")
            df = df[~df.index.duplicated(keep="first")]
        df.fillna("-9999", inplace=True)
        return df.astype("int32").T.max()

    def _get_dynamic_pointers(self):
        """Build a DataFrame of dynamic keyword byte-offsets from the RSSPEC file.

        Returns
        -------
        pd.DataFrame
            Pointer values per keyword (rows) and simulation timestep (columns).
        """
        dynamic_names = self._read_all_names("RSSPEC")
        dynamic_pointers = self._read_all_pointers("RSSPEC")
        df = None
        for i, (names, pointers) in enumerate(
            zip(dynamic_names, dynamic_pointers, strict=False)
        ):
            df0 = pd.DataFrame(pointers, index=names, columns=[i])
            df = df0 if df is None else df.join(df0, how="outer")
            df = df[~df.index.duplicated(keep="first")]
        df.fillna("-9999", inplace=True)
        df = df.astype("int32")
        df.columns = self.get_seqnum_dates().index
        return df

    def _get_all_pointers(self):
        """Merge static and dynamic pointer DataFrames into one time-indexed table.

        Returns
        -------
        pd.DataFrame
            Combined pointer table indexed by simulation date, columns are keyword names.
        """
        all_pointers = pd.concat(
            [self._get_static_pointers(), self._get_dynamic_pointers()]
        )
        # all_pointers = all_pointers.fillna(method="ffill", axis=1).astype("int32").T
        all_pointers = all_pointers.ffill(axis=1).astype("int32").T
        all_pointers.columns = [byte2str(column) for column in all_pointers.columns]
        return self.get_seqnum_dates().join(all_pointers)

    def get_dimens(self):
        """Read grid dimensions (NI, NJ, NK) from the RSSPEC file.

        Returns
        -------
        collections.namedtuple
            Named tuple ``DIMENS(ni, nj, nk)`` with integer grid extents.
        """
        with open(self.filename + ".RSSPEC", "rb") as f:
            rsspec = mmap(f.fileno(), 0, access=1)  # Read-only access
        Dimens = namedtuple("DIMENS", "ni, nj, nk")
        ni, nj, nk = unpack_from(">3i", rsspec, offset=60)
        return Dimens(ni, nj, nk)

    def is_dual(self):
        """Check whether the model uses a dual-porosity formulation.

        Returns
        -------
        bool
            ``True`` if LOGIHEAD[14] is non-empty, indicating dual porosity.
        """
        return (
            len(
                EclArray(
                    self.filename + ".INIT", keyword="LOGIHEAD", with_fakes=False
                ).array[14]
            )
            != 0
        )

    def get_actnum(self):
        """Return the active-cell mask derived from the PORV array.

        Returns
        -------
        numpy.ma.MaskedArray
            Array of pore volumes with zero-PORV cells masked out.
        """
        porv_array = EclArray(
            self.filename + ".INIT", keyword="PORV", with_fakes=True
        ).array
        return ma.masked_equal(porv_array, 0)

    def get_seqnum_dates(self, condensed=True):
        """Build a DataFrame mapping sequence numbers to simulation dates.

        Parameters
        ----------
        condensed : bool, optional
            If ``True`` (default) return only the DATETIME column; otherwise return
            all ITIME fields (DAY, MONTH, YEAR, HOUR, etc.).

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by SEQNUM with a ``DATETIME`` column (or full ITIME columns).
        """
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
        """Read a 3-D property array at a given simulation date.

        Parameters
        ----------
        prop : str
            Eclipse property keyword (e.g. ``"PRESSURE"``, ``"SWAT"``).
        date : datetime or None, optional
            Simulation date to read. Defaults to the first available date.

        Returns
        -------
        np.ndarray
            3-D array of shape ``(NI, NJ, NK)`` with inactive cells set to NaN.

        Raises
        ------
        ValueError
            If *prop* is not in the model or *date* is not available.
        """
        warnings.filterwarnings("ignore")
        seqnum_dates = self.get_seqnum_dates()
        ni, nj, nk = self.get_dimens()
        if prop.upper() not in self._get_all_pointers().columns:
            raise ValueError(f"There is no {prop} property")
        if date is None:
            # Take the first date
            date = seqnum_dates.iloc[0, -1]
        if date not in seqnum_dates["DATETIME"]:
            raise ValueError(f"There is no {date} date among available restart dates")
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
        logger = setup_logging()
        logger.info(
            f"No {prop} value at {date}. Assuming zero for plotting \
              "
        )
        return np.zeros((nk, nj, ni)).T

    def read_prop_time(self, prop, i, j, k):
        """Return a time series of *prop* at grid cell (i, j, k).

        Parameters
        ----------
        prop : str
            Eclipse property keyword (e.g. ``"PRESSURE"``).
        i, j, k : int
            1-based grid indices.

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame indexed by datetime with the property time series.
        """
        dates = self._get_all_pointers()["DATETIME"]
        values = [
            self.read_prop_array(prop, date)[i - 1, j - 1, k - 1] for date in dates
        ]
        return pd.DataFrame(values, index=dates, columns=[f"{prop}@({i}, {j}, {k})"])

    def read_vectors(self):
        """Read all summary vectors from the SMSPEC/UNSMRY binary files.

        Returns
        -------
        pd.DataFrame
            DataFrame with a MultiIndex of (Vector, Well/Group, Cell/Region, Units)
            and one row per simulation ministep.
        """
        smspec = self.filename + ".SMSPEC"
        nlist, ni, nj, nk = EclArray(smspec, keyword="DIMENS", with_fakes=False).array[
            :4
        ]
        logger.debug(f"nlist: {nlist}, ni: {ni}, nj: {nj}, nk: {nk}")
        keywords = byte2str(EclArray(smspec, keyword="KEYWORDS", with_fakes=True).array)
        logger.debug(f"keywords: {keywords}")
        wgnames = byte2str(EclArray(smspec, keyword="WGNAMES", with_fakes=True).array)
        logger.debug(f"wgnames: {wgnames}")
        nums = EclArray(smspec, keyword="NUMS", with_fakes=True).array
        logger.debug(f"nums: {nums}")
        units = byte2str(EclArray(smspec, keyword="UNITS", with_fakes=True).array)
        logger.debug(f"units: {units}")
        logger.debug("LENGTHS")
        logger.debug("-------")
        logger.debug(f"keywords: {len(keywords)}")
        logger.debug(f"wgnames: {len(wgnames)}")
        logger.debug(f"nums: {len(nums)}")
        logger.debug(f"units: {len(units)}")
        logger.debug("ZIPS")
        logger.debug("-------")
        for i in zip(keywords, wgnames, nums, units, strict=False):
            logger.warning(i)
        new_nums = []
        for keyword, num in zip(keywords, nums, strict=False):
            if (keyword.startswith(("C", "B"))) and num > 0:
                k = int((num - 1) / (ni * nj) - 0.00001) + 1
                j = int((num - (k - 1) * ni * nj) / ni - 0.00001) + 1
                i = num - (j - 1) * ni - (k - 1) * ni * nj
                num = str(f"({i}, {j}, {k})")
            else:
                # Convert NUMs to strings for subsequent plotting
                num = str(num)
            new_nums.append(num)
        nums = new_nums
        logger.debug("NUMS CONVERTED")
        logger.debug("-------")
        for i in nums:
            logger.debug(i)
        params = self._read_all_arrays("UNSMRY", "PARAMS", True)
        logger.warning(params)
        headers = pd.MultiIndex.from_tuples(
            list(zip(*[keywords, wgnames, nums, units], strict=False)),
            names=["Vector", "Well/Group", "Cell/Region", "Units"],
        )
        df = pd.DataFrame(params, columns=headers).sort_index(axis=1)
        df.index.name = "MINISTEP"
        self.vectors_df = df
        return df

    def get_vectors_shape(self):
        """Return the shape of the cached vectors DataFrame.

        Returns
        -------
        tuple[int, int] or None
            ``(n_timesteps, n_vectors)`` if vectors have been read, else ``None``.
        """
        if self.vectors_df is not None:
            return self.vectors_df.shape
        return None

    def get_vector_names(self):
        """Return sorted unique top-level vector keyword names from the vectors DataFrame.

        Returns
        -------
        list[str] or None
            Sorted vector names (e.g. ``["FOPT", "WBHP", ...]``), or ``None`` if not yet read.
        """
        if self.vectors_df is not None:
            return sorted(set(self.vectors_df.columns.get_level_values(0)))
        return None

    def get_vector_column(self, vector_name):
        """Extract all rows for *vector_name* as a flat single-column DataFrame.

        Parameters
        ----------
        vector_name : str
            Top-level Eclipse summary keyword (e.g. ``"WBHP"``).

        Returns
        -------
        pd.DataFrame or None
            Single-column DataFrame named *vector_name*, or ``None`` if not yet read.
        """
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
        return None


def is_valid_vector(vector_name):
    """Check whether a string is a recognised Eclipse summary vector keyword.

    Parameters
    ----------
    vector_name : str
        Keyword to validate against the known vector list.

    Returns
    -------
    bool
        ``True`` if ``vector_name`` appears in the ``ecl_vectors`` list.
    """
    valid_vectors = ecl_vectors
    return vector_name in valid_vectors


def loss_compute_abs(a, b):
    """Compute the batch-normalised sum of absolute differences between two tensors.

    Parameters
    ----------
    a : torch.Tensor
        Predicted tensor; first dimension is the batch size.
    b : torch.Tensor
        Target tensor of the same shape as ``a``.

    Returns
    -------
    torch.Tensor
        Scalar sum of absolute differences divided by the batch size.
    """
    return torch.sum(torch.abs(a - b) / a.shape[0])


class LpLoss:
    """Lp Loss."""

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        """Initialise an Lp-norm loss for finite-element-style grids.

        Parameters
        ----------
        d : int, optional
            Spatial dimension of the domain (default 2).
        p : int, optional
            Order of the Lp norm (default 2 → L2).
        size_average : bool, optional
            If ``True`` return the mean over examples; otherwise return the sum.
        reduction : bool, optional
            If ``False`` return per-example norms without aggregation.
        """
        super().__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        """Compute the absolute Lp loss weighted by grid spacing ``h = 1/(N-1)``.

        Parameters
        ----------
        x : torch.Tensor
            Predicted values, shape ``(batch, N, ...)``.
        y : torch.Tensor
            Ground-truth values, same shape as *x*.

        Returns
        -------
        torch.Tensor
            Scalar (or per-example) absolute Lp loss.
        """
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        """Compute the relative Lp loss (normalised by the norm of *y*).

        Parameters
        ----------
        x : torch.Tensor
            Predicted values, shape ``(batch, ...)``.
        y : torch.Tensor
            Ground-truth values, same shape as *x*.

        Returns
        -------
        torch.Tensor
            Scalar (or per-example) relative Lp loss.
        """
        num_examples = x.size()[0]
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y):
        """Evaluate the relative Lp loss (calls :meth:`rel`).

        Parameters
        ----------
        x : torch.Tensor
            Predicted values.
        y : torch.Tensor
            Ground-truth values.

        Returns
        -------
        torch.Tensor
            Relative Lp loss scalar.
        """
        return self.rel(x, y)


def Getit(data, input_keys, output_keys, output_keys2):
    """Extract three subsets of arrays from a data dictionary by key lists.

    Parameters
    ----------
    data : dict
        Source dictionary mapping string keys to array-like values.
    input_keys : list of str
        Keys to extract into the input variable dictionary.
    output_keys : list of str
        Keys to extract into the first output variable dictionary.
    output_keys2 : list of str
        Keys to extract into the second output variable dictionary.

    Returns
    -------
    invar : dict
        Input variables extracted from ``data``.
    outvar : dict
        First set of output variables extracted from ``data``.
    outvar2 : dict
        Second set of output variables extracted from ``data``.
    """
    logger = setup_logging()
    _ks = [k for k in data if not k.startswith("__")]
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
    """Extract two subsets of arrays from a data dictionary by key lists.

    Parameters
    ----------
    data : dict
        Source dictionary mapping string keys to array-like values; deleted after extraction.
    input_keys : list of str
        Keys to extract into the input variable dictionary.
    output_keys : list of str
        Keys to extract into the output variable dictionary.

    Returns
    -------
    invar : dict
        Input variables extracted from ``data``.
    outvar : dict
        Output variables extracted from ``data``.
    """
    logger = setup_logging()
    _ks = [k for k in data if not k.startswith("__")]
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
    """Compute gas viscosity as a quadratic function of pressure.

    Parameters
    ----------
    p : torch.Tensor
        Reservoir pressure tensor in consistent pressure units.

    Returns
    -------
    torch.Tensor
        Gas dynamic viscosity tensor with the same shape as ``p``.
    """
    # Average reservoir pressure
    return 3e-6 * p**2 + 1e-6 * p + 0.0133


def calc_rs(p_bub, p, device):
    """Compute solution gas-oil ratio as a function of pressure.

    Parameters
    ----------
    p_bub : torch.Tensor
        Bubble-point pressure scalar tensor.
    p : torch.Tensor
        Reservoir pressure tensor.
    device : str
        PyTorch device string (e.g., ``"cuda:0"`` or ``"cpu"``).

    Returns
    -------
    torch.Tensor
        Solution GOR tensor with the same shape as ``p``.
    """
    device1 = device
    rs_factor = torch.where(
        p < p_bub,
        torch.tensor(1.0).to(device1, torch.float32),
        torch.tensor(1e-6).to(device1, torch.float32),
    )
    return (
        (178.11**2) / 5.615 * (torch.pow(p / p_bub, 1.3) * rs_factor + (1 - rs_factor))
    )


def calc_dp(p_bub, p_atm, p):
    """Compute pressure differential used in gas formation volume factor calculations.

    Parameters
    ----------
    p_bub : torch.Tensor
        Bubble-point pressure scalar tensor.
    p_atm : torch.Tensor
        Atmospheric (reference) pressure scalar tensor.
    p : torch.Tensor
        Reservoir pressure tensor.

    Returns
    -------
    torch.Tensor
        Effective pressure differential tensor with the same shape as ``p``.
    """
    return torch.where(p < p_bub, p_atm - p, p_atm - p_bub)


def calc_bg(p_bub, p_atm, p):
    """Compute gas formation volume factor as a function of pressure.

    Parameters
    ----------
    p_bub : torch.Tensor
        Bubble-point pressure scalar tensor.
    p_atm : torch.Tensor
        Atmospheric (reference) pressure scalar tensor.
    p : torch.Tensor
        Average reservoir pressure tensor.

    Returns
    -------
    torch.Tensor
        Gas FVF tensor with the same shape as ``p``.
    """
    # P is average reservoir pressure
    return torch.divide(1, torch.exp(1.7e-3 * calc_dp(p_bub, p_atm, p)))


def calc_bo(p_bub, p_atm, CFO, p):
    """Compute oil formation volume factor as a function of pressure.

    Parameters
    ----------
    p_bub : torch.Tensor
        Bubble-point pressure scalar tensor.
    p_atm : torch.Tensor
        Atmospheric (reference) pressure scalar tensor.
    CFO : float
        Oil compressibility factor above bubble point.
    p : torch.Tensor
        Average reservoir pressure tensor.

    Returns
    -------
    torch.Tensor
        Oil FVF tensor with the same shape as ``p``.
    """
    # p is average reservoir pressure
    exp_term1 = torch.where(p < p_bub, -8e-5 * (p_atm - p), -8e-5 * (p_atm - p_bub))
    exp_term2 = -CFO * torch.where(p < p_bub, torch.zeros_like(p), p - p_bub)
    return torch.divide(1, torch.exp(exp_term1) * torch.exp(exp_term2))


def fit_scale_abs(tensor, target_min, target_max, tensor_min, tensor_max):
    """Normalise a tensor by its known maximum value.

    Parameters
    ----------
    tensor : np.ndarray or torch.Tensor
        Data to normalise.
    target_min : float
        Intended normalised minimum (currently unused).
    target_max : float
        Intended normalised maximum (currently unused).
    tensor_min : float
        Known minimum of ``tensor`` (currently unused).
    tensor_max : float
        Known maximum of ``tensor``; used as the divisor.

    Returns
    -------
    np.ndarray or torch.Tensor
        Tensor divided by ``tensor_max``.
    """
    # Rescale between target min and target max
    return tensor / tensor_max


def StoneIIModel(params, device, Sg, Sw):
    """Compute three-phase relative permeabilities using the Stone-II model.

    Parameters
    ----------
    params : dict
        Model parameters dict with keys ``k_rwmax``, ``k_romax``, ``k_rgmax``,
        ``n``, ``p``, ``q``, ``m``, ``Swi``, ``Sor`` as torch.Tensor scalars.
    device : str
        PyTorch device string (e.g., ``"cuda:0"`` or ``"cpu"``).
    Sg : torch.Tensor
        Gas saturation field tensor.
    Sw : torch.Tensor
        Water saturation field tensor.

    Returns
    -------
    krw : torch.Tensor
        Water relative permeability with the same shape as ``Sw``.
    kro : torch.Tensor
        Oil relative permeability with the same shape as ``Sw``.
    krg : torch.Tensor
        Gas relative permeability with the same shape as ``Sg``.
    """
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
    """Compute oil production rates at fixed well locations using the Peacemann model.

    Parameters
    ----------
    UO : float
        Oil viscosity in consistent units.
    BO : float
        Oil formation volume factor.
    UW : float
        Water viscosity (currently unused inside this function).
    BW : float
        Water FVF (currently unused inside this function).
    DZ : float
        Perforation thickness.
    RE : float
        Drainage radius.
    device : str
        PyTorch device string.
    max_inn_fcn : float
        Input normalisation scale factor (currently unused).
    max_out_fcn : float
        Output normalisation scale factor (currently unused).
    paramz : dict
        Stone-II model parameter dictionary.
    p_bub : torch.Tensor
        Bubble-point pressure scalar tensor.
    p_atm : torch.Tensor
        Atmospheric pressure scalar tensor.
    steppi : int
        Number of time steps.
    CFO : float
        Oil compressibility above bubble point.
    sgas : torch.Tensor
        Gas saturation tensor of shape ``(N, T, nz, nx, ny)``.
    swater : torch.Tensor
        Water saturation tensor of the same shape as ``sgas``.
    pressure : torch.Tensor
        Pressure tensor of the same shape as ``sgas``.
    permeability : torch.Tensor
        Permeability tensor of shape ``(N, 1, nz, nx, ny)``.

    Returns
    -------
    torch.Tensor
        Oil rate tensor of the same shape as ``sgas`` (negative at producer cells).
    """
    qoil = torch.zeros_like(sgas).to(device)
    skin = 0
    rwell = 200
    pwf_producer = 100

    def process_location(i, j, k, l_index):
        """Compute the Peacemann oil-rate contribution for one well location.

        Parameters
        ----------
        i, j : int
            Batch and spatial grid indices for the well location.
        k : int
            Well index within the producer list.
        l_index : int
            Lateral index along the well trajectory.

        Returns
        -------
        torch.Tensor
            Negative oil flow rate (negative convention = production) for this location.
        """
        pre1 = pressure[i, j, :, :, :]
        sg1 = sgas[i, j, :, k, l_index]
        sw1 = swater[i, j, :, k, l_index]
        _krw, kro, _krg = StoneIIModel(paramz, device, sg1, sw1)
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


# linear_interp is imported from utils.array_utils above.


def RelPerm(Sa, Sg, SWI, SWR, SWOW, SWOG):
    """Compute three-phase relative permeabilities using table interpolation.

    Parameters
    ----------
    Sa : torch.Tensor
        Water saturation tensor.
    Sg : torch.Tensor
        Gas saturation tensor.
    SWI : float
        Irreducible water saturation.
    SWR : float
        Residual water saturation.
    SWOW : torch.Tensor
        Water-oil relative permeability table of shape ``(n, 3)``: columns are
        ``[Sw, KROW, KRW]``.
    SWOG : torch.Tensor
        Gas-oil relative permeability table of shape ``(n, 3)``: columns are
        ``[Sg, KROG, KRG]``.

    Returns
    -------
    KRW : torch.Tensor
        Water relative permeability tensor.
    KRO : torch.Tensor
        Oil relative permeability tensor.
    KRG : torch.Tensor
        Gas relative permeability tensor.
    """
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
    """Load and compile geostatistical parameters for the Norne field model.

    Parameters
    ----------
    nx : int
        Number of grid cells in the x direction.
    ny : int
        Number of grid cells in the y direction.
    nz : int
        Number of grid cells in the z direction.

    Returns
    -------
    dict
        Dictionary containing porosity, permeability, NTG, fault-multiplier, and
        relative-permeability geostatistical parameters for ensemble generation.
    """
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
    for i, (mean_, std_) in enumerate(zip(z_means, z_stds, strict=False), start=1):
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
