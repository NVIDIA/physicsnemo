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

"""Canonical Eclipse binary file reader (EclArray, EclBinaryParser) shared across all sub-modules."""

# Standard Library
import os
import warnings
import re
from struct import unpack_from
from collections import namedtuple
from mmap import mmap

# Third-party Libraries
import numpy as np
import pandas as pd
import numpy.ma as ma

# Local Modules
from utils.logging_utils import setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Module-level data tables (required by EclArray / EclBinaryParser)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Standalone helper functions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


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
