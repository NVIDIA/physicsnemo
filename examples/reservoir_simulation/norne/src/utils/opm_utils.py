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
               OPM RESERVOIR DATA UTILITY FUNCTIONS
=====================================================================

Consolidated utility functions for OPM/Eclipse reservoir data.
Functions here were previously duplicated across multiple modules;
all callers now import from this single canonical location.

Canonical sources:
- compare/sequential/misc_gather_utils.py  (parse_egrid, parse_unrst,
  _check_and_fetch_*, _parse_*, _fetch_keyword_data, Get_fault,
  read_faults, assign_faults, Get_falt, copy_files)
- compare/sequential/misc_plotting_utils.py (Get_Time, historydata,
  simulation_data_types)
- data_extract/opm_extract_rates.py  (process_data, process_data2,
  get_dyna, get_dyna2, convert_to_list, extract_tuples, read_compdats,
  read_compdats2, process_dataframe, extract_qs, Remove_folder,
  process_and_print, process_task)

@Author : Clement Etienam
"""

# Standard Library
import gc
import gzip
import math
import os
import pickle
import re
import shutil
import fnmatch
from collections import OrderedDict
from shutil import rmtree
from struct import unpack

# Third-party Libraries
import numpy as np
import pandas as pd
from gstools import SRF, Gaussian
from hydra.utils import to_absolute_path
from sklearn.preprocessing import MinMaxScaler

# Local Modules
from utils.logging_utils import setup_logging

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# RSM file line-skip sentinels for NORNE_ATW2013.RSM
# Each section within the RSM file begins at a fixed line offset.  The groups
# below correspond to well oil, water, and gas production sections followed by
# the water- and gas-injection sections.  Consecutive sections within a group
# are separated by _RSM_SECTION_STRIDE lines.
# ---------------------------------------------------------------------------
_RSM_SECTION_STRIDE = 870  # lines between consecutive RSM sections

_RSM_OIL_SEC1 = 47873  # first  oil-rate RSM section
_RSM_OIL_SEC2 = 48743  # second oil-rate RSM section
_RSM_OIL_SEC3 = 49613  # third  oil-rate RSM section
_RSM_OIL_SEC4 = 50483  # fourth oil-rate RSM section

_RSM_WATER_SEC1 = 40913  # first  water-rate RSM section
_RSM_WATER_SEC2 = 41783  # second water-rate RSM section
_RSM_WATER_SEC3 = 42653  # third  water-rate RSM section
_RSM_WATER_SEC4 = 43523  # fourth water-rate RSM section

_RSM_GAS_SEC1 = 54833  # first  gas-rate RSM section
_RSM_GAS_SEC2 = 55703  # second gas-rate RSM section
_RSM_GAS_SEC3 = 56573  # third  gas-rate RSM section
_RSM_GAS_SEC4 = 57443  # fourth gas-rate RSM section

_RSM_WINJ_SEC1 = 72237  # water-injection RSM section
_RSM_GINJ_SEC1 = 73977  # gas-injection   RSM section

# ---------------------------------------------------------------------------
# Module-level ECL type/constant tables
# ---------------------------------------------------------------------------

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
# simulation_data_types
# (canonical: compare/sequential/misc_plotting_utils.py)
# ---------------------------------------------------------------------------


def simulation_data_types():
    """
    Return standard ECL simulation data-type lookup tables and keyword lists.

    Returns
    -------
    type_dict : dict
        Mapping of bytes keyword (e.g. b'INTE') to format character string.
    ecl_extensions : list of str
        Recognised Eclipse file extensions (e.g. '.UNRST').
    dynamic_props : list of str
        Names of dynamic reservoir properties (e.g. 'PRESSURE', 'SWAT').
    ecl_vectors : list of str
        Names of recognised ECL summary vectors (e.g. 'WOPR', 'WBHP').
    static_props : list of str
        Names of static reservoir properties (e.g. 'PERMX', 'PORO').
    SUPPORTED_DATA_TYPES : dict
        Mapping of type string to (element_size, fmt_char, element_skip).
    """
    _SUPPORTED_DATA_TYPES = {
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
        _SUPPORTED_DATA_TYPES,
    )


# ---------------------------------------------------------------------------
# ECL binary parsing helpers
# (canonical: compare/sequential/misc_gather_utils.py)
# ---------------------------------------------------------------------------


def _check_and_fetch_type_info(data_type):
    """
    Look up element size, format character, and skip interval for an ECL data type.

    Parameters
    ----------
    data_type : str
        ECL keyword data type string (e.g. 'INTE', 'REAL', 'CHAR').

    Returns
    -------
    type_info : tuple
        (element_size_bytes, fmt_char, element_skip) from SUPPORTED_DATA_TYPES.

    Raises
    ------
    ValueError
        If data_type is not present in SUPPORTED_DATA_TYPES.
    """
    try:
        return SUPPORTED_DATA_TYPES[data_type]
    except KeyError as exc:
        raise ValueError(f"Unknown datatype {data_type}.") from exc


def _check_and_fetch_file(path, pattern, return_relative=False):
    """
    Search a directory for files matching a glob pattern.

    Parameters
    ----------
    path : str
        Directory path to search.
    pattern : str
        Glob-style pattern string (e.g. '*.UNRST').
    return_relative : bool, optional
        If True return paths relative to path; otherwise return absolute. Default is False.

    Returns
    -------
    found : list of str
        Matching file paths (absolute or relative depending on return_relative).
    """
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
    """
    Sequentially read all keyword sections from an ECL binary file.

    Parameters
    ----------
    path : str
        Path to the binary ECL file (e.g. .UNRST or .EGRID).
    attrs : list of str or None, optional
        If provided, only sections whose name is in attrs are decoded.
        If None, all sections are decoded. Default is None.

    Returns
    -------
    header : bytes
        The first 4-byte file header.
    sections : dict
        Mapping of keyword name to list of decoded np.ndarray values.
    """
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
    """
    Parse selected keyword sections from an ECL binary file.

    Parameters
    ----------
    path : str
        Path to the binary ECL file.
    attrs : list of str or str or None, optional
        Keyword names to extract. Must not be None or empty.

    Returns
    -------
    sections : dict
        Mapping of keyword name to list of decoded np.ndarray values.

    Raises
    ------
    ValueError
        If attrs is None or empty.
    """
    if attrs is None:
        raise ValueError("Keyword attribute cannot be empty")
    if isinstance(attrs, str):
        attrs = [attrs]
    attrs = [attr.strip().upper() for attr in attrs]
    _, sections = _parse_keywords(path, attrs)
    return sections


def _fetch_keyword_data(section):
    """
    Decode a packed binary ECL keyword section into a numpy array.

    Parameters
    ----------
    section : tuple
        Tuple of (n_elements, data_type, element_size, fmt, element_skip, binary_data)
        as produced by _parse_keywords.

    Returns
    -------
    decoded_section : np.ndarray
        Decoded and skip-stripped array of the keyword values.
    """
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


def parse_egrid(path_to_result):
    """
    Read GRIDHEAD and ACTNUM attributes from an EGRID binary file.

    Parameters
    ----------
    path_to_result : str
        Path to the .EGRID file.

    Returns
    -------
    egrid : dict
        Dictionary with keys 'GRIDHEAD' and 'ACTNUM' containing parsed numpy arrays.
    """
    egrid_path = path_to_result
    attrs = ("GRIDHEAD", "ACTNUM")
    return _parse_ech_bin(egrid_path, attrs)


def parse_unrst(path_to_result):
    """
    Read PRESSURE, SGAS, and SWAT state arrays from a UNRST binary restart file.

    Parameters
    ----------
    path_to_result : str
        Path to the .UNRST restart file.

    Returns
    -------
    states : dict
        Dictionary with keys 'PRESSURE', 'SGAS', 'SWAT' containing lists of numpy arrays.
    """
    unrst_path = path_to_result
    attrs = ("PRESSURE", "SGAS", "SWAT")
    return _parse_ech_bin(unrst_path, attrs)


# ---------------------------------------------------------------------------
# Fault utilities
# (canonical: compare/sequential/misc_gather_utils.py)
# ---------------------------------------------------------------------------


def Get_fault(filename):
    """
    Parse a DATA file and return all fault names listed under MULTFLT.

    Parameters
    ----------
    filename : str
        Path to the Eclipse-style DATA file containing a MULTFLT keyword.

    Returns
    -------
    fault_names : list of str
        Sorted list of unique fault names found in the MULTFLT section.
    """
    with open(filename) as file:
        injector_gas = set()  # Set to collect gas injector well names
        start_collecting_welspecs = False
        for line in file:
            stripped_line = line.strip()
            # Skip lines that are comments
            if stripped_line.startswith("--"):
                continue
            # Start collecting data after finding 'WELSPECS'
            if "MULTFLT" in stripped_line:
                start_collecting_welspecs = True
                continue
            # Stop collecting data when encountering a line that starts with '/'
            if start_collecting_welspecs and stripped_line.startswith("/"):
                start_collecting_welspecs = False
                continue
            # If collecting from WELSPECS, process the data
            if start_collecting_welspecs:
                parts = stripped_line.split()
                fault_name = parts[0].strip("'")
                injector_gas.add(fault_name)
    return sorted(injector_gas)


def read_faults(filename, well_names):
    """
    Parse a DATA file and extract FAULTS keyword entries for specified fault names.

    Parameters
    ----------
    filename : str
        Path to the Eclipse-style DATA file containing a FAULTS keyword.
    well_names : list of str
        Fault names to filter (reuses the well_names parameter convention).

    Returns
    -------
    data : list of tuple
        Each tuple is (fault_name, i1, i2, j1, j2, k1, k2) for matching FAULTS rows.
    """
    with open(filename) as file:
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


def assign_faults(well_indices, nx, ny, nz, well_amount, data):
    """
    Fill a 3-D fault multiplier array using parsed FAULTS index ranges and values.

    Parameters
    ----------
    well_indices : list of tuple
        Fault entries with (fault_name, i1, i2, j1, j2, k1, k2) structure.
    nx : int
        Grid dimension in the X direction.
    ny : int
        Grid dimension in the Y direction.
    nz : int
        Grid dimension in the Z direction.
    well_amount : list or set
        Ordered collection of fault names used to align with data values.
    data : np.ndarray or list
        Multiplier values indexed to match the ordered fault names.

    Returns
    -------
    faultm : np.ndarray
        3-D float16 array of shape (nx, ny, nz) filled with fault multipliers.
    """
    faultm = np.ones((nx, ny, nz), dtype=np.float16)
    unique_well_names = OrderedDict()
    for _idx, tuple_entry in enumerate(well_indices):
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


def Get_falt(source_dir, nx, ny, nz, floatz, N, filename_fault, FAULT_INCLUDE):
    """
    Build a stacked fault multiplier array for an ensemble of N realisations.

    Parameters
    ----------
    source_dir : str
        Directory containing the fault include file.
    nx : int
        Grid dimension in the X direction.
    ny : int
        Grid dimension in the Y direction.
    nz : int
        Grid dimension in the Z direction.
    floatz : np.ndarray
        2-D array of shape (n_faults, N) with multiplier values per realisation.
    N : int
        Number of ensemble members.
    filename_fault : str
        Path to the DATA file containing the FAULTS keyword.
    FAULT_INCLUDE : str
        Filename of the fault multiplier include file (relative to source_dir).

    Returns
    -------
    flt_stack : np.ndarray
        Array of shape (N, 1, nx, ny, nz) with stacked fault multiplier grids.
    """
    Fault = np.ones((nx, ny, nz), dtype=np.float16)
    flt = []
    for k in range(N):
        floatts = floatz[:, k]
        fault_temp = Get_fault(os.path.join(source_dir, FAULT_INCLUDE))
        fault_data = read_faults(filename_fault, fault_temp)
        Fault = assign_faults(fault_data, nx, ny, nz, fault_temp, floatts)
        flt.append(Fault)
    flt = np.stack(flt, axis=0)[:, None, :, :, :]
    return np.stack(flt, axis=0)


def copy_files(source_dir, dest_dir):
    """
    Copy all files from a source directory into a destination directory.

    Parameters
    ----------
    source_dir : str
        Path to the directory whose files will be copied.
    dest_dir : str
        Path to the target directory.

    Returns
    -------
    None
    """
    files = os.listdir(source_dir)
    for file in files:
        shutil.copy(os.path.join(source_dir, file), dest_dir)


# ---------------------------------------------------------------------------
# Time loading
# (canonical: compare/sequential/misc_plotting_utils.py)
# ---------------------------------------------------------------------------


def Get_Time(nx, ny, nz, steppi, steppi_indices, N):
    """
    Load training time data from disk and tile it into an ensemble time array.

    Parameters
    ----------
    nx : int
        Grid dimension in the X direction.
    ny : int
        Grid dimension in the Y direction.
    nz : int
        Grid dimension in the Z direction.
    steppi : int
        Number of time steps in the training data.
    steppi_indices : np.ndarray
        1-based indices selecting a subset of time steps (currently unused in tiling).
    N : int
        Number of ensemble members to tile the time array over.

    Returns
    -------
    Timee : np.ndarray
        Array of shape (N, steppi, nx, ny, nz) with broadcast time values.
    """
    logger.info("Load simulated labelled training data")
    with gzip.open(to_absolute_path("../data/data_train.pkl.gz"), "rb") as f2:
        mat = pickle.load(f2)
    X_data1 = mat
    del mat
    gc.collect()
    Time = X_data1["Time"]
    np_array2 = np.zeros(Time.shape[1])
    for mm in range(Time.shape[1]):
        np_array2[mm] = Time[0, mm, 0, 0, 0]
    Timee = []
    for _k in range(N):
        check = np.ones((nx, ny, nz), dtype=np.float16)
        unie = []
        for zz in range(len(np_array2)):
            aa = np_array2[zz] * check
            unie.append(aa)
        Time = np.stack(unie, axis=0)
        Timee.append(Time)
    return np.stack(Timee, axis=0)


def historydata(timestep, steppi, steppi_indices, source_dir):
    """
    Load historical well production and injection rates from the Norne RSM file.

    Parameters
    ----------
    timestep : np.ndarray
        Row indices (1-based) selecting the time steps from the RSM file.
    steppi : int
        Number of output time steps expected in the returned arrays.
    steppi_indices : np.ndarray
        1-based indices used to sub-select rows from the full time series.
    source_dir : str
        Directory containing the NORNE_ATW2013.RSM file.

    Returns
    -------
    DATA : dict
        Dictionary with keys 'OIL', 'WATER', 'GAS', 'WATER_INJ', 'WGAS_inj'
        each holding an np.ndarray of shape (steppi, n_wells).
    DATA2 : np.ndarray
        Vertically stacked 1-D array concatenating all production/injection columns.
    """
    WOIL1 = np.zeros((steppi, 22))
    WWATER1 = np.zeros((steppi, 22))
    WGAS1 = np.zeros((steppi, 22))
    WWINJ1 = np.zeros((steppi, 9))
    WGASJ1 = np.zeros((steppi, 4))
    indices = timestep
    logger.info("Get the Well Oil Production Rate")
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM")) as f:
        for i, line in enumerate(f):
            if i < _RSM_OIL_SEC1:
                continue
            if "---" in line:
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1 = df[[2, 3, 4, 5, 6, 8]].values
    B_2H = A1[:, 0][indices - 1]
    D_1H = A1[:, 1][indices - 1]
    D_2H = A1[:, 2][indices - 1]
    B_4H = A1[:, 3][indices - 1]
    D_4H = A1[:, 4][indices - 1]
    E_3H = A1[:, 5][indices - 1]
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM")) as f:
        for i, line in enumerate(f):
            if i < _RSM_OIL_SEC2:
                continue
            if "---" in line:
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A2 = df[[1, 4, 5, 7, 9]].values
    B_1H = A2[:, 0][indices - 1]
    B_3H = A2[:, 1][indices - 1]
    E_1H = A2[:, 2][indices - 1]
    E_2H = A2[:, 3][indices - 1]
    E_4AH = A2[:, 4][indices - 1]
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM")) as f:
        for i, line in enumerate(f):
            if i < _RSM_OIL_SEC3:
                continue
            if "---" in line:
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A3 = df[[2, 4, 7, 8, 9]].values
    D_3AH = A3[:, 0][indices - 1]
    E_3AH = A3[:, 1][indices - 1]
    B_4BH = A3[:, 2][indices - 1]
    D_4AH = A3[:, 3][indices - 1]
    D_1CH = A3[:, 4][indices - 1]
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM")) as f:
        for i, line in enumerate(f):
            if i < _RSM_OIL_SEC4:
                continue
            if "---" in line:
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A4 = df[[2, 4, 5, 6, 8, 9]].values
    B_4DH = A4[:, 0][indices - 1]
    E_3CH = A4[:, 1][indices - 1]
    E_2AH = A4[:, 2][indices - 1]
    D_3BH = A4[:, 3][indices - 1]
    B_1BH = A4[:, 4][indices - 1]
    K_3H = A4[:, 5][indices - 1]
    WOIL1[:, 0] = B_1BH.ravel()[steppi_indices - 1]
    WOIL1[:, 1] = B_1H.ravel()[steppi_indices - 1]
    WOIL1[:, 2] = B_2H.ravel()[steppi_indices - 1]
    WOIL1[:, 3] = B_3H.ravel()[steppi_indices - 1]
    WOIL1[:, 4] = B_4BH.ravel()[steppi_indices - 1]
    WOIL1[:, 5] = B_4DH.ravel()[steppi_indices - 1]
    WOIL1[:, 6] = B_4H.ravel()[steppi_indices - 1]
    WOIL1[:, 7] = D_1CH.ravel()[steppi_indices - 1]
    WOIL1[:, 8] = D_1H.ravel()[steppi_indices - 1]
    WOIL1[:, 9] = D_2H.ravel()[steppi_indices - 1]
    WOIL1[:, 10] = D_3AH.ravel()[steppi_indices - 1]
    WOIL1[:, 11] = D_3BH.ravel()[steppi_indices - 1]
    WOIL1[:, 12] = D_4AH.ravel()[steppi_indices - 1]
    WOIL1[:, 13] = D_4H.ravel()[steppi_indices - 1]
    WOIL1[:, 14] = E_1H.ravel()[steppi_indices - 1]
    WOIL1[:, 15] = E_2AH.ravel()[steppi_indices - 1]
    WOIL1[:, 16] = E_2H.ravel()[steppi_indices - 1]
    WOIL1[:, 17] = E_3AH.ravel()[steppi_indices - 1]
    WOIL1[:, 18] = E_3CH.ravel()[steppi_indices - 1]
    WOIL1[:, 19] = E_3H.ravel()[steppi_indices - 1]
    WOIL1[:, 20] = E_4AH.ravel()[steppi_indices - 1]
    WOIL1[:, 21] = K_3H.ravel()[steppi_indices - 1]
    logger.info("Get the Well water Production Rate")
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM")) as f:
        for i, line in enumerate(f):
            if i < _RSM_WATER_SEC1:
                continue
            if "---" in line:
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1w = df[[2, 3, 4, 5, 6, 8]].values
    B_2Hw = A1w[:, 0][indices - 1]
    D_1Hw = A1w[:, 1][indices - 1]
    D_2Hw = A1w[:, 2][indices - 1]
    B_4Hw = A1w[:, 3][indices - 1]
    D_4Hw = A1w[:, 4][indices - 1]
    E_3Hw = A1w[:, 5][indices - 1]
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM")) as f:
        for i, line in enumerate(f):
            if i < _RSM_WATER_SEC2:
                continue
            if "---" in line:
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A2w = df[[1, 4, 5, 7, 9]].values
    B_1Hw = A2w[:, 0][indices - 1]
    B_3Hw = A2w[:, 1][indices - 1]
    E_1Hw = A2w[:, 2][indices - 1]
    E_2Hw = A2w[:, 3][indices - 1]
    E_4AHw = A2w[:, 4][indices - 1]
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM")) as f:
        for i, line in enumerate(f):
            if i < _RSM_WATER_SEC3:
                continue
            if "---" in line:
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A3w = df[[2, 4, 7, 8, 9]].values
    D_3AHw = A3w[:, 0][indices - 1]
    E_3AHw = A3w[:, 1][indices - 1]
    B_4BHw = A3w[:, 2][indices - 1]
    D_4AHw = A3w[:, 3][indices - 1]
    D_1CHw = A3w[:, 4][indices - 1]
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM")) as f:
        for i, line in enumerate(f):
            if i < _RSM_WATER_SEC4:
                continue
            if "---" in line:
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A4w = df[[2, 4, 5, 6, 8, 9]].values
    B_4DHw = A4w[:, 0][indices - 1]
    E_3CHw = A4w[:, 1][indices - 1]
    E_2AHw = A4w[:, 2][indices - 1]
    D_3BHw = A4w[:, 3][indices - 1]
    B_1BHw = A4w[:, 4][indices - 1]
    K_3Hw = A4w[:, 5][indices - 1]
    WWATER1[:, 0] = B_1BHw.ravel()[steppi_indices - 1]
    WWATER1[:, 1] = B_1Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 2] = B_2Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 3] = B_3Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 4] = B_4BHw.ravel()[steppi_indices - 1]
    WWATER1[:, 5] = B_4DHw.ravel()[steppi_indices - 1]
    WWATER1[:, 6] = B_4Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 7] = D_1CHw.ravel()[steppi_indices - 1]
    WWATER1[:, 8] = D_1Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 9] = D_2Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 10] = D_3AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 11] = D_3BHw.ravel()[steppi_indices - 1]
    WWATER1[:, 12] = D_4AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 13] = D_4Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 14] = E_1Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 15] = E_2AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 16] = E_2Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 17] = E_3AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 18] = E_3CHw.ravel()[steppi_indices - 1]
    WWATER1[:, 19] = E_3Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 20] = E_4AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 21] = K_3Hw.ravel()[steppi_indices - 1]
    logger.info("Get the Well Gas Production Rate")
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM")) as f:
        for i, line in enumerate(f):
            if i < _RSM_GAS_SEC1:
                continue
            if "---" in line:
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1g = df[[2, 3, 4, 5, 6, 8]].values
    B_2Hg = A1g[:, 0][indices - 1]
    D_1Hg = A1g[:, 1][indices - 1]
    D_2Hg = A1g[:, 2][indices - 1]
    B_4Hg = A1g[:, 3][indices - 1]
    D_4Hg = A1g[:, 4][indices - 1]
    E_3Hg = A1g[:, 5][indices - 1]
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM")) as f:
        for i, line in enumerate(f):
            if i < _RSM_GAS_SEC2:
                continue
            if "---" in line:
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A2g = df[[1, 4, 5, 7, 9]].values
    B_1Hg = A2g[:, 0][indices - 1]
    B_3Hg = A2g[:, 1][indices - 1]
    E_1Hg = A2g[:, 2][indices - 1]
    E_2Hg = A2g[:, 3][indices - 1]
    E_4AHg = A2g[:, 4][indices - 1]
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM")) as f:
        for i, line in enumerate(f):
            if i < _RSM_GAS_SEC3:
                continue
            if "---" in line:
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A3g = df[[2, 4, 7, 8, 9]].values
    D_3AHg = A3g[:, 0][indices - 1]
    E_3AHg = A3g[:, 1][indices - 1]
    B_4BHg = A3g[:, 2][indices - 1]
    D_4AHg = A3g[:, 3][indices - 1]
    D_1CHg = A3g[:, 4][indices - 1]
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM")) as f:
        for i, line in enumerate(f):
            if i < _RSM_GAS_SEC4:
                continue
            if "---" in line:
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A4g = df[[2, 4, 5, 6, 8, 9]].values
    B_4DHg = A4g[:, 0][indices - 1]
    E_3CHg = A4g[:, 1][indices - 1]
    E_2AHg = A4g[:, 2][indices - 1]
    D_3BHg = A4g[:, 3][indices - 1]
    B_1BHg = A4g[:, 4][indices - 1]
    K_3Hg = A4g[:, 5][indices - 1]
    WGAS1[:, 0] = B_1BHg.ravel()[steppi_indices - 1]
    WGAS1[:, 1] = B_1Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 2] = B_2Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 3] = B_3Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 4] = B_4BHg.ravel()[steppi_indices - 1]
    WGAS1[:, 5] = B_4DHg.ravel()[steppi_indices - 1]
    WGAS1[:, 6] = B_4Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 7] = D_1CHg.ravel()[steppi_indices - 1]
    WGAS1[:, 8] = D_1Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 9] = D_2Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 10] = D_3AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 11] = D_3BHg.ravel()[steppi_indices - 1]
    WGAS1[:, 12] = D_4AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 13] = D_4Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 14] = E_1Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 15] = E_2AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 16] = E_2Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 17] = E_3AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 18] = E_3CHg.ravel()[steppi_indices - 1]
    WGAS1[:, 19] = E_3Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 20] = E_4AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 21] = K_3Hg.ravel()[steppi_indices - 1]
    logger.info("Get the Well water injection Rate")
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM")) as f:
        for i, line in enumerate(f):
            if i < _RSM_WINJ_SEC1:
                continue
            if "---" in line:
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1win = df[[1, 2, 3, 4, 5, 6, 7, 8, 9]].values
    C_1Hwin = A1win[:, 0][indices - 1]
    C_2Hwin = A1win[:, 1][indices - 1]
    C_3Hwin = A1win[:, 2][indices - 1]
    C_4Hwin = A1win[:, 3][indices - 1]
    C_4AHwin = A1win[:, 4][indices - 1]
    F_1Hwin = A1win[:, 5][indices - 1]
    F_2Hwin = A1win[:, 6][indices - 1]
    F_3Hwin = A1win[:, 7][indices - 1]
    F_4Hwin = A1win[:, 8][indices - 1]
    WWINJ1[:, 0] = C_1Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 1] = C_2Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 2] = C_3Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 3] = C_4AHwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 4] = C_4Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 5] = F_1Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 6] = F_2Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 7] = F_3Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 8] = F_4Hwin.ravel()[steppi_indices - 1]
    logger.info("Get the Well Gas injection Rate")
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM")) as f:
        for i, line in enumerate(f):
            if i < _RSM_GINJ_SEC1:
                continue
            if "---" in line:
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1gin = df[[1, 3, 4, 5]].values
    C_1Hgin = A1gin[:, 0][indices - 1]
    C_3Hgin = A1gin[:, 1][indices - 1]
    C_4Hgin = A1gin[:, 2][indices - 1]
    C_4AHgin = A1gin[:, 3][indices - 1]
    WGASJ1[:, 0] = C_1Hgin.ravel()[steppi_indices - 1]
    WGASJ1[:, 1] = C_3Hgin.ravel()[steppi_indices - 1]
    WGASJ1[:, 2] = C_4AHgin.ravel()[steppi_indices - 1]
    WGASJ1[:, 3] = C_4Hgin.ravel()[steppi_indices - 1]
    DATA = {
        "OIL": WOIL1,
        "WATER": WWATER1,
        "GAS": WGAS1,
        "WATER_INJ": WWINJ1,
        "WGAS_inj": WGASJ1,
    }
    oil = np.reshape(WOIL1, (-1, 1), "F")
    water = np.reshape(WWATER1, (-1, 1), "F")
    gas = np.reshape(WGAS1, (-1, 1), "F")
    winj = np.reshape(WWINJ1, (-1, 1), "F")
    gasinj = np.reshape(WGASJ1, (-1, 1), "F")
    DATA2 = np.vstack([oil, water, gas, winj, gasinj])
    return DATA, DATA2


# ---------------------------------------------------------------------------
# Rate / well data processing
# (canonical: data_extract/opm_extract_rates.py)
# ---------------------------------------------------------------------------


def find_first_numeric_row(df):
    """Return first row index where all values are numeric, else None."""
    for i in range(len(df)):
        if df.iloc[i].apply(np.isreal).all():
            return i
    return None


def process_data(data):
    """Build a well-index dict from a list of COMPDAT-style tuples.

    Parameters
    ----------
    data : list[tuple]
        Each tuple is ``(well_name, i, j, k, k2)`` with 1-based indices.

    Returns
    -------
    dict
        Mapping of well name -> list of 0-based ``(i, j, k, k2)`` tuples.
    """
    well_indices = {}
    for entry in data:
        if entry[0] not in well_indices:
            well_indices[entry[0]] = []
        well_indices[entry[0]].append(
            (int(entry[1]) - 1, int(entry[2]) - 1, int(entry[3]) - 1, int(entry[4]) - 1)
        )
    return well_indices


def process_data2(data):
    """Build a well-index dict from WELSPECS ``(name, i, j)`` tuples.

    Parameters
    ----------
    data : list[tuple]
        Each tuple is ``(well_name, i, j)`` with 1-based I/J indices.

    Returns
    -------
    dict
        Mapping of well name -> list of 0-based ``(i, j)`` index tuples.
    """
    well_indices = {}
    for entry in data:
        well_name = entry[0]
        if well_name not in well_indices:
            well_indices[well_name] = []
        i_index = int(entry[1]) - 1
        j_index = int(entry[2]) - 1
        well_indices[well_name].append((i_index, j_index))
    return well_indices


def get_dyna(steppi, well_indices, swatuse):
    """Compute per-timestep mean grid values at completion intervals per well.

    Parameters
    ----------
    steppi : int
        Number of timesteps.
    well_indices : dict
        Mapping of well name -> list of ``(i, j, k, l)`` completion tuples
        (0-based indices).
    swatuse : np.ndarray
        4-D array of shape ``(steppi, nx, ny, nz)`` with the grid property
        (e.g., water saturation) to sample.

    Returns
    -------
    np.ndarray
        Array of shape ``(steppi, n_wells)`` with mean completion values.
    """
    mean_big_all = []
    for xx in range(steppi):
        mean_big = []
        for list1 in well_indices.values():
            temp_perm_values = [
                swatuse[xx, i_idx, j_idx, k_idx]
                if k_idx == l_idx
                else swatuse[xx, i_idx, j_idx, k_idx : l_idx + 1]
                for i_idx, j_idx, k_idx, l_idx in list1
            ]
            mean_all = np.mean(temp_perm_values)
            mean_big.append(mean_all)
        mean_big_all.append(mean_big)
    return np.array(mean_big_all)


def get_dyna2(
    steppi, well_indices, well_indicesg, well_indiceso, swatuse, gasuse, oiluse, Q, Qg
):
    """Distribute injection/production rates from wells into 4-D grid arrays.

    Assigns the average per-completion rate from `Q` and `Qg` into the
    corresponding cells of `swatuse`, `gasuse`, and sets `oiluse` to -1
    at producer completions.

    Parameters
    ----------
    steppi : int
        Number of timesteps.
    well_indices : list[tuple]
        COMPDAT tuples ``(name, i, j, k, k2)`` for water injectors.
    well_indicesg : list[tuple]
        COMPDAT tuples ``(name, i, j, k, k2)`` for gas injectors.
    well_indiceso : list[tuple]
        COMPDAT tuples ``(name, i, j, k, k2)`` for oil producers.
    swatuse : np.ndarray
        4-D water rate grid of shape ``(steppi, nx, ny, nz)``; updated in-place.
    gasuse : np.ndarray
        4-D gas rate grid of shape ``(steppi, nx, ny, nz)``; updated in-place.
    oiluse : np.ndarray
        4-D oil rate grid of shape ``(steppi, nx, ny, nz)``; updated in-place.
    Q : np.ndarray
        Water injection rates of shape ``(steppi, n_water_wells)``.
    Qg : np.ndarray
        Gas injection rates of shape ``(steppi, n_gas_wells)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Updated ``(swatuse, gasuse, oiluse)`` grids.
    """
    unique_well_names = OrderedDict()
    for _idx, tuple_entry in enumerate(well_indices):
        well_name = tuple_entry[0]
        if well_name not in unique_well_names:
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
            total_valueg = Q[xx, q_idxg]
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


def convert_to_list(well_data):
    """Flatten a well-index dict to a list of ``(i, j, well_name)`` tuples.

    Parameters
    ----------
    well_data : dict
        Mapping of well name -> list of ``(i, j)`` index tuples.

    Returns
    -------
    list[tuple]
        Flat list of ``(i, j, well_name)`` tuples.
    """
    output_list = []
    for well_name, indices in well_data.items():
        for i, j in indices:
            output_list.append((i, j, well_name))
    return output_list


def extract_tuples(set1, set2, set3, tuples_list):
    """Partition `tuples_list` into three groups based on well-name membership.

    Parameters
    ----------
    set1 : set[str]
        Well names for the gas-injector group.
    set2 : set[str]
        Well names for the water-injector group.
    set3 : set[str]
        Well names for the producer group.
    tuples_list : list[tuple]
        Source tuples of the form ``(i, j, well_name)``.

    Returns
    -------
    tuple[list, list, list]
        ``(gas_injectors, water_injectors, producers)`` each sorted by well
        name; producers excludes wells already in set1 or set2.
    """
    extracted_set1 = [tup for tup in tuples_list if tup[2] in set1]
    extracted_set1.sort(key=lambda x: x[2])
    extracted_set2 = [tup for tup in tuples_list if tup[2] in set2]
    extracted_set2.sort(key=lambda x: x[2])
    combined_set = list(set1) + list(set2)
    extracted_set3 = [tup for tup in tuples_list if tup[2] in set3]
    extracted_set3.sort(key=lambda x: x[2])
    final_remaining_list = [tup for tup in extracted_set3 if tup[2] not in combined_set]
    final_remaining_list.sort(key=lambda x: x[2])
    return extracted_set1, extracted_set2, final_remaining_list


def read_compdats(filename, well_names):
    """Read `COMPDAT` entries for the provided well names.

    Returns a list of tuples `(well, i, j, k, k2)` extracted directly from
    the deck file, preserving string tokens as parsed.
    """
    with open(filename) as file:
        start_collecting = False
        data = []
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith("--"):
                continue
            if "COMPDAT" in stripped_line:
                start_collecting = True
                continue
            if start_collecting and stripped_line.startswith("/"):
                start_collecting = False
                continue
            if start_collecting and stripped_line:
                parts = stripped_line.split()
                well_name = parts[0].strip("'")
                if well_name in well_names:
                    data.append((well_name, parts[1], parts[2], parts[3], parts[4]))
    return data


def read_compdats2(filename, file_path):
    """Parse `WELSPECS`/`WCONINJE` and producer names for NORNE decks.

    See also `read_compdats` for parsing the `COMPDAT` section.
    """
    with open(filename) as file:
        data_gas = []
        data_water = []
        data_oil = []
        injector_gas = set()
        injector_water = set()
        producer_oil = set()
        start_collecting_welspecs = False
        start_collecting_wconinje = False
        start_collecting_wconhist = False
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith("--"):
                continue
            if "WELSPECS" in stripped_line:
                start_collecting_welspecs = True
                continue
            if start_collecting_welspecs and stripped_line.startswith("/"):
                start_collecting_welspecs = False
                continue
            if "WCONINJE" in stripped_line:
                start_collecting_wconinje = True
                continue
            if start_collecting_wconinje and stripped_line.startswith("/"):
                start_collecting_wconinje = False
                continue
            if "WCONHIST" in stripped_line:
                start_collecting_wconhist = True
                continue
            if start_collecting_wconhist and stripped_line.startswith("/"):
                start_collecting_wconhist = False
                continue
            if start_collecting_welspecs:
                parts = stripped_line.split()
                if len(parts) > 5:
                    well_name = parts[0].strip("'")
                    i = parts[2]
                    j = parts[3]
                    if parts[5].strip("'") == "GAS":
                        data_gas.append((well_name, i, j))
                    elif parts[5].strip("'") == "WATER":
                        data_water.append((well_name, i, j))
                    elif parts[5].strip("'") == "OIL":
                        data_oil.append((well_name, i, j))
            if start_collecting_wconinje:
                parts = stripped_line.split()
                if len(parts) > 3:
                    well_name = parts[0].strip("'")
                    fluid_type = parts[1].strip("'")
                    if fluid_type == "GAS":
                        injector_gas.add(well_name)
                    elif fluid_type == "WATER":
                        injector_water.add(well_name)
            if start_collecting_wconhist:
                parts = stripped_line.split()
                if len(parts) > 3:
                    well_name = parts[0].strip("'")
                    producer_oil.add(well_name)
    data = convert_to_list(process_data2(data_oil))
    data.sort(key=lambda x: x[2])
    with open(file_path) as file:
        lines = file.readlines()
    well_namesoil = set()
    capture = False
    for line in lines:
        line = line.strip()
        if line == "WOPR":
            capture = True
            continue
        if capture:
            if line == "/":
                break
            well_name = line.strip(" '")
            well_namesoil.add(well_name)
    gas_injectors, water, oil = extract_tuples(
        injector_gas, injector_water, well_namesoil, data
    )
    return gas_injectors, oil, water


def process_dataframe(name, producer_well_names, vectors):
    """Extract numeric data for a vector and the TIME column from UNSMRY.

    Parameters
    ----------
    name : str
        Vector name (e.g., 'WOPR').
    producer_well_names : list[str]
        Well names to select column subsets by.
    vectors : pandas.DataFrame
        Multi-indexed UNSMRY dataframe from `EclBinaryParser.read_vectors()`.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        2-D array for the requested vector and 1-D TIME array, both
        starting from the first fully numeric row; None if not found.
    """
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
    else:
        result_array = None
    Time = vectors["TIME"]
    start_row = find_first_numeric_row(Time)
    if start_row is not None:
        numeric_df = Time.iloc[start_row:]
        time_array = numeric_df.to_numpy()
    else:
        time_array = None
    return result_array, time_array


def extract_qs(steppi, steppi_indices, filenameui, injectors, gas_injectors, filename):
    """Extract per-timestep gas and water injection rates for given wells.

    Parameters
    ----------
    steppi : int
        Number of timesteps used for sampling.
    steppi_indices : np.ndarray | int
        1-based indices used to sample rows from UNSMRY vectors.
    filenameui : str
        Base path (without extension) to Eclipse summary files.
    injectors : list[tuple]
        Water injector metadata; last element of each tuple is the well name.
    gas_injectors : list[tuple]
        Gas injector metadata; last element of each tuple is the well name.
    filename : str
        Path to additional deck context (unused here but kept for parity).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of shape `(steppi, nwells)` for gas (WGIR) and water (WWIR)
        injection rates sampled at `steppi_indices`.
    """
    from compare.sequential.misc_forward_utils import EclBinaryParser

    well_namesg = [entry[-1] for entry in gas_injectors]
    well_namesw = [entry[-1] for entry in injectors]
    unsmry_file = filenameui
    parser = EclBinaryParser(unsmry_file)
    vectorsdd = parser.read_vectors()
    namez = "WGIR"
    dfaa = vectorsdd[namez]
    filtered_columns = [
        coll
        for coll in dfaa.columns
        if any(well_namee in coll for well_namee in well_namesg)
    ]
    filtered_df = dfaa[filtered_columns]
    filtered_df = filtered_df[well_namesg]
    start_row = find_first_numeric_row(filtered_df)
    if start_row is None:
        raise ValueError(
            "WGIR table contains no numeric rows; check vector summary file."
        )
    numeric_df = filtered_df.iloc[start_row:]
    final_arrayg = numeric_df.to_numpy()
    final_arrayg[final_arrayg <= 0] = 0
    outg = final_arrayg[steppi_indices - 1, :].astype(float)
    outg[outg <= 0] = 0
    namez = "WWIR"
    dfaa = vectorsdd[namez]
    filtered_columns = [
        coll
        for coll in dfaa.columns
        if any(well_namee in coll for well_namee in well_namesw)
    ]
    filtered_df = dfaa[filtered_columns]
    filtered_df = filtered_df[well_namesw]
    start_row = find_first_numeric_row(filtered_df)
    if start_row is None:
        raise ValueError(
            "WWIR table contains no numeric rows; check vector summary file."
        )
    numeric_df = filtered_df.iloc[start_row:]
    final_arrayw = numeric_df.to_numpy()
    final_arrayw[final_arrayw <= 0] = 0
    outw = final_arrayw[steppi_indices - 1, :].astype(float)
    outw[outw <= 0] = 0
    return outg, outw


def Remove_folder(N_ens, straa):
    """Delete ``N_ens`` numbered ensemble folders ``straa + str(jj)``.

    Parameters
    ----------
    N_ens : int
        Number of ensemble folders to remove.
    straa : str
        Common path prefix; folder names are ``straa + str(jj)``.

    Returns
    -------
    None
    """
    for jj in range(N_ens):
        folderr = straa + str(jj)
        rmtree(folderr)


def process_and_print(data_dict, dict_name):
    """Replace NaN/Inf with ``1e-6``, clip to float32, and log array stats.

    Parameters
    ----------
    data_dict : dict
        Mapping of string keys to ``np.ndarray`` values; modified in-place.
    dict_name : str
        Human-readable name of the dict used in log messages.

    Returns
    -------
    None
    """
    from utils.array_utils import clip_and_convert_to_float32

    for key in data_dict:
        _n_nan = int(np.sum(np.isnan(data_dict[key])))
        _n_inf = int(np.sum(np.isinf(data_dict[key])))
        if _n_nan > 0:
            logger.warning(
                "Replacing %d NaN values in %s[%s] with 1e-6", _n_nan, dict_name, key
            )
        if _n_inf > 0:
            logger.warning(
                "Replacing %d Inf values in %s[%s] with 1e-6", _n_inf, dict_name, key
            )
        data_dict[key][np.isnan(data_dict[key])] = 1e-6
        data_dict[key][np.isinf(data_dict[key])] = 1e-6
        data_dict[key] = clip_and_convert_to_float32(data_dict[key])
    for key in data_dict:
        logger.info(f"For key '{key}' in {dict_name}:")


def process_task(k, x, y, z, seed, minn, maxx, minnp, maxxp, var, len_scale):
    """Generate one Gaussian random-field realisation for a parallel worker.

    Parameters
    ----------
    k : int
        Realisation index (unused in body; provided for parallelism bookkeeping).
    x : np.ndarray
        1-D coordinate array along the x-axis.
    y : np.ndarray
        1-D coordinate array along the y-axis.
    z : np.ndarray
        1-D coordinate array along the z-axis.
    seed : int
        Random seed for reproducible field generation.
    minn : float
        Lower bound for the permeability-range rescaling.
    maxx : float
        Upper bound for the permeability-range rescaling.
    minnp : float
        Lower bound for the porosity-range rescaling.
    maxxp : float
        Upper bound for the porosity-range rescaling.
    var : float
        Variance of the Gaussian covariance model.
    len_scale : float
        Length scale of the Gaussian covariance model.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Flattened permeability-like and porosity-like realisations.
    """
    model = Gaussian(dim=3, var=var, len_scale=len_scale)
    srf = SRF(model)
    aoutt = srf.structured([x, y, z], seed=seed)
    foo = np.reshape(aoutt, (-1, 1), "F")
    clfy = MinMaxScaler(feature_range=(minn, maxx))
    clfy.fit(foo)
    fout = clfy.transform(foo)
    clfy1 = MinMaxScaler(feature_range=(minnp, maxxp))
    clfy1.fit(foo)
    fout1 = clfy1.transform(foo)
    return np.ravel(fout), np.ravel(fout1)
