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
                    DATA GATHERING AND AGGREGATION UTILITIES
=====================================================================

This module provides utilities for gathering and aggregating data from reservoir
simulation results. It includes functions for data collection, processing,
validation, and result aggregation.

Key Features:
- Data collection from simulation outputs
- Result aggregation and processing
- Data validation and quality checks
- File I/O operations with proper error handling
- Statistical analysis and metrics calculation
- Visualization data preparation
- Ensemble data management

Usage:
    from compare.batch.misc_gather import (
        gather_results,
        aggregate_ensemble_data,
        process_simulation_outputs
    )

Inputs:
    - Simulation output files
    - Configuration parameters
    - Data arrays for processing
    - File paths for I/O operations

Outputs:
    - Aggregated simulation results
    - Processed data arrays
    - Statistical summaries
    - Logged status messages

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import re
import logging
import warnings
from collections import OrderedDict, namedtuple
from struct import unpack_from
from mmap import mmap

# 🔧 Third-party Libraries
import numpy as np
import numpy.matlib
import numpy.ma as ma
import pandas as pd

# 📦 Local Modules
# No external compare.batch imports needed - all functions are defined locally


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


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


def process_data2(data):
    well_indices = {}
    for entry in data:
        well_name = entry[0]
        if well_name not in well_indices:
            well_indices[well_name] = []
        i_index = int(entry[1]) - 1  # Convert to zero-based index
        j_index = int(entry[2]) - 1  # Convert to zero-based index
        well_indices[well_name].append((i_index, j_index))
    return well_indices


def find_first_numeric_row(df):
    """Find the first row in the DataFrame where all data is numeric."""
    for i in range(len(df)):
        if df.iloc[i].apply(np.isreal).all():
            return i
    return None


def convert_to_list(well_data):
    output_list = []
    for well_name, indices in well_data.items():
        for i, j in indices:
            output_list.append((i, j, well_name))
    return output_list


def extract_tuples(set1, set2, set3, tuples_list):
    # Extract tuples for set1
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
    with open(filename, "r") as file:
        start_collecting = False
        data = []  # List to collect all entries
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


def process_dataframe(name, producer_well_names, vectors):
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
        logger = setup_logging()
        logger.info(f"Numeric data from {name} processed successfully.")
    else:
        logger = setup_logging()
        logger.info(f"No numeric rows found in the DataFrame for {name}.")
        result_array = None
    Time = vectors["TIME"]
    start_row = find_first_numeric_row(Time)
    if start_row is not None:
        numeric_df = Time.iloc[start_row:]
        time_array = numeric_df.to_numpy()
        logger = setup_logging()
        logger.info(f"Numeric data from {name} processed successfully.")
    else:
        logger = setup_logging()
        logger.info(f"No numeric rows found in the DataFrame for {name}.")
        time_array = None
    return result_array, time_array


def extract_qs(steppi, steppi_indices, filenameui, injectors, gass, filename):
    well_namesg = [entry[-1] for entry in gass]  # gas injectors well names
    well_namesw = [entry[-1] for entry in injectors]  # water injectors well names
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
    if start_row is not None:
        numeric_df = filtered_df.iloc[start_row:]
        all_arrays = numeric_df.to_numpy()
    else:
        all_arrays = None
    final_arrayg = all_arrays
    if final_arrayg is None:
        return None, None
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
    if start_row is not None:
        numeric_df = filtered_df.iloc[start_row:]
        all_arrays = numeric_df.to_numpy()
    else:
        all_arrays = None
    final_arrayg = all_arrays
    if final_arrayg is None:
        return outg, None
    final_arrayg[final_arrayg <= 0] = 0
    outw = final_arrayg[steppi_indices - 1, :].astype(float)
    outw[outw <= 0] = 0
    return outg, outw


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


def get_dyna2(
    steppi, well_indices, well_indicesg, well_indiceso, swatuse, gasuse, oiluse, Q, Qg
):
    unique_well_names = OrderedDict()
    for idx, tuple_entry in enumerate(well_indices):
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
                # print(i_idx, j_idx, k_idx)
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
    for idx, tuple_entry in enumerate(well_indicesg):
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
    for idx, tuple_entry in enumerate(well_indiceso):
        well_nameo = tuple_entry[0]
        if well_nameo not in unique_well_nameso:
            unique_well_nameso[well_nameo] = len(unique_well_nameso)
    well_name_to_indexo = {name: index for index, name in enumerate(unique_well_nameso)}
    for xx in range(steppi):
        for well_nameo, q_idxo in well_name_to_indexo.items():
            # Find all tuples corresponding to this well name to update swatuse accordingly
            entries_for_wello = [t for t in well_indiceso if t[0] == well_nameo]
            for _, i_idx, j_idx, k_idx, l_idx in entries_for_wello:
                # print(i_idx, j_idx, k_idx)
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
