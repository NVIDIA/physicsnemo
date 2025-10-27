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
                    PLOTTING UTILITIES AND HELPERS
=====================================================================

This module provides utility functions for plotting and visualization in reservoir
simulation. It includes helper functions for plot styling, color management,
and figure optimization.

Key Features:
- Plot styling and customization utilities
- Color mapping and management
- Figure optimization and memory management
- Plot helper functions
- Visualization utilities
- Performance optimization for plotting

Usage:
    from compare.batch.misc_plotting_utils import (
        setup_plot_style,
        create_color_maps,
        manage_figures
    )

Inputs:
    - Plot configuration parameters
    - Styling options
    - Color specifications
    - Figure management settings

Outputs:
    - Configured plot styles
    - Color maps and palettes
    - Optimized figures
    - Plot utilities

@Author : Clement Etienam
"""

# 🛠 Standard Library
import gc
import gzip
import pickle
import logging
from collections import OrderedDict

# 🔧 Third-party Libraries
import numpy as np
import numpy.matlib
import pandas as pd
# import scipy.io as sio

# 📦 Local Modules
from hydra.utils import to_absolute_path
from compare.batch.misc_gather import (
    process_data2,
    convert_to_list,
    extract_tuples,
)


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Forward problem")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def simulation_data_types():
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


def Get_Time(nx, ny, nz, steppi, steppi_indices, N):
    logger = setup_logging()
    logger.info("Load simulated labelled training data")
    with gzip.open(to_absolute_path("../data/data_train.pkl.gz"), "rb") as f2:
        mat = pickle.load(f2)
    X_data1 = mat
    del mat
    gc.collect()
    # mat = sio.loadmat(to_absolute_path("../data/conversions.mat"))
    Time = X_data1["Time"]  # * mat ["maxT"]
    np_array2 = np.zeros((Time.shape[1]))
    for mm in range(Time.shape[1]):
        np_array2[mm] = Time[0, mm, 0, 0, 0]
    Timee = []
    for k in range(N):
        check = np.ones((nx, ny, nz), dtype=np.float16)
        unie = []
        for zz in range(len(np_array2)):
            aa = np_array2[zz] * check
            unie.append(aa)
        Time = np.stack(unie, axis=0)
        Timee.append(Time)
    Timee = np.stack(Timee, axis=0)
    return Timee


def assign_faults(well_indices, nx, ny, nz, well_amount, data):
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
            # print(f"Updating faultm for well '{well_name}' at indices ({i_idx},{j_idx},{k_idx}) to ({i1_idx},{j1_idx},{k1_idx}) with value {average_value}")
            faultm[
                int(i_idx) - 1 : int(i1_idx),
                int(j_idx) - 1 : int(j1_idx),
                int(k_idx) - 1 : int(k1_idx),
            ] = average_value
    return faultm


def Get_fault(filename):
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


def read_faults(filename, well_names):
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
                # Strip quotes from the well name
                well_name = parts[0].strip("'")
                # print("Parts found:", parts)
                if well_name in well_names:
                    # Extract and add the tuple of the required columns to the list
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


def read_compdats2(filename, file_path):
    with open(filename, "r") as file:
        data_gas = []  # List to collect gas entries
        data_water = []  # List to collect water entries
        data_oil = []  # List to collect oil entries
        injector_gas = set()  # Set to collect gas injector well names
        injector_water = set()  # Set to collect water injector well names
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
                if (
                    len(parts) > 5
                ):  # Ensure the line has enough parts to avoid index errors
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
                if (
                    len(parts) > 3
                ):  # Ensure the line has enough parts to avoid index errors
                    well_name = parts[0].strip("'")
                    fluid_type = parts[1].strip("'")
                    if fluid_type == "GAS":
                        injector_gas.add(well_name)
                    elif fluid_type == "WATER":
                        injector_water.add(well_name)
            if start_collecting_wconhist:
                parts = stripped_line.split()
                if (
                    len(parts) > 3
                ):  # Ensure the line has enough parts to avoid index errors
                    well_name = parts[0].strip("'")
                    producer_oil.add(well_name)
    data = convert_to_list(process_data2(data_oil))
    data.sort(key=lambda x: x[2])
    with open(file_path, "r") as file:
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
    gass, water, oil = extract_tuples(injector_gas, injector_water, well_namesoil, data)
    return gass, oil, water


def historydata(timestep, steppi, steppi_indices):
    WOIL1 = np.zeros((steppi, 22))
    WWATER1 = np.zeros((steppi, 22))
    WGAS1 = np.zeros((steppi, 22))
    WWINJ1 = np.zeros((steppi, 9))
    WGASJ1 = np.zeros((steppi, 4))
    indices = timestep
    logger = setup_logging()
    logger.info(" Get the Well Oil Production Rate")
    lines = []
    with open(to_absolute_path("../NORNE/NORNE_ATW2013.RSM"), "r") as f:
        for i, line in enumerate(f):
            if i < 47873:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
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
    with open(to_absolute_path("../NORNE/NORNE_ATW2013.RSM"), "r") as f:
        for i, line in enumerate(f):
            if i < 48743:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
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
    with open(to_absolute_path("../NORNE/NORNE_ATW2013.RSM"), "r") as f:
        for i, line in enumerate(f):
            if i < 49613:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
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
    with open(to_absolute_path("../NORNE/NORNE_ATW2013.RSM"), "r") as f:
        for i, line in enumerate(f):
            if i < 50483:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
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
    logger = setup_logging()
    logger.info(" Get the Well water Production Rate")
    lines = []
    with open(to_absolute_path("../NORNE/NORNE_ATW2013.RSM"), "r") as f:
        for i, line in enumerate(f):
            if i < 40913:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
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
    with open(to_absolute_path("../NORNE/NORNE_ATW2013.RSM"), "r") as f:
        for i, line in enumerate(f):
            if i < 41783:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
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
    with open(to_absolute_path("../NORNE/NORNE_ATW2013.RSM"), "r") as f:
        for i, line in enumerate(f):
            if i < 42653:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
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
    with open(to_absolute_path("../NORNE/NORNE_ATW2013.RSM"), "r") as f:
        for i, line in enumerate(f):
            if i < 43523:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
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
    logger = setup_logging()
    logger.info(" Get the Well Gas Production Rate")
    lines = []
    with open(to_absolute_path("../NORNE/NORNE_ATW2013.RSM"), "r") as f:
        for i, line in enumerate(f):
            if i < 54833:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
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
    with open(to_absolute_path("../NORNE/NORNE_ATW2013.RSM"), "r") as f:
        for i, line in enumerate(f):
            if i < 55703:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
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
    with open(to_absolute_path("../NORNE/NORNE_ATW2013.RSM"), "r") as f:
        for i, line in enumerate(f):
            if i < 56573:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
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
    with open(to_absolute_path("../NORNE/NORNE_ATW2013.RSM"), "r") as f:
        for i, line in enumerate(f):
            if i < 57443:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
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
    logger = setup_logging()
    logger.info(" Get the Well water injection Rate")
    lines = []
    with open(to_absolute_path("../NORNE/NORNE_ATW2013.RSM"), "r") as f:
        for i, line in enumerate(f):
            if i < 72237:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
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
    logger = setup_logging()
    logger.info(" Get the Well Gas injection Rate")
    lines = []
    with open(to_absolute_path("../NORNE/NORNE_ATW2013.RSM"), "r") as f:
        for i, line in enumerate(f):
            if i < 73977:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
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
