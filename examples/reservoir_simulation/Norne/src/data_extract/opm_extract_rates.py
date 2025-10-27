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

OPM (Open Porous Media) extraction utilities for Eclipse/Flow-style
reservoir simulations. This module provides helpers to:

- Read Eclipse summary files (SMSPEC/UNSMRY) to obtain vector time series
  and the TIME column.
- Parse deck sections (e.g., COMPDAT, WELSPECS, WCONINJE, WCONHIST) to
  discover well names and completion intervals.
- Derive training/test tensors for surrogate models (FNO/PINO) from
  simulator outputs, including gridded pressures and saturations.
- Compute well-rate arrays for gas and water injectors/producers.
- Set up logging and tracking backends for distributed runs.

Main entry points
-----------------
- read_compdats / read_compdats2: Parse deck files to extract per-well
  completion tuples.
- extract_qs: Build per-step injection rates for specified wells.
- Get_data_FFNN / Get_data_FFNN1: Construct model inputs/targets arrays
  for feed-forward neural networks.
- historydata: Read historical NORNE RSM rates and assemble matrices.
- InitializeLoggers: Configure distributed logging and MLflow tracking.

Notes
-----
- File paths used here are expected to be absolute or resolved via
  hydra.utils.to_absolute_path when running under Hydra.
- Many functions assume Eclipse binary files (EGRID/UNRST/SMSPEC) are present
  in the working directory referenced by the deck path.

@Author: Clement Etienam
"""

import os
import time
import logging
import warnings
from omegaconf import DictConfig
import re
import shutil
import numpy as np
import numpy.linalg
import numpy.matlib
import pandas as pd
from typing import Tuple
from shutil import rmtree
from collections import OrderedDict
from cpuinfo import get_cpu_info
from sklearn.preprocessing import MinMaxScaler
from gstools import SRF, Gaussian
import torch
import hashlib

# Move heavy imports below or gate at runtime to satisfy E402
from hydra.utils import to_absolute_path
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.distributed import DistributedManager
import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
from data_extract.opm_extract_props_geostats import clip_and_convert_to_float32
from compare.batch.misc_forward_utils import EclBinaryParser


def is_available():
    """Return True if `nvidia-smi` is callable, else False.

    This is a lightweight probe for logging; it does not guarantee CUDA
    availability inside frameworks such as PyTorch.
    """
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
    logger.info("GPU Available with CUDA")
else:
    logger.info("No GPU Available")

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


def clip_and_convert_to_float3(array):
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min
    array_clipped = np.clip(array, min_float32, max_float32)
    # array_clipped = round_array_to_4dp(array_clipped)
    return array_clipped.astype(np.float32)


def Make_correct(array):
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


def Split_Matrix(matrix, sizee):
    x_split = np.split(matrix, sizee, axis=0)
    return x_split


def extract_qs(steppi, steppi_indices, filenameui, injectors, gass, filename):
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
    gass : list[tuple]
        Gas injector metadata; last element of each tuple is the well name.
    filename : str
        Path to additional deck context (unused here but kept for parity).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of shape `(steppi, nwells)` for gas (WGIR) and water (WWIR)
        injection rates sampled at `steppi_indices`.
    """
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
    """Build FFNN input/output arrays from Eclipse vectors and grid tensors.

    Constructs per-timestep features by aggregating grid values around
    completion intervals and concatenating global statistics, then extracts
    target rate vectors for selected producer wells.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(innn, ouut)` shaped `(N, steppi, features)` and
        `(N, steppi, lenwels*N_pr)` respectively.
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
                # logger.info(f"Numeric data from {namey} processed successfully.")
            else:
                # logger.warning(f"No numeric rows found in the DataFrame for {namey}.")
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


def find_first_numeric_row(df):
    """Return first row index where all values are numeric, else None.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose rows will be scanned.

    Returns
    -------
    int | None
        Zero-based index of the first fully-numeric row, or None if absent.
    """
    for i in range(len(df)):
        if df.iloc[i].apply(np.isreal).all():
            return i
    return None


def process_data(data):
    well_indices = {}
    for entry in data:
        if entry[0] not in well_indices:
            well_indices[entry[0]] = []
        well_indices[entry[0]].append(
            (int(entry[1]) - 1, int(entry[2]) - 1, int(entry[3]) - 1, int(entry[4]) - 1)
        )
    return well_indices


def read_compdats2(filename, file_path):
    """Parse `WELSPECS`/`WCONINJE` and producer names for NORNE decks.

    See also `read_compdats` for parsing the `COMPDAT` section.
    """
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


def convert_to_list(well_data):
    output_list = []
    for well_name, indices in well_data.items():
        for i, j in indices:
            output_list.append((i, j, well_name))
    return output_list


def extract_tuples(set1, set2, set3, tuples_list):
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
        # logger.info(f"Numeric data from {name} processed successfully.")
    else:
        # logger.warning(f"No numeric rows found in the DataFrame for {name}.")
        result_array = None
    Time = vectors["TIME"]
    start_row = find_first_numeric_row(Time)
    if start_row is not None:
        numeric_df = Time.iloc[start_row:]
        time_array = numeric_df.to_numpy()
        # logger.info(f"Numeric data from {name} processed successfully.")
    else:
        # logger.warning(f"No numeric rows found in the DataFrame for {name}.")
        time_array = None
    return result_array, time_array


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
    """Build FFNN input/output arrays using a fixed `folder` per sample.

    Same outputs as `Get_data_FFNN`, but the caller provides the folder path
    where UNSMRY files reside for all iterations.
    """
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
            start_row = find_first_numeric_row(filtered_df)
            if start_row is not None:
                numeric_df = filtered_df.iloc[start_row:]
                result_array = numeric_df.to_numpy()
                print(f"Numeric data from {namey} processed successfully.")
            else:
                print(f"No numeric rows found in the DataFrame for {namey}.")
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


def Remove_folder(N_ens, straa):
    for jj in range(N_ens):
        folderr = straa + str(jj)
        rmtree(folderr)


def historydata(timestep, steppi, steppi_indices, source_dir):
    """Load historical NORNE RSM data and assemble category matrices.

    Returns two structures: a dict mapping category names to `(steppi, n)`
    matrices, and a vertically stacked vectorised array with all categories
    concatenated column-wise for convenience.
    """
    WOIL1 = np.zeros((steppi, 22))
    WWATER1 = np.zeros((steppi, 22))
    WGAS1 = np.zeros((steppi, 22))
    WWINJ1 = np.zeros((steppi, 9))
    WGASJ1 = np.zeros((steppi, 4))
    indices = timestep
    logger.info("Get the Well Oil Production Rate")
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM"), "r") as f:
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
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM"), "r") as f:
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
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM"), "r") as f:
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
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM"), "r") as f:
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
    logger.info("Get the Well water Production Rate")
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM"), "r") as f:
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
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM"), "r") as f:
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
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM"), "r") as f:
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
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM"), "r") as f:
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
    logger.info("Get the Well Gas Production Rate")
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM"), "r") as f:
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
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM"), "r") as f:
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
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM"), "r") as f:
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
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM"), "r") as f:
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
    logger.info("Get the Well water injection Rate")
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM"), "r") as f:
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
    logger.info("Get the Well Gas injection Rate")
    lines = []
    with open(os.path.join(source_dir, "NORNE_ATW2013.RSM"), "r") as f:
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


def replace_nan_with_zero(tensor):
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    invalid_mask = nan_mask | inf_mask
    valid_elements = tensor[~invalid_mask]  # Elements that are not NaN or Inf
    if valid_elements.numel() > 0:  # Ensure there are valid elements to calculate mean
        mean_value = valid_elements.mean()
    else:
        mean_value = torch.tensor(1e-6, device=tensor.device)
    return torch.where(invalid_mask, mean_value, tensor)


def interp_torch(cuda, reference_matrix1, reference_matrix2, tensor1):
    chunk_size = 1
    chunks = torch.chunk(tensor1, chunks=chunk_size, dim=0)
    processed_chunks = []
    for start_idx in range(chunk_size):
        interpolated_chunk = linear_interp(
            chunks[start_idx], reference_matrix1, reference_matrix2
        )
        processed_chunks.append(interpolated_chunk)
    torch.cuda.empty_cache()
    return processed_chunks


def get_model_hash(model):
    state_dict = model.state_dict()
    state_bytes = torch.save(state_dict, None, _use_new_zipfile_serialization=False)
    return hashlib.sha256(state_bytes).hexdigest()


def write_RSM(data, Time, Name, well_names):
    groups = ["WOPR(bbl/day)", "WWPR(bbl/day)", "WGPR(scf/day)"]
    columns = well_names  # ['L1', 'L2', 'L3', 'LU1', 'LU2',
    headers = pd.MultiIndex.from_product([groups, columns])
    df = pd.DataFrame(data, columns=headers)
    df.insert(0, "Time(days)", Time)
    with pd.ExcelWriter(Name + ".xlsx", engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Sheet1")
        writer.sheets["Sheet1"] = worksheet
        header_format = workbook.add_format({"bold": True, "align": "center"})
        worksheet.write(0, 0, "Time(days)", header_format)
        col = 1
        for sub_col in columns * len(groups):
            worksheet.write(1, col, sub_col)
            col += 1
        col = 1
        for group in groups:
            end_col = col + len(columns) - 1
            worksheet.merge_range(0, col, 0, end_col, group, header_format)
            col = end_col + 1
        for row_num, row_data in enumerate(df.values):
            worksheet.write_row(row_num + 2, 0, row_data)
        time_data = np.arange(1, 101)
        for row_num, time_val in enumerate(time_data):
            worksheet.write(row_num + 2, 0, time_val)
        worksheet.set_column(0, 0, 12)  # 'Time(days)' column
        worksheet.set_column(1, col, 10)  # Data columns


def sort_key(s):
    return int(re.search(r"\d+", s).group())


def process_and_print(data_dict, dict_name):
    for key in data_dict.keys():
        data_dict[key][np.isnan(data_dict[key])] = 1e-6
        data_dict[key][np.isinf(data_dict[key])] = 1e-6
        data_dict[key] = clip_and_convert_to_float32(data_dict[key])
    for key, value in data_dict.items():
        logger.info(f"For key '{key}' in {dict_name}:")


def normalize_tensors_adjusted(tensor_dict):
    normalized_dict = {}
    for key, tensor in tensor_dict.items():
        tensor = tensor.to(torch.float32)
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        if max_val - min_val > 0:
            tensor = (tensor - min_val) / (max_val - min_val)  # ✅ Out-of-place
            perturbation = torch.clamp(
                torch.normal(
                    mean=0.1, std=0.01, size=tensor.size(), device=tensor.device
                ),
                min=0.1,
            )
            tensor = tensor * 0.9 + perturbation  # ✅ Out-of-place
        else:
            perturbation = torch.clamp(
                torch.normal(
                    mean=0.1, std=0.01, size=tensor.size(), device=tensor.device
                ),
                min=0.1,
            )
            tensor = torch.zeros_like(tensor) + perturbation  # ✅ Out-of-place
        normalized_dict[key] = tensor
        del min_val, max_val, perturbation  # Free memory of intermediate variables
    return normalized_dict


def process_task(k, x, y, z, seed, minn, maxx, minnp, maxxp, var, len_scale):
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


def safe_mean_std(data):
    mean = np.mean(data, axis=None)
    std = np.std(data, axis=None, ddof=1)
    if np.isinf(std):
        std = None
    return mean, std


def safe_rmtree(path, retries=3, delay=1):
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
                logger.info(f"Removed folder: {path}")
            else:
                logger.info(f"Folder does not exist: {path}")
            break
        except OSError as e:
            if "Device or resource busy" in str(e):
                logger.warning(f"Resource busy: {path}. Retrying...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to remove {path}: {e}")
    else:
        logger.error(f"Failed to remove directory: {path} after {retries} attempts.")


def InitializeLoggers(cfg: DictConfig) -> Tuple[DistributedManager, PythonLogger]:
    """Initialise distributed logging, MLflow tracking, and rank context.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with logging and experiment settings.

    Returns
    -------
    tuple[DistributedManager, PythonLogger]
        Distributed context and a rank-aware logger wrapper.
    """

    DistributedManager.initialize()
    dist_manager = DistributedManager()
    logger = PythonLogger(name=f"PhyNeMo Reservoir_Characterisation{dist_manager.rank}")
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(dist_manager.rank)
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(dist_manager.rank % torch.cuda.device_count())
    logger.info(
        f"Process {os.getenv('RANK')}: torch.cuda.device_count() = {torch.cuda.device_count()}"
    )
    logger.info(
        f"Process {os.getenv('RANK')}: Visible GPUs = {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
    )
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device_id = dist_manager.rank % gpu_count  # Map rank to available GPUs
        torch.cuda.set_device(device_id)
        logger.info(
            f"Process {dist_manager.rank} is using GPU {device_id}: {torch.cuda.get_device_name(device_id)}"
        )
    else:
        logger.info(f"Process {dist_manager.rank} is using CPU")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    experiment_name = "PhyNeMo-Reservoir Modelling"
    experiment_id = None
    if dist_manager.rank == 0:
        try:
            tracking_dir = os.path.join(os.getcwd(), "mlruns")

            if os.path.exists(tracking_dir):
                logger.info(f"Removing existing directory: {tracking_dir}")
                os.system(f"rm -rf {tracking_dir}")  # Force delete using shell command
                # shutil.rmtree(tracking_dir)  # Remove directory and all contents
            else:
                os.makedirs(tracking_dir, exist_ok=True)
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{tracking_dir}"
            mlflow.set_tracking_uri(f"file://{tracking_dir}")
            client = MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.info(f"[MLflow] Creating new experiment: {experiment_name}")
                experiment_id = client.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"[MLflow] Using existing experiment ID: {experiment_id}")
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name="PhyNeMo-Training")
            logger.info(
                f"[MLflow] Started run for experiment '{experiment_name}' with ID {experiment_id}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize MLFlow on rank 0: {e}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return dist_manager, RankZeroLoggingWrapper(logger, dist_manager)


def on_rm_error(func, path, exc_info):
    logger.warning(f"Error removing {path}. Retrying...")
    try:
        os.chmod(path, 0o777)  # Change permissions to ensure it can be deleted
        func(path)  # Retry removing
    except Exception as e:
        logger.error(f"Failed to remove {path}: {e}")


def check_and_remove_dirs(directories, response):
    for directory in directories:
        if os.path.exists(directory) and os.path.isdir(directory):
            # response = user_response.lower().strip()
            if response == "yes":
                logger.info(f"Removing: {directory} ...")
                try:
                    shutil.rmtree(directory, onerror=on_rm_error)
                    logger.info(f"Successfully removed: {directory}")
                except Exception as e:
                    logger.error(f"Error removing {directory}: {e}")
            else:
                logger.info(f"Skipped: {directory}")
        else:
            logger.info(f"Directory '{directory}' does not exist.")


def are_models_equal(model1, model2):
    return all(
        torch.equal(param1, param2)
        for param1, param2 in zip(
            model1.state_dict().values(), model2.state_dict().values()
        )
    )
