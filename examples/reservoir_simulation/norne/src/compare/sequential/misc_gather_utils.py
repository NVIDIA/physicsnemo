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
                    SEQUENTIAL GATHER UTILITIES - CORE FUNCTIONS
=====================================================================

This module provides core gather utilities for sequential FVM
surrogate model comparisons. It includes functions for data processing,
file handling, and analysis.

Key Features:
- Data type definitions for simulation data
- File parsing and data extraction utilities
- Data processing and conversion functions
- Simulation and analysis utilities

Usage:
    from compare.sequential.misc_gather_utils import (
        get_all_properties,
        ensemble_pytorch,
        copy_files,
        save_files
    )

@Author : Clement Etienam
"""

import os
import math
import re
import shlex
import shutil
import subprocess
from collections import OrderedDict
from struct import unpack
import fnmatch

import numpy as np
import numpy.matlib
import torch
from skimage.transform import resize as rzz

from hydra.utils import to_absolute_path
from compare.sequential.misc_forward import (
    get_dyna2,
)

from compare.sequential.misc_gather import (
    EclBinaryParser,
    extract_qs,
)


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


def find_first_numeric_row(df):
    """Find the first row in the DataFrame where all data is numeric."""
    for i in range(len(df)):
        if df.iloc[i].apply(np.isreal).all():
            return i
    return None


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
    # injector_gas.sort(key=lambda x: x[2])
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
                # print("Started collecting data after COMPDAT")
                continue
            if start_collecting and stripped_line.startswith("/"):
                start_collecting = False
                # print("Stopped collecting data after encountering '/'")
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
            # print(f"Updating faultm for well '{well_name}' at indices ({i_idx},{j_idx},{k_idx}) to ({i1_idx},{j1_idx},{k1_idx}) with value {average_value}")
            faultm[
                int(i_idx) - 1 : int(i1_idx),
                int(j_idx) - 1 : int(j1_idx),
                int(k_idx) - 1 : int(k1_idx),
            ] = average_value
    return faultm


def get_all_properties(
    folder,
    nx,
    ny,
    nz,
    oldfolder,
    steppi,
    steppi_indices,
    filenameui,
    injectors,
    gas_injectors,
    filename,
    compdat_datag,
    compdat_dataw,
    compdat_data,
    input_variables,
    output_variables,
    filename_fault,
    FAULT_INCLUDE,
):
    """
    Load and assemble all simulation output properties for a single ensemble folder.

    Parameters
    ----------
    folder : str
        Path to the simulation run directory to change into.
    nx : int
        Grid dimension in the X direction.
    ny : int
        Grid dimension in the Y direction.
    nz : int
        Grid dimension in the Z direction.
    oldfolder : str
        Original working directory to restore after processing.
    steppi : int
        Number of output time steps to extract.
    steppi_indices : np.ndarray
        1-based indices of time steps to select.
    filenameui : str
        Base path to the UNSMRY/SMSPEC binary summary files.
    injectors : list of tuple
        Water injector COMPDAT entries.
    gas_injectors : list of tuple
        Gas injector COMPDAT entries.
    filename : str
        Path to the DATA file used for COMPDAT parsing.
    compdat_datag : list of tuple
        Pre-parsed gas COMPDAT data entries.
    compdat_dataw : list of tuple
        Pre-parsed water COMPDAT data entries.
    compdat_data : list of tuple
        Pre-parsed oil producer COMPDAT data entries.
    input_variables : list of str
        Names of input property variables (e.g. ['PERM', 'FAULT']).
    output_variables : list of str
        Names of output state variables (e.g. ['PRESSURE', 'SWAT']).
    filename_fault : str
        Path to the DATA file containing the FAULTS keyword.
    FAULT_INCLUDE : str
        Path to the fault multiplier include file (multflt.dat or similar).

    Returns
    -------
    return_values : dict
        Dictionary containing requested output arrays keyed by variable name
        (e.g. 'PRESSURE', 'SWAT', 'SGAS', 'SOIL', 'Time', 'actnum', 'QG', 'QW', 'QO', 'FAULT').
    """
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
    for slices in zip(*(filtered_states[var] for var in filtered_states), strict=False):
        for state_var, var_name in zip(slices, filtered_states, strict=False):
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
        steppi, steppi_indices, filenameui, injectors, gas_injectors, filename
    )
    awater, agas, aoil = get_dyna2(
        steppi, compdat_dataw, compdat_datag, compdat_data, Qw, Qg, Qo, seew, seeg
    )
    if "FAULT" in input_variables:
        float_parameters = []
        file_path = FAULT_INCLUDE  # "multflt.dat"
        with open(file_path) as file:
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


def fit_operation(tensor, target_min, target_max, tensor_min, tensor_max):
    """
    Normalise a tensor by dividing by its maximum value.

    Parameters
    ----------
    tensor : np.ndarray
        Array to normalise.
    target_min : float
        Unused target minimum (reserved for interface consistency).
    target_max : float
        Unused target maximum (reserved for interface consistency).
    tensor_min : float
        Unused observed minimum (reserved for interface consistency).
    tensor_max : float
        Divisor used to normalise the tensor.

    Returns
    -------
    rescaled_tensor : np.ndarray
        Tensor divided by tensor_max.
    """
    return tensor / tensor_max


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
        fault_data = read_faults(filename_fault, fault_temp)  # OIl
        Fault = assign_faults(fault_data, nx, ny, nz, fault_temp, floatts)
        flt.append(Fault)
    flt = np.stack(flt, axis=0)[:, None, :, :, :]
    return np.stack(flt, axis=0)


def ensemble_pytorch(
    param,
    nx,
    ny,
    nz,
    Ne,
    effective,
    oldfolder,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    minQ,
    maxQ,
    minQw,
    maxQw,
    minQg,
    maxQg,
    steppi,
    device,
    steppi_indices,
    input_variables,
    cfg,
):
    """
    Assemble a normalised PyTorch input dictionary for an ensemble of realisations.

    Parameters
    ----------
    param : dict
        Ensemble parameter arrays keyed by variable name (e.g. 'PERM', 'PORO', 'FAULT').
    nx : int
        Grid dimension in the X direction.
    ny : int
        Grid dimension in the Y direction.
    nz : int
        Grid dimension in the Z direction.
    Ne : int
        Number of ensemble members.
    effective : np.ndarray
        Active-cell mask array (unused here, reserved for future use).
    oldfolder : str
        Original working directory (unused, reserved for interface consistency).
    target_min : float
        Target minimum for normalisation (forwarded to fit_operation).
    target_max : float
        Target maximum for normalisation (forwarded to fit_operation).
    minK : float
        Minimum permeability value for normalisation.
    maxK : float
        Maximum permeability value for normalisation.
    minT : float
        Minimum time value (unused, reserved).
    maxT : float
        Maximum time value (unused, reserved).
    minP : float
        Minimum pressure value for normalisation.
    maxP : float
        Maximum pressure value for normalisation.
    minQ : float
        Minimum water injection rate (unused, reserved).
    maxQ : float
        Maximum water injection rate (unused, reserved).
    minQw : float
        Minimum water rate (unused, reserved).
    maxQw : float
        Maximum water rate (unused, reserved).
    minQg : float
        Minimum gas rate (unused, reserved).
    maxQg : float
        Maximum gas rate (unused, reserved).
    steppi : int
        Number of output time steps.
    device : str
        PyTorch device string (e.g. 'cuda' or 'cpu').
    steppi_indices : np.ndarray
        1-based indices of selected time steps.
    input_variables : list of str
        Variable names to include (e.g. ['PERM', 'PORO', 'PINI', 'FAULT']).
    cfg : object
        Hydra config object providing PROPS and FAULT_INCLUDEE / FAULT_DATA paths.

    Returns
    -------
    inn : dict
        Dictionary of torch.Tensor arrays on the specified device, keyed by
        lowercase variable name (e.g. 'perm', 'poro', 'pini', 'fault').
    """
    if "FAULT" in input_variables:
        Fault = np.ones((nx, ny, nz), dtype=np.float16)
        flt = []
        for k in range(Ne):
            floatts = param["FAULT"][:, k]
            fault_temp = Get_fault(cfg.custom.FAULT_INCLUDEE)
            fault_data = read_faults(
                to_absolute_path(cfg.custom.FAULT_DATA), fault_temp
            )  # OIl
            Fault = assign_faults(fault_data, nx, ny, nz, fault_temp, floatts)
            flt.append(Fault)
        flt = np.stack(flt, axis=0)[:, None, :, :, :]
        faultz = np.stack(flt, axis=0)
        faultz = faultz / 100
    ini_ensembles = {}
    if "PERM" in input_variables:
        ini_ensembles["perm"] = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    if "PORO" in input_variables:
        ini_ensembles["poro"] = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    if "PINI" in input_variables:
        ini_ensembles["pini"] = cfg.custom.PROPS.initial_pressure * np.ones(
            (Ne, 1, nz, nx, ny), dtype=np.float32
        )  # * effea[None,None,:,:,:]
    if "SINI" in input_variables:
        ini_ensembles["sini"] = cfg.custom.PROPS.initial_water_saturation * np.ones(
            (Ne, 1, nz, nx, ny), dtype=np.float32
        )  # * effea[None,None,:,:,:]
    if "FAULT" in input_variables:
        ini_ensembles["fault"] = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    if "SGINI" in input_variables:
        ini_ensembles["sgini"] = 1e-3 * np.ones((Ne, 1, nz, nx, ny), dtype=np.float32)
    if "SOINI" in input_variables:
        ini_ensembles["soini"] = 0.8 * np.ones((Ne, 1, nz, nx, ny), dtype=np.float32)
    for kk in range(Ne):
        if "PERM" in input_variables:
            a = np.reshape(param["PERM"][:, kk], (nx, ny, nz), "F")  # * effective
        if "PORO" in input_variables:
            a1 = np.reshape(param["PORO"][:, kk], (nx, ny, nz), "F")  # * effective
        for my in range(nz):
            if "PERM" in input_variables:
                ini_ensembles["perm"][kk, 0, my, :, :] = a[:, :, my]  # Permeability
            if "PORO" in input_variables:
                ini_ensembles["poro"][kk, 0, my, :, :] = a1[:, :, my]  # Porosity
            if "FAULT" in input_variables:
                ini_ensembles["fault"][kk, 0, my, :, :] = faultz[
                    kk, 0, :, :, my
                ]  # fault
    # Initial_pressure
    if "PINI" in input_variables:
        ini_ensembles["pini"] = fit_operation(
            ini_ensembles["pini"], target_min, target_max, minP, maxP
        )
    # Permeability
    if "PERM" in input_variables:
        ini_ensembles["perm"] = fit_operation(
            ini_ensembles["perm"], target_min, target_max, minK, maxK
        )
    # Prepare the dictionary dynamically
    inn = {}
    if "PERM" in input_variables:
        inn["perm"] = torch.from_numpy(ini_ensembles["perm"]).to(
            device, dtype=torch.float32
        )
    if "PORO" in input_variables:
        inn["poro"] = torch.from_numpy(ini_ensembles["poro"]).to(
            device, dtype=torch.float32
        )
    if "PINI" in input_variables:
        inn["pini"] = torch.from_numpy(ini_ensembles["pini"]).to(
            device, dtype=torch.float32
        )
    if "SINI" in input_variables:
        inn["sini"] = torch.from_numpy(ini_ensembles["sini"]).to(
            device, dtype=torch.float32
        )
    if "FAULT" in input_variables:
        inn["fault"] = torch.from_numpy(ini_ensembles["fault"]).to(
            device, dtype=torch.float32
        )
    if "SGINI" in input_variables:
        inn["sgini"] = torch.from_numpy(ini_ensembles["sgini"]).to(
            device, dtype=torch.float32
        )

    if "SOINI" in input_variables:
        inn["soini"] = torch.from_numpy(ini_ensembles["soini"]).to(
            device, dtype=torch.float32
        )
    return inn


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


def save_files(
    perm, poro, perm2, dest_dir, oldfolder, FAULT_INCLUDE, PERMX_INCLUDE, PORO_INCLUDE
):
    """
    Write permeability, porosity, and fault multiplier data to their include files.

    Parameters
    ----------
    perm : np.ndarray
        Permeability values to write to the PERMX include file.
    poro : np.ndarray
        Porosity values to write to the PORO include file.
    perm2 : np.ndarray
        Fault multiplier values to update within FAULT_INCLUDE.
    dest_dir : str
        Destination directory to change into before writing files.
    oldfolder : str
        Original working directory to restore after writing.
    FAULT_INCLUDE : str
        Filename of the fault multiplier include file.
    PERMX_INCLUDE : str
        Filename for the permeability output include file.
    PORO_INCLUDE : str
        Filename for the porosity output include file.

    Returns
    -------
    None
    """
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
    with open(FAULT_INCLUDE) as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.strip() == "MULTFLT":
            continue
        parts = line.split(" ")
        if len(parts) > 1 and parts[1].replace(".", "", 1).replace("/", "").isdigit():
            parts[1] = str(my_array[my_array_index])
            lines[i] = " ".join(parts)
            my_array_index += 1
    with open(FAULT_INCLUDE, "w") as file:
        file.writelines(lines)
    os.chdir(oldfolder)


def convert_back(rescaled_tensor, target_min, target_max, min_val, max_val):
    """
    Reverse a max-normalisation by multiplying back by the original maximum.

    Parameters
    ----------
    rescaled_tensor : np.ndarray or torch.Tensor
        Normalised tensor produced by fit_operation.
    target_min : float
        Unused (reserved for interface consistency).
    target_max : float
        Unused (reserved for interface consistency).
    min_val : float
        Unused (reserved for interface consistency).
    max_val : float
        Original maximum value used during normalisation.

    Returns
    -------
    tensor : np.ndarray or torch.Tensor
        Tensor scaled back to original range.
    """
    return rescaled_tensor * max_val


def Run_simulator(dest_dir, oldfolder, string_simulation):
    """
    Execute a simulation command in the specified directory then restore the original path.

    Parameters
    ----------
    dest_dir : str
        Directory to change into before running the simulation.
    oldfolder : str
        Original working directory to restore after simulation.
    string_simulation : str
        Shell command string to execute (passed to os.system).

    Returns
    -------
    None
    """
    os.chdir(dest_dir)
    # Parse the command into argv tokens so we can run without a shell, avoiding
    # the shell-injection surface that ``os.system`` exposes when the command
    # string is built from configuration.
    subprocess.run(
        shlex.split(string_simulation, posix=os.name != "nt"),
        shell=False,
        check=False,
    )
    os.chdir(oldfolder)
