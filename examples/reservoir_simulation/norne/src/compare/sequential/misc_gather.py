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
                    SEQUENTIAL DATA GATHERING UTILITIES
=====================================================================

This module provides data gathering utilities for sequential FVM
surrogate model comparisons. It includes functions for data processing,
file handling, and analysis.

Key Features:
- Data type definitions for simulation data
- File parsing and data extraction
- Data processing and conversion utilities
- Binary file handling

Usage:
    from compare.sequential.misc_gather import (
        read_compdats,
        process_data2,
        convert_to_list,
        extract_tuples
    )

@Author : Clement Etienam
"""

from collections import OrderedDict
import numpy as np
from utils.logging_utils import setup_logging
from utils.array_utils import find_first_numeric_row
from utils.ecl_binary import EclBinaryParser


def process_data2(data):
    """
    Build a mapping from well names to lists of (i, j) grid indices.

    Parameters
    ----------
    data : list of tuple
        Each tuple contains (well_name, i_str, j_str, ...) from COMPDAT entries.

    Returns
    -------
    well_indices : dict
        Keys are well name strings; values are lists of (i_index, j_index) tuples (0-based).
    """
    well_indices = {}
    for entry in data:
        well_name = entry[0]
        if well_name not in well_indices:
            well_indices[well_name] = []
        i_index = int(entry[1]) - 1  # Convert to zero-based index
        j_index = int(entry[2]) - 1  # Convert to zero-based index
        well_indices[well_name].append((i_index, j_index))
    return well_indices


# find_first_numeric_row is imported from utils.array_utils above.


def convert_to_list(well_data):
    """
    Flatten a well-indices dict into a list of (i, j, well_name) tuples.

    Parameters
    ----------
    well_data : dict
        Mapping of well name to list of (i, j) index pairs.

    Returns
    -------
    output_list : list of tuple
        Each element is (i_index, j_index, well_name).
    """
    output_list = []
    for well_name, indices in well_data.items():
        for i, j in indices:
            output_list.append((i, j, well_name))
    return output_list


def extract_tuples(set1, set2, set3, tuples_list):
    """
    Partition a list of (i, j, name) tuples into three categorised groups.

    Parameters
    ----------
    set1 : set
        Well names for the first group.
    set2 : set
        Well names for the second group.
    set3 : set
        Well names for the third group (remaining after removing set1 and set2 names).
    tuples_list : list of tuple
        Each tuple is (i_index, j_index, well_name).

    Returns
    -------
    extracted_set1 : list of tuple
        Tuples whose well name is in set1, sorted by name.
    extracted_set2 : list of tuple
        Tuples whose well name is in set2, sorted by name.
    final_remaining_list : list of tuple
        Tuples from set3 not also in set1 or set2, sorted by name.
    """
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
    """
    Parse a reservoir DATA file and extract COMPDAT entries for specified wells.

    Parameters
    ----------
    filename : str
        Path to the Eclipse-style DATA file containing COMPDAT keyword.
    well_names : list of str
        Names of wells whose COMPDAT rows should be returned.

    Returns
    -------
    data : list of tuple
        Each tuple is (well_name, i, j, k1, k2) for matching COMPDAT lines.
    """
    with open(filename) as file:
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
    """
    Extract numeric data and time arrays for a given vector and producer wells.

    Parameters
    ----------
    name : str
        Name of the ECL vector (e.g. 'WOPR') to look up in vectors.
    producer_well_names : list of str
        Well names used to filter relevant columns in the DataFrame.
    vectors : dict
        Mapping of vector name to pd.DataFrame returned by EclBinaryParser.read_vectors.

    Returns
    -------
    result_array : np.ndarray or None
        Numeric rows for the filtered columns, or None if no numeric rows exist.
    time_array : np.ndarray or None
        Numeric rows from the TIME vector, or None if no numeric rows exist.
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
        logger = setup_logging(__name__)
        logger.info(f"Numeric data from {name} processed successfully.")
    else:
        logger = setup_logging(__name__)
        logger.info(f"No numeric rows found in the DataFrame for {name}.")
        result_array = None
    Time = vectors["TIME"]
    start_row = find_first_numeric_row(Time)
    if start_row is not None:
        numeric_df = Time.iloc[start_row:]
        time_array = numeric_df.to_numpy()
        logger = setup_logging(__name__)
        logger.info(f"Numeric data from {name} processed successfully.")
    else:
        logger = setup_logging(__name__)
        logger.info(f"No numeric rows found in the DataFrame for {name}.")
        time_array = None
    return result_array, time_array


def extract_qs(steppi, steppi_indices, filenameui, injectors, gas_injectors, filename):
    """
    Read gas and water injection rates from an UNSMRY binary summary file.

    Parameters
    ----------
    steppi : int
        Total number of time steps in the simulation.
    steppi_indices : np.ndarray
        1-based indices of the time steps to extract.
    filenameui : str
        Base path (without extension) to the UNSMRY/SMSPEC files.
    injectors : list of tuple
        Water injector entries; each tuple's last element is the well name.
    gas_injectors : list of tuple
        Gas injector entries; each tuple's last element is the well name.
    filename : str
        Path to the DATA file (unused directly but forwarded to helpers).

    Returns
    -------
    outg : np.ndarray or None
        Gas injection rates array of shape (len(steppi_indices), n_gas_wells).
    outw : np.ndarray or None
        Water injection rates array of shape (len(steppi_indices), n_water_wells).
    """
    well_namesg = [entry[-1] for entry in gas_injectors]  # gas injectors well names
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
    """
    Compute per-timestep mean property values at each well's grid locations.

    Parameters
    ----------
    steppi : int
        Number of time steps to iterate over.
    well_indices : dict
        Mapping of well identifier to list of (i, j, k, l) index tuples.
    swatuse : np.ndarray
        4-D array of shape (steppi, nx, ny, nz) containing the property values.

    Returns
    -------
    outt2 : np.ndarray
        Array of shape (steppi, n_wells) with mean values per timestep per well.
    """
    mean_big_all = []
    for xx in range(steppi):
        mean_big = []  # Collects mean values for this particular timestep
        for list1 in well_indices.values():  # Direct access to lists via .items()
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
    """
    Distribute injection and production rates into 4-D grid arrays for all wells.

    Parameters
    ----------
    steppi : int
        Number of simulation time steps.
    well_indices : list of tuple
        Water injector entries with (well_name, i, j, k, l) structure.
    well_indicesg : list of tuple
        Gas injector entries with (well_name, i, j, k, l) structure.
    well_indiceso : list of tuple
        Oil producer entries with (well_name, i, j, k, l) structure.
    swatuse : np.ndarray
        4-D array (steppi, nx, ny, nz) to be filled with water injection rates.
    gasuse : np.ndarray
        4-D array (steppi, nx, ny, nz) to be filled with gas injection rates.
    oiluse : np.ndarray
        4-D array (steppi, nx, ny, nz) to be filled with oil production flags.
    Q : np.ndarray
        2-D array (steppi, n_water_wells) of water injection rates per well.
    Qg : np.ndarray
        2-D array (steppi, n_gas_wells) of gas injection rates per well.

    Returns
    -------
    swatuse : np.ndarray
        Updated water injection rate grid, shape (steppi, nx, ny, nz).
    gasuse : np.ndarray
        Updated gas injection rate grid, shape (steppi, nx, ny, nz).
    oiluse : np.ndarray
        Updated oil production flag grid, shape (steppi, nx, ny, nz).
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
