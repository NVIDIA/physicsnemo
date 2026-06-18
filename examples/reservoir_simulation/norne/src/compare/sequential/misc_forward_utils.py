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
                    SEQUENTIAL FORWARD UTILITIES - CORE FUNCTIONS
=====================================================================

This module provides core forward utilities for sequential FVM
surrogate model comparisons. It includes functions for data processing,
model operations, and analysis.

Key Features:
- Data type definitions for simulation data
- Forward processing and model operations
- Data transformation and analysis
- Machine learning utilities

Usage:
    from compare.sequential.misc_forward_utils import (
        setup_simulation_parameters,
        validate_input_data,
        process_ensemble_results
    )

@Author : Clement Etienam
"""

import numpy as np
from hydra.utils import to_absolute_path
from utils.logging_utils import setup_logging
from utils.array_utils import find_first_numeric_row
from utils.ecl_binary import EclBinaryParser
from utils.path_utils import pushd

logger = setup_logging("inference")


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
    """Assemble FFNN inputs/targets from summary vectors and grid tensors.

    Parameters mirror the batch variant. Returns `(innn, ouut)` arrays for
    sequential pipelines.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Input and output arrays for FFNN training.
    """
    well_indices = process_data(unique_entries)
    ouut = np.zeros((N, steppi, lenwels * N_pr))
    innn = np.zeros((N, steppi, (4 * N_pr) + 2))
    producer_well_names = [well[-1] for well in producer_wells]
    for i in range(N):
        folder = to_absolute_path("../RUNS/Realisation" + str(i))
        with pushd(folder):
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
                    logger = setup_logging(__name__)
                    logger.info(f"Numeric data from {namey} processed successfully.")
                else:
                    logger = setup_logging(__name__)
                    logger.info(f"No numeric rows found in the DataFrame for {namey}.")
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
            for indices_list in well_indices.values():
                values = [
                    permuse[i_idx, j_idx, k_idx]
                    if k_idx == l_idx
                    else permuse[i_idx, j_idx, k_idx : l_idx + 1]
                    for i_idx, j_idx, k_idx, l_idx in indices_list
                ]
                mean_big.append(np.mean(values))
            permxx = np.tile(mean_big, (steppi, 1))
            a3 = get_dyna(steppi, well_indices, water_use[steppi_indices - 1])
            a2 = get_dyna(steppi, well_indices, gas_use[steppi_indices - 1])
            a5 = get_dyna(steppi, well_indices, oil_use[steppi_indices - 1])
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
    return innn, ouut


# convert_back is imported from utils.array_utils above.


def process_data(data):
    """Parse well location entries into a dict of zero-indexed grid index tuples.

    Parameters
    ----------
    data : list of tuple
        Each entry is ``(well_name, i, j, k_start, k_end)`` with 1-based indices.

    Returns
    -------
    dict
        Mapping from well name (str) to list of ``(i, j, k_start, k_end)`` tuples.
    """
    well_indices = {}
    for entry in data:
        if entry[0] not in well_indices:
            well_indices[entry[0]] = []
        well_indices[entry[0]].append(
            (int(entry[1]) - 1, int(entry[2]) - 1, int(entry[3]) - 1, int(entry[4]) - 1)
        )
    return well_indices


# find_first_numeric_row is imported from utils.array_utils above.


def get_dyna(steppi, well_indices, swatuse):
    """Compute per-well mean property values from a 4-D field over all timesteps.

    Parameters
    ----------
    steppi : int
        Number of simulation time steps.
    well_indices : dict
        Mapping from well name to list of ``(i, j, k_start, k_end)`` grid tuples.
    swatuse : np.ndarray
        4-D field array of shape ``(steppi, nz, nx, ny)`` to sample.

    Returns
    -------
    np.ndarray
        2-D array of mean well values, shape ``(steppi, n_wells)``.
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
    """Assemble FFNN inputs/targets from a fixed simulation folder and grid tensors.

    Parameters
    ----------
    folder : str
        Directory containing the Eclipse output files to read.
    oldfolder : str
        Original working directory to restore after processing.
    N : int
        Number of ensemble members (reads the same folder N times).
    pressure : np.ndarray
        5-D pressure field, shape ``(N, steppi, nz, nx, ny)``.
    Sgas : np.ndarray
        5-D gas saturation field, shape ``(N, steppi, nz, nx, ny)``.
    Swater : np.ndarray
        5-D water saturation field, shape ``(N, steppi, nz, nx, ny)``.
    Soil : np.ndarray
        5-D oil saturation field, shape ``(N, steppi, nz, nx, ny)``.
    perm : np.ndarray
        5-D permeability field, shape ``(N, 1, nz, nx, ny)``.
    Time : np.ndarray
        5-D time field, shape ``(N, steppi, nz, nx, ny)``.
    steppi : int
        Number of time steps to include in output arrays.
    steppi_indices : np.ndarray
        1-D array of 1-based timestep indices to select from simulation output.
    N_pr : int
        Number of producer wells.
    producer_wells : list
        List of producer well descriptors; last element of each entry is the name.
    unique_entries : list of tuple
        Well location entries ``(name, i, j, k_start, k_end)`` with 1-based indices.
    filenameui : str
        Filename of the Eclipse UNSMRY/SMSPEC file pair to parse.
    well_measurements : list of str
        Measurement types to extract (e.g. ``['WOPR', 'WWPR', 'WGPR']``).
    lenwels : int
        Number of well-measurement types.

    Returns
    -------
    innn : np.ndarray
        Input feature array, shape ``(N, steppi, (4 * N_pr) + 2)``.
    ouut : np.ndarray
        Target output array, shape ``(N, steppi, lenwels * N_pr)``.
    """
    well_indices = process_data(unique_entries)
    ouut = np.zeros((N, steppi, lenwels * N_pr))
    innn = np.zeros((N, steppi, (4 * N_pr) + 2))
    producer_well_names = [well[-1] for well in producer_wells]
    for i in range(N):
        with pushd(folder):
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
                # Extract numeric data and convert to numpy array
                start_row = find_first_numeric_row(filtered_df)
                if start_row is not None:
                    numeric_df = filtered_df.iloc[start_row:]
                    result_array = numeric_df.to_numpy()
                    logger = setup_logging(__name__)
                    logger.info(f"Numeric data from {namey} processed successfully.")
                else:
                    logger = setup_logging(__name__)
                    logger.info(f"No numeric rows found in the DataFrame for {namey}.")
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
            for indices_list in well_indices.values():
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
    return innn, ouut
