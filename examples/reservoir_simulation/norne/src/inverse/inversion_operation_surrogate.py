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
                    INVERSE SURROGATE OPERATIONS MODULE
=====================================================================

This module provides surrogate model operations for inverse problems
in reservoir simulation. It includes neural network models, ensemble
operations, and data processing utilities.

Key Features:
- Forward model ensemble operations
- Neural network surrogate models
- Data processing and conversion utilities
- Ensemble Kalman filter operations

@Author : Clement Etienam
"""

import numpy as np
import numpy.matlib
import pandas as pd
from utils.logging_utils import setup_logging

logger = setup_logging("inverse problem")

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


def historydatano(timestep, steppi, steppi_indices, N_pr):
    """Load Norne field production history from RSM CSV files for history matching.

    Parameters
    ----------
    timestep : int
        Number of time steps in the source data (used to slice raw arrays).
    steppi : int
        Number of output time steps to include.
    steppi_indices : np.ndarray
        Indices selecting which rows from the 246-step source arrays to retain.
    N_pr : int
        Number of producer wells (expected to be 22 for the Norne model).

    Returns
    -------
    DATA : dict
        Dictionary with keys ``'OIL'``, ``'WATER'``, ``'GAS'`` mapping to
        ``np.ndarray`` of shape ``(steppi, N_pr)``.
    DATA2 : np.ndarray
        Column-stacked rates of shape ``(steppi*N_pr*3, 1)``.
    new : np.ndarray
        Horizontally stacked ``[WOIL1, WWATER1, WGAS1]`` of shape ``(steppi, 3*N_pr)``.
    """
    WOIL1 = np.zeros((steppi, N_pr))
    WWATER1 = np.zeros((steppi, N_pr))
    WGAS1 = np.zeros((steppi, N_pr))
    steppii = 246
    A2oilsim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=1545, sep=r"\s+", header=None
    )

    B_1BHoilsim = A2oilsim[5].values[:steppii]
    B_1Hoilsim = A2oilsim[6].values[:steppii]
    B_2Hoilsim = A2oilsim[7].values[:steppii]
    B_3Hoilsim = A2oilsim[8].values[:steppii]
    B_4BHoilsim = A2oilsim[9].values[:steppii]
    A22oilsim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=1801, sep=r"\s+", header=None
    )
    B_4DHoilsim = A22oilsim[1].values[:steppii]
    B_4Hoilsim = A22oilsim[2].values[:steppii]
    D_1CHoilsim = A22oilsim[3].values[:steppii]
    D_1Hoilsim = A22oilsim[4].values[:steppii]
    D_2Hoilsim = A22oilsim[5].values[:steppii]
    D_3AHoilsim = A22oilsim[6].values[:steppii]
    D_3BHoilsim = A22oilsim[7].values[:steppii]
    D_4AHoilsim = A22oilsim[8].values[:steppii]
    D_4Hoilsim = A22oilsim[9].values[:steppii]
    A222oilsim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=2057, sep=r"\s+", header=None
    )

    E_1Hoilsim = A222oilsim[1].values[:steppii]
    E_2AHoilsim = A222oilsim[2].values[:steppii]
    E_2Hoilsim = A222oilsim[3].values[:steppii]
    E_3AHoilsim = A222oilsim[4].values[:steppii]
    E_3CHoilsim = A222oilsim[5].values[:steppii]
    E_3Hoilsim = A222oilsim[6].values[:steppii]
    E_4AHoilsim = A222oilsim[7].values[:steppii]
    K_3Hoilsim = A222oilsim[8].values[:steppii]

    WOIL1[:, 0] = B_1BHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 1] = B_1Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 2] = B_2Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 3] = B_3Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 4] = B_4BHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 5] = B_4DHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 6] = B_4Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 7] = D_1CHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 8] = D_1Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 9] = D_2Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 10] = D_3AHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 11] = D_3BHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 12] = D_4AHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 13] = D_4Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 14] = E_1Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 15] = E_2AHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 16] = E_2Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 17] = E_3AHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 18] = E_3CHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 19] = E_3Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 20] = E_4AHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 21] = K_3Hoilsim.ravel()[steppi_indices - 1]
    # IMPORT FOR WATER
    A2watersim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=2313, sep=r"\s+", header=None
    )
    B_1BHwatersim = A2watersim[9].values[:steppii]

    A22watersim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=2569, sep=r"\s+", header=None
    )
    B_1Hwatersim = A22watersim[1].values[:steppii]
    B_2Hwatersim = A22watersim[2].values[:steppii]
    B_3Hwatersim = A22watersim[3].values[:steppii]
    B_4BHwatersim = A22watersim[4].values[:steppii]
    B_4DHwatersim = A22watersim[5].values[:steppii]
    B_4Hwatersim = A22watersim[6].values[:steppii]
    D_1CHwatersim = A22watersim[7].values[:steppii]
    D_1Hwatersim = A22watersim[8].values[:steppii]
    D_2Hwatersim = A22watersim[9].values[:steppii]

    A222watersim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=2825, sep=r"\s+", header=None
    )
    D_3AHwatersim = A222watersim[1].values[:steppii]
    D_3BHwatersim = A222watersim[2].values[:steppii]
    D_4AHwatersim = A222watersim[3].values[:steppii]
    D_4Hwatersim = A222watersim[4].values[:steppii]
    E_1Hwatersim = A222watersim[5].values[:steppii]
    E_2AHwatersim = A222watersim[6].values[:steppii]
    E_2Hwatersim = A222watersim[7].values[:steppii]
    E_3AHwatersim = A222watersim[8].values[:steppii]
    E_3CHwatersim = A222watersim[9].values[:steppii]

    A222watersim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=3081, sep=r"\s+", header=None
    )
    E_3Hwatersim = A222watersim[1].values[:steppii]
    E_4AHwatersim = A222watersim[2].values[:steppii]
    K_3Hwatersim = A222watersim[3].values[:steppii]

    WWATER1[:, 0] = B_1BHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 1] = B_1Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 2] = B_2Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 3] = B_3Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 4] = B_4BHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 5] = B_4DHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 6] = B_4Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 7] = D_1CHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 8] = D_1Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 9] = D_2Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 10] = D_3AHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 11] = D_3BHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 12] = D_4AHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 13] = D_4Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 14] = E_1Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 15] = E_2AHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 16] = E_2Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 17] = E_3AHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 18] = E_3CHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 19] = E_3Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 20] = E_4AHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 21] = K_3Hwatersim.ravel()[steppi_indices - 1]

    # GAS PRODUCTION RATE
    A2gassim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=1033, sep=r"\s+", header=None
    )
    B_1BHgassim = A2gassim[1].values[:steppii]
    B_1Hgassim = A2gassim[2].values[:steppii]
    B_2Hgassim = A2gassim[3].values[:steppii]
    B_3Hgassim = A2gassim[4].values[:steppii]
    B_4BHgassim = A2gassim[5].values[:steppii]
    B_4DHgassim = A2gassim[6].values[:steppii]
    B_4Hgassim = A2gassim[7].values[:steppii]
    D_1CHgassim = A2gassim[8].values[:steppii]
    D_1Hgassim = A2gassim[9].values[:steppii]

    A22gassim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=1289, sep=r"\s+", header=None
    )
    D_2Hgassim = A22gassim[1].values[:steppii]
    D_3AHgassim = A22gassim[2].values[:steppii]
    D_3BHgassim = A22gassim[3].values[:steppii]
    D_4AHgassim = A22gassim[4].values[:steppii]
    D_4Hgassim = A22gassim[5].values[:steppii]
    E_1Hgassim = A22gassim[6].values[:steppii]
    E_2AHgassim = A22gassim[7].values[:steppii]
    E_2Hgassim = A22gassim[8].values[:steppii]
    E_3AHgassim = A22gassim[9].values[:steppii]

    A222gassim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=1545, sep=r"\s+", header=None
    )
    E_3CHgassim = A222gassim[1].values[:steppii]
    E_3Hgassim = A222gassim[2].values[:steppii]
    E_4AHgassim = A222gassim[3].values[:steppii]
    K_3Hgassim = A222gassim[4].values[:steppii]

    WGAS1[:, 0] = B_1BHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 1] = B_1Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 2] = B_2Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 3] = B_3Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 4] = B_4BHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 5] = B_4DHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 6] = B_4Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 7] = D_1CHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 8] = D_1Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 9] = D_2Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 10] = D_3AHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 11] = D_3BHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 12] = D_4AHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 13] = D_4Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 14] = E_1Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 15] = E_2AHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 16] = E_2Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 17] = E_3AHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 18] = E_3CHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 19] = E_3Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 20] = E_4AHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 21] = K_3Hgassim.ravel()[steppi_indices - 1]

    DATA = {"OIL": WOIL1, "WATER": WWATER1, "GAS": WGAS1}

    oil = np.reshape(WOIL1, (-1, 1), "F")
    water = np.reshape(WWATER1, (-1, 1), "F")
    gas = np.reshape(WGAS1, (-1, 1), "F")

    # Get data for history matching
    DATA2 = np.vstack([oil, water, gas])
    new = np.hstack([WOIL1, WWATER1, WGAS1])
    return DATA, DATA2, new


def historydata2(timestep, steppi, steppi_indices):
    """Load full Norne field history (oil, water, gas, water/gas injection) from RSM file.

    Parameters
    ----------
    timestep : int
        Array of time indices used to slice raw RSM data (referred to as ``indices``).
    steppi : int
        Number of output time steps to include.
    steppi_indices : np.ndarray
        Secondary indices applied after the first slice to select final time steps.

    Returns
    -------
    DATA : dict
        Dictionary with keys ``'OIL'``, ``'WATER'``, ``'GAS'``, ``'WATER_INJ'``,
        ``'WGAS_inj'`` each mapping to an ``np.ndarray`` of shape
        ``(steppi, n_wells)`` for the respective well type.
    DATA2 : np.ndarray
        Column-stacked oil, water, and gas rates of shape ``(steppi*22*3, 1)``.
    new : np.ndarray
        Horizontally stacked ``[WOIL1, WWATER1, WGAS1]`` of shape ``(steppi, 66)``.
    """
    WOIL1 = np.zeros((steppi, 22))
    WWATER1 = np.zeros((steppi, 22))
    WGAS1 = np.zeros((steppi, 22))
    WWINJ1 = np.zeros((steppi, 9))
    WGASJ1 = np.zeros((steppi, 4))
    indices = timestep
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM") as f:
        for i, line in enumerate(f):
            if (
                i < _RSM_OIL_SEC1
            ):  # Skip header lines; this RSM section starts at line 47873
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
    with open("../simulator_data/NORNE_ATW2013.RSM") as f:
        for i, line in enumerate(f):
            if (
                i < _RSM_OIL_SEC2
            ):  # Skip header lines; this RSM section starts at line 48743
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

    # Open the file and read lines until '---' is found
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM") as f:
        for i, line in enumerate(f):
            if (
                i < _RSM_OIL_SEC3
            ):  # Skip header lines; this RSM section starts at line 49613
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
    with open("../simulator_data/NORNE_ATW2013.RSM") as f:
        for i, line in enumerate(f):
            if (
                i < _RSM_OIL_SEC4
            ):  # Skip header lines; this RSM section starts at line 50483
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
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM") as f:
        for i, line in enumerate(f):
            if (
                i < _RSM_WATER_SEC1
            ):  # Skip header lines; this RSM section starts at line 40913
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
    with open("../simulator_data/NORNE_ATW2013.RSM") as f:
        for i, line in enumerate(f):
            if (
                i < _RSM_WATER_SEC2
            ):  # Skip header lines; this RSM section starts at line 41783
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
    with open("../simulator_data/NORNE_ATW2013.RSM") as f:
        for i, line in enumerate(f):
            if (
                i < _RSM_WATER_SEC3
            ):  # Skip header lines; this RSM section starts at line 42653
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
    with open("../simulator_data/NORNE_ATW2013.RSM") as f:
        for i, line in enumerate(f):
            if (
                i < _RSM_WATER_SEC4
            ):  # Skip header lines; this RSM section starts at line 43523
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
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM") as f:
        for i, line in enumerate(f):
            if (
                i < _RSM_GAS_SEC1
            ):  # Skip header lines; this RSM section starts at line 54833
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
    with open("../simulator_data/NORNE_ATW2013.RSM") as f:
        for i, line in enumerate(f):
            if (
                i < _RSM_GAS_SEC2
            ):  # Skip header lines; this RSM section starts at line 55703
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
    with open("../simulator_data/NORNE_ATW2013.RSM") as f:
        for i, line in enumerate(f):
            if (
                i < _RSM_GAS_SEC3
            ):  # Skip header lines; this RSM section starts at line 56573
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
    with open("../simulator_data/NORNE_ATW2013.RSM") as f:
        for i, line in enumerate(f):
            if (
                i < _RSM_GAS_SEC4
            ):  # Skip header lines; this RSM section starts at line 57443
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
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM") as f:
        for i, line in enumerate(f):
            if (
                i < _RSM_WINJ_SEC1
            ):  # Skip header lines; this RSM section starts at line 72237
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
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM") as f:
        for i, line in enumerate(f):
            if (
                i < _RSM_GINJ_SEC1
            ):  # Skip header lines; this RSM section starts at line 73977
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
    DATA2 = np.vstack([oil, water, gas])
    new = np.hstack([WOIL1, WWATER1, WGAS1])
    return DATA, DATA2, new
