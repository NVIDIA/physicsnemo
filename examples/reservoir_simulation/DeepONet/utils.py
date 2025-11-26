# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
Utility functions for CO2 sequestration data processing.

This module provides denormalization functions and other utilities
specific to the CO2 sequestration dataset.
"""

import numpy as np


# ==============================================================================
# Denormalization Functions
# ==============================================================================


def dnorm_dP(dP):
    """
    Denormalize pressure change (dP) from normalized to physical units (bar).

    Args:
        dP: Normalized pressure change values

    Returns:
        Denormalized pressure change in bar
    """
    dP = dP * 18.772821433027488
    dP = dP + 4.172939172019009
    return dP


def dnorm_inj(a):
    """
    Denormalize injection rate to physical units (MT/yr).

    Args:
        a: Normalized injection rate

    Returns:
        Injection rate in MT/yr (megatons per year)
    """
    return (a * (3e6 - 3e5) + 3e5) / (1e6 / 365 * 1000 / 1.862)


def dnorm_temp(a):
    """
    Denormalize temperature to physical units (°C).

    Args:
        a: Normalized temperature (0-1)

    Returns:
        Temperature in °C (30-180°C range)
    """
    return a * (180 - 30) + 30


def dnorm_P(a):
    """
    Denormalize initial pressure to physical units (bar).

    Args:
        a: Normalized pressure (0-1)

    Returns:
        Initial pressure in bar (100-300 bar range)
    """
    return a * (300 - 100) + 100


def dnorm_lam(a):
    """
    Denormalize lambda parameter to physical units (-).

    Args:
        a: Normalized lambda (0-1)

    Returns:
        Lambda parameter (0.3-0.7 range)
    """
    return a * 0.4 + 0.3


def dnorm_Swi(a):
    """
    Denormalize initial water saturation (Swi) to physical units (-).

    Args:
        a: Normalized Swi (0-1)

    Returns:
        Initial water saturation (0.1-0.3 range)
    """
    return a * 0.2 + 0.1


# ==============================================================================
# Helper Functions
# ==============================================================================


def extract_reservoir_mask(x_plot):
    """
    Extract the reservoir mask from input data.

    The permeability map (channel 0) indicates the active reservoir region.

    Args:
        x_plot: Input array with shape (..., H, W, T, C)

    Returns:
        tuple: (mask, thickness) where mask is boolean array and thickness is int
    """
    mask = x_plot[0, :, :, 0, 0] != 0
    thickness = int(np.sum(mask[:, 0]))
    return mask, thickness


def denormalize_inputs(x_plot):
    """
    Denormalize all input parameters from a sample.

    Args:
        x_plot: Input array with shape (1, H, W, T, C)

    Returns:
        dict: Dictionary with denormalized parameters
    """
    params = {
        "injection_rate": dnorm_inj(x_plot[0, 0, 0, 0, 4]),
        "temperature": dnorm_temp(x_plot[0, 0, 0, 0, 6]),
        "initial_pressure": dnorm_P(x_plot[0, 0, 0, 0, 5]),
        "Swi": dnorm_Swi(x_plot[0, 0, 0, 0, 7]),
        "lambda": dnorm_lam(x_plot[0, 0, 0, 0, 8]),
    }
    return params
