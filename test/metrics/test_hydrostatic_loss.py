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

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
import torch

from physicsnemo.metrics.climate.hydrostasy import (
    HydrostaticBalance,
)

xr = pytest.importorskip("xarray")

def test_constant_temperature(rtol: float = 1e-3, atol: float = 1e-3):
    R = 287  # J K^{-1} kg^{-1}
    g0 = 9.81  # m s^{-2}
    T = 273.15  # K
    z_pressure_levels = {0: 850.0, 1: 500.0, 2: 250.0, 3: 50.0}
    z_pressure_levels = {
        c: p * 100 for c, p in z_pressure_levels.items()
    }  # convert hPa to Pa
    anchor_z_channel = 0
    anchor_T_channel = 4

    # Create HydrostaticBalance constraint object
    constraint = HydrostaticBalance(
        z_pressure_levels, anchor_z_channel, anchor_T_channel, R, g0
    )

    x = torch.zeros((1, 12, 5, 1, 1))
    # Set Z
    z = (
        R
        * T
        / g0
        * torch.log(
            z_pressure_levels[anchor_z_channel]
            / torch.Tensor(list(z_pressure_levels.values()))
        )
    )
    x[:, :, 0:4, :, :] = z.view(1, 1, -1, 1, 1)
    # Set Tv at 850 hPa
    x[:, :, 4, :, :] = T + torch.zeros_like(x[:, :, 4, :, :])

    Tv = constraint(x)

    print(Tv[0, 0, :, 0, 0])
    assert torch.allclose(T * torch.ones_like(Tv), Tv, rtol=rtol, atol=atol,)


@pytest.mark.parametrize("N", [10, 20])
def test_constant_lapse_rate(N: int, rtol: float = 1e-3, atol: float = 1e-3):
    R = 287  # J K^{-1} kg^{-1}
    g0 = 9.81  # m s^{-2}
    T0 = 273.15  # K
    z0 = 0  # m
    Gamma = 9.8 / 1000  # K m^{-1}
    p = np.logspace(np.log10(850.0), np.log10(50.0), num=N)
    # z_pressure_levels = {0: 850., 1: 500., 2: 250., 3: 50.}
    # z_pressure_levels = {i: 850. - 50.*i for i in range(17)}
    z_pressure_levels = {i: p[i] for i in range(p.shape[0])}
    z_pressure_levels = {
        c: p * 100 for c, p in z_pressure_levels.items()
    }  # convert hPa to Pa
    anchor_z_channel = 0
    anchor_T_channel = len(z_pressure_levels)

    # Create HydrostaticBalance constraint object
    constraint = HydrostaticBalance(
        z_pressure_levels, anchor_z_channel, anchor_T_channel, R, g0
    )

    x = torch.zeros((1, 12, anchor_T_channel + 1, 1, 1))
    # Set T
    T = T0 * torch.pow(
        torch.Tensor(list(z_pressure_levels.values()))
        / z_pressure_levels[anchor_z_channel],
        Gamma * R / g0,
    )
    print("T: ", T)
    z = z0 + (T0 - T) / Gamma
    print("z: ", z)
    x[:, :, 0:anchor_T_channel, :, :] = z.view(1, 1, -1, 1, 1)
    # Set Tv at 850 hPa
    x[:, :, anchor_T_channel, :, :] = T[0] + torch.zeros_like(
        x[:, :, anchor_T_channel, :, :]
    )

    Tv = constraint(x)

    print(Tv[0, 0, :, 0, 0])
    # assert torch.allclose(T * torch.ones_like(Tv), Tv)
    return p, z, T, Tv[0, 0, constraint.z_channels, 0, 0]


@pytest.mark.parametrize("N", [10, 20])
def test_dual_lapse_rate(N: int, rtol: float = 1e-3, atol: float = 1e-3):
    R = 287  # J K^{-1} kg^{-1}
    g0 = 9.81  # m s^{-2}
    T0 = 273.15  # K
    z0 = 0  # m
    Gamma1 = 9.8 / 1000  # K m^{-1}
    Gamma2 = Gamma1 / 2.0
    zc = 10000  # m
    p = np.logspace(np.log10(850.0), np.log10(50.0), num=N)
    # z_pressure_levels = {0: 850., 1: 500., 2: 250., 3: 50.}
    # z_pressure_levels = {i: 850. - 50.*i for i in range(17)}
    z_pressure_levels = {i: p[i] for i in range(p.shape[0])}
    z_pressure_levels = {
        c: p * 100 for c, p in z_pressure_levels.items()
    }  # convert hPa to Pa
    anchor_z_channel = 0
    anchor_T_channel = len(z_pressure_levels)

    # Create HydrostaticBalance constraint object
    constraint = HydrostaticBalance(
        z_pressure_levels, anchor_z_channel, anchor_T_channel, R, g0
    )

    x = torch.zeros((1, 12, anchor_T_channel + 1, 1, 1))
    # Set T
    T1 = T0 * torch.pow(
        torch.Tensor(list(z_pressure_levels.values()))
        / z_pressure_levels[anchor_z_channel],
        Gamma1 * R / g0,
    )
    z1 = z0 + (T0 - T1) / Gamma1

    Tc = T0 - Gamma1 * (zc - z0)
    Pc = z_pressure_levels[anchor_z_channel] * (Tc / T0) ** (g0 / (Gamma1 * R))
    T2 = Tc * torch.pow(
        torch.Tensor(list(z_pressure_levels.values())) / Pc, Gamma2 * R / g0
    )
    z2 = zc + (Tc - T2) / Gamma2

    z = torch.zeros_like(z1)
    z = torch.where(z1 < zc, z1, z2)
    T = torch.where(z1 < zc, T1, T2)

    x[:, :, 0:anchor_T_channel, :, :] = z.view(1, 1, -1, 1, 1)
    # Set Tv at 850 hPa
    x[:, :, anchor_T_channel, :, :] = T[0] + torch.zeros_like(
        x[:, :, anchor_T_channel, :, :]
    )

    Tv = constraint(x)

    print(Tv[0, 0, :, 0, 0])
    # assert torch.allclose(T * torch.ones_like(Tv), Tv)
    return p, z, T, Tv[0, 0, constraint.z_channels, 0, 0]
