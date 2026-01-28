# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regridding utilities for lat-lon to HEALPix conversion."""

import earth2grid
import torch
from earth2grid import healpix


def add_south_pole_mean(x: torch.Tensor) -> torch.Tensor:
    """Add south pole using zonal mean of southernmost latitude.

    Aurora outputs 720 lat (patch_size=4 constraint) → adds 721st via zonal mean.
    Other models output 721 lat → pass-through unchanged.
    """
    if x.shape[-2] == 721:
        return x

    pole_values = x[..., -1:, :].mean(dim=-1, keepdim=True)
    pole_values = pole_values.expand(*x.shape[:-2], 1, x.shape[-1])

    return torch.cat([x, pole_values], dim=-2)


def get_latlon_bilinear_regridder(latlon_grid, regrid_level, dtype):
    """Create bilinear regridder from lat-lon grid to HEALPix grid."""
    hpx_grid = healpix.Grid(level=regrid_level, pixel_order=healpix.PixelOrder.NEST)
    return earth2grid.get_regridder(latlon_grid, hpx_grid).to(dtype)


class ConservativeRegridder(torch.nn.Module):
    """
    Bilinear regridder to high-res HPX then block average. More conservative than direct bilinear.
    Matches ERA5 HPX64 processing.
    """

    def __init__(
        self, latlon_grid=None, regrid_level=8, out_level=6, dtype=torch.float32
    ):
        super().__init__()

        if latlon_grid is None:
            latlon_grid = earth2grid.latlon.equiangular_lat_lon_grid(
                nlat=721, nlon=1440
            )
        self.regridder = get_latlon_bilinear_regridder(latlon_grid, regrid_level, dtype)
        self.coarsen_factor = 4 ** (regrid_level - out_level)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.regridder(x)
        return x.reshape(x.shape[:-1] + (-1, self.coarsen_factor)).mean(-1)
