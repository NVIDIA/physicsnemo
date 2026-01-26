# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import functools

import config.environment as config
import earth2grid
import numpy as np
import torch
import zarr
from utils.storage import get_storage_options

from datasets import catalog


@functools.cache
def load_lfrac(hpx_level) -> torch.Tensor:
    src_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat=768, nlon=1536)
    hpx_grid = earth2grid.healpix.Grid(
        level=hpx_level, pixel_order=earth2grid.healpix.NEST
    )
    regridder = earth2grid.get_regridder(src_grid, hpx_grid)

    # get static iputs
    land_data = zarr.open_group(
        config.UFS_LAND_DATA_ZARR,
        storage_options=get_storage_options(config.UFS_LAND_DATA_PROFILE),
    )
    land_fraction = land_data["lfrac"][:]
    land_fraction = regridder(torch.from_numpy(land_fraction).to(torch.float64))
    return land_fraction


@functools.cache
def load_orography() -> np.ndarray:
    entry = catalog.ufs()
    group = entry.to_zarr()
    return group["orog"][:]
