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
"""Static geospatial data loaders (orography, land fraction).

These functions load time-invariant fields used as conditioning inputs.
Data paths are read from environment variables set in ``.env``.
"""

import functools
import os

import numpy as np
import torch

from physicsnemo.core.version_check import OptionalImport

earth2grid = OptionalImport("earth2grid")
zarr = OptionalImport("zarr")


@functools.cache
def load_lfrac(hpx_level: int) -> torch.Tensor:
    """Load land fraction data regridded to HEALPix NEST ordering.

    Reads from the zarr path specified by ``UFS_LAND_DATA_ZARR``.

    Args:
        hpx_level: HEALPix resolution level.
    """
    src_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat=768, nlon=1536)
    hpx_grid = earth2grid.healpix.Grid(
        level=hpx_level, pixel_order=earth2grid.healpix.NEST
    )
    regridder = earth2grid.get_regridder(src_grid, hpx_grid)

    land_data_path = os.environ["UFS_LAND_DATA_ZARR"]
    land_data = zarr.open_group(land_data_path)
    land_fraction = land_data["lfrac"][:]
    land_fraction = regridder(torch.from_numpy(land_fraction).to(torch.float64))
    return land_fraction


@functools.cache
def load_orography() -> np.ndarray:
    """Load orography (surface elevation) on HEALPix level-6 NEST grid.

    Reads from the zarr path specified by ``UFS_HPX6_ZARR``.
    """
    ufs_zarr_path = os.environ["UFS_HPX6_ZARR"]
    group = zarr.open_group(ufs_zarr_path)
    return group["orog"][:]
