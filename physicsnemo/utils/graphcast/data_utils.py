# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import os

import netCDF4 as nc
import torch
from torch import Tensor
from torch.nn.functional import interpolate

from .graph_utils import deg2rad


class StaticData:
    """Class to load static data from netCDF files. Static data includes land-sea mask,
    geopotential, and latitude-longitude coordinates.

    Parameters
    ----------
    static_dataset_path : str
        Path to directory containing static data.
    latitudes : Tensor
        Tensor with shape (lat,) that includes latitudes.
    longitudes : Tensor
        Tensor with shape (lon,) that includes longitudes.
    """

    def __init__(
        self,
        static_dataset_path: str,
        latitudes: Tensor,
        longitudes: Tensor,
    ) -> None:  # pragma: no cover
        self.lsm_path = os.path.join(static_dataset_path, "land_sea_mask.nc")
        self.geop_path = os.path.join(static_dataset_path, "geopotential.nc")
        self.lat = latitudes
        self.lon = longitudes

    def get_lsm(self) -> Tensor:  # pragma: no cover
        """Get land-sea mask from netCDF file.

        Returns
        -------
        Tensor
            Land-sea mask with shape (1, 1, lat, lon).
        """
        ds = torch.tensor(nc.Dataset(self.lsm_path)["lsm"], dtype=torch.float32)
        ds = torch.unsqueeze(ds, dim=0)
        ds = interpolate(ds, size=(self.lat.size(0), self.lon.size(0)), mode="bilinear")
        return ds

    def get_geop(self, normalize: bool = True) -> Tensor:  # pragma: no cover
        """Get geopotential from netCDF file.

        Parameters
        ----------
        normalize : bool, optional
            Whether to normalize the geopotential, by default True

        Returns
        -------
        Tensor
            Normalized geopotential with shape (1, 1, lat, lon).
        """
        ds = torch.tensor(nc.Dataset(self.geop_path)["z"], dtype=torch.float32)
        ds = torch.unsqueeze(ds, dim=0)
        ds = interpolate(ds, size=(self.lat.size(0), self.lon.size(0)), mode="bilinear")
        if normalize:
            ds = (ds - ds.mean()) / ds.std()
        return ds

    def get_lat_lon(self) -> Tensor:  # pragma: no cover
        """Computes cosine of latitudes and sine and cosine of longitudes.

        Returns
        -------
        Tensor
            Tensor with shape (1, 3, lat, lon) tha includes cosine of latitudes,
            sine and cosine of longitudes.
        """

        # cos latitudes
        cos_lat = torch.cos(deg2rad(self.lat))
        cos_lat = cos_lat.view(1, 1, self.lat.size(0), 1)
        cos_lat_mg = cos_lat.expand(1, 1, self.lat.size(0), self.lon.size(0))

        # sin longitudes
        sin_lon = torch.sin(deg2rad(self.lon))
        sin_lon = sin_lon.view(1, 1, 1, self.lon.size(0))
        sin_lon_mg = sin_lon.expand(1, 1, self.lat.size(0), self.lon.size(0))

        # cos longitudes
        cos_lon = torch.cos(deg2rad(self.lon))
        cos_lon = cos_lon.view(1, 1, 1, self.lon.size(0))
        cos_lon_mg = cos_lon.expand(1, 1, self.lat.size(0), self.lon.size(0))

        outvar = torch.cat((cos_lat_mg, sin_lon_mg, cos_lon_mg), dim=1)
        return outvar

    def get(self) -> Tensor:  # pragma: no cover
        """Get all static data.

        Returns
        -------
        Tensor
            Tensor with shape (1, 5, lat, lon) that includes land-sea mask,
            geopotential, cosine of latitudes, sine and cosine of longitudes.
        """
        lsm = self.get_lsm()
        geop = self.get_geop()
        lat_lon = self.get_lat_lon()
        return torch.concat((lsm, geop, lat_lon), dim=1)
