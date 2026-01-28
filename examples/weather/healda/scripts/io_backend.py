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
"""Regridding Zarr IO Backend for HEALPix output."""

import itertools
import os
from typing import Any

import earth2grid
import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr
from earth2grid import healpix, latlon
from earth2studio.utils.type import CoordSystem
from regridding import ConservativeRegridder, add_south_pole_mean


class RegriddingZarrBackend:
    """Interface for a generic IO backend. Assume 0.25 output grid."""

    _VAR_MAP = {
        "t2m": "tas",
        "u10m": "uas",
        "v10m": "vas",
        "u100m": "100u",
        "v100m": "100v",
        "msl": "pres_msl",
    }

    def __init__(
        self,
        zarr_path,
        times,
        out_vars,
        n_ensemble,
        nsteps,
        rank,
        regrid_conservative=False,
        init_zarr_path=None,
    ):
        self.out_vars = out_vars
        self.n_ensemble = n_ensemble
        self.nsteps = nsteps
        self.regrid_conservative = regrid_conservative
        self.init_zarr_path = init_zarr_path
        if rank == 0:
            self._init_group(zarr_path, times)

        # Load original HEALPix init data for t=0 bypass
        if init_zarr_path:
            self.init_ds = xr.open_zarr(init_zarr_path)
        else:
            self.init_ds = None

        self.group = zarr.open_consolidated(zarr_path)
        self.times = pd.DatetimeIndex(self.group["time"][:].astype("datetime64[s]"))

        # Regridder expects 721 lat after add_south_pole_mean()
        ll_grid = latlon.equiangular_lat_lon_grid(
            nlat=721, nlon=1440, includes_south_pole=True
        )

        if not regrid_conservative:
            hpx_grid = healpix.Grid(
                level=6, pixel_order=earth2grid.healpix.PixelOrder.NEST
            )
            self.regridder_to_hpx = earth2grid.get_regridder(ll_grid, hpx_grid).float()
        else:
            self.regridder_to_hpx = ConservativeRegridder(
                latlon_grid=ll_grid, regrid_level=8, out_level=6
            )

    def _regrid(self, array):
        regridder = self.regridder_to_hpx.to(array.device)
        return regridder(array)

    def _update_coords(self, zarr_path, requested_times):
        nensemble = self.n_ensemble
        first_var = self.out_vars[0]
        group = zarr.open_group(zarr_path, mode="a")
        if first_var in group:
            existing_array = group[first_var]
            existing_ensemble_size = existing_array.shape[1]

            # Validate times match
            if "time" in group:
                existing_times = pd.DatetimeIndex(
                    group["time"][:].astype("datetime64[s]")
                )
                requested_times_s = pd.DatetimeIndex(
                    np.asarray(requested_times).astype("datetime64[s]")
                )
                if not existing_times.equals(requested_times_s):
                    raise ValueError(
                        f"Existing zarr has times {existing_times[0]} to {existing_times[-1]} "
                        f"({len(existing_times)} times), but requested {requested_times_s[0]} to "
                        f"{requested_times_s[-1]} ({len(requested_times_s)} times). "
                        "Please delete the zarr or use a different output path."
                    )

            if existing_ensemble_size > nensemble:
                raise ValueError(
                    f"Existing zarr has {existing_ensemble_size} ensemble members, "
                    f"but requested {nensemble}. Cannot resize down. "
                    "Please delete the zarr or use a different output path."
                )
            elif existing_ensemble_size < nensemble:
                print(
                    f"Resizing zarr from {existing_ensemble_size} to {nensemble} ensemble members"
                )
                # Resize all variable arrays along ensemble dimension
                for field in self.out_vars:
                    if field in group:
                        var_array = group[field]
                        new_shape = list(var_array.shape)
                        new_shape[1] = nensemble
                        var_array.resize(new_shape)

                # Update ensemble coordinate array
                ensemble_v = group["ensemble"]
                ensemble_array = np.arange(nensemble)
                ensemble_v.resize(ensemble_array.shape)
                ensemble_v[:] = ensemble_array

                zarr.consolidate_metadata(group.store)
                print(
                    f"Resized zarr structure at {zarr_path} (HEALPix level 6, {nensemble} ensemble members)"
                )
                return
            else:
                print(f"Zarr exists with {nensemble} ensemble members. Reusing.")
                return

    def _init_group(self, zarr_path, times):
        os.makedirs(zarr_path, exist_ok=True)
        group = zarr.open_group(zarr_path, mode="a")

        zarr_exists = len(group) > 0 and any(key in group for key in self.out_vars)
        if zarr_exists:
            return self._update_coords(zarr_path, times)

        # Create new zarr structure
        print(f"Creating zarr at {zarr_path}")
        spatial_shape = (49152,)  # HEALPix cells
        total_times = len(times)  # Use full task count for dimensions
        for field in self.out_vars:
            group.create_array(
                field,
                shape=(
                    total_times,
                    self.n_ensemble,
                    self.nsteps + 1,
                    *spatial_shape,
                ),  # time, ensemble, step, cells
                chunks=(1, 1, self.nsteps + 1, *spatial_shape),
                fill_value=float("NaN"),
                dimension_names=("time", "ensemble", "lead_time", "cells"),
                dtype="f",
            )

        # Store actual datetime values for time coordinate
        time_v = group.create_array(
            "time",
            dtype=np.int64,
            shape=(total_times,),
            chunks=(total_times,),
            dimension_names=["time"],
        )
        # global_tasks contains indices, get actual times
        # Ensure times are unique (floor to seconds and check for duplicates)
        times_s = np.asarray(times).astype("datetime64[s]")
        unique_times, counts = np.unique(times_s, return_counts=True)
        if len(unique_times) != len(times_s):
            duplicates = unique_times[counts > 1]
            raise ValueError(
                f"Duplicate times found after converting to seconds: {duplicates}"
            )
        time_v[:] = times_s.astype(np.int64)
        time_v.attrs["units"] = "seconds since 1970-01-01 00:00:00"
        time_v.attrs["calendar"] = "standard"

        ensemble_array = np.arange(self.n_ensemble)
        ensemble_v = group.create_array(
            "ensemble",
            dtype=np.int32,
            shape=ensemble_array.shape,
            chunks=ensemble_array.shape,
            dimension_names=["ensemble"],
        )
        ensemble_v[:] = ensemble_array
        ensemble_v.attrs["description"] = "ensemble member index"

        # Store forecast step information (hours from initial time)
        forecast_hours = np.arange(0, (self.nsteps + 1) * 6, 6)  # 6-hour steps
        step_v = group.create_array(
            "lead_time",
            dtype=np.int32,
            shape=forecast_hours.shape,
            chunks=forecast_hours.shape,
            dimension_names=["lead_time"],
        )
        step_v[:] = forecast_hours
        step_v.attrs["description"] = "Forecast lead time in hours"

        # Add global attributes for HEALPix grid
        group.attrs["grid_type"] = "HEALPix"
        group.attrs["healpix_level"] = 6
        group.attrs["healpix_nside"] = 64
        group.attrs["healpix_ncells"] = 49152
        group.attrs["healpix_pixel_order"] = "NEST"
        group.attrs["description"] = "ensemble forecasts in HEALPix format"

        zarr.consolidate_metadata(group.store)
        print(
            f"Created zarr structure at {zarr_path} (HEALPix level 6, {self.n_ensemble} ensemble members)"
        )

    def add_array(
        self, coords: CoordSystem, array_name: str | list[str], **kwargs: dict[str, Any]
    ) -> None:
        """
        Add an array with `array_name` to the existing IO backend object.

        Parameters
        ----------
        coords : OrderedDict
            Ordered dictionary of representing the dimensions and coordinate data
            of x.
        array_name : str
            Name of the arrays that will be initialized with coordinates as dimensions.
        kwargs : dict[str, Any], optional
            Optional keyword arguments that will be passed to the IO backend constructor.
        """
        return

    def flush(self):
        return

    def write(
        self,
        x: torch.Tensor | list[torch.Tensor],
        coords: CoordSystem,
        array_name: str | list[str],
    ) -> None:
        """
        Write data to the current backend using the passed array_name.

        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            Tensor(s) to be written to zarr store.
        coords : OrderedDict
            Coordinates of the passed data.
        array_name : str | list[str]
            Name(s) of the array(s) that will be written to.
        """
        coords_time = pd.DatetimeIndex(coords["time"]).floor("s")
        time_idx = self.times.get_indexer(coords_time)
        if np.any(time_idx == -1):
            raise ValueError(
                f"Time mismatch: {coords_time[time_idx == -1]} not in {self.times}"
            )

        time_idx = np.atleast_1d(time_idx)
        step = coords["lead_time"] // np.timedelta64(6, "h")
        step = np.atleast_1d(step)
        if "ensemble" in coords:
            ensemble_idx = coords["ensemble"]
        else:
            # handle aurora case where no ensemble dim and coords has [t0-6, t0]
            ensemble_idx = np.array([0])
            step = step[-1:]
        ensemble_idx = np.atleast_1d(ensemble_idx)

        # Ensure all models output 721 lat before regridding
        x_processed = [add_south_pole_mean(array) for array in x]
        regridded = {
            name: self._regrid(array) for name, array in zip(array_name, x_processed)
        }

        for var_name in array_name:
            nt = len(time_idx)
            ne = len(ensemble_idx)
            nstep = len(step)
            array = regridded[var_name].cpu().numpy()

            # Handle deterministic models (no ensemble dimension)
            if array.ndim == 3:  # Missing ensemble dimension
                array = array[:, np.newaxis, :, :]  # Add ensemble dimension

            # Map e2studio var name to init zarr var name
            init_var = None
            if self.init_ds is not None:
                if var_name in self.init_ds:
                    init_var = var_name
                elif var_name.upper() in self.init_ds:
                    init_var = var_name.upper()
                elif self._VAR_MAP.get(var_name) in self.init_ds:
                    init_var = self._VAR_MAP[var_name]
            can_bypass_t0 = init_var is not None

            for i, j, k in itertools.product(range(nt), range(ne), range(nstep)):
                if step[k] == 0 and can_bypass_t0:
                    # Write original analysis data directly without regridding effects
                    orig = self.init_ds[init_var].sel(time=coords_time[i]).values
                    orig_tensor = torch.as_tensor(orig)
                    orig_nest = earth2grid.healpix.reorder(
                        orig_tensor,
                        earth2grid.healpix.HEALPIX_PAD_XY,
                        earth2grid.healpix.PixelOrder.NEST,
                    )
                    self.group[var_name][time_idx[i], ensemble_idx[j], 0] = (
                        orig_nest.numpy()
                    )
                else:
                    # Write regridded data
                    self.group[var_name][time_idx[i], ensemble_idx[j], step[k]] = array[
                        i, j, k
                    ]
