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
"""
Simplified forecast script for running weather model inference.

Supports: FCN3, Aurora, FengWu, Pangu, and Mock/Persistence models.
Output: HEALPix level 6 zarr format.

Usage:
    python -m forecast.forecast --init_path /path/to/init.zarr --out_dir /path/to/output

Install notes:
    FCN3: pip install "makani @ git+https://github.com/NVIDIA/modulus-makani.git"
    Aurora: pip install "microsoft-aurora @ git+https://github.com/NickGeneva/aurora.git"
    Pangu: pip install earth2studio[pangu]
    FengWu: pip install earth2studio[onnx]
"""

import argparse
import gc
import logging
import os
from collections import OrderedDict
from typing import Any, Literal

import earth2grid
import numpy as np
import pandas as pd
import torch
import xarray as xr
from earth2grid import healpix, latlon
from earth2studio.models.px import Persistence
from earth2studio.models.px.aurora import VARIABLES as AURORA_VARIABLES
from earth2studio.models.px.aurora import Aurora
from earth2studio.models.px.fcn3 import FCN3
from earth2studio.models.px.fcn3 import VARIABLES as FCN3_VARIABLES
from earth2studio.models.px.pangu import VARIABLES as PANGU_VARIABLES
from earth2studio.models.px.pangu import Pangu6
from earth2studio.perturbation import Zero
from earth2studio.px.fengwu import VARIABLES as FENGWU_VARIABLES
from earth2studio.px.fengwu import FengWu
from earth2studio.run import deterministic as e2s_deterministic
from earth2studio.run import ensemble as e2s_ensemble

# from fengwu_model import FengWu, VARIABLES as FENGWU_VARIABLES
from io_backend import RegriddingZarrBackend

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------


class ModelConfig:
    """Configuration for different prognostic models."""

    def __init__(
        self,
        name: str,
        variables: list[str],
        workflow: Literal["ensemble", "deterministic"],
        model: torch.nn.Module,
        nlat: int = 721,
        nlon: int = 1440,
    ):
        self.name = name
        self.variables = variables
        self.workflow = workflow
        self.model = model
        self.nlat = nlat
        self.nlon = nlon

    @classmethod
    def fcn3(cls) -> "ModelConfig":
        return cls(
            name="fcn3",
            variables=FCN3_VARIABLES,
            workflow="ensemble",
            model=FCN3.load_model(FCN3.load_default_package()),
        )

    @classmethod
    def aurora(cls) -> "ModelConfig":
        return cls(
            name="aurora",
            variables=AURORA_VARIABLES,
            workflow="deterministic",
            model=Aurora.load_model(Aurora.load_default_package()),
            nlat=720,
            nlon=1440,
        )

    @classmethod
    def pangu(cls) -> "ModelConfig":
        return cls(
            name="pangu",
            variables=PANGU_VARIABLES,
            workflow="deterministic",
            model=Pangu6.load_model(Pangu6.load_default_package()),
        )

    @classmethod
    def fengwu(cls) -> "ModelConfig":
        return cls(
            name="fengwu",
            variables=FENGWU_VARIABLES,
            workflow="deterministic",
            model=FengWu.load_model(FengWu.load_default_package()),
        )

    @classmethod
    def mock(cls) -> "ModelConfig":
        grid = earth2grid.latlon.equiangular_lat_lon_grid(721, 1440)
        coords = OrderedDict([("lat", grid.lat.ravel()), ("lon", grid.lon)])
        return cls(
            name="mock",
            variables=FCN3_VARIABLES,
            workflow="ensemble",
            model=Persistence(FCN3_VARIABLES, coords),
        )


# -----------------------------------------------------------------------------
# Data Source Wrapper
# -----------------------------------------------------------------------------


class DataArrayZarr:
    """
    E2Studio-compatible wrapper around xarray zarr that regrids from HEALPix to lat-lon.
    Output order is always: (time, variable, lat, lon).
    """

    _2D_VAR_MAP = {
        "tcwv": "tcwv",
        "t2m": "tas",
        "u10m": "uas",
        "v10m": "vas",
        "u100m": "100u",
        "v100m": "100v",
        "msl": "pres_msl",
    }

    def __init__(
        self,
        file_path: str,
        *,
        nlat: int = 721,
        nlon: int = 1440,
        xr_open_kwargs: dict[str, Any] | None = None,
    ):
        self.file_path = file_path
        self.nlat = nlat
        self.nlon = nlon
        self.include_south_pole = nlat == 721

        xr_open_kwargs = xr_open_kwargs or {}
        ds = xr.open_zarr(file_path, **xr_open_kwargs)

        if isinstance(ds, xr.Dataset):
            da = ds.to_array(dim="variable")
        else:
            da = ds

        dims = list(da.dims)
        self._on_latlon_grid = "lat" in da.dims and "lon" in da.dims

        if not self._on_latlon_grid and "cells" not in da.dims:
            raise ValueError("DataArray must have 'cells' dimension for non-latlon grids")

        # Move 'time' and 'variable' to front
        for lead in ("variable", "time"):
            if lead in dims:
                dims.remove(lead)
                dims.insert(0, lead)
        da = da.transpose(*dims, missing_dims="ignore")

        self.da = da
        self.has_type_dim = "type" in da.dims
        self._setup_regridding()

    def _setup_regridding(self):
        self._regridder = None
        self._lat_coords = None
        self._lon_coords = None

        if self._on_latlon_grid:
            return

        hpx_grid = healpix.Grid(level=6, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY)
        ll_grid = latlon.equiangular_lat_lon_grid(
            nlat=self.nlat, nlon=self.nlon, includes_south_pole=self.include_south_pole
        )

        self._regridder = earth2grid.get_regridder(hpx_grid, ll_grid).float().cuda()

        lat_arr = np.asarray(ll_grid.lat).squeeze()
        if lat_arr.ndim != 1:
            lat_arr = lat_arr[:, 0]
        lon_arr = np.asarray(ll_grid.lon).squeeze()
        if lon_arr.ndim != 1:
            lon_arr = lon_arr[0, :]
        self._lat_coords = lat_arr
        self._lon_coords = lon_arr

    def _regrid(self, np_block: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np_block).to(torch.float32).cuda()
        return self._regridder(t).cpu().float().numpy()

    def __call__(self, time, variable) -> xr.DataArray:
        if isinstance(variable, str):
            variable = [variable]

        dataset_variables = []
        available_vars = self.da.coords["variable"].values
        ours_to_fcn3 = {}

        for v in variable:
            if v in available_vars:
                dataset_variables.append(v)
            elif v.upper() in available_vars:
                dataset_variables.append(v.upper())
            elif v in self._2D_VAR_MAP:
                mapped_var = self._2D_VAR_MAP[v]
                if mapped_var in available_vars:
                    dataset_variables.append(mapped_var)
                else:
                    raise ValueError(f"Mapped variable {mapped_var} for {v} not found")
            else:
                raise ValueError(f"Variable {v} not found. Available: {available_vars}")

            our_var_name = dataset_variables[-1]
            ours_to_fcn3[our_var_name] = v

        data = self.da.sel(time=time, variable=dataset_variables)
        if self.has_type_dim:
            data = data.isel(type=0)

        if self._on_latlon_grid:
            return data

        np_block = np.ascontiguousarray(data.values.astype(np.float32))
        out = self._regrid(np_block)

        out_dims = list(data.dims[:-1]) + ["lat", "lon"]
        out_coords = {k: data.coords[k].values for k in data.dims if k != "cells"}
        out_coords["variable"] = [ours_to_fcn3[v] for v in out_coords["variable"]]
        out_coords["lat"] = self._lat_coords
        out_coords["lon"] = self._lon_coords

        return xr.DataArray(out, dims=out_dims, coords=out_coords)


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)


def main(argv=None):
    """Run forecast inference.

    Supports multiple weather models with HEALPix zarr output format.
    Single-GPU, batch_size=1 for simplicity.
    """
    parser = argparse.ArgumentParser(description="Run weather forecast inference")
    parser.add_argument(
        "--init_path", type=str, required=True, help="Path to input zarr"
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--num_steps", type=int, default=40, help="Number of forecast steps (6h each)"
    )
    parser.add_argument(
        "--num_times", type=int, default=4, help="Number of initial times to forecast"
    )
    parser.add_argument(
        "--num_ensemble", type=int, default=1, help="Number of ensemble members"
    )
    parser.add_argument(
        "--model",
        type=str.lower,
        choices=["fcn3", "aurora", "pangu", "fengwu", "mock"],
        default="mock",
        help="Model to use for inference",
    )
    parser.add_argument(
        "--no-bfloat16", action="store_false", dest="bfloat16", default=True
    )
    args = parser.parse_args(argv)

    setup_logging()

    # Create model config
    if args.model == "fcn3":
        model_config = ModelConfig.fcn3()
    elif args.model == "aurora":
        model_config = ModelConfig.aurora()
    elif args.model == "pangu":
        model_config = ModelConfig.pangu()
    elif args.model == "fengwu":
        model_config = ModelConfig.fengwu()
    else:
        model_config = ModelConfig.mock()

    # Ensemble size
    nensemble = 1 if model_config.workflow == "deterministic" else args.num_ensemble

    logger.info(f"Model: {model_config.name}, workflow: {model_config.workflow}")
    logger.info(f"Ensemble members: {nensemble}, steps: {args.num_steps}")

    # Load data
    if not os.path.exists(args.init_path):
        raise FileNotFoundError(f"Zarr not found: {args.init_path}")

    ds = DataArrayZarr(args.init_path, nlat=model_config.nlat, nlon=model_config.nlon)
    times = pd.to_datetime(ds.da.time.values)

    # Filter to 00/12 UTC and limit to num_times
    mask = times.hour.isin([0, 12])
    times = times[mask]
    times = times[: args.num_times]

    logger.info(f"Running {len(times)} forecasts: {times[0]} to {times[-1]}")

    # Setup output
    os.makedirs(args.out_dir, exist_ok=True)
    zarr_path = os.path.join(args.out_dir, "forecast.zarr")

    io_backend = RegriddingZarrBackend(
        zarr_path=zarr_path,
        times=times,
        out_vars=model_config.variables,
        n_ensemble=nensemble,
        nsteps=args.num_steps,
    )

    # Run inference
    model = model_config.model
    perturbation = Zero()

    for i, t in enumerate(times):
        logger.info(f"[{i + 1}/{len(times)}] Forecasting from {t}")

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=args.bfloat16
        ):
            if model_config.workflow == "deterministic":
                e2s_deterministic(
                    time=[t],
                    nsteps=args.num_steps,
                    prognostic=model,
                    data=ds,
                    io=io_backend,
                    output_coords={"variable": np.array(model_config.variables)},
                )
            else:
                e2s_ensemble(
                    time=[t],
                    nsteps=args.num_steps,
                    nensemble=nensemble,
                    prognostic=model,
                    data=ds,
                    io=io_backend,
                    perturbation=perturbation,
                    output_coords={"variable": np.array(model_config.variables)},
                )

        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Forecast complete. Output: {zarr_path}")


if __name__ == "__main__":
    main()
