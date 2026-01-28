#!/usr/bin/env python
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
Score forecasts against reference (e.g., ERA5).
Computes ensemble metrics (RMSE, spread, CRPS) per variable and lead time.
"""

import argparse
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Literal, Optional

import earth2grid
import numpy as np
import torch
import xarray as xr
from ensemble_metrics import unbiased_ensemble_metrics
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

HPX_LEVEL = 6

DEFAULT_SCORE_FIELDS = ["Z500", "U500", "T850", "Q700", "t2m", "msl", "u10m", "tcwv"]

# Surface field aliases: canonical name -> list of equivalent names to search for
SURFACE_ALIASES = {
    "t2m": ["t2m", "tas", "2t", "2m_temperature"],
    "u10m": ["u10m", "uas", "10u", "10m_u_component_of_wind"],
    "v10m": ["v10m", "vas", "10v", "10m_v_component_of_wind"],
    "msl": ["msl", "pres_msl", "mean_sea_level_pressure"],
    "tcwv": ["tcwv", "total_column_water_vapour"],
}


def resolve_field(ds: xr.Dataset, field: str) -> Optional[xr.DataArray]:
    """Resolve field with case-insensitive matching, level-dimension, and surface aliases.

    Handles:
    - Exact match (Z500)
    - Case-insensitive (Z500 vs z500)
    - Level-dimension selection (z with level=500 -> Z500)
    - Surface field aliases (t2m -> tas, 2t, etc.)
    """
    # Exact match
    if field in ds.data_vars:
        return ds[field]
    # Case-insensitive match (handles Z500 vs z500)
    for var in ds.data_vars:
        if var.lower() == field.lower():
            return ds[var]
    # Level-dimension selection (e.g., Z500 -> z.sel(level=500))
    if len(field) > 1 and field[0].isalpha() and field[1:].isdigit():
        base = field[0].lower()  # 'z' from 'Z500'
        level = int(field[1:])  # 500 from 'Z500'
        for var in [base, base.upper()]:
            if var in ds.data_vars and "level" in ds[var].dims:
                return ds[var].sel(level=level)
    # Surface field aliases (exact match only)
    if field in SURFACE_ALIASES:
        for alias in SURFACE_ALIASES[field]:
            if alias in ds.data_vars:
                return ds[alias]
    return None


def get_common_fields(
    datasets: list[xr.Dataset], fields: list[str] = None
) -> list[str]:
    """Get fields that exist in all datasets."""
    fields = fields or DEFAULT_SCORE_FIELDS
    common = []
    for f in fields:
        found_in_all = all(resolve_field(ds, f) is not None for ds in datasets)
        if found_in_all:
            common.append(f)
        else:
            # Log which dataset is missing the field
            missing_in = [
                i for i, ds in enumerate(datasets) if resolve_field(ds, f) is None
            ]
            logger.warning(
                f"Skipping field '{f}' - not found in dataset(s): {missing_in}"
            )
    return common


def open_any(path: str, storage_options: Optional[dict] = None) -> xr.Dataset:
    """Open dataset from zarr or netcdf, with optional S3 support."""
    if path.endswith(".zarr"):
        return xr.open_zarr(path, storage_options=storage_options)
    return xr.open_dataset(path, storage_options=storage_options)


def setup_hpx_regridder(input_format: Literal["nest", "ring", "hpxpadxy"]) -> callable:
    """Create regridder to convert input format to NEST."""
    if input_format == "nest":
        return lambda x: x
    src_order = {
        "ring": earth2grid.healpix.PixelOrder.RING,
        "hpxpadxy": earth2grid.healpix.HEALPIX_PAD_XY,
    }[input_format]
    return lambda x: earth2grid.healpix.reorder(
        torch.as_tensor(x), src_order, earth2grid.healpix.PixelOrder.NEST
    ).float()


def get_lead_time_hours(forecast: xr.Dataset) -> np.ndarray:
    """Get lead time values in hours as integers."""
    lead_time = forecast.lead_time
    if np.issubdtype(lead_time.dtype, np.timedelta64):
        if lead_time.dtype == np.dtype("timedelta64[ns]"):
            return (lead_time / np.timedelta64(1, "h")).astype(int).values
        return lead_time.astype("timedelta64[h]").astype(int).values
    return lead_time.values


def get_valid_times(forecast: xr.Dataset) -> xr.DataArray:
    """Calculate valid times for each forecast step."""
    lead_hours = get_lead_time_hours(forecast)
    lead_deltas = np.array([np.timedelta64(int(h), "h") for h in lead_hours])
    lead_time = xr.DataArray(
        lead_deltas, dims=["lead_time"], coords={"lead_time": lead_hours}
    )
    times = forecast.time.expand_dims(lead_time=lead_hours)
    return times + lead_time


@torch.no_grad()
def compute_metrics_for_field(
    reference_ds: xr.Dataset,
    forecast_ds: xr.Dataset,
    field: str,
    forecast_format: Literal["nest", "ring", "hpxpadxy"],
    reference_format: Literal["nest", "ring", "hpxpadxy"],
    worker_id: int = 0,
    num_gpus: int = 1,
) -> xr.Dataset:
    """Compute metrics for a single field."""
    gpu_id = worker_id % max(num_gpus, 1)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    forecast_regridder = setup_hpx_regridder(forecast_format)
    reference_regridder = setup_hpx_regridder(reference_format)

    reference = resolve_field(reference_ds, field)
    forecast = resolve_field(forecast_ds, field)
    if reference is None or forecast is None:
        raise ValueError(f"Field {field} not found in datasets")

    if "ensemble" not in forecast.dims:
        forecast = forecast.expand_dims(ensemble=[0])

    valid_times = get_valid_times(forecast)
    forecast = forecast.sel(time=valid_times.time)
    reference = reference.sel(time=valid_times)

    metrics_list = []
    nan_warning_count = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Processing {len(forecast.time)} time steps for {field}")

    for t in tqdm(range(len(forecast.time)), desc=f"Processing {field}"):
        forecast_t = forecast_regridder(
            torch.as_tensor(forecast.isel(time=t).values)
        ).to(device)
        forecast_t = forecast_t.permute(1, 2, 0)  # [lead_time, cells, ensemble]
        reference_t = reference_regridder(
            torch.as_tensor(reference.isel(time=t).values)
        ).to(device)

        # NaN handling
        if torch.isnan(forecast_t).any():
            if nan_warning_count < 2:
                logger.warning(f"NaN in forecast for {field} at time {t}")
                nan_warning_count += 1
            forecast_t = torch.nan_to_num(
                forecast_t, nan=forecast_t[~torch.isnan(forecast_t)].mean()
            )

        metrics_t = unbiased_ensemble_metrics(forecast_t, reference_t)
        metrics_t["mse_m0"] = (forecast_t[:, :, 0] - reference_t) ** 2

        metrics_list.append({k: v.cpu() for k, v in metrics_t.items()})
        del reference_t, forecast_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Combine metrics
    metrics = {
        key: torch.stack([m[key] for m in metrics_list], dim=0)
        for key in metrics_list[0].keys()
    }

    def reduce_avg(x):
        return x.mean(dim=(0, -1))

    data_vars = {
        "rmse_ens": (
            ("lead_time",),
            torch.sqrt(torch.clamp(reduce_avg(metrics["mse"]), min=0)).numpy(),
        ),
        "rmse_m0": (
            ("lead_time",),
            torch.sqrt(torch.clamp(reduce_avg(metrics["mse_m0"]), min=0)).numpy(),
        ),
    }

    # Ensemble metrics (spread, CRPS, SSR)
    if "variance" in metrics:
        spread = torch.sqrt(reduce_avg(metrics["variance"]))
        rmse_ens = torch.sqrt(torch.clamp(reduce_avg(metrics["mse"]), min=0))
        R = forecast_ds.sizes.get("ensemble", 1)
        # SSR = sqrt((R+1)/R) * spread / rmse
        ssr = (
            torch.sqrt(torch.tensor((R + 1) / R))
            * spread
            / torch.clamp(rmse_ens, min=1e-9)
        )

        data_vars["spread"] = (("lead_time",), spread.numpy())
        data_vars["crps"] = (("lead_time",), reduce_avg(metrics["crps"]).numpy())
        data_vars["ssr"] = (("lead_time",), ssr.numpy())

    return xr.Dataset(
        data_vars,
        coords={"lead_time": get_lead_time_hours(forecast_ds), "field": field},
    )


def score_forecast(
    reference_path: str,
    forecast_path: str,
    fields: list[str],
    forecast_format: Literal["nest", "ring", "hpxpadxy"] = "nest",
    reference_format: Literal["nest", "ring", "hpxpadxy"] = "hpxpadxy",
) -> xr.Dataset:
    """Score forecast against reference with parallel processing across fields."""
    reference_ds = open_any(reference_path)
    forecast_ds = open_any(forecast_path)

    fields = get_common_fields([reference_ds, forecast_ds], fields)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logger.info(f"Scoring {len(fields)} fields on {num_gpus} GPUs")

    worker = partial(
        compute_metrics_for_field,
        reference_ds,
        forecast_ds,
        forecast_format=forecast_format,
        reference_format=reference_format,
        num_gpus=num_gpus,
    )

    results = []
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(len(fields), 8), mp_context=mp_context
    ) as executor:
        futures = {
            executor.submit(worker, field, worker_id=i): field
            for i, field in enumerate(fields)
        }
        for future in tqdm(futures.keys(), desc="Scoring fields"):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error scoring {futures[future]}: {e}")

    if not results:
        raise RuntimeError("No fields were successfully scored")

    combined = xr.concat(results, dim="field")
    combined.lead_time.attrs = {"long_name": "forecast lead time", "units": "hours"}
    return combined


def main():
    multiprocessing.set_start_method(
        "spawn"
    )  # Required for CUDA with ProcessPoolExecutor

    parser = argparse.ArgumentParser(description="Score forecasts against reference")
    parser.add_argument(
        "--forecast_path",
        type=str,
        required=True,
        help="Path to forecast zarr (from forecast.py)",
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        required=True,
        help="Path to reference zarr (from inference.py --use_analysis)",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save metrics (.nc)"
    )
    parser.add_argument(
        "--forecast_format",
        type=str,
        default="nest",
        choices=["nest", "ring", "hpxpadxy"],
        help="HEALPix format of forecast",
    )
    parser.add_argument(
        "--reference_format",
        type=str,
        default="hpxpadxy",
        choices=["nest", "ring", "hpxpadxy"],
        help="HEALPix format of reference",
    )
    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=None,
        help=f"Fields (default: {DEFAULT_SCORE_FIELDS})",
    )
    args = parser.parse_args()

    fields = args.fields or DEFAULT_SCORE_FIELDS
    logger.info(f"Forecast: {args.forecast_path}")
    logger.info(f"Reference: {args.reference_path}")
    logger.info(f"Scoring {len(fields)} fields: {fields}")

    metrics = score_forecast(
        reference_path=args.reference_path,
        forecast_path=args.forecast_path,
        fields=fields,
        forecast_format=args.forecast_format,
        reference_format=args.reference_format,
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    metrics.to_netcdf(args.output_path)
    logger.info(f"Saved metrics to {args.output_path}")


if __name__ == "__main__":
    main()
