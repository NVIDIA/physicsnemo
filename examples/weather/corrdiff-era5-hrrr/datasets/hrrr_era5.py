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

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from physicsnemo.utils.zenith_angle import cos_zenith_angle

from .dataset import DownscalingDataset


class HRRRERA5Dataset(DownscalingDataset):
    """Same-time ERA5 -> HRRR downscaling samples over the contiguous US.

    Expects the data under ``data_path`` with the following layout:

    | <data_path>
    | -- hrrr-conus/<year>.zarr         # "HRRR" (time, channel, y, x), "latitude",
    |                                   # "longitude" (y, x), and "valid_data" (time)
    | -- era5-conus/<year>.zarr         # "data" (time, channel, latitude, longitude)
    | -- invariants/invariants_HRRR.nc  # static fields on the HRRR grid
    | -- stats/hrrr_era5_stats.json     # {"hrrr"|"era5"|"invariant": {var: {"mean", "std"}}}

    Each sample is a dictionary with the following normalized arrays:

    - ``state``: the HRRR target state, shape ``(C_state, H, W)``
    - ``background``: the ERA5 fields bilinearly interpolated to the HRRR grid,
      followed by the cosine of the solar zenith angle and the invariants,
      shape ``(C_background, H, W)``
    - ``scalar_conditions``: sine and cosine of the time of year followed by
      one presence flag per droppable ERA5 variable, shape ``(C_scalar,)``

    Training samples are random patches of shape ``patch_shape`` cropped out of
    the domain, while the position of each validation patch depends only on
    the sample index. With ``patch_shape=None`` each sample covers the full
    domain.

    During training, the dataset replaces each ERA5 variable in
    ``drop_variables`` by its climatological mean (zero in normalized units)
    with probability ``drop_prob``, and sets its presence flag to zero. This
    lets the model run without these inputs. Validation and inference never
    drop inputs.

    Parameters
    ----------
    data_path : str
        Root directory of the data.
    years : list[int]
        Years of data to use.
    time_range_train : tuple[str, str]
        Training samples have ``time_range_train[0] <= time < time_range_train[1]``.
    time_range_valid : tuple[str, str]
        Validation samples have ``time_range_valid[0] <= time < time_range_valid[1]``.
    invariant_variables : list[str]
        Static fields appended to the background channels.
    crop_window : tuple[tuple[int, int], tuple[int, int]]
        ``((y0, y1), (x0, x1))`` window of the HRRR grid used as the domain.
    patch_shape : tuple[int, int] | None
        Shape of the patches, or ``None`` to use the full domain.
    drop_variables : list[str]
        ERA5 variables that the dataset randomly drops during training.
    drop_prob : float
        Probability of dropping each variable in ``drop_variables``.
    train : bool
        Whether to serve training or validation samples.
    """

    def __init__(
        self,
        data_path: str,
        years: list[int],
        time_range_train: tuple[str, str],
        time_range_valid: tuple[str, str],
        invariant_variables: list[str] = ["elev_mean", "elev_std", "lsm_mean"],
        crop_window: tuple[tuple[int, int], tuple[int, int]] = ((17, 1041), (3, 1795)),
        patch_shape: tuple[int, int] | None = (256, 256),
        drop_variables: list[str] = [],
        drop_prob: float = 0.0,
        train: bool = True,
    ):
        self.train = train
        self.crop_window = crop_window
        self.crop = tuple(slice(i0, i1) for (i0, i1) in crop_window)
        self.domain_shape = tuple(i1 - i0 for (i0, i1) in crop_window)
        self.patch_shape = None if patch_shape is None else tuple(patch_shape)
        self.invariant_variables = list(invariant_variables)
        self.drop_variables = list(drop_variables)
        self.drop_prob = drop_prob

        self.hrrr_paths = {
            year: os.path.join(data_path, "hrrr-conus", f"{year}.zarr")
            for year in years
        }
        self.era5_paths = {
            year: os.path.join(data_path, "era5-conus", f"{year}.zarr")
            for year in years
        }
        self.hrrr_datasets = {}  # opened on first access in each dataloader worker
        self.era5_datasets = {}

        # Valid sample times (valid HRRR data with matching ERA5 data),
        # channel names, and HRRR coordinates
        times = []
        for year in years:
            with xr.open_zarr(self.hrrr_paths[year]) as ds:
                hrrr_times = ds["time"].values[ds["valid_data"].values]
            with xr.open_zarr(self.era5_paths[year]) as ds:
                era5_times = ds["time"].values
            times.append(hrrr_times[np.isin(hrrr_times, era5_times)])
        with xr.open_zarr(self.hrrr_paths[years[0]]) as ds:
            self.hrrr_channels = [str(c) for c in ds["channel"].values]
            self.lat = ds["latitude"].values[self.crop]
            self.lon = ds["longitude"].values[self.crop]
        with xr.open_zarr(self.era5_paths[years[0]]) as ds:
            self.era5_channels = [str(c) for c in ds["channel"].values]
            era5_lat = ds["latitude"].values
            era5_lon = ds["longitude"].values

        (t0, t1) = (
            np.datetime64(t) for t in (time_range_train if train else time_range_valid)
        )
        self.times = np.concatenate(times)
        self.times = self.times[(self.times >= t0) & (self.times < t1)]

        # Fractional ERA5 grid indices of each HRRR pixel, used for the bilinear
        # interpolation of ERA5 to the HRRR grid
        i = (self.lat - era5_lat[0]) / (era5_lat[1] - era5_lat[0])
        j = (self.lon - era5_lon[0]) / (era5_lon[1] - era5_lon[0])
        if not (
            (i.min() >= 0)
            and (i.max() < len(era5_lat) - 1)
            and (j.min() >= 0)
            and (j.max() < len(era5_lon) - 1)
        ):
            raise ValueError("The ERA5 grid must cover the HRRR domain.")
        self.i0 = i.astype(np.int64)
        self.di = (i - self.i0).astype(np.float32)
        self.j0 = j.astype(np.int64)
        self.dj = (j - self.j0).astype(np.float32)

        # Normalization statistics and the normalized invariants
        with open(os.path.join(data_path, "stats", "hrrr_era5_stats.json")) as f:
            stats = json.load(f)
        (self.hrrr_mean, self.hrrr_std) = _load_stats(stats, "hrrr", self.hrrr_channels)
        (self.era5_mean, self.era5_std) = _load_stats(stats, "era5", self.era5_channels)
        (inv_mean, inv_std) = _load_stats(stats, "invariant", self.invariant_variables)
        invariants_path = os.path.join(data_path, "invariants", "invariants_HRRR.nc")
        with xr.open_dataset(invariants_path) as ds:
            invariants = np.stack(
                [ds[v].values[self.crop] for v in self.invariant_variables]
            )
        self.invariants = (invariants - inv_mean) / inv_std

        self.drop_indices = [self.era5_channels.index(v) for v in self.drop_variables]

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        time = pd.Timestamp(self.times[idx])
        (y0, x0) = self._patch_origin(idx)
        (h, w) = self.image_shape()
        patch = (slice(y0, y0 + h), slice(x0, x0 + w))

        # HRRR target state; the patch origin is relative to the crop window
        hrrr = self._open_dataset(self.hrrr_datasets, self.hrrr_paths, time.year)
        (gy0, gx0) = (self.crop_window[0][0] + y0, self.crop_window[1][0] + x0)
        state = hrrr["HRRR"].sel(time=time)[:, gy0 : gy0 + h, gx0 : gx0 + w].values
        state = (state - self.hrrr_mean) / self.hrrr_std

        # ERA5 interpolated to the HRRR grid, then the zenith angle and invariants
        era5 = self._open_dataset(self.era5_datasets, self.era5_paths, time.year)
        era5 = era5["data"].sel(time=time).values
        era5 = _bilinear_interpolation(
            era5, self.i0[patch], self.di[patch], self.j0[patch], self.dj[patch]
        )
        era5 = (era5 - self.era5_mean) / self.era5_std
        presence_flags = self._drop_variables(era5)
        cos_zenith = cos_zenith_angle(
            time.to_pydatetime(), lon=self.lon[patch], lat=self.lat[patch]
        )
        background = np.concatenate(
            [era5, cos_zenith[None], self.invariants[(slice(None),) + patch]]
        )

        # Time of year (one period = 365.25 days) and the input presence flags
        start_of_year = pd.Timestamp(year=time.year, month=1, day=1)
        year_phase = 2 * np.pi * ((time - start_of_year) / pd.Timedelta(days=365.25))
        scalar_conditions = np.concatenate(
            [[np.sin(year_phase), np.cos(year_phase)], presence_flags]
        )

        return {
            "background": background.astype(np.float32),
            "state": state.astype(np.float32),
            "scalar_conditions": scalar_conditions.astype(np.float32),
        }

    def _patch_origin(self, idx: int) -> tuple[int, int]:
        """Top-left corner of the patch within the domain."""
        if self.patch_shape is None:
            return (0, 0)
        max_origin = [
            d - p + 1
            for (d, p) in zip(self.domain_shape, self.patch_shape, strict=True)
        ]
        if self.train:
            return tuple(int(np.random.randint(m)) for m in max_origin)
        rng = np.random.default_rng(idx)  # same validation patches every time
        return tuple(int(rng.integers(m)) for m in max_origin)

    def _drop_variables(self, era5: np.ndarray) -> np.ndarray:
        """Randomly zero out the droppable ERA5 channels in place (training only)
        and return their presence flags (1 = kept, 0 = dropped)."""
        flags = np.ones(len(self.drop_indices), dtype=np.float32)
        if self.train:
            for k, c in enumerate(self.drop_indices):
                if np.random.random() < self.drop_prob:
                    era5[c] = 0.0
                    flags[k] = 0.0
        return flags

    @staticmethod
    def _open_dataset(datasets: dict, paths: dict, year: int) -> xr.Dataset:
        if year not in datasets:
            datasets[year] = xr.open_dataset(
                paths[year], engine="zarr", chunks=None, mask_and_scale=False
            )
        return datasets[year]

    def background_channels(self) -> list[str]:
        """ERA5 channels, the cosine of the solar zenith angle, and the invariants."""
        return self.era5_channels + ["cos_zenith"] + self.invariant_variables

    def state_channels(self) -> list[str]:
        """HRRR channels."""
        return self.hrrr_channels

    def scalar_condition_channels(self) -> list[str]:
        """Time of year and the presence flags of the droppable variables."""
        return ["sin_year", "cos_year"] + [f"{v}_present" for v in self.drop_variables]

    def image_shape(self) -> tuple[int, int]:
        """Shape (height, width) of the samples."""
        return self.domain_shape if self.patch_shape is None else self.patch_shape

    def latitude(self) -> np.ndarray:
        """Latitude of the full domain. It has the shape ``image_shape()`` only
        with ``patch_shape=None``, which inference uses."""
        return self.lat

    def longitude(self) -> np.ndarray:
        """Longitude of the full domain. It has the shape ``image_shape()`` only
        with ``patch_shape=None``, which inference uses."""
        return self.lon

    def denormalize_background(self, x: np.ndarray) -> np.ndarray:
        """Convert the ERA5 channels of the background to physical units,
        leaving the zenith angle and invariant channels unchanged."""
        x = x.copy()
        n = len(self.era5_channels)
        x[..., :n, :, :] = x[..., :n, :, :] * self.era5_std + self.era5_mean
        return x

    def denormalize_state(self, x: np.ndarray) -> np.ndarray:
        """Convert the state from normalized data to physical units."""
        return x * self.hrrr_std + self.hrrr_mean


def _load_stats(
    stats: dict, group: str, variables: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of ``variables``, shaped ``(C, 1, 1)``."""
    mean = np.array([stats[group][v]["mean"] for v in variables], dtype=np.float32)
    std = np.array([stats[group][v]["std"] for v in variables], dtype=np.float32)
    return (mean[:, None, None], std[:, None, None])


def _bilinear_interpolation(
    x: np.ndarray, i0: np.ndarray, di: np.ndarray, j0: np.ndarray, dj: np.ndarray
) -> np.ndarray:
    """Bilinearly interpolate ``x`` of shape ``(C, Ny, Nx)`` at the fractional
    grid indices ``(i0 + di, j0 + dj)``."""
    f0 = x[:, i0, j0] + dj * (x[:, i0, j0 + 1] - x[:, i0, j0])
    f1 = x[:, i0 + 1, j0] + dj * (x[:, i0 + 1, j0 + 1] - x[:, i0 + 1, j0])
    return f0 + di * (f1 - f0)
