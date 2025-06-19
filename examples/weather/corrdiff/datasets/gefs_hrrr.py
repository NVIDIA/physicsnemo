# Data loader for TWC MVP: GEFS and HRRR forecasts
# adapted from https://gitlab-master.nvidia.com/earth-2/corrdiff-internal/-/blob/dpruitt/hrrr/explore/dpruitt/hrrr/datasets/hrrr.py

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

from datetime import datetime, timedelta
import glob
import logging
import os
from typing import Iterable, Tuple, Union, List
import copy

import cftime
import dask
import json
import numpy as np
import torch
import xarray as xr
import cv2

from physicsnemo.distributed import DistributedManager

from datasets.base import ChannelMetadata, DownscalingDataset

# """
# TO DO LIST:
# missing samples file
# mean and std
# """


hrrr_stats_channels = [
    "u10m",
    "v10m",
    "t2m",
    "precip",
    "cat_snow",
    "cat_ice",
    "cat_freez",
    "cat_rain",
    "refc",
]
gefs_surface_channels = [
    "u10m",
    "v10m",
    "t2m",
    "q2m",
    "sp",
    "msl",
    "precipitable_water",
]
gefs_isobaric_channels = [
    "u1000",
    "u925",
    "u850",
    "u700",
    "u500",
    "u250",
    "v1000",
    "v925",
    "v850",
    "v700",
    "v500",
    "v250",
    "z1000",
    "z925",
    "z850",
    "z700",
    "z500",
    "z200",
    "t1000",
    "t925",
    "t850",
    "t700",
    "t500",
    "t100",
    "r1000",
    "r925",
    "r850",
    "r700",
    "r500",
    "r100",
]


def convert_datetime_to_cftime(
    time: datetime, cls=cftime.DatetimeGregorian
) -> cftime.DatetimeGregorian:
    """Convert a Python datetime object to a cftime DatetimeGregorian object."""
    return cls(time.year, time.month, time.day, time.hour, time.minute, time.second)


def time_range(
    start_time: datetime,
    end_time: datetime,
    step: timedelta,
    inclusive: bool = False,
):
    """Like the Python `range` iterator, but with datetimes."""
    t = start_time
    while (t <= end_time) if inclusive else (t < end_time):
        yield t
        t += step


class HrrrForecastGEFSDataset(DownscalingDataset):
    """
    Paired dataset object serving time-synchronized pairs of GEFS (surface and isobaric) and HRRR samples
    Expects data to be stored under directory specified by 'location'
        GEFS under <root_dir>/gefs/
        HRRR under <root_dir>/hrrr/
    Within each directory, there should be one zarr file per
    year containing the data of interest.
    """

    def __init__(
        self,
        *,
        data_path: str,
        stats_path: str,
        input_surface_variables: Union[List[str], None] = None,
        input_isobaric_variables: Union[List[str], None] = None,
        output_variables: Union[List[str], None] = None,
        prob_variables: Union[List[str], None] = None,
        train: bool = True,
        normalize: bool = True,
        train_years: Iterable[int] = (2020, 2021, 2022, 2023),
        valid_years: Iterable[int] = (2024,),
        hrrr_window: Union[Tuple[Tuple[int, int], Tuple[int, int]], None] = None,
        sample_shape: Tuple[int, int] = [-1, -1],
        ds_factor: int = 1,
        shard: bool = False,
        overfit: bool = False,
        use_all: bool = False,
    ):
        dask.config.set(
            scheduler="synchronous"
        )  # for threadsafe multiworker dataloaders
        self.data_path = data_path
        self.train = train
        self.normalize = normalize
        self.output_variables = output_variables
        self.prob_channels = prob_variables
        self.input_surface_variables = input_surface_variables
        self.input_isobaric_variables = input_isobaric_variables
        self.input_variables = input_surface_variables + input_isobaric_variables
        self.train_years = list(train_years)
        self.valid_years = list(valid_years)
        self.hrrr_window = hrrr_window
        self.sample_shape = sample_shape
        self.ds_factor = ds_factor
        self.shard = shard
        self.use_all = use_all
        self.input_isobaric_variables_load = input_isobaric_variables
        self.output_variables_load = copy.deepcopy(output_variables)
        self._get_files_stats()
        self._gefs_name_edit()
        self.overfit = overfit

        with open(stats_path, "r") as f:
            stats = json.load(f)

        (self.input_mean, self.input_std) = _load_stats(
            stats, self.input_variables, "input"
        )
        (self.output_mean, self.output_std) = _load_stats(
            stats, self.output_variables, "output"
        )
        self.prob_channel_index = [
            self.output_variables_load.index(prob_channel)
            for prob_channel in self.prob_channels
        ]

    def _gefs_name_edit(self):
        """
        Handle naming bugs in preprocessed gefs dataset
        """
        if "cat_none" in self.output_variables_load:
            self.output_variables_load.remove("cat_none")
        for i in range(len(self.input_isobaric_variables_load)):
            if self.input_isobaric_variables_load[i] == "z200":
                self.input_isobaric_variables_load[i] = "z250"
            elif self.input_isobaric_variables_load[i] == "t100":
                self.input_isobaric_variables_load[i] = "t250"
            elif "r" in self.input_isobaric_variables_load[i]:
                self.input_isobaric_variables_load[
                    i
                ] = self.input_isobaric_variables_load[i].replace("r", "q")
                if self.input_isobaric_variables_load[i] == "q100":
                    self.input_isobaric_variables_load[i] = "q250"

    def _get_files_stats(self):
        """
        Scan directories and extract metadata for GEFS (surface and isobaric) and HRRR

        Note: This makes the assumption that the lowest numerical year has the
        correct channel ordering for the means
        """

        # GEFS surface parsing
        self.ds_gefs_surface = {}
        gefs_surface_paths = glob.glob(
            os.path.join(self.data_path, "gefs", "*surface*.zarr"), recursive=True
        )
        gefs_surface_years = [
            os.path.basename(x).replace(".zarr", "")[-10:]
            for x in gefs_surface_paths
            if "stats" not in x
        ]
        self.gefs_surface_paths = dict(zip(gefs_surface_years, gefs_surface_paths))

        # keep only training or validation years
        years = self.train_years if self.train else self.valid_years
        self.gefs_surface_paths = {
            year: path
            for (year, path) in self.gefs_surface_paths.items()
            if int(year[:4]) in years
        }
        self.n_years_surface = len(self.gefs_surface_paths)
        first_key = list(self.gefs_surface_paths.keys())[0]

        with xr.open_zarr(self.gefs_surface_paths[first_key], consolidated=True) as ds:
            self.base_gefs_surface_channels = list(ds.channel.values)
            self.gefs_surface_lat = ds.lat
            self.gefs_surface_lon = ds.lon % 360

        self.ds_gefs_isobaric = {}
        gefs_isobaric_paths = [
            path.replace("surface", "isobaric") for path in gefs_surface_paths
        ]
        gefs_isobaric_years = [
            os.path.basename(x).replace(".zarr", "")[-10:]
            for x in gefs_isobaric_paths
            if "stats" not in x
        ]
        self.gefs_isobaric_paths = dict(zip(gefs_isobaric_years, gefs_isobaric_paths))

        # keep only training or validation years
        years = self.train_years if self.train else self.valid_years
        self.gefs_isobaric_paths = {
            year: path
            for (year, path) in self.gefs_isobaric_paths.items()
            if int(year[:4]) in years
        }
        self.n_years_surface = len(self.gefs_isobaric_paths)
        first_key = list(self.gefs_isobaric_paths.keys())[0]

        # with xr.open_zarr(gefs_isobaric_paths[0], consolidated=True) as ds:
        with xr.open_zarr(self.gefs_isobaric_paths[first_key], consolidated=True) as ds:
            self.base_gefs_isobaric_channels = list(ds.channel.values)
            self.gefs_isobaric_lat = ds.lat
            self.gefs_isobaric_lon = ds.lon % 360

        # HRRR parsing
        self.ds_hrrr = {}
        hrrr_paths = glob.glob(
            os.path.join(self.data_path, "hrrr_forecast", "*[!backup].zarr"),
            recursive=True,
        )
        hrrr_years = [
            os.path.basename(x).replace(".zarr", "")[-10:]
            for x in hrrr_paths
            if "stats" not in x
        ]
        self.hrrr_paths = dict(zip(hrrr_years, hrrr_paths))
        # keep only training or validation years
        self.hrrr_paths = {
            year: path
            for (year, path) in self.hrrr_paths.items()
            if int(year[:4]) in years
        }
        self.years = set([int(key[:4]) for key in self.hrrr_paths.keys()])

        # assert set(gefs_surface_years) == set(hrrr_years) == set(gefs_isobaric_years), 'Number of years for GEFS surface, GEFS isobaric, and HRRR must match'
        with xr.open_zarr(hrrr_paths[0], consolidated=True) as ds:
            self.base_hrrr_channels = list(ds.channel.values)
            self.hrrr_lat = ds.lat
            self.hrrr_lon = ds.lon % 360

        if self.hrrr_window is None:
            self.hrrr_window = (
                (0, self.hrrr_lat.shape[0]),
                (0, self.hrrr_lat.shape[1]),
            )

        self.n_samples_total = self.compute_total_samples()

    def __len__(self):
        return len(self.valid_samples)

    def crop_to_fit(self, x):
        """
        Crop HRRR to get nicer dims
        """
        ((y0, y1), (x0, x1)) = self._get_crop_box()
        return x[..., y0:y1, x0:x1]

    def to_datetime(self, date):
        timestamp = (date - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(
            1, "s"
        )
        return datetime.utcfromtimestamp(timestamp)

    def compute_total_samples(self):

        # Loop through all years and count the total number of samples

        first_year = min(self.years)
        last_year = max(self.years)
        if first_year == 2020:
            first_sample = datetime(
                year=2020, month=12, day=2, hour=0, minute=0, second=0
            )
        else:
            first_sample = datetime(
                year=first_year, month=1, day=1, hour=0, minute=0, second=0
            )
        if last_year == 2024:
            last_sample = datetime(
                year=2024, month=7, day=29, hour=18, minute=0, second=0
            )
        else:
            last_sample = datetime(
                year=last_year, month=12, day=31, hour=18, minute=0, second=0
            )

        logging.info("First sample is {}".format(first_sample))
        logging.info("Last sample is {}".format(last_sample))

        all_datetimes = time_range(
            first_sample, last_sample, step=timedelta(hours=6), inclusive=True
        )

        all_datetimes = set(dt for dt in all_datetimes if dt.year in self.years)
        all_datetimes = set(
            time.strftime("%Y%m%d%H") + f"f{f:02d}"
            for time in all_datetimes
            for f in range(0, 25, 3)
        )

        self.valid_samples = list(all_datetimes)  # exclude missing samples
        logging.info(
            "Total datetimes in training set are {} of which {} are valid".format(
                len(all_datetimes), len(self.valid_samples)
            )
        )

        if self.shard:  # use only part of dataset in each training process
            dist_manager = DistributedManager()
            self.valid_samples = np.array_split(
                self.valid_samples, dist_manager.world_size
            )[dist_manager.rank]

        return len(self.valid_samples)

    def normalize_input(self, x):
        x = x.astype(np.float32)
        if self.normalize:
            x -= self.input_mean
            x /= self.input_std
        return x

    def denormalize_input(self, x):
        x = x.astype(np.float32)
        if self.normalize:
            x *= self.input_std
            x += self.input_mean
        return x

    def _get_gefs_surface(self, ts, lat, lon):
        """
        Retrieve GEFS surface samples from zarr files
        """
        gefs_surface_handle = self._get_ds_handles(
            self.ds_gefs_surface, self.gefs_surface_paths, ts
        )
        data = gefs_surface_handle.sel(time=ts, channel=self.input_surface_variables)
        data["x"] = data.lat.values[:, 0]
        data["y"] = data.lon.values[0, :] % 360
        gefs_surface_field = data.interp(x=lat, y=lon)["values"].values
        if len(gefs_surface_field.shape) == 4:
            gefs_surface_field = gefs_surface_field[0]
        return gefs_surface_field

    def _get_gefs_isobaric(self, ts, lat, lon):
        """
        Retrieve GEFS isobaric samples from zarr files
        """
        gefs_isobaric_handle = self._get_ds_handles(
            self.ds_gefs_isobaric, self.gefs_isobaric_paths, ts
        )
        data = gefs_isobaric_handle.sel(time=ts, channel=self.input_isobaric_variables)
        data["x"] = data.lat.values[:, 0]
        data["y"] = data.lon.values[0, :] % 360
        gefs_isobaric_field = data.interp(x=lat, y=lon)["values"].values
        if len(gefs_isobaric_field.shape) == 4:
            gefs_isobaric_field = gefs_isobaric_field[0]
        return gefs_isobaric_field

    def normalize_output(self, x):
        x = x.astype(np.float32)
        if self.normalize:
            x -= self.output_mean
            x /= self.output_std
        return x

    def denormalize_output(self, x):
        x = x.astype(np.float32)
        if self.normalize:
            x *= self.output_std
            x += self.output_mean
        return x

    def _get_hrrr(self, ts, crop_box):
        """
        Retrieve HRRR samples from zarr files
        """

        hrrr_handle = self._get_ds_handles(
            self.ds_hrrr, self.hrrr_paths, ts, mask_and_scale=False
        )
        ds_channel_names = list(np.array(hrrr_handle.channel))
        ((y0, y1), (x0, x1)) = crop_box
        hrrr_field = hrrr_handle.sel(time=ts, channel=self.output_variables_load)[
            "values"
        ][..., y0:y1, x0:x1].values
        if len(hrrr_field.shape) == 4:
            hrrr_field = hrrr_field[0]

        is_precip = np.sum(
            hrrr_field[
                self.prob_channel_index,
            ],
            axis=0,
            keepdims=True,
        )
        hrrr_non_precip = (is_precip == 0).astype(float)
        hrrr_field = np.concatenate((hrrr_field, hrrr_non_precip), axis=0)
        prob_channel_index = self.prob_channel_index + [hrrr_field.shape[0] - 1]
        hrrr_field[prob_channel_index] = hrrr_field[prob_channel_index] / np.sum(
            hrrr_field[
                prob_channel_index,
            ],
            axis=0,
            keepdims=True,
        )
        hrrr_field = self.normalize_output(hrrr_field)
        return hrrr_field

    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data (same for input and output)."""
        ((y_start, y_end), (x_start, x_end)) = self.hrrr_window
        return (y_end - y_start, x_end - x_start)

    def _get_crop_box(self):
        if self.sample_shape == [-1, -1]:
            return self.hrrr_window

        ((y_start, y_end), (x_start, x_end)) = self.hrrr_window

        y0 = np.random.randint(y_start, y_end - self.sample_shape[0] + 1)
        y1 = y0 + self.sample_shape[0]
        x0 = np.random.randint(x_start, x_end - self.sample_shape[1] + 1)
        x1 = x0 + self.sample_shape[1]
        return ((y0, y1), (x0, x1))

    def __getitem__(self, global_idx):
        """Return a tuple of:
        - hrrr_field: High-resolution HRRR output data
        - gefs_field: Low-resolution GEFS input data
        - lead_time_label: Lead time
        """
        if self.overfit:
            global_idx = 42
        time_index = self._global_idx_to_datetime(global_idx)
        ((y0, y1), (x0, x1)) = crop_box = self._get_crop_box()
        lon = self.hrrr_lon[y0:y1, x0:x1]
        lat = self.hrrr_lat[y0:y1, x0:x1]
        gefs_surface_sample = self._get_gefs_surface(time_index, lon=lon, lat=lat)
        gefs_isobaric_sample = self._get_gefs_isobaric(time_index, lon=lon, lat=lat)
        if self.ds_factor > 1:
            gefs_surface_sample = self._create_lowres_(
                gefs_surface_sample, factor=self.ds_factor
            )
            gefs_isobaric_sample = self._create_lowres_(
                gefs_isobaric_sample, factor=self.ds_factor
            )
        hrrr_sample = self._get_hrrr(time_index, crop_box=crop_box)
        gefs_sample = np.concatenate(
            (gefs_surface_sample, gefs_isobaric_sample), axis=0
        )
        gefs_sample = self.normalize_input(gefs_sample)
        torch.cuda.nvtx.range_pop()
        return hrrr_sample, gefs_sample, int(time_index[-2:]) // 3

    def _global_idx_to_datetime(self, global_idx):
        """
        Parse a global sample index and return the input/target timstamps as datetimes
        """
        return self.valid_samples[global_idx]

    def _get_ds_handles(self, handles, paths, ts, mask_and_scale=True):
        """
        Return handles for the appropriate year
        """
        if ts[:4] == "2020":
            name = "2020_12_12"
        elif ts[:4] == "2024":
            name = "2024_01_07"
        elif "01" <= ts[4:6] <= "06":
            name = ts[:4] + "_01_06"
        elif "07" <= ts[4:6] <= "12":
            name = ts[:4] + "_07_12"
        else:
            raise Exception("wrong time")

        if name in handles:
            ds_handle = handles[name]
        else:
            ds_handle = xr.open_zarr(
                paths[name], consolidated=True, mask_and_scale=mask_and_scale
            )
            handles[name] = ds_handle
        return ds_handle

    @staticmethod
    def _create_lowres_(x, factor=4):
        # downsample the high res imag
        x = x.transpose(1, 2, 0)
        x = x[::factor, ::factor, :]  # 8x8x3  #subsample
        # upsample with bicubic interpolation to bring the image to the nominal size
        x = cv2.resize(
            x, (x.shape[1] * factor, x.shape[0] * factor), interpolation=cv2.INTER_CUBIC
        )  # 32x32x3
        x = x.transpose(2, 0, 1)  # 3x32x32
        return x

    def latitude(self):
        return self.hrrr_lat if self.train else self.crop_to_fit(self.hrrr_lat)

    def longitude(self):
        return self.hrrr_lon if self.train else self.crop_to_fit(self.hrrr_lon)

    def time(self):
        return self.valid_samples

    def get_prob_channel_index(self):
        """
        Get prob_channel_index list one more dimension
        """
        return self.prob_channel_index + [len(self.output_variables) - 1]

    def input_channels(self):
        return [ChannelMetadata(name=n) for n in self.input_variables]

    def output_channels(self):
        return [ChannelMetadata(name=n) for n in self.output_variables]

    def info(self):
        return {
            "input_normalization": (
                self.input_mean.squeeze(),
                self.input_std.squeeze(),
            ),
            "target_normalization": (
                self.output_mean.squeeze(),
                self.output_std.squeeze(),
            ),
        }


def _load_stats(stats, variables, group):
    mean = np.array([stats[group][v]["mean"] for v in variables])[:, None, None].astype(
        np.float32
    )
    std = np.array([stats[group][v]["std"] for v in variables])[:, None, None].astype(
        np.float32
    )
    return (mean, std)
