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
import logging
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime, timedelta
import dask
import xarray as xr


def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed % (2**32 - 1))


def get_dataset(params, train):
    """
    Get the dataset for the given parameters and training flag.
    """
    if params.task == "downscale":
        return HrrrEra5DatasetDownscale(
            params,
            train=train,
            location=params.location,
        )
    else:
        raise ValueError("Unsupported dataset type: {}".format(task))


def get_absolute_path(relative_path):
    """
    Get the absolute path for the given relative path.
    """
    root = os.getenv("DATA_ROOT", "/")
    relative_path = relative_path.lstrip("/")  # strip leading "/"
    location = os.path.join(root, relative_path)
    return location


def get_inference_data_loader(params):
    """
    Get the inference data loader for the given parameters.
    """
    dataset = get_dataset(params, train=False)
    dataloader = DataLoader(
        dataset,
        batch_size=int(params.local_batch_size),
        num_workers=params.num_data_workers,
        shuffle=False,
        sampler=None,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader, dataset


def get_data_loader(params, distributed, train):
    """
    Get the data loader for the given parameters and training flag.
    """
    dataset = get_dataset(params, train=train)
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None

    dataloader = DataLoader(
        dataset,
        batch_size=int(params.local_batch_size),
        num_workers=params.num_data_workers,
        shuffle=(sampler is None),
        sampler=sampler if train else None,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


class HrrrEra5DatasetDownscale(Dataset):
    """
    Paired dataset object serving time-synchronized pairs of ERA5 and HRRR samples
    Expects data to be stored under directory specified by 'location'
        ERA5 under <root_dir>/era5/
        HRRR under <root_dir>/hrrr/
    """

    def __init__(self, params, train, location: str):
        dask.config.set(
            scheduler="synchronous"
        )  # for threadsafe multiworker dataloaders
        self.params = params
        self.location = location
        self.train = train
        self.path_suffix = "train" if train else "valid"
        self.normalize = True
        self.conus_dataset_name = params.conus_dataset_name
        self.boundary_padding_pixels = params.boundary_padding_pixels
        self._get_files_stats()
        self.hrrr_stats = self.params.hrrr_stats
        # self.means_hrrr = np.load(os.path.join(self.location, self.conus_dataset_name, self.hrrr_stats, 'means.npy'))[:, None, None]
        # self.stds_hrrr = np.load(os.path.join(self.location, self.conus_dataset_name, self.hrrr_stats, 'stds.npy'))[:, None, None]
        self.means_hrrr = np.array(params.diffusion_channels_means)[:, None, None]
        self.stds_hrrr = np.array(params.diffusion_channels_stds)[:, None, None]
        # self.means_era5 = np.load(os.path.join(self.location, 'era5', 'stats', 'means.npy'))[:, None, None]
        # self.stds_era5 = np.load(os.path.join(self.location, 'era5', 'stats', 'stds.npy'))[:, None, None]

    def _get_hrrr_channel_names(self):

        base_hrrr_channels = self.hrrr_channels
        kept_hrrr_channels = base_hrrr_channels

        if len(self.params.exclude_channels) > 0:
            kept_hrrr_channels = [
                x for x in base_hrrr_channels if x not in self.params.exclude_channels
            ]

        return base_hrrr_channels, kept_hrrr_channels

    def _get_era5_channel_names(self):

        return list(self.era5_channels.values)

    def _get_files_stats(self):
        """
        Scan directories and extract metadata for ERA5 and HRRR
        """
        # HRRR parsing

        self.hrrr_paths = glob.glob(
            os.path.join(self.location, self.conus_dataset_name, "**", "????.zarr"),
            recursive=True,
        )
        print("list of all hrrr paths: ", self.hrrr_paths)
        self.hrrr_paths = sorted(
            self.hrrr_paths, key=lambda x: int(os.path.basename(x).replace(".zarr", ""))
        )
        if self.train:
            # keep only years specified in the params.train_years list
            self.hrrr_paths = [
                x
                for x in self.hrrr_paths
                if int(os.path.basename(x).replace(".zarr", ""))
                in self.params.train_years
            ]
            self.years = [
                int(os.path.basename(x).replace(".zarr", "")) for x in self.hrrr_paths
            ]
        else:
            # keep only years specified in the params.valid_years list
            self.hrrr_paths = [
                x
                for x in self.hrrr_paths
                if int(os.path.basename(x).replace(".zarr", ""))
                in self.params.valid_years
            ]
            self.years = [
                int(os.path.basename(x).replace(".zarr", "")) for x in self.hrrr_paths
            ]

        print("list of all hrrr paths after filtering: ", self.hrrr_paths)
        self.n_years = len(self.hrrr_paths)
        self.n_samples_total = self.compute_total_samples()

        # sort paths by year
        years = [int(os.path.basename(x).replace(".zarr", "")) for x in self.hrrr_paths]
        print("years: ", years)
        print("self.years: ", self.years)
        assert (
            years == self.years
        ), "Number of years for ERA5 in %s and HRRR in %s must match" % (
            os.path.join(self.location, "era5/*.zarr"),
            os.path.join(self.location, "hrrr/*.zarr"),
        )
        with xr.open_zarr(self.hrrr_paths[0], consolidated=True) as ds:
            self.hrrr_channels = list(ds.channel.values)
            self.hrrr_lat = ds.latitude[
                0 : self.params.hrrr_img_size[0], 0 : self.params.hrrr_img_size[1]
            ]
            self.hrrr_lon = ds.longitude[
                0 : self.params.hrrr_img_size[0], 0 : self.params.hrrr_img_size[1]
            ]
            self.era5_lat_window, self.era5_lon_window = self._construct_era5_window()
        self.ds_hrrr = [
            xr.open_zarr(self.hrrr_paths[i], consolidated=True, mask_and_scale=False)
            for i in range(self.n_years)
        ]

        if self.boundary_padding_pixels > 0:

            self.era5_lat, self.era5_lon = self._construct_era5_window()

        else:

            self.era5_lat = self.hrrr_lat
            self.era5_lon = self.hrrr_lon

    def __len__(self):
        return self.n_samples_total

    def crop_to_fit(self, x):
        """
        Crop HRRR to get nicer dims
        """
        return x[:, 0 : self.params.hrrr_img_size[0], 0 : self.params.hrrr_img_size[1]]

    def to_datetime(self, date):

        timestamp = (date - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(
            1, "s"
        )
        return datetime.utcfromtimestamp(timestamp)

    def compute_total_samples(self):
        """
        Loop through all years and count the total number of samples
        """

        if self.params.localtest:

            all_datetimes_ = []

            for year in self.years:
                times = xr.open_zarr(
                    os.path.join(
                        self.location,
                        self.conus_dataset_name,
                        self.path_suffix,
                        str(year) + ".zarr",
                    ),
                    consolidated=True,
                ).time.values
                all_datetimes_.extend(times)

            all_datetimes = [self.to_datetime(x) for x in all_datetimes_]
            all_datetimes = all_datetimes[:-2]

        else:

            first_year = sorted(self.years)[0]
            last_year = sorted(self.years)[-1]
            if first_year == 2018:
                first_sample = datetime(
                    year=first_year, month=8, day=1, hour=1, minute=0, second=0
                )  # marks transition of hrrr model version
                logging.info("First sample is {}".format(first_sample))
            else:
                first_sample = datetime(
                    year=first_year, month=1, day=1, hour=0, minute=0, second=0
                )
                logging.info("First sample is {}".format(first_sample))
            last_sample = datetime(
                year=last_year, month=12, day=31, hour=23, minute=0, second=0
            )
            all_datetimes = [
                first_sample + timedelta(hours=x)
                for x in range(
                    int((last_sample - first_sample).total_seconds() / 3600) + 1
                )
            ]

        missing_samples = np.load(
            os.path.join(self.location, "missing_samples.npy"), allow_pickle=True
        )

        missing_samples = set(missing_samples)  # hash for faster lookup

        self.valid_samples = [x for x in all_datetimes if (x not in missing_samples)]

        logging.info(
            "Total datetimes in training set are {} of which {} are valid".format(
                len(all_datetimes), len(self.valid_samples)
            )
        )

        return len(self.valid_samples)

    def _normalize_era5(self, img):
        # if self.normalize:
        #     img -= self.means_era5
        #     img /= self.stds_era5
        return torch.as_tensor(img)

    def _get_era5(self, ts):
        """
        Retrieve ERA5 samples from zarr files
        """

        era5_handle = self._get_ds_handles(self.ds_era5, self.era5_paths, ts)

        era5_field = (
            era5_handle.sel(time=ts, channel=self.era5_channels)
            .interp(latitude=self.era5_lat, longitude=self.era5_lon)
            .data.values
        )

        era5_field = self._normalize_era5(era5_field)

        return era5_field

    def _normalize_hrrr(self, img):
        tp_index = self.hrrr_channels.index("tp")
        img[tp_index] = np.log(img[tp_index] + 0.0001)
        if self.normalize:
            img -= self.means_hrrr
            img /= self.stds_hrrr

        if len(self.params.exclude_channels) > 0:
            drop_channel_indices = [
                self.hrrr_channels.index(x) for x in self.params.exclude_channels
            ]
            img = np.delete(img, drop_channel_indices, axis=0)
        if self.params.crop_hrrr:
            img = self.crop_to_fit(img)

        return torch.as_tensor(img)

    def _get_hrrr(self, ts):
        """
        Retrieve HRRR samples from zarr files
        """
        hrrr_handle = self._get_ds_handles(self.ds_hrrr, self.hrrr_paths, ts)

        hrrr_field = hrrr_handle.sel(time=ts).HRRR.values

        hrrr_field = self._normalize_hrrr(hrrr_field)

        return hrrr_field

    def __getitem__(self, global_idx):
        """
        Return data as a dict (so we can potentially add extras, metadata, etc if desired
        """
        time_index = self._global_idx_to_datetime(global_idx)
        hrrr_sample = self._get_hrrr(time_index)

        return {
            "hrrr": hrrr_sample,
            #'time':(np.datetime64(time_pair[0]), np.datetime64(time_pair[1]))
        }

    def _global_idx_to_datetime(self, global_idx):
        """
        Parse a global sample index and return the input/target timstamps as datetimes
        """
        time_index = self.valid_samples[global_idx]

        return time_index

    def _get_ds_handles(self, handles, paths, ts):
        """
        Return handles for the appropriate year
        """
        year = ts.year
        year_idx = self.years.index(year)
        ds_handle = handles[year_idx]

        return ds_handle

    def _construct_era5_window(self):
        """
        Build custom indexing window to subselect HRRR region from ERA5
        """

        logging.info(
            "Constructing ERA5 window, extending HRRR domain by {} pixels in each direction".format(
                self.boundary_padding_pixels
            )
        )
        from pyproj import Proj

        proj_params = {
            "a": 6371229,
            "b": 6371229,
            "proj": "lcc",
            "lon_0": 262.5,
            "lat_0": 38.5,
            "lat_1": 38.5,
            "lat_2": 38.5,
        }  # from HRRR source grib file
        proj = Proj(proj_params)
        x, y = proj(self.hrrr_lon.values, self.hrrr_lat.values)

        dx = round(
            x[0, 1] - x[0, 0]
        )  # grid spacing (this is pretty darn close to constant)
        dy = round(y[1, 0] - y[0, 0])

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        boundary_padding_pixels = self.boundary_padding_pixels

        new_x_min = x_min - boundary_padding_pixels * dx
        new_x_max = x_max + boundary_padding_pixels * dx
        new_y_min = y_min - boundary_padding_pixels * dy
        new_y_max = y_max + boundary_padding_pixels * dy

        new_x = np.arange(new_x_min, new_x_max, dx)

        new_y = np.arange(new_y_min, new_y_max, dy)

        new_x, new_y = np.meshgrid(new_x, new_y)

        new_lons, new_lats = proj(new_x, new_y, inverse=True)

        added_pixels_x = new_x.shape[1] - self.hrrr_lon.shape[1]
        added_pixels_y = new_x.shape[0] - self.hrrr_lon.shape[0]

        logging.info(
            "Added {} pixels in x, {} pixels in y".format(
                added_pixels_x, added_pixels_y
            )
        )

        assert (
            added_pixels_x == 2 * boundary_padding_pixels
        ), "requested {} padding pixels but got {} in x due to interpolation round off errors".format(
            boundary_padding_pixels, added_pixels_x
        )
        assert (
            added_pixels_y == 2 * boundary_padding_pixels
        ), "requested {} padding pixels but got {} in y due to interpolation round off errors".format(
            boundary_padding_pixels, added_pixels_y
        )

        era5_lats = xr.DataArray(new_lats, dims=("y", "x"))
        era5_lons = xr.DataArray(new_lons, dims=("y", "x"))
        coords = {"latitude": era5_lats, "longitude": era5_lons}
        era5_lats = era5_lats.assign_coords(coords)
        era5_lons = era5_lons.assign_coords(coords)
        return era5_lats, era5_lons
