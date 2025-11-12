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

from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
    interp,
)
import pandas as pd
from physicsnemo.utils.zenith_angle import cos_zenith_angle_from_timestamp, cos_zenith_angle
import random
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


class SolarDataset(DownscalingDataset):
    """
    Paired dataset object serving time-synchronized pairs of ERA5 and Envision-wind samples
    Expects data to be stored under directory specified by 'location'
        ERA5 under <root_dir>/ERA5/
        Solar under <root_dir>/H08/
    Within ERA5 directory, there should be one zarr file per year containing the data of interest.
    Within H08 directory, there should be many nc file per hour containing the data of interest.
    """

    def __init__(
        self,
        *,
        data_path: str,
        stats_path: str,
        input_variables: Union[List[str], None] = None,
        output_variables: Union[List[str], None] = None,
        invariant_variables: Union[List[str], None] = ("dem"),
        train: bool = True,
        normalize: bool = True,
        train_years: Iterable[int] = (2022,),
        valid_years: Iterable[int] = (2021,),
        sample_shape: Tuple[int, int] = [-1, -1],
        ds_factor: int = 1,
        shard: bool = False,
        overfit: bool = False,
        use_all: bool = False,
        normal_way: str = "min_max",
        generating: bool = False,
        stride_train: int = 80,
        stride_gen: int = 160,
        window_size: int = 320,
        solar: bool = True
    ):
        dask.config.set(
            scheduler="synchronous"
        )  # for threadsafe multiworker dataloaders
        self.data_path = data_path
        self.train = train
        self.normalize = normalize
        self.output_variables = output_variables
        self.input_variables = input_variables
        self.invariant_variables = invariant_variables 
        self.train_years = list(train_years)
        self.valid_years = list(valid_years)
        self.normal_way = normal_way
        self.solar = solar
        
        self.sample_shape = sample_shape
        self.ds_factor = ds_factor
        self.shard = shard
        self.use_all = use_all
        self.output_variables_load = copy.deepcopy(output_variables)
        
        self._get_files_stats()
        self.overfit = overfit
        
        self.window_size = window_size
        self.generating = generating
        if not self.generating:
            self.stride_train = stride_train
            self.windows = self.get_windows(stride=self.stride_train)
            logging.info(f"The num of training windows is: {len(self.windows)}")
        else:
            self.stride_gen = stride_gen
        self.era5_input_variables = self.input_variables


        with open(stats_path, "r") as f:
            stats = json.load(f)
        
        (self.input_center, self.input_scale) = _load_stats(
            stats, self.input_variables, "era5", self.normal_way
        )
        (self.output_center, self.output_scale) = _load_stats(
            stats, self.output_variables, "h08", self.normal_way
        )
        if self.invariant_variables is not None:
            (self.inv_center, self.inv_scale) = _load_stats(
            stats, self.invariant_variables, "inv", self.normal_way
            )
            self.invs = self._get_inv()
        
        lon_input_grid,lat_input_grid = np.meshgrid(self.era5_lon, self.era5_lat)
        self.lon_output_grid,self.lat_output_grid = np.meshgrid(self.H08_lon, self.H08_lat)
        self._interpolator = interp.LatLonInterpolation(
                lat_input_grid,
                lon_input_grid,
                self.lat_output_grid,
                self.lon_output_grid,
            )
        
    def _apply_window(self,x,window):
        ((y_start, y_end), (x_start, x_end)) = window
        """
        x: [,,, hight,width]
        Crop the data to the H08-window size
        """
        if len(x.shape)==2:
            return x[y_start:y_end,x_start:x_end]
        elif len(x.shape)==3:
            return x[:,y_start:y_end,x_start:x_end]
        
    def _get_files_stats(self):
        """
        Scan directories and extract metadata for H08 and ERA5

        We assume: 
        - ERA5 files are at self.data_path/LRdata/ with name era5_YYYY_opt.zarr
        - HR files are at self.data_path/HRdata/ with folder name H08_YYYY_hourly 
        """

        # training or validating, different files will be read
        years_to_use = self.train_years if self.train else self.valid_years
        logging.info(f"years_to_use: {years_to_use}")
        # ERA5 parsing
        self.ds_era5 = {}
        LR_paths_all = glob.glob(
            os.path.join(self.data_path, "LRdata", "era5_*_opt.zarr")
        )
        logging.info(f"LR_paths_all: {LR_paths_all}")
        # Get years from paths. e.g. '.../era5_2021_opt.zarr' -> '2021'
        era5_years = [os.path.basename(p).split('.')[0].split('_')[1] for p in LR_paths_all]
        self.era5_paths = dict(zip(era5_years, LR_paths_all))
        logging.info(f"era5_years: {era5_years}")
        # Only keep the years to be used
        self.era5_paths = {
            year: path
            for (year, path) in self.era5_paths.items()
            if int(year) in years_to_use
        }
    
        # Use the first year to load metadata
        first_era5_key = sorted(self.era5_paths.keys())[0]
        with xr.open_zarr(self.era5_paths[first_era5_key], consolidated=True) as ds:
            
            self.era5_lat = ds['latitude'].values
            self.era5_lon = ds['longitude'].values

        # H08 parsing
        self.ds_H08 = {}
        HR_paths_all = glob.glob(
            os.path.join(self.data_path, "HRdata", "H08_*_hourly")
        )
    
        H08_years = [os.path.basename(p).split('.')[0].split('_')[1] for p in HR_paths_all]
        self.H08_paths = dict(zip(H08_years, HR_paths_all))
        self.H08_paths = {
            year: path
            for (year, path) in self.H08_paths.items()
            if int(year) in years_to_use
        }
        
        first_H08_key = sorted(self.H08_paths.keys())[0] #folds
        #the first path self.H08_paths[first_H08_key]
        nc_file = glob.glob(os.path.join(self.H08_paths[first_H08_key], '*.nc'))[0]
        logging.info(f"We achieve the lat/lon from {nc_file}")
        with xr.open_dataset(nc_file) as ds:
            self.H08_lat = ds['latitude'].values
            self.H08_lon = ds['longitude'].values

        # Get all years
        self.years = set([int(key) for key in self.H08_paths.keys()])
        self.n_samples_total = self.compute_total_samples()

    def __len__(self):
        return len(self.valid_samples)-1


    def compute_total_samples(self):

        # count the total number of samples from valid_time of H08 files
        all_datetimes = set()
        # Loop self.H08_paths.values() from _get_files_stats
        for year, path in self.H08_paths.items():
            logging.info(f"Reading {year} H08 files: {path}")
            nc_files = glob.glob(os.path.join(path, 'H08_*.nc'))
            for file_path in nc_files:
                filename = os.path.basename(file_path)
                datetime_str = '_'.join(filename.split('_')[1:3])
                datetime_obj = np.datetime64(datetime.strptime(datetime_str, '%Y%m%d_%H%M'))
                all_datetimes.add(datetime_obj)
        self.valid_samples = sorted(list(all_datetimes))
    
        logging.info(
            "Scan done. We have {} samlpes".format(len(self.valid_samples))
        )
        logging.info(f"The first time: {self.valid_samples[0]}")
        logging.info(f"The last time: {self.valid_samples[-1]}")

        # prepare data for distributed training 
        if self.shard:
            dist_manager = DistributedManager()
            self.valid_samples = np.array_split(
                self.valid_samples, dist_manager.world_size
            )[dist_manager.rank]
            logging.info(
                f"(Rank {dist_manager.rank}) "
                f"has {len(self.valid_samples)} samples"
            )

        return len(self.valid_samples)

    def normalize_input(self, x):
        x = x.astype(np.float32)
        if self.normalize:
            x -= self.input_center
            x /= self.input_scale
        return x

    def denormalize_input(self, x):
        x = x.astype(np.float32)
        if self.normalize:
            x *= self.input_scale
            x += self.input_center
        return x

    def normalize_output(self, x):
        x = x.astype(np.float32)
        if self.normalize:
            x -= self.output_center
            x /= self.output_scale
        return x

    def denormalize_output(self, x):
        x = x.astype(np.float32)
        
        if self.normalize:
            x *= self.output_scale
            x += self.output_center
        return x
    

    def _interp(self,x):
        
        x = torch.from_numpy(x).unsqueeze(0)
        x = self._interpolator(x.float()).squeeze(0).numpy()
        return x
    def _get_inv(self):
        file_path = os.path.join(self.data_path, "dem.nc")
        
        ds = xr.open_dataset(file_path)
        invs = []
        for inv in self.invariant_variables:
            invs.append(ds[inv].values)
        invs = np.stack(invs)
        invs = (invs - self.inv_center)/self.inv_scale
        
        return invs
    def _get_era5(self, ts):
        """
        Retrieve ERA5 samples from zarr files given valid_time
        """
        year = ts.astype('datetime64[Y]').astype(int) + 1970
        year_str = str(year)
        
        #cache the handle
        if year_str not in self.ds_era5:
            era5_path = self.era5_paths[year_str]
            self.ds_era5[year_str] = xr.open_zarr(era5_path, consolidated=True)
        #get the handle
        era5_handle = self.ds_era5[year_str]
        
        era5_field = []
        for var in self.input_variables:
            era5_field.append(era5_handle[var].sel(valid_time=ts,method='nearest').values)
        era5_field = np.stack(era5_field)

        if len(era5_field.shape) == 4:
            era5_field = era5_field[0]

        era5_field = self._interp(era5_field)
        
        era5_field = self.normalize_input(era5_field)

        return era5_field

    def _get_H08(self, ts):
        """
        Retrieve H08 samples from nc files given valid_time
        """
        
        year = ts.astype('datetime64[Y]').astype(int) + 1970
        year_str = str(year)
        #ts --> nc_filename
        H08_path = self.H08_paths[year_str]
        ts_pd = pd.to_datetime(ts)
        datetime_str = ts_pd.strftime('%Y%m%d_%H%M')
        filename = f"H08_{datetime_str}_hourly.nc"
        file_path = os.path.join(H08_path, filename)
        H08_handle = xr.open_dataset(file_path)
        H08_field = []
        for var in self.output_variables:
            H08_field.append(H08_handle[var].values)
        H08_field = np.stack(H08_field)

        if len(H08_field.shape) == 4:
            H08_field = H08_field[0]
        H08_field = self.normalize_output(H08_field)
        return H08_field

    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data (same for input and output)."""
        
        return (self.window_size, self.window_size)

    def compute_sza(self, ts):
        """Compute solar zenith angle for given coordinates.
        """
        grid = np.meshgrid(self.H08_lon, self.H08_lat)
        lon, lat = grid[0].reshape(-1), grid[1].reshape(-1)

        pd_ts = pd.to_datetime(ts)
        yy, mm, dd, hh = pd_ts.year, pd_ts.month, pd_ts.day, pd_ts.hour
        
        zeith_arr = []
        for miint in range(6):
            ztime = datetime(yy, mm, dd, hh, miint * 10, 0)
            zeith = cos_zenith_angle(ztime, lon, lat).reshape((len(self.H08_lat),len(self.H08_lon)))
            
            zeith_arr.append(zeith)

        zeith = np.stack(zeith_arr)
        
        return zeith
    
    def get_windows(self, stride=8):
        window_size = 320 #self.image_shape[0]
        height, width = len(self.H08_lat),len(self.H08_lon) 
        
        if window_size > height or window_size > width:
            raise ValueError("window_size cannot be larger than the panorama dimensions")
        h_starts = list(range(0, height - window_size, stride))
        w_starts = list(range(0, width - window_size, stride))
    
        if (height - window_size) not in h_starts:
            h_starts.append(height - window_size)
        
        if (width - window_size) not in w_starts:
            w_starts.append(width - window_size)
        
        windows = []
        for h_s in h_starts:
            for w_s in w_starts:
                h_e = h_s + window_size
                w_e = w_s + window_size
                windows.append((h_s, h_e, w_s, w_e))
            
        return windows

    def __getitem__(self, global_idx):
        """Return a tuple of:
        - H08_field: High-resolution H08 output data
        - era5_field: Low-resolution ERA5 input data (interpolated)
        - lead_time_label: Lead time
        """
        time_index = self._global_idx_to_datetime(global_idx)
        
        H08_sample = self._get_H08(time_index)
        era5_sample_T = self._get_era5(time_index)
        time_index_1 = self._global_idx_to_datetime(global_idx+1)
        era5_sample_T_1 = self._get_era5(time_index_1)
        era5_sample = np.stack([era5_sample_T, era5_sample_T_1], axis=1)
        C,H,W = era5_sample_T.shape
        era5_sample = era5_sample.reshape(C*2, H, W) 
        zeith = self.compute_sza(time_index)
        
        era5_sample = np.concatenate([era5_sample,zeith],axis=0)
        
        if np.isnan(era5_sample).any() or np.isnan(H08_sample).any():
            logging.info(f"We find nan in sample at {time_index}")
        torch.cuda.nvtx.range_pop()
        if self.invariant_variables is not None:
            img_lr = np.concatenate([era5_sample,self.invs],axis=0)
        else:
            img_lr = era5_sample
        
        if not self.generating:
            #when training, we randomly select a window
            idx = random.randint(0,len(self.windows)-1)
            window = ((self.windows[idx][0], self.windows[idx][1]),(self.windows[idx][2], self.windows[idx][3]))
            
            img_lr = self._apply_window(img_lr,window)
            H08_sample = self._apply_window(H08_sample,window)
        else:
            #when generating, we return all sliding windows
            window = self.get_windows(stride=160)
            logging.info(f"window0:{window[0]}")
            logging.info(f"windows:{window}")
            
        pd_ts = pd.to_datetime(time_index)
        yy, mm, dd, hh = pd_ts.year, pd_ts.month, pd_ts.day, pd_ts.hour
        
        return H08_sample, img_lr, window, (yy, mm, dd, hh)

    def _global_idx_to_datetime(self, global_idx):
        """
        Parse a global sample index and return the input/target timstamps as datetimes
        """
        return self.valid_samples[global_idx]

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
        return self.H08_lat #if self.train else self.crop_to_fit(self.H08_lat)

    def longitude(self):
        return self.H08_lon #if self.train else self.crop_to_fit(self.H08_lon)
    
    def input_channels(self):
        era5_variables = self.input_variables
        self.era5_input = era5_variables 
        era5_variables_1 = [s + '_1' for s in self.input_variables] 
        variables = era5_variables +  era5_variables_1 + ['zeith0','zeith1','zeith2','zeith3','zeith4','zeith5']
        #[ var for pair in zip(era5_variables, era5_variables_1) for var in pair] + ['zeith0','zeith1','zeith2','zeith3','zeith4','zeith5']
        if self.invariant_variables is not None:
            variables += self.invariant_variables
            return [ChannelMetadata(name=n) for n in variables]
        else:
            return [ChannelMetadata(name=n) for n in variables]
        
    def output_channels(self):
        variables = self.output_variables + [s + '_1' for s in self.output_variables]+ [s + '_2' for s in self.output_variables]+ [s + '_3' for s in self.output_variables]+ [s + '_4' for s in self.output_variables]+ [s + '_5' for s in self.output_variables]
        return [ChannelMetadata(name=n) for n in variables]
    
    def time(self):
        return self.valid_samples




def _load_stats(stats, variables, group, normal_way = "min_max"):

    if normal_way == "min_max":
        center = np.array([stats[group][v]["center"] for v in variables])[:, None, None].astype(
            np.float32
        )
        scale = np.array([stats[group][v]["scale"] for v in variables])[:, None, None].astype(
            np.float32
        )
    elif normal_way == "mean_std":
        center = np.array([stats[group][v]["mean"] for v in variables])[:, None, None].astype(
            np.float32
        )
        scale = np.array([stats[group][v]["std"] for v in variables])[:, None, None].astype(
            np.float32
        )

    return (center, scale)
