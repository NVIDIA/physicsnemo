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
import sys
import os
import gc

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from sda.score import *
from sda.utils import *

import xarray as xr
from training.utils.diffusions.networks import get_preconditioned_architecture
import torch
import xskillscore as xs
from configs.ord_ph import (
    path_to_hrrr,
    path_to_pretrained,
    station_locations,
    val_station_path,
    isd_path,
)

hr = xr.open_zarr(path_to_hrrr, mask_and_scale=False)

ngpu = torch.cuda.device_count()


def denorm(x):
    x *= stds[:, np.newaxis, np.newaxis]
    x += means[:, np.newaxis, np.newaxis]
    x[..., 2, :, :] = np.exp(x[..., 2, :, :] - 1e-4)
    return x


models = {}
for rank in range(ngpu):
    model = get_preconditioned_architecture(
        name="ddpmpp-cwb-v0",
        resolution=128,
        target_channels=3,
        conditional_channels=0,
        label_dim=0,
    )
    # print(model)
    state = torch.load(path_to_pretrained)
    model.load_state_dict(state, strict=False)
    model.to(device=rank)
    models[rank] = model
# model = model.to(device)

u10_mean, u10_std = -0.262, 2.372
v10_mean, v10_std = 0.865, 4.115
logtp_mean, logtp_std = -8.117, 2.489
means, stds = np.array([u10_mean, v10_mean, logtp_mean]), np.array(
    [u10_std, v10_std, logtp_std]
)

ds_regrid = xr.open_dataset(isd_path)

stat_loc = xr.open_dataarray(station_locations)

valid = np.load(val_station_path)
job = os.getenv("SLURM_ARRAY_TASK_ID")
# job = 0
num_leave = int(job)
valid = valid[:num_leave]

bool_array = stat_loc.values.astype(bool)
for indices in valid:
    bool_array[indices[0], indices[1]] = False
tune = np.zeros_like(bool_array)
for indices in valid:
    tune[indices[0], indices[1]] = True


def process_day(day):
    day = int(day)
    index = day % ngpu

    target_time = slice(day * 24, (day + 1) * 24 - 12)
    u10 = ds_regrid.isel(DATE=target_time).u10.values
    v10 = ds_regrid.isel(DATE=target_time).v10.values
    tp = np.log(ds_regrid.isel(DATE=target_time).tp.values + 0.0001)
    obs = np.array([u10, v10, tp])
    obs -= means[:, np.newaxis, np.newaxis, np.newaxis]
    obs /= stds[:, np.newaxis, np.newaxis, np.newaxis]
    obs = obs.transpose(3, 0, 2, 1)
    obs = torch.tensor(obs)  # .to(device=index)

    u10 = hr.isel(time=target_time).sel(channel="10u").HRRR.values
    v10 = hr.isel(time=target_time).sel(channel="10v").HRRR.values
    tp = np.log(hr.isel(time=target_time).sel(channel="tp").HRRR.values + 0.0001)
    hrrr = np.array([u10, v10, tp])
    hrrr -= means[:, np.newaxis, np.newaxis, np.newaxis]
    hrrr /= stds[:, np.newaxis, np.newaxis, np.newaxis]
    hrrr = hrrr.transpose(1, 0, 2, 3)
    hrrr = torch.tensor(hrrr)

    inf_obs = obs.where(torch.tensor(np.tile(bool_array, (len(obs), 3, 1, 1))), np.nan)
    val_obs = obs.where(torch.tensor(np.tile(tune, (len(obs), 3, 1, 1))), np.nan)

    mask = ~np.isnan(inf_obs).bool()

    def A(x):
        return x[mask]

    with torch.cuda.device(index):
        y_star = A(inf_obs)
        sde = VPSDE(  # from this we use sample()
            GaussianScore_from_denoiser(  # from this, sample() uses VPSDE.eps() which is GaussianScore_from_denoiser.forward()
                y_star,
                A=A,
                std=0.1,
                gamma=0.001,
                sde=VPSDE_from_denoiser(
                    models[index], shape=()
                ),  # which calls VPSDE_from_denoiser.eps_from_denoiser() which calls the network
            ),  # .to(device=index),
            shape=inf_obs.shape,
        ).cuda()

    x = sde.sample(steps=64, corrections=2, tau=0.3, makefigs=False).cpu()
    x = denorm(x)
    val_obs = denorm(val_obs)
    hrrr = denorm(hrrr)

    mse = np.nanmean((x - val_obs) ** 2, axis=(0, 2, 3))
    mse_hr = np.nanmean((hrrr - val_obs.numpy()) ** 2, axis=(0, 2, 3))
    obs, hrrr, inf_obs, val_obs, x, sde = None, None, None, None, None, None
    gc.collect()
    np.save(f"MSE_OBS_denorm_{num_leave}_{day:05d}", mse)
    np.save(f"MSE_HRR_denorm_{num_leave}_{day:05d}", mse_hr)


ngpu = torch.cuda.device_count()

start = 0
stop = 364

with ThreadPoolExecutor(ngpu) as pool:
    lazy_jobs = pool.map(process_day, range(start, stop))
    list(lazy_jobs)  # explicitly run jobs
