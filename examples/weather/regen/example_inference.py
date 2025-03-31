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
#!/usr/bin/env python
# coding: utf-8


import sys

sys.path.append("./training")  # for loading pickled models which need training/utils
import numpy as np
import torch
from sda.score import GaussianScore_from_denoiser, VPSDE_from_denoiser, VPSDE
from training.utils.diffusions.networks import get_preconditioned_architecture
import torch
import xarray as xr
from scipy.stats import norm
import matplotlib

matplotlib.rcParams.update({"font.size": 14})
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from configs.base import (
    path_to_model_state,
    path_to_hrrr,
    isd_path,
    station_locations,
    path_to_pretrained,
)
import scipy.interpolate


def bounds():
    dy = lat_g[0] - lat_g[1]
    dx = lon_g[1] - lon_g[0]
    north_west = [lat_g[0] + dy / 2, lon_g[0] - dx / 2]
    south_east = [lat_g[-1] - dy / 2, lon_g[-1] + dx / 2]
    return [north_west, south_east]


def interpolate(x):
    x = np.asarray(x)

    values = x.ravel()
    values_g = scipy.interpolate.griddata(
        (lat.ravel(), lon.ravel()), values, (lat_g[:, None], lon_g)
    )
    return values_g


u10_mean, u10_std = -0.262, 2.372
v10_mean, v10_std = 0.865, 4.115
logtp_mean, logtp_std = -8.117, 2.489


means, stds = np.array([u10_mean, v10_mean, logtp_mean]), np.array(
    [u10_std, v10_std, logtp_std]
)


print("cuda available? ", torch.cuda.is_available())


## Load Obs data


ds_regrid = xr.open_dataset(isd_path)


# ## Initialize Denoiser


device = torch.device("cuda:0")

# load pretrained model
model = get_preconditioned_architecture(
    name="ddpmpp-cwb-v0",
    resolution=128,
    target_channels=3,
    conditional_channels=0,
    label_dim=0,
    spatial_embedding=None,
)

modelpath = path_to_model_state
load_state_dict = modelpath.endswith(".pth")

state = torch.load(modelpath)
if load_state_dict:
    model.load_state_dict(state, strict=True)
else:
    model = state["net"]
model.eval().to(device=device).to(memory_format=torch.channels_last)


# used for interpolation
hr = xr.open_zarr(path_to_hrrr, mask_and_scale=False)
lat = hr.latitude.values
lon = hr.longitude.values

lat_g = np.linspace(lat.max(), lat.min(), 256)
lon_g = np.linspace(lon.min(), lon.max(), 256)


print("Running example inference")


def do_inference(time, n_steps):
    idx = ds_regrid.indexes["DATE"]
    (time_loc,) = idx.get_indexer([time])

    target_time = slice(time_loc, time_loc + 1)
    u10 = ds_regrid.isel(DATE=target_time).u10.values
    v10 = ds_regrid.isel(DATE=target_time).v10.values
    tp = np.log(ds_regrid.isel(DATE=target_time).tp.values + 0.0001)
    obs = np.array([u10, v10, tp])
    obs -= means[:, np.newaxis, np.newaxis, np.newaxis]
    obs /= stds[:, np.newaxis, np.newaxis, np.newaxis]
    obs = obs.transpose(3, 0, 2, 1)
    obs = torch.tensor(obs)
    mask = ~np.isnan(obs).bool()

    hr = xr.open_zarr(path_to_hrrr, mask_and_scale=False)
    target_time = slice(
        3530,
        3531,
    )
    u10 = hr.isel(time=target_time).sel(channel="10u").HRRR.values
    v10 = hr.isel(time=target_time).sel(channel="10v").HRRR.values
    tp = np.log(hr.isel(time=target_time).sel(channel="tp").HRRR.values + 0.0001)
    hrrr = np.array([u10, v10, tp])
    hrrr -= means[:, np.newaxis, np.newaxis, np.newaxis]
    hrrr /= stds[:, np.newaxis, np.newaxis, np.newaxis]
    hrrr = hrrr.transpose(1, 0, 2, 3)
    hrrr = torch.tensor(hrrr)

    # ## Inpaint

    stat_loc = xr.open_dataarray(station_locations)
    bool_array = stat_loc.values.astype(bool)
    bool_arrayt = torch.tensor(np.tile(bool_array, (3, 1, 1)))

    hrrr = hrrr.tile(3, 1, 1, 1)

    conc = obs
    print(conc.shape)

    mask = ~torch.isnan(obs)

    def A(x):
        return x[mask]

    y_star = A(conc)

    sde = VPSDE(  # from this we use sample()
        GaussianScore_from_denoiser(  # from this, sample() uses VPSDE.eps() which is GaussianScore_from_denoiser.forward()
            y_star,
            A=A,
            std=0.1,
            gamma=0.001,
            sde=VPSDE_from_denoiser(
                model, shape=()
            ),  # which calls VPSDE_from_denoiser.eps_from_denoiser() which calls the network
        ),
        shape=conc.shape,
    ).cuda()

    lat = hr.latitude.values
    lon = hr.longitude.values

    def denorm(x):
        x *= stds[:, np.newaxis, np.newaxis]
        x += means[:, np.newaxis, np.newaxis]
        x[..., 2, :, :] = np.exp(x[..., 2, :, :] - 1e-4)
        return x

    sample = (
        sde.sample(steps=n_steps, corrections=2, tau=0.3, makefigs=False).cpu().numpy()
    )

    return lat, lon, denorm(sample), denorm(obs), denorm(hrrr)


def main():
    import datetime
    import plotting

    time = datetime.datetime(2017, 5, 28, 3)
    lat, lon, pred, obs, hrrr = do_inference(time, n_steps=16)
    plotting.plot_sample(pred, obs, lat, lon)
    print(f"saving example outputs to output.png")
    plt.title(time.isoformat())
    plt.savefig("output.png")


if __name__ == "__main__":
    main()
