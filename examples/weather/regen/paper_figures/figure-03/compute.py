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

import numpy as np
import torch
import xarray as xr
from sda.score import GaussianScore_from_denoiser, VPSDE_from_denoiser, VPSDE
from training.utils.diffusions.networks import get_preconditioned_architecture
from configs.ord_ph import (
    path_to_hrrr,
    path_to_pretrained,
    station_locations,
    val_station_path,
)

u10_mean, u10_std = -0.262, 2.372
v10_mean, v10_std = 0.865, 4.115
logtp_mean, logtp_std = -8.117, 2.489


means, stds = (
    np.array([u10_mean, v10_mean, logtp_mean]),
    np.array([u10_std, v10_std, logtp_std]),
)


torch.cuda.is_available()


device = torch.device("cuda:0")

model = get_preconditioned_architecture(
    name="ddpmpp-cwb-v0",
    resolution=128,
    target_channels=3,
    conditional_channels=0,
    label_dim=0,
)
state = torch.load(path_to_pretrained)
model.load_state_dict(state, strict=False)
model = model.to(device)

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
print(hr.isel(time=target_time).time)

# ## Inpaint


stat_loc = xr.open_dataarray(station_locations)
bool_array = stat_loc.values.astype(bool)
bool_arrayt = torch.tensor(np.tile(bool_array, (3, 1, 1)))


bool_arrayt.shape


# hrrr=hrrr.tile(3,1,1,1)
mask = np.ones_like(hrrr)
mask[:, 1, ...] = 0
# mask[:,0,...]=0
mask = torch.tensor(mask).bool()


def A(x):
    """
    Mask the observations to the valid locations.
    """
    return x[mask]


# y_star = torch.normal(A(hrrr), 0.1)
y_star = A(hrrr)


y_star.shape


sde = VPSDE(  # from this we use sample()
    GaussianScore_from_denoiser(  # from this, sample() uses VPSDE.eps() which is GaussianScore_from_denoiser.forward()
        y_star,
        A=A,
        std=0.1,
        gamma=0.01,
        sde=VPSDE_from_denoiser(
            model, shape=()
        ),  # which calls VPSDE_from_denoiser.eps_from_denoiser() which calls the network
    ),
    shape=hrrr.shape,
).cuda()

x = sde.sample(steps=256, corrections=10, tau=0.3, makefigs=False).cpu()


# torch.save(x,'gen_10v_stde-1_gammae-2_steps256_corrections10_tau3e-1')


# y=torch.load('gen_10v_stde-1_gammae-2_steps256_corrections10_tau3e-1')


def denorm(x):
    x *= stds[:, np.newaxis, np.newaxis]
    # print(x.shape)
    x += means[:, np.newaxis, np.newaxis]
    x[..., 2, :, :] = np.exp(x[..., 2, :, :] - 1e-4)
    return x


dhrrr = denorm(hrrr)
dx = denorm(x)
lats = hr.sel(channel="10v").isel(time=234).latitude
lons = hr.sel(channel="10v").isel(time=234).longitude
