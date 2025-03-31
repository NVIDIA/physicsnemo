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
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys

import h5py
import numpy as np
import torch
from sda.score import GaussianScore_from_denoiser
import matplotlib.pyplot as plt

from utils import *
import PIL
import xarray as xr
import xskillscore as xs

from scipy.stats import norm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from configs.ord_ph import (
    path_to_hrrr,
    path_to_pretrained,
    station_locations,
    val_station_path,
    isd_path,
)
from training.utils.diffusions.networks import get_preconditioned_architecture
import torch

u10_mean, u10_std = -0.262, 2.372
v10_mean, v10_std = 0.865, 4.115
logtp_mean, logtp_std = -8.117, 2.489
tp_mean, tp_std = 0.14272778, 1.4051849


def denorm(x):
    x *= stds[:, np.newaxis, np.newaxis]
    x += means[:, np.newaxis, np.newaxis]
    x[..., 2, :, :] = np.exp(x[..., 2, :, :]) - 1e-4
    return x


import matplotlib

matplotlib.rcParams.update({"font.size": 14})

means, stds = np.array([u10_mean, v10_mean, logtp_mean]), np.array(
    [u10_std, v10_std, logtp_std]
)


torch.cuda.is_available()


ds_regrid = xr.open_dataset(isd_path)

target_time = slice(
    4700,
    4701,
)
u10 = ds_regrid.isel(DATE=target_time).u10.values
v10 = ds_regrid.isel(DATE=target_time).v10.values
tp = np.log(ds_regrid.isel(DATE=target_time).tp.values + 0.0001)
obs = np.array([u10, v10, tp])


obs -= means[:, np.newaxis, np.newaxis, np.newaxis]
obs /= stds[:, np.newaxis, np.newaxis, np.newaxis]


obs = obs.transpose(3, 0, 2, 1)
obs = torch.tensor(obs)


mask = ~np.isnan(obs).bool()

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


def A(x):
    """
    Mask the observations to the valid locations.
    """
    return x[mask]


y_star = A(obs)


class GaussianScore_from_denoiser_draw(GaussianScore_from_denoiser):
    """
    Draw the denoising process for a single variable at a single time.
    """

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        if self.detach:
            eps = self.sde.eps_from_denoiser(x, t, c)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            if not self.detach:
                eps = self.sde.eps_from_denoiser(x, t, c)
            print(t)
            x_ = self.sde.eps(x / mu, sigma / mu)  # (x - sigma * eps) / mu
            z = x_.clone().detach()
            fig = plt.figure(figsize=(11, 5))
            ax = fig.add_subplot(1, 2, 1, projection=projection)
            im = ax.pcolormesh(
                lons,
                lats,
                denorm(z.cpu().numpy())[0, 1],
                vmax=6,
                vmin=-6,
                cmap="RdBu",
                transform=ccrs.PlateCarree(),
                alpha=1,
            )
            addf(ax)
            cbar = fig.colorbar(im, ax=ax, shrink=0.7)
            cbar.set_label("10v [m/s]", size=12)
            np.save(f"denoising_process/other_time/x_denoised_t_{t}", z.cpu().numpy())
            ax = fig.add_subplot(1, 2, 2, projection=projection)
            y = x.clone().detach()
            im = ax.pcolormesh(
                lons,
                lats,
                denorm(y.cpu().numpy())[0, 1],
                vmax=6,
                vmin=-6,
                cmap="RdBu",
                transform=ccrs.PlateCarree(),
                alpha=1,
            )
            addf(ax)
            cbar = fig.colorbar(im, ax=ax, shrink=0.7)
            cbar.set_label("10v [m/s]", size=12)
            plt.show()
            np.save(f"denoising_process/other_time/x_noisy_t_{t}", y.cpu().numpy())
            err = self.y - self.A(x_)
            var = self.std**2 + self.gamma * (sigma / mu) ** 2

            log_p = -(err**2 / var).sum() / 2

        (s,) = torch.autograd.grad(log_p, x)

        return eps - sigma * s


hr = xr.open_zarr(path_to_hrrr, mask_and_scale=False)


plt.imshow(
    hr.HRRR.sel(channel="10v").isel(time=4700).values,
    vmax=6,
    vmin=-6,
    cmap="RdBu",
)

lats = hr.sel(channel="10v").isel(time=234).latitude
lons = hr.sel(channel="10v").isel(time=234).longitude


reader = shpreader.Reader("/home/pmanshausen/county_files/tl_2017_40_place.shp")

for k, i in enumerate(reader.records()):

    if i.attributes["NAME"] == "Oklahoma City":

        ok = i.geometry
    if i.attributes["NAME"] == "Tulsa":
        t = i.geometry
counties = list((ok, t))

COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

# %%
projection = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)


def addf(ax):
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAKES, alpha=0.8)
    ax.add_feature(
        cfeature.RIVERS.with_scale("10m"), alpha=0.8, edgecolor="gray", zorder=2
    )
    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="black", linewidth=2.5)
    ax.annotate(
        "Oklahoma City", (-97.9964, 35.7076), transform=ccrs.PlateCarree(), size=10
    )
    ax.annotate("Tulsa", (-96.05, 36.35), transform=ccrs.PlateCarree(), size=10)
    ax.add_feature(COUNTIES, facecolor="gray", edgecolor="none", linewidth=1, alpha=0.5)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=1,
        color="gray",
        alpha=0.3,
        linestyle="--",
        x_inline=False,
        y_inline=False,
    )


stat_loc = xr.open_dataarray("station_locations_on_grid.nc")
bool_array = stat_loc.values.astype(bool)

import glob

files_denoised = glob.glob("denoising_process/x_denoised_t_*")
files_noisy = glob.glob("denoising_process/x_noisy_t_*")
lats = np.load("figure_data/latitudes.npy")
lons = np.load("figure_data/longitudes.npy")
files_denoised.sort()
files_denoised.reverse()
files_noisy.sort()
files_noisy.reverse()


stat_loc = xr.open_dataarray("station_locations_on_grid.nc")
bool_array = stat_loc.values.astype(bool)
ps_obs = A(np.load(files_denoised[20])[:, 1])
ps_obs *= stds[1]
ps_obs += means[1]
ps_obs

latss = lats[bool_array]
lonss = lons[bool_array]
toplotn = ps_obs
plt.scatter(
    lonss,
    latss,
    c=toplotn,
    cmap="RdBu",
    marker="^",
    vmax=6,
    vmin=-6,
    edgecolor="black",
    s=50,
)

i = 32
projection = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)
fig = plt.figure(figsize=(27, 5))
ax = fig.add_subplot(1, 5, 1, projection=projection)
im = ax.pcolormesh(
    lons,
    lats,
    denorm(np.load(files_noisy[i]))[0, 1],
    vmax=6,
    vmin=-6,
    cmap="RdBu",
    transform=ccrs.PlateCarree(),
    alpha=1,
    rasterized=True,
)
addf(ax)
ax.set_title("Noisy state $x_t$")

ax = fig.add_subplot(1, 5, 2, projection=projection)
im = ax.pcolormesh(
    lons,
    lats,
    denorm(np.load(files_denoised[i]))[0, 1],
    vmax=6,
    vmin=-6,
    cmap="RdBu",
    transform=ccrs.PlateCarree(),
    alpha=1,
    rasterized=True,
)
addf(ax)
ax.set_title("Denoised state $\hat{x}_t$")
time = float(files_noisy[i].split("_")[-1][:-4])

ax = fig.add_subplot(1, 5, 3, projection=projection)
ps_obs = A(np.load(files_denoised[i])[:, 1])
ps_obs *= stds[1]
ps_obs += means[1]
toplotn = ps_obs
im = ax.scatter(
    lonss,
    latss,
    c=toplotn,
    cmap="RdBu",
    marker="^",
    vmax=6,
    vmin=-6,
    edgecolor="black",
    transform=ccrs.PlateCarree(),
    s=50,
    zorder=2,
)
addf(ax)
ax.set_title("$\mathcal{A}(\hat{x}_t)$")

ax = fig.add_subplot(1, 5, 4, projection=projection)
im = ax.scatter(
    lonss,
    latss,
    c=denorm(obs)[0, 1][mask],
    cmap="RdBu",
    marker="^",
    vmax=6,
    vmin=-6,
    edgecolor="black",
    transform=ccrs.PlateCarree(),
    s=50,
    zorder=2,
)
addf(ax)
ax.set_title("Observations $y$")


ax = fig.add_subplot(1, 5, 5)
oplot = denorm(obs)[0, 1][mask]
ax.bar(range(len(oplot)), oplot, alpha=0.5)
ax.bar(range(len(toplotn)), toplotn, alpha=0.5)
ax.legend(["$\mathcal{A}(\hat{x}_t)$", "$y$"])
ax.set_xlabel("station number")
ax.set_ylabel("observation")
ax.set_title("Difference")


plt.savefig(
    f"denoising_process/method_anim/denoising_plot_step_{f'{i:02d}'}.png", dpi=300
)

plt.close()


files_denoised = glob.glob("denoising_process/other_time/x_denoised_t_*")
files_noisy = glob.glob("denoising_process/other_time/x_noisy_t_*")
files_denoised.sort()
files_denoised.reverse()
files_noisy.sort()
files_noisy.reverse()


for i in range(len(files_noisy)):
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 2, 1, projection=projection)
    im = ax.pcolormesh(
        lons,
        lats,
        denorm(np.load(files_noisy[i]))[0, 1],
        vmax=6,
        vmin=-6,
        cmap="RdBu",
        transform=ccrs.PlateCarree(),
        alpha=1,
    )
    addf(ax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(
        "10v [m/s]",
    )
    ax.set_title("Noisy state")
    ax = fig.add_subplot(1, 2, 2, projection=projection)
    y = x.clone().detach()
    im = ax.pcolormesh(
        lons,
        lats,
        denorm(np.load(files_denoised[i]))[0, 1],
        vmax=6,
        vmin=-6,
        cmap="RdBu",
        transform=ccrs.PlateCarree(),
        alpha=1,
    )
    addf(ax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(
        "10v [m/s]",
    )
    ax.set_title("Denoised state")
    time = float(files_noisy[i].split("_")[-1][:-4])
    fig.suptitle(f"Backwards diffusion time = {np.round(time, 2)}")
    plt.savefig(f"denoising_process/plot_step_{f'{i:02d}'}.png", dpi=100)
    plt.close()

glob.glob("denoising_process/plot_step_*").sort()


from PIL import Image


png_files = glob.glob("denoising_process/plot_step_*")
png_files.sort()

images = [Image.open(file) for file in png_files]

images[0].save(
    "denoising_process/guided_denoising.gif",
    save_all=True,
    append_images=images[1:],
    optimize=False,
    duration=200,
    loop=0,
)


plt.close()


from PIL import Image

png_files = glob.glob("denoising_process/method_anim/*")
png_files.sort()


images = [Image.open(file) for file in png_files]


images[0].save(
    "denoising_process/method_anim/method.gif",
    save_all=True,
    append_images=images[1:],
    optimize=False,
    duration=200,
    loop=0,
)
