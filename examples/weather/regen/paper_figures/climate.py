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
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import joblib
import cartopy.crs as crs

sys.path.insert(0, "../../../../")
sys.path.insert(0, "../")
from utils import savefig, subplot
import matplotlib

matplotlib.rcParams["font.family"] = "sans-serif"
import config
from matplotlib.ticker import MaxNLocator


POINTS_PER_INCH = 72
PAPER_WIDTH = 397.48499 / POINTS_PER_INCH

PROJECT = "/path/to/project"
DATA_PATH = "figure_data/climate.pkl"
n_samples = 5000

unconditional_samples = os.path.join(
    PROJECT, "inferences", "uncondSample", "samples.nc"
)
conditional_samples = os.path.join(PROJECT, "inferences", "peters_step_64_all.nc")

fields = ["10u", "10v", "tp"]
sources = [
    "hrrr",
    "sampled",
    "sda",
]


def prepare_data():
    # load conditional samples
    FIGURE_DATA = {}
    loaded_data = {}

    def add_conditional_samples():
        # TODO use same samples as the truth
        # currently gives an error
        # IndexError: index 8649 is out of bounds for axis 0 with size 8640

        ds = xr.open_dataset(conditional_samples, group="prediction")
        index = np.random.choice(ds.sizes["time"], n_samples)

        ds = ds.isel(time=index, ensemble=0)
        for field in fields:
            loaded_data.setdefault("sda", {})[field] = ds[field].values

    add_conditional_samples()

    # load hrrr data
    hrrr = xr.open_zarr(config.path_to_hrrr, mask_and_scale=False)
    hrrr_index = np.random.choice(hrrr.sizes["time"], n_samples)
    # TODO confirm validation period
    hrrr = hrrr.isel(time=hrrr_index).HRRR.to_dataset("channel")
    for field in fields:
        data = loaded_data.setdefault("hrrr", {})
        data[field] = hrrr[field].load().values

    # load sampled data
    samples = xr.open_dataset(unconditional_samples, mask_and_scale=False)
    subsamples = samples.isel(sample=slice(n_samples))
    for field in fields:
        data = loaded_data.setdefault("sampled", {})
        field_name = {"10u": "u10m", "10v": "v10m", "2t": "t2m"}.get(field, field)
        data[field] = subsamples[field_name].load().values

    lat = samples.lat.values
    lon = samples.lon.values

    # %%
    # these are the validation stations used.
    # see sda/experiments/corrdiff/evaluation/evaluate_performance_table_metrics_parallel.py
    nleave = 10
    x, y = np.load("../evenmore_random_val_stations.npy")[:nleave].T
    lat_validation = lat[x, y]
    lon_validation = lon[x, y]

    x, y = np.load("../evenmore_random_val_stations.npy")[nleave:].T
    lat_station = lat[x, y]
    lon_station = lon[x, y]

    def filter_tp(source):
        tp = loaded_data[source]["tp"]
        tp_filtered = tp[tp.max(axis=(1, 2)) <= 200]
        loaded_data[source]["tp"] = tp_filtered
        print(len(tp_filtered) / len(tp))

    filter_tp("sda")
    filter_tp("sampled")

    def compute_hist(source, field="tp", min=None, max=None):
        tp = loaded_data[source][field].ravel()
        counts, edges = np.histogram(tp, bins=50)
        return counts, edges

    FIGURE_DATA["histograms"] = {
        (source, field): compute_hist(source, field)
        for source in sources
        for field in fields
    }

    FIGURE_DATA["time_means"] = {
        (source, field): loaded_data[source][field].mean(0)
        for source in sources
        for field in fields
    }
    FIGURE_DATA["lat"] = lat
    FIGURE_DATA["lon"] = lon
    FIGURE_DATA["lat_station"] = lat_station
    FIGURE_DATA["lon_station"] = lon_station
    FIGURE_DATA["lat_validation"] = lat_validation
    FIGURE_DATA["lon_validation"] = lon_validation
    return FIGURE_DATA


def plot_hist(source, field="tp", min=None, max=None):
    counts, edges = FIGURE_DATA["histograms"][(source, field)]
    plt.step(edges[1:], counts / counts.sum(), label=source)


def plot_time_mean(source, field, min, max, **kwargs):
    tmean = FIGURE_DATA["time_means"][(source, field)]
    plt.pcolormesh(
        FIGURE_DATA["lon"],
        FIGURE_DATA["lat"],
        tmean,
        vmin=min,
        vmax=max,
        **kwargs,
        transform=crs.PlateCarree(),
        rasterized=True,
    )
    plt.title(source)
    cb = plt.colorbar(orientation="horizontal")
    cb.locator = MaxNLocator(nbins=5)
    cb.update_ticks()
    add_stations(plt.gca())
    if (field == "10u") or (field == "10v"):
        unit = "m/s"
    else:
        unit = "mm/h"
    cb.set_label(f"{field} [{unit}]")


def add_stations(ax):
    plt.scatter(
        FIGURE_DATA["lon_station"],
        FIGURE_DATA["lat_station"],
        transform=crs.PlateCarree(),
        marker="^",
        edgecolor="black",
        s=40,
        color="none",
    )
    plt.scatter(
        FIGURE_DATA["lon_validation"],
        FIGURE_DATA["lat_validation"],
        marker="p",
        transform=crs.PlateCarree(),
        edgecolor="black",
        s=40,
        color="none",
    )


def plot_time_mean_for_field(field, min, max, **kwargs):
    pos = 100 + 10 * len(sources)
    plt.figure(figsize=(12, 5))
    subplot(pos + 1)
    plot_time_mean("hrrr", field, min, max, **kwargs)
    subplot(pos + 2)
    plot_time_mean("sampled", field, min, max, **kwargs)
    subplot(pos + 3)
    plot_time_mean("sda", field, min, max, **kwargs)
    savefig(f"figures/time_mean/{field}")


if __name__ == "__main__":
    while True:
        try:
            FIGURE_DATA = joblib.load(DATA_PATH)
            break
        except FileNotFoundError:
            joblib.dump(prepare_data(), DATA_PATH)

    plt.figure(figsize=(PAPER_WIDTH, 1.8))
    ax = plt.subplot(131)
    plot_hist("hrrr", max=60, min=-1)
    plot_hist("sampled", max=60, min=1)
    plot_hist("sda", max=60, min=1)
    plt.yscale("log")
    plt.xlabel("tp [mm/h]")
    plt.legend()

    plt.subplot(132, sharey=ax)
    plot_hist("hrrr", "10u", -15, 15)
    plot_hist("sampled", "10u", -15, 15)
    plot_hist("sda", "10u", -15, 15)
    plt.xlabel("u10 [m/s]")
    plt.yscale("log")

    plt.subplot(133, sharey=ax)
    plot_hist("hrrr", "10v", -15, 15)
    plot_hist("sampled", "10v", -15, 15)
    plot_hist("sda", "10v", -15, 15)
    plt.xlabel("v10 [m/s]")
    plt.yscale("log")
    plt.tight_layout()
    savefig("figures/histograms")

    plot_time_mean_for_field("tp", 0, 0.2, cmap="Blues")
    plot_time_mean_for_field("10u", -2, 2, cmap="RdBu_r")
    plot_time_mean_for_field("10v", -5, 5, cmap="RdBu_r")
