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

"""Score the generated samples


Saves a netCDF of crps and other scores. Depends on time, but space and ensemble have been reduced::

    netcdf scores {
dimensions:
        metric = 4 ;
        time = 205 ;
variables:
        double eastward_wind_10m(metric, time) ;
                eastward_wind_10m:_FillValue = NaN ;
        double maximum_radar_reflectivity(metric, time) ;
                maximum_radar_reflectivity:_FillValue = NaN ;
        double northward_wind_10m(metric, time) ;
                northward_wind_10m:_FillValue = NaN ;
        double temperature_2m(metric, time) ;
                temperature_2m:_FillValue = NaN ;
        int64 time(time) ;
                time:units = "hours since 1990-01-01" ;
                time:calendar = "standard" ;
        string metric(metric) ;
}


"""
# %%
import multiprocessing
from functools import partial
import argparse

import tqdm
import dask.diagnostics
import dask

import xarray as xr

try:
    import xskillscore
except ImportError:
    raise ImportError("xskillscore not installed. Try `pip install xskillscore`")


def open_samples(f):
    """
    Open prediction and truth samples from a dataset file.

    Parameters:
        f: Path to the dataset file.

    Returns:
        tuple: A tuple containing truth, prediction, and root datasets.
    """
    root = xr.open_dataset(f)
    pred = xr.open_dataset(f, group="prediction")
    truth = xr.open_dataset(f, group="truth")

    pred = pred.merge(root)
    truth = truth.merge(root)

    truth = truth.set_coords(["lon", "lat"])
    pred = pred.set_coords(["lon", "lat"])
    return truth, pred, root


# compute metrics in parallel for performance reasons
def process(i, path, n_ensemble):
    truth, pred, root = open_samples(path)
    truth = truth.isel(time=slice(i, i + 1)).load()
    if n_ensemble > 0:
        pred = pred.isel(time=slice(i, i + 1), ensemble=slice(0, n_ensemble))
    pred = pred.load()
    dim = ["x", "y"]

    a = xskillscore.rmse(truth, pred.mean("ensemble"), dim=dim)
    b = xskillscore.crps_ensemble(truth, pred, member_dim="ensemble", dim=dim)

    c = pred.std("ensemble").mean(dim)
    crps_mean = xskillscore.crps_ensemble(
        truth,
        pred.mean("ensemble").expand_dims("ensemble"),
        member_dim="ensemble",
        dim=dim,
    )

    metrics = (
        xr.concat([a, b, c, crps_mean], dim="metric")
        .assign_coords(metric=["rmse", "crps", "std_dev", "mae"])
        .load()
    )
    return metrics


def main(path: str, output: str, n_ensemble: int == -1):

    truth, pred, root = open_samples(path)

    with multiprocessing.Pool(32) as pool:
        metrics = []
        for metric in tqdm.tqdm(
            pool.imap(
                partial(process, path=path, n_ensemble=n_ensemble),
                range(truth.sizes["time"]),
            ),
            total=truth.sizes["time"],
        ):
            metrics.append(metric)

    metrics = xr.concat(metrics, dim="time")
    metrics.attrs["n_ensemble"] = n_ensemble

    # to netcdf with single threaded scheduler to avoid deadlocks
    with dask.config.set(scheduler="single-threaded"):
        metrics.to_netcdf(output, mode="w")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--n-ensemble", type=int, default=-1)
    args = parser.parse_args()

    main(args.path, args.output, args.n_ensemble)
