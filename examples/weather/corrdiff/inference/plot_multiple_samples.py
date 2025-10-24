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

import joblib
import argparse

import matplotlib.pyplot as plt
import xarray


def plot_samples(netcdf_file, output_dir, n_samples):
    """Plot multiple samples"""
    root = xarray.open_dataset(netcdf_file)
    ds = (
        xarray.open_dataset(netcdf_file, group="prediction")
        .merge(root)
        .set_coords(["lat", "lon"])
    )
    truth = (
        xarray.open_dataset(netcdf_file, group="truth")
        .merge(root)
        .set_coords(["lat", "lon"])
    )
    os.makedirs(output_dir, exist_ok=True)

    # concatenate truth data and ensemble mean as an "ensemble" member for easy
    # plotting
    truth_expanded = truth.assign_coords(ensemble="truth").expand_dims("ensemble")
    ens_mean = (
        ds.mean("ensemble")
        .assign_coords(ensemble="ensemble_mean")
        .expand_dims("ensemble")
    )
    # add [0, 1, 2, ...] to ensemble dim
    ds["ensemble"] = [str(i) for i in range(ds.sizes["ensemble"])]
    merged = xarray.concat([truth_expanded, ens_mean, ds], dim="ensemble")

    # plot the variables in parallel
    def plot(v):
        print(v)
        # 2 is for the ensemble and
        merged[v][: n_samples + 2, :].plot(row="time", col="ensemble")
        plt.savefig(f"{output_dir}/{v}.png")

    joblib.Parallel(n_jobs=8)(joblib.delayed(plot)(v) for v in merged)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the positional arguments
    parser.add_argument("--netcdf_file", help="Path to the NetCDF file")
    parser.add_argument("--output_dir", help="Path to the output directory")
    # Add the optional argument
    parser.add_argument("--n-samples", help="Number of samples", default=5, type=int)
    # Parse the arguments
    args = parser.parse_args()
    main(args.netcdf_file, args.output_dir, args.n_samples)
