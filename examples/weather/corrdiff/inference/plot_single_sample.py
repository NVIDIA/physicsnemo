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
import argparse

import cftime
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np


def pattern_correlation(x, y):
    """Pattern correlation"""
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.mean((x - mx) ** 2)
    vy = np.mean((y - my) ** 2)

    a = np.mean((x - mx) * (y - my))
    b = np.sqrt(vx * vy)

    return a / b


def plot_channels(group, time_idx: int):
    """Plot channels"""
    # weather sub-plot
    num_channels = len(group.variables)
    ncols = 4
    fig, axs = plt.subplots(
        nrows=(
            num_channels // ncols
            if num_channels % ncols == 0
            else num_channels // ncols + 1
        ),
        ncols=ncols,
        sharex=True,
        sharey=True,
        constrained_layout=True,
        figsize=(15, 15),
    )

    for ch, ax in zip(sorted(group.variables), axs.flat):
        # label row
        x = group[ch][time_idx]
        ax.set_title(ch)
        ax.imshow(x)


def channel_eq(a, b):
    """Check if two channels are equal in variable and pressure."""
    variable_equal = a["variable"] == b["variable"]
    pressure_is_nan = np.isnan(a["pressure"]) and np.isnan(b["pressure"])
    pressure_equal = a["pressure"] == b["pressure"]
    return variable_equal and (pressure_equal or pressure_is_nan)


def channel_repr(channel):
    """Return a string representation of a channel with variable and pressure."""
    v = channel["variable"]
    pressure = channel["pressure"]
    return f"{v}\n Pressure: {pressure}"


def get_clim(output_channels, f):
    """Get color limits (clim) for output channels based on prediction and truth data."""
    colorlimits = {}
    for ch in range(len(output_channels)):
        channel = output_channels[ch]
        y = f["prediction"][channel][:]
        truth = f["truth"][channel][:]

        vmin = min([y.min(), truth.min()])
        vmax = max([y.max(), truth.max()])
        colorlimits[channel] = (vmin, vmax)
    return colorlimits


def main(file, output_dir, sample):
    """Plot single sample"""
    os.makedirs(output_dir, exist_ok=True)
    f = nc.Dataset(file, "r")

    # for c in f.time:
    output_channels = list(f["prediction"].variables)
    v = f["time"]
    times = cftime.num2date(v, units=v.units, calendar=v.calendar)

    def plot_panel(ax, data, **kwargs):
        return ax.pcolormesh(f["lon"], f["lat"], data, cmap="RdBu_r", **kwargs)

    colorlimits = get_clim(output_channels, f)
    for idx in range(len(times)):
        print("idx", idx)
        # weather sub-plot
        fig, axs = plt.subplots(
            nrows=len(output_channels),
            ncols=3,
            sharex=True,
            sharey=True,
            constrained_layout=True,
            figsize=(12, 12),
        )
        row = axs[0]
        row[0].set_title("Input")
        row[1].set_title("Generated")
        row[2].set_title("Truth")

        for ch in range(len(output_channels)):
            channel = output_channels[ch]
            row = axs[ch]

            # label row

            y = f["prediction"][channel][sample, idx]
            truth = f["truth"][channel][idx]

            # search for input_channel
            input_channels = list(f["input"].variables)
            if channel in input_channels:
                x = f["input"][channel][idx]
            else:
                x = None

            vmin, vmax = colorlimits[channel]

            def plot_panel(ax, data, **kwargs):
                if channel == "maximum_radar_reflectivity":
                    return ax.pcolormesh(
                        f["lon"], f["lat"], data, cmap="magma", vmin=0, vmax=vmax
                    )
                if channel == "temperature_2m":
                    return ax.pcolormesh(
                        f["lon"], f["lat"], data, cmap="magma", vmin=vmin, vmax=vmax
                    )
                else:
                    if vmin < 0 < vmax:
                        bound = max(abs(vmin), abs(vmax))
                        vmin1 = -bound
                        vmax1 = bound
                    else:
                        vmin1 = vmin
                        vmax1 = vmax
                    return ax.pcolormesh(
                        f["lon"], f["lat"], data, cmap="RdBu_r", vmin=vmin1, vmax=vmax1
                    )

            if x is not None:
                plot_panel(row[0], x)
                pc_x = pattern_correlation(x, truth)
                label_x = pc_x
                row[0].set_title(f"Input, Pattern correlation: {label_x:.2f}")

            im = plot_panel(row[1], y)
            plot_panel(row[2], truth)

            cb = plt.colorbar(im, ax=row.tolist())
            cb.set_label(channel)

            pc_y = pattern_correlation(y, truth)
            label_y = pc_y
            row[1].set_title(f"Generated, Pattern correlation: {label_y:.2f}")

        for ax in axs[-1]:
            ax.set_xlabel("longitude [deg E]")

        for ax in axs[:, 0]:
            ax.set_ylabel("latitude [deg N]")

        time = times[idx]
        plt.suptitle(f"Time {time.isoformat()}")
        plt.savefig(f"{output_dir}/{time.isoformat()}.sample.png")

        plot_channels(f["input"], idx)
        plt.savefig(f"{output_dir}/{time.isoformat()}.input.png")


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add the positional arguments
    parser.add_argument("--netcdf_file", help="Path to the NetCDF file")
    parser.add_argument("--output_dir", help="Path to the output directory")
    # Add the optional argument
    parser.add_argument("--sample", help="Sample to plot", default=0, type=int)
    # Parse the arguments
    args = parser.parse_args()
    main(args.netcdf_file, args.output_dir, args.sample)
