# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

import math
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.colors import LogNorm


def detect_variable_keys(root: zarr.group) -> List[str]:
    """Return variable array names to plot.

    A variable is defined as a 3D array with shape (time, hrrr_y, hrrr_x).
    Coordinate and non-3D arrays are skipped.
    """
    skip_names = {"time", "lat", "lon", "hrrr_y", "hrrr_x"}
    variable_keys: List[str] = []
    for key, arr in root.arrays():
        if key in skip_names:
            continue
        try:
            shape = arr.shape
        except Exception:
            continue
        if len(shape) == 3:
            variable_keys.append(key)
    # Keep a deterministic order
    variable_keys.sort()
    return variable_keys


def make_subplot_grid(n: int) -> (int, int):
    """Compute a pleasant rows/cols grid for n subplots."""
    if n <= 0:
        return 0, 0
    cols = min(4, max(1, int(math.ceil(math.sqrt(n)))))
    rows = int(math.ceil(n / cols))
    return rows, cols


def plot_time_slice(root: zarr.group, idx: int, variables: Sequence[str]) -> None:
    """Plot each variable slice at time index idx in subplots."""
    rows, cols = make_subplot_grid(len(variables))
    if rows == 0:
        raise ValueError("No variables to plot.")

    fig, axes = plt.subplots(
        rows, cols, figsize=(4.5 * cols, 3.8 * rows), squeeze=False
    )
    axes_flat = axes.ravel()

    for ax in axes_flat[len(variables) :]:
        ax.axis("off")

    for ax, var in zip(axes_flat, variables):
        data = root[var][idx, :, :]
        if var in {"aerot", "sde", "tp"}:
            positive = data[data > 0]
            if positive.size == 0:
                im = ax.imshow(data, origin="lower", cmap="viridis")
            else:
                vmin = float(np.nanmin(positive))
                vmax = float(np.nanmax(data))
                vmax = vmin * 10 if not np.isfinite(vmax) or vmax <= vmin else vmax
                im = ax.imshow(
                    data,
                    origin="lower",
                    cmap="viridis",
                    norm=LogNorm(vmin=max(vmin, 1e-6), vmax=vmax),
                )
        else:
            im = ax.imshow(data, origin="lower", cmap="viridis")
        ax.set_title(var)
        ax.set_xlabel("hrrr_x")
        ax.set_ylabel("hrrr_y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"zarr_plot_{idx:05d}.jpg")


def main() -> None:
    root = zarr.open_group(
        store="s3://hrrr-surface-sda/zarr-v1",
        mode="r",
        storage_options={"endpoint_url": "https://pdx.s8k.io"},
    )
    time = root["time"][:]
    idx = np.where(time == np.datetime64("2023-03-30T00:00:00"))[0][0]
    variables = detect_variable_keys(root)

    plot_time_slice(root, idx, variables)


if __name__ == "__main__":
    main()
