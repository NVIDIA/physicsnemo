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

"""Quick visualization of what the VTKHDF dataloader produces.

Loads one simulation through ``vtkhdf_reader.load_vtkhdf_file`` and plots the merged
structural mesh (top view, x-y) at several timesteps, colored by von Mises stress -- i.e.
exactly the ``coords`` and ``stress_vm`` target the datapipe feeds the model. Saves a PNG.

Run:  python visualize_bumper.py [sim_id]   (e.g. sim_00002)
"""

import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vtkhdf_reader as R  # noqa: E402

CRASH_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(CRASH_DIR, "..", "..", "..", ".."))
DATA_DIR = os.environ.get("BUMPER_DATA_DIR", os.path.join(WORKSPACE, "simulations"))
MASTER_CSV = os.path.join(WORKSPACE, "bumper_beam_master_with_split.csv")


def velocity_for(sim_id):
    if not os.path.isfile(MASTER_CSV):
        return None
    with open(MASTER_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("sim_name") == sim_id:
                return float(row["velocity_kmh"])
    return None


def main():
    sim_id = sys.argv[1] if len(sys.argv) > 1 else None
    if sim_id is None:
        # prefer a higher-velocity sim for a more dramatic crush
        files = R.find_sim_files(DATA_DIR)
        best, best_v = None, -1
        for p in files:
            sid = os.path.splitext(os.path.basename(p))[0]
            v = velocity_for(sid) or 0
            if v > best_v:
                best, best_v = sid, v
        sim_id = best or os.path.splitext(os.path.basename(files[0]))[0]

    path = os.path.join(DATA_DIR, sim_id, f"{sim_id}.vtkhdf")
    coords, src, dst, targets = R.load_vtkhdf_file(path)
    vm = targets["stress_vm"]  # [T, N]
    T = coords.shape[0]
    v_kmh = velocity_for(sim_id)
    print(f"{sim_id}: coords {coords.shape}, {T} steps, velocity {v_kmh} km/h")

    frames = np.linspace(0, T - 1, 6).astype(int)
    vmax = np.percentile(vm[frames], 99) or 1.0  # shared color scale, robust to outliers

    # fixed extents so the crush is visible across panels
    x, y = coords[..., 0], coords[..., 1]
    xlim = (x.min(), x.max())
    ylim = (y.min(), y.max())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    sc = None
    for ax, t in zip(axes.ravel(), frames):
        sc = ax.scatter(
            coords[t, :, 0], coords[t, :, 1],
            c=vm[t], s=2, cmap="inferno", vmin=0, vmax=vmax,
        )
        ax.set_title(f"t = {t}/{T - 1}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    fig.colorbar(sc, ax=axes, shrink=0.7, label="von Mises stress (GPa)")
    title = f"Bumper crash {sim_id}  (top view, colored by von Mises stress)"
    if v_kmh is not None:
        title += f"  -  impact {v_kmh:.0f} km/h"
    fig.suptitle(title, fontsize=14)

    out = os.path.join(WORKSPACE, f"bumper_crash_{sim_id}.png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
