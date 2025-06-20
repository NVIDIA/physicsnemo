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

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

files = {
    "Raw Sensitivities": Path(__file__).parent
    / "gradient_checking_results"
    / "drag_gradients_raw.txt",
    "Smooth Sensitivities": Path(__file__).parent
    / "gradient_checking_results"
    / "drag_gradients_smooth.txt",
}

plt.figure(figsize=(9, 7))

for name, file in files.items():
    data = np.loadtxt(file, delimiter=",")
    epsilon = data[:, 0]
    drag = data[:, 1]

    baseline_drag = drag[epsilon == 0][0]
    drag_delta = drag - baseline_drag

    plt.plot(
        epsilon,
        drag_delta,
        ".",
        label=name,
        color="C0" if name == "Raw Sensitivities" else "C1",
    )

x = np.unique(np.concatenate([line.get_xdata() for line in plt.gca().get_lines()]))
x_minscale = np.min(np.abs(x[x != 0]))
x_maxscale = np.max(np.abs(x[x != 0]))

x_minscale = 1e-15
# x_maxscale=1e0

sorted_x = np.sort(
    np.concatenate(
        (
            np.logspace(np.log10(x_minscale) - 3, np.log10(x_maxscale), 100),
            np.abs(epsilon),
        )
    )
)
sorted_x = np.concatenate((-sorted_x[::-1], sorted_x))

analytical_gradient = -376531.0

plt.plot(
    sorted_x,
    analytical_gradient * sorted_x,
    "--k",
    label="Adjoint-Based Gradient",
    zorder=1.9,
)

# Set up logit axes with symmetric linear range
plt.xscale("symlog", linthresh=x_minscale)
plt.yscale("symlog", linthresh=np.abs(analytical_gradient) * x_minscale)

xax = plt.gca().xaxis
yax = plt.gca().yaxis

for ax in ["x", "y"]:
    for kind in ["major", "minor"]:
        a = plt.gca().xaxis if ax == "x" else plt.gca().yaxis
        locator = ticker.SymmetricalLogLocator(
            base=1000 if kind == "major" else 10,
            linthresh=x_minscale
            if ax == "x"
            else np.abs(analytical_gradient) * x_minscale,
        )
        locator.numticks = 1000

        if kind == "major":
            a.set_major_locator(locator)
        else:
            a.set_minor_locator(locator)

plt.xticks(rotation=15, va="top")
plt.yticks(rotation=15, va="center")
plt.axhline(0, color="k", linewidth=1.2, zorder=1, alpha=0.2)
plt.axvline(0, color="k", linewidth=1.2, zorder=1, alpha=0.2)


# Add grid for better readability
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.xlabel("Epsilon [m / (N/m)]")
plt.ylabel("Net change in Drag Force, relative to baseline [N]")
plt.title("Adjoint-Predicted Gradient vs. Finite-Differences")
plt.legend(loc="lower left")
plt.show()
