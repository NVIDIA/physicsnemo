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

"""Plot compressor output parity and predicted pressure jumps of a run."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from basis import quad_edges
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

COLOR = "#cb7449"
GLOBALS = (
    ("Massflow", "mass flow"),
    ("Compression_ratio", "compression ratio"),
    ("Efficiency", "efficiency"),
)


def configure_style():
    """Use a white canvas with light axes."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#7d8790",
            "axes.linewidth": 0.7,
            "xtick.color": "#38434e",
            "ytick.color": "#38434e",
            "text.color": "#202b35",
            "axes.labelcolor": "#202b35",
            "pdf.fonttype": 42,
        }
    )


def save(figure, output_dir, name):
    """Write a PNG preview and a PDF of a figure."""
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"{name}.{extension}",
            dpi=250,
            bbox_inches="tight",
            pad_inches=0.08,
        )
    plt.close(figure)


def plot_parity(cases, output_dir):
    """Compare predicted and true compressor outputs of every evaluated case."""
    with cases.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    figure, axes = plt.subplots(
        1, len(GLOBALS), figsize=(12.5, 3.9), layout="constrained"
    )
    for axis, (name, label) in zip(axes, GLOBALS):
        truth = np.array([float(row[f"{name}_truth"]) for row in rows])
        prediction = np.array([float(row[f"{name}_prediction"]) for row in rows])
        low = min(truth.min(), prediction.min())
        high = max(truth.max(), prediction.max())
        limits = (low - 0.05 * (high - low), high + 0.05 * (high - low))
        axis.plot(limits, limits, color="#9aa4ad", linewidth=0.8, zorder=1)
        axis.scatter(
            truth,
            prediction,
            s=16,
            color=COLOR,
            edgecolors="white",
            linewidths=0.4,
            zorder=2,
        )
        axis.set(
            xlim=limits,
            ylim=limits,
            aspect="equal",
            xlabel=f"True {label}",
            ylabel=f"Predicted {label}",
        )
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_locator(MaxNLocator(4))
    save(figure, output_dir, "compressor_parity")


def draw_surface(axis, points, quads, values, limits, cmap, side):
    """Draw one side of the blade surface in an orthographic view."""
    center = points.mean(axis=0)
    _, _, frame = np.linalg.svd(points - center, full_matrices=False)
    frame *= np.sign(frame[np.arange(3), np.abs(frame).argmax(axis=1)])[:, None]
    view = side * frame[2] + 0.12 * frame[1]
    view /= np.linalg.norm(view)
    up = frame[0] - (frame[0] @ view) * view
    up /= np.linalg.norm(up)
    right = np.cross(up, view)
    relative = points - center
    screen = np.stack((relative @ right, relative @ up), axis=-1)
    depth = (relative @ view)[quads].mean(axis=1)
    order = np.argsort(depth)
    colors = plt.get_cmap(cmap)(Normalize(*limits)(values[quads].mean(axis=1)))
    faces = PolyCollection(
        screen[quads[order]],
        facecolors=colors[order],
        edgecolors=colors[order],
        linewidths=0.2,
    )
    axis.add_collection(faces)
    axis.set_xlim(screen[:, 0].min(), screen[:, 0].max())
    axis.set_ylim(screen[:, 1].min(), screen[:, 1].max())
    axis.set_aspect("equal")
    axis.set_axis_off()


def vertex_maximum(values, edges, count):
    """Assign each vertex the largest value on its incident edges."""
    result = np.zeros(count)
    np.maximum.at(result, edges[:, 0], values)
    np.maximum.at(result, edges[:, 1], values)
    return result


def plot_pressure_jumps(prediction_path, output_dir):
    """Show reference and predicted pressure jumps and their error on one case."""
    with np.load(prediction_path) as saved:
        case = dict(saved)
    points, quads = case["points"].astype(np.float64), case["quads"]
    edges = quad_edges(quads, len(points))

    def jumps(pressure):
        pressure = pressure.astype(np.float64)
        scale = float(case["pressure_scale"])
        return (pressure[edges[:, 0]] - pressure[edges[:, 1]]) / scale

    truth = jumps(case["field_truth"][:, 1])
    predicted = jumps(case["field_prediction"][:, 1])
    magnitudes = [
        vertex_maximum(np.abs(values), edges, len(points))
        for values in (truth, predicted)
    ]
    error = vertex_maximum(np.abs(predicted - truth), edges, len(points))
    magnitude_limits = (0.0, max(values.max() for values in magnitudes))
    jump_label = r"$|\Delta p| / \sigma_p$"
    columns = (
        (magnitudes[0], magnitude_limits, "viridis", f"Reference\n{jump_label}"),
        (magnitudes[1], magnitude_limits, "viridis", f"Prediction\n{jump_label}"),
        (error, (0.0, error.max()), "magma", r"Jump error / $\sigma_p$"),
    )
    figure, axes = plt.subplots(2, len(columns), figsize=(3.05 * len(columns), 6.8))
    figure.subplots_adjust(wspace=0.015, hspace=0.04, bottom=0.15, top=0.995)
    for column, (values, limits, cmap, label) in enumerate(columns):
        for row, side in enumerate((1, -1)):
            draw_surface(axes[row, column], points, quads, values, limits, cmap, side)
        colorbar = figure.colorbar(
            ScalarMappable(norm=Normalize(*limits), cmap=cmap),
            ax=list(axes[:, column]),
            orientation="horizontal",
            fraction=0.035,
            pad=0.015,
            shrink=0.9,
            aspect=24,
        )
        colorbar.outline.set_visible(False)
        colorbar.locator = MaxNLocator(3)
        colorbar.update_ticks()
        colorbar.set_label(label)
    save(figure, output_dir, "pressure_jumps")


def main():
    """Render the figures of an evaluated run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/geotransolver"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample", type=int, default=613)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    evaluation = args.run_dir / "evaluation" / args.split
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_parity(evaluation / "cases.csv", args.output_dir)
    plot_pressure_jumps(
        evaluation / "predictions" / f"sample_{args.sample:06d}.npz", args.output_dir
    )


if __name__ == "__main__":
    main()
