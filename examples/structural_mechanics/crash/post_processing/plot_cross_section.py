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

"""
Script to plot surface cross-section at y=12.153 from VTP files for all
available time steps, combined into a single grid image.
"""

import os
import re
import argparse
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


def extract_timestep_from_path(file_path):
    """
    Extracts the time step from a file path.
    """
    filename = os.path.basename(file_path)
    numbers = re.findall(r"\d+", filename)
    if numbers:
        return numbers[-1]
    return "Unknown"


def load_vtp_and_get_surface_points(vtp_file):
    """
    Loads a VTP file and extracts surface points at the y=12.153 cross-section.
    """
    if not os.path.exists(vtp_file):
        raise FileNotFoundError(f"VTP file not found: {vtp_file}")
    mesh = pv.read(vtp_file)
    points = mesh.points
    y_cross_section = 12.153
    tolerance = 8.0  # Use a wider tolerance for point clouds without edge data
    indices = np.where(np.abs(points[:, 1] - y_cross_section) < tolerance)[0]
    return points[indices, 0], points[indices, 2]


def plot_timestep_on_ax(ax, pred_vtp_file, exact_vtp_file, with_edges=False):
    """
    Plots a single timestep comparison of predicted vs. exact data on a
    provided Matplotlib axis object.

    Args:
        ax (matplotlib.axes.Axes): The subplot axis to plot on.
        pred_vtp_file (str): Path to the predicted VTP file.
        exact_vtp_file (str): Path to the exact VTP file.
        with_edges (bool): If True, plots mesh edges.
    """
    timestep = extract_timestep_from_path(exact_vtp_file)
    y_cross_section = 12.153

    # --- Data Loading ---
    if with_edges:
        pred_mesh = pv.read(pred_vtp_file)
        exact_mesh = pv.read(exact_vtp_file)
        pred_points = pred_mesh.points
        exact_points = exact_mesh.points

        tolerance = 1e-6  # Use a tight tolerance for precise edge slicing
        pred_indices = np.where(
            np.abs(pred_points[:, 1] - y_cross_section) < tolerance
        )[0]
        exact_indices = np.where(
            np.abs(exact_points[:, 1] - y_cross_section) < tolerance
        )[0]

        pred_x, pred_z = pred_points[pred_indices, 0], pred_points[pred_indices, 2]
        exact_x, exact_z = (
            exact_points[exact_indices, 0],
            exact_points[exact_indices, 2],
        )
    else:
        pred_x, pred_z = load_vtp_and_get_surface_points(pred_vtp_file)
        exact_x, exact_z = load_vtp_and_get_surface_points(exact_vtp_file)

    # --- Plotting ---
    ax.scatter(pred_x, pred_z, c="red", s=2, alpha=0.7, label="Predicted")
    ax.scatter(exact_x, exact_z, c="blue", s=2, alpha=0.7, label="Exact")

    if with_edges:
        if hasattr(pred_mesh, "lines") and pred_mesh.lines.size > 0:
            lines = pred_mesh.lines.reshape(-1, 3)
            for line in lines:
                if line[0] == 2:
                    p1_idx, p2_idx = line[1], line[2]
                    if p1_idx < len(pred_points) and p2_idx < len(pred_points):
                        p1, p2 = pred_points[p1_idx], pred_points[p2_idx]
                        if (
                            np.abs(p1[1] - y_cross_section) < tolerance
                            and np.abs(p2[1] - y_cross_section) < tolerance
                        ):
                            ax.plot(
                                [p1[0], p2[0]],
                                [p1[2], p2[2]],
                                "r-",
                                alpha=0.5,
                                linewidth=0.5,
                            )

        if hasattr(exact_mesh, "lines") and exact_mesh.lines.size > 0:
            lines = exact_mesh.lines.reshape(-1, 3)
            for line in lines:
                if line[0] == 2:
                    p1_idx, p2_idx = line[1], line[2]
                    if p1_idx < len(exact_points) and p2_idx < len(exact_points):
                        p1, p2 = exact_points[p1_idx], exact_points[p2_idx]
                        if (
                            np.abs(p1[1] - y_cross_section) < tolerance
                            and np.abs(p2[1] - y_cross_section) < tolerance
                        ):
                            ax.plot(
                                [p1[0], p2[0]],
                                [p1[2], p2[2]],
                                "b-",
                                alpha=0.5,
                                linewidth=0.5,
                            )

    # --- Formatting ---
    ax.set_title(f"Timestep: {timestep}")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(markerscale=4)


def main():
    parser = argparse.ArgumentParser(
        description="Plot surface cross-sections for all timesteps in a single grid plot."
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Path to the directory containing predicted VTP files.",
    )
    parser.add_argument(
        "--exact_dir",
        type=str,
        required=True,
        help="Path to the directory containing exact VTP files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="comparison_grid.png",
        help="Output file path for the combined plot.",
    )
    parser.add_argument(
        "--with_edges",
        action="store_true",
        help="Include edge information in the plot.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the plot after saving."
    )
    args = parser.parse_args()

    # Verify directories exist
    for d in [args.pred_dir, args.exact_dir]:
        if not os.path.isdir(d):
            print(f"Error: Directory not found: {d}")
            return

    # Find and map VTP files by timestep
    pred_files = {
        extract_timestep_from_path(f): os.path.join(args.pred_dir, f)
        for f in os.listdir(args.pred_dir)
        if f.endswith(".vtp")
    }
    exact_files = {
        extract_timestep_from_path(f): os.path.join(args.exact_dir, f)
        for f in os.listdir(args.exact_dir)
        if f.endswith(".vtp")
    }

    common_timesteps = sorted(
        list(set(pred_files.keys()) & set(exact_files.keys())), key=int
    )

    if not common_timesteps:
        print("No common timesteps found between the two directories.")
        return

    num_plots = len(common_timesteps)
    print(f"Found {num_plots} common timesteps. Creating a grid plot...")

    # Calculate grid size
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))

    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5), squeeze=False)
    flat_axes = axes.flatten()

    # Loop and plot on each subplot
    for i, timestep in enumerate(common_timesteps):
        ax = flat_axes[i]
        print(f"  Plotting timestep {timestep}...")
        pred_file = pred_files[timestep]
        exact_file = exact_files[timestep]

        plot_timestep_on_ax(ax, pred_file, exact_file, args.with_edges)

        # Add axis labels only to the outer plots to reduce clutter
        if i % cols == 0:
            ax.set_ylabel("Z-coordinate")
        if i >= num_plots - cols:
            ax.set_xlabel("X-coordinate")

    # Hide unused subplots
    for i in range(num_plots, len(flat_axes)):
        flat_axes[i].axis("off")

    # Final adjustments and save
    fig.suptitle(f"Cross-Section Comparison at y=12.153", fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])  # Adjust layout for main title

    plt.savefig(args.output_file, dpi=200)
    print(f"\n✅ Grid plot saved to: {args.output_file}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
