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
Derives and plots velocity and acceleration from averaged position data over two probe point sets
('driver' and 'passenger'), producing four curves per plot:
- Driver (Ground Truth), Driver (Predicted), Passenger (Ground Truth), Passenger (Predicted).
"""

import os
import re
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt


def extract_timestep_from_path(file_path: str) -> str:
    filename = os.path.basename(file_path)
    numbers = re.findall(r"\d+", filename)
    return numbers[-1] if numbers else "0"


def parse_point_set(spec: str) -> List[int]:
    """
    Parse a comma/space-separated list of integers and inclusive ranges like '70658-70659'.
    Example: '70658-70659, 70664, 70676-70679' -> [70658, 70659, 70664, 70676, 70677, 70678, 70679]
    """
    if not spec:
        return []
    ids: List[int] = []
    for token in re.split(r"[,\s]+", spec.strip()):
        if not token:
            continue
        if "-" in token:
            a_str, b_str = token.split("-", 1)
            a = int(a_str)
            b = int(b_str)
            if b < a:
                a, b = b, a
            ids.extend(range(a, b + 1))
        else:
            ids.append(int(token))
    return sorted(set(ids))


def load_averaged_series(
    vtp_dir: str, point_ids: List[int], dt: float, position_array: str
) -> pd.DataFrame | None:
    """
    Load VTPs from vtp_dir, average positions over point_ids for the given position_array ('prediction' or 'exact'),
    and return a DataFrame with Time, Position, Velocity, Acceleration (per axis).
    """
    if not os.path.isdir(vtp_dir):
        print(f"❌ Error: Directory not found: {vtp_dir}")
        return None

    try:
        vtp_files: Dict[int, str] = {
            int(extract_timestep_from_path(f)): os.path.join(vtp_dir, f)
            for f in os.listdir(vtp_dir)
            if f.lower().endswith(".vtp")
        }
    except (ValueError, TypeError):
        print(
            f"❌ Error: Could not extract integer timesteps from filenames in {vtp_dir}."
        )
        return None

    if not vtp_files:
        print(f"❌ Error: No .vtp files found in {vtp_dir}")
        return None

    if not point_ids:
        print("❌ Error: Empty point set provided.")
        return None

    sorted_timesteps = sorted(vtp_files.keys())
    rows = []
    for step in sorted_timesteps:
        filepath = vtp_files[step]
        try:
            mesh = pv.read(filepath, progress_bar=False)
            if position_array not in mesh.point_data:
                continue

            arr = mesh.point_data[position_array]
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            valid_ids = [pid for pid in point_ids if 0 <= pid < mesh.n_points]
            if not valid_ids:
                continue

            subset = arr[valid_ids]  # (k, dim)
            avg = subset.mean(axis=0)  # (dim,)
            avg3 = np.zeros(3, dtype=float)
            avg3[: min(3, avg.shape[0])] = avg[: min(3, avg.shape[0])]
            avg3 /= 1000.0  # convert to m

            rows.append(
                {
                    "Time (s)": step * dt,
                    "Timestep": step,
                    "Position_X": float(avg3[0]),
                    "Position_Y": float(avg3[1]),
                    "Position_Z": float(avg3[2]),
                }
            )
        except Exception as e:
            print(f"❌ Error processing file {filepath}: {e}")

    if not rows:
        print("Could not extract any averaged position data.")
        return None

    df = pd.DataFrame(rows).sort_values(by="Time (s)").reset_index(drop=True)

    # Derive velocity and acceleration using central differences
    for axis in ["X", "Y", "Z"]:
        df[f"Velocity_{axis}"] = np.gradient(df[f"Position_{axis}"], df["Time (s)"])
        df[f"Acceleration_{axis}"] = np.gradient(df[f"Velocity_{axis}"], df["Time (s)"])

    df.fillna(0, inplace=True)
    return df


def plot_kinematics(
    driver_gt: pd.DataFrame,
    driver_pred: pd.DataFrame,
    passenger_gt: pd.DataFrame,
    passenger_pred: pd.DataFrame,
    output_plot: str,
):
    """
    2x3 plots; each subplot shows two curves (GT vs Pred) for X only:
    - Row 0: Driver (red solid = GT, red dashed = Pred)
    - Row 1: Passenger (blue solid = GT, blue dashed = Pred)
    Columns: Displacement X, Velocity X, Acceleration X
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    fig.suptitle(
        "Kinematics (x-direction ): Driver (top) vs Passenger (bottom) toe pan| Ground Truth vs Predicted",
        fontsize=16,
    )

    def get_limits(dfs: List[pd.DataFrame], col: str):
        vals = np.concatenate([df[col].to_numpy() for df in dfs if col in df])
        if vals.size == 0:
            return (-1, 1)
        vmin, vmax = float(vals.min()), float(vals.max())
        margin = (vmax - vmin) * 0.05
        if margin == 0:
            margin = 1.0
        return (vmin - margin, vmax + margin)

    # Shared limits per row (driver vs passenger) for each X component
    pos_lim_driver = get_limits([driver_gt, driver_pred], "Position_X")
    vel_lim_driver = get_limits([driver_gt, driver_pred], "Velocity_X")
    acc_lim_driver = get_limits([driver_gt, driver_pred], "Acceleration_X")

    pos_lim_pass = get_limits([passenger_gt, passenger_pred], "Position_X")
    vel_lim_pass = get_limits([passenger_gt, passenger_pred], "Velocity_X")
    acc_lim_pass = get_limits([passenger_gt, passenger_pred], "Acceleration_X")

    components = [
        ("Position_X", "Displacement (m)"),
        ("Velocity_X", "Velocity (m/s)"),
        ("Acceleration_X", "Acceleration (m/s²)"),
    ]

    # Row 0: Driver (red)
    for j, (comp, label) in enumerate(components):
        ax = axes[0, j]
        ax.plot(
            driver_gt["Time (s)"],
            driver_gt[comp],
            color="red",
            linewidth=2,
            label="Driver GT",
        )
        ax.plot(
            driver_pred["Time (s)"],
            driver_pred[comp],
            color="red",
            linestyle="--",
            linewidth=2,
            label="Driver Pred",
        )
        if comp.startswith("Position"):
            ax.set_ylim(pos_lim_driver)
        elif comp.startswith("Velocity"):
            ax.set_ylim(vel_lim_driver)
        else:
            ax.set_ylim(acc_lim_driver)
        ax.set_title(label, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    # Row 1: Passenger (blue)
    for j, (comp, label) in enumerate(components):
        ax = axes[1, j]
        ax.plot(
            passenger_gt["Time (s)"],
            passenger_gt[comp],
            color="blue",
            linewidth=2,
            label="Passenger GT",
        )
        ax.plot(
            passenger_pred["Time (s)"],
            passenger_pred[comp],
            color="blue",
            linestyle="--",
            linewidth=2,
            label="Passenger Pred",
        )
        if comp.startswith("Position"):
            ax.set_ylim(pos_lim_pass)
        elif comp.startswith("Velocity"):
            ax.set_ylim(vel_lim_pass)
        else:
            ax.set_ylim(acc_lim_pass)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    for ax in axes[1, :]:
        ax.set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(output_plot, dpi=300)
    print(f"\n📈 Plot saved to: {output_plot}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot derived kinematics (Driver & Passenger | GT vs Pred) from averaged point positions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help='Directory with predicted VTP files ("prediction" array).',
    )
    parser.add_argument(
        "--exact_dir",
        type=str,
        required=True,
        help='Directory with ground truth VTP files ("exact" array).',
    )
    parser.add_argument(
        "--driver_points",
        type=str,
        required=True,
        help='Driver point IDs/ranges (e.g., "70658-70659,70664,70676-70679").',
    )
    parser.add_argument(
        "--passenger_points",
        type=str,
        required=True,
        help="Passenger point IDs/ranges.",
    )
    parser.add_argument(
        "--dt", type=float, default=1.0, help="Time step size Δt in seconds."
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="driver_passenger_gt_pred_kinematics.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--save_csv", action="store_true", help="Save the processed data to CSV files."
    )
    args = parser.parse_args()

    driver_ids = parse_point_set(args.driver_points)
    passenger_ids = parse_point_set(args.passenger_points)

    # Load series
    print("--- Loading Driver (Ground Truth) ---")
    driver_gt = load_averaged_series(
        args.exact_dir, driver_ids, args.dt, position_array="exact"
    )
    if driver_gt is None:
        return

    print("--- Loading Driver (Predicted) ---")
    driver_pred = load_averaged_series(
        args.pred_dir, driver_ids, args.dt, position_array="prediction"
    )
    if driver_pred is None:
        return

    print("--- Loading Passenger (Ground Truth) ---")
    passenger_gt = load_averaged_series(
        args.exact_dir, passenger_ids, args.dt, position_array="exact"
    )
    if passenger_gt is None:
        return

    print("--- Loading Passenger (Predicted) ---")
    passenger_pred = load_averaged_series(
        args.pred_dir, passenger_ids, args.dt, position_array="prediction"
    )
    if passenger_pred is None:
        return

    if args.save_csv:
        driver_gt.to_csv(
            "driver_ground_truth_kinematics.csv", index=False, float_format="%.6e"
        )
        driver_pred.to_csv(
            "driver_predicted_kinematics.csv", index=False, float_format="%.6e"
        )
        passenger_gt.to_csv(
            "passenger_ground_truth_kinematics.csv", index=False, float_format="%.6e"
        )
        passenger_pred.to_csv(
            "passenger_predicted_kinematics.csv", index=False, float_format="%.6e"
        )
        print("\n💾 Saved CSVs for driver/passenger (GT/Pred)")

    plot_kinematics(
        driver_gt, driver_pred, passenger_gt, passenger_pred, args.output_plot
    )


if __name__ == "__main__":
    main()
