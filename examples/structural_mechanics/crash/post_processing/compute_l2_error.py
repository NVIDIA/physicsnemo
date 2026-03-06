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

import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


def read_vtp_points(file_path: str) -> np.ndarray:
    mesh = pv.read(file_path)
    return mesh.points


def relative_position_error(x_pred: np.ndarray, x_true: np.ndarray) -> float:
    if x_pred.shape != x_true.shape:
        raise ValueError(
            f"Point arrays must match. Got {x_pred.shape} vs {x_true.shape}"
        )
    diff = (x_pred - x_true).ravel()
    denom = x_true.ravel()
    num = np.linalg.norm(diff, ord=2)
    den = np.linalg.norm(denom, ord=2)
    if den < 1e-12:
        return 0.0 if np.linalg.norm(x_pred.ravel(), ord=2) < 1e-12 else float("inf")
    return float(num / den)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Relative Position Error Curve (multi-run support)"
    )
    parser.add_argument(
        "--predicted_dir", type=str, help="Directory containing predicted VTP files"
    )
    parser.add_argument(
        "--exact_dir", type=str, help="Directory containing exact VTP files"
    )
    parser.add_argument(
        "--predicted_parent", type=str, help="Parent dir with run subdirs (predicted)"
    )
    parser.add_argument(
        "--exact_parent", type=str, help="Parent dir with run subdirs (exact)"
    )
    parser.add_argument(
        "--output_plot", type=str, default="rel_pos_curve.png", help="Output plot path"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional CSV (timestep, mean, std, count)",
    )
    return parser.parse_args()


_integer_re = re.compile(r"(\d+)")


def extract_timestep_from_name(filename: str) -> int:
    base = os.path.basename(filename)
    matches = _integer_re.findall(base)
    if not matches:
        raise ValueError(f"No integer timestep found in filename: {filename}")
    return int(matches[-1])


def discover_runs(parent_dir: str) -> List[str]:
    runs = []
    if not parent_dir or not os.path.isdir(parent_dir):
        return runs
    for root, dirs, files in os.walk(parent_dir):
        if any(f.endswith(".vtp") for f in files):
            runs.append(root)
    return sorted(set(runs))


def collect_vtps_by_timestep(dir_path: str) -> Dict[int, str]:
    vtps: Dict[int, str] = {}
    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".vtp")])
    for f in files:
        ts = extract_timestep_from_name(f)
        vtps[ts] = os.path.join(dir_path, f)
    return vtps


def compute_single_run_relpos(pred_dir: str, exact_dir: str) -> Dict[int, float]:
    pred_map = collect_vtps_by_timestep(pred_dir)
    exact_map = collect_vtps_by_timestep(exact_dir)
    results: Dict[int, float] = {}
    for ts in sorted(set(pred_map).intersection(exact_map)):
        x_pred = read_vtp_points(pred_map[ts])
        x_true = read_vtp_points(exact_map[ts])
        val = relative_position_error(x_pred, x_true)
        if np.isfinite(val):
            results[ts] = val
    return results


def aggregate_runs(predicted_parent: str, exact_parent: str) -> Dict[int, List[float]]:
    pred_runs = discover_runs(predicted_parent)
    exact_runs = discover_runs(exact_parent)

    def rel(path, parent):
        return os.path.relpath(path, parent)

    exact_rel_to_dir = {rel(d, exact_parent): d for d in exact_runs}
    aggregated: Dict[int, List[float]] = defaultdict(list)

    for pred_run_dir in pred_runs:
        key = rel(pred_run_dir, predicted_parent)
        exact_run_dir = exact_rel_to_dir.get(key)
        if exact_run_dir is None:
            continue
        per_run = compute_single_run_relpos(pred_run_dir, exact_run_dir)
        for ts, v in per_run.items():
            aggregated[ts].append(v)

    return aggregated


def maybe_single_run(predicted_dir: str, exact_dir: str) -> Dict[int, List[float]]:
    if not predicted_dir or not exact_dir:
        return {}
    if not (os.path.isdir(predicted_dir) and os.path.isdir(exact_dir)):
        return {}
    has_pred = any(f.endswith(".vtp") for f in os.listdir(predicted_dir))
    has_exact = any(f.endswith(".vtp") for f in os.listdir(exact_dir))
    if not (has_pred and has_exact):
        return {}
    per_run = compute_single_run_relpos(predicted_dir, exact_dir)
    aggregated: Dict[int, List[float]] = defaultdict(list)
    for ts, v in per_run.items():
        aggregated[ts].append(v)
    return aggregated


def save_csv(
    path: str,
    timesteps: List[int],
    means: List[float],
    stds: List[float],
    counts: List[int],
) -> None:
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestep",
                "mean_relative_position_error",
                "std_relative_position_error",
                "num_runs",
            ]
        )
        for t, m, s, c in zip(timesteps, means, stds, counts):
            w.writerow([t, m, s, c])


def main():
    args = parse_args()

    aggregated: Dict[int, List[float]] = {}
    if args.predicted_parent and args.exact_parent:
        aggregated = aggregate_runs(args.predicted_parent, args.exact_parent)
    if not aggregated and args.predicted_dir and args.exact_dir:
        aggregated = maybe_single_run(args.predicted_dir, args.exact_dir)
    if not aggregated:
        raise SystemExit(
            "No paired runs or .vtp files found. Provide valid parent or single-run directories."
        )

    timesteps = sorted(aggregated.keys())
    if not timesteps:
        raise SystemExit("No valid timesteps found.")

    timesteps_plot = [ts + 1 for ts in timesteps]

    means: List[float] = []
    stds: List[float] = []
    counts: List[int] = []
    for ts in timesteps:
        vals = np.array(aggregated[ts], dtype=float)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
        counts.append(int(vals.size))

    plt.figure(figsize=(10, 6))
    plt.plot(
        timesteps_plot,
        means,
        "-o",
        color="red",
        label="Relative $L^2$ position error for the test dataset",
    )
    lower = np.clip(np.array(means) - np.array(stds), 0.0, None)
    upper = np.array(means) + np.array(stds)
    plt.fill_between(
        timesteps_plot, lower, upper, color="red", alpha=0.2, label="±1 Std Dev"
    )
    plt.xlabel("Time Step")
    plt.ylabel("Relative $L^2$ position error")
    plt.title("Relative $L^2$ Position Error")
    plt.grid(True)
    plt.legend()
    plt.savefig(args.output_plot, dpi=300)
    print(f"Saved plot to {args.output_plot}")

    if args.output_csv:
        save_csv(args.output_csv, timesteps_plot, means, stds, counts)
        print(f"Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
