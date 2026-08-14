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

import csv
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import zarr


def read_daily_stats(csv_path: str) -> Dict[str, List[Tuple[float, float]]]:
    grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            var = row["variable"]
            x_val = float(row["x"])
            x2_val = float(row["x2"])
            grouped[var].append((x_val, x2_val))
    return grouped


def pooled_mean_variance(
    entries: List[Tuple[float, float]], n_per_entry: int
) -> Tuple[float, float, int]:
    """Compute pooled mean/variance across entries with equal sample size per entry.

    - entries: list of (sum_x_i, sum_x2_i) where each statistic was computed over n_per_entry samples
    - returns: (mean, variance, total_samples)
    """
    if not entries:
        return float("nan"), float("nan"), 0

    x = np.array([m for m, _ in entries], dtype=np.float64)
    x2 = np.array([s for _, s in entries], dtype=np.float64)
    n = int(n_per_entry)
    k = int(len(entries))
    total_n = n * k

    mu = np.sum(x) / total_n
    variance = np.sum(x2) / total_n - mu**2
    return mu, variance, total_n


def main() -> None:
    root = zarr.open_group(
        store="s3://hrrr-surface-sda/zarr-v1",
        mode="r",
        storage_options={
            "endpoint_url": "https://pdx.s8k.io",
            "profile": "physicsnemo"
        },
    )
    n_grid = root["lat"].shape[0] * root["lat"].shape[1]
    grouped = read_daily_stats("daily_random_moments.csv")

    out_path = "stats.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["variable", "log_scale", "n_samples", "mean", "variance", "eps"]
        )
        for var, stats in sorted(grouped.items()):
            mu, var_val, total_n = pooled_mean_variance(stats, n_grid)
            writer.writerow(
                [
                    var,
                    (var in ["tp", "aerot"]),
                    f"{total_n:.16g}",
                    f"{mu:.16g}",
                    f"{var_val:.16g}",
                    1e-8,
                ]
            )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
