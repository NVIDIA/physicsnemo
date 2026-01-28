#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Plot multi-panel metric comparison (RMSE, CRPS, spread, etc.)."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

METRIC_LABELS = {
    "crps": "CRPS",
    "rmse_ens": "RMSE",
    "rmse_m0": "RMSE",
    "spread": "Spread",
    "ssr": "Spread-Skill Ratio",
}


def main():
    parser = argparse.ArgumentParser(description="Plot metric comparison panels")
    parser.add_argument("--stats", nargs="+", required=True, help="Metrics .nc files")
    parser.add_argument("--labels", nargs="+", required=True, help="System labels")
    parser.add_argument(
        "--metric",
        default="rmse_ens",
        help="Metric (rmse_ens, rmse_m0, crps, spread, ssr)",
    )
    parser.add_argument("--fields", nargs="+", default=None, help="Fields to plot")
    parser.add_argument("--output_path", default="panel.pdf", help="Output file")
    parser.add_argument("--max_lead_time", type=int, default=None)
    args = parser.parse_args()

    if len(args.stats) != len(args.labels):
        raise ValueError("Number of stats files must match number of labels")

    # Load datasets
    datasets = [xr.open_dataset(f, decode_timedelta=False) for f in args.stats]

    # Auto-detect fields if not specified
    if args.fields is None:
        field_sets = [set(ds.field.values) for ds in datasets if "field" in ds.dims]
        common = set.intersection(*field_sets) if field_sets else set()
        preferred = ["Z500", "T850", "U500", "Q700", "msl", "t2m", "u10m", "tcwv"]
        fields = [f for f in preferred if f in common]
        fields.extend(sorted(f for f in common if f not in fields))
        fields = fields[:8]
    else:
        fields = args.fields

    if not fields:
        raise ValueError("No common fields found across datasets")

    # Setup figure
    ncols = min(4, len(fields))
    nrows = (len(fields) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.5 * nrows), sharex=True)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    # Plot each field
    for idx, field in enumerate(fields):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        for ds, label in zip(datasets, args.labels):
            # Get lead time
            lead = ds["lead_time"]
            if np.issubdtype(lead.dtype, np.timedelta64):
                x = (lead / np.timedelta64(1, "h")).astype(int).values
            else:
                x = lead.values

            # Get metric data (case-insensitive field match)
            if args.metric not in ds:
                print(f"Metric {args.metric} not in {label}, skipping")
                continue

            ds_fields = {str(f).lower(): str(f) for f in ds.field.values}
            field_key = ds_fields.get(field.lower())
            if field_key is None:
                continue

            y = ds[args.metric].sel(field=field_key).values
            if y.ndim > 1:
                y = y.mean(axis=tuple(range(y.ndim - 1)))

            if args.max_lead_time:
                mask = x <= args.max_lead_time
                x, y = x[mask], y[mask]

            ax.plot(x, y, label=label, linewidth=1.5)

        ax.set_title(field)
        ax.grid(True, alpha=0.3)
        if row == nrows - 1:
            ax.set_xlabel("Lead time (hours)")
        if col == 0:
            ax.set_ylabel(METRIC_LABELS.get(args.metric, args.metric.upper()))

    # Hide unused axes
    for idx in range(len(fields), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    axes[0, 0].legend(loc="best", fontsize=10)
    plt.tight_layout()
    fig.savefig(args.output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
