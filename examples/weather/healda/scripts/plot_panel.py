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
"""
Plot multi-panel metric comparison (CRPS, RMSE, ACC, etc.).
Self-contained - no external utils dependencies.
"""

import argparse
import re
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import MaxNLocator

# =============================================================================
# Constants
# =============================================================================

SYSTEM_COLORS = {
    "DA": "#009E73",
    "ERA5": "#000000",
    "IFS": "#56B4E9",
    "AURORA": "#0072B2",
    "PANGU": "#CC79A7",
    "FENGWU": "#E69F00",
    "GFS": "#D55E00",
    "FCN3": "#009E73",
}

METRIC_LABELS = {
    "crps": "CRPS",
    "rmse": "RMSE",
    "rmse_ens": "RMSE",
    "rmse_member": "RMSE",
    "acc": "ACC",
    "acc_ens": "ACC",
    "acc_member": "ACC",
    "ssr": "Spread-Skill Ratio",
    "spread": "Spread",
}

FIELD_UNITS = {
    "z": "m² s⁻²",
    "t": "K",
    "u": "m s⁻¹",
    "v": "m s⁻¹",
    "msl": "Pa",
    "t2m": "K",
    "u10m": "m s⁻¹",
    "v10m": "m s⁻¹",
    "tcwv": "kg m⁻²",
}


# =============================================================================
# Utility Functions (inlined)
# =============================================================================


def get_system_color(label: str) -> str:
    """Get color for a system based on its label."""
    label_upper = label.upper()
    for key in SYSTEM_COLORS:
        if key in label_upper:
            return SYSTEM_COLORS[key]
    return plt.rcParams["axes.prop_cycle"].by_key()["color"][0]


def get_field_units(field: str) -> str:
    """Get units for a field."""
    field_lower = field.lower()
    if field_lower in FIELD_UNITS:
        return FIELD_UNITS[field_lower]
    match = re.match(r"^([a-z]+)\d*$", field_lower)
    if match:
        base_field = match.group(1)
        if base_field in FIELD_UNITS:
            return FIELD_UNITS[base_field]
    return ""


def load_metrics(filepath: str) -> xr.Dataset:
    """Load metrics NetCDF file."""
    return xr.open_dataset(filepath, decode_timedelta=False)


def lead_hours_from_ds(ds: xr.Dataset) -> np.ndarray:
    """Extract lead time in hours from dataset."""
    lead = ds["lead_time"]
    if np.issubdtype(lead.dtype, np.timedelta64):
        return (lead / np.timedelta64(1, "h")).astype("int64").values
    return np.asarray(lead.values, dtype=np.int64)


def resolve_field_name(ds: xr.Dataset, field: str) -> str:
    """Resolve field name with case-insensitive matching."""
    if "field" not in ds.dims:
        raise ValueError("Dataset does not have a 'field' dimension")
    ds_fields = {str(f).lower(): str(f) for f in ds.field.values}
    resolved = ds_fields.get(field.lower())
    if resolved is None:
        raise KeyError(
            f"Field '{field}' not found. Available: {sorted(ds_fields.values())}"
        )
    return resolved


def extract_metric_curve(
    ds: xr.Dataset, metric_name: str, field: str, system_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract metric curve for field. Returns (lead_hours, values)."""
    lead_hours = lead_hours_from_ds(ds)
    field_resolved = resolve_field_name(ds, field)

    if "ssr" in metric_name.lower():
        if "spread_error" not in ds:
            return lead_hours, np.array([])
        metric = ds["spread_error"].sel(field=field_resolved)
    elif metric_name not in ds:
        return lead_hours, np.array([])
    else:
        metric = ds[metric_name].sel(field=field_resolved)

    if "system" in metric.dims:
        metric = metric.isel(system=system_idx)

    dims_to_avg = [d for d in metric.dims if d != "lead_time"]
    if dims_to_avg:
        metric = metric.mean(dim=dims_to_avg)

    return lead_hours, metric.values


def create_figure(
    nrows: int, ncols: int, width: float = 12.0, height_per_row: float = 2.5
) -> Tuple[plt.Figure, np.ndarray]:
    """Create figure with proper sizing."""
    height = min(height_per_row * nrows, 11.0)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, height), sharex=True, constrained_layout=True
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    return fig, axes


def add_subplot_labels(axes: np.ndarray, x_offset: float = -0.12, fontsize: int = 10):
    """Add (a), (b), (c), ... labels to panels."""
    for idx, ax in enumerate(axes.flat):
        if ax.get_visible():
            ax.text(
                x_offset,
                1.05,
                chr(97 + idx),
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight="bold",
                va="bottom",
                ha="right",
            )


def add_units_label(ax, units: str, fontsize: int = 9):
    """Add units label above axis."""
    if units:
        ax.text(
            0.0,
            1.02,
            f"[{units}]",
            transform=ax.transAxes,
            fontsize=fontsize,
            va="bottom",
            ha="left",
        )


# =============================================================================
# Main Plotting
# =============================================================================


def plot_metric_panel(
    stats_files: List[str],
    labels: List[str],
    metric: str = "rmse_ens",
    fields: Optional[List[str]] = None,
    output_path: str = "panel.pdf",
    max_lead_time: Optional[int] = None,
):
    """
    Plot multi-panel metric comparison.

    Args:
        stats_files: List of paths to metric .nc files
        labels: Labels for each system
        metric: Metric to plot (rmse_ens, crps, acc_ens, etc.)
        fields: Fields to plot (auto-detect if None)
        output_path: Output file path
        max_lead_time: Maximum lead time in hours (None = all)
    """
    # Load all datasets
    datasets = [load_metrics(f) for f in stats_files]

    # Auto-detect fields if not specified
    if fields is None:
        field_sets = [set(ds.field.values) for ds in datasets if "field" in ds.dims]
        common = set.intersection(*field_sets) if field_sets else set()
        preferred = ["Z500", "T850", "U500", "Q700", "msl", "t2m", "u10m", "tcwv"]
        fields = [f for f in preferred if f in common]
        fields.extend(sorted(f for f in common if f not in fields))
        fields = fields[:8]

    if not fields:
        raise ValueError("No common fields found across datasets")

    # Setup figure
    nfields = len(fields)
    ncols = min(4, nfields)
    nrows = (nfields + ncols - 1) // ncols

    fig, axes = create_figure(nrows, ncols)

    # Plot each field
    for idx, field in enumerate(fields):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        for ds, label in zip(datasets, labels):
            try:
                x, y = extract_metric_curve(ds, metric, field)
                if len(y) == 0:
                    continue
                if max_lead_time:
                    mask = x <= max_lead_time
                    x, y = x[mask], y[mask]
                color = get_system_color(label)
                ax.plot(x, y, label=label, color=color, linewidth=1.5)
            except (KeyError, ValueError) as e:
                print(f"Skipping {field} for {label}: {e}")

        ax.set_title(field)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        units = get_field_units(field)
        add_units_label(ax, units)

        if row == nrows - 1:
            ax.set_xlabel("Lead time (hours)")
        if col == 0:
            ylabel = METRIC_LABELS.get(metric, metric.upper())
            ax.set_ylabel(ylabel)

    # Hide unused axes
    for idx in range(nfields, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Add labels and legend
    add_subplot_labels(axes)

    handles, lbls = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            lbls,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(4, len(lbls)),
            frameon=True,
        )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot metric comparison panels")
    parser.add_argument("--stats", nargs="+", required=True, help="Metrics .nc files")
    parser.add_argument("--labels", nargs="+", required=True, help="System labels")
    parser.add_argument("--metric", default="rmse_ens", help="Metric to plot")
    parser.add_argument("--fields", nargs="+", default=None, help="Fields to plot")
    parser.add_argument("--output_path", default="panel.pdf", help="Output file")
    parser.add_argument("--max_lead_time", type=int, default=None)
    args = parser.parse_args()

    if len(args.stats) != len(args.labels):
        raise ValueError("Number of stats files must match number of labels")

    plot_metric_panel(
        stats_files=args.stats,
        labels=args.labels,
        metric=args.metric,
        fields=args.fields,
        output_path=args.output_path,
        max_lead_time=args.max_lead_time,
    )


if __name__ == "__main__":
    main()
