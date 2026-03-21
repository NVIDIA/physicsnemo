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
Read parquet statistics and recommend normalization parameters.

When aggregate (Welford) parquet files are present alongside per-sample
files, the script uses exact global moments (mean, std, skewness,
kurtosis) computed over every data point.  Otherwise it falls back to
sample-level approximations.

Usage::

    python -m src.recommend_normalization
    python -m src.recommend_normalization --fields pressure wss
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COMP_LABELS = {0: "x", 1: "y", 2: "z", 3: "w"}


def _load_per_sample(stats_dir: Path) -> pd.DataFrame:
    """Load per-sample parquet files (excluding *_aggregate.parquet)."""
    files = sorted(f for f in stats_dir.glob("*.parquet") if "_aggregate" not in f.stem)
    if not files:
        raise FileNotFoundError(f"No per-sample parquet files in {stats_dir}")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df["dataset"] = f.stem
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _load_aggregates(stats_dir: Path) -> pd.DataFrame | None:
    """Load Welford aggregate parquet files if they exist."""
    files = sorted(stats_dir.glob("*_aggregate.parquet"))
    if not files:
        return None
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        stem = f.stem.replace("_aggregate", "")
        df["dataset"] = stem
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _comp_suffix(component: int, n_components: int | None = None) -> str:
    if n_components is not None and n_components <= 1:
        return ""
    if component < 0:
        return ""
    return f"[{COMP_LABELS.get(component, str(component))}]"


def _filter_df(df: pd.DataFrame, fields: list[str] | None) -> pd.DataFrame:
    if fields is None:
        return df
    pattern = "|".join(fields)
    return df[df["field_key"].str.contains(pattern, case=False)]


def _fmt(v: float, width: int = 10) -> str:
    if v == 0:
        return f"{'0':>{width}}"
    if abs(v) >= 1e5 or abs(v) < 1e-3:
        return f"{v:>{width}.3e}"
    return f"{v:>{width}.4f}"


def _print_aggregate_report(agg_df: pd.DataFrame, per_sample_df: pd.DataFrame) -> None:
    """Print report using exact Welford aggregates."""
    for (field_key, comp), agg_group in agg_df.groupby(["field_key", "component"]):
        n_comp_guess = agg_group.iloc[0].get("n_components", None)
        if n_comp_guess is None:
            n_comp_guess = 1 if comp < 0 else 3
        display = f"{field_key}{_comp_suffix(comp, n_comp_guess)}"

        # Merge across datasets for this field+component
        total_count = int(agg_group["count"].sum())
        datasets = ", ".join(sorted(agg_group["dataset"].unique()))

        # Weighted merge of mean/var across datasets
        counts = agg_group["count"].values.astype(float)
        means = agg_group["mean"].values
        varis = agg_group["var"].values
        total = counts.sum()

        global_mean = (counts * means).sum() / total
        # Parallel variance merge
        global_var = (counts * varis).sum() / total + (
            counts * (means - global_mean) ** 2
        ).sum() / total
        global_std = global_var**0.5
        global_min = agg_group["min"].min()
        global_max = agg_group["max"].max()
        global_abs_mean = (counts * agg_group["abs_mean"].values).sum() / total

        # Skewness/kurtosis are exact per-dataset but need care for cross-dataset.
        # Show per-dataset values when multiple datasets exist.
        skew_vals = agg_group["skewness"].values
        kurt_vals = agg_group["kurtosis"].values

        print(f"\n{'=' * 82}")
        print(f"  {display}   ({total_count:,} total points from: {datasets})")
        print(f"{'=' * 82}")
        print(f"  exact global moments (Welford):")
        print(
            f"    mean={_fmt(global_mean)}  std={_fmt(global_std)}  "
            f"range=[{_fmt(global_min)}, {_fmt(global_max)}]"
        )
        print(f"    abs_mean={_fmt(global_abs_mean)}")

        # Per-dataset breakdown
        ds_list = sorted(agg_group["dataset"].unique())
        if len(ds_list) > 1:
            print(f"  per-dataset:")
            for _, row in agg_group.iterrows():
                print(
                    f"    {row['dataset']}: "
                    f"mean={_fmt(row['mean'])}  std={_fmt(row['std'])}  "
                    f"range=[{_fmt(row['min'])}, {_fmt(row['max'])}]  "
                    f"skew={_fmt(row['skewness'], 7)}  kurt={_fmt(row['kurtosis'], 7)}"
                )
        else:
            row = agg_group.iloc[0]
            print(
                f"    skewness={_fmt(row['skewness'])}  "
                f"excess_kurtosis={_fmt(row['kurtosis'])}"
            )

        # Normalization schemes
        schemes = _compute_schemes_from_exact(
            global_mean, global_std, global_min, global_max, global_abs_mean
        )
        _print_scheme_table(schemes)


def _compute_schemes_from_exact(
    mean: float, std: float, vmin: float, vmax: float, abs_mean: float
) -> list[dict]:
    schemes = []

    sigma = std if std > 0 else 1.0
    schemes.append(
        {
            "scheme": "z-score",
            "formula": "(x - mean) / std",
            "params": f"mean={_fmt(mean)} std={_fmt(sigma)}",
            "post_mean": 0.0,
            "post_std": 1.0,
            "post_min": (vmin - mean) / sigma,
            "post_max": (vmax - mean) / sigma,
        }
    )

    center = (vmax + vmin) / 2
    hr = (vmax - vmin) / 2
    if hr == 0:
        hr = 1.0
    schemes.append(
        {
            "scheme": "min-max",
            "formula": "(x - center) / half_range",
            "params": f"center={_fmt(center)} hr={_fmt(hr)}",
            "post_mean": (mean - center) / hr,
            "post_std": std / hr if hr > 0 else 0.0,
            "post_min": -1.0,
            "post_max": 1.0,
        }
    )

    max_abs = max(abs(vmin), abs(vmax))
    if max_abs == 0:
        max_abs = 1.0
    schemes.append(
        {
            "scheme": "max-abs",
            "formula": "x / max|x|",
            "params": f"scale={_fmt(max_abs)}",
            "post_mean": mean / max_abs,
            "post_std": std / max_abs,
            "post_min": vmin / max_abs,
            "post_max": vmax / max_abs,
        }
    )

    return schemes


def _print_scheme_table(schemes: list[dict]) -> None:
    print()
    hdr = (
        f"  {'scheme':<10} {'params':<36} "
        f"{'mean':>8} {'std':>8} {'min':>10} {'max':>10}"
    )
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")
    for s in schemes:
        print(
            f"  {s['scheme']:<10} {s['params']:<36} "
            f"{_fmt(s['post_mean'], 8)} {_fmt(s['post_std'], 8)} "
            f"{_fmt(s['post_min'])} {_fmt(s['post_max'])}"
        )


def _print_fallback_report(df: pd.DataFrame) -> None:
    """Fall back to per-sample approximations (no aggregate files found)."""
    print("  [note: no aggregate files found; using per-sample approximations]")
    for (field_key, comp), group in df.groupby(["field_key", "component"]):
        n_comp = group["n_components"].iloc[0]
        display = f"{field_key}{_comp_suffix(comp, n_comp)}"
        n = len(group)
        datasets = ", ".join(sorted(group["dataset"].unique()))

        print(f"\n{'=' * 82}")
        print(f"  {display}   ({n} samples from: {datasets})")
        print(f"{'=' * 82}")

        means = group["mean"].values
        stds = group["std"].values
        mins = group["min"].values
        maxs = group["max"].values
        n_vals = group["n_spatial"].values

        mu = (means * n_vals).sum() / n_vals.sum()
        within_var = (stds**2 * n_vals).sum() / n_vals.sum()
        between_var = means.var()
        sigma = (within_var + between_var) ** 0.5

        print(
            f"  approx:  mean={_fmt(mu)}  std={_fmt(sigma)}  "
            f"range=[{_fmt(mins.min())}, {_fmt(maxs.max())}]"
        )

        schemes = _compute_schemes_from_exact(
            mu,
            sigma,
            mins.min(),
            maxs.max(),
            (group["abs_mean"].values * n_vals).sum() / n_vals.sum(),
        )
        _print_scheme_table(schemes)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend normalization parameters from collected statistics"
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default="stats",
        help="Directory containing parquet stats files",
    )
    parser.add_argument(
        "--fields",
        nargs="*",
        type=str,
        default=None,
        help="Filter to fields containing these substrings (e.g. pressure wss)",
    )
    args = parser.parse_args()

    stats_dir = Path(args.stats_dir)
    per_sample_df = _load_per_sample(stats_dir)
    per_sample_df = _filter_df(per_sample_df, args.fields)

    agg_df = _load_aggregates(stats_dir)
    if agg_df is not None:
        agg_df = _filter_df(agg_df, args.fields)

    if per_sample_df.empty:
        print("No matching fields found.")
        return

    n_datasets = per_sample_df["dataset"].nunique()
    n_fields = per_sample_df.groupby(["field_key", "component"]).ngroups
    n_samples = per_sample_df["sample_index"].nunique()
    print(
        f"Loaded {n_samples} samples, {n_fields} field components "
        f"from {n_datasets} dataset(s)"
    )

    if agg_df is not None and not agg_df.empty:
        print("  [using exact Welford aggregates]")
        _print_aggregate_report(agg_df, per_sample_df)
    else:
        _print_fallback_report(per_sample_df)

    print()


if __name__ == "__main__":
    main()
