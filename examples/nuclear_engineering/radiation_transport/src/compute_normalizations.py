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

"""Standalone CLI to compute flux + material statistics over a mesh data root.

Run this once before training to produce the two YAML statistics files the
training pipeline expects:

    <output_dir>/<case>_flux_stats.yaml
    <output_dir>/<case>_material_stats.yaml

Usage::

    python src/compute_normalizations.py \\
        --data_path <DATA_ROOT>/lattice \\
        --case_type lattice \\
        --split_file <DATA_ROOT>/splits/lattice_splits.json \\
        --output_dir <DATA_ROOT>/stats

The flux statistics walk the training split of the dataset, log-clip the raw
``scalar_flux`` field, and accumulate (mean, std, min, max) plus the clip
threshold the training pipeline must use. The material statistics walk the
training split, read the precomputed ``sigma_a / sigma_s / sigma_t / Q``
fields from each store, and accumulate per-property (mean, std, min, max)
across all cells.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

from dataset import RTEBaseDataset
from transforms import MaterialPropertyExtractor


def compute_flux_statistics(
    data_path: Path,
    case_type: str,
    output_file: Path,
    split_file: Path,
    clip_threshold: float = 1e-8,
) -> Dict[str, float]:
    """Compute flux normalization statistics from the training split.

    Args:
        data_path: path to the mesh stores for one case.
        case_type: ``"lattice"`` or ``"hohlraum"``.
        output_file: destination YAML path.
        split_file: split JSON used to select the training split.
        clip_threshold: minimum flux value before ``log10``.
    Returns:
        The statistics dict written to ``output_file``.
    """
    print(f"Computing flux statistics for {case_type} [final time only]")
    print(f"Data path: {data_path}")
    print(f"Split file: {split_file}")

    dataset = RTEBaseDataset(
        data_path=data_path,
        case_type=case_type,
        phase="train",
        split_file=split_file,
    )

    print(f"\nProcessing {len(dataset)} training simulations...")

    n_samples = 0
    sum_log_flux = 0.0
    sum_log_flux_sq = 0.0
    min_log_flux = float("inf")
    max_log_flux = float("-inf")

    for i in range(len(dataset)):
        sample, _ = dataset[i]
        flux = sample["scalar_flux"]
        if isinstance(flux, torch.Tensor):
            flux = flux.detach().cpu().numpy()
        flux = np.asarray(flux)

        # ``scalar_flux`` from the reader is shape (T, n_cells) with T=2
        # (first + final snapshots). The target the model predicts is the
        # final-time only.
        if flux.ndim > 1:
            flux = flux[-1]

        # match training-pipeline preprocessing
        flux = np.clip(flux, clip_threshold, None)
        log_flux = np.log10(flux + clip_threshold)

        n = log_flux.size
        n_samples += n
        sum_log_flux += float(np.sum(log_flux))
        sum_log_flux_sq += float(np.sum(log_flux**2))
        min_log_flux = min(min_log_flux, float(np.min(log_flux)))
        max_log_flux = max(max_log_flux, float(np.max(log_flux)))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} simulations")

    mean = sum_log_flux / n_samples
    variance = (sum_log_flux_sq / n_samples) - (mean**2)
    std = float(np.sqrt(max(variance, 0.0)))

    stats = {
        "log_flux_mean": float(mean),
        "log_flux_std": float(std),
        "log_flux_min": float(min_log_flux),
        "log_flux_max": float(max_log_flux),
        "clip_threshold": float(clip_threshold),
        "num_samples": int(n_samples),
        "num_simulations": len(dataset),
        "case_type": case_type,
    }

    stats["note"] = "computed from the final-time snapshot only"

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)

    print("\nFlux statistics:")
    print(f"  Mean (log flux): {mean:.6f}")
    print(f"  Std  (log flux): {std:.6f}")
    print(f"  Min  (log flux): {min_log_flux:.6f}")
    print(f"  Max  (log flux): {max_log_flux:.6f}")
    print(f"  Total samples:   {n_samples:,}")
    print(f"\nSaved to: {output_file}")

    return stats


def compute_material_statistics(
    data_path: Path,
    case_type: str,
    output_file: Path,
    split_file: Path,
) -> Dict[str, Dict[str, float]]:
    """Compute per-property material statistics from the training split.

    Args:
        data_path: path to the mesh stores for one case.
        case_type: ``"lattice"`` or ``"hohlraum"``.
        output_file: destination YAML path.
        split_file: split JSON used to select the training split.

    Returns:
        The nested statistics dict written to ``output_file``.
    """
    print(f"\nComputing material statistics for {case_type}")
    print(f"Data path: {data_path}")
    print(f"Split file: {split_file}")

    dataset = RTEBaseDataset(
        data_path=data_path,
        case_type=case_type,
        phase="train",
        split_file=split_file,
    )
    extractor = MaterialPropertyExtractor()
    print(f"Dataset loaded: {len(dataset)} samples")

    print("\nAccumulating physical_properties...")

    # We track count, running mean, and M2 (sum of squared deviations from
    # the running mean); the population std is sqrt(M2 / count)
    prop_names = ("sigma_a", "sigma_s", "sigma_t", "Q")
    count = 0
    mean_running = np.zeros(len(prop_names), dtype=np.float64)
    m2_running = np.zeros(len(prop_names), dtype=np.float64)
    min_running = np.full(len(prop_names), np.inf, dtype=np.float64)
    max_running = np.full(len(prop_names), -np.inf, dtype=np.float64)

    for i in range(len(dataset)):
        td, _ = dataset[i]
        sample = extractor(td)
        props = sample["physical_properties"]
        if isinstance(props, torch.Tensor):
            props = props.detach().cpu().numpy()
        # Cast to float64 for the accumulator; the on-disk tensors are fp32.
        props = np.asarray(props, dtype=np.float64)
        n_i = props.shape[0]
        if n_i == 0:
            continue

        # Per-batch sufficient stats (mean and M2) for combination
        batch_mean = props.mean(axis=0)
        batch_m2 = ((props - batch_mean) ** 2).sum(axis=0)

        new_count = count + n_i
        delta = batch_mean - mean_running
        mean_running = mean_running + delta * (n_i / new_count)
        m2_running = m2_running + batch_m2 + (delta**2) * (count * n_i / new_count)
        count = new_count

        np.minimum(min_running, props.min(axis=0), out=min_running)
        np.maximum(max_running, props.max(axis=0), out=max_running)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} samples")

    if count == 0:
        raise RuntimeError(
            "compute_material_statistics: dataset produced zero cells; "
            "cannot compute stats."
        )

    std_running = np.sqrt(m2_running / count)

    stats = {
        name: {
            "mean": float(mean_running[j]),
            "std": float(std_running[j]),
            "min": float(min_running[j]),
            "max": float(max_running[j]),
        }
        for j, name in enumerate(prop_names)
    }

    print("\nMaterial statistics:")
    print("-" * 60)
    for prop_name, prop_stats in stats.items():
        print(f"{prop_name}:")
        for stat_name, value in prop_stats.items():
            print(f"  {stat_name:6s}: {value:10.4f}")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved to: {output_file}")

    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute flux + material normalization statistics over a mesh data "
            "root. Emits two YAML files: <case>_flux_stats.yaml and "
            "<case>_material_stats.yaml in the output directory."
        )
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to the mesh data root for one case (e.g. <DATA_ROOT>/lattice).",
    )
    parser.add_argument(
        "--case_type",
        type=str,
        required=True,
        choices=["lattice", "hohlraum"],
        help="Case type.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write the two YAML statistics files into.",
    )
    parser.add_argument(
        "--split_file",
        type=Path,
        required=True,
        help="Required split JSON; statistics are computed on its training split.",
    )
    parser.add_argument(
        "--clip_threshold",
        type=float,
        default=1e-8,
        help="Flux clip threshold used during log-transform (default: 1e-8).",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry: compute and write flux + material statistics YAMLs."""
    args = _parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    flux_output = output_dir / f"{args.case_type}_flux_stats.yaml"
    material_output = output_dir / f"{args.case_type}_material_stats.yaml"

    print("=" * 80)
    print("COMPUTE NORMALIZATIONS")
    print("=" * 80)

    compute_flux_statistics(
        data_path=args.data_path,
        case_type=args.case_type,
        output_file=flux_output,
        split_file=args.split_file,
        clip_threshold=args.clip_threshold,
    )

    compute_material_statistics(
        data_path=args.data_path,
        case_type=args.case_type,
        output_file=material_output,
        split_file=args.split_file,
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"  Flux stats:     {flux_output}")
    print(f"  Material stats: {material_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
