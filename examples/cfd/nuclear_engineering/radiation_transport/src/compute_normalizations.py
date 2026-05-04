# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Standalone CLI to compute flux + material statistics over a zarr root.

Run this once before training to produce the two YAML statistics files the
training pipeline expects:

    <output_dir>/<case>_flux_stats.yaml
    <output_dir>/<case>_material_stats.yaml

Usage::

    python compute_normalizations.py \\
        --data_path <DATA_ROOT>/lattice \\
        --case_type lattice \\
        --split_file <DATA_ROOT>/splits/lattice_splits.json \\
        --output_dir <DATA_ROOT>/stats

The flux statistics walk the training split of the dataset, log-clip the raw
``scalar_flux`` field, and accumulate (mean, std, min, max) plus the clip
threshold the training pipeline must use. The material statistics walk the
training split through a minimal transform pipeline that derives the
per-point ``physical_properties`` tensor (sigma_a, sigma_s, sigma_t, Q) and
records (mean, std, min, max) for each component.

The on-disk YAML schema matches the originals so that ``load_flux_stats`` /
``load_material_stats`` in ``dataset.py`` consume them unchanged.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

# Flat-import shim: when invoked as ``python compute_normalizations.py`` the
# script's own directory is already on ``sys.path``; when invoked from
# elsewhere we make sure sibling modules are importable.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dataset import RTEBaseDataset  # noqa: E402
from loader import RTEDataPipe  # noqa: E402
from material import MaterialPropertyExtractor  # noqa: E402
from transforms import (  # noqa: E402
    Compose,
    RTEFluxLogClip,
    SpatialSampler,
    SteadyStateSampler,
)


# =========================================================================
# Flux statistics
# =========================================================================
#
# Walks the training split of ``RTEBaseDataset`` (no transforms, no
# adapter). For each simulation, applies the same log-clip preprocessing
# the training pipeline uses, and accumulates global mean / std / min / max
# in single precision. Output schema matches the legacy
# ``compute_flux_statistics.py`` so the existing ``load_flux_stats`` reader
# works unchanged.


def compute_flux_statistics(
    data_path: Path,
    case_type: str,
    output_file: Path,
    split_file: Path,
    clip_threshold: float = 1e-8,
) -> Dict[str, float]:
    """Compute flux normalization statistics from the training split.

    Args:
        data_path: path to the zarr stores for one case.
        case_type: ``"lattice"`` or ``"hohlraum"``.
        output_file: destination YAML path.
        split_file: split JSON used to select the training split.
        clip_threshold: minimum flux value before ``log10``.
    Returns:
        The statistics dict written to ``output_file``.
    """
    print(f"Computing flux statistics for {case_type} [steady state]")
    print(f"Data path: {data_path}")
    print(f"Split file: {split_file}")

    dataset = RTEBaseDataset(
        data_path=data_path,
        case_type=case_type,
        phase="train",
        split_file=split_file,
        load_material_properties=False,
        load_geometric_features=False,
    )

    print(f"\nProcessing {len(dataset)} training simulations...")

    n_samples = 0
    sum_log_flux = 0.0
    sum_log_flux_sq = 0.0
    min_log_flux = float("inf")
    max_log_flux = float("-inf")

    for i in range(len(dataset)):
        sample = dataset[i]
        flux = sample["scalar_flux"]
        if isinstance(flux, torch.Tensor):
            flux = flux.detach().cpu().numpy()
        flux = np.asarray(flux)

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

    stats["note"] = "computed from first and final snapshots only (steady state)"

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


# =========================================================================
# Material statistics
# =========================================================================
#
# Walks the training split through a minimal transform pipeline:
#
#     RTEFluxLogClip -> SteadyStateSampler -> MaterialPropertyExtractor -> SpatialSampler
#
# The flux log-clip step is required because the dataset reader produces a
# steady-state flux tensor; the sampler picks the first/final pair, the
# material extractor produces ``physical_properties`` with shape (N, 4), and
# ``SpatialSampler`` subsamples to a fixed point count for speed. Per-property
# stats are written in the schema the existing ``load_material_stats`` reader
# expects.


def compute_material_statistics(
    data_path: Path,
    case_type: str,
    output_file: Path,
    flux_stats_file: Path,
    split_file: Path,
    clip_threshold: float = 1e-8,
    num_spatial_points: int = 2048,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute per-property material statistics from the training split.

    Args:
        data_path: path to the zarr stores for one case.
        case_type: ``"lattice"`` or ``"hohlraum"``.
        output_file: destination YAML path.
        flux_stats_file: path to the flux stats YAML produced by
            :func:`compute_flux_statistics`. Required because the transform
            pipeline runs ``RTEFluxLogClip`` first.
        split_file: split JSON used to select the training split.
        clip_threshold: flux clip threshold used by the flux transform.
        num_spatial_points: number of points per simulation drawn by
            ``SpatialSampler``.
        seed: RNG seed for spatial sampling.

    Returns:
        The nested statistics dict written to ``output_file``.
    """
    print(f"\nComputing material statistics for {case_type}")
    print(f"Data path: {data_path}")
    print(f"Flux stats: {flux_stats_file}")
    print(f"Split file: {split_file}")

    if not Path(flux_stats_file).exists():
        raise FileNotFoundError(
            f"Flux statistics file not found: {flux_stats_file}\n"
            "Compute flux statistics before material statistics."
        )

    transforms = Compose(
        [
            RTEFluxLogClip(
                normalization_stats_file=flux_stats_file,
                clip_threshold=clip_threshold,
            ),
            SteadyStateSampler(),
            MaterialPropertyExtractor(case_type=case_type),
            SpatialSampler(num_points=num_spatial_points, seed=seed),
        ]
    )

    print("\nCreating dataset (this may take a moment)...")
    dataset = RTEDataPipe(
        data_path=data_path,
        transforms=transforms,
        adapter=None,
        case_type=case_type,
        phase="train",
        split_file=split_file,
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    print("\nAccumulating physical_properties...")
    all_sigma_a, all_sigma_s, all_sigma_t, all_Q = [], [], [], []

    for i in range(len(dataset)):
        sample = dataset.get_transformed_sample(i)
        if "physical_properties" not in sample:
            raise KeyError(
                f"Sample {i} is missing 'physical_properties'. "
                "MaterialPropertyExtractor did not produce expected output."
            )
        props = sample["physical_properties"]
        if isinstance(props, torch.Tensor):
            props = props.detach().cpu().numpy()
        all_sigma_a.append(props[:, 0])
        all_sigma_s.append(props[:, 1])
        all_sigma_t.append(props[:, 2])
        all_Q.append(props[:, 3])
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} samples")

    all_sigma_a = np.concatenate(all_sigma_a)
    all_sigma_s = np.concatenate(all_sigma_s)
    all_sigma_t = np.concatenate(all_sigma_t)
    all_Q = np.concatenate(all_Q)

    stats = {
        "sigma_a": {
            "mean": float(np.mean(all_sigma_a)),
            "std": float(np.std(all_sigma_a)),
            "min": float(np.min(all_sigma_a)),
            "max": float(np.max(all_sigma_a)),
        },
        "sigma_s": {
            "mean": float(np.mean(all_sigma_s)),
            "std": float(np.std(all_sigma_s)),
            "min": float(np.min(all_sigma_s)),
            "max": float(np.max(all_sigma_s)),
        },
        "sigma_t": {
            "mean": float(np.mean(all_sigma_t)),
            "std": float(np.std(all_sigma_t)),
            "min": float(np.min(all_sigma_t)),
            "max": float(np.max(all_sigma_t)),
        },
        "Q": {
            "mean": float(np.mean(all_Q)),
            "std": float(np.std(all_Q)),
            "min": float(np.min(all_Q)),
            "max": float(np.max(all_Q)),
        },
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


# =========================================================================
# CLI entry
# =========================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute flux + material normalization statistics over a zarr root. "
            "Emits two YAML files: <case>_flux_stats.yaml and "
            "<case>_material_stats.yaml in the output directory."
        )
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to the zarr root for one case (e.g. <DATA_ROOT>/lattice).",
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
    parser.add_argument(
        "--num_spatial_points",
        type=int,
        default=2048,
        help="Points per simulation for material stats subsampling (default: 2048).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for material-stats sampling (default: 42).",
    )
    return parser.parse_args()


def main() -> int:
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
        flux_stats_file=flux_output,
        split_file=args.split_file,
        clip_threshold=args.clip_threshold,
        num_spatial_points=args.num_spatial_points,
        seed=args.seed,
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"  Flux stats:     {flux_output}")
    print(f"  Material stats: {material_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
