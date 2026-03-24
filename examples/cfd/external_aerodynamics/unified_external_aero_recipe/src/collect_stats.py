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
Collect per-field statistics from surface mesh datasets and write to parquet.

Thin CLI wrapper around :class:`~physicsnemo.datapipes.statistics.FieldStatisticsCollector`.
Loads dataset configs from YAML, instantiates the reader via Hydra, iterates
meshes, and writes cached parquet files.

By default, statistics are collected on **raw** meshes (no transforms).  Use
``--with-transforms`` to apply the deterministic portion of the pipeline
(e.g. ``InjectMetadata``, ``NonDimensionalizeByMetadata``, ``RenameMeshFields``)
so that the resulting statistics describe the non-dimensionalized fields that
the model actually sees.  Stochastic augmentations and terminal conversion
transforms are automatically skipped.

Usage::

    python -m src.collect_stats
    python -m src.collect_stats --output stats/
    python -m src.collect_stats --force
    python -m src.collect_stats --configs conf/dataset/drivaer_ml_surface.yaml
    python -m src.collect_stats --with-transforms
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

import sys

# Ensure sibling modules (nondim, datasets) are importable regardless of
# how this script is launched (python -m src.collect_stats, python src/collect_stats.py, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import physicsnemo.datapipes  # noqa: F401  (registers ${dp:...} resolvers)
from physicsnemo.datapipes import MeshDataset
from physicsnemo.datapipes.statistics import FieldStatisticsCollector

import nondim  # noqa: F401  (registers InjectMetadata, NonDimensionalizeByMetadata)

logging.basicConfig(level=logging.INFO, format="%(message)s")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "conf" / "dataset"

DEFAULT_CONFIGS = [
    CONFIG_DIR / "drivaer_ml_surface.yaml",
    CONFIG_DIR / "shift_suv_estate.yaml",
]

# Transforms to skip when running with --with-transforms.
# Stochastic augmentations would change stats on every run; terminal
# transforms convert Mesh -> TensorDict which the collector can't handle;
# NormalizeMeshFields would be self-referential.
_SKIP_TRANSFORMS = {
    "RandomRotateMesh",
    "RandomTranslateMesh",
    "RandomScaleMesh",
    "SubsampleMesh",
    "MeshToTensorDict",
    "RestructureTensorDict",
    "ComputeCellCentroids",
    "NormalizeMeshFields",
}


def _build_deterministic_transforms(cfg) -> list:
    """Instantiate only the deterministic, non-terminal transforms from a config."""
    from datasets import _inject_metadata_into_transform

    metadata = OmegaConf.to_container(
        OmegaConf.select(cfg, "metadata", default=OmegaConf.create({})),
        resolve=True,
    )
    transforms = []
    if "transforms" not in cfg.pipeline or not cfg.pipeline.transforms:
        return transforms
    for t_cfg in cfg.pipeline.transforms:
        target = t_cfg.get("_target_", "")
        class_name = target.split(":")[-1] if ":" in target else target.split(".")[-1]
        if class_name in _SKIP_TRANSFORMS:
            continue
        if metadata:
            t_cfg = _inject_metadata_into_transform(t_cfg, metadata)
        transforms.append(hydra.utils.instantiate(t_cfg))
    return transforms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect surface mesh field statistics to parquet"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stats",
        help="Output directory for parquet files (one per dataset)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cached stats exist",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        type=str,
        default=None,
        help="YAML config file paths (default: all in conf/dataset/)",
    )
    parser.add_argument(
        "--with-transforms",
        action="store_true",
        help=(
            "Apply deterministic pipeline transforms (e.g. "
            "InjectMetadata, NonDimensionalizeByMetadata, RenameMeshFields) "
            "before collecting stats.  Stochastic augmentations and terminal "
            "transforms are automatically skipped."
        ),
    )
    args = parser.parse_args()

    config_paths = [Path(p) for p in args.configs] if args.configs else DEFAULT_CONFIGS
    output_dir = Path(args.output)
    suffix = "_transformed" if args.with_transforms else ""

    t0 = time.perf_counter()

    for path in config_paths:
        if not path.exists():
            print(f"Config not found: {path}")
            continue

        cfg = OmegaConf.load(path)
        name = OmegaConf.select(cfg, "name", default=path.stem)
        datadir = OmegaConf.select(cfg, "train_datadir", default="")

        if not Path(datadir).exists():
            print(f"Skipping {name}: data dir {datadir} not found")
            continue

        reader = hydra.utils.instantiate(cfg.pipeline.reader)

        transforms = None
        if args.with_transforms:
            transforms = _build_deterministic_transforms(cfg)
            skipped = [
                t.get("_target_", "").split(":")[-1]
                for t in (cfg.pipeline.transforms or [])
                if t.get("_target_", "").split(":")[-1] in _SKIP_TRANSFORMS
            ]
            print(f"  Applying transforms (skipped: {skipped})")

        dataset = MeshDataset(reader, transforms=transforms)

        parquet_path = output_dir / f"{name}{suffix}.parquet"
        collector = FieldStatisticsCollector(
            output_path=parquet_path,
            force=args.force,
        )

        print(f"\n{name}: {len(dataset)} samples -> {parquet_path}")
        table = collector.collect(dataset)
        print(f"  {table.num_rows} rows, {table.num_columns} columns")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    with torch.no_grad():
        main()
