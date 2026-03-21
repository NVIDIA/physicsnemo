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
raw meshes (no augmentation), and writes cached parquet files.

Usage::

    python -m src.collect_stats
    python -m src.collect_stats --output stats/
    python -m src.collect_stats --force
    python -m src.collect_stats --configs conf/dataset/drivaer_ml_surface.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

import physicsnemo.datapipes  # noqa: F401  (registers ${dp:...} resolvers)
from physicsnemo.datapipes import MeshDataset
from physicsnemo.datapipes.statistics import FieldStatisticsCollector

logging.basicConfig(level=logging.INFO, format="%(message)s")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "conf" / "dataset"

DEFAULT_CONFIGS = [
    CONFIG_DIR / "drivaer_ml_surface.yaml",
    CONFIG_DIR / "shift_suv_estate.yaml",
]


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
    args = parser.parse_args()

    config_paths = [Path(p) for p in args.configs] if args.configs else DEFAULT_CONFIGS
    output_dir = Path(args.output)

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
        dataset = MeshDataset(reader)

        parquet_path = output_dir / f"{name}.parquet"
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
