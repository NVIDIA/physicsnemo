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
Benchmark the external aerodynamics surface mesh datapipes.

Builds individual and combined datapipes from Hydra-instantiable YAML
configs and measures loading throughput, per-sample timing, and prints
data shape summaries.

Usage::

    python -m src.benchmark
    python -m src.benchmark --max-samples 20
    python -m src.benchmark --configs conf/dataset/drivaer_ml_surface.yaml
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

_RECIPE_ROOT = Path(__file__).resolve().parent.parent
if str(_RECIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECIPE_ROOT))

import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from physicsnemo.datapipes import DataLoader, FunctionCollator

from src.datasets import (
    build_surface_dataset,
    build_multi_surface_dataset,
    load_dataset_config,
)


CONFIG_DIR = Path(__file__).resolve().parent.parent / "conf" / "dataset"

DEFAULT_CONFIGS = [
    CONFIG_DIR / "drivaer_ml_surface.yaml",
    CONFIG_DIR / "shift_suv_estate.yaml",
]


def _component_stats_lines(t: torch.Tensor, prefix: str) -> list[str]:
    """Per-component mean/std/min/max lines for the last dimension."""
    f = t.float()
    if f.ndim < 2:
        f = f.unsqueeze(-1)
    n_comp = f.shape[-1]
    flat = f.reshape(-1, n_comp)
    labels = list("xyzw"[:n_comp]) if n_comp <= 4 else [str(i) for i in range(n_comp)]
    lines = []
    for c, label in enumerate(labels):
        col = flat[:, c]
        lines.append(
            f"{prefix}  {label}: mean={col.mean().item():+.4g}  "
            f"std={col.std().item():.4g}  "
            f"min={col.min().item():+.4g}  "
            f"max={col.max().item():+.4g}"
        )
    return lines


def _print_data_summary(data, indent: int = 0, show_stats: bool = False) -> None:
    """Print tensor shapes (and optionally per-component stats) in a data object."""
    prefix = " " * indent
    if isinstance(data, TensorDict):
        for key in sorted(data.keys()):
            val = data[key]
            if isinstance(val, TensorDict):
                print(f"{prefix}{key}/")
                _print_data_summary(val, indent + 2, show_stats=show_stats)
            elif hasattr(val, "shape"):
                print(f"{prefix}{key}: {tuple(val.shape)} {val.dtype}")
                if show_stats:
                    for line in _component_stats_lines(val, prefix):
                        print(line)
    elif hasattr(data, "points"):
        print(
            f"{prefix}points={tuple(data.points.shape)}, "
            f"cells={tuple(data.cells.shape)}"
        )


def _report_times(times: list[float], label: str) -> None:
    """Print summary statistics for a list of timings."""
    if not times:
        print(f"  {label}: no samples")
        return

    total = sum(times)
    mean = statistics.mean(times)
    med = statistics.median(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    p95 = sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0]

    print(
        f"  {label} ({len(times)} samples): "
        f"total={total:.2f}s  mean={mean:.4f}s  median={med:.4f}s  "
        f"std={std:.4f}s  p95={p95:.4f}s  "
        f"min={min(times):.4f}s  max={max(times):.4f}s"
    )
    if total > 0:
        print(f"  throughput: {len(times) / total:.2f} samples/sec")


def benchmark_dataset(name: str, dataset, max_samples: int) -> None:
    """Benchmark a single dataset or MultiDataset."""
    n_total = len(dataset)
    n_bench = min(max_samples, n_total)

    print(f"\n{'=' * 60}")
    print(f"{name}  ({n_total} samples, benchmarking {n_bench})")
    print(f"{'=' * 60}")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=FunctionCollator(lambda samples: samples[0]),
        use_streams=False,
        prefetch_factor=0,
    )

    warmup_iter = iter(dataloader)
    for _ in range(min(2, n_bench)):
        try:
            _ = next(warmup_iter)
        except StopIteration:
            break
    del warmup_iter

    times = []
    for i, batch in enumerate(dataloader):
        t0 = time.perf_counter()
        data, metadata = batch
        dt = time.perf_counter() - t0
        times.append(dt)

        src = metadata.get("source_path", "unknown")
        ds_idx = metadata.get("dataset_index", "")
        header = f"  sample {i}"
        if ds_idx != "":
            header += f" (ds={ds_idx})"
        print(f"\n{header}  src={src}  dt={dt:.4f}s")
        _print_data_summary(data, indent=4, show_stats=True)

        if i + 1 >= max_samples:
            break

    print()
    _report_times(times, name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark surface mesh datapipes")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Max samples to benchmark per pipeline",
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

    valid_cfgs: list[tuple[Path, DictConfig]] = []
    for path in config_paths:
        if not path.exists():
            print(f"Config not found: {path}")
            continue
        cfg = load_dataset_config(path)
        datadir = OmegaConf.select(cfg, "train_datadir", default="")
        if not Path(datadir).exists():
            print(f"Skipping {path.stem}: data dir {datadir} not found")
            continue
        valid_cfgs.append((path, cfg))

    for path, cfg in valid_cfgs:
        ds = build_surface_dataset(cfg)
        benchmark_dataset(path.stem, ds, args.max_samples)

    if len(valid_cfgs) > 1:
        cfgs = [cfg for _, cfg in valid_cfgs]
        multi_ds = build_multi_surface_dataset(*cfgs)
        benchmark_dataset("combined", multi_ds, args.max_samples)


if __name__ == "__main__":
    with torch.no_grad():
        main()
