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
Benchmark loading AirFRANS samples from the .pt cache.

Measures per-sample and aggregate wall-clock time for torch.load from the
cached .pt files produced by preprocess_cache.py.

Usage
-----
    AIRFRANS_DATA_DIR=/path/to/Dataset python benchmark_cache.py

    # Custom cache dir / fewer samples
    AIRFRANS_DATA_DIR=/path/to/Dataset python benchmark_cache.py \\
        --cache-dir ./cache/full_train --n-samples 20
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

from dataset import AirFRANSDataSet


def time_samples(dataset: AirFRANSDataSet, n_samples: int) -> list[float]:
    actual_n = min(n_samples, len(dataset))
    if actual_n == 0:
        print("  No samples available.")
        return []

    print(f"  Warming up (1 sample)...")
    _ = dataset[0]

    print(f"  Timing {actual_n} samples...")
    times: list[float] = []
    for i in range(actual_n):
        t0 = time.perf_counter()
        _ = dataset[i]
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times


def print_table(label: str, times: list[float]) -> None:
    n = len(times)
    if n == 0:
        return
    total = sum(times)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0.0
    median = statistics.median(times)
    throughput = n / total if total > 0 else 0.0

    header = f"{'':30s} {'Samples':>7s} {'Total':>10s} {'Mean':>10s} {'Median':>10s} {'Std':>10s} {'Throughput':>12s}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    print(
        f"{label:<30s} {n:>7d} {total:>10.3f}s {mean:>9.3f}s "
        f"{median:>9.3f}s {std:>9.3f}s {throughput:>10.2f}/s"
    )
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark AirFRANS .pt cache loading"
    )
    parser.add_argument(
        "--task",
        choices=["full", "scarce", "reynolds", "aoa"],
        default="full",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory (default: ./cache/{task}_{split})",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples to time (clamped to dataset size)",
    )
    args = parser.parse_args()

    data_dir_env = os.environ.get("AIRFRANS_DATA_DIR")
    if not data_dir_env:
        raise ValueError("Set the AIRFRANS_DATA_DIR environment variable.")
    data_dir = Path(data_dir_env)

    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else Path("cache") / f"{args.task}_{args.split}"
    )

    paths = AirFRANSDataSet.get_split_paths(data_dir, task=args.task, split=args.split)
    dataset = AirFRANSDataSet(sample_paths=paths, cache_dir=cache_dir)

    cached_count = sum(
        1 for p in paths
        if (cache_dir / p.name).with_suffix(".pt").exists()
    )

    print(f"=== AirFRANS Cache Benchmark ===")
    print(f"  Task:      {args.task}")
    print(f"  Split:     {args.split}")
    print(f"  Samples:   {len(dataset)}")
    print(f"  Cached:    {cached_count}/{len(dataset)}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  N samples: {min(args.n_samples, len(dataset))}")
    print()

    if cached_count < min(args.n_samples, len(dataset)):
        print(
            "  WARNING: Not all samples are cached. Uncached samples will be "
            "preprocessed on the fly, skewing timing results. Run "
            "preprocess_cache.py first."
        )
        print()

    times = time_samples(dataset, args.n_samples)
    print_table("Cache load (torch.load .pt)", times)


if __name__ == "__main__":
    main()
