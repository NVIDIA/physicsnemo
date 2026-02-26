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
Preprocess and cache the AirFRANS dataset using the original dataloader.

Usage
-----
    # Full train split (default)
    AIRFRANS_DATA_DIR=/path/to/Dataset python preprocess_cache.py

    # Specific task/split
    AIRFRANS_DATA_DIR=/path/to/Dataset python preprocess_cache.py --task scarce --split train

    # Custom cache directory
    AIRFRANS_DATA_DIR=/path/to/Dataset python preprocess_cache.py --cache-dir /fast/ssd/cache
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from dataset import AirFRANSDataSet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess and cache AirFRANS samples"
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

    print(f"Task:      {args.task}")
    print(f"Split:     {args.split}")
    print(f"Samples:   {len(dataset)}")
    print(f"Data dir:  {data_dir}")
    print(f"Cache dir: {cache_dir}")
    print()

    t_total = time.perf_counter()
    for i in range(len(dataset)):
        t0 = time.perf_counter()
        _ = dataset[i]
        elapsed = time.perf_counter() - t0
        print(f"  [{i + 1:4d}/{len(dataset)}] {paths[i].name:40s} {elapsed:.2f}s")

    total = time.perf_counter() - t_total
    print()
    print(f"Done. {len(dataset)} samples in {total:.1f}s ({total / len(dataset):.2f}s/sample)")
    print(f"Cached to: {cache_dir.resolve()}")


if __name__ == "__main__":
    main()
