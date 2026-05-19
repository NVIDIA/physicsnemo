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

"""Create fixed test/pool split manifests for DrivAerStar active learning.

Run once to produce JSON manifests that define exactly which samples
are in the test set (100 per class) and pool (remaining).  All
subsequent AL experiments read these manifests instead of re-splitting.

Usage::

    python create_manifests.py \\
        --class_F /data/datasets/drivaerstar/surface_files_zarr/class_F/val \\
        --class_N /data/datasets/drivaerstar/surface_files_zarr/class_N/val \\
        --class_E /data/datasets/drivaerstar/surface_files_zarr/class_E/val \\
        --test_per_class 100 \\
        --seed 42 \\
        --out_dir manifests/
"""

import argparse
import json
from pathlib import Path

import numpy as np


def list_sample_names(zarr_dir: str) -> list[str]:
    """List sample directory names within a zarr val directory."""
    p = Path(zarr_dir)
    samples = sorted([d.name for d in p.iterdir() if d.is_dir()])
    return samples


def main():
    """CLI entry point: build per-class test/pool split manifests for AL."""
    parser = argparse.ArgumentParser(description="Create AL split manifests")
    parser.add_argument("--class_F", required=True, help="Path to class_F/val zarr dir")
    parser.add_argument("--class_N", required=True, help="Path to class_N/val zarr dir")
    parser.add_argument("--class_E", required=True, help="Path to class_E/val zarr dir")
    parser.add_argument("--test_per_class", type=int, default=100)
    parser.add_argument(
        "--pool_per_class",
        type=int,
        default=500,
        help="Max samples per class in the AL pool (rest discarded)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="manifests/")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    classes = {
        "F": args.class_F,
        "N": args.class_N,
        "E": args.class_E,
    }

    summary = {}
    for cls_label, zarr_path in classes.items():
        samples = list_sample_names(zarr_path)
        n = len(samples)
        perm = rng.permutation(n)

        test_idx = sorted(perm[: args.test_per_class].tolist())
        remaining = perm[args.test_per_class :]
        if args.pool_per_class is not None and len(remaining) > args.pool_per_class:
            remaining = remaining[: args.pool_per_class]
        pool_idx = sorted(remaining.tolist())

        test_names = [samples[i] for i in test_idx]
        pool_names = [samples[i] for i in pool_idx]

        manifest = {
            "class": cls_label,
            "zarr_path": zarr_path,
            "total_samples": n,
            "seed": args.seed,
            "test_per_class": args.test_per_class,
            "test_indices": test_idx,
            "test_names": test_names,
            "pool_indices": pool_idx,
            "pool_names": pool_names,
        }

        fname = f"manifest_class_{cls_label}.json"
        with open(out_dir / fname, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote {out_dir / fname}: {len(test_idx)} test, {len(pool_idx)} pool")

        summary[cls_label] = {
            "total": n,
            "test": len(test_idx),
            "pool": len(pool_idx),
        }

    print(f"\nSummary:")
    total_test = 0
    total_pool = 0
    for cls_label, counts in summary.items():
        print(
            f"  {cls_label}: {counts['total']} total -> {counts['test']} test + {counts['pool']} pool"
        )
        total_test += counts["test"]
        total_pool += counts["pool"]
    print(f"  Total: {total_test} test + {total_pool} pool = {total_test + total_pool}")


if __name__ == "__main__":
    main()
