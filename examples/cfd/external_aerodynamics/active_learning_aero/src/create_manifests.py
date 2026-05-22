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

"""Create fixed test/pool split manifests for active learning.

Run once to produce JSON manifests that define exactly which samples
are in the test set (N per class) and pool (remaining). All subsequent
AL experiments read these manifests instead of re-splitting.

Class labels are supplied as ``LABEL=PATH`` pairs to ``--zarr-paths``,
so the same script works for any dataset (e.g. ShiftSUV's SE/SF or
DrivAerStar's F/N/E).

Usage::

    python create_manifests.py \\
        --zarr-paths \\
            SE=/path/to/shift_suv_estateback_zarr/val \\
            SF=/path/to/shift_suv_fastback_zarr/val \\
        --test-samples-per-class 100 \\
        --seed 42 \\
        --output-dir src/manifests
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


def _parse_label_path(s: str) -> tuple[str, str]:
    """Parse a ``LABEL=PATH`` argument into a ``(label, path)`` tuple."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"Expected LABEL=PATH, got {s!r} (missing '=')."
        )
    label, _, path = s.partition("=")
    if not label or not path:
        raise argparse.ArgumentTypeError(
            f"Expected LABEL=PATH with both fields non-empty, got {s!r}."
        )
    return label, path


def main():
    """CLI entry point: build per-class test/pool split manifests for AL."""
    parser = argparse.ArgumentParser(description="Create AL split manifests")
    parser.add_argument(
        "--zarr-paths",
        type=_parse_label_path,
        nargs="+",
        required=True,
        metavar="LABEL=PATH",
        help="One or more class label / zarr-val-dir pairs, e.g. SE=/data/.../val",
    )
    parser.add_argument("--test-samples-per-class", type=int, default=100)
    parser.add_argument(
        "--pool-per-class",
        type=int,
        default=500,
        help="Max samples per class in the AL pool (rest discarded)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="manifests/")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    classes = dict(args.zarr_paths)
    if len(classes) != len(args.zarr_paths):
        raise SystemExit("Duplicate class label in --zarr-paths.")

    summary = {}
    for cls_label, zarr_path in classes.items():
        samples = list_sample_names(zarr_path)
        n = len(samples)
        perm = rng.permutation(n)

        test_idx = sorted(perm[: args.test_samples_per_class].tolist())
        remaining = perm[args.test_samples_per_class :]
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
            "test_per_class": args.test_samples_per_class,
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
