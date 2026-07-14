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

r"""Merge sharded ``predictions_shard*.npz`` files into one ``predictions.npz``.

`inference.py` run with ``num_shards>1`` writes one file per shard. This
concatenates the per-case arrays back into a single dump identical in schema
to a single-GPU run, so the downstream `superwing_metrics` / `superwing_forces`
scripts run on it unchanged.

Per-case arrays (fields, coefficients, ids, ...) are concatenated along axis 0.
The two dataset-level arrays ``target_mean`` / ``target_std`` are identical
across shards and are copied from the first file. Cases are de-duplicated by
``case_ids`` (a no-op for strided sharding, a safety net otherwise), which also
yields a deterministic case-id-sorted order.

Usage::

    # Shell expands the glob to the shard files:
    python -m src.postprocessing.merge_shards \
        --shards outputs/infer/shard*/inference/predictions_shard*.npz \
        --output outputs/infer/predictions.npz

    # Or point at a directory to search recursively:
    python -m src.postprocessing.merge_shards \
        --shards outputs/infer --output outputs/infer/predictions.npz
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

# Arrays that are dataset-level (not per-case) and so are copied, not stacked.
STATIC_KEYS: tuple[str, ...] = ("target_mean", "target_std")


def _resolve_shard_files(inputs: list[str]) -> list[Path]:
    """Expand CLI inputs (files, globs, or directories) to a shard-file list."""
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.rglob("predictions_shard*.npz")))
            continue
        matched = sorted(glob.glob(item))  # handle an unexpanded glob string
        files.extend(Path(m) for m in matched) if matched else files.append(p)
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def merge_shards(*, shard_files: list[Path], output: Path) -> int:
    """Concatenate shard npz files into ``output``; return the case count."""
    if not shard_files:
        raise FileNotFoundError("No shard files given / found to merge.")
    for f in shard_files:
        if not f.exists():
            raise FileNotFoundError(f"Shard file not found: {f}")

    loaded = [np.load(f, allow_pickle=False) for f in shard_files]
    keys = list(loaded[0].keys())

    merged: dict[str, np.ndarray] = {}
    for key in keys:
        if key in STATIC_KEYS:
            merged[key] = loaded[0][key]
        else:
            merged[key] = np.concatenate([d[key] for d in loaded], axis=0)

    if "case_ids" in merged:
        # First occurrence of each case id, in sorted (deterministic) order.
        _, first_idx = np.unique(merged["case_ids"], return_index=True)
        keep = np.sort(first_idx)
        for key in keys:
            if key not in STATIC_KEYS:
                merged[key] = merged[key][keep]

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    return int(merged["case_ids"].shape[0]) if "case_ids" in merged else -1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shards",
        required=True,
        nargs="+",
        help="Shard npz files, a glob, or a directory to search recursively.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path for the merged predictions.npz.",
    )
    return p.parse_args()


def main() -> None:
    """Command-line entry point -- see module docstring."""
    args = _parse_args()
    shard_files = _resolve_shard_files(list(args.shards))
    n_cases = merge_shards(shard_files=shard_files, output=Path(args.output))
    print(f"Merged {len(shard_files)} shard(s) -> {args.output} ({n_cases} cases)")


if __name__ == "__main__":
    main()
