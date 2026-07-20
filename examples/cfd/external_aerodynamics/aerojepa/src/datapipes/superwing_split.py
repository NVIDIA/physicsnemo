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

r"""Train/val/test split-manifest builder for the SuperWing dataset.

Splits are made by **wing geometry** (column 0 of ``index.npy``) so that
all operating conditions for a held-out wing are unseen at evaluation
time. The output JSON contains both the geometry-level partition and the
sample-row indices into ``data.npy`` / ``index.npy`` that fall in each
split.

Run as a script:

.. code-block:: bash

    python -m src.datapipes.superwing_split \
        --root-dir /path/to/SuperWing_Dataset \
        --output /path/to/split_by_geometry.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


SUPERWING_INDEX_NUM_COLS: int = 12


def build_superwing_split_manifest(
    *,
    root_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> dict:
    r"""Build a geometry-level train/val/test split manifest.

    Parameters
    ----------
    root_dir : str
        Directory containing ``index.npy`` / ``geom0.npy`` / ``data.npy``.
    train_ratio : float, optional
        Fraction of wing geometries used for training. Default ``0.8``.
    val_ratio : float, optional
        Fraction of wing geometries used for validation. The remainder
        becomes the test split. Default ``0.1``.
    seed : int, optional
        Seed for the geometry permutation. Default ``0``.

    Returns
    -------
    dict
        Manifest dict with both geometry- and sample-level partitions
        (keys ``train_geom_idx``, ``val_geom_idx``, ``test_geom_idx``,
        ``train_sample_idx``, ``val_sample_idx``, ``test_sample_idx``).

    Raises
    ------
    FileNotFoundError
        If ``index.npy`` is missing from ``root_dir``.
    ValueError
        If ``index.npy`` does not have shape ``(N, 12)``, or if
        ``train_ratio + val_ratio > 1``.
    """
    index_path = os.path.join(root_dir, "index.npy")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.npy not found at {index_path}")
    index = np.load(index_path)
    if index.ndim != 2 or int(index.shape[1]) != SUPERWING_INDEX_NUM_COLS:
        raise ValueError(
            f"Unexpected index.npy shape {tuple(index.shape)}; "
            f"expected (N, {SUPERWING_INDEX_NUM_COLS})."
        )

    geom_idx_per_sample = index[:, 0].astype(np.int64)
    unique_geoms = np.unique(geom_idx_per_sample)
    n_geoms = int(unique_geoms.shape[0])

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_geoms)
    n_train = int(round(n_geoms * float(train_ratio)))
    n_val = int(round(n_geoms * float(val_ratio)))
    n_test = n_geoms - n_train - n_val
    if n_test < 0:
        raise ValueError("train_ratio + val_ratio must be <= 1")

    train_geoms = unique_geoms[perm[:n_train]].tolist()
    val_geoms = unique_geoms[perm[n_train : n_train + n_val]].tolist()
    test_geoms = unique_geoms[perm[n_train + n_val :]].tolist()

    train_set = set(train_geoms)
    val_set = set(val_geoms)
    test_set = set(test_geoms)

    train_samples: list[int] = []
    val_samples: list[int] = []
    test_samples: list[int] = []
    for sample_idx, gid in enumerate(geom_idx_per_sample.tolist()):
        if gid in train_set:
            train_samples.append(int(sample_idx))
        elif gid in val_set:
            val_samples.append(int(sample_idx))
        elif gid in test_set:
            test_samples.append(int(sample_idx))

    return {
        "version": 1,
        "root_dir": root_dir,
        "split_by": "geometry",
        "seed": int(seed),
        "n_total_geoms": n_geoms,
        "n_total_samples": int(geom_idx_per_sample.shape[0]),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(1.0 - train_ratio - val_ratio),
        "train_geom_idx": [int(v) for v in train_geoms],
        "val_geom_idx": [int(v) for v in val_geoms],
        "test_geom_idx": [int(v) for v in test_geoms],
        "train_sample_idx": train_samples,
        "val_sample_idx": val_samples,
        "test_sample_idx": test_samples,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root-dir",
        required=True,
        help="Directory containing index.npy / geom0.npy / data.npy.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output JSON path for the split manifest.",
    )
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    """Command-line entry point — see module docstring."""
    args = _parse_args()
    manifest = build_superwing_split_manifest(
        root_dir=args.root_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"Wrote split manifest to {out_path}: "
        f"geoms train/val/test = "
        f"{len(manifest['train_geom_idx'])}/"
        f"{len(manifest['val_geom_idx'])}/"
        f"{len(manifest['test_geom_idx'])}; "
        f"samples train/val/test = "
        f"{len(manifest['train_sample_idx'])}/"
        f"{len(manifest['val_sample_idx'])}/"
        f"{len(manifest['test_sample_idx'])}"
    )


if __name__ == "__main__":
    main()
