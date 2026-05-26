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

r"""Normalization-stat builder for the SuperWing dataset.

Computes:

* Per-channel mean / std for the three surface target fields
  (``Cp``, ``Cf_tau``, ``Cf_z``) over the training subset.
* Mean / std for the operating-condition columns kept as model input
  (default: ``aoa`` at column 2 and ``mach`` at column 3 of ``index.npy``).
* Global xyz min/max across training geometries' cell-centric meshes
  (``geom0``), used to map coordinates into ``[-1, 1]``.

The output JSON is consumed by :class:`SuperWingDataset` at
``__getitem__`` time.

Run as a script:

.. code-block:: bash

    python -m src.datapipes.superwing_normalization \
        --root-dir /path/to/SuperWing_Dataset \
        --split-manifest /path/to/split_by_geometry.json \
        --output /path/to/normalization_stats_train.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np


SUPERWING_TARGET_CHANNELS: list[str] = ["Cp", "Cf_tau", "Cf_z"]
SUPERWING_DEFAULT_GEN_PARAM_COLUMNS: list[int] = [2, 3]  # aoa, mach
SUPERWING_DEFAULT_GEN_PARAM_NAMES: list[str] = ["aoa", "mach"]


def _load_split(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_superwing_normalization_stats(
    *,
    root_dir: str,
    split_manifest_path: str,
    gen_param_columns: Sequence[int] = SUPERWING_DEFAULT_GEN_PARAM_COLUMNS,
    gen_param_names: Sequence[str] = SUPERWING_DEFAULT_GEN_PARAM_NAMES,
    max_target_samples: int = 2048,
    seed: int = 0,
    save_path: str | None = None,
) -> dict:
    r"""Compute and (optionally) persist normalization statistics.

    Parameters
    ----------
    root_dir : str
        Directory containing ``index.npy`` / ``geom0.npy`` / ``data.npy``.
    split_manifest_path : str
        JSON produced by :func:`build_superwing_split_manifest` — its
        train partition drives both the geometry and field statistics.
    gen_param_columns : Sequence[int], optional
        Indices into ``index.npy`` columns to keep as gen-params. Default
        ``[2, 3]`` (``aoa``, ``mach``).
    gen_param_names : Sequence[str], optional
        Human-readable names matching ``gen_param_columns``. Must have
        equal length. Default ``["aoa", "mach"]``.
    max_target_samples : int, optional
        Cap on the number of training samples streamed for target stats
        (a random subset is drawn). ``0`` means use all train samples.
        Default ``2048``.
    seed : int, optional
        Seed for the target-sample subsampling. Default ``0``.
    save_path : str or None, optional
        If set, write the stats to this path as JSON.

    Returns
    -------
    dict
        Stats dict ready to be consumed by :class:`SuperWingDataset`.

    Raises
    ------
    ValueError
        If ``gen_param_columns`` and ``gen_param_names`` have different
        lengths, or the train partition is empty.
    FileNotFoundError
        If any required file under ``root_dir`` is missing.
    """
    if len(gen_param_columns) != len(gen_param_names):
        raise ValueError(
            "gen_param_columns and gen_param_names must have equal length."
        )

    manifest = _load_split(split_manifest_path)
    train_geom_idx = sorted(int(v) for v in manifest["train_geom_idx"])
    train_sample_idx = sorted(int(v) for v in manifest["train_sample_idx"])
    if not train_geom_idx or not train_sample_idx:
        raise ValueError("Split manifest has empty train partition.")

    index_path = os.path.join(root_dir, "index.npy")
    geom_path = os.path.join(root_dir, "geom0.npy")
    data_path = os.path.join(root_dir, "data.npy")
    for p in (index_path, geom_path, data_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file missing: {p}")

    index = np.load(index_path)
    geom = np.load(geom_path, mmap_mode="r")
    data = np.load(data_path, mmap_mode="r")

    # Gen-params mean/std over train rows.
    gp_cols = np.asarray(list(gen_param_columns), dtype=np.int64)
    gp_train = index[np.asarray(train_sample_idx, dtype=np.int64)][:, gp_cols].astype(
        np.float64
    )
    gp_mean = gp_train.mean(axis=0)
    gp_std = gp_train.std(axis=0)
    gp_std = np.where(gp_std < 1e-12, 1.0, gp_std)

    # xyz min/max over training geometries' full (3, 128, 256) cell grids.
    xyz_min = np.full((3,), np.inf, dtype=np.float64)
    xyz_max = np.full((3,), -np.inf, dtype=np.float64)
    for gid in train_geom_idx:
        g = np.asarray(geom[int(gid)], dtype=np.float64)
        flat = g.reshape(3, -1)
        xyz_min = np.minimum(xyz_min, flat.min(axis=1))
        xyz_max = np.maximum(xyz_max, flat.max(axis=1))

    # Per-channel mean/std for the surface field, streamed in two passes.
    rng = np.random.default_rng(int(seed))
    train_sample_arr = np.asarray(train_sample_idx, dtype=np.int64)
    if max_target_samples > 0 and train_sample_arr.size > max_target_samples:
        sub_idx = rng.choice(
            train_sample_arr, size=int(max_target_samples), replace=False
        )
        sub_idx = np.sort(sub_idx)
    else:
        sub_idx = train_sample_arr

    n = 0
    s1 = np.zeros((3,), dtype=np.float64)
    s2 = np.zeros((3,), dtype=np.float64)
    t_min = np.full((3,), np.inf, dtype=np.float64)
    t_max = np.full((3,), -np.inf, dtype=np.float64)
    for sid in sub_idx.tolist():
        d = np.asarray(data[int(sid)], dtype=np.float64)
        flat = d.reshape(3, -1)
        s1 += flat.sum(axis=1)
        s2 += np.square(flat).sum(axis=1)
        t_min = np.minimum(t_min, flat.min(axis=1))
        t_max = np.maximum(t_max, flat.max(axis=1))
        n += int(flat.shape[1])
    if n == 0:
        raise RuntimeError("No target samples accumulated; cannot compute stats.")
    t_mean = s1 / float(n)
    t_var = np.maximum(s2 / float(n) - np.square(t_mean), 0.0)
    t_std = np.sqrt(t_var)
    t_std = np.where(t_std < 1e-12, 1.0, t_std)

    stats: dict = {
        "version": 1,
        "schema": "superwing",
        "n_train_geoms": len(train_geom_idx),
        "n_train_samples_used_for_targets": int(sub_idx.shape[0]),
        "target_channels": list(SUPERWING_TARGET_CHANNELS),
        "target_mean": t_mean.astype(np.float32).tolist(),
        "target_std": t_std.astype(np.float32).tolist(),
        "target_min": t_min.astype(np.float32).tolist(),
        "target_max": t_max.astype(np.float32).tolist(),
        "gen_params_columns": [int(v) for v in gen_param_columns],
        "gen_params_names": list(gen_param_names),
        "gen_params_mean": gp_mean.astype(np.float32).tolist(),
        "gen_params_std": gp_std.astype(np.float32).tolist(),
        "xyz_min": xyz_min.astype(np.float32).tolist(),
        "xyz_max": xyz_max.astype(np.float32).tolist(),
    }

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-dir", required=True)
    p.add_argument(
        "--split-manifest",
        required=True,
        help="JSON manifest produced by build_superwing_split_manifest.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output JSON path for normalization stats.",
    )
    p.add_argument(
        "--gen-param-columns",
        type=int,
        nargs="+",
        default=SUPERWING_DEFAULT_GEN_PARAM_COLUMNS,
        help="Indices into index.npy columns to use as gen_params.",
    )
    p.add_argument(
        "--gen-param-names",
        type=str,
        nargs="+",
        default=SUPERWING_DEFAULT_GEN_PARAM_NAMES,
        help="Names corresponding to --gen-param-columns.",
    )
    p.add_argument(
        "--max-target-samples",
        type=int,
        default=2048,
        help=(
            "Cap on number of training samples streamed for target stats. "
            "0 = all train samples."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    """Command-line entry point — see module docstring."""
    args = _parse_args()
    stats = compute_superwing_normalization_stats(
        root_dir=args.root_dir,
        split_manifest_path=args.split_manifest,
        gen_param_columns=args.gen_param_columns,
        gen_param_names=args.gen_param_names,
        max_target_samples=args.max_target_samples,
        seed=args.seed,
        save_path=args.output,
    )
    print(f"Wrote SuperWing normalization stats to {args.output}")
    print(f"  target_mean = {stats['target_mean']}")
    print(f"  target_std  = {stats['target_std']}")
    print(f"  gen_params_mean = {stats['gen_params_mean']}")
    print(f"  gen_params_std  = {stats['gen_params_std']}")
    print(f"  xyz_min = {stats['xyz_min']}")
    print(f"  xyz_max = {stats['xyz_max']}")


if __name__ == "__main__":
    main()
