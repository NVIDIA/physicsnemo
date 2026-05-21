# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute per-channel mean/std from ARCO/ERA5 for FGN normalization.

Writes an ``.npz`` with arrays ``mean`` and ``std`` (each of shape
``(len(variables),)``) in the same variable order you pass in. The output
is consumed by `datasets.arco.ArcoFGNDataset` via the ``stats_path`` config
key.

Usage
-----
    python scripts/compute_arco_stats.py \\
        --variables u10m v10m t2m msl z500 q850 \\
        --samples 128 --stride 4 --output stats.npz

Defaults target the 83-channel Table A.1 schema; ``--samples`` draws random
timestamps uniformly from the training window. Uses a Welford-style online
accumulator to avoid materialising the full sample stack in memory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Paper Table A.1 defaults -- keep in sync with datasets.arco.DEFAULT_STATE.
DEFAULT_ATMOS_VARS = ("z", "q", "t", "u", "v", "w")
DEFAULT_LEVELS = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)
DEFAULT_SURFACE = ("t2m", "u10m", "v10m", "msl", "sst")
DEFAULT_STATE = tuple(
    [f"{v}{lvl}" for v in DEFAULT_ATMOS_VARS for lvl in DEFAULT_LEVELS]
    + list(DEFAULT_SURFACE)
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variables",
        nargs="+",
        default=list(DEFAULT_STATE),
        help="ARCO compact-name variables, in the order you want the stats "
        "arrays to be laid out. Default: paper Table A.1 (83 channels).",
    )
    p.add_argument(
        "--start",
        type=str,
        default="1979-01-01",
        help="Earliest sample time (ISO date).",
    )
    p.add_argument(
        "--end",
        type=str,
        default="2018-01-01",
        help="Latest sample time (ISO date, exclusive).",
    )
    p.add_argument(
        "--step-hours",
        type=int,
        default=6,
        help="Sampling cadence; restricts timestamps to a 6h grid by default.",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Number of random timestamps to average over.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Spatial stride applied to the 721x1440 grid to cut fetch cost.",
    )
    p.add_argument(
        "--tp-accumulation-hours",
        type=int,
        default=None,
        help="If set to N, any variable named tp{N:02d} (e.g. tp06 for N=6) "
        "in --variables is treated as a paper §3 N-hour accumulation of "
        "ARCO hourly ``tp``, not a native ARCOLexicon key. Matches "
        "ArcoFGNDataset.tp_accumulation_hours so stats and training see "
        "the same representation.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .npz path.",
    )
    return p.parse_args()


def iter_sample_times(
    start: datetime,
    end: datetime,
    step_hours: int,
    samples: int,
    rng: np.random.Generator,
):
    total_hours = int((end - start).total_seconds() // 3600)
    max_offset = total_hours // step_hours
    if max_offset <= 0:
        raise ValueError("start/end window is too narrow for the requested step_hours")
    offsets = rng.integers(0, max_offset, size=samples)
    return [start + timedelta(hours=int(o) * step_hours) for o in offsets]


def _accumulate_tp(arco, time: datetime, hours: int) -> np.ndarray:
    """Sum ``hours`` hourly ARCO ``tp`` values ending at ``time``.

    Mirrors ArcoFGNDataset._fetch_tp_accumulation's semantics (ARCO ``tp``
    is the hourly accumulation during ``[t-1h, t]``; an N-hour window
    ending at T sums values at T-N+1, ..., T).
    """
    window = [time - timedelta(hours=hours - 1 - j) for j in range(hours)]
    da = arco(time=window, variable=["tp"])
    hourly = np.asarray(da.values, dtype=np.float32)[:, 0]  # (N, 721, 1440)
    return hourly.sum(axis=0)


def main() -> None:
    args = parse_args()
    from earth2studio.data import ARCO

    rng = np.random.default_rng(args.seed)
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    times = iter_sample_times(start, end, args.step_hours, args.samples, rng)
    arco = ARCO(cache=True, verbose=True)

    # Figure out which slot (if any) is the tp{N} accumulation. Exclude it
    # from the bulk fetch just like ArcoFGNDataset does, and splice the
    # accumulation back into the channel tensor before computing stats.
    tp_name: str | None = None
    tp_idx: int | None = None
    arco_vars = list(args.variables)
    if args.tp_accumulation_hours is not None:
        tp_name = f"tp{args.tp_accumulation_hours:02d}"
        if tp_name in args.variables:
            tp_idx = args.variables.index(tp_name)
            arco_vars = [v for v in args.variables if v != tp_name]

    # Welford online accumulator per channel.
    n_channels = len(args.variables)
    count = np.int64(0)
    mean = np.zeros(n_channels, dtype=np.float64)
    m2 = np.zeros(n_channels, dtype=np.float64)

    for i, t in enumerate(times):
        da = arco(time=[t], variable=arco_vars)
        fetched = np.asarray(da.values, dtype=np.float32)[0]  # (V', 721, 1440)
        if args.stride > 1:
            fetched = fetched[:, :: args.stride, :: args.stride]

        # Embed fetched channels + (optional) tp accumulation into a single
        # tensor whose channel axis matches args.variables order.
        arr = np.zeros(
            (n_channels, fetched.shape[-2], fetched.shape[-1]), dtype=np.float32
        )
        if tp_idx is None:
            arr[:] = fetched
        else:
            arr[:tp_idx] = fetched[:tp_idx]
            arr[tp_idx + 1 :] = fetched[tp_idx:]
            tp_acc = _accumulate_tp(arco, t, args.tp_accumulation_hours)
            if args.stride > 1:
                tp_acc = tp_acc[:: args.stride, :: args.stride]
            arr[tp_idx] = tp_acc

        # SST NaN imputation with global min (paper A.1.1) so the stats are
        # computed on the same representation the training pipeline sees.
        if "sst" in args.variables:
            sst_idx = args.variables.index("sst")
            sst = arr[sst_idx]
            nan_mask = np.isnan(sst)
            if nan_mask.any():
                sst[nan_mask] = float(np.nanmin(sst))
                arr[sst_idx] = sst

        flat = arr.reshape(n_channels, -1).astype(np.float64)
        for v in range(n_channels):
            col = flat[v]
            n_col = col.size
            delta = col - mean[v]
            total = count + n_col
            mean[v] += delta.sum() / total
            m2[v] += (delta * (col - mean[v])).sum()
        count += flat.shape[1]

        if (i + 1) % 16 == 0 or i == len(times) - 1:
            print(f"[{i + 1}/{len(times)}] running mean[0]={mean[0]:.4g}")

    var = m2 / max(count - 1, 1)
    std = np.sqrt(var).astype(np.float32)
    mean = mean.astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        mean=mean,
        std=std,
        variables=np.array(list(args.variables), dtype=object),
    )
    print(f"wrote {args.output} (variables={n_channels}, samples={len(times)})")


if __name__ == "__main__":
    main()
