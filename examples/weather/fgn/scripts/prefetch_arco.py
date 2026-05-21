# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-warm the earth2studio ARCO local cache for an FGN training window.

The default FGN config streams ERA5 chunks from Google's public ARCO store
at training time. For small state-variable counts that's fine, but at the
paper's 83-channel schema (5 surface + 78 atmospheric) the first epoch is
dominated by GCS fetch latency rather than GPU compute.

This script iterates over the (time, variable) combinations that
``ArcoFGNDataset`` would need for a given window and calls the ARCO data
source so earth2studio populates its on-disk cache. Subsequent training runs
hitting the same cache location read locally and run at GPU speed.

Fetches are issued in monthly batches to stay within GCS per-request limits.
earth2studio's ARCO data source fires all (time, variable) pairs in the batch
concurrently via asyncio, so each batch saturates available network bandwidth.

Fetches:
  - State variables at ``step_hours`` cadence, covering
    ``[start - history_frames*step_hours, end]`` (so the earliest sample
    has all its prior frames cached). The derived ``tp{N}`` placeholder names
    are excluded — we handle total precipitation separately.
  - Hourly ``tp`` over ``[start - (history_frames*step_hours + N - 1), end]``
    so every sample's ``tp_accumulation_hours`` backward-window is cached.
  - Invariants ``z``, ``lsm`` at ``static_date`` (cheap, one-off fetch).

earth2studio ≥0.15.0a0 dynamically reads ``valid_time_stop`` from the ARCO
zarr metadata, extending coverage to 2025-12-31.

Runs fine on a CPU-only slurm node — no GPU needed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--start", default="2024-01-01", help="Window start (ISO, inclusive)."
    )
    p.add_argument("--end", default="2025-01-01", help="Window end (ISO, exclusive).")
    p.add_argument("--step-hours", type=int, default=6)
    p.add_argument("--history-frames", type=int, default=2)
    p.add_argument(
        "--tp-accumulation-hours",
        type=int,
        default=6,
        help="N for tp{N:02d} accumulation. 0 = skip tp fetch.",
    )
    p.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Override state variables. Default: ArcoFGNDataset.DEFAULT_STATE (82 ARCO vars).",
    )
    p.add_argument(
        "--static-date",
        default="2016-01-01",
        help="Date for one-off invariants fetch (z, lsm).",
    )
    p.add_argument(
        "--batch-days",
        type=int,
        default=31,
        help="Days of data per time-batch. Default: 31.",
    )
    p.add_argument(
        "--var-group-size",
        type=int,
        default=10,
        help="Variables per sub-batch. Limits concurrent GCS requests to "
        "batch_days_timestamps × var_group_size. Default: 10.",
    )
    p.add_argument(
        "--no-tp",
        dest="include_tp",
        action="store_false",
        help="Skip hourly tp prefetch.",
    )
    p.add_argument(
        "--no-invariants",
        dest="include_invariants",
        action="store_false",
        help="Skip invariants prefetch.",
    )
    p.set_defaults(include_tp=True, include_invariants=True)
    return p.parse_args()


def _window_times(start: datetime, end: datetime, step_hours: int) -> list[datetime]:
    out: list[datetime] = []
    t = start
    while t <= end:
        out.append(t)
        t += timedelta(hours=step_hours)
    return out


def _batch(
    times: list[datetime], batch_days: int, step_hours: int = 1
) -> list[list[datetime]]:
    """Split a time list into chunks covering at most batch_days of real time."""
    n = max(1, batch_days * 24 // step_hours)
    return [times[i : i + n] for i in range(0, len(times), n)]


def main() -> int:
    args = parse_args()
    # Silence earth2studio's per-fetch DEBUG lines — they flood log files at scale.
    import sys

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    from datasets.arco import DEFAULT_STATE  # noqa: E402 — after sys.path patch
    from earth2studio.data import ARCO  # noqa: E402

    state_vars = list(args.variables) if args.variables else list(DEFAULT_STATE)
    # Filter the derived tp{N} placeholder — not a valid ARCOLexicon key.
    fetch_vars = [v for v in state_vars if not v.startswith("tp")]

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    static_date = datetime.fromisoformat(args.static_date)

    # Warm window: shift start back so the first sample's history frames are cached.
    warm_start_state = start - timedelta(hours=args.history_frames * args.step_hours)
    warm_end_state = end
    times_state = _window_times(warm_start_state, warm_end_state, args.step_hours)

    tp_acc = args.tp_accumulation_hours
    fetch_tp = args.include_tp and tp_acc > 0
    if fetch_tp:
        warm_start_tp = warm_start_state - timedelta(hours=tp_acc - 1)
        times_tp = _window_times(warm_start_tp, warm_end_state, step_hours=1)

    n_state_req = len(times_state) * len(fetch_vars)
    n_tp_req = len(times_tp) * 1 if fetch_tp else 0
    print(
        f"State : {len(fetch_vars)} vars × {len(times_state)} timestamps"
        f" = {n_state_req:,} requests  ({warm_start_state} → {warm_end_state})"
    )
    if fetch_tp:
        print(
            f"TP    : 1 var  × {len(times_tp)} hourly timestamps"
            f" = {n_tp_req:,} requests"
        )
    # --var-group-size: number of variables per sub-batch. Keeps concurrent
    # GCS requests to batch_days_timestamps × var_group_size to avoid GCS
    # rate-limit timeouts at large scale.
    var_group_size = args.var_group_size
    var_groups = [
        fetch_vars[i : i + var_group_size]
        for i in range(0, len(fetch_vars), var_group_size)
    ]
    total_state_batches = len(_batch(times_state, args.batch_days, args.step_hours))
    print(
        f"Batch : {args.batch_days} days × {var_group_size} vars/group"
        f" → {total_state_batches} time-batches × {len(var_groups)} var-groups"
        f" = {total_state_batches * len(var_groups)} calls\n"
    )

    # async_timeout per call (one time-batch × one var-group).
    arco = ARCO(cache=True, async_timeout=3600)

    # --- state variables: time-batch outer loop, var-group inner loop ---
    state_batches = _batch(times_state, args.batch_days, step_hours=args.step_hours)
    n_tb = len(state_batches)
    n_vg = len(var_groups)
    for ti, batch_times in enumerate(state_batches, 1):
        for vi, vgroup in enumerate(var_groups, 1):
            print(
                f"[state {ti}/{n_tb} vars {vi}/{n_vg}]  "
                f"{batch_times[0].date()} → {batch_times[-1].date()}"
                f"  ({len(batch_times)} steps × {len(vgroup)} vars"
                f" = {len(batch_times) * len(vgroup)} requests)"
            )
            arco(time=batch_times, variable=vgroup)

    # --- hourly tp, batched by time only (1 var, small requests) ---
    if fetch_tp:
        tp_batches = _batch(times_tp, args.batch_days, step_hours=1)
        for i, batch_times in enumerate(tp_batches, 1):
            print(
                f"[tp    {i}/{len(tp_batches)}]  "
                f"{batch_times[0].date()} → {batch_times[-1].date()}"
                f"  ({len(batch_times)} hourly steps)"
            )
            arco(time=batch_times, variable=["tp"])

    # --- invariants (one-off) ---
    if args.include_invariants:
        print(f"\n[invariants] z, lsm at {static_date.date()}")
        arco(time=[static_date], variable=["z", "lsm"])

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
