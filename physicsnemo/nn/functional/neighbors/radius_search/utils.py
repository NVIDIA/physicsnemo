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

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import torch


_CAPTURE_LOCK = threading.Lock()
_CAPTURE_COUNT = 0
_TIMING_LOCK = threading.Lock()
_TIMING_EVENTS: list[tuple[str, Any, Any]] = []


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def radius_search_timing_enabled() -> bool:
    return _env_flag("PHYSICSNEMO_RADIUS_SEARCH_TIMING")


class _RadiusSearchTimer:
    def __init__(self, implementation: str):
        self.implementation = implementation
        self.start_event: Any | None = None
        self.end_event: Any | None = None

    def __enter__(self):
        if not radius_search_timing_enabled():
            return self
        if not torch.cuda.is_available():
            return self
        if getattr(torch.compiler, "is_compiling", lambda: False)():
            return self

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.start_event is not None and self.end_event is not None:
            self.end_event.record()
            with _TIMING_LOCK:
                _TIMING_EVENTS.append(
                    (self.implementation, self.start_event, self.end_event)
                )
        return False


def time_radius_search(implementation: str) -> _RadiusSearchTimer:
    return _RadiusSearchTimer(implementation)


def reset_radius_search_timing() -> None:
    with _TIMING_LOCK:
        _TIMING_EVENTS.clear()


def summarize_radius_search_timing() -> dict[str, float | int]:
    with _TIMING_LOCK:
        events = list(_TIMING_EVENTS)
        _TIMING_EVENTS.clear()

    summary: dict[str, float | int] = {
        "neighbor_search_ms": 0.0,
        "neighbor_search_calls": 0,
    }
    if not events:
        return summary

    for _, _, end_event in events:
        end_event.synchronize()

    by_impl_ms: dict[str, float] = {}
    by_impl_calls: dict[str, int] = {}
    total_ms = 0.0
    for implementation, start_event, end_event in events:
        elapsed_ms = float(start_event.elapsed_time(end_event))
        total_ms += elapsed_ms
        by_impl_ms[implementation] = by_impl_ms.get(implementation, 0.0) + elapsed_ms
        by_impl_calls[implementation] = by_impl_calls.get(implementation, 0) + 1

    summary["neighbor_search_ms"] = total_ms
    summary["neighbor_search_calls"] = len(events)
    for implementation, elapsed_ms in sorted(by_impl_ms.items()):
        key = implementation.replace("-", "_")
        summary[f"neighbor_search_{key}_ms"] = elapsed_ms
        summary[f"neighbor_search_{key}_calls"] = by_impl_calls[implementation]
    return summary


def _capture_enabled_for(implementation: str) -> bool:
    capture_dir = os.getenv("PHYSICSNEMO_RADIUS_SEARCH_CAPTURE_DIR")
    if not capture_dir:
        return False

    allowed = os.getenv("PHYSICSNEMO_RADIUS_SEARCH_CAPTURE_IMPLS")
    if not allowed:
        return True
    return implementation in {item.strip() for item in allowed.split(",") if item.strip()}


def maybe_capture_radius_search(
    implementation: str,
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None,
    return_dists: bool,
    return_points: bool,
    *,
    num_neighbors: torch.Tensor | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Optionally save radius-search inputs for standalone replay/benchmarking.

    Capture is disabled unless ``PHYSICSNEMO_RADIUS_SEARCH_CAPTURE_DIR`` is set.
    ``PHYSICSNEMO_RADIUS_SEARCH_CAPTURE_LIMIT`` limits files per process; the
    default is 1 to avoid accidentally filling disks during long training runs.
    Set the limit to 0 for unlimited capture.
    """
    if not _capture_enabled_for(implementation):
        return
    if getattr(torch.compiler, "is_compiling", lambda: False)():
        return

    global _CAPTURE_COUNT
    limit = _env_int("PHYSICSNEMO_RADIUS_SEARCH_CAPTURE_LIMIT", 1)
    with _CAPTURE_LOCK:
        if limit > 0 and _CAPTURE_COUNT >= limit:
            return
        capture_id = _CAPTURE_COUNT
        _CAPTURE_COUNT += 1

    capture_dir = Path(os.environ["PHYSICSNEMO_RADIUS_SEARCH_CAPTURE_DIR"])
    capture_dir.mkdir(parents=True, exist_ok=True)
    rank = _env_int("RANK", 0)
    filename = f"rank{rank:03d}_{capture_id:06d}_{implementation}.pt"
    path = capture_dir / filename

    metadata: dict[str, Any] = {
        "implementation": implementation,
        "radius": float(radius),
        "max_points": max_points,
        "return_dists": bool(return_dists),
        "return_points": bool(return_points),
        "points_shape": tuple(points.shape),
        "queries_shape": tuple(queries.shape),
        "points_dtype": str(points.dtype),
        "queries_dtype": str(queries.dtype),
        "points_device": str(points.device),
        "queries_device": str(queries.device),
        "rank": rank,
        "capture_id": capture_id,
        "path": str(path),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    payload: dict[str, Any] = {
        "metadata": metadata,
        "points": points.detach().cpu(),
        "queries": queries.detach().cpu(),
        "radius": float(radius),
        "max_points": max_points,
        "return_dists": bool(return_dists),
        "return_points": bool(return_points),
    }
    if num_neighbors is not None and num_neighbors.numel() > 0:
        payload["num_neighbors"] = num_neighbors.detach().cpu()

    torch.save(payload, path)
    with (capture_dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(metadata, sort_keys=True) + "\n")


def validate_inputs(points: torch.Tensor, queries: torch.Tensor):
    """Validate and normalize inputs to (B, N, 3) shape.

    Returns ``(points, queries, was_unbatched)``.
    """
    if points.ndim == 2 and queries.ndim == 2:
        return points.unsqueeze(0), queries.unsqueeze(0), True
    elif points.ndim == 3 and queries.ndim == 3:
        if points.shape[0] != queries.shape[0]:
            raise ValueError(
                f"Batch dimensions must match: points has {points.shape[0]}, "
                f"queries has {queries.shape[0]}"
            )
        return points, queries, False
    else:
        raise ValueError(
            f"points and queries must be 2D (N, 3) or 3D (B, N, 3), "
            f"got {points.ndim}D and {queries.ndim}D"
        )


def format_returns(
    indices: torch.Tensor,
    points: torch.Tensor,
    distances: torch.Tensor,
    return_dists: bool,
    return_points: bool,
):
    """Select which tensors to include in the radius search return tuple.

    Always includes ``indices``. Appends ``points`` if ``return_points`` is True,
    and ``distances`` if ``return_dists`` is True.
    """
    if return_points:
        if return_dists:
            return indices, points, distances
        return indices, points

    if return_dists:
        return indices, distances

    return indices
