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

"""Opt-in, low-overhead per-stage wall-clock timing for the datapipe.

Enabled with ``PHYSICSNEMO_DATAPIPE_TIMING=1`` (off by default; when off,
:func:`record` and :func:`tick` are effectively no-ops). Wall-clock is the right
measure here: we care about *host-blocking* time on the producer (disk load,
``pin_memory``'s ``cudaHostAlloc``) and consumer (host-to-device launch, any
host syncs inside the BVH build / SDF) threads -- i.e. what actually starves the
pipeline -- not async GPU kernel time. No CUDA syncs are inserted, so the
measurement does not perturb stream overlap.

Stats accumulate across the producer (worker) threads and the consumer (main)
thread under a lock, and an aggregate (mean / p50 / p95 per stage) is logged
every ``PHYSICSNEMO_DATAPIPE_TIMING_EVERY`` consumed samples (default 100).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


def _truthy(val: str | None) -> bool:
    return val is not None and val.lower() not in ("", "0", "false", "no", "off")


_ENABLED = _truthy(os.environ.get("PHYSICSNEMO_DATAPIPE_TIMING"))
_LOG_EVERY = max(1, int(os.environ.get("PHYSICSNEMO_DATAPIPE_TIMING_EVERY", "100")))


def enabled() -> bool:
    """Return ``True`` when datapipe stage timing is enabled."""
    return _ENABLED


class _Stats:
    """Thread-safe accumulator of per-stage wall-clock samples."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._times: dict[str, list[float]] = defaultdict(list)
        self._samples = 0

    def add(self, stage: str, dt: float) -> None:
        with self._lock:
            self._times[stage].append(dt)

    def tick(self) -> None:
        with self._lock:
            self._samples += 1
            if self._samples % _LOG_EVERY != 0:
                return
            lines = []
            for stage in sorted(self._times):
                arr = sorted(self._times[stage])
                n = len(arr)
                if n == 0:
                    continue
                mean = sum(arr) / n
                p50 = arr[n // 2]
                p95 = arr[min(n - 1, int(0.95 * n))]
                total = sum(arr)
                lines.append(
                    f"  {stage:28s} n={n:5d} "
                    f"mean={mean * 1e3:8.2f}ms p50={p50 * 1e3:8.2f}ms "
                    f"p95={p95 * 1e3:8.2f}ms total={total * 1e3:9.1f}ms"
                )
            self._times.clear()
            logger.info(
                "datapipe stage timing over last %d samples:\n%s",
                _LOG_EVERY,
                "\n".join(lines),
            )


_STATS = _Stats()


@contextmanager
def record(stage: str) -> Iterator[None]:
    """Time the wrapped block under ``stage`` (no-op when timing is disabled)."""
    if not _ENABLED:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _STATS.add(stage, time.perf_counter() - t0)


def tick() -> None:
    """Count one consumed sample and log aggregate stats every N samples."""
    if _ENABLED:
        _STATS.tick()
