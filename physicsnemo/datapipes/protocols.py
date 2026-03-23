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
Base class for dataset components.

Provides :class:`DatasetBase`, an ABC that owns the thread-based prefetch
infrastructure shared by :class:`Dataset`, :class:`MeshDataset`, and any
future dataset implementations.  The user-facing extension points are
**Readers** and **Transforms**, not dataset subclasses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import torch


@dataclass
class _PrefetchResult:
    """Result of a stream-aware prefetch operation.

    Used by :class:`Dataset` and :class:`MeshDataset` to carry data,
    metadata, and an optional CUDA event through the prefetch pipeline.
    """

    index: int
    data: Any = None
    metadata: Optional[dict[str, Any]] = field(default=None)
    error: Optional[Exception] = field(default=None)
    event: Optional[torch.cuda.Event] = field(default=None)


class DatasetBase(ABC):
    """Abstract base for datasets compatible with :class:`DataLoader`.

    Subclasses implement :meth:`_load` (the actual data-loading pipeline)
    and :meth:`__len__`.  Everything else — ``__getitem__`` with prefetch
    cache lookup, thread-pool prefetching, cancellation, cleanup — is
    provided here.

    Both :class:`Dataset` and :class:`MeshDataset` override
    :meth:`prefetch` and :meth:`__getitem__` to add CUDA-stream
    support via :class:`_PrefetchResult`.
    """

    def __init__(self, *, num_workers: int = 2) -> None:
        self._prefetch_futures: dict[int, Future] = {}
        self._executor: Optional[ThreadPoolExecutor] = None
        self._num_workers = num_workers

    @abstractmethod
    def _load(self, index: int) -> tuple[Any, dict[str, Any]]:
        """Load and return a single sample ``(data, metadata)``.

        This is the hook that subclasses must implement.  It is called
        both synchronously (from ``__getitem__``) and asynchronously
        (from the prefetch thread pool).
        """
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    # ------------------------------------------------------------------
    # Concrete interface
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        """Return sample *index*, using a prefetched result when available."""
        future = self._prefetch_futures.pop(index, None)
        if future is not None:
            return future.result()  # re-raises on error
        return self._load(index)

    def prefetch(
        self,
        index: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Submit *index* for background loading in a worker thread.

        The ``stream`` parameter is accepted for interface compatibility
        but ignored by the default implementation.  :class:`Dataset`
        overrides this to run GPU transfers on the given stream.
        """
        if index in self._prefetch_futures:
            return
        executor = self._ensure_executor()
        self._prefetch_futures[index] = executor.submit(self._load, index)

    def cancel_prefetch(self, index: Optional[int] = None) -> None:
        """Discard prefetch results (already-running tasks still complete)."""
        if index is None:
            self._prefetch_futures.clear()
        else:
            self._prefetch_futures.pop(index, None)

    def close(self) -> None:
        """Drain in-flight prefetches and shut down the thread pool."""
        for future in self._prefetch_futures.values():
            try:
                future.result(timeout=30.0)
            except Exception:  # noqa: BLE001, S110
                pass
        self._prefetch_futures.clear()

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._num_workers,
                thread_name_prefix="datapipe_prefetch",
            )
        return self._executor

    def __iter__(self) -> Iterator[tuple[Any, dict[str, Any]]]:
        for i in range(len(self)):
            yield self[i]

    def __enter__(self) -> "DatasetBase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
