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

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, TypeVar

_T = TypeVar("_T")
_U = TypeVar("_U")


class _PrefetchMap(Generic[_U]):
    """Iterable that applies a function with one-step lookahead prefetching.

    Preserves ``__len__`` from the source iterable so that progress bars
    (tqdm) can display a total count.
    """

    def __init__(self, iterable: Iterable, fn: Callable[..., _U]) -> None:
        self._iterable = iterable
        self._fn = fn

    def __iter__(self) -> Iterator[_U]:
        with ThreadPoolExecutor(max_workers=1) as pool:
            it = iter(self._iterable)
            try:
                future = pool.submit(self._fn, next(it))
            except StopIteration:
                return

            for item in it:
                next_future = pool.submit(self._fn, item)
                yield future.result()
                future = next_future

            yield future.result()

    def __len__(self) -> int:
        return len(self._iterable)  # type: ignore[arg-type]


def prefetch_map(iterable: Iterable[_T], fn: Callable[[_T], _U]) -> _PrefetchMap[_U]:
    """Apply *fn* to each element, overlapping ``fn(next)`` with consumption of current.

    Submits ``fn(element)`` to a single background thread one step ahead
    of the consumer.  ``ThreadPoolExecutor`` does *not* sidestep the GIL;
    the overlap is real only when ``fn`` performs operations that release
    the GIL while running.  Operations that typically release the GIL and
    so benefit from prefetching:

    * CUDA kernel dispatch (asynchronous launches return immediately).
    * Host-to-device copies via ``tensor.to(device, non_blocking=True)``.
    * NumPy and PyTorch C++ kernels on tensors above the small-op threshold.
    * File and network I/O.

    Operations that hold the GIL or force synchronization, and so will
    *not* benefit (and may even serialize work):

    * Pure-Python computation.
    * ``tensor.item()`` and other host-side reads from device tensors.
    * Explicit syncs such as ``torch.cuda.synchronize`` or blocking
      ``tensor.cpu()`` calls.

    Typical good fit: wrap a DataLoader to overlap CPU-bound sample
    preparation (subsampling, geometry precomputation, host-to-device
    transfer) with GPU-bound forward/backward of the previous sample.
    For pure-Python heavy ``fn``, prefer a ``ProcessPoolExecutor``-based
    pattern instead.

    Parameters
    ----------
    iterable : Iterable[T]
        Source of raw items (e.g., a DataLoader).
    fn : Callable[[T], U]
        Preparation function applied to each item. Should be safe to call
        from a background thread (no shared mutable state with the main
        thread).

    Yields
    ------
    U
        Prepared items, one step behind the background thread.
    """
    return _PrefetchMap(iterable, fn)
