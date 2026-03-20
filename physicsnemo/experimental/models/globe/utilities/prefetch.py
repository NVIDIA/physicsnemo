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
from typing import TypeVar

_T = TypeVar("_T")
_U = TypeVar("_U")


def prefetch_map(iterable: Iterable[_T], fn: Callable[[_T], _U]) -> Iterator[_U]:
    """Apply *fn* to each element, overlapping ``fn(next)`` with consumption of current.

    Submits ``fn(element)`` to a single background thread one step ahead
    of the consumer.  Tensor operations release the GIL during C++/CUDA
    execution, so CPU-bound preparation of the next sample overlaps with
    GPU-bound processing of the current sample.

    Typical use: wrap a DataLoader to overlap CPU-bound sample preparation
    (subsampling, geometry precomputation, host-to-device transfer) with
    GPU-bound forward/backward processing of the previous sample.

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

    Notes
    -----
    If the iterator is not fully consumed (e.g., due to an early ``break``
    or an exception in the caller), the in-flight background task is not
    forcibly interrupted but the main thread will not block waiting for it.
    The background thread runs to completion on its own and is joined at
    interpreter shutdown.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        it = iter(iterable)
        try:
            future = pool.submit(fn, next(it))
        except StopIteration:
            return

        for item in it:
            next_future = pool.submit(fn, item)
            yield future.result()
            future = next_future

        yield future.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
