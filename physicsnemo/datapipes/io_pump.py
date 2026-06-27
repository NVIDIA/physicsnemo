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
IOPump - A self-driving I/O producer that keeps the IO pipeline primed.

The pump owns a dedicated dispatcher thread that lazily pulls work items from a
source iterator and submits each for background loading, keeping
a *bounded* number of samples in flight at all times. Pulling lazily means
the source may be arbitrarily large (or effectively unbounded): the
dispatcher only advances it when a backpressure slot is free, so memory
stays bounded regardless of source length.

Dispatch is decoupled from the consumer's cadence: the dispatcher keeps
topping the pipeline off while the main thread is busy launching kernels or
running the model, so a ready sample is (almost) always waiting when the
consumer asks for the next one.

The pump is agnostic to *how* a sample is produced and consumed: it drives
opaque work items through a user-provided ``dispatch_fn`` (which starts the
background load and returns a handle) and hands those handles back to the
consumer in the exact order they were pulled. Correlation is purely
positional (FIFO), so work items need not be hashable or unique.

A source may interleave :data:`BATCH_BOUNDARY` markers between work items;
the pump forwards them to the consumer in order without consuming a
backpressure slot, letting the consumer reassemble dynamically-sized
batches without knowing the batch layout up front.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Iterable, Iterator

# Public marker a source yields to delimit the end of one batch. A distinct
# sentinel object so it can never collide with a real work item.
BATCH_BOUNDARY = object()

# Internal marker the dispatcher pushes once the source is exhausted (or the
# pump is stopped), telling the consumer to finish iterating.
_DONE = object()


class _PumpError:
    """Wraps an exception raised on the dispatcher thread.

    Forwarded through the ready queue so the consumer re-raises it on the
    main thread instead of blocking forever waiting for items that will
    never arrive.
    """

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class IOPump:
    """Bounded, self-driving prefetch dispatcher.

    A dedicated dispatcher thread pulls work items from ``source``,
    acquires a backpressure slot, calls ``dispatch_fn(work_item)`` to start
    the background load, and makes the returned handle available to the
    consumer in FIFO order via iteration. Slots are released as the
    consumer advances, keeping at most ``depth`` samples in flight.

    Parameters
    ----------
    source : Iterable
        Work items to load, optionally interleaved with
        :data:`BATCH_BOUNDARY` markers. Consumed lazily, one item at a
        time, only as backpressure slots free up.
    dispatch_fn : Callable[[Any], Any]
        Called on the dispatcher thread to start loading a work item (for
        example ``dataset.submit(work_item, stream=...)``). It must be
        non-blocking and thread-safe and must not launch device kernels;
        it returns an opaque handle that the consumer later turns into a
        sample.
    depth : int
        Maximum number of samples dispatched but not yet consumed. Acts as
        both the backpressure valve and the jitter buffer that hides
        consumer stalls. Clamped to at least 1.

    Notes
    -----
    A pump instance is single-consumer. Iterate it with a single thread
    (the main/launcher thread). Call :meth:`stop` (or use it as a context
    manager) to tear down the dispatcher thread; already-submitted loads
    are left to complete and are reaped by the owning dataset.
    """

    def __init__(
        self,
        source: Iterable[Any],
        dispatch_fn: Callable[[Any], Any],
        depth: int,
    ) -> None:
        self._source = source
        self._dispatch_fn = dispatch_fn
        self._depth = max(1, int(depth))
        self._slots = threading.Semaphore(self._depth)
        self._ready_queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="datapipe_pump",
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Dispatcher thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Dispatcher loop: keep ``depth`` samples in flight, in order."""
        source = iter(self._source)
        while not self._stop.is_set():
            try:
                item = next(source)
            except StopIteration:
                break
            except BaseException as exc:  # noqa: BLE001
                # A failing source must surface to the consumer, not hang it.
                self._ready_queue.put(_PumpError(exc))
                return

            if item is BATCH_BOUNDARY:
                # Boundaries are bookkeeping, not work: forward without
                # consuming a slot.
                self._ready_queue.put(BATCH_BOUNDARY)
                continue

            # Backpressure: block until the consumer frees a slot. This is
            # also where lazy pulling is enforced -- the source is not
            # advanced again until there is room in flight.
            self._slots.acquire()
            if self._stop.is_set():
                break
            try:
                handle = self._dispatch_fn(item)
            except BaseException as exc:  # noqa: BLE001
                # A failing dispatch must surface to the consumer, not hang it.
                self._ready_queue.put(_PumpError(exc))
                return
            self._ready_queue.put(handle)

        self._ready_queue.put(_DONE)

    # ------------------------------------------------------------------
    # Consumer side (single consumer / the main thread)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Any]:
        """Yield ready handles (and batch boundaries) in FIFO order.

        Yields each loaded sample's handle in the order its work item was
        pulled, and forwards :data:`BATCH_BOUNDARY` markers in place.
        Releases a backpressure slot after each handle is consumed (i.e.
        on the next iteration), so the dispatcher can refill the pipeline
        as the consumer advances. Returns once the source is exhausted.

        Yields
        ------
        object
            Either a handle returned by ``dispatch_fn`` or
            :data:`BATCH_BOUNDARY`.
        """
        while True:
            item = self._ready_queue.get()
            if item is _DONE:
                return
            if isinstance(item, _PumpError):
                raise item.exc
            if item is BATCH_BOUNDARY:
                yield BATCH_BOUNDARY
                continue
            yield item
            # Consumer has finished with this sample; free a slot.
            self._slots.release()

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop the dispatcher thread and release its resources.

        Idempotent. Unblocks the dispatcher if it is waiting on a slot,
        then joins it briefly. In-flight background loads already submitted
        via ``dispatch_fn`` are not cancelled; the owning dataset reaps
        them.
        """
        if self._stop.is_set():
            return
        self._stop.set()
        # Unblock the dispatcher if it is parked acquiring a slot.
        self._slots.release()
        self._thread.join(timeout=5.0)

    def __enter__(self) -> "IOPump":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
