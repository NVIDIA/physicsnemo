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
Base classes for dataset components.

Provides two abstractions consumed by :class:`~physicsnemo.datapipes.DataLoader`:

- :class:`DatasetBase` -- map-style datasets. Owns the thread-based
  prefetch infrastructure (a producer/consumer split plus a FIFO
  ``submit``/``consume`` primitive) shared by :class:`Dataset`,
  :class:`MeshDataset`, and any future implementation.
- :class:`IterableDatasetBase` -- generator-style datasets that produce
  data directly on the main thread (online simulation and other
  stream-sensitive workloads). No prefetch, no length, no indexing.

The user-facing extension points are **Readers** and **Transforms**, not
dataset subclasses.
"""

from __future__ import annotations

import contextlib
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import torch
from tensordict import is_tensor_collection


@contextlib.contextmanager
def preprocessing_stream(stream: Optional["torch.cuda.Stream"]):
    """Bind torch to *stream* for the host-to-device copy + transforms.

    Within the block torch's current stream is set to *stream*, so the
    host-to-device copy and any device kernels launched by transforms run
    on the same stream the data was copied on. A ``None`` stream is a
    no-op (run on the current stream).

    Parameters
    ----------
    stream : torch.cuda.Stream, optional
        Stream to bind, or ``None`` to run on the current stream.
    """
    if stream is None:
        yield
        return
    with torch.cuda.stream(stream):
        yield


def record_consumer_stream(data: Any, stream: torch.cuda.Stream) -> None:
    """Mark every CUDA tensor in *data* as used by *stream*.

    Tensors allocated on a preprocessing stream but consumed on the compute
    stream must be recorded against the consumer
    (:meth:`torch.Tensor.record_stream`), so the caching allocator does not
    recycle their blocks for new prep-stream allocations while consumer
    kernels are still reading them.  The event-based gating between the two
    streams only orders *kernels* (compute after preprocessing); block reuse
    after a host-side free is an allocator decision this call defers.

    Walks tensor collections (``TensorDict`` and tensorclasses such as
    ``Mesh``/``DomainMesh``), mappings, lists, and tuples; non-tensor leaves
    and CPU tensors are ignored.

    Parameters
    ----------
    data : Any
        Sample data whose CUDA tensors will be consumed on *stream*.
    stream : torch.cuda.Stream
        The consumer (compute) stream.
    """
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            data.record_stream(stream)
    elif is_tensor_collection(data):
        for leaf in data.values(include_nested=True, leaves_only=True):
            if isinstance(leaf, torch.Tensor) and leaf.is_cuda:
                leaf.record_stream(stream)
    elif isinstance(data, Mapping):
        for value in data.values():
            record_consumer_stream(value, stream)
    elif isinstance(data, (list, tuple)):
        for value in data:
            record_consumer_stream(value, stream)


@dataclass
class HostPayload:
    """A sample produced by the (thread-safe) I/O stage, staged on the host.

    A ``HostPayload`` is the boundary object between the I/O producer and
    the main-thread consumer. It carries a CPU ``TensorDict`` (ideally
    pinned, so the subsequent host-to-device copy can be asynchronous)
    plus metadata. It is produced by a worker thread, which must not
    launch device kernels.

    Parameters
    ----------
    work_item : Any
        The work item this payload was produced from -- an ``int`` index
        for map-style datasets, or any opaque descriptor for
        descriptor-driven sources.
    data : Any, optional
        Host ``TensorDict`` (or mesh) payload. ``None`` on error.
    metadata : dict, optional
        Per-sample metadata produced by the reader.
    error : Exception, optional
        Exception captured during production, re-raised on consumption.
    """

    work_item: Any
    data: Any = None
    metadata: Optional[dict[str, Any]] = field(default=None)
    error: Optional[Exception] = field(default=None)


@dataclass
class PrefetchHandle:
    """Handle to one in-flight prefetch returned by :meth:`DatasetBase.submit`.

    Bundles the producer ``Future`` with the CUDA stream the consumer
    should use for the host-to-device copy and transforms. The handle is
    correlated to its sample purely by identity/order, so opaque work
    items need not be hashable or unique.

    Parameters
    ----------
    future : concurrent.futures.Future
        Future resolving to the producer's :class:`HostPayload`.
    stream : torch.cuda.Stream, optional
        Stream assigned for the consume step, or ``None`` for the current
        stream.
    """

    future: Future
    stream: Optional[torch.cuda.Stream] = None


class DatasetBase(ABC):
    """Abstract base for map-style datasets compatible with :class:`DataLoader`.

    Subclasses implement :meth:`_load` (the synchronous data-loading
    pipeline) and :meth:`__len__`. Everything else -- indexing, the
    ``submit``/``consume`` prefetch primitive, the index-keyed
    ``prefetch``/``__getitem__`` convenience API, cancellation, and
    cleanup -- is provided here.

    Producer / consumer split
    --------------------------
    Prefetching is split into two stages so that **no device kernels are
    launched off the main thread** (device kernels must share the model's
    single launching thread):

    - :meth:`_load_host` is the **producer**. It runs on a worker thread
      and performs only thread-safe work: reading, decoding, and staging
      into host memory. It returns a :class:`HostPayload`.
    - :meth:`_consume` is the **consumer**. It runs on the thread that
      calls :meth:`consume` / :meth:`__getitem__` (the main thread, in
      practice) and performs the host-to-device transfer and device
      transforms on the assigned CUDA stream.

    :class:`Dataset` and :class:`MeshDataset` override both hooks to
    perform the real split. The default implementations fall back to
    running the full :meth:`_load` on the worker for any subclass that
    does not override them.
    """

    def __init__(
        self,
        *,
        num_workers: int = 2,
    ) -> None:
        self._executor: Optional[ThreadPoolExecutor] = None
        self._num_workers = num_workers
        self._lock = threading.Lock()
        # Futures still in flight, tracked so close() can drain them.
        self._inflight: set[Future] = set()
        # Index-keyed handles backing the prefetch()/__getitem__ compat API.
        self._prefetch_handles: dict[int, PrefetchHandle] = {}
        # Per-sample preprocessing CUDA events recorded by _consume when
        # invoked with defer_sync=True.  The compute-stream wait on these is
        # deferred to the DataLoader, which inserts it right before the batch
        # is yielded (i.e. after the previous batch's model work is enqueued)
        # so preprocessing genuinely overlaps the prior batch's compute.
        # Always accessed on the main thread so no locking is needed.
        self._events_pending: list = []

    def _pop_events(self) -> list:
        """Return and clear the pending preprocessing-event list.

        Called by the DataLoader after each ``consume()`` (or batch of
        consumes) made with ``defer_sync=True`` to retrieve the CUDA events
        that ``_consume`` recorded but did not yet wait on.  The DataLoader
        inserts ``compute_stream.wait_event`` for each returned event right
        before the corresponding batch is yielded.

        Returns
        -------
        list
            CUDA events recorded during the most recent deferred
            ``consume()`` call(s).  Empty when none were recorded (e.g. no
            stream, CPU target, or ``defer_sync=False``).
        """
        lst = self._events_pending
        self._events_pending = []
        return lst

    @abstractmethod
    def _load(self, index: int) -> tuple[Any, dict[str, Any]]:
        """Load and return a single sample ``(data, metadata)``.

        This is the synchronous, full-pipeline hook that subclasses must
        implement. It is called directly from :meth:`__getitem__` when the
        index was not prefetched.
        """
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    # ------------------------------------------------------------------
    # Producer / consumer hooks (overridden by Dataset / MeshDataset)
    # ------------------------------------------------------------------

    def _load_host(self, work_item: Any) -> HostPayload:
        """Producer stage: load *work_item* into a :class:`HostPayload`.

        Runs on a worker thread and must not launch device kernels. The
        default implementation runs the full :meth:`_load` pipeline for
        backward compatibility; subclasses that use device transforms
        override this to stop at host staging.

        Parameters
        ----------
        work_item : Any
            Work item to load (an ``int`` index by default).

        Returns
        -------
        HostPayload
            Payload carrying the host data and metadata, or a captured
            error.
        """
        try:
            data, metadata = self._load(work_item)
            return HostPayload(work_item=work_item, data=data, metadata=metadata)
        except Exception as e:  # noqa: BLE001
            return HostPayload(work_item=work_item, error=e)

    def _consume(
        self,
        payload: HostPayload,
        stream: Optional[torch.cuda.Stream] = None,
        *,
        defer_sync: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        """Consumer stage: turn a :class:`HostPayload` into ``(data, metadata)``.

        Runs on the calling (main) thread. The default implementation
        unwraps the payload and re-raises any captured error; the
        ``stream`` and ``defer_sync`` arguments are ignored. Subclasses
        override this to perform the host-to-device transfer and device
        transforms.

        Parameters
        ----------
        payload : HostPayload
            Producer payload from :meth:`_load_host`.
        stream : torch.cuda.Stream, optional
            Stream for the consume step.
        defer_sync : bool, default=False
            When True (and a stream is used), record the preprocessing CUDA
            event into :attr:`_events_pending` instead of making the compute
            stream wait on it here.  The DataLoader then inserts the wait
            just before the batch is yielded so preprocessing overlaps the
            previous batch's compute.  Ignored by this default implementation.

        Returns
        -------
        tuple[Any, dict[str, Any]]
            The sample data and its metadata.
        """
        if payload.error is not None:
            raise payload.error
        return payload.data, payload.metadata

    # ------------------------------------------------------------------
    # FIFO prefetch primitive (used by the DataLoader's pump)
    # ------------------------------------------------------------------

    def submit(
        self,
        work_item: Any,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> PrefetchHandle:
        """Submit *work_item* for background loading and return its handle.

        Only the (thread-safe) producer stage runs on the worker pool. The
        returned :class:`PrefetchHandle` is later passed to :meth:`consume`
        on the main thread. Safe to call from a dispatcher thread distinct
        from the consumer.

        Parameters
        ----------
        work_item : Any
            Work item to load.
        stream : torch.cuda.Stream, optional
            Stream the consumer should use for this sample.

        Returns
        -------
        PrefetchHandle
            Handle bundling the producer future and the assigned stream.
        """
        executor = self._ensure_executor()
        future = executor.submit(self._load_host, work_item)
        with self._lock:
            self._inflight.add(future)
        future.add_done_callback(self._discard_inflight)
        return PrefetchHandle(future=future, stream=stream)

    def consume(
        self, handle: PrefetchHandle, *, defer_sync: bool = False
    ) -> tuple[Any, dict[str, Any]]:
        """Resolve a :meth:`submit` handle into ``(data, metadata)``.

        Blocks until the producer future is ready, then runs the consumer
        stage on the calling thread (host-to-device transfer and device
        transforms on the handle's stream).

        Parameters
        ----------
        handle : PrefetchHandle
            Handle returned by :meth:`submit`.
        defer_sync : bool, default=False
            Forwarded to :meth:`_consume`.  When True, the compute-stream
            wait on this sample's preprocessing event is deferred to the
            caller (the DataLoader) via :attr:`_events_pending`.  Standalone
            callers leave this False so the wait is enqueued inline and the
            result is immediately safe to use on the current stream.

        Returns
        -------
        tuple[Any, dict[str, Any]]
            The sample data and its metadata.
        """
        payload = handle.future.result()  # re-raises producer errors via _consume
        return self._consume(payload, handle.stream, defer_sync=defer_sync)

    # ------------------------------------------------------------------
    # Concrete interface
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        """Return sample *index*, using a prefetched result when available.

        When the index was prefetched via :meth:`prefetch`, the pending
        handle is consumed on the calling thread (so device transforms run
        here, not on the worker). Otherwise the sample is loaded
        synchronously.
        """
        with self._lock:
            handle = self._prefetch_handles.pop(index, None)
        if handle is not None:
            return self.consume(handle)
        return self._load(index)

    def prefetch(
        self,
        index: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Start prefetching *index*, retrievable later via :meth:`__getitem__`.

        Index-keyed convenience wrapper around :meth:`submit`. A repeated
        prefetch of an in-flight index is a no-op. Not thread-safe for
        concurrent calls prefetching the same index; the DataLoader uses
        :meth:`submit` / :meth:`consume` instead.

        Parameters
        ----------
        index : int
            Sample index to prefetch.
        stream : torch.cuda.Stream, optional
            Stream the consumer should use for this sample.
        """
        with self._lock:
            if index in self._prefetch_handles:
                return
        handle = self.submit(index, stream)
        with self._lock:
            if index in self._prefetch_handles:
                return  # raced; the extra handle's future is reaped on completion
            self._prefetch_handles[index] = handle

    def cancel_prefetch(self, index: Optional[int] = None) -> None:
        """Discard prefetch handles (already-running tasks still complete)."""
        with self._lock:
            if index is None:
                self._prefetch_handles.clear()
            else:
                self._prefetch_handles.pop(index, None)

    def close(self) -> None:
        """Drain in-flight prefetches and shut down the thread pool."""
        with self._lock:
            futures = list(self._inflight)
            self._prefetch_handles.clear()
        for future in futures:
            try:
                future.result(timeout=30.0)
            except Exception:  # noqa: BLE001, S110
                pass
        with self._lock:
            self._inflight.clear()

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _discard_inflight(self, future: Future) -> None:
        """Done-callback: drop a finished future from the in-flight set."""
        with self._lock:
            self._inflight.discard(future)

    def _ensure_executor(self) -> ThreadPoolExecutor:
        with self._lock:
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


class IterableDatasetBase(ABC):
    """Abstract base for generator-style datasets driven on the main thread.

    Unlike :class:`DatasetBase`, an iterable dataset has no length and no
    indexing: it produces data by iteration only. The
    :class:`~physicsnemo.datapipes.DataLoader` drives it entirely on the
    main thread (no worker pool), so :meth:`__iter__` may freely launch
    device kernels and use CUDA streams -- the property that makes online
    simulation safe here but unsafe on the worker-pool preload path.

    Emission modes
    --------------
    - **Per-sample** (default, ``yields_batches = False``): :meth:`__iter__`
      yields ``(data, metadata)`` for one sample at a time and the loader
      collates ``batch_size`` of them.
    - **Self-batching** (``yields_batches = True``): :meth:`__iter__` yields
      a fully-formed batch already and the loader passes it through
      unchanged (``batch_size``/``drop_last`` do not apply).

    Subclasses may optionally implement :meth:`set_epoch` and
    :meth:`set_generator` for reproducible seeding.
    """

    # When True, __iter__ yields ready-made batches and the loader does not
    # re-collate (e.g. an online simulator that produces a batch per step).
    yields_batches: bool = False

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """Yield samples ``(data, metadata)`` or ready-made batches.

        Returns
        -------
        Iterator
            Per-sample ``(data, metadata)`` tuples, or full batches when
            :attr:`yields_batches` is True.
        """
        ...

    def set_epoch(self, epoch: int) -> None:
        """Reseed for *epoch* (no-op by default).

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """

    def set_generator(self, generator: torch.Generator) -> None:
        """Seed the dataset's randomness from *generator* (no-op by default).

        Parameters
        ----------
        generator : torch.Generator
            Parent generator supplied by the DataLoader.
        """

    def __enter__(self) -> "IterableDatasetBase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release any resources held by the dataset (no-op by default)."""
