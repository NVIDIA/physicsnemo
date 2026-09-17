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
DataLoader - Batched iteration over datasets with prefetching.

The DataLoader orchestrates efficient batch loading by leveraging
the Dataset's prefetching capabilities with CUDA streams.
By default, returns batched TensorDict for PyTorch DataLoader compatibility.
When collate_metadata=True, returns (TensorDict, list[dict]) tuples.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any, Callable, Iterator, Optional, Sequence

import torch
from tensordict import TensorDict
from torch.utils.data import RandomSampler, Sampler, SequentialSampler

from physicsnemo.datapipes._rng import fork_generator
from physicsnemo.datapipes.collate import Collator, get_collator
from physicsnemo.datapipes.io_pump import BATCH_BOUNDARY, IOPump
from physicsnemo.datapipes.protocols import (
    DatasetBase,
    IterableDatasetBase,
    preprocessing_stream,
    record_consumer_stream,
)
from physicsnemo.datapipes.registry import register


@register()
class DataLoader:
    """
    Batched iteration over a Dataset with stream-based prefetching.

    Unlike PyTorch's DataLoader which uses CPU multiprocessing, this
    DataLoader uses CUDA streams to overlap data loading, preprocessing,
    and collation. This is more efficient for SciML workloads where:

    - Datasets are huge
    - Batches are small
    - Preprocessing benefits from GPU acceleration

    Features:

    - Stream-based parallelism (one stream per sample in flight)
    - Toggleable prefetching for debugging
    - Compatible with PyTorch samplers (DistributedSampler, etc.)
    - Familiar torch DataLoader interface

    Two data paths
    --------------
    The path is selected by dataset type:

    - **Map-style** (:class:`~physicsnemo.datapipes.protocols.DatasetBase`):
      a dispatcher thread (:class:`~physicsnemo.datapipes.io_pump.IOPump`)
      lazily submits sample loads to a worker pool and forwards batch
      boundaries, while the main thread consumes handles (host-to-device
      transfer + transforms on a preprocessing stream).
    - **Iterable** (:class:`~physicsnemo.datapipes.protocols.IterableDatasetBase`):
      a generator dataset driven main-thread-only (no sampler, no pump, no
      worker pool). ``len()`` is undefined and ``shuffle``/``sampler`` are
      ignored; generation runs on a preprocessing stream with the same
      event handoff so it overlaps training.

    Concurrency model
    -----------------
    A dedicated dispatcher thread keeps the I/O pipeline primed by
    submitting sample loads ahead of consumption, bounded by
    ``prefetch_factor`` batches worth of in-flight samples. The main
    thread is the sole consumer: it performs all host-to-device transfers
    and GPU transforms on the prefetch streams, so transforms run on the
    assigned preprocessing stream and overlap the compute stream.

    For the pipeline to stay primed, the main thread must not block: keep
    reader output (optionally) pinned (so host-to-device copies are asynchronous) and
    avoid host readbacks (``.item()``), data-dependent shapes, and
    GIL-bound pure-Python transforms on the launch path.

    Examples
    --------
    >>> from physicsnemo.datapipes import DataLoader, Dataset, HDF5Reader, Normalize
    >>>
    >>> dataset = Dataset(  # doctest: +SKIP
    ...     HDF5Reader("data.h5", fields=["input", "target"]),
    ...     transforms=Normalize(["input"], method="mean_std", means={"input": 0.0}, stds={"input": 1.0}),
    ...     device="cuda",  # Automatic GPU transfer
    ... )
    >>> loader = DataLoader(dataset, batch_size=16, shuffle=True)  # doctest: +SKIP
    >>>
    >>> for batch in loader:  # doctest: +SKIP
    ...     output = model(batch["input"])

    With DistributedSampler:

    >>> from torch.utils.data.distributed import DistributedSampler
    >>> sampler = DistributedSampler(dataset)  # doctest: +SKIP
    >>> loader = DataLoader(dataset, batch_size=16, sampler=sampler)  # doctest: +SKIP
    """

    def __init__(
        self,
        dataset: DatasetBase | IterableDatasetBase,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        drop_last: bool = False,
        collate_fn: Optional[
            Collator
            | Callable[
                [Sequence[tuple[TensorDict, dict[str, Any]]]],
                tuple[TensorDict, list[dict[str, Any]]],
            ]
        ] = None,
        collate_metadata: bool = False,
        prefetch_factor: int = 2,
        num_streams: int = 4,
        use_streams: bool = True,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the DataLoader.

        Parameters
        ----------
        dataset : DatasetBase or IterableDatasetBase
            Dataset to load from. A map-style :class:`DatasetBase`
            (e.g. :class:`Dataset`, :class:`MeshDataset`) or an
            :class:`IterableDatasetBase` generator dataset.
        batch_size : int, default=1
            Number of samples per batch.
        shuffle : bool, default=False
            If True, shuffle indices each epoch. Ignored if sampler provided.
        sampler : Sampler, optional
            Custom sampler for index generation. If provided, shuffle is ignored.
        drop_last : bool, default=False
            If True, drop the last incomplete batch.
        collate_fn : Collator or Callable, optional
            Function to collate samples into batches. Defaults to stacking.
        collate_metadata : bool, default=False
            If True, collate metadata into a list of dicts. Set to False for
            compatibility with PyTorch DataLoader. Only used when collate_fn
            is None (uses default collator).
        prefetch_factor : int, default=2
            Number of batches to prefetch ahead. Set to 0 to disable prefetching.
        num_streams : int, default=4
            Number of CUDA streams for prefetching.
        use_streams : bool, default=True
            If True, use CUDA streams for overlap. Set False for debugging
            or CPU-only operation.
        seed : int, optional
            Master seed for all pipeline randomness.  When set, the
            DataLoader derives independent generators for the sampler,
            reader, and every stochastic transform, making the full
            pipeline reproducible.  Use :meth:`set_epoch` to vary the
            random sequence across epochs while staying deterministic.

        Raises
        ------
        ValueError
            If batch_size < 1.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        # Depth restored by enable_prefetch(); a loader constructed with
        # prefetch_factor=0 re-enables at the default depth of 2.
        self._saved_prefetch_factor = prefetch_factor if prefetch_factor > 0 else 2
        self.num_streams = num_streams
        self.use_streams = use_streams and torch.cuda.is_available()
        self._seed = seed
        # Iterable (generator) datasets are driven main-thread-only: no
        # sampler, no worker-pool prefetch (see _iter_iterable).
        self._iterable = isinstance(dataset, IterableDatasetBase)

        # Build master generator and fork for sampler + dataset
        sampler_generator: torch.Generator | None = None
        if seed is not None:
            master = torch.Generator()
            master.manual_seed(seed)
            # Fork: child 0 → sampler, child 1 → dataset
            forks = fork_generator(master, 2)
            sampler_generator = forks[0]
            if hasattr(dataset, "set_generator"):
                dataset.set_generator(forks[1])

        # Handle sampler. Iterable datasets have no indices, so they carry
        # no sampler and ignore shuffle.
        if self._iterable:
            if sampler is not None or shuffle:
                warnings.warn(
                    "shuffle/sampler are ignored for iterable datasets; "
                    "the generator controls sample order.",
                    stacklevel=2,
                )
            self.sampler = None
        elif sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(dataset, generator=sampler_generator)
        else:
            self.sampler = SequentialSampler(dataset)

        # Handle collation
        self.collate_fn = get_collator(collate_fn, collate_metadata=collate_metadata)

        # Create CUDA streams: prefetch uses several round-robin streams; the
        # iterable path uses the first as its preprocessing stream.
        self._streams: list[torch.cuda.Stream] = []
        if self.use_streams:
            for _ in range(num_streams):
                self._streams.append(torch.cuda.Stream())

    def __len__(self) -> int:
        """
        Return the number of batches.

        Returns
        -------
        int
            Number of batches in the dataloader.

        Raises
        ------
        TypeError
            If the dataset is iterable (generator-style), which has no
            defined length.
        """
        if self._iterable:
            raise TypeError("len() is undefined for an iterable (generator) dataset")
        n_samples = (
            len(self.sampler) if hasattr(self.sampler, "__len__") else len(self.dataset)
        )
        if self.drop_last:
            return n_samples // self.batch_size
        return (n_samples + self.batch_size - 1) // self.batch_size

    def _generate_batches(self) -> Iterator[list[int]]:
        """
        Generate batches of indices.

        Yields
        ------
        list[int]
            List of sample indices for each batch.
        """
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __iter__(
        self,
    ) -> Iterator[TensorDict | tuple[TensorDict, list[dict[str, Any]]]]:
        """
        Iterate over batches.

        Uses the self-priming :class:`IOPump` to overlap host-side I/O
        (on the dataset's worker threads) with main-thread consumption
        whenever ``prefetch_factor > 0``. This threaded producer path is
        independent of CUDA streams: when streams are enabled (and CUDA
        is available) each prefetched sample is also assigned a stream so
        the host-to-device copy and GPU transforms overlap; otherwise the
        same path runs with ``stream=None`` (still overlapping disk I/O
        with the main thread). Set ``prefetch_factor=0`` for fully
        synchronous iteration.

        Yields
        ------
        TensorDict or tuple[TensorDict, list[dict[str, Any]]]
            Batched TensorDict if collate_metadata=False (default),
            or tuple of (batched TensorDict, list of metadata dicts)
            if collate_metadata=True.
        """
        if self._iterable:
            yield from self._iter_iterable()
        elif self.prefetch_factor > 0:
            yield from self._iter_prefetch()
        else:
            yield from self._iter_simple()

    def _iter_simple(
        self,
    ) -> Iterator[TensorDict | tuple[TensorDict, list[dict[str, Any]]]]:
        """
        Simple synchronous iteration without prefetching.

        Yields
        ------
        TensorDict or tuple[TensorDict, list[dict[str, Any]]]
            Collated batch.
        """
        for batch_indices in self._generate_batches():
            samples = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(samples)

    def _work_stream(self) -> Iterator[Any]:
        """Lazily yield sampler indices delimited by :data:`BATCH_BOUNDARY`.

        Buffers at most one batch of indices (never the whole epoch), so
        arbitrarily long samplers stream without up-front materialization.
        A boundary is emitted after each full batch; a trailing partial
        batch is emitted only when ``drop_last`` is False.

        Yields
        ------
        int or object
            Sample indices, with :data:`BATCH_BOUNDARY` after each batch.
        """
        batch: list[int] = []
        for index in self.sampler:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield from batch
                yield BATCH_BOUNDARY
                batch = []
        if batch and not self.drop_last:
            yield from batch
            yield BATCH_BOUNDARY

    def _iter_prefetch(
        self,
    ) -> Iterator[TensorDict | tuple[TensorDict, list[dict[str, Any]]]]:
        """
        Iteration driven by a self-priming prefetch pump with one-batch lookahead.

        A dedicated dispatcher thread (the :class:`IOPump`) lazily pulls
        the index stream and submits sample loads to the dataset's worker
        pool, keeping a bounded number of samples in flight regardless of
        the consumer's cadence. The main thread is a pure drain loop: it
        pulls ready handles in order, runs the per-sample consume step
        (host-to-device transfer plus GPU transforms on the assigned
        stream), and reassembles batches from the boundary markers the
        pump forwards.

        Stream assignment is optional and decoupled from the threaded
        producer: when CUDA streams are enabled a stream is round-robined
        per sample (so preprocessing overlaps the previous batch's compute);
        otherwise dispatch passes ``stream=None`` and the path still
        overlaps host-side I/O with main-thread consumption.

        Because dispatch lives off the main thread, the pipeline stays
        primed even while the main thread is blocked launching kernels or
        running the model. All device-kernel launches happen here, on the
        single main thread.

        One-batch lookahead for preprocessing stream overlap
        ----------------------------------------------------
        Before yielding batch N, this method eagerly drains the pump for
        batch N+1's items and calls ``consume(..., defer_sync=True)`` on
        each. ``consume`` enqueues the host-to-device transfer and GPU
        transforms on a preprocessing stream asynchronously and *records* a
        CUDA event, but -- with ``defer_sync=True`` -- does **not** make the
        compute stream wait on it yet. The wait is inserted here, by
        :func:`gate_compute_stream`, immediately before the owning batch is
        yielded.

        This ordering is the whole point of the lookahead. If ``_consume``
        made the compute stream wait on batch N+1's event during the
        lookahead drain (i.e. before yielding batch N), that wait would be
        enqueued on the compute stream *ahead* of batch N's model kernels,
        so batch N's model would block on batch N+1's preprocessing -- the
        opposite of overlap. By deferring the wait, the compute-stream order
        becomes ``..., model_{N-1}, wait(prep_N), model_N, ...``: batch N's
        preprocessing (already in flight on its own stream) overlaps batch
        N-1's compute, and each model only blocks on its own batch's
        preprocessing.

        With ``prefetch_factor >= 2`` the IOPump has already dispatched
        batch N+1's disk-I/O ahead of time, so the ``future.result()``
        inside ``consume()`` returns immediately and the lookahead drain
        adds negligible host-side latency.

        Yields
        ------
        TensorDict or tuple[TensorDict, list[dict[str, Any]]]
            Collated batch.
        """
        # Streams are an optional accelerator on top of the threaded
        # producer; only assign them when actually available.
        use_streams = self.use_streams and len(self._streams) > 0

        # The compute stream the training loop runs on. Captured once on the
        # main thread; the deferred per-batch preprocessing waits are
        # enqueued onto it right before each batch is yielded.
        compute_stream = torch.cuda.current_stream() if use_streams else None

        # Round-robin a stream per sample at dispatch time (when enabled);
        # submit returns a handle the consumer resolves in order.
        stream_counter = itertools.count()

        def dispatch(index: int) -> Any:
            stream = (
                self._streams[next(stream_counter) % self.num_streams]
                if use_streams
                else None
            )
            return self.dataset.submit(index, stream=stream)

        # Depth = prefetch_factor batches worth of samples kept in flight
        # (at least the stream count when streams drive the overlap).
        depth = max(self.prefetch_factor * self.batch_size, 1)
        if use_streams:
            depth = max(depth, self.num_streams)

        pump = IOPump(self._work_stream(), dispatch, depth=depth)
        pump_iter = iter(pump)

        def drain_one_batch() -> tuple[list[Any], bool]:
            """Consume pump items up to the next BATCH_BOUNDARY.

            Each non-boundary item is passed through
            :meth:`~physicsnemo.datapipes.protocols.DatasetBase.consume`
            with ``defer_sync=True``, which enqueues the host-to-device copy
            and GPU transforms on a preprocessing stream (asynchronously)
            and records a CUDA event *without* making the compute stream wait
            on it. The wait is inserted later by :func:`gate_compute_stream`.

            Returns
            -------
            tuple[list, bool]
                ``(samples, has_batch)`` where *has_batch* is ``True`` when
                a ``BATCH_BOUNDARY`` was found and ``False`` when the pump
                was exhausted without one (only possible for an empty or
                fully-consumed source).
            """
            samples: list[Any] = []
            for item in pump_iter:
                if item is BATCH_BOUNDARY:
                    return samples, True
                samples.append(self.dataset.consume(item, defer_sync=True))
            return samples, False

        def gate_compute_stream(events: list) -> None:
            """Make the compute stream wait on a batch's preprocessing events.

            Called immediately before a batch is yielded, so the wait is
            ordered *after* the previous batch's model kernels (already
            enqueued by the prior iteration's ``yield``). The batch's
            preprocessing -- launched on its side stream during the lookahead
            drain -- thus overlaps the previous batch's compute, and the
            model only blocks on its own batch's preprocessing.

            A no-op when streams are disabled (``events`` is empty).
            """
            if compute_stream is not None:
                for event in events:
                    compute_stream.wait_event(event)

        try:
            # Prime: consume the first batch's items, enqueueing their
            # preprocessing stream work (event recorded, compute-stream wait
            # deferred to gate_compute_stream below).
            current_samples, has_first = drain_one_batch()
            # Per-batch preprocessing events whose compute-stream wait was
            # deferred; gate_compute_stream consumes these right before yield.
            current_events = self.dataset._pop_events()
            if not has_first:
                # Source was empty or had no boundary; yield whatever arrived.
                if current_samples:
                    gate_compute_stream(current_events)
                    yield self.collate_fn(current_samples)
                return

            while True:
                # Eagerly drain the NEXT batch before yielding the current
                # one. This enqueues batch N+1's H2D transfer and GPU
                # transforms on the preprocessing streams so they run
                # concurrently with the training loop's compute-stream work
                # for batch N (forward / backward / optimizer).
                next_samples, has_next = drain_one_batch()
                next_events = self.dataset._pop_events()

                # Gate the compute stream on the CURRENT batch's preprocessing
                # *now* -- after the previous iteration's yield enqueued the
                # previous batch's model work, and after the NEXT batch's
                # preprocessing was launched above on its own stream. This is
                # what lets the next batch's preprocessing overlap this
                # batch's compute instead of blocking it.
                gate_compute_stream(current_events)

                # Yield the current batch. While the training loop runs,
                # the preprocessing streams are already working on the next
                # batch.
                yield self.collate_fn(current_samples)

                if not has_next:
                    # Pump exhausted after the lookahead drain; yield the
                    # final partial batch (empty when drop_last trimmed it)
                    # then stop.
                    if next_samples:
                        gate_compute_stream(next_events)
                        yield self.collate_fn(next_samples)
                    break

                current_samples = next_samples
                current_events = next_events
        finally:
            # Stop the dispatcher (handles early break / exhaustion) and
            # drop any prefetched-but-unconsumed handles.
            pump.stop()
            self.dataset.cancel_prefetch()

    def _iter_iterable(
        self,
    ) -> Iterator[Any]:
        """
        Main-thread-only iteration for generator (iterable) datasets.

        There is no worker pool: the dataset's generator runs on the main
        thread, so it may freely launch device kernels / use streams. Each
        item is generated on a preprocessing stream (when streams are
        enabled) and handed to the compute stream via a CUDA event, so
        generation of the next item can overlap training on the current
        one. A generator that forces a host readback simply serializes
        itself.

        Two emission modes are supported (see :class:`IterableDatasetBase`):
        per-sample items are collated into ``batch_size`` batches (with
        ``drop_last`` trimming the trailing partial batch); a self-batching
        generator (``yields_batches = True``) has each batch passed through
        unchanged.

        Yields
        ------
        Any
            Collated batches, or generator-produced batches when the
            dataset is self-batching.
        """
        use_stream = self.use_streams and len(self._streams) > 0
        prep_stream = self._streams[0] if use_stream else None
        compute_stream = torch.cuda.current_stream() if use_stream else None
        self_batching = getattr(self.dataset, "yields_batches", False)

        iterator = iter(self.dataset)
        samples: list[Any] = []
        while True:
            # Generate the next item on the preprocessing stream, then order
            # it before the compute stream without blocking the host.
            with preprocessing_stream(prep_stream):
                try:
                    item = next(iterator)
                except StopIteration:
                    break
            if use_stream:
                event = torch.cuda.Event()
                event.record(prep_stream)
                compute_stream.wait_event(event)
                # The compute stream will read this item's tensors; record it
                # so the allocator does not recycle their blocks for the
                # generator's next allocations while those reads are pending.
                # This protects allocator reuse only: a generator that reuses
                # its *own* output buffers in place across iterations still
                # races with pending compute-stream reads.
                record_consumer_stream(item, compute_stream)

            if self_batching:
                yield item
                continue

            samples.append(item)
            if len(samples) == self.batch_size:
                yield self.collate_fn(samples)
                samples = []

        if not self_batching and samples and not self.drop_last:
            yield self.collate_fn(samples)

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for the sampler and the full data pipeline.

        Propagates the epoch to the sampler (for
        :class:`~torch.utils.data.distributed.DistributedSampler`),
        the dataset, reader, and every stochastic transform so all
        RNG streams are reseeded deterministically.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def enable_prefetch(self) -> None:
        """Re-enable prefetching after :meth:`disable_prefetch`.

        Restores the threaded prefetch pump on every platform; CUDA
        streams are re-enabled only when CUDA is available (they are an
        optional accelerator on top of the threaded producer). A loader
        constructed with ``prefetch_factor=0`` is enabled at the default
        depth of 2. Takes effect at the next iteration.
        """
        if self.prefetch_factor == 0:
            self.prefetch_factor = self._saved_prefetch_factor
        if torch.cuda.is_available():
            if not self._streams:
                for _ in range(self.num_streams):
                    self._streams.append(torch.cuda.Stream())
            self.use_streams = True

    def disable_prefetch(self) -> None:
        """Return to fully synchronous iteration (useful for debugging).

        Disables both the threaded prefetch pump (``prefetch_factor`` is
        set to 0; the previous value is restored by
        :meth:`enable_prefetch`) and the CUDA stream handoff. Takes
        effect at the next iteration; an in-flight iterator is
        unaffected.
        """
        if self.prefetch_factor > 0:
            self._saved_prefetch_factor = self.prefetch_factor
            self.prefetch_factor = 0
        self.use_streams = False
        # Iterable datasets have no prefetch machinery to cancel;
        # MultiDataset duck-types DatasetBase, so feature-detect rather
        # than isinstance.
        cancel = getattr(self.dataset, "cancel_prefetch", None)
        if cancel is not None:
            cancel()

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation of the DataLoader.
        """
        return (
            f"DataLoader(\n"
            f"  dataset={self.dataset},\n"
            f"  batch_size={self.batch_size},\n"
            f"  shuffle={self.shuffle},\n"
            f"  drop_last={self.drop_last},\n"
            f"  prefetch_factor={self.prefetch_factor},\n"
            f"  num_streams={self.num_streams},\n"
            f"  use_streams={self.use_streams}\n"
            f")"
        )
