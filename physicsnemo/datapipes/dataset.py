# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
Dataset - Combines a Reader with a transform pipeline.

The Dataset is the primary interface for accessing preprocessed data.
It wraps a Reader and applies transforms to produce ready-to-use TensorDicts.
Supports prefetching with CUDA streams for overlapped IO and computation,
and automatic device transfer when device parameter is specified.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.readers.base import Reader
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.base import Transform
from physicsnemo.datapipes.transforms.compose import Compose
from physicsnemo.distributed import DistributedManager


@dataclass
class _PrefetchResult:
    """Result of a prefetch operation."""

    index: int
    data: Optional[TensorDict] = None
    metadata: Optional[dict[str, Any]] = None
    error: Optional[Exception] = None
    event: Optional[torch.cuda.Event] = None  # For stream sync


@register()
class Dataset:
    """
    A dataset combining a Reader with a transform pipeline.

    The Dataset provides a torch-like interface for accessing data:

    - Indexing: dataset[i] returns transformed sample i
    - Iteration: for sample in dataset
    - Length: len(dataset)
    - Prefetching: dataset.prefetch(i, stream) for async loading

    The pipeline is: Reader → Transforms → Sample

    Prefetching Model
    -----------------
    The dataset supports prefetching samples using a thread pool.
    When a CUDA stream is provided, GPU operations (device transfer,
    GPU transforms) happen on that stream, allowing overlap with
    other computation.

    >>> # Start prefetching
    >>> dataset.prefetch(0, stream=stream0)  # doctest: +SKIP
    >>> dataset.prefetch(1, stream=stream1)  # doctest: +SKIP
    >>>
    >>> # Retrieve results (waits if not ready)
    >>> sample_0 = dataset[0]  # Uses prefetched result  # doctest: +SKIP

    Examples
    --------
    >>> from physicsnemo.datapipes import Dataset, HDF5Reader, Normalize
    >>>
    >>> reader = HDF5Reader("data.h5", fields=["pressure", "velocity"])  # doctest: +SKIP
    >>> transforms = Normalize(  # doctest: +SKIP
    ...     ["pressure"],
    ...     method="mean_std",
    ...     means={"pressure": 0.0},  # doctest: +SKIP
    ...     stds={"pressure": 1.0},  # doctest: +SKIP
    ... )
    >>>
    >>> dataset = Dataset(reader, transforms=transforms, device="cuda")  # doctest: +SKIP
    >>> sample, metadata = dataset[0]  # doctest: +SKIP
    """

    def __init__(
        self,
        reader: Reader,
        *,
        transforms: Optional[Transform | Sequence[Transform]] = None,
        device: Optional[str | torch.device] = None,
        num_workers: int = 2,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        reader : Reader
            Data reader providing raw samples.
        transforms : Transform or Sequence[Transform], optional
            Transform or sequence of transforms to apply.
            If a sequence, they are composed in order.
        device : str or torch.device, optional
            Target device for automatic transfer (e.g., "cuda", "cuda:0").
            If None, no automatic transfer is performed (data stays on CPU).
            When specified, data is transferred to this device before transforms.
            If device is "auto", will select the device with distributed manager.
            Auto device falls back to CPU.
        num_workers : int, default=2
            Number of worker threads for prefetching.

        Raises
        ------
        TypeError
            If reader is not a Reader instance.
        """
        if not isinstance(reader, Reader):
            raise TypeError(
                f"reader must be a Reader instance, got {type(reader).__name__}"
            )

        self.reader = reader
        self.num_workers = num_workers
        if device == "auto":
            if torch.cuda.is_available():
                if DistributedManager.is_initialized():
                    device = DistributedManager().device
                else:
                    device = "cuda:0"
            else:
                device = "cpu"

        # Now, instantiate the device if not already done:
        match device:
            case torch.device():
                self.target_device = device
            case str():
                self.target_device = torch.device(device)
            case None:
                self.target_device = None

        # Handle transforms
        if transforms is None:
            self.transforms: Optional[Transform] = None
        elif isinstance(transforms, Transform):
            self.transforms = transforms
        elif isinstance(transforms, Sequence):
            if len(transforms) == 0:
                self.transforms = None
            elif len(transforms) == 1:
                self.transforms = transforms[0]
            else:
                self.transforms = Compose(transforms)
        else:
            raise TypeError(
                f"transforms must be Transform, Sequence[Transform], or None, "
                f"got {type(transforms).__name__}"
            )

        # Share device with transforms so their internal state is on the right device
        if self.target_device is not None and self.transforms is not None:
            self.transforms.to(self.target_device)

        # Prefetch state - using thread-safe dict for results
        # Key: index, Value: Future[_PrefetchResult]
        self._prefetch_futures: dict[int, Future[_PrefetchResult]] = {}
        self._executor: Optional[ThreadPoolExecutor] = None

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """
        Lazily create the thread pool executor.

        Returns
        -------
        ThreadPoolExecutor
            The thread pool executor for prefetching.
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="datapipe_prefetch",
            )
        return self._executor

    def _load_and_transform(
        self,
        index: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> _PrefetchResult:
        """
        Load a sample and apply transforms. Called by worker threads.

        Parameters
        ----------
        index : int
            Sample index.
        stream : torch.cuda.Stream, optional
            Optional CUDA stream for GPU operations.

        Returns
        -------
        _PrefetchResult
            PrefetchResult with data, metadata, or error.
        """
        result = _PrefetchResult(index=index)

        try:
            # Load from reader (CPU, potentially slow IO)
            data, metadata = self.reader[index]

            # Auto-transfer to target device if specified
            if self.target_device is not None:
                if stream is not None:
                    with torch.cuda.stream(stream):
                        data = data.to(self.target_device, non_blocking=True)
                else:
                    data = data.to(self.target_device, non_blocking=True)

            # Apply transforms (data is now on target device if specified)
            if self.transforms is not None:
                if stream is not None:
                    with torch.cuda.stream(stream):
                        data = self.transforms(data)
                    # Record event for synchronization
                    result.event = torch.cuda.Event()
                    result.event.record(stream)
                else:
                    data = self.transforms(data)

            result.data = data
            result.metadata = metadata

        except Exception as e:
            result.error = e

        return result

    def prefetch(
        self,
        index: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        Start prefetching a sample asynchronously.

        The sample will be loaded in a background thread. If a CUDA stream
        is provided, GPU operations happen on that stream.

        Call __getitem__ to retrieve the result (it will wait if needed).

        Parameters
        ----------
        index : int
            Sample index to prefetch.
        stream : torch.cuda.Stream, optional
            Optional CUDA stream for GPU operations.
        """
        # Don't prefetch if already in flight
        if index in self._prefetch_futures:
            return

        executor = self._ensure_executor()
        future = executor.submit(self._load_and_transform, index, stream)
        self._prefetch_futures[index] = future

    def prefetch_batch(
        self,
        indices: Sequence[int],
        streams: Optional[Sequence[torch.cuda.Stream]] = None,
    ) -> None:
        """
        Start prefetching multiple samples.

        Parameters
        ----------
        indices : Sequence[int]
            Sample indices to prefetch.
        streams : Sequence[torch.cuda.Stream], optional
            Optional CUDA streams, one per index. If shorter than
            indices, streams are cycled. If None, no streams used.
        """
        for i, idx in enumerate(indices):
            stream = None
            if streams:
                stream = streams[i % len(streams)]
            self.prefetch(idx, stream=stream)

    def __getitem__(self, index: int) -> tuple[TensorDict, dict[str, Any]]:
        """
        Get a transformed sample by index.

        If the index was prefetched, returns the prefetched result
        (waiting for completion if necessary). Otherwise loads synchronously.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        tuple[TensorDict, dict[str, Any]]
            Tuple of (TensorDict with transformed data, metadata dict).

        Raises
        ------
        IndexError
            If index is out of range.
        Exception
            If prefetch failed, re-raises the error.
        """
        # Check if prefetched
        future = self._prefetch_futures.pop(index, None)

        if future is not None:
            # Wait for prefetch to complete
            result = future.result()

            if result.error is not None:
                raise result.error

            # Sync stream if needed
            if result.event is not None:
                result.event.synchronize()

            return result.data, result.metadata

        # Not prefetched, load synchronously
        data, metadata = self.reader[index]

        # Auto-transfer to target device if specified
        if self.target_device is not None:
            data = data.to(self.target_device, non_blocking=True)

        # Apply transforms
        if self.transforms is not None:
            data = self.transforms(data)

        return data, metadata

    def cancel_prefetch(self, index: Optional[int] = None) -> None:
        """
        Cancel prefetch requests.

        Note: Already-running tasks will complete, but results are discarded.

        Parameters
        ----------
        index : int, optional
            Specific index to cancel. If None, cancels all.
        """
        if index is None:
            # Cancel all - just clear the dict, let futures complete
            self._prefetch_futures.clear()
        else:
            self._prefetch_futures.pop(index, None)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.reader)

    def __iter__(self) -> Iterator[tuple[TensorDict, dict[str, Any]]]:
        """
        Iterate over all samples.

        Note: This does NOT automatically prefetch. For prefetched iteration,
        use the DataLoader which manages prefetching strategy.

        Yields
        ------
        tuple[TensorDict, dict[str, Any]]
            Tuple of (transformed data, metadata) for each sample.
        """
        for i in range(len(self)):
            yield self[i]

    @property
    def field_names(self) -> list[str]:
        """
        List of field names in samples (from reader).

        Returns
        -------
        list[str]
            Field names available in samples.
        """
        return self.reader.field_names

    @property
    def prefetch_count(self) -> int:
        """
        Number of items currently being prefetched.

        Returns
        -------
        int
            Count of in-flight prefetch operations.
        """
        return len(self._prefetch_futures)

    def close(self) -> None:
        """
        Close the dataset and stop prefetching.

        Waits for any in-flight prefetch tasks to complete before shutdown.
        This prevents "cannot schedule new futures after shutdown" errors
        from libraries like zarr that use async I/O internally.
        """
        # Wait for any in-flight prefetch tasks to complete before shutdown.
        # This prevents "cannot schedule new futures after shutdown" errors
        # from libraries like zarr that use async I/O internally.
        for future in self._prefetch_futures.values():
            try:
                future.result(timeout=30.0)  # Wait up to 30s per task
            except Exception:  # noqa: BLE001, S110
                pass  # Ignore errors during shutdown

        self._prefetch_futures.clear()

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        self.reader.close()

    def __enter__(self) -> "Dataset":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation of the Dataset.
        """
        transform_str = repr(self.transforms) if self.transforms else "None"
        return f"Dataset(\n  reader={self.reader},\n  transforms={transform_str}\n)"
