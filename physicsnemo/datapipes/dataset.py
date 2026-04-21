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
Dataset - Combines a Reader with a transform pipeline.

The Dataset is the primary interface for accessing preprocessed data.
It wraps a Reader and applies transforms to produce ready-to-use TensorDicts.
Supports prefetching with CUDA streams for overlapped IO and computation,
and automatic device transfer when device parameter is specified.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from tensordict import TensorDict

from physicsnemo.datapipes._rng import fork_generator
from physicsnemo.datapipes.protocols import DatasetBase, _PrefetchResult
from physicsnemo.datapipes.readers.base import Reader
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.base import Transform
from physicsnemo.datapipes.transforms.compose import Compose
from physicsnemo.distributed import DistributedManager


@register()
class Dataset(DatasetBase):
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
        super().__init__(num_workers=num_workers)

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

        match device:
            case torch.device():
                self.target_device = device
            case str():
                self.target_device = torch.device(device)
            case None:
                self.target_device = None

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

        if self.target_device is not None and self.transforms is not None:
            self.transforms.to(self.target_device)

    # ------------------------------------------------------------------
    # DatasetBase implementation
    # ------------------------------------------------------------------

    def _load(self, index: int) -> tuple[TensorDict, dict[str, Any]]:
        """Synchronous load: reader → device transfer → transforms."""
        data, metadata = self.reader[index]

        if self.target_device is not None:
            data = data.to(self.target_device, non_blocking=True)

        if self.transforms is not None:
            data = self.transforms(data)

        return data, metadata

    def __len__(self) -> int:
        return len(self.reader)

    # ------------------------------------------------------------------
    # RNG management
    # ------------------------------------------------------------------

    def _flat_transforms(self) -> list[Transform]:
        """Return transforms as a flat list (unwrapping Compose)."""
        if self.transforms is None:
            return []
        if isinstance(self.transforms, Compose):
            return list(self.transforms.transforms)
        return [self.transforms]

    def set_generator(self, generator: torch.Generator) -> None:
        """Distribute forked generators to the reader and every stochastic transform.

        Forks *generator* into ``1 + len(flat_transforms)`` independent
        children: the first goes to the reader, the rest map 1-to-1 to
        the transform list (deterministic transforms silently ignore
        theirs).

        Parameters
        ----------
        generator : torch.Generator
            Parent generator (typically forked from the DataLoader's
            master generator).
        """
        flat = self._flat_transforms()
        n_children = 1 + len(flat)
        children = fork_generator(generator, n_children)

        # Child 0 → reader
        if hasattr(self.reader, "set_generator"):
            self.reader.set_generator(children[0])

        # Children 1..N → transforms (deterministic ones ignore via base no-op)
        for child, t in zip(children[1:], flat):
            if self.target_device is not None and self.target_device != child.device:
                dev_gen = torch.Generator(device=self.target_device)
                dev_gen.manual_seed(child.initial_seed())
                t.set_generator(dev_gen)
            else:
                t.set_generator(child)

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch to the reader and every transform.

        Reseeds all generators assigned via :meth:`set_generator` so
        each epoch produces a different but deterministic random
        sequence.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        if hasattr(self.reader, "set_epoch"):
            self.reader.set_epoch(epoch)

        for t in self._flat_transforms():
            t.set_epoch(epoch)

    # ------------------------------------------------------------------
    # Stream-aware prefetch (overrides DatasetBase defaults)
    # ------------------------------------------------------------------

    def _load_and_transform(
        self,
        index: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> _PrefetchResult:
        """
        Load a sample and apply transforms with optional CUDA stream.

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
            data, metadata = self.reader[index]

            if self.target_device is not None:
                if stream is not None:
                    with torch.cuda.stream(stream):
                        data = data.to(self.target_device, non_blocking=True)
                else:
                    data = data.to(self.target_device, non_blocking=True)

            if self.transforms is not None:
                if stream is not None:
                    with torch.cuda.stream(stream):
                        data = self.transforms(data)
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

        When a CUDA stream is provided, GPU operations (device transfer
        and transforms) run on that stream for overlap with computation.

        Parameters
        ----------
        index : int
            Sample index to prefetch.
        stream : torch.cuda.Stream, optional
            Optional CUDA stream for GPU operations.
        """
        if index in self._prefetch_futures:
            return

        executor = self._ensure_executor()
        future = executor.submit(self._load_and_transform, index, stream)
        self._prefetch_futures[index] = future

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
        future = self._prefetch_futures.pop(index, None)

        if future is not None:
            result = future.result()

            if isinstance(result, _PrefetchResult):
                if result.error is not None:
                    raise result.error
                if result.event is not None:
                    torch.cuda.current_stream().wait_event(result.event)
                return result.data, result.metadata

            return result

        return self._load(index)

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

    def close(self) -> None:
        """
        Close the dataset and stop prefetching.

        Waits for any in-flight prefetch tasks to complete before shutdown.
        This prevents "cannot schedule new futures after shutdown" errors
        from libraries like zarr that use async I/O internally.
        """
        super().close()
        self.reader.close()

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
