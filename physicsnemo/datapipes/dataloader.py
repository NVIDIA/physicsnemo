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
DataLoader - Batched iteration over datasets with prefetching.

The DataLoader orchestrates efficient batch loading by leveraging
the Dataset's prefetching capabilities with CUDA streams.
By default, returns batched TensorDict for PyTorch DataLoader compatibility.
When collate_metadata=True, returns (TensorDict, list[dict]) tuples.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional, Sequence

import torch
from tensordict import TensorDict
from torch.utils.data import RandomSampler, Sampler, SequentialSampler

from physicsnemo.datapipes.collate import Collator, get_collator
from physicsnemo.datapipes.dataset import Dataset
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
        dataset: Dataset,
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
    ) -> None:
        """
        Initialize the DataLoader.

        Parameters
        ----------
        dataset : Dataset
            Dataset to load from.
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
        self.num_streams = num_streams
        self.use_streams = use_streams and torch.cuda.is_available()

        # Handle sampler
        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

        # Handle collation
        self.collate_fn = get_collator(collate_fn, collate_metadata=collate_metadata)

        # Create CUDA streams for prefetching
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
        """
        n_samples = len(self.dataset)
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

        Uses stream-based prefetching when enabled to overlap IO,
        GPU transfers, and computation.

        Yields
        ------
        TensorDict or tuple[TensorDict, list[dict[str, Any]]]
            Batched TensorDict if collate_metadata=False (default),
            or tuple of (batched TensorDict, list of metadata dicts)
            if collate_metadata=True.
        """
        if self.prefetch_factor > 0 and self.use_streams:
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

    def _iter_prefetch(
        self,
    ) -> Iterator[TensorDict | tuple[TensorDict, list[dict[str, Any]]]]:
        """
        Iteration with stream-based prefetching.

        Strategy:

        1. Prefetch `prefetch_factor` batches worth of samples
        2. As we yield batches, prefetch more to keep the pipeline full
        3. Each sample in a batch uses a different stream for overlap

        Yields
        ------
        TensorDict or tuple[TensorDict, list[dict[str, Any]]]
            Collated batch.
        """
        # Collect all batches upfront for prefetch planning
        all_batches = list(self._generate_batches())
        if not all_batches:
            return

        num_prefetch_batches = min(self.prefetch_factor, len(all_batches))
        stream_idx = 0

        # Start initial prefetch
        prefetched_up_to = 0
        for batch_idx in range(num_prefetch_batches):
            for sample_idx in all_batches[batch_idx]:
                stream = self._streams[stream_idx % self.num_streams]
                self.dataset.prefetch(sample_idx, stream=stream)
                stream_idx += 1
            prefetched_up_to = batch_idx + 1

        # Yield batches and prefetch more
        for batch_idx, batch_indices in enumerate(all_batches):
            # Collect samples (uses prefetched if available)
            samples = [self.dataset[idx] for idx in batch_indices]
            batch = self.collate_fn(samples)

            # Prefetch next batch if available
            next_prefetch_idx = prefetched_up_to
            if next_prefetch_idx < len(all_batches):
                for sample_idx in all_batches[next_prefetch_idx]:
                    stream = self._streams[stream_idx % self.num_streams]
                    self.dataset.prefetch(sample_idx, stream=stream)
                    stream_idx += 1
                prefetched_up_to += 1

            yield batch

        # Clean up any remaining prefetch state
        self.dataset.cancel_prefetch()

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for the sampler.

        Required for DistributedSampler to shuffle properly across epochs.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def enable_prefetch(self) -> None:
        """
        Enable stream-based prefetching.

        Raises
        ------
        RuntimeError
            If CUDA is not available.
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available, cannot enable stream prefetching"
            )

        if not self._streams:
            for _ in range(self.num_streams):
                self._streams.append(torch.cuda.Stream())

        self.use_streams = True

    def disable_prefetch(self) -> None:
        """Disable prefetching (useful for debugging)."""
        self.use_streams = False
        self.dataset.cancel_prefetch()

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
