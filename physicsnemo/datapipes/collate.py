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
Collation utilities - Batch multiple (TensorDict, metadata) tuples.

Collators combine multiple (TensorDict, dict) tuples from Dataset into a single
batched output suitable for model consumption. By default, returns just the
batched TensorDict for PyTorch DataLoader compatibility. When collate_metadata=True,
returns a tuple of (TensorDict, list[dict]).

The default collator stacks TensorDicts along batch dimension using TensorDict.stack().
Metadata collation is optional and disabled by default.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence

import torch
from tensordict import TensorDict


def _collate_metadata(metadata_list: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Collate metadata from multiple samples.

    Simply returns the list of metadata dicts as-is. Each metadata dict
    corresponds to one sample in the batch.

    Parameters
    ----------
    metadata_list : Sequence[dict[str, Any]]
        Sequence of metadata dicts.

    Returns
    -------
    list[dict[str, Any]]
        List of metadata dicts.
    """
    return list(metadata_list)


class Collator(ABC):
    """
    Abstract base class for collators.

    Collators take a sequence of (TensorDict, dict) tuples and combine them
    into a batched output. By default, returns just the batched TensorDict
    for PyTorch DataLoader compatibility. When collate_metadata=True, returns
    a tuple of (TensorDict, list[dict]).

    Examples
    --------
    >>> class MyCollator(Collator):
    ...     def __call__(
    ...         self,
    ...         samples: Sequence[tuple[TensorDict, dict]]
    ...     ) -> TensorDict:
    ...         # Custom batching logic
    ...         ...
    """

    @abstractmethod
    def __call__(
        self, samples: Sequence[tuple[TensorDict, dict[str, Any]]]
    ) -> TensorDict | tuple[TensorDict, list[dict[str, Any]]]:
        """
        Collate a batch of samples.

        Parameters
        ----------
        samples : Sequence[tuple[TensorDict, dict[str, Any]]]
            Sequence of (TensorDict, metadata dict) tuples to batch.

        Returns
        -------
        TensorDict or tuple[TensorDict, list[dict[str, Any]]]
            Batched TensorDict, or tuple of (batched TensorDict, list of metadata dicts)
            if collate_metadata=True.
        """
        raise NotImplementedError


class DefaultCollator(Collator):
    """
    Default collator that stacks TensorDicts along a new batch dimension.

    Uses TensorDict.stack() to efficiently batch all tensors, creating
    shape [batch_size, ...original_shape] for each field.

    All samples must have:

    - The same tensor keys
    - Tensors with matching shapes (per key)
    - Tensors on the same device

    By default, returns just the batched TensorDict for PyTorch DataLoader
    compatibility. Set collate_metadata=True to also return metadata.

    Examples
    --------
    >>> data1 = TensorDict({"x": torch.randn(10, 3)}, device="cpu")
    >>> data2 = TensorDict({"x": torch.randn(10, 3)}, device="cpu")
    >>> samples = [
    ...     (data1, {"file": "a.h5"}),
    ...     (data2, {"file": "b.h5"}),
    ... ]
    >>> collator = DefaultCollator()
    >>> batched_data = collator(samples)
    >>> batched_data["x"].shape
    torch.Size([2, 10, 3])

    With metadata collation enabled:

    >>> collator = DefaultCollator(collate_metadata=True)
    >>> batched_data, metadata_list = collator(samples)
    >>> metadata_list
    [{'file': 'a.h5'}, {'file': 'b.h5'}]
    """

    def __init__(
        self,
        *,
        stack_dim: int = 0,
        keys: Optional[list[str]] = None,
        collate_metadata: bool = False,
    ) -> None:
        """
        Initialize the collator.

        Parameters
        ----------
        stack_dim : int, default=0
            Dimension along which to stack tensors.
        keys : list[str], optional
            If provided, only collate these tensor keys. Others are ignored.
        collate_metadata : bool, default=False
            If True, collate metadata into list. Default is False for
            compatibility with PyTorch DataLoader.
        """
        self.stack_dim = stack_dim
        self.keys = keys
        self.collate_metadata = collate_metadata

    def __call__(
        self, samples: Sequence[tuple[TensorDict, dict[str, Any]]]
    ) -> TensorDict | tuple[TensorDict, list[dict[str, Any]]]:
        """
        Collate samples by stacking TensorDicts.

        Parameters
        ----------
        samples : Sequence[tuple[TensorDict, dict[str, Any]]]
            Sequence of (TensorDict, metadata) tuples to batch.

        Returns
        -------
        TensorDict or tuple[TensorDict, list[dict[str, Any]]]
            Batched TensorDict if collate_metadata=False (default),
            or tuple of (batched TensorDict, list of metadata dicts)
            if collate_metadata=True.

        Raises
        ------
        ValueError
            If samples is empty or samples have mismatched keys/shapes.
        """
        if not samples:
            raise ValueError("Cannot collate empty sequence of samples")

        # Separate data and metadata
        data_list = [data for data, _ in samples]

        # Use TensorDict.stack() for efficient batching
        if self.keys is not None:
            # Filter to only requested keys
            data_list = [data.select(*self.keys) for data in data_list]

        batched_data = torch.stack(data_list, dim=self.stack_dim)

        # Collate metadata only if requested
        if self.collate_metadata:
            metadata_list = [meta for _, meta in samples]
            return batched_data, _collate_metadata(metadata_list)

        return batched_data


class ConcatCollator(Collator):
    """
    Collator that concatenates tensors along an existing dimension.

    Unlike DefaultCollator which creates a new batch dimension, this
    concatenates along an existing dimension. Useful for point clouds
    or other variable-length data where you want to combine all points.

    Optionally adds batch indices to track which points came from which sample.
    By default, returns just the batched TensorDict for PyTorch DataLoader
    compatibility. Set collate_metadata=True to also return metadata.

    Examples
    --------
    >>> data1 = TensorDict({"points": torch.randn(100, 3)})
    >>> data2 = TensorDict({"points": torch.randn(150, 3)})
    >>> samples = [
    ...     (data1, {"file": "a.h5"}),
    ...     (data2, {"file": "b.h5"}),
    ... ]
    >>> collator = ConcatCollator(dim=0, add_batch_idx=True)
    >>> batched_data = collator(samples)
    >>> batched_data["points"].shape
    torch.Size([250, 3])
    >>> batched_data["batch_idx"].shape
    torch.Size([250])

    With metadata collation enabled:

    >>> collator = ConcatCollator(dim=0, add_batch_idx=True, collate_metadata=True)
    >>> batched_data, metadata_list = collator(samples)
    >>> metadata_list
    [{'file': 'a.h5'}, {'file': 'b.h5'}]
    """

    def __init__(
        self,
        *,
        dim: int = 0,
        add_batch_idx: bool = True,
        batch_idx_key: str = "batch_idx",
        keys: Optional[list[str]] = None,
        collate_metadata: bool = False,
    ) -> None:
        """
        Initialize the collator.

        Parameters
        ----------
        dim : int, default=0
            Dimension along which to concatenate.
        add_batch_idx : bool, default=True
            If True, add a tensor of batch indices.
        batch_idx_key : str, default="batch_idx"
            Key for the batch index tensor.
        keys : list[str], optional
            If provided, only collate these tensor keys.
        collate_metadata : bool, default=False
            If True, collate metadata into lists. Default is False for
            compatibility with PyTorch DataLoader.
        """
        self.dim = dim
        self.add_batch_idx = add_batch_idx
        self.batch_idx_key = batch_idx_key
        self.keys = keys
        self.collate_metadata = collate_metadata

    def __call__(
        self, samples: Sequence[tuple[TensorDict, dict[str, Any]]]
    ) -> TensorDict | tuple[TensorDict, list[dict[str, Any]]]:
        """
        Collate samples by concatenating tensors.

        Parameters
        ----------
        samples : Sequence[tuple[TensorDict, dict[str, Any]]]
            Sequence of (TensorDict, metadata) tuples to batch.

        Returns
        -------
        TensorDict or tuple[TensorDict, list[dict[str, Any]]]
            Batched TensorDict if collate_metadata=False (default),
            or tuple of (batched TensorDict, list of metadata dicts)
            if collate_metadata=True.

        Raises
        ------
        ValueError
            If samples is empty.
        """
        if not samples:
            raise ValueError("Cannot collate empty sequence of samples")

        # Separate data
        data_list = [data for data, _ in samples]

        first_data = data_list[0]
        keys = self.keys if self.keys else list(first_data.keys())
        device = first_data.device

        batched_tensors = {}
        sizes = []  # Track sizes for batch indices

        for key in keys:
            tensors = []
            for data in data_list:
                if key not in data.keys():
                    raise ValueError(f"Data missing key '{key}'")
                tensor = data[key]
                tensors.append(tensor)
                if key == keys[0]:  # Track sizes from first key
                    sizes.append(tensor.shape[self.dim])

            batched_tensors[key] = torch.cat(tensors, dim=self.dim)

        # Add batch indices
        if self.add_batch_idx:
            batch_indices = []
            for i, size in enumerate(sizes):
                batch_indices.append(
                    torch.full((size,), i, dtype=torch.long, device=device)
                )
            batched_tensors[self.batch_idx_key] = torch.cat(batch_indices, dim=0)

        # Create batched TensorDict
        batched_data = TensorDict(batched_tensors, device=device)

        # Collate metadata only if requested
        if self.collate_metadata:
            metadata_list = [meta for _, meta in samples]
            return batched_data, _collate_metadata(metadata_list)

        return batched_data


class FunctionCollator(Collator):
    """
    Collator that wraps a user-provided function.

    Allows using any function as a collator without subclassing.

    Examples
    --------
    >>> def my_collate(samples):
    ...     # Custom logic
    ...     data_list = [d for d, _ in samples]
    ...     metadata_list = [m for _, m in samples]
    ...     return torch.stack(data_list), metadata_list
    >>> collator = FunctionCollator(my_collate)
    """

    def __init__(
        self,
        fn: Callable[
            [Sequence[tuple[TensorDict, dict[str, Any]]]],
            tuple[TensorDict, list[dict[str, Any]]],
        ],
    ) -> None:
        """
        Initialize with a collation function.

        Parameters
        ----------
        fn : Callable
            Function that takes a sequence of (TensorDict, dict) tuples
            and returns a (TensorDict, list[dict]) tuple.
        """
        self.fn = fn

    def __call__(
        self, samples: Sequence[tuple[TensorDict, dict[str, Any]]]
    ) -> TensorDict | tuple[TensorDict, list[dict[str, Any]]]:
        """Apply the wrapped function."""
        return self.fn(samples)


# Default collator instance
_default_collator = DefaultCollator()


def default_collate(
    samples: Sequence[tuple[TensorDict, dict[str, Any]]],
) -> tuple[TensorDict, list[dict[str, Any]]]:
    """
    Default collation function using stacking.

    Convenience function that uses DefaultCollator.
    Metadata is collated into a list of dicts.

    Parameters
    ----------
    samples : Sequence[tuple[TensorDict, dict[str, Any]]]
        Sequence of (TensorDict, metadata) tuples to batch.

    Returns
    -------
    tuple[TensorDict, list[dict[str, Any]]]
        Tuple of (batched TensorDict, list of metadata dicts).
    """
    return _default_collator(samples)


def concat_collate(
    samples: Sequence[tuple[TensorDict, dict[str, Any]]],
    dim: int = 0,
    add_batch_idx: bool = True,
) -> tuple[TensorDict, list[dict[str, Any]]]:
    """
    Collation function using concatenation.

    Convenience function that uses ConcatCollator.
    Metadata is collated into a list of dicts.

    Parameters
    ----------
    samples : Sequence[tuple[TensorDict, dict[str, Any]]]
        Sequence of (TensorDict, metadata) tuples to batch.
    dim : int, default=0
        Dimension along which to concatenate.
    add_batch_idx : bool, default=True
        If True, add batch index tensor.

    Returns
    -------
    tuple[TensorDict, list[dict[str, Any]]]
        Tuple of (batched TensorDict, list of metadata dicts).
    """
    collator = ConcatCollator(dim=dim, add_batch_idx=add_batch_idx)
    return collator(samples)


def get_collator(
    collate_fn: Collator
    | Callable[
        [Sequence[tuple[TensorDict, dict[str, Any]]]],
        tuple[TensorDict, list[dict[str, Any]]],
    ]
    | None = None,
    *,
    collate_metadata: bool = False,
) -> Collator:
    """
    Get a Collator instance from various input types.

    Parameters
    ----------
    collate_fn : Collator or Callable, optional
        Collator, callable, or None (uses default).
    collate_metadata : bool, default=False
        If True, collate metadata into list. Only used when collate_fn is None.
        Default is False for compatibility with PyTorch DataLoader.

    Returns
    -------
    Collator
        Collator instance.

    Raises
    ------
    TypeError
        If collate_fn is not a Collator, callable, or None.
    """
    if collate_fn is None:
        return DefaultCollator(collate_metadata=collate_metadata)
    elif isinstance(collate_fn, Collator):
        return collate_fn
    elif callable(collate_fn):
        return FunctionCollator(collate_fn)
    else:
        raise TypeError(
            f"collate_fn must be Collator, callable, or None, "
            f"got {type(collate_fn).__name__}"
        )
