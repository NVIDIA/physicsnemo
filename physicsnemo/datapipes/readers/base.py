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
Reader base class - Abstract interface for data sources.

Readers are simple, transactional data loaders. They load data from sources
and return TensorDict instances with CPU tensors plus separate metadata dicts.
Device transfers and threading are handled elsewhere (Dataset and DataLoader).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Iterator

import torch
from tensordict import TensorDict

logger = logging.getLogger(__name__)


class Reader(ABC):
    """
    Abstract base class for data readers.

    Readers are intentionally simple and transactional:

    - Load data from a source (file, database, etc.)
    - Return (TensorDict, metadata_dict) tuples with CPU tensors
    - No threading, no prefetching, no device transfers

    This design makes custom readers easy to implement. Users only need to:

    1. Implement ``_load_sample(index)`` to load raw data
    2. Implement ``__len__()`` to return dataset size

    Device transfers are handled automatically by Dataset (if device parameter set).
    Threading/prefetching is handled by the DataLoader.

    Examples
    --------
    Custom reader implementation:

    >>> class MyReader(Reader):  # doctest: +SKIP
    ...     def __init__(self, path: str, **kwargs):
    ...         super().__init__(**kwargs)
    ...         self.data = load_my_data(path)
    ...
    ...     def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
    ...         return {"x": torch.from_numpy(self.data[index])}
    ...
    ...     def __len__(self) -> int:
    ...         return len(self.data)

    Subclasses must implement:

    - ``_load_sample(index: int) -> dict[str, torch.Tensor]``
    - ``__len__() -> int``

    Optionally override:

    - ``_get_field_names() -> list[str]``
    - ``_get_sample_metadata(index: int) -> dict[str, Any]``
    - ``close()``
    """

    def __init__(
        self,
        *,
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
        coordinated_subsampling: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the reader.

        Parameters
        ----------
        pin_memory : bool, default=False
            If True, place tensors in pinned (page-locked) memory.
            This enables faster async CPU→GPU transfers later.
            Only use if you plan to move data to GPU.
        include_index_in_metadata : bool, default=True
            If True, include sample index in metadata.
        coordinated_subsampling : dict[str, Any], optional
            Optional dict to configure coordinated subsampling at construction
            time. If provided, must contain:

            - ``n_points``: Number of points to read from each target tensor
            - ``target_keys``: List of tensor keys to apply subsampling to

            This allows configuration via Hydra. Readers that don't support
            coordinated subsampling will ignore this parameter.
        """
        self.pin_memory = pin_memory
        self.include_index_in_metadata = include_index_in_metadata
        self._coordinated_subsampling_config = coordinated_subsampling

    @abstractmethod
    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """
        Load raw data for a single sample.

        This is the main method to implement. Load data from your source
        and return it as a dictionary of CPU tensors.

        Parameters
        ----------
        index : int
            Sample index (0 to len-1).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping field names to CPU tensors.

        Raises
        ------
        IndexError
            If index is out of range.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        raise NotImplementedError

    def _get_field_names(self) -> list[str]:
        """
        Return the list of field names in samples.

        Override this to provide field names without loading a sample.
        Default implementation loads sample 0 and extracts keys.

        Returns
        -------
        list[str]
            List of field names available in samples.
        """
        if len(self) == 0:
            return []
        data = self._load_sample(0)
        return list(data.keys())

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """
        Return metadata for a sample.

        Override this to provide source-specific metadata (filenames, etc.).
        Default implementation returns empty dict (index added separately).

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        dict[str, Any]
            Dictionary of metadata (not tensors).
        """
        return {}

    @property
    def _supports_coordinated_subsampling(self) -> bool:
        """
        Return True if this reader supports coordinated subsampling.

        Override this property in subclasses that implement coordinated
        subsampling.

        Returns
        -------
        bool
            True if coordinated subsampling is supported.
        """
        return False

    @property
    def field_names(self) -> list[str]:
        """
        List of field names available in samples.

        Returns
        -------
        list[str]
            Field names.
        """
        return self._get_field_names()

    def __getitem__(self, index: int) -> tuple[TensorDict, dict[str, Any]]:
        """
        Load and return a single sample.

        Parameters
        ----------
        index : int
            Sample index. Supports negative indexing.

        Returns
        -------
        tuple[TensorDict, dict[str, Any]]
            Tuple of (TensorDict with CPU tensors, metadata dict).

        Raises
        ------
        IndexError
            If index is out of range.
        """
        # Handle negative indexing
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for reader with {len(self)} samples"
            )

        # Load data
        data_dict = self._load_sample(index)

        # Build metadata
        metadata = self._get_sample_metadata(index)
        if self.include_index_in_metadata:
            metadata["index"] = index

        # Pin memory if requested
        if self.pin_memory:
            data_dict = {k: v.pin_memory() for k, v in data_dict.items()}

        # Create TensorDict
        data = TensorDict(data_dict, device=torch.device("cpu"))

        return data, metadata

    def __iter__(self) -> Iterator[tuple[TensorDict, dict[str, Any]]]:
        """
        Iterate over all samples.

        Yields
        ------
        tuple[TensorDict, dict[str, Any]]
            Tuple of (TensorDict with CPU tensors, metadata dict) for each sample.

        Raises
        ------
        RuntimeError
            If a sample fails to load, wrapping the original exception with
            context about which sample failed.
        """
        for i in range(len(self)):
            try:
                yield self[i]
            except Exception as e:
                error_msg = f"Sample {i} failed with exception: {type(e).__name__}: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

    def close(self) -> None:
        """
        Clean up resources (file handles, connections, etc.).

        Override this in subclasses that hold open resources.
        """
        pass

    def __enter__(self) -> "Reader":
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
            String representation of the reader.
        """
        return (
            f"{self.__class__.__name__}(len={len(self)}, pin_memory={self.pin_memory})"
        )
