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
ZarrReader - Read data from Zarr arrays.

Supports reading from a directory of Zarr groups, one sample per group.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

from physicsnemo.datapipes.readers.base import Reader
from physicsnemo.datapipes.registry import register


@register()
class ZarrReader(Reader):
    """
    Read samples from Zarr groups.

    Zarr is a chunked, compressed array format ideal for large scientific datasets.
    Each Zarr group in the directory represents one sample. Supports loading both
    arrays and attributes from Zarr groups.

    Examples
    --------
    Basic usage:

    >>> # Directory with sample_0.zarr, sample_1.zarr, ...
    >>> # Each contains arrays like "positions", "features", etc.
    >>> reader = ZarrReader("data_dir/", group_pattern="sample_*.zarr")  # doctest: +SKIP
    >>> data, metadata = reader[0]  # Returns (TensorDict, dict) tuple  # doctest: +SKIP

    Load only specific fields:

    >>> reader = ZarrReader("data_dir/", fields=["positions", "velocity"])  # doctest: +SKIP
    >>> data, metadata = reader[0]  # doctest: +SKIP

    Load attributes from Zarr groups:

    >>> # If the Zarr group has attributes like "timestep" or "scale_factor",
    >>> # you can request them as fields:
    >>> reader = ZarrReader("data_dir/", fields=["positions", "timestep", "scale_factor"])  # doctest: +SKIP
    >>> data, metadata = reader[0]  # data["timestep"] contains the attribute value  # doctest: +SKIP

    With coordinated subsampling for large arrays:

    >>> reader = ZarrReader(  # doctest: +SKIP
    ...     "data_dir/",
    ...     coordinated_subsampling={
    ...         "n_points": 50000,
    ...         "target_keys": ["volume_coords", "volume_fields"],
    ...     }
    ... )
    >>> data, metadata = reader[0]  # doctest: +SKIP
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fields: Optional[list[str]] = None,
        default_values: Optional[dict[str, torch.Tensor]] = None,
        group_pattern: str = "*.zarr",
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
        coordinated_subsampling: Optional[dict[str, Any]] = None,
        cache_stores: bool = True,
    ) -> None:
        """
        Initialize the Zarr reader.

        Parameters
        ----------
        path : str or Path
            Path to directory containing Zarr groups.
        fields : list[str], optional
            List of array or attribute names to load. If None, loads all
            available arrays from each group. When a field name matches an
            attribute key (and not an array), the attribute value will be
            converted to a tensor. Note: string attributes are not supported.
        default_values : dict[str, torch.Tensor], optional
            Dictionary mapping field names to default tensors.
            If a field in ``fields`` is not found in the file but has an
            entry here, the default tensor is used instead of raising an
            error. Useful for optional fields.
        group_pattern : str, default="*.zarr"
            Glob pattern for finding Zarr groups.
        pin_memory : bool, default=False
            If True, place tensors in pinned memory for faster GPU transfer.
        include_index_in_metadata : bool, default=True
            If True, include sample index in metadata.
        coordinated_subsampling : dict[str, Any], optional
            Optional dict to configure coordinated subsampling. If provided,
            must contain ``n_points`` (int) and ``target_keys`` (list of str).
        cache_stores : bool, default=True
            If True, cache opened zarr stores to avoid repeated opening and
            prevent executor shutdown errors. Set to False if memory is a
            concern with many groups.

        Raises
        ------
        ImportError
            If zarr is not installed.
        FileNotFoundError
            If path doesn't exist.
        ValueError
            If no Zarr groups found in directory.
        """
        if not HAS_ZARR:
            raise ImportError(
                "zarr is required for ZarrReader. Install with: pip install zarr"
            )

        super().__init__(
            pin_memory=pin_memory,
            include_index_in_metadata=include_index_in_metadata,
            coordinated_subsampling=coordinated_subsampling,
        )

        self.path = Path(path).expanduser().resolve()
        self._user_fields = fields
        self.default_values = default_values or {}
        self.group_pattern = group_pattern
        self._cache_stores = cache_stores
        self._cached_stores: dict[Path, Any] = {}  # Cache for opened zarr stores

        if not self.path.exists():
            raise FileNotFoundError(f"Path not found: {self.path}")

        if not self.path.is_dir():
            raise ValueError(
                f"Path must be a directory containing Zarr groups: {self.path}"
            )

        # Detect mode: single group or directory of groups
        self._single_group_mode = self._is_zarr_group(self.path)

        if self._single_group_mode:
            # Single Zarr group - samples indexed along first dimension
            self._groups = [self.path]
            root = zarr.open(self.path, mode="r")

            if isinstance(root, zarr.Array):
                raise ValueError(
                    f"Expected Zarr group with named arrays, got single array at "
                    f"{self.path}. Path should be a Zarr group containing named arrays."
                )

            self._available_fields = list(root.array_keys())

            # Get length from first array's first dimension
            if not self._available_fields:
                raise ValueError(f"Zarr group {self.path} contains no arrays")

            first_array = root[self._available_fields[0]]
            self._length = first_array.shape[0]
        else:
            # Directory containing multiple Zarr groups
            self._groups = sorted(
                [p for p in self.path.glob(group_pattern) if self._is_zarr_group(p)]
            )

            if not self._groups:
                raise ValueError(
                    f"No Zarr groups matching '{group_pattern}' found in {self.path}"
                )

            self._length = len(self._groups)

            # Discover available fields from first group
            root = zarr.open(self._groups[0], mode="r")
            if isinstance(root, zarr.Array):
                raise ValueError(
                    f"Expected Zarr group with named arrays, got single array at "
                    f"{self._groups[0]}. Each sample should be a Zarr group containing "
                    f"named arrays."
                )
            self._available_fields = list(root.array_keys())

    @property
    def fields(self) -> list[str]:
        """Fields that will be loaded (user-specified or all available)."""
        if self._user_fields is not None:
            return self._user_fields
        return self._available_fields

    def _open_zarr_store(self, path: Path) -> Any:
        """
        Open a zarr store, using cache if enabled.

        This prevents the "cannot schedule new futures after shutdown" error
        by reusing opened stores instead of repeatedly calling zarr.open().

        Parameters
        ----------
        path : Path
            Path to the zarr group.

        Returns
        -------
        Any
            Opened zarr group.
        """
        if self._cache_stores:
            if path not in self._cached_stores:
                self._cached_stores[path] = zarr.open(path, mode="r")
            return self._cached_stores[path]
        else:
            return zarr.open(path, mode="r")

    def _is_zarr_group(self, path: Path) -> bool:
        """
        Check if a path is a Zarr group.

        A Zarr group is identified by the presence of a zarr.json file (v3)
        or .zgroup file (v2).
        """
        return (path / "zarr.json").exists() or (path / ".zgroup").exists()

    def _select_random_sections_from_slice(
        self,
        slice_start: int,
        slice_stop: int,
        n_points: int,
    ) -> slice:
        """
        Select a random contiguous slice from a range.

        Parameters
        ----------
        slice_start : int
            Start index of the available range.
        slice_stop : int
            Stop index of the available range (exclusive).
        n_points : int
            Number of points to sample.

        Returns
        -------
        slice
            A slice object representing the random contiguous section.

        Raises
        ------
        ValueError
            If the range is smaller than n_points.
        """
        total_points = slice_stop - slice_start

        if total_points < n_points:
            raise ValueError(
                f"Slice size {total_points} is less than the number of points "
                f"{n_points} requested for subsampling"
            )

        start = np.random.randint(slice_start, slice_stop - n_points + 1)
        return slice(start, start + n_points)

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Load a single sample from a Zarr group."""
        if self._single_group_mode:
            # Single group: index into first dimension of each array
            group_path = self._groups[0]
            root = self._open_zarr_store(group_path)
        else:
            # Directory mode: each group is one sample
            group_path = self._groups[index]
            root = self._open_zarr_store(group_path)

        data = {}
        fields_to_load = self.fields

        # Discover available arrays and attributes for this sample at runtime
        available_arrays = set(root.array_keys())
        available_attrs = set(root.attrs.keys()) if hasattr(root, "attrs") else set()
        available = available_arrays | available_attrs

        # Check for missing required fields (check both arrays and attributes)
        required_fields = set(fields_to_load) - set(self.default_values.keys())
        missing_fields = required_fields - available
        if missing_fields:
            raise KeyError(
                f"Required fields {missing_fields} not found in {group_path}. "
                f"Available arrays: {list(available_arrays)}, "
                f"Available attributes: {list(available_attrs)}"
            )

        # Determine subsample slice if coordinated subsampling is enabled
        subsample_slice = None
        target_keys_set = set()
        if self._coordinated_subsampling_config is not None:
            n_points = self._coordinated_subsampling_config["n_points"]
            target_keys_set = set(self._coordinated_subsampling_config["target_keys"])

            # Find slice from first available target key
            for field in target_keys_set:
                if field in root:
                    if self._single_group_mode:
                        # In single group mode, subsample along dimensions after the first
                        array_shape = root[field].shape[1]
                    else:
                        array_shape = root[field].shape[0]
                    subsample_slice = self._select_random_sections_from_slice(
                        0, array_shape, n_points
                    )
                    break

        # Load each field
        for field in fields_to_load:
            if field in root:
                if self._single_group_mode:
                    # Single group mode: index into first dimension
                    if subsample_slice is not None and field in target_keys_set:
                        # Apply subsampling on dimensions after the first
                        data[field] = torch.from_numpy(
                            root[field][index, subsample_slice]
                        )
                    else:
                        data[field] = torch.from_numpy(root[field][index])
                else:
                    # Directory mode: load entire array or subsample
                    if subsample_slice is not None and field in target_keys_set:
                        data[field] = torch.from_numpy(root[field][subsample_slice])
                    else:
                        data[field] = torch.from_numpy(root[field][:])

            elif field in available_attrs:
                # Load from attributes (discovered at runtime for this sample)
                attr_value = root.attrs[field]
                data[field] = self._convert_attr_to_tensor(attr_value, field)

            elif field in self.default_values:
                data[field] = self.default_values[field].clone()

        return data

    def _convert_attr_to_tensor(self, value: Any, field_name: str) -> torch.Tensor:
        """
        Convert an attribute value to a torch.Tensor.

        Parameters
        ----------
        value : Any
            The attribute value to convert.
        field_name : str
            Name of the field (for error messages).

        Returns
        -------
        torch.Tensor
            A tensor containing the attribute value.

        Raises
        ------
        TypeError
            If the attribute value cannot be converted to a tensor.
        """
        try:
            match value:
                case np.ndarray():
                    return torch.from_numpy(value)
                case list() | tuple():
                    return torch.tensor(value)
                case int() | float() | bool():
                    return torch.tensor(value)
                case str():
                    raise TypeError(
                        f"Cannot convert string attribute '{field_name}' to tensor. "
                        f"String attributes are not supported."
                    )
                case _:
                    # Try to convert via numpy
                    return torch.from_numpy(np.asarray(value))
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Cannot convert attribute '{field_name}' of type {type(value).__name__} "
                f"to tensor: {e}"
            ) from e

    def __len__(self) -> int:
        """Return number of samples."""
        return self._length

    def _get_field_names(self) -> list[str]:
        """Return field names that will be loaded."""
        return self.fields

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return metadata for a sample including source info."""
        if self._single_group_mode:
            return {
                "source_file": str(self._groups[0]),
                "source_filename": self._groups[0].name,
                "sample_index": index,
            }
        else:
            return {
                "source_file": str(self._groups[index]),
                "source_filename": self._groups[index].name,
            }

    @property
    def _supports_coordinated_subsampling(self) -> bool:
        """Zarr reader supports coordinated subsampling."""
        return True

    def close(self) -> None:
        """Close resources and cached zarr stores."""
        # Clear cached stores to allow garbage collection
        # This helps prevent executor shutdown issues
        self._cached_stores.clear()
        super().close()

    def __repr__(self) -> str:
        subsample_info = ""
        if self._coordinated_subsampling_config is not None:
            cfg = self._coordinated_subsampling_config
            subsample_info = f", subsampling={cfg['n_points']} points"

        return (
            f"ZarrReader("
            f"path={self.path}, "
            f"len={len(self)}, "
            f"fields={self.fields}, "
            f"cache_stores={self._cache_stores}"
            f"{subsample_info})"
        )
