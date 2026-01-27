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
NumpyReader - Read data from NumPy .npz files.

Supports reading from single .npz files or directories of .npz files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from physicsnemo.datapipes.readers.base import Reader
from physicsnemo.datapipes.registry import register


@register()
class NumpyReader(Reader):
    """
    Read samples from NumPy .npz files.

    Supports two modes:
    1. Single .npz file: samples indexed along first dimension of each array
    2. Directory of .npz files: one sample per file

    Example (single .npz):
        >>> # data.npz with arrays "positions" (N, 100, 3), "features" (N, 100)
        >>> reader = NumpyReader("data.npz", fields=["positions", "features"])  # doctest: +SKIP
        >>> data, metadata = reader[0]  # Returns (TensorDict, dict) tuple  # doctest: +SKIP
        >>> # Or load all arrays:
        >>> reader = NumpyReader("data.npz")  # fields=None loads all  # doctest: +SKIP

    Example (directory):
        >>> # Directory with sample_0.npz, sample_1.npz, ...
        >>> reader = NumpyReader("data_dir/", file_pattern="sample_*.npz")  # doctest: +SKIP
        >>> data, metadata = reader[0]  # Returns (TensorDict, dict) tuple  # doctest: +SKIP
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fields: Optional[list[str]] = None,
        default_values: Optional[dict[str, torch.Tensor]] = None,
        file_pattern: str = "*.npz",
        index_key: Optional[str] = None,
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
        coordinated_subsampling: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the NumPy reader.

        Parameters
        ----------
        path : str or Path
            Path to .npz file or directory of .npz files.
        fields : list[str], optional
            List of array names to load. If None, loads all available
            arrays from the file.
        default_values : dict[str, torch.Tensor], optional
            Dictionary mapping field names to default tensors.
            If a field in ``fields`` is not found in the file but has an
            entry here, the default tensor is used instead of raising an
            error. Useful for optional fields.
        file_pattern : str, default="*.npz"
            Glob pattern for finding files (directory mode).
        index_key : str, optional
            If provided, use this array to determine sample count.
        pin_memory : bool, default=False
            If True, place tensors in pinned memory for faster GPU transfer.
        include_index_in_metadata : bool, default=True
            If True, include sample index in metadata.
        coordinated_subsampling : dict[str, Any], optional
            Optional dict to configure coordinated subsampling (directory mode
            only). If provided, must contain ``n_points`` (int) and
            ``target_keys`` (list of str).

        Raises
        ------
        FileNotFoundError
            If path doesn't exist.
        ValueError
            If no files found in directory or unsupported file type.
        """
        super().__init__(
            pin_memory=pin_memory,
            include_index_in_metadata=include_index_in_metadata,
            coordinated_subsampling=coordinated_subsampling,
        )

        self.path = Path(path).expanduser().resolve()
        self._user_fields = fields
        self.default_values = default_values or {}
        self.file_pattern = file_pattern
        self.index_key = index_key

        if not self.path.exists():
            raise FileNotFoundError(f"Path not found: {self.path}")

        # Determine mode based on path
        self._mode: str  # "single" or "directory"
        self._files: Optional[list[Path]] = None
        self._data: Optional[np.lib.npyio.NpzFile] = None
        self._available_fields: list[str] = []

        if self.path.is_dir():
            self._setup_directory_mode()
        elif self.path.suffix == ".npz":
            self._setup_single_file_mode()
        else:
            raise ValueError(
                f"Unsupported file type: {self.path.suffix}. "
                f"Expected .npz file or directory of .npz files."
            )

    def _setup_directory_mode(self) -> None:
        """Set up reader for directory of .npz files."""
        self._mode = "directory"
        self._files = sorted(self.path.glob(self.file_pattern))
        if not self._files:
            raise ValueError(
                f"No files matching '{self.file_pattern}' found in {self.path}"
            )
        self._length = len(self._files)

        # Discover available fields from first file
        with np.load(self._files[0]) as npz:
            self._available_fields = list(npz.files)

    def _setup_single_file_mode(self) -> None:
        """Set up reader for single .npz file."""
        self._mode = "single"
        self._data = np.load(self.path)
        self._available_fields = list(self._data.files)

        # Determine length from index_key or first field
        if self.index_key is not None:
            self._length = self._data[self.index_key].shape[0]
        elif self._available_fields:
            self._length = self._data[self._available_fields[0]].shape[0]
        else:
            self._length = 0

    @property
    def fields(self) -> list[str]:
        """Fields that will be loaded (user-specified or all available)."""
        if self._user_fields is not None:
            return self._user_fields
        return self._available_fields

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

    def _load_from_npz(
        self,
        npz: np.lib.npyio.NpzFile,
        index: Optional[int] = None,
        file_path: Optional[Path] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Load data from an npz file.

        Parameters
        ----------
        npz : np.lib.npyio.NpzFile
            The loaded npz file object.
        index : int, optional
            Sample index to load (for single file mode with indexed arrays).
            None for directory mode (load entire arrays).
        file_path : Path, optional
            Path to the file (for error messages).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping field names to tensors.
        """
        data = {}
        fields_to_load = self.fields

        # Check for missing required fields
        required_fields = set(fields_to_load) - set(self.default_values.keys())
        missing_fields = required_fields - set(npz.files)
        if missing_fields:
            path_str = str(file_path) if file_path else str(self.path)
            raise KeyError(
                f"Required fields {missing_fields} not found in {path_str}. "
                f"Available: {list(npz.files)}"
            )

        # Determine subsample slice if coordinated subsampling is enabled
        subsample_slice = None
        target_keys_set = set()
        if self._coordinated_subsampling_config is not None:
            n_points = self._coordinated_subsampling_config["n_points"]
            target_keys_set = set(self._coordinated_subsampling_config["target_keys"])

            # Find slice from first available target key
            for field in target_keys_set:
                if field in npz.files:
                    array_shape = npz[field].shape[0]
                    subsample_slice = self._select_random_sections_from_slice(
                        0, array_shape, n_points
                    )
                    break

        # Load each field
        for field in fields_to_load:
            if field in npz.files:
                arr = npz[field]

                # Apply indexing if provided (single file mode)
                if index is not None:
                    arr = arr[index]

                # Apply subsampling if this field is a target
                if subsample_slice is not None and field in target_keys_set:
                    arr = arr[subsample_slice]
                elif index is None:
                    # Directory mode: load full array
                    arr = arr[:]

                data[field] = torch.from_numpy(np.array(arr))

            elif field in self.default_values:
                data[field] = self.default_values[field].clone()

        return data

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Load a single sample."""
        if self._mode == "directory":
            file_path = self._files[index]
            with np.load(file_path) as npz:
                return self._load_from_npz(npz, index=None, file_path=file_path)
        else:  # single
            return self._load_from_npz(self._data, index=index)

    def __len__(self) -> int:
        """Return number of samples."""
        return self._length

    def _get_field_names(self) -> list[str]:
        """Return field names that will be loaded."""
        return self.fields

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return metadata for a sample including source file info."""
        if self._mode == "directory":
            return {
                "source_file": str(self._files[index]),
                "source_filename": self._files[index].name,
            }
        else:
            return {
                "source_file": str(self.path),
                "source_filename": self.path.name,
            }

    @property
    def _supports_coordinated_subsampling(self) -> bool:
        """NumPy reader supports coordinated subsampling in directory mode."""
        return self._mode == "directory"

    def close(self) -> None:
        """Close file handles."""
        super().close()
        if self._data is not None:
            if hasattr(self._data, "close"):
                self._data.close()
            self._data = None

    def __repr__(self) -> str:
        subsample_info = ""
        if self._coordinated_subsampling_config is not None:
            cfg = self._coordinated_subsampling_config
            subsample_info = f", subsampling={cfg['n_points']} points"

        return (
            f"NumpyReader("
            f"path={self.path}, "
            f"mode={self._mode}, "
            f"len={len(self)}, "
            f"fields={self.fields}"
            f"{subsample_info})"
        )
