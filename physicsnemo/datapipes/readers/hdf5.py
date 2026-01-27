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
HDF5Reader - Read data from HDF5 files.

Supports reading from single HDF5 files or directories of HDF5 files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from physicsnemo.datapipes.readers.base import Reader
from physicsnemo.datapipes.registry import register


@register()
class HDF5Reader(Reader):
    """
    Read samples from HDF5 files.

    Supports two modes:

    1. Single file with samples indexed along first dimension of datasets
    2. Directory of HDF5 files, one sample per file

    Examples
    --------
    Single file mode:

    >>> # File structure: data.h5 with datasets "pressure" (N, 100), "velocity" (N, 100, 3)
    >>> reader = HDF5Reader("data.h5", fields=["pressure", "velocity"])  # doctest: +SKIP
    >>> data, metadata = reader[0]  # Returns (TensorDict, dict) tuple  # doctest: +SKIP
    >>> data["pressure"].shape  # torch.Size([100])  # doctest: +SKIP

    Directory mode:

    >>> # Directory with sample_0.h5, sample_1.h5, ...
    >>> reader = HDF5Reader("data_dir/", file_pattern="sample_*.h5")  # doctest: +SKIP
    >>> data, metadata = reader[0]  # Loads all datasets from sample_0.h5  # doctest: +SKIP
    """

    def __init__(
        self,
        path: Path | str,
        *,
        fields: Optional[list[str]] = None,
        file_pattern: str = "*.h5",
        index_key: Optional[str] = None,
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
    ) -> None:
        """
        Initialize the HDF5 reader.

        Parameters
        ----------
        path : Path or str
            Path to HDF5 file or directory containing HDF5 files.
        fields : list[str], optional
            List of dataset names to load. If None, loads all datasets.
        file_pattern : str, default="*.h5"
            Glob pattern for finding files (directory mode only).
        index_key : str, optional
            If provided, use this dataset to determine sample count
            instead of inferring from first dimension.
        pin_memory : bool, default=False
            If True, place tensors in pinned memory for faster GPU transfer.
        include_index_in_metadata : bool, default=True
            If True, include sample index in metadata.

        Raises
        ------
        ImportError
            If h5py is not installed.
        FileNotFoundError
            If path doesn't exist.
        ValueError
            If no HDF5 files found in directory.
        """
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required for HDF5Reader. Install with: pip install h5py"
            )

        super().__init__(
            pin_memory=pin_memory,
            include_index_in_metadata=include_index_in_metadata,
        )

        self.path = Path(path)
        self.fields = fields
        self.file_pattern = file_pattern
        self.index_key = index_key

        if not self.path.exists():
            raise FileNotFoundError(f"Path not found: {self.path}")

        # Determine mode: single file or directory
        self._is_directory = self.path.is_dir()

        if self._is_directory:
            # Directory mode: each file is a sample
            self._files = sorted(self.path.glob(file_pattern))
            if not self._files:
                raise ValueError(
                    f"No files matching '{file_pattern}' found in {self.path}"
                )
            self._length = len(self._files)
            self._h5_file = None

            # Discover fields from first file
            if self.fields is None:
                with h5py.File(self._files[0], "r") as f:
                    self.fields = [
                        k for k in f.keys() if isinstance(f[k], h5py.Dataset)
                    ]
        else:
            # Single file mode: samples indexed along first dimension
            self._files = None
            self._h5_file = h5py.File(self.path, "r")

            # Discover fields
            if self.fields is None:
                self.fields = [
                    k
                    for k in self._h5_file.keys()
                    if isinstance(self._h5_file[k], h5py.Dataset)
                ]

            # Determine length
            if self.index_key is not None:
                self._length = self._h5_file[self.index_key].shape[0]
            elif self.fields:
                self._length = self._h5_file[self.fields[0]].shape[0]
            else:
                self._length = 0

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """
        Load a single sample from HDF5.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping field names to tensors.

        Raises
        ------
        KeyError
            If a requested field is not found in the file.
        """
        data = {}

        if self._is_directory:
            # Directory mode: load all datasets from the file
            file_path = self._files[index]
            with h5py.File(file_path, "r") as f:
                for field in self.fields:
                    if field not in f:
                        raise KeyError(
                            f"Field '{field}' not found in {file_path}. "
                            f"Available: {list(f.keys())}"
                        )
                    arr = f[field][:]
                    data[field] = torch.from_numpy(arr)
        else:
            # Single file mode: index into datasets
            for field in self.fields:
                if field not in self._h5_file:
                    raise KeyError(
                        f"Field '{field}' not found in {self.path}. "
                        f"Available: {list(self._h5_file.keys())}"
                    )
                arr = self._h5_file[field][index]
                data[field] = torch.from_numpy(arr)

        return data

    def __len__(self) -> int:
        """
        Return number of samples.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self._length

    def _get_field_names(self) -> list[str]:
        """
        Return field names.

        Returns
        -------
        list[str]
            List of field names available in samples.
        """
        return self.fields if self.fields else []

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """
        Return metadata for a sample including source file info.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        dict[str, Any]
            Metadata dictionary with source file information.
        """
        if self._is_directory:
            return {
                "source_file": str(self._files[index]),
                "source_filename": self._files[index].name,
            }
        else:
            return {
                "source_file": str(self.path),
                "source_filename": self.path.name,
            }

    def close(self) -> None:
        """Close HDF5 file handle."""
        super().close()
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation of the reader.
        """
        mode = "directory" if self._is_directory else "file"
        return (
            f"HDF5Reader("
            f"path={self.path}, "
            f"mode={mode}, "
            f"len={len(self)}, "
            f"fields={self.fields})"
        )
