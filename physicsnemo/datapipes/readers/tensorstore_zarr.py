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
TensorStoreZarrReader - High-performance async reader for Zarr files using TensorStore.

Provides faster I/O than standard Zarr library through async operations and
optimized caching. Supports coordinated subsampling for large arrays.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.datapipes.readers.base import Reader
from physicsnemo.datapipes.registry import register

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

# Check if tensorstore is available
TENSORSTORE_AVAILABLE = check_version_spec("tensorstore", hard_fail=False)

if TENSORSTORE_AVAILABLE:
    ts = importlib.import_module("tensorstore")


@register()
class TensorStoreZarrReader(Reader):
    r"""
    High-performance async reader for Zarr files using TensorStore.

    This reader provides faster I/O than the standard ZarrReader through async
    operations, optimized caching, and concurrent data fetching. It's particularly
    beneficial for large datasets on networked storage or cloud storage.

    This is a drop-in replacement for ZarrReader with identical interface.
    Each Zarr group in the directory represents one sample.

    Examples
    --------
    Basic usage:

    >>> # Directory with sample_0.zarr, sample_1.zarr, ...
    >>> reader = TensorStoreZarrReader("data_dir/", group_pattern="sample_*.zarr")  # doctest: +SKIP
    >>> data, metadata = reader[0]  # Returns (TensorDict, dict) tuple  # doctest: +SKIP

    Load only specific fields:

    >>> reader = TensorStoreZarrReader("data_dir/", fields=["positions", "velocity"])  # doctest: +SKIP
    >>> data, metadata = reader[0]  # doctest: +SKIP

    With coordinated subsampling for large arrays:

    >>> reader = TensorStoreZarrReader(  # doctest: +SKIP
    ...     "data_dir/",
    ...     coordinated_subsampling={
    ...         "n_points": 50000,
    ...         "target_keys": ["volume_coords", "volume_fields"],
    ...     }
    ... )
    >>> data, metadata = reader[0]  # doctest: +SKIP

    Performance Tips:
        - Increase ``cache_bytes_limit`` for better performance on repeated access
        - Increase ``data_copy_concurrency`` and ``file_io_concurrency`` for
          parallel workloads
        - Use coordinated subsampling when reading subsets of large arrays
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fields: Optional[list[str]] = None,
        default_values: Optional[dict[str, torch.Tensor]] = None,
        group_pattern: str = "*.zarr",
        cache_bytes_limit: int = 10_000_000,
        data_copy_concurrency: int = 72,
        file_io_concurrency: int = 72,
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
        coordinated_subsampling: Optional[dict[str, Any]] = None,
        domain_parallel: Optional[dict[str, Any]] = None,
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> None:
        """
        Initialize the TensorStore Zarr reader.

        Parameters
        ----------
        path : str or Path
            Path to directory containing Zarr groups.
        fields : list[str], optional
            List of array names to load. If None, loads all available
            arrays from each group.
        default_values : dict[str, torch.Tensor], optional
            Dictionary mapping field names to default tensors.
            If a field in ``fields`` is not found in the file but has an
            entry here, the default tensor is used instead of raising an
            error. Useful for optional fields.
        group_pattern : str, default="*.zarr"
            Glob pattern for finding Zarr groups.
        cache_bytes_limit : int, default=10_000_000
            Total cache size in bytes (default: 10 MB).
        data_copy_concurrency : int, default=72
            Limit for concurrent data copy operations.
        file_io_concurrency : int, default=72
            Limit for concurrent file I/O operations.
        pin_memory : bool, default=False
            If True, place tensors in pinned memory.
        include_index_in_metadata : bool, default=True
            If True, include sample index in metadata.
        coordinated_subsampling : dict[str, Any], optional
            Optional dict to configure coordinated subsampling. If provided,
            must contain ``n_points`` (int) and ``target_keys`` (list of str).
        domain_parallel : dict[str, Any], optional
            Optional dict to configure domain-parallel (rank-local) reading;
            see :class:`~physicsnemo.datapipes.readers.base.Reader`. Sharded
            keys read only this rank's chunk (of the coordinated window when
            subsampling is also configured).
        device_mesh : torch.distributed.device_mesh.DeviceMesh, optional
            1-D device mesh for domain-parallel reading; required with
            ``domain_parallel``. Constructed and injected at runtime.

        Raises
        ------
        ImportError
            If TensorStore is not installed.
        FileNotFoundError
            If path doesn't exist.
        ValueError
            If no Zarr groups found.
        """
        if not TENSORSTORE_AVAILABLE:
            raise ImportError(
                "TensorStore is required for TensorStoreZarrReader but is not installed.\n"
                "Install it with: pip install tensorstore\n"
                "See https://google.github.io/tensorstore/ for more information."
            )

        super().__init__(
            pin_memory=pin_memory,
            include_index_in_metadata=include_index_in_metadata,
            coordinated_subsampling=coordinated_subsampling,
            domain_parallel=domain_parallel,
            device_mesh=device_mesh,
        )

        self.path = Path(path).expanduser().resolve()
        self._user_fields = fields
        self.default_values = default_values or {}
        self.group_pattern = group_pattern

        if not self.path.exists():
            raise FileNotFoundError(f"Path not found: {self.path}")

        if not self.path.is_dir():
            raise ValueError(
                f"Path must be a directory containing Zarr groups: {self.path}"
            )

        # Find all Zarr groups
        self._groups = sorted(
            [
                p
                for p in self.path.glob(group_pattern)
                if p.is_dir()
                and ((p / ".zgroup").exists() or (p / "zarr.json").exists())
            ]
        )

        if not self._groups:
            raise ValueError(
                f"No Zarr groups matching '{group_pattern}' found in {self.path}"
            )

        self._length = len(self._groups)

        # Discover available fields from first group
        self._available_fields = self._discover_fields(self._groups[0])

        # Create TensorStore context with caching config
        self._context = ts.Context(
            {
                "cache_pool": {"total_bytes_limit": cache_bytes_limit},
                "data_copy_concurrency": {"limit": data_copy_concurrency},
                "file_io_concurrency": {"limit": file_io_concurrency},
            }
        )

        # Spec template for opening Zarr arrays
        self._spec_template = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": None,
            },
        }

    def _discover_fields(self, group_path: Path) -> list[str]:
        """Discover array names in a Zarr group (v2 or v3 format)."""
        fields = []

        # List subdirectories that are zarr arrays
        for item in group_path.iterdir():
            if not item.is_dir():
                continue

            # Zarr v2: arrays have .zarray metadata file
            if (item / ".zarray").exists():
                fields.append(item.name)
            # Zarr v3: arrays have zarr.json with node_type="array"
            elif (item / "zarr.json").exists():
                try:
                    with open(item / "zarr.json") as f:
                        metadata = json.load(f)
                    if metadata.get("node_type") == "array":
                        fields.append(item.name)
                except (json.JSONDecodeError, OSError):
                    # Skip malformed or unreadable metadata
                    pass

        return sorted(fields)

    @property
    def fields(self) -> list[str]:
        """Fields that will be loaded (user-specified or all available)."""
        if self._user_fields is not None:
            return self._user_fields
        return self._available_fields

    def _read_attributes(self, group_path: Path) -> dict[str, Any]:
        """Read attributes from a Zarr group (v2 or v3)."""
        store_spec = {"driver": "file", "path": str(group_path)}
        store = ts.KvStore.open(store_spec).result()

        keys = store.list().result()

        # Try Zarr v3 format first
        if b"/zarr.json" in keys:
            zarr_json = store.read(b"/zarr.json").result()
            metadata = json.loads(zarr_json.value)
            if "attributes" in metadata:
                return {k: torch.tensor(v) for k, v in metadata["attributes"].items()}
            return {}

        # Try Zarr v2 format
        elif b"/.zattrs" in keys:
            zarr_attrs = store.read(b"/.zattrs").result()
            metadata = json.loads(zarr_attrs.value)
            return {k: torch.tensor(v) for k, v in metadata.items()}

        return {}

    def _open_stores(
        self, index: int
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """Open the sample's array stores (async, metadata-only) + attributes."""
        group_path = self._groups[index]

        # Read attributes (stored as tensors in sample)
        attributes = self._read_attributes(group_path)

        # Determine which fields to read
        fields_from_arrays = set(self.fields) - set(attributes.keys())

        # Check for missing required fields using cached available fields
        # (discovered once during __init__ from the first group)
        available = set(self._available_fields)
        required_fields = fields_from_arrays - set(self.default_values.keys())
        missing_fields = required_fields - available
        if missing_fields:
            raise KeyError(
                f"Required fields {missing_fields} not found in {group_path}. "
                f"Available: {list(available)}"
            )

        # Open all array stores asynchronously (metadata only, no data read)
        read_futures = {}
        for key in fields_from_arrays:
            if key not in available:
                continue

            spec = {
                "driver": "auto",
                "kvstore": {
                    "driver": "file",
                    "path": str(group_path / key),
                },
            }
            read_futures[key] = ts.open(
                spec, create=False, open=True, context=self._context
            )

        # Wait for opens to complete
        stores = {key: future.result() for key, future in read_futures.items()}
        return stores, attributes

    def _finalize_sample(
        self, tensor_futures: dict[str, Any], attributes: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Resolve read futures, merge attributes and default values."""
        data = {
            key: torch.as_tensor(future.result(), dtype=torch.float32)
            for key, future in tensor_futures.items()
        }
        data.update(attributes)
        for key, default_value in self.default_values.items():
            if key not in data:
                data[key] = default_value.clone()
        return data

    def _read_sample(
        self, index: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, tuple[int, ...]]]:
        """Read a sample: the whole window, or this rank's share of it.

        Store opens are metadata-only, so placements (when domain parallelism
        is on) resolve before any data is read. Attributes and default values
        always replicate.
        """
        # Per-sample generator: reproducible regardless of read order/thread.
        generator = self._index_generator(index)
        stores, attributes = self._open_stores(index)
        window, windowed_keys = self._window_indices(
            lambda key: stores[key].shape[0] if key in stores else None,
            generator,
        )

        shapes = {
            key: (
                len(window)
                if window is not None and key in windowed_keys
                else store.shape[0],
                *store.shape[1:],
            )
            for key, store in stores.items()
        }
        selection, sharded = self._selection_plan(
            shapes, window, windowed_keys, generator
        )

        # Trigger async reads of each key's selection
        tensor_futures = {
            key: store[selection[key]].read() for key, store in stores.items()
        }
        return self._finalize_sample(tensor_futures, attributes), sharded

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Load a single sample from a Zarr group using TensorStore."""
        data, _ = self._read_sample(index)
        return data

    def _load_sample_domain_parallel(
        self, index: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, tuple[int, ...]]]:
        """Load this rank's chunk of a single sample using TensorStore."""
        return self._read_sample(index)

    def __len__(self) -> int:
        """Return number of samples."""
        return self._length

    def _get_field_names(self) -> list[str]:
        """Return field names that will be loaded."""
        return self.fields

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return metadata for a sample."""
        return {
            "source_file": str(self._groups[index]),
            "source_filename": self._groups[index].name,
        }

    @property
    def _supports_coordinated_subsampling(self) -> bool:
        """TensorStore Zarr reader supports coordinated subsampling."""
        return True

    @property
    def _supports_domain_parallel(self) -> bool:
        """TensorStore Zarr reader supports domain-parallel reading."""
        return True

    def __repr__(self) -> str:
        subsample_info = ""
        if self._coordinated_subsampling_config is not None:
            cfg = self._coordinated_subsampling_config
            subsample_info = f", subsampling={cfg['n_points']} points"

        return (
            f"TensorStoreZarrReader("
            f"path={self.path}, "
            f"len={len(self)}, "
            f"fields={self.fields}"
            f"{subsample_info})"
        )
