# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Domain-parallel Zarr dataset for Transolver CFD data.

This dataset implements domain parallelism where multiple ranks can read chunks
of data cooperatively from zarr files. It supports conversion to ShardTensor
objects for distributed training.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from functools import lru_cache

import numpy as np
import zarr
import zarrs

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})


import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.utils.data import Dataset

from physicsnemo.distributed import DistributedManager
from physicsnemo.distributed.shard_tensor import ShardTensor
from physicsnemo.distributed._shard_tensor_spec import (
    ShardTensorSpec,
    _stride_from_contiguous_shape_C_style,
)
from physicsnemo.distributed.utils import compute_split_shapes
from physicsnemo.utils.profiling import profile

import threading


def get_filenames(data_path: Path, exclude_dirs: bool = True) -> List[str]:
    """Get list of filenames from data directory.

    Args:
        data_path: Path to the directory containing files.
        exclude_dirs: If True, exclude directories from the result.

    Returns:
        Sorted list of filenames with .zarr extension.
    """
    filenames = []
    for item in data_path.iterdir():
        if item.suffix in [".zarr"]:
            filenames.append(item.name)
    return sorted(filenames)


def _read_chunk_into_array(
    cpu_array: np.ndarray,
    zarr_array: zarr.Array,
    cpu_slice: slice,
    zarr_slice: slice = None,
) -> None:
    """Helper function to read a chunk from zarr into numpy array.

    Args:
        cpu_array: The destination numpy array.
        zarr_array: The source zarr array.
        cpu_slice: The slice in the cpu_array to write to.
        zarr_slice: The slice in the zarr_array to read from. If None, uses cpu_slice.
    """
    if zarr_slice is None:
        zarr_slice = cpu_slice
    cpu_array[cpu_slice] = zarr_array[zarr_slice]


@lru_cache
def to_torch_dtype(dtype: np.dtype) -> torch.dtype:
    """Convert a numpy dtype to a torch dtype.

    Args:
        dtype: Numpy dtype to convert.

    Returns:
        Corresponding torch dtype.
    """
    temp = torch.from_numpy(np.empty((), dtype=dtype))
    return temp.dtype


class DomainParallelZarrDataset(Dataset):
    """
    PyTorch dataset for domain-parallel reading of Zarr files.

    This dataset supports:
    - Efficient chunk-aligned reading for large data
    - GPU memory streaming (disk -> numpy -> torch.Tensor -> GPU)
    - Domain parallelism via DeviceMesh configuration
    - Conversion to ShardTensor objects for domain-parallel usage

    This dataset supports multiple processes cooperating to read the data, and optimizations
    for single process reading.
    - The dataset will only read the keys from the `keys_to_read` parameter.
    - For keys in `large_keys`, the dataset will spawn threads to read the data,
      where each thread is aligned with a zarr chunk whenever possible.
    - If domain parallelism is enabled, then the large and non-large keys may have alterate behavior:
      - For large keys, each rank will participate in the read.  Each rank will read a portion of the data,
        designed to be approximately equal portions per process.  Depending on the final output placement
        of that key, it will be allgathered on the GPU (replicated) or converted inplace to ShardTensor.
      - For small keys, each rank will read each key in its entirety.  If the output placement is sharded,
        the keys will be converted in place.  A future update may enable round-robin reading, where each
        process will read only a selection of keys and broadcast to partners.

    Technical details:
    - CPU memory is allocated with torch, and if pin_memory is True, it's allocated directly into
      pinned memory space.  Numpy wraps the torch allocated memory, and Zarr streams the data into it.
    - The threading is done on the zarr read only - typically the main bottleneck is the memory allocation
      with pytorch.
      - There might be a benefit to using threads on the torch alloc - but would require mutliple levels of threading.


    Args:
        data_path: Path to directory containing zarr files
        device_mesh: Optional DeviceMesh for domain parallelism. If None, reads full data locally.
        placements: Placement specifications for sharding. Must match mesh dimensions.
        max_workers: Maximum number of threads for parallel I/O
        pin_memory: Whether to pin the CPU memory of the allocated tensors.  If True, the memory is allocated
                    in pinned memory space.  If False, the memory is allocated in the default memory space.
                    Pinning is faster for GPU transfers, but the available pinned memory is smaller.
        keys_to_read: Specific keys to read from zarr files. If None, reads all standard keys.
        large_keys: Keys that should use chunk-aligned reading (example: high res volume data)
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        device_mesh: DeviceMesh | None = None,
        placements: Placement | dict[str, Placement] | None = None,
        max_workers: int = 4,
        pin_memory: bool = True,
        keys_to_read: list[str] | None = None,
        large_keys: set[str] | None = None,
    ):
        super().__init__()

        # Initialize distributed manager if not already done
        if not DistributedManager.is_initialized():
            DistributedManager.initialize()

        self.dm = DistributedManager()
        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.device_mesh = device_mesh
        self.placements = placements
        self.max_workers = max_workers
        self.pin_memory = pin_memory
        self.keys_to_read = keys_to_read or set()
        self.large_keys = large_keys or set()

        self.data_loader_stream = torch.cuda.Stream()
        self.gpu_transfer_compete = None

        # Validate distributed configuration
        if self.device_mesh is not None:
            if self.device_mesh.ndim != 1:
                raise ValueError(
                    "DomainParallelZarrDataset requires a single axis DeviceMesh if used"
                )
            if self.placements is None:
                raise ValueError(
                    "placements must be specified when device_mesh is provided"
                )
            if isinstance(self.placements, dict):
                for key, placement in self.placements.items():
                    if len(placement) != 1:
                        raise ValueError(
                            f"placements must be a single placement for each key, got {placement} for key {key}"
                        )
                # Check that each key to read has a placement:
                for key in self.keys_to_read:
                    if key not in self.placements:
                        raise ValueError(
                            f"placements must specify a placement for each key to read, missing placement for key {key}"
                        )
            elif isinstance(self.placements, tuple):
                if len(self.placements) != 1:
                    raise ValueError(
                        f"placements must be a length-1 tuple of placement if not a dict, got {self.placements}"
                    )
            else:
                raise ValueError(
                    f"placements must be a dict or tuple, got {type(self.placements)}"
                )

        # Get list of zarr files
        self.filenames = get_filenames(self.data_path, exclude_dirs=True)
        if not self.filenames:
            raise ValueError(f"No zarr files found in {self.data_path}")

        # Create thread pool for parallel I/O
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # We cache ShardTensorSpecs for each tensor read, based on how they are
        # read and not how they are meant to be.  They get converted in the end.
        self.tensor_specs = {}

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.filenames)

    def __del__(self):
        """Clean up thread pool on destruction."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

    def _get_slice_boundaries(
        self, zarr_array: zarr.Array
    ) -> Tuple[int, int, tuple | None]:
        # Determine what slice this rank should read
        if self.device_mesh is not None:
            # How many splits to make?
            n_splits = dist.get_world_size(group=self.device_mesh.get_group())
            # What rank is this one?
            this_rank = self.device_mesh.get_local_rank()

            sections = compute_split_shapes(zarr_array.shape[0], n_splits)

            global_chunk_start = sum(sections[:this_rank])
            global_chunk_stop = global_chunk_start + sections[this_rank]

            chunk_sizes = tuple(
                (section,) + zarr_array.shape[1:] for section in sections
            )

        else:
            global_chunk_start, global_chunk_stop = 0, zarr_array.shape[0]
            chunk_sizes = None

        return global_chunk_start, global_chunk_stop, chunk_sizes

    def create_tensor_spec(
        self,
        zarr_array: zarr.Array,
        key: str,
        placements: tuple[Placement],
        sharding_shapes: dict[int, tuple[int]] | None = None,
    ) -> ShardTensorSpec:
        # Unpack the batch index:
        shape = (1,) + zarr_array.shape
        # Don't forget to unpack it in the sharding shapes too:
        if sharding_shapes is not None:
            for k in sharding_shapes.keys():
                sharding_shapes[k] = tuple((1,) + s for s in sharding_shapes[k])

        stride = _stride_from_contiguous_shape_C_style(shape)

        meta = TensorMeta(shape, stride, to_torch_dtype(zarr_array.dtype))
        return ShardTensorSpec(
            mesh=self.device_mesh,
            placements=placements,
            tensor_meta=meta,
            _sharding_shapes=sharding_shapes,
        )

    @profile
    def _read_key_chunk_aligned(
        self, zarr_group: zarr.Group, key: str, futures: list[Future]
    ) -> np.ndarray:
        """
        Read a key using chunk-aligned I/O for efficiency.
        Implements domain parallelism if device_mesh is configured.

        Args:
            zarr_group: The zarr group to read from.
            key: The key to read.
            futures: List to append thread futures to.

        Returns:
            Torch tensor containing the read data.
        """

        zarr_array = zarr_group[key]

        global_chunk_start, global_chunk_stop, chunk_sizes = self._get_slice_boundaries(
            zarr_array
        )
        if chunk_sizes is not None:
            self.tensor_specs[key] = self.create_tensor_spec(
                zarr_array, key, (Shard(1),), {0: chunk_sizes}
            )

        # Calculate the shape of data this rank will read
        local_shape = [global_chunk_stop - global_chunk_start] + list(
            zarr_array.shape[1:]
        )

        # Pre-allocate result array
        torch_output = torch.empty(
            local_shape,
            dtype=to_torch_dtype(zarr_array.dtype),
            pin_memory=self.pin_memory,
        )
        # Share the memory buffer with numpy:
        result = torch_output.numpy()

        # For chunk-aligned reading, align with zarr's chunk boundaries
        # This is easiest when we precompute start/end slices
        zarr_chunk_size = zarr_array.chunks[0]

        # Generate the global list of chunk boundaries first (then apply corrections)
        slice_starts = list(range(0, zarr_array.shape[0], zarr_chunk_size))
        slice_stops = [start + zarr_chunk_size for start in slice_starts]

        # Correct the last stop:
        slice_stops[-1] = zarr_array.shape[0]

        # Now, select all the slices that this rank is responsible for:
        # These are all the boundary points exclusively within the global chunk slice:
        local_boundaries = [
            s for s in slice_starts if s >= global_chunk_start and s < global_chunk_stop
        ]

        # Fix the boundaries:
        if global_chunk_start not in local_boundaries:
            zarr_slice_starts = [global_chunk_start] + local_boundaries
        else:
            zarr_slice_starts = [s for s in local_boundaries]

        # The stops are the +1 locations
        zarr_slice_stops = [s for s in zarr_slice_starts[1:]]

        # Handle the end situation
        if global_chunk_stop not in zarr_slice_stops:
            zarr_slice_stops.append(global_chunk_stop)
        else:
            # It's already in the list of boundaries
            # We have to reduce the list of starts:
            zarr_slice_starts = zarr_slice_starts[:-1]

        slice_sizes = [
            stop - start for stop, start in zip(zarr_slice_stops, zarr_slice_starts)
        ]

        cpu_slice_starts = [0]
        cpu_slice_stops = [slice_sizes[0]]
        for slice_size in slice_sizes[1:]:
            cpu_slice_starts.append(cpu_slice_stops[-1])
            cpu_slice_stops.append(cpu_slice_starts[-1] + slice_size)

        # Now, spawn threads to do each of those reads.

        for i in range(len(slice_sizes)):
            zarr_slice = np.s_[zarr_slice_starts[i] : zarr_slice_stops[i]]
            cpu_slice = np.s_[cpu_slice_starts[i] : cpu_slice_stops[i]]

            future = self.executor.submit(
                _read_chunk_into_array,
                result,
                zarr_array,
                cpu_slice,
                zarr_slice,
            )
            futures.append(future)

        return torch_output

    @profile
    def _read_key_standard(
        self, zarr_group: zarr.Group, key: str, futures: list[Future]
    ) -> torch.Tensor:
        """Read a key with simple I/O (for small arrays).

        Args:
            zarr_group: The zarr group to read from.
            key: The key to read.
            futures: List to append thread futures to.

        Returns:
            Torch tensor containing the read data.
        """
        zarr_array = zarr_group[key]

        # Handle scalar values
        if zarr_array.shape == ():
            if self.device_mesh is not None:
                self.tensor_specs[key] = self.create_tensor_spec(
                    zarr_array,
                    key,
                    (Replicate(),),
                )

            output = torch.from_numpy(np.array(zarr_array))
            if self.pin_memory:
                output = output.pin_memory()
            return output

        global_chunk_start, global_chunk_stop, chunk_sizes = self._get_slice_boundaries(
            zarr_array
        )
        if chunk_sizes is not None:
            self.tensor_specs[key] = self.create_tensor_spec(
                zarr_array, key, (Shard(1),), {0: chunk_sizes}
            )

        # Calculate the shape of data this rank will read
        local_shape = [global_chunk_stop - global_chunk_start] + list(
            zarr_array.shape[1:]
        )

        # data = np.empty(zarr_array.shape, dtype=zarr_array.dtype)
        output = torch.empty(
            local_shape,
            dtype=to_torch_dtype(zarr_array.dtype),
            pin_memory=self.pin_memory,
        )
        data = output.numpy()
        slice = np.s_[:]

        # The zarr slice is not the numpy slice if we're sharding
        if chunk_sizes is not None:
            zarr_slice = np.s_[global_chunk_start:global_chunk_stop]
        else:
            zarr_slice = np.s_[:]

        futures.append(
            self.executor.submit(
                _read_chunk_into_array,
                data,
                zarr_array,
                slice,
                zarr_slice,
            )
        )

        return output

    @profile
    def _read_zarr_file(self, filepath: Path) -> Dict[str, np.ndarray]:
        """Read data from a zarr file.

        Args:
            filepath: Path to the zarr file.

        Returns:
            Dictionary mapping keys to numpy arrays or torch tensors.
        """
        zarr_group = zarr.open_group(str(filepath), mode="r")
        # group_keys = list(zarr_group.keys())

        data = {}
        futures = []

        # Process each key
        for key in self.keys_to_read:
            # if key not in group_keys:
            #     continue

            if key in self.large_keys:
                # Use chunk-aligned reading for large data
                data[key] = self._read_key_chunk_aligned(zarr_group, key, futures)
            else:
                # Use simple reading for other data
                data[key] = self._read_key_standard(zarr_group, key, futures)

        # Wait for all futures to complete
        for future in futures:
            future.result()

        return data

    @profile
    def _move_to_gpu(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to torch tensors and move to GPU if available.

        Args:
            data: Dictionary of key to torch tensor.

        Returns:
            Dictionary of key to torch tensor on GPU if available.
        """
        result = {}

        with torch.cuda.stream(self.data_loader_stream):
            for key, array in data.items():
                # Move to GPU if available
                if self.dm.cuda:
                    result[key] = data[key].to(self.dm.device, non_blocking=True)

            self.gpu_transfer_compete = torch.cuda.Event()
            self.gpu_transfer_compete.record(self.data_loader_stream)

        return result

    def _convert_to_shard_tensors(
        self, tensors: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, ShardTensor]]:
        """Convert tensors to ShardTensor objects for distributed training.

        Args:
            tensors: Dictionary of key to torch tensor.

        Returns:
            Dictionary of key to torch tensor or ShardTensor.
        """
        if self.device_mesh is None:
            return tensors

        result = {}

        for key, tensor in tensors.items():
            # Create a ShardTensor with whatever layout the data is actually in:
            st = ShardTensor.__new__(
                ShardTensor,
                local_tensor=tensor,
                spec=self.tensor_specs[key],
                requires_grad=False,  # By default, the data pipe output doesn't need a grad.
            )

            # Find out the desired placement:
            if tensor.numel() > 1:
                if isinstance(self.placements, dict):
                    target_placement = self.placements[key]
                else:
                    target_placement = self.placements
            else:
                target_placement = (Replicate(),)

            # Redistribute if necessary:
            # (Recall that this is one dimensional mesh only)
            if st._spec.placements[0] != target_placement[0]:
                st = st.redistribute(placements=target_placement)

            result[key] = st

        return result

    def preload(self, idx: int) -> None:
        """
        Asynchronously preload the data for the given index (up to CPU, not GPU).
        Only one preload operation is supported at a time.

        Args:
            idx: Index of the sample to preload.
        """
        if hasattr(self, "_preload_thread") and self._preload_thread is not None:
            # Optionally, wait for previous preload to finish or raise error
            self._preload_thread.join()

        self._preload_result = None
        self._preload_exception = None

        def _preload_worker():
            try:
                filename = self.filenames[idx]
                filepath = self.data_path / filename
                data = self._read_zarr_file(filepath)
                # Convert to torch tensors
                data = self._move_to_gpu(data)
                self._preload_result = (idx, data)
            except Exception as e:
                self._preload_exception = e

        self._preload_thread = threading.Thread(target=_preload_worker)
        self._preload_thread.start()

    def get_preloaded(self) -> Tuple[int, Dict[str, np.ndarray]]:
        """
        Retrieve the preloaded data (blocking if not ready).

        Returns:
            (idx, data) tuple where data is a dictionary of key to numpy array or torch tensor.

        Raises:
            RuntimeError: If no preload is in progress.
            Exception: If preload failed.
        """
        if not hasattr(self, "_preload_thread") or self._preload_thread is None:
            raise RuntimeError("No preload in progress. Call preload(idx) first.")

        self._preload_thread.join()
        self._preload_thread = None

        if self._preload_exception is not None:
            raise self._preload_exception

        return self._preload_result

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        """
        This supports the DALI iterator.
        """
        if self.i >= len(self.filenames):
            self.i = 0
            raise StopIteration
        data = self._read_zarr_file(self.data_path / self.filenames[self.i])

        self.i += 1
        return tuple(data[key].unsqueeze(0) for key in self.keys_to_read)

    def __len__(self):
        return len(self.filenames)

    @profile
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | ShardTensor]:
        """
        Get a data sample.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing tensors/ShardTensors for the requested data
        """
        if idx >= len(self.filenames):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.filenames)}"
            )

        if hasattr(self, "_preload_result") and self._preload_result is not None:
            preload_idx, preload_data = self._preload_result
            if preload_idx == idx:
                data = preload_data
                self._preload_result = None  # Clear after use
            else:
                # Preloaded data is for a different idx, ignore it
                data = self._read_zarr_file(self.data_path / self.filenames[idx])
                data = self._move_to_gpu(data)
        else:
            filename = self.filenames[idx]
            filepath = self.data_path / filename
            # Read data from zarr file
            data = self._read_zarr_file(filepath)
            data = self._move_to_gpu(data)

        # This blocks until the preprocessing has transferred to GPU
        if self.gpu_transfer_compete is not None:
            torch.cuda.current_stream().wait_event(self.gpu_transfer_compete)

        # Add a batch index:
        data = {key: value.unsqueeze(0) for key, value in data.items()}

        # Convert to ShardTensors if using domain parallelism
        if self.device_mesh is not None:
            data = self._convert_to_shard_tensors(data)

        return data


# TODO: Additional features to consider implementing:
# 1. Caching of metadata to avoid repeated zarr.open_group calls
# 2. Asynchronous prefetching of next sample while processing current
# 3. More sophisticated sharding strategies (e.g., spatial partitioning)
# 4. Support for different placement strategies per key
# 5. Memory-mapped reading for very large datasets
# 6. Compression/decompression handling
