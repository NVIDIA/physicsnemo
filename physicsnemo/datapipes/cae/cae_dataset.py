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

import importlib
import json
import pathlib
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import Replicate, Shard

from physicsnemo.core.version_check import OptionalImport, check_version_spec
from physicsnemo.distributed.utils import compute_split_shapes
from physicsnemo.domain_parallel import ShardTensor, ShardTensorSpec

zarr = OptionalImport("zarr")

TENSORSTORE_AVAILABLE = check_version_spec("tensorstore", hard_fail=False)
PV_AVAILABLE = check_version_spec("pyvista", hard_fail=False)

# Abstractions:
# - want to read npy/npz/.zarr/.stl/.vtp files
# - Need to share next level abstractions
# - Domain parallel dataloading is supported: output will be ShardTensor instead.
# - need to be able to configure preprocessing
# - CPU -> GPU transfer happens here, needs to be isolated in it's own stream
# - Output of dataloader should be torch.Tensor objects.


"""
This datapipe handles reading files from Zarr and piping into torch.Tensor objects.

It's expected that the files are organized as groups, with each .zarr
file representing one training example.  To improve IO performance, the files 
should be chunked for each array.  The reader takes a list of keys in the 
group to read, and will not read keys that are not specified.  The exception
is if _no_ keys are passed, in which case _all_ keys will be read.
"""


class BackendReader(ABC):
    """
    Abstract base class for backend readers.
    """

    def __init__(
        self,
        keys_to_read: list[str] | None,
        keys_to_read_if_available: dict[str, torch.Tensor] | None,
    ) -> None:
        """
        Initialize the backend reader.
        """
        self.keys_to_read = keys_to_read
        self.keys_to_read_if_available = keys_to_read_if_available

        self.volume_sampling_size = None

        self.is_volumetric = any(["volume" in key for key in self.keys_to_read])

    @abstractmethod
    def read_file(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
        """
        Read a file and return a dictionary of tensors.
        """
        pass

    @abstractmethod
    def read_file_attributes(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
        """
        Read the attributes of a file and return a dictionary of tensors.
        """
        pass

    @abstractmethod
    def read_file_sharded(
        self, filename: pathlib.Path, device_mesh: torch.distributed.DeviceMesh
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
        """
        Read a file and return a dictionary of tensors ready to convert to ShardTensors.

        NOTE: this function does not actually convert torch tensors to ShardTensors.
        It's possible that the conversion, in some cases, can be a collective function.
        Due to the async nature of the loader, we don't rely on any ordering of
        collectives and defer them to the last possible minute.

        Additionally, these functions return CPU tensors and we don't actually
        define shard tensors on cpu.

        So, the dataset itself will convert a local tensor + shard info to shard tensor
        after the cpu-> gpu movement.
        """
        pass

    def fill_optional_keys(
        self, data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Fill missing keys with the keys from the keys_to_read_if_available dictionary.
        """
        for key in self.keys_to_read_if_available:
            if key not in data.keys():
                data[key] = self.keys_to_read_if_available[key]
        return data

    def _get_slice_boundaries(
        self, array_shape: tuple[int], this_rank: int, n_splits: int, split_dim: int = 0
    ) -> tuple[int, int, tuple | None]:
        """
        For an array, determine the slice boundaries for parallel reading.

        Args:
            array_shape: The total shape of the target array.
            this_rank: The rank of the distributed process.
            n_splits: The size of the distributed process.
            split_dim: The dimension to split, default is 0.

        Returns:
            The slice boundaries for parallel reading.
        """
        # Determine what slice this rank should read

        sections = compute_split_shapes(array_shape[split_dim], n_splits)

        global_chunk_start = sum(sections[:this_rank])
        global_chunk_stop = global_chunk_start + sections[this_rank]

        chunk_sizes = tuple(
            array_shape[:split_dim] + (section,) + array_shape[split_dim + 1 :]
            for section in sections
        )

        return global_chunk_start, global_chunk_stop, chunk_sizes

    def set_volume_sampling_size(self, volume_sampling_size: int):
        """
        Set the volume sampling size.  When set, the readers will
        assume the volumetric data is shuffled on disk and read only
        contiguous chunks of the data up to the sampling size.


        Args:
            volume_sampling_size: The total size of the volume sampling.

        """
        self.volume_sampling_size = volume_sampling_size

    def select_random_sections_from_slice(
        self,
        slice_start: int,
        slice_stop: int,
        n_points: int,
    ) -> slice:
        """

        select the contiguous chunks of the volume data to read.

        Args:
            n_volume_points: The number of points to sample from the volume.

        Returns:
            A tuple of the start and stop indices of the contiguous chunks.
        """

        if slice_stop - slice_start < n_points:
            raise ValueError(
                f"Slice size {slice_stop - slice_start} is less than the number of points {n_points}"
            )

        # Choose a random start point that will fit the entire n_points region:
        start = np.random.randint(slice_start, slice_stop - n_points)
        return slice(start, start + n_points)


class NpyFileReader(BackendReader):
    """
    Reader for numpy files.
    """

    def __init__(
        self,
        keys_to_read: list[str] | None,
        keys_to_read_if_available: dict[str, torch.Tensor] | None,
    ) -> None:
        super().__init__(keys_to_read, keys_to_read_if_available)

    def read_file(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
        """
        Read a file and return a dictionary of tensors.
        """
        data = np.load(filename, allow_pickle=True).item()

        missing_keys = set(self.keys_to_read) - set(data.keys())

        if len(missing_keys) > 0:
            raise ValueError(f"Keys {missing_keys} not found in file {filename}")

        data = {key: torch.from_numpy(data[key]) for key in self.keys_to_read}

        return self.fill_optional_keys(data)

    def read_file_attributes(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
        """
        NPY doesn't read the attributes of a file and return a dictionary of tensors.
        Returning nothing.
        """
        return {}

    def read_file_sharded(
        self, filename: pathlib.Path, device_mesh: torch.distributed.DeviceMesh
    ) -> dict[str, ShardTensor]:
        pass

    def set_volume_sampling_size(self, volume_sampling_size: int):
        """
        This is not supported for npy files.
        """
        raise NotImplementedError(
            "volume sampling directly from disk is not supported for npy files."
        )


class NpzFileReader(BackendReader):
    """
    Reader for npz files.
    """

    def __init__(
        self,
        keys_to_read: list[str] | None,
        keys_to_read_if_available: dict[str, torch.Tensor] | None,
    ) -> None:
        super().__init__(keys_to_read, keys_to_read_if_available)

    def read_file(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
        """
        Read a file and return a dictionary of tensors.
        """
        in_data = np.load(filename)

        keys_found = set(in_data.keys())
        keys_missing = set(self.keys_to_read) - keys_found
        if len(keys_missing) > 0:
            raise ValueError(f"Keys {keys_missing} not found in file {filename}")

        # Make sure to select the slice outside of the loop.
        if self.is_volumetric:
            volume_key = next(key for key in in_data.keys() if "volume" in key)
            volume_shape = in_data[volume_key].shape[0]
            if self.volume_sampling_size is not None:
                volume_slice = self.select_random_sections_from_slice(
                    0,
                    volume_shape,
                    self.volume_sampling_size,
                )
            else:
                volume_slice = slice(0, volume_shape)

        # This is a slower basic way to do this, to be improved:
        data = {}
        for key in self.keys_to_read:
            if "volume" not in key:
                data[key] = torch.from_numpy(in_data[key][:])
            else:
                data[key] = torch.from_numpy(in_data[key][volume_slice])

        # data = {key: torch.from_numpy(in_data[key][:]) for key in self.keys_to_read}

        return self.fill_optional_keys(data)

    def read_file_attributes(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
        """
        NPZ doesn't read the attributes of a file and return a dictionary of tensors.
        Returning nothing.
        """
        return {}

    def read_file_sharded(
        self, filename: pathlib.Path, device_mesh: torch.distributed.DeviceMesh
    ) -> dict[str, ShardTensor]:
        pass

    def set_volume_sampling_size(self, volume_sampling_size: int):
        """
        This is not supported for npz files.
        """
        raise NotImplementedError(
            "volume sampling directly from disk is not supported for npz files."
        )


class ZarrFileReader(BackendReader):
    """
    Reader for zarr files.
    """

    def __init__(
        self,
        keys_to_read: list[str] | None,
        keys_to_read_if_available: dict[str, torch.Tensor] | None,
    ) -> None:
        super().__init__(keys_to_read, keys_to_read_if_available)

    def read_file_attributes(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
        """
        Read the attributes of a file and return a dictionary of tensors.
        """
        group = zarr.open_group(filename, mode="r")
        return group.attrs

    def read_file(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
        """
        Read a file and return a dictionary of tensors.
        """
        group = zarr.open_group(filename, mode="r")

        attributes = self.read_file_attributes(filename)

        missing_keys = (
            set(self.keys_to_read) - set(group.keys()) - set(attributes.keys())
        )
        data = {}

        if len(missing_keys) > 0:
            raise ValueError(f"Keys {missing_keys} not found in file {filename}")

        # Read in attributes:
        for key in self.keys_to_read:
            if key in attributes.keys():
                data[key] = torch.tensor(attributes[key])

        # Make sure to select the slice outside of the loop.
        if self.is_volumetric:
            volume_key = next(key for key in group.keys() if "volume" in key)
            volume_shape = group[volume_key].shape[0]
            if self.volume_sampling_size is not None:
                volume_slice = self.select_random_sections_from_slice(
                    0,
                    volume_shape,
                    self.volume_sampling_size,
                )
            else:
                volume_slice = slice(0, volume_shape)

        # This is a slower basic way to do this, to be improved:
        for key in self.keys_to_read:
            # Don't read things that came from attributes, potentially;
            if key in data.keys():
                continue

            if group[key].shape == ():
                data[key] = torch.from_numpy(np.array(group[key])).to(torch.float32)
            else:
                if "volume" not in key:
                    data[key] = torch.from_numpy(group[key][:])
                else:
                    data[key] = torch.from_numpy(group[key][volume_slice])

        return self.fill_optional_keys(data)

    def read_file_sharded(
        self, filename: pathlib.Path, device_mesh: torch.distributed.DeviceMesh
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
        """
        Read a file and return a dictionary of tensors.
        """

        # We need the coordinates of this GPU:
        this_rank = device_mesh.get_local_rank()
        domain_size = dist.get_world_size(group=device_mesh.get_group())

        group = zarr.open_group(filename, mode="r")

        attributes = self.read_file_attributes(filename)

        missing_keys = (
            set(self.keys_to_read) - set(group.keys()) - set(attributes.keys())
        )

        if len(missing_keys) > 0:
            raise ValueError(f"Keys {missing_keys} not found in file {filename}")

        data = {}

        # Read in attributes:
        for key in self.keys_to_read:
            if key in attributes.keys():
                data[key] = torch.tensor(attributes[key])

        specs = {}
        for key in self.keys_to_read:
            # Skip attributes:
            if key in data.keys():
                continue

            # Open the array in zarr without reading it and get info:
            zarr_array = group[key]
            array_shape = zarr_array.shape
            if array_shape == ():
                # Read scalars from every rank and use replicate sharding
                raw_data = torch.from_numpy(zarr_array[:])
                placement = [
                    Replicate(),
                ]
                chunk_sizes = None
            else:
                target_dim = 0
                if array_shape[target_dim] < domain_size:
                    # If the array is smaller than the number of ranks,
                    # again read and use replicate sharding:
                    raw_data = torch.from_numpy(zarr_array[:])
                    placement = [
                        Replicate(),
                    ]
                    chunk_sizes = None
                else:
                    # Read partially from the data and use Shard(target_dim) sharding
                    chunk_start, chunk_stop, chunk_sizes = self._get_slice_boundaries(
                        zarr_array.shape, this_rank, domain_size
                    )
                    raw_data = torch.from_numpy(zarr_array[chunk_start:chunk_stop])
                    placement = [
                        Shard(target_dim),
                    ]

                    # Turn chunk sizes into a dict over mesh dim 0:
                    chunk_sizes = {0: chunk_sizes}

            #
            data[key] = raw_data
            specs[key] = (placement, chunk_sizes)

        # Patch in the optional keys:
        data = self.fill_optional_keys(data)
        for key in data.keys():
            if key not in specs:
                specs[key] = (
                    [
                        Replicate(),
                    ],
                    {},
                )

        return data, specs


if PV_AVAILABLE:
    pv = importlib.import_module("pyvista")

    class VTKFileReader(BackendReader):
        """
        Reader for vtk files.
        """

        def __init__(
            self,
            keys_to_read: list[str] | None,
            keys_to_read_if_available: dict[str, torch.Tensor] | None,
        ) -> None:
            super().__init__(keys_to_read, keys_to_read_if_available)

            self.stl_file_keys = [
                "stl_coordinates",
                "stl_centers",
                "stl_faces",
                "stl_areas",
            ]
            self.vtp_file_keys = [
                "surface_mesh_centers",
                "surface_normals",
                "surface_mesh_sizes",
                "CpMeanTrim",
                "pMeanTrim",
                "wallShearStressMeanTrim",
            ]
            self.vtu_file_keys = [
                "volume_mesh_centers",
                "volume_fields",
            ]

            self.exclude_patterns = [
                "single_solid",
            ]

        def get_file_name(self, dir_name: pathlib.Path, extension: str) -> pathlib.Path:
            """
            Get the file name for a given directory and extension.
            """
            # >>> matches = [p for p in list(dir_name.iterdir()) if p.suffix == ".stl" and not any(pattern in p.name for pattern in exclude_patterns)]
            matches = [
                p
                for p in dir_name.iterdir()
                if p.suffix == extension
                and not any(pattern in p.name for pattern in self.exclude_patterns)
            ]
            if len(matches) == 0:
                raise FileNotFoundError(f"No {extension} files found in {dir_name}")
            fname = matches[0]
            return dir_name / fname

        def read_file(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
            """
            Read a set of files and return a dictionary of tensors.
            """

            # This reader attempts to only read what's necessary, and not more.
            # So, the functions that do the reading are each "one file" functions
            # and we open them for processing only when necessary.

            return_data = {}

            # Note that this reader is, already, running in a background thread.
            # It may or may not help to further thread these calls.
            if any(key in self.stl_file_keys for key in self.keys_to_read):
                stl_path = self.get_file_name(filename, ".stl")
                stl_data = self.read_data_from_stl(stl_path)
                return_data.update(stl_data)
            if any(key in self.vtp_file_keys for key in self.keys_to_read):
                vtp_path = self.get_file_name(filename, ".vtp")
                vtp_data = self.read_data_from_vtp(vtp_path)
                return_data.update(vtp_data)
            if any(key in self.vtu_file_keys for key in self.keys_to_read):
                raise NotImplementedError("VTU files are not supported yet.")

            return self.fill_optional_keys(return_data)

        def read_file_sharded(
            self, filename: pathlib.Path, parallel_rank: int, parallel_size: int
        ) -> tuple[dict[str, torch.Tensor], dict[str, ShardTensorSpec]]:
            """
            Read a file and return a dictionary of tensors.
            """
            raise NotImplementedError("Not implemented yet.")

        def read_data_from_stl(
            self,
            stl_path: str,
        ) -> dict:
            """
            Reads surface mesh data from an STL file and prepares a batch dictionary for inference.

            Args:
                stl_path (str): Path to the STL file.

            Returns:
                dict: Batch dictionary with mesh faces and coordinates as torch tensors.
            """

            mesh = pv.read(stl_path)

            batch = {}

            faces = mesh.faces.reshape(-1, 4)
            faces = faces[:, 1:]

            batch["stl_faces"] = faces.flatten()

            batch["stl_coordinates"] = mesh.points
            batch["surface_normals"] = mesh.cell_normals

            batch = {k: torch.from_numpy(v) for k, v in batch.items()}

            return batch

        def read_data_from_vtp(self, vtp_path: str) -> dict:
            """
            Read vtp file from a file
            """

            raise NotImplementedError("Not implemented yet.")

        def set_volume_sampling_size(self, volume_sampling_size: int):
            """
            This is not supported for vtk files.
            """
            raise NotImplementedError(
                "volume sampling directly from disk is not supported for vtk files."
            )
else:

    class VTKFileReader(BackendReader):
        """
        Dummy reader for vtk files.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "CAE Dataset: VTKFileReader is not available without pyvista.\n"
                "Please see https://docs.pyvista.org/getting-started/installation.html for installation instructions."
            )


if TENSORSTORE_AVAILABLE:
    ts = importlib.import_module("tensorstore")

    class TensorStoreZarrReader(BackendReader):
        """
        Reader for tensorstore zarr files.
        """

        def __init__(
            self,
            keys_to_read: list[str] | None,
            keys_to_read_if_available: dict[str, torch.Tensor] | None,
            cache_bytes_limit: int = 10_000_000,
            data_copy_concurrency: int = 72,
            file_io_concurrency: int = 72,
        ) -> None:
            super().__init__(keys_to_read, keys_to_read_if_available)

            self.spec_template = {
                "driver": "auto",
                "kvstore": {
                    "driver": "file",
                    "path": None,
                },
            }

            self.context = ts.Context(
                {
                    "cache_pool": {"total_bytes_limit": cache_bytes_limit},
                    "data_copy_concurrency": {"limit": data_copy_concurrency},
                    "file_io_concurrency": {"limit": file_io_concurrency},
                }
            )

        def read_file_attributes(
            self, filename: pathlib.Path
        ) -> dict[str, torch.Tensor]:
            """
            Read the attributes of a file and return a dictionary of tensors.
            """
            store_spec = self.spec_template["kvstore"].copy()
            store_spec["path"] = str(filename)
            store = ts.KvStore.open(store_spec).result()

            keys = store.list().result()

            def to_tensor_dict(attributes_dict):
                attributes = {}
                for k, v in attributes_dict.items():
                    try:
                        attributes[k] = torch.tensor(v)
                    except (TypeError, ValueError, RuntimeError):  # noqa PERF203
                        pass
                return attributes

            # Zarr 3 check:
            if b"/zarr.json" in keys:
                zarr_json = store.read(b"/zarr.json").result()
                # load into json's parser:
                attributes_dict = json.loads(zarr_json.value)["attributes"]
                return to_tensor_dict(attributes_dict)
            elif b"/.zattrs" in keys:
                # Zarr 2:
                zarr_attrs = store.read(b"/.zattrs").result()
                attributes_dict = json.loads(zarr_attrs.value)
                return to_tensor_dict(attributes_dict)
            else:
                return {}

        def read_file(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
            """
            Read a file and return a dictionary of tensors.
            """

            # We need to figure out, first, which keys are attributes.
            attributes = self.read_file_attributes(filename)

            local_keys_to_read = set(self.keys_to_read) - set(attributes.keys())

            # Trigger an async open of each data item:
            read_futures = {}
            for key in local_keys_to_read:
                spec = self.spec_template.copy()
                spec["kvstore"]["path"] = str(filename) + "/" + str(key)

                read_futures[key] = ts.open(
                    spec, create=False, open=True, context=self.context
                )

            # Wait for all the opens to conclude:
            read_futures = {
                key: read_futures[key].result() for key in read_futures.keys()
            }

            # Make sure to select the slice outside of the loop.
            if self.is_volumetric:
                volume_key = next(key for key in read_futures.keys() if "volume" in key)
                volume_shape = read_futures[volume_key].shape[0]
                if self.volume_sampling_size is not None:
                    volume_slice = self.select_random_sections_from_slice(
                        0,
                        volume_shape,
                        self.volume_sampling_size,
                    )
                else:
                    volume_slice = slice(0, volume_shape)

            # Trigger an async read of each data item:
            # (Each item will be a numpy ndarray after this:)
            tensor_futures = {}
            for key in local_keys_to_read:
                if "volume" not in key:
                    tensor_futures[key] = read_futures[key].read()
                # For the volume data, read the slice:
                else:
                    tensor_futures[key] = read_futures[key][volume_slice].read()

            # Convert them to torch tensors:
            # (make sure to block for the result)
            data = {
                key: torch.as_tensor(tensor_futures[key].result(), dtype=torch.float32)
                for key in local_keys_to_read
            }

            # Patch in the attributes:
            data.update(attributes)

            return self.fill_optional_keys(data)

        def read_file_sharded(
            self, filename: pathlib.Path, device_mesh: torch.distributed.DeviceMesh
        ) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
            """
            Read a file and return a dictionary of tensors.
            """
            # We need to figure out, first, which keys are attributes.
            attributes = self.read_file_attributes(filename)

            local_keys_to_read = set(self.keys_to_read) - set(attributes.keys())

            # We need the coordinates of this GPU:
            this_rank = device_mesh.get_local_rank()
            domain_size = dist.get_world_size(group=device_mesh.get_group())

            # This pulls a list of store objects in tensorstore:
            stores = {}
            for key in local_keys_to_read:
                spec = self.spec_template.copy()
                spec["kvstore"]["path"] = str(filename) + "/" + str(key)

                stores[key] = ts.open(
                    spec, create=False, open=True, context=self.context
                )

            stores = {key: stores[key].result() for key in stores.keys()}

            data = {}
            specs = {}
            for key in local_keys_to_read:
                # Open the array in zarr without reading it and get info:
                store = stores[key]
                array_shape = store.shape
                if array_shape == ():
                    # Read scalars from every rank and use replicate sharding
                    _slice = np.s_[:]
                    # raw_data = torch.from_numpy(store[:])
                    placement = [
                        Replicate(),
                    ]
                    chunk_sizes = None
                else:
                    target_dim = 0
                    if array_shape[target_dim] < domain_size:
                        # If the array is smaller than the number of ranks,
                        # again read and use replicate sharding:
                        _slice = np.s_[:]
                        # raw_data = torch.from_numpy(store[:])
                        placement = [
                            Replicate(),
                        ]
                        chunk_sizes = None
                    else:
                        # Read partially from the data and use Shard(target_dim) sharding
                        chunk_start, chunk_stop, chunk_sizes = (
                            self._get_slice_boundaries(
                                store.shape, this_rank, domain_size
                            )
                        )
                        _slice = np.s_[chunk_start:chunk_stop]
                        # raw_data = torch.from_numpy(zarr_array[chunk_start:chunk_stop])
                        placement = [
                            Shard(target_dim),
                        ]

                        # Turn chunk sizes into a dict over mesh dim 0:
                        chunk_sizes = {0: chunk_sizes}

                # Trigger the reads as async:
                data[key] = store[_slice].read()
                specs[key] = (placement, chunk_sizes)

            # Finally, await the full data read:
            for key in local_keys_to_read:
                data[key] = torch.as_tensor(data[key].result())

            # Patch in the optional keys:
            data = self.fill_optional_keys(data)
            for key in data.keys():
                if key not in specs:
                    specs[key] = (
                        [
                            Replicate(),
                        ],
                        {},
                    )

            return data, specs

else:

    class TensorStoreZarrReader(BackendReader):
        """
        Null reader for tensorstore zarr files.
        """

        def __init__(
            self,
            keys_to_read: list[str] | None,
            keys_to_read_if_available: dict[str, torch.Tensor] | None,
        ) -> None:
            # Raise an exception on construction if we get here:
            raise NotImplementedError(
                "TensorStoreZarrReader is not available without tensorstore.  `pip install tensorstore`."
            )


def is_vtk_directory(file: pathlib.Path) -> bool:
    """
    Check if a file is a vtk directory.
    """
    return file.is_dir() and all(
        [f.suffix in [".vtp", ".stl", ".vtu", ".vtk", ".csv"] for f in file.iterdir()]
    )


class CAEDataset:
    """
    Dataset reader for DrivaerML and similar datasets.  In general, this
    dataset supports reading dictionary-like data, and returning a
    dictionary of torch.Tensor objects.

    When constructed, the user must pass a directory of data examples.
    The dataset will inspect the folder, identify all children, and decide:
    - If every file is a directory ending in .zarr, the zarr reader is used.
    - If every file is .npy, the .npy reader is used.
    - If every file is .npz, the .npz reader is used.
    - If every file is a directory without an extension, it's assumed to be .stl/.vtp/.vtu

    The user can optionally force one path with a parameter.

    The flow of this dataset is:
    - Load data from file, using a thread.
        - Each individual file reading tool may or may not have it's own threading
          or multi processing enabled.  That's up to it.  This just does async
          loading.
        - Data should come out of the readers in dict{str : torch.Tensor} format
    - The data is transferred from CPU to GPU in a separate stream.

    Users can call __getitem__(i), which will trigger the pipeline,
    or they can call `preload(i)`, which will start the pipeline for index `i`.
    Subsequent calls to `__getitem__(i)` should be faster since the IO is in
    progress or complete.

    Using the `__iter__` functionality will automatically enable preloading.

    """

    def __init__(
        self,
        data_dir: str | pathlib.Path,
        keys_to_read: list[str] | None,
        keys_to_read_if_available: dict[str, torch.Tensor] | None,
        output_device: torch.device,
        preload_depth: int = 2,
        pin_memory: bool = False,
        device_mesh: torch.distributed.DeviceMesh | None = None,
        placements: dict[str, torch.distributed.tensor.Placement] | None = None,
        consumer_stream: torch.cuda.Stream | None = None,
    ) -> None:
        if isinstance(data_dir, str):
            data_dir = pathlib.Path(data_dir)

        # Verify the data directory exists:
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")

        # Verify the data directory is a directory:
        if not data_dir.is_dir():
            raise NotADirectoryError(f"Data directory {data_dir} is not a directory")

        self._keys_to_read = keys_to_read

        # Make sure the optional keys are on the right device:
        self._keys_to_read_if_available = {
            k: v.to(output_device) for k, v in keys_to_read_if_available.items()
        }

        self.file_reader, self._filenames = self._infer_file_type_and_filenames(
            data_dir
        )

        self.pin_memory = pin_memory

        # Check the file names; some can be read well in parallel, while others
        # are not parallelizable.

        self._length = len(self._filenames)

        self.output_device = output_device
        if output_device.type == "cuda":
            self._data_loader_stream = torch.cuda.Stream()
        else:
            self._data_loader_stream = None

        self.device_mesh = device_mesh
        self.placements = placements
        # This tracks global tensor info
        # so we can convert to ShardTensor at the right time.
        self.shard_spec = {}

        if self.device_mesh is not None:
            if self.device_mesh.ndim != 1:
                raise ValueError("Device mesh must be one dimensional")

        # This is thread storage for data preloading:
        self._preload_queue = {}
        self._transfer_events = {}
        self.preload_depth = preload_depth
        self.preload_executor = ThreadPoolExecutor(max_workers=max(1, preload_depth))

        if consumer_stream is None and self.output_device.type == "cuda":
            consumer_stream = torch.cuda.current_stream()

        self.consumer_stream = consumer_stream

    def set_indices(self, indices: list[int]):
        """
        Set the indices for the dataset for this epoch.
        """

        # TODO - this needs to block while anything is in the preprocess queue.

        self.indices = indices

    def idx_to_index(self, idx):
        if hasattr(self, "indices"):
            return self.indices[idx]

        return idx

    def _infer_file_type_and_filenames(
        self, data_dir: pathlib.Path
    ) -> tuple[str, list[str]]:
        """
        Infer the file type and filenames from the data directory.
        """

        # We validated the directory exists and is a directory already.

        # List the files:
        files = list(data_dir.iterdir())

        # Initialize the file reader object
        # Note that for some of these, they could be functions
        # But others benefit from having a state, so we use classes:

        if all(file.suffix == ".npy" for file in files):
            file_reader = NpyFileReader(
                self._keys_to_read, self._keys_to_read_if_available
            )
            return file_reader, files
        elif all(file.suffix == ".npz" for file in files):
            file_reader = NpzFileReader(
                self._keys_to_read, self._keys_to_read_if_available
            )
            return file_reader, files
        elif all(file.suffix == ".zarr" and file.is_dir() for file in files):
            if TENSORSTORE_AVAILABLE:
                file_reader = TensorStoreZarrReader(
                    self._keys_to_read, self._keys_to_read_if_available
                )
            else:
                file_reader = ZarrFileReader(
                    self._keys_to_read, self._keys_to_read_if_available
                )
            return file_reader, files
        elif all(is_vtk_directory(file) for file in files):
            file_reader = VTKFileReader(
                self._keys_to_read, self._keys_to_read_if_available
            )
            return file_reader, files
            # Each "file" here is a directory of .vtp, stl, etc.
        else:
            # TODO - support folders of stl, vtp, vtu.
            raise ValueError(f"Unsupported file type: {files[0]}")

    def _move_to_gpu(
        self, data: dict[str, torch.Tensor], idx: int
    ) -> dict[str, torch.Tensor]:
        """Convert numpy arrays to torch tensors and move to GPU if available.

        Args:
            data: Dictionary of key to torch tensor.

        Returns:
            Dictionary of key to torch tensor on GPU if available.
        """

        if self.output_device.type != "cuda":
            return data

        result = {}

        with torch.cuda.stream(self._data_loader_stream):
            for key in data.keys():
                if data[key].device == self.output_device:
                    result[key] = data[key]
                    continue
                if self.pin_memory:
                    result[key] = (
                        data[key].pin_memory().to(self.output_device, non_blocking=True)
                    )
                else:
                    result[key] = data[key].to(self.output_device, non_blocking=True)
                # Move to GPU if available
                # result[key] = data[key].to(self.output_device, non_blocking=True)
                result[key].record_stream(self.consumer_stream)

        # Mark the consumer stream:
        transfer_event = torch.cuda.Event()
        transfer_event.record(self._data_loader_stream)
        self._transfer_events[idx] = transfer_event

        return result

    def _convert_to_shard_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        filename: str,
    ) -> dict[str, ShardTensor]:
        """Convert tensors to ShardTensor objects for distributed training.

        Args:
            tensors: Dictionary of key to torch tensor.

        Returns:
            Dictionary of key to torch tensor or ShardTensor.
        """

        if self.device_mesh is None:
            return tensors

        spec_dict = self.shard_spec.pop(filename)
        result = {}
        for key in tensors.keys():
            placement, chunk_sizes = spec_dict[key]

            result[key] = ShardTensor.from_local(
                local_tensor=tensors[key],
                device_mesh=self.device_mesh,
                placements=placement,
                sharding_shapes=chunk_sizes,
            )

        return result

    def preload(self, idx: int) -> None:
        """
        Asynchronously preload the data for the given index (up to CPU, not GPU).
        Only one preload operation is supported at a time.

        Args:
            idx: Index of the sample to preload.
        """
        if idx in self._preload_queue:
            # Skip items that are already in the queue
            return

        def _preload_worker():
            data = self._read_file(self._filenames[idx])
            if "stl_faces" in data:
                data["stl_faces"] = data["stl_faces"].to(torch.int32)
            # Convert to torch tensors
            return self._move_to_gpu(data, idx)

        self._preload_queue[idx] = self.preload_executor.submit(_preload_worker)

    def get_preloaded(self, idx: int) -> dict[str, torch.Tensor] | None:
        """
        Retrieve the preloaded data (blocking if not ready).

        Returns:
            (idx, data) tuple where data is a dictionary of key to numpy array or torch tensor.

        Raises:
            RuntimeError: If no preload is in progress.
            Exception: If preload failed.
        """

        if idx not in self._preload_queue:
            return None

        result = self._preload_queue[
            idx
        ].result()  # This will block until the result is ready
        self._preload_queue.pop(idx)  # Clear the future after getting the result

        return result

    def __iter__(self):
        # When starting the iterator method, start loading the data
        # at idx = 0, idx = 1
        # Start preprocessing at idx = 0, when the load completes

        self.i = 0

        N = len(self.indices) if hasattr(self, "indices") else len(self)
        for i in range(self.preload_depth):
            # Trigger the dataset to start loading index 0:
            if N > i + 1:
                self.preload(self.idx_to_index(self.i + i))

        return self

    def __next__(self):
        N = len(self.indices) if hasattr(self, "indices") else len(self._filenames)

        # Iteration bounds are based on the counter, not the random-access index
        if self.i >= N:
            self.i = 0
            raise StopIteration

        # This is the file random access index
        target_index = self.idx_to_index(self.i)

        # Before returning, put the next two target indexes into the queue:
        for preload_i in range(self.preload_depth):
            next_iteration_index = self.i + preload_i + 1
            if N > next_iteration_index:
                preload_idx = self.idx_to_index(next_iteration_index)
                self.preload(preload_idx)

        # Send up the random-access data:
        data = self.__getitem__(target_index)

        self.i += 1

        return data

    def __len__(self):
        return len(self._filenames)

    def _read_file(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
        """
        Read a file and return a dictionary of tensors.
        """
        if self.device_mesh is not None:
            tensor_dict, spec_dict = self.file_reader.read_file_sharded(
                filename, self.device_mesh
            )
            self.shard_spec[filename] = spec_dict
            return tensor_dict
        else:
            return self.file_reader.read_file(filename)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | ShardTensor]:
        """
        Get a data sample.

        Flow is:
        - Read data, or get preloaded data if this idx is preloaded.
        - Move data to GPU, if needed.
            - Preloading data will move to GPU if it can.
        - If domain parallelism is enabled, convert to ShardTensors.
        - Return

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing tensors/ShardTensors for the requested data
        """

        if idx >= len(self._filenames):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self._filenames)}"
            )

        # Attempt to get preloaded data:
        data = self.get_preloaded(idx)
        if data is None:
            # Read data from zarr file
            data = self._read_file(self._filenames[idx])
            data = self._move_to_gpu(data, idx)

        # This blocks until the preprocessing has transferred to GPU
        if idx in self._transfer_events:
            self.consumer_stream.wait_event(self._transfer_events[idx])
            self._transfer_events.pop(idx)

        # Convert to ShardTensors if using domain parallelism
        if self.device_mesh is not None:
            data = self._convert_to_shard_tensors(data, self._filenames[idx])

        return data

    def set_volume_sampling_size(self, volume_sampling_size: int):
        """
        Set the volume sampling size.  When set, the readers will
        assume the volumetric data is shuffled on disk and read only
        contiguous chunks of the data up to the sampling size.

        Args:
            volume_sampling_size: The total size of the volume sampling.
        """
        self.file_reader.set_volume_sampling_size(volume_sampling_size)

    def close(self):
        """
        Explicitly close the dataset and cleanup resources, including the ThreadPoolExecutor.
        """
        if hasattr(self, "preload_executor") and self.preload_executor is not None:
            self.preload_executor.shutdown(wait=True)
            self.preload_executor = None

    def __del__(self):
        """
        Cleanup resources when the dataset is destroyed.
        """
        self.close()


def compute_mean_std_min_max(
    dataset: CAEDataset, field_keys: list[str], max_samples: int = 20
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the mean, standard deviation, minimum, and maximum for a specified field
    across all samples in a dataset.

    Uses a numerically stable online algorithm for mean and variance.

    Args:
        dataset (CAEDataset): The dataset to process.
        field_key (str): The key for the field to normalize.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            mean, std, min, max tensors for the field.
    """
    N = {}
    mean = {}
    M2 = {}  # Sum of squares of differences from the current mean
    min_val = {}
    max_val = {}

    # Read the first data item to get the shapes:
    example_data = dataset[0]

    # Create placeholders for the accumulators:
    for key in field_keys:
        N[key] = torch.zeros(1, dtype=torch.int64, device=example_data[key].device)
        mean[key] = torch.zeros(
            example_data[key].shape[-1],
            device=example_data[key].device,
            dtype=torch.float64,
        )
        M2[key] = torch.zeros(
            example_data[key].shape[-1],
            device=example_data[key].device,
            dtype=torch.float64,
        )
        min_val[key] = torch.full(
            (example_data[key].shape[-1],),
            float("inf"),
            device=example_data[key].device,
        )
        max_val[key] = torch.full(
            (example_data[key].shape[-1],),
            float("-inf"),
            device=example_data[key].device,
        )

    global_start = time.perf_counter()
    start = time.perf_counter()
    data_list = np.arange(len(dataset))
    np.random.shuffle(data_list)
    for i, j in enumerate(data_list):
        data = dataset[j]
        if i >= max_samples:
            break

        for field_key in field_keys:
            field_data = data[field_key]

            # Compute batch statistics
            batch_mean = field_data.mean(axis=(0))
            batch_M2 = ((field_data - batch_mean) ** 2).sum(axis=(0))
            batch_n = field_data.shape[0]

            # Update running mean and M2 (Welford's algorithm)
            delta = batch_mean - mean[field_key]
            N[field_key] += batch_n  # batch_n should also be torch.int64
            mean[field_key] = mean[field_key] + delta * (batch_n / N[field_key])
            M2[field_key] = (
                M2[field_key]
                + batch_M2
                + delta**2 * (batch_n * N[field_key]) / N[field_key]
            )

        end = time.perf_counter()
        iteration_time = end - start
        print(
            f"on iteration {i} of {max_samples}, time: {iteration_time:.2f} seconds for file: {j}"
        )
        start = time.perf_counter()

    var = {}
    std = {}
    for field_key in field_keys:
        var[field_key] = M2[field_key] / (
            N[field_key].item() - 1
        )  # Convert N to Python int for division
        std[field_key] = torch.sqrt(var[field_key])

    start = time.perf_counter()
    for i, j in enumerate(data_list):
        data = dataset[j]
        if i >= max_samples:
            break

        for field_key in field_keys:
            field_data = data[field_key]

            batch_n = field_data.shape[0]

            # # Update min/max

            mean_sample = mean[field_key]
            std_sample = std[field_key]
            mask = torch.ones_like(field_data, dtype=torch.bool)
            for v in range(field_data.shape[-1]):
                outliers = (field_data[:, v] < mean_sample[v] - 9.0 * std_sample[v]) | (
                    field_data[:, v] > mean_sample[v] + 9.0 * std_sample[v]
                )
                mask[:, v] = ~outliers

            batch_min = []
            batch_max = []
            for v in range(field_data.shape[-1]):
                batch_min.append(field_data[mask[:, v], v].min())
                batch_max.append(field_data[mask[:, v], v].max())

            batch_min = torch.stack(batch_min)
            batch_max = torch.stack(batch_max)

            min_val[field_key] = torch.minimum(min_val[field_key], batch_min)
            max_val[field_key] = torch.maximum(max_val[field_key], batch_max)

        end = time.perf_counter()
        iteration_time = end - start
        print(
            f"on iteration {i} of {max_samples}, time: {iteration_time:.2f} seconds for file: {j}"
        )
        start = time.perf_counter()

    global_end = time.perf_counter()
    global_time = global_end - global_start

    print(f"Total time: {global_time:.2f} seconds for {max_samples} samples")

    return mean, std, min_val, max_val
