# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from dataclasses import dataclass
from typing import Tuple, Union
import fsspec

import numpy as np
import torch
import zarr

try:
    import nvidia.dali as dali
    import nvidia.dali.plugin.pytorch as dali_pth
except ImportError:
    raise ImportError(
        "DALI dataset requires NVIDIA DALI package to be installed. "
        + "The package can be installed at:\n"
        + "https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html"
    )

from physicsnemo.datapipes.datapipe import Datapipe
from physicsnemo.datapipes.meta import DatapipeMetaData

Tensor = torch.Tensor


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "SeqZarrDatapipe"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True


class SeqZarrDatapipe(Datapipe):
    """
    DALI data pipeline for loading sequences from a Zarr array.

    This data pipeline is designed to be general given a Zarr dataset.

    Parameters
    ----------
    file_mapping : fsspec.mapping.FSMap
        Fsspec file mapping (e.g. fsspec.get_mapper("s3://bucket/path"))
    batch_size : int, optional
        Batch size, by default 1
    num_steps : int, optional
        Number of steps to predict, by default 2
    shuffle : bool, optional
        Shuffle data, by default True
    device : Union[str, torch.device], optional
        Device to use, by default "cuda"
    process_rank : int, optional
        Process rank, by default 0
    world_size : int, optional
        World size, by default 1
    """

    def __init__(
        self,
        file_mapping: fsspec.mapping.FSMap,
        variables: list,
        batch_size: int = 1,
        num_steps: int = 2,
        shuffle: bool = True,
        device: Union[str, torch.device] = "cuda",
        process_rank: int = 0,
        world_size: int = 1,
        batch: bool = False,
        parallel: bool = True,
        num_threads: int = 2,
        prefetch_queue_depth: int = 2,
        py_num_workers: int = 1,
        py_start_method: str = "spawn",
    ):
        super().__init__(meta=MetaData())

        # Store parameters
        self.file_mapping = file_mapping
        self.variables = variables
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.shuffle = shuffle
        self.batch = batch
        self.parallel = parallel
        self.num_threads = num_threads
        self.prefetch_queue_depth = prefetch_queue_depth
        self.py_num_workers = py_num_workers
        self.py_start_method = py_start_method

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda" and device.index == None:
            device = torch.device("cuda:0")
        self.device = device

        # Set up parallel
        self.process_rank = process_rank
        self.world_size = world_size

        # Outputs of pipeline
        self.pipe_outputs = self.variables

        # Create pipeline
        self.pipe = self._create_pipeline()

    def _create_pipeline(self) -> dali.Pipeline:
        """Create DALI pipeline

        Returns
        -------
        dali.Pipeline
            DALI pipeline
        """
        pipe = dali.Pipeline(
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            prefetch_queue_depth=self.prefetch_queue_depth,
            py_num_workers=self.py_num_workers,
            device_id=self.device.index,
            py_start_method=self.py_start_method,
        )

        with pipe:
            # Zarr source
            source = SeqZarrSource(
                self.file_mapping,
                self.variables,
                num_steps=self.num_steps,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                process_rank=self.process_rank,
                world_size=self.world_size,
                batch=self.batch,
            )

            # Update length of dataset
            self.total_length = len(source) // self.batch_size

            # Read current batch
            data = dali.fn.external_source(
                source,
                num_outputs=len(self.pipe_outputs),
                parallel=self.parallel,
                batch=self.batch,
                prefetch_queue_depth=self.prefetch_queue_depth,
                device="cpu",
            )

            if self.device.type == "cuda":
                # Move tensors to GPU as external_source won't do that
                data = [d.gpu() for d in data]

            # Set outputs
            pipe.set_outputs(*data)

        return pipe

    def __iter__(self):
        # Reset the pipeline before creating an iterator to enable epochs.
        self.pipe.reset()
        # Create DALI PyTorch iterator.
        return dali_pth.DALIGenericIterator([self.pipe], self.pipe_outputs)

    def __len__(self):
        return self.total_length


class SeqZarrSource:
    """
    DALI Source for loading a zarr array.
    The arrays will be indexed along the first dimension (usually time).

    Parameters
    ----------
    file_mapping : fsspec.mapping.FSMap
        Fsspec file mapping (e.g. fsspec.get_mapper("s3://bucket/path"))
    num_steps : int
        Number of steps to predict
    batch_size : int, optional
        Batch size, by default 1
    shuffle : bool, optional
        Shuffle data, by default True
    process_rank : int, optional
        Process rank, by default 0
    world_size : int, optional
        World size, by default 1
    batch : bool, optional
        Batch data, by default False
    """

    def __init__(
        self,
        file_mapping: fsspec.mapping.FSMap,
        variables: list,
        num_steps: int,
        batch_size: int = 1,
        shuffle: bool = True,
        process_rank: int = 0,
        world_size: int = 1,
        batch: bool = False,
    ):
        # Set up parameters
        self.file_mapping = file_mapping
        self.variables = variables
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch = batch

        # Check if all zarr arrays have the same first dimension
        _zarr_dataset = zarr.open(self.file_mapping, mode="r")
        self.first_dim = _zarr_dataset[variables[0]].shape[0]
        for variable in self.variables:
            if _zarr_dataset[variable].shape[0] != self.first_dim:
                raise ValueError("All zarr arrays must have the same first dimension.")

        # Get number of samples
        self.indices = np.arange(
            batch_size
            * world_size
            * ((self.first_dim - self.num_steps) // batch_size // world_size)
        )
        self.indices = np.array_split(self.indices, world_size)[process_rank]

        # Get number of full batches, ignore possible last incomplete batch for now.
        self.num_batches = len(self.indices) // self.batch_size

        # Set up last epoch
        self.last_epoch = None

        # Set zarr dataset
        self.zarr_dataset = None

        # Set call
        if self.batch:
            self._call = self._batch_call
            self.batch_mapping = np.stack(
                np.array_split(
                    self.indices[
                        : len(self.indices) - len(self.indices) % self.batch_size
                    ],
                    self.batch_size,
                ),
                axis=1,
            )
        else:
            self._call = self._sample_call

    def _batch_call(
        self,
        sample_info: dali.types.BatchInfo,
    ) -> Tuple[Tensor, Tensor, np.ndarray, np.ndarray, np.ndarray]:
        # Open Zarr dataset
        if self.zarr_dataset is None:
            self.zarr_dataset = zarr.open(self.file_mapping, mode="r")

        if sample_info >= self.batch_mapping.shape[0]:
            raise StopIteration()

        # Get batch indices
        batch_idx = self.batch_mapping[sample_info]
        time_idx = np.concatenate(
            [idx + np.arange(self.num_steps) for idx in batch_idx]
        )

        # Get data
        data = []

        # Get slices
        for i, variable in enumerate(self.variables):
            batch_data = self.zarr_dataset[variable][time_idx]
            data.append(
                np.reshape(
                    batch_data, (self.batch_size, self.num_steps, *batch_data.shape[1:])
                )
            )

        return tuple(data)

    def _sample_call(
        self,
        sample_info: dali.types.SampleInfo,
    ) -> Tuple[Tensor, Tensor, np.ndarray, np.ndarray, np.ndarray]:
        # Open Zarr dataset
        if self.zarr_dataset is None:
            self.zarr_dataset = zarr.open(self.file_mapping, mode="r")

        if sample_info.iteration >= self.num_batches:
            raise StopIteration()

        # Shuffle before the next epoch starts
        if self.shuffle and sample_info.epoch_idx != self.last_epoch:
            # All workers use the same rng seed so the resulting
            # indices are the same across workers
            np.random.default_rng(seed=sample_info.epoch_idx).shuffle(self.indices)
            self.last_epoch = sample_info.epoch_idx

        # Get local indices from global index
        idx = self.indices[sample_info.idx_in_epoch]

        # Make time indices
        time_idx = idx + np.arange(self.num_steps)

        # Get data
        data = []

        # Get slices
        for i, variable in enumerate(self.variables):
            data.append(self.zarr_dataset[variable][time_idx])

        return tuple(data)

    def __call__(
        self, sample_info: Union[dali.types.SampleInfo, dali.types.BatchInfo]
    ) -> Tuple[Tensor, Tensor, np.ndarray, np.ndarray, np.ndarray]:
        return self._call(sample_info)

    def __len__(self):
        if self.batch:
            return self.batch_mapping.shape[0] * self.batch_size
        else:
            return len(self.indices)
