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


import numpy as np
import torch
import vtk

try:
    import nvidia.dali as dali
    import nvidia.dali.plugin.pytorch as dali_pth
except ImportError:
    raise ImportError(
        "DALI dataset requires NVIDIA DALI package to be installed. "
        + "The package can be installed at:\n"
        + "https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html"
    )

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Union

from torch import Tensor

from modulus.datapipes.datapipe import Datapipe
from modulus.datapipes.meta import DatapipeMetaData

from .readers import read_cgns, read_vtp, read_vtu


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "MeshDatapipe"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True


class MeshDatapipe(Datapipe):
    """DALI data pipeline for mesh data

    Parameters
    ----------
    data_dir : str
        Directory where ERA5 data is stored
    variables : List[str, None]
        Ordered list of variables to be loaded from the files
    num_variables : int
        Number of variables to be loaded from the files
    file_format : str, optional
        File format of the data, by default "vtp"
        Supported formats: "vtp", "vtu", "cgns"
    stats_dir : Union[str, None], optional
        Directory where statistics are stored, by default None
        If provided, the statistics are used to normalize the attributes
    batch_size : int, optional
        Batch size, by default 1
    num_steps : int, optional
        Number of timesteps are included in the output variables, by default 1
    shuffle : bool, optional
        Shuffle dataset, by default True
    num_workers : int, optional
        Number of workers, by default 1
    device: Union[str, torch.device], optional
        Device for DALI pipeline to run on, by default cuda
    process_rank : int, optional
        Rank ID of local process, by default 0
    world_size : int, optional
        Number of training processes, by default 1
    cache_data : False, optional
        Whether to cache the data in memory for faster access in subsequent epochs, by default False
    """

    def __init__(
        self,
        data_dir: str,
        variables: List[str],
        num_variables: int,
        file_format: str = "vtp",
        stats_dir: Union[str, None] = None,
        batch_size: int = 1,
        num_samples: int = 1,
        shuffle: bool = True,
        num_workers: int = 1,
        device: Union[str, torch.device] = "cuda",
        process_rank: int = 0,
        world_size: int = 1,
        cache_data: bool = False,
    ):
        super().__init__(meta=MetaData())
        self.file_format = file_format
        self.variables = variables
        self.num_variables = num_variables
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.data_dir = Path(data_dir)
        self.stats_dir = Path(stats_dir) if stats_dir is not None else None
        self.num_samples = num_samples
        self.process_rank = process_rank
        self.world_size = world_size
        self.cache_data = cache_data

        # if self.batch_size > 1:
        #     raise NotImplementedError("Batch size greater than 1 is not supported yet")

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)
        # Need a index id if cuda
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        self.device = device

        # check root directory exists
        if not self.data_dir.is_dir():
            raise IOError(f"Error, data directory {self.data_dir} does not exist")

        self.parse_dataset_files()
        self.load_statistics()

        self.pipe = self._create_pipeline()

    def parse_dataset_files(self) -> None:
        """Parses the data directory for valid files and determines training samples

        Raises
        ------
        ValueError
            In channels specified or number of samples per year is not valid
        """
        # get all input data files
        match self.file_format:
            case "vtp":
                pattern = "*.vtp"
            case "vtu":
                pattern = "*.vtu"
            case "cgns":
                pattern = "*.cgns"
            case _:
                raise NotImplementedError(
                    f"Data type {self.file_format} is not supported yet"
                )

        self.data_paths = sorted(str(path) for path in self.data_dir.glob(pattern))

        for data_path in self.data_paths:
            self.logger.info(f"File found: {data_path}")
        self.total_samples = len(self.data_paths)

        if self.num_samples > self.total_samples:
            raise ValueError(
                "Number of requested samples is greater than the total number of available samples!"
            )
        self.logger.info(
            f"Total number of samples: {self.total_samples}, number of requested samples: {self.num_samples}"
        )

    def load_statistics(
        self,
    ) -> None:  # TODO generalize and combine with climate/era5_hdf5 datapipes
        """Loads statistics from pre-computed numpy files

        The statistic files should be of name global_means.npy and global_std.npy with
        a shape of [1, C] located in the stat_dir.

        Raises
        ------
        IOError
            If mean or std numpy files are not found
        AssertionError
            If loaded numpy arrays are not of correct size
        """
        # If no stats dir we just skip loading the stats
        if self.stats_dir is None:
            self.mu = None
            self.std = None
            return
        # load normalisation values
        mean_stat_file = self.stats_dir / Path("global_means.npy")
        std_stat_file = self.stats_dir / Path("global_stds.npy")

        if not mean_stat_file.exists():
            raise IOError(f"Mean statistics file {mean_stat_file} not found")
        if not std_stat_file.exists():
            raise IOError(f"Std statistics file {std_stat_file} not found")

        # has shape [1, C]
        self.mu = np.load(str(mean_stat_file))[:, 0 : self.num_variables]
        # has shape [1, C]
        self.sd = np.load(str(std_stat_file))[:, 0 : self.num_variables]

        if not self.mu.shape == self.sd.shape == (1, self.num_variables):
            raise AssertionError("Error, normalisation arrays have wrong shape")

    def _create_pipeline(self) -> dali.Pipeline:
        """Create DALI pipeline

        Returns
        -------
        dali.Pipeline
            Mesh DALI pipeline
        """
        pipe = dali.Pipeline(
            batch_size=self.batch_size,
            num_threads=2,
            prefetch_queue_depth=2,
            py_num_workers=self.num_workers,
            device_id=self.device.index,
            py_start_method="spawn",
        )

        with pipe:
            source = MeshDaliExternalSource(
                data_paths=self.data_paths,
                file_format=self.file_format,
                variables=self.variables,
                num_samples=self.num_samples,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                process_rank=self.process_rank,
                world_size=self.world_size,
                cache_data=self.cache_data,
            )
            # Update length of dataset
            self.length = len(source) // self.batch_size
            # Read current batch.
            vertices, attributes, edges = dali.fn.external_source(
                source,
                num_outputs=3,
                parallel=True,
                batch=False,
            )

            if self.device.type == "cuda":
                # Move tensors to GPU as external_source won't do that.
                vertices = vertices.gpu()
                attributes = attributes.gpu()
                edges = edges.gpu()

            # Normalize attributes if statistics are available.
            if self.stats_dir is not None:
                attributes = dali.fn.normalize(attributes, mean=self.mu, stddev=self.sd)

            # Set outputs.
            pipe.set_outputs(vertices, attributes, edges)

        return pipe

    def __iter__(self):
        # Reset the pipeline before creating an iterator to enable epochs.
        self.pipe.reset()
        # Create DALI PyTorch iterator.
        return dali_pth.DALIGenericIterator([self.pipe], ["vertices", "x", "edges"])

    def __len__(self):
        return self.length


class MeshDaliExternalSource:
    """DALI Source for lazy-loading with caching of mesh data

    Parameters
    ----------
    data_paths : Iterable[str]
        Directory where data is stored
    num_samples : int
        Total number of training samples
    batch_size : int, optional
        Batch size, by default 1
    shuffle : bool, optional
        Shuffle dataset, by default True
    process_rank : int, optional
        Rank ID of local process, by default 0
    world_size : int, optional
        Number of training processes, by default 1
    cache_data : False, optional
        Whether to cache the data in memory for faster access in subsequent epochs, by default False

    Note
    ----
    For more information about DALI external source operator:
    https://docs.nvidia.com/deeplearning/dali/archives/dali_1_13_0/user-guide/docs/examples/general/data_loading/parallel_external_source.html
    """

    def __init__(
        self,
        data_paths: Iterable[str],
        file_format: str,
        variables: List[str],
        num_samples: int,
        batch_size: int = 1,
        shuffle: bool = True,
        process_rank: int = 0,
        world_size: int = 1,
        cache_data: bool = False,
    ):
        self.data_paths = list(data_paths)
        self.file_format = file_format
        self.variables = variables
        # Will be populated later once each worker starts running in its own process.
        self.poly_data = None
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache_data = cache_data

        self.last_epoch = None

        self.indices = np.arange(num_samples)
        # Shard from indices if running in parallel
        self.indices = np.array_split(self.indices, world_size)[process_rank]

        # Get number of full batches, ignore possible last incomplete batch for now.
        # Also, DALI external source does not support incomplete batches in parallel mode.
        self.num_batches = len(self.indices) // self.batch_size

        self.mesh_reader_fn = self.mesh_reader()
        self.parse_vtk_data_fn = self.parse_vtk_data()

        if self.cache_data:
            # Make cache for the data
            self.data_cache = {}
            for data_path in self.data_paths:
                self.data_cache[data_path] = None

    def __call__(self, sample_info: dali.types.SampleInfo) -> Tuple[Tensor, Tensor]:
        if sample_info.iteration >= self.num_batches:
            raise StopIteration()

        # Shuffle before the next epoch starts.
        if self.shuffle and sample_info.epoch_idx != self.last_epoch:

            # All workers use the same rng seed so the resulting
            # indices are the same across workers.
            np.random.default_rng(seed=sample_info.epoch_idx).shuffle(self.indices)
            self.last_epoch = sample_info.epoch_idx

        # Get local indices from global index.
        idx = self.indices[sample_info.idx_in_epoch]

        # if self.poly_data is None:  # TODO check
        # This will be called once per worker. Workers are persistent,
        # so there is no need to explicitly close the files - this will be done
        # when corresponding pipeline/dataset is destroyed.
        if self.cache_data:
            processed_data = self.data_cache.get(self.data_paths[idx])
            if processed_data is None:
                data = self.mesh_reader_fn(self.data_paths[idx])
                processed_data = self.parse_vtk_data_fn(data, self.variables)
                self.data_cache[self.data_paths[idx]] = processed_data
        else:
            data = self.mesh_reader_fn(self.data_paths[idx])
            processed_data = self.parse_vtk_data_fn(data, self.variables)

        return processed_data

    def __len__(self):
        return len(self.indices)

    def mesh_reader(self):
        if self.file_format == "vtp":
            return read_vtp
        if self.file_format == "vtu":
            return read_vtu
        if self.file_format == "cgns":
            return read_cgns
        else:
            raise NotImplementedError(
                f"Data type {self.file_format} is not supported yet"
            )

    def parse_vtk_data(self):
        if self.file_format == "vtp":
            return _parse_vtk_polydata
        elif self.file_format in ["vtu", "cgns"]:
            return _parse_vtk_unstructuredgrid
        else:
            raise NotImplementedError(
                f"Data type {self.file_format} is not supported yet"
            )


def _parse_vtk_polydata(polydata, variables):
    # Fetch vertices
    points = polydata.GetPoints()
    if points is None:
        raise ValueError("Failed to get points from the polydata.")
    vertices = torch.tensor(
        np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())]),
        dtype=torch.float32,
    )

    # Fetch node attributes  # TODO modularize
    attributes = []
    point_data = polydata.GetPointData()
    if point_data is None:
        raise ValueError("Failed to get point data from the unstructured grid.")
    for array_name in variables:
        try:
            array = point_data.GetArray(array_name)
        except ValueError:
            raise ValueError(
                f"Failed to get array {array_name} from the unstructured grid."
            )
        array_data = np.zeros(
            (points.GetNumberOfPoints(), array.GetNumberOfComponents())
        )
        for j in range(points.GetNumberOfPoints()):
            array.GetTuple(j, array_data[j])
        attributes.append(torch.tensor(array_data, dtype=torch.float32))
    attributes = torch.cat(attributes, dim=-1)
    # TODO torch.cat is usually very inefficient when the number of items is large.
    # If possible, the resulting tensor should be pre-allocated and filled in during the loop.

    # Fetch edges
    polys = polydata.GetPolys()
    if polys is None:
        raise ValueError("Failed to get polygons from the polydata.")
    polys.InitTraversal()
    edges = []
    id_list = vtk.vtkIdList()
    for _ in range(polys.GetNumberOfCells()):
        polys.GetNextCell(id_list)
        num_ids = id_list.GetNumberOfIds()
        edges = [
            (id_list.GetId(j), id_list.GetId((j + 1) % num_ids)) for j in range(num_ids)
        ]
    edges = torch.tensor(edges, dtype=torch.long)

    return vertices, attributes, edges


def _parse_vtk_unstructuredgrid(grid, variables):
    # Fetch vertices
    points = grid.GetPoints()
    if points is None:
        raise ValueError("Failed to get points from the unstructured grid.")
    vertices = torch.tensor(
        np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())]),
        dtype=torch.float32,
    )

    # Fetch node attributes  # TODO modularize
    attributes = []
    point_data = grid.GetPointData()
    if point_data is None:
        raise ValueError("Failed to get point data from the unstructured grid.")
    for array_name in variables:
        try:
            array = point_data.GetArray(array_name)
        except ValueError:
            raise ValueError(
                f"Failed to get array {array_name} from the unstructured grid."
            )
        array_data = np.zeros(
            (points.GetNumberOfPoints(), array.GetNumberOfComponents())
        )
        for j in range(points.GetNumberOfPoints()):
            array.GetTuple(j, array_data[j])
        attributes.append(torch.tensor(array_data, dtype=torch.float32))
    if variables:
        attributes = torch.cat(attributes, dim=-1)
    else:
        attributes = torch.zeros((1,), dtype=torch.float32)

    # Return a dummy tensor of zeros for edges since they are not directly computable
    return (
        vertices,
        attributes,
        torch.zeros((0, 2), dtype=torch.long),
    )  # Dummy tensor for edges
