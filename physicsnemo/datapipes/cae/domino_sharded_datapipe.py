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

from dataclasses import asdict

import torch

from physicsnemo.utils.version_check import check_module_requirements

from .domino_datapipe import DoMINODataPipe

# Prevent importing this module if the minimum version of pytorch is not met.
check_module_requirements("physicsnemo.distributed.shard_tensor")

from torch.distributed.tensor.placement_types import (  # noqa: E402
    Replicate,
    Shard,
)

from physicsnemo.distributed.shard_tensor import ShardTensor  # noqa: E402


class ShardedDoMINODataPipe(DoMINODataPipe):
    """
    An extension of the DoMINODataPipe for domain parallel training.

    How this works:
    1. the preprocessing is done in cupy or numpy in the base class, which we
       want to keep.
    2. Dataloading is done on one file per idx in __getitem__.  For sharded data,
       we want to load one file per mesh and shard or replicate the data as needed.
    3. The sharding can be either on the grid or the point clouds.  We shard the grids
       after loading point data, so data loading only worries about the point clouds.
    4. For numpy files (.npz, .npy), each rank reads the whole file and takes only
       the data it needs, in the end.  Because data loading is the bulk of the time,
       this preprocesses everything independently and then shards.
    5. For Zarr files, each rank can read slices of the data independently.  So
       infer the chunk size, based on the number of ranks in the mesh and sharding,
       and then read the right slice.
    6. For some of the pipeline, we need the full data.  So it gets gathered locally.
    7. After preprocessing, the data is chunked into appropriate shards and sent out.
    8. This file provides a wrapper function for the collate function (like a decorator)
       that will turn appropriate cupy into tensors and then into shard tensors.

    """

    def __init__(
        self,
        input_path,
        model_type,
        domain_mesh,
        shard_point_cloud,
        shard_grid,
        **config_overrides,
    ):

        # if 'gpu_output' not in config_overrides:
        config_overrides["gpu_output"] = True

        # First, initialize the super class.
        super().__init__(
            input_path,
            model_type,
            **config_overrides,
        )

        self.domain_mesh = domain_mesh

        self.shard_point_cloud = shard_point_cloud
        self.shard_grid = shard_grid

        # These are keys that are point-like
        self.point_cloud_keys = [
            "volume_fields",
            "pos_volume_closest",
            "pos_volume_center_of_mass",
            "pos_surface_center_of_mass",
            "geometry_coordinates",
            "surface_mesh_centers",
            "surface_mesh_neighbors",
            "sdf_nodes",
            "surface_normals",
            "surface_neighbors_normals",
            "surface_areas",
            "surface_neighbors_areas",
            "volume_mesh_centers",
            "surface_fields",
        ]

        # These keys are grid-like
        self.grid_keys = [
            "grid",
            "surf_grid",
            "sdf_grid",
            "sdf_surf_grid",
        ]

        # These keys are scalar-like and should never be sharded
        self.scalar_keys = [
            "global_params_values",
            "global_params_reference",
            "surface_min_max",
            "volume_min_max",
            "length_scale",
        ]

    def __getitem__(self, idx):

        single_dict = super().__getitem__(idx)

        # Here, we're assuming that the data is already replicated.
        # Turn all the pieces of the dict into ShardTensors with that placement.
        default_placement = [
            Replicate(),
        ]
        for key, value in single_dict.items():
            if isinstance(value, torch.Tensor):
                single_dict[key] = ShardTensor.from_local(
                    value, self.domain_mesh, default_placement
                )

        # # Now, shard the data.
        sharding = [
            Shard(0),
        ]
        if self.shard_point_cloud:
            for key in self.point_cloud_keys:
                if key in single_dict:
                    single_dict[key] = single_dict[key].redistribute(
                        placements=sharding
                    )

        if self.shard_grid:
            for key in self.grid_keys:
                if key in single_dict:
                    single_dict[key] = single_dict[key].redistribute(
                        placements=sharding
                    )

        return single_dict


def create_sharded_domino_dataset(
    base_dataset,
    domain_mesh,
    shard_point_cloud,
    shard_grid,
):

    # Pull off the data path, model type, and config_dict:
    data_path = base_dataset.config.data_path
    model_type = base_dataset.model_type
    config_dict = asdict(base_dataset.config)

    # Make sure the input path is not included in the config_dict:
    config_dict.pop("data_path")

    # Use the configuration of the base dataset to create a sharded dataset:
    return ShardedDoMINODataPipe(
        input_path=data_path,
        model_type=model_type,
        domain_mesh=domain_mesh,
        shard_point_cloud=shard_point_cloud,
        shard_grid=shard_grid,
        **config_dict,
    )
