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

import copy

import pytest
import torch
from torch.distributed.tensor import distribute_module
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.models.domino import DoMINO
from physicsnemo.models.domino.config import DEFAULT_MODEL_PARAMS

# Conv processor for faster tests; same shapes as DEFAULT_MODEL_PARAMS.
_DOMINO_TEST_CONFIG = copy.deepcopy(DEFAULT_MODEL_PARAMS)
_DOMINO_TEST_CONFIG.geometry_rep.geo_processor.processor_type = "conv"


def generate_synthetic_data(shard_grid, shard_points, npoints=100, config=None):
    """
    Generate synthetic data for the DoMINO model.
    Args:
        shard_grid: Whether to shard the grid.
        shard_points: Whether to shard the points.
        npoints: Number of points.
        config: DoMINO config for grid size and neighbor counts. Defaults to DEFAULT_MODEL_PARAMS.
    Returns:
        input_dict: Dictionary of input tensors.
    """
    if config is None:
        config = DEFAULT_MODEL_PARAMS
    dm = DistributedManager()

    bsize = 1
    nx, ny, nz = config.interp_res
    num_neigh = config.num_neighbors_surface
    global_features = 2

    device = dm.device

    pos_normals_closest_vol = torch.randn(bsize, npoints, 3).to(device)
    pos_normals_com_vol = torch.randn(bsize, npoints, 3).to(device)
    pos_normals_com_surface = torch.randn(bsize, npoints, 3).to(device)
    geom_centers = torch.randn(bsize, npoints, 3).to(device)
    grid = torch.randn(bsize, nx, ny, nz, 3).to(device)
    surf_grid = torch.randn(bsize, nx, ny, nz, 3).to(device)
    sdf_grid = torch.randn(bsize, nx, ny, nz).to(device)
    sdf_surf_grid = torch.randn(bsize, nx, ny, nz).to(device)
    sdf_nodes = torch.randn(bsize, npoints, 1).to(device)
    surface_coordinates = torch.randn(bsize, npoints, 3).to(device)
    surface_neighbors = torch.randn(bsize, npoints, num_neigh, 3).to(device)
    surface_normals = torch.randn(bsize, npoints, 3).to(device)
    surface_neighbors_normals = torch.randn(bsize, npoints, num_neigh, 3).to(device)
    surface_sizes = torch.rand(bsize, npoints).to(device)
    surface_neighbors_sizes = torch.rand(bsize, npoints, num_neigh).to(device)
    volume_coordinates = torch.randn(bsize, npoints, 3).to(device)
    vol_grid_max_min = torch.randn(bsize, 2, 3).to(device)
    surf_grid_max_min = torch.randn(bsize, 2, 3).to(device)
    global_params_values = torch.randn(bsize, global_features, 1).to(device)
    global_params_reference = torch.randn(bsize, global_features, 1).to(device)
    input_dict = {
        "pos_volume_closest": pos_normals_closest_vol,
        "pos_volume_center_of_mass": pos_normals_com_vol,
        "pos_surface_center_of_mass": pos_normals_com_surface,
        "geometry_coordinates": geom_centers,
        "grid": grid,
        "surf_grid": surf_grid,
        "sdf_grid": sdf_grid,
        "sdf_surf_grid": sdf_surf_grid,
        "sdf_nodes": sdf_nodes,
        "surface_mesh_centers": surface_coordinates,
        "surface_mesh_neighbors": surface_neighbors,
        "surface_normals": surface_normals,
        "surface_neighbors_normals": surface_neighbors_normals,
        "surface_areas": surface_sizes,
        "surface_neighbors_areas": surface_neighbors_sizes,
        "volume_mesh_centers": volume_coordinates,
        "volume_min_max": vol_grid_max_min,
        "surface_min_max": surf_grid_max_min,
        "global_params_reference": global_params_values,
        "global_params_values": global_params_reference,
    }

    return input_dict


def convert_input_dict_to_shard_tensor(
    input_dict, point_placements, grid_placements, mesh
):
    # Strategy: convert the point clouds to replicated tensors, and
    # grid objects to sharded tensors

    non_sharded_keys = [
        "volume_min_max",
        "surface_min_max",
        "global_params_reference",
        "global_params_values",
    ]

    sharded_dict = {}

    for key, value in input_dict.items():
        # Skip non-tensor values
        if not isinstance(value, torch.Tensor):
            continue

        # Skip keys that should not be sharded
        if key in non_sharded_keys:
            sharded_dict[key] = scatter_tensor(
                value,
                0,
                mesh,
                [
                    Replicate(),
                ],
                global_shape=value.shape,
                dtype=value.dtype,
                requires_grad=value.requires_grad,
            )
            continue

        if "grid" in key:
            sharded_dict[key] = scatter_tensor(
                value,
                0,
                mesh,
                grid_placements,
                global_shape=value.shape,
                dtype=value.dtype,
                requires_grad=value.requires_grad,
            )
        else:
            sharded_dict[key] = scatter_tensor(
                value,
                0,
                mesh,
                point_placements,
                global_shape=value.shape,
                dtype=value.dtype,
                requires_grad=value.requires_grad,
            )

    return sharded_dict


@pytest.mark.multigpu_static
@pytest.mark.parametrize(
    "shard_grid",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "shard_points",
    [
        True,
    ],
)
def test_domino_distributed(
    distributed_mesh,
    shard_grid,
    shard_points,
):
    """Test DoMINO distributed forward pass"""

    dm = DistributedManager()

    # Construct DoMINO model (conv processor for faster tests)
    model = DoMINO(
        input_features=3,
        output_features_vol=5,
        output_features_surf=4,
        model_parameters=_DOMINO_TEST_CONFIG,
    ).to(dm.device)

    npoints = 500

    # Create data:
    input_dict = generate_synthetic_data(
        shard_grid, shard_points, npoints, config=_DOMINO_TEST_CONFIG
    )

    # Scatter the data
    point_placements = (Shard(1),) if shard_points else (Replicate(),)
    grid_placements = (Shard(1),) if shard_grid else (Replicate(),)

    sharded_input_dict = convert_input_dict_to_shard_tensor(
        input_dict, point_placements, grid_placements, distributed_mesh
    )

    model = distribute_module(model, device_mesh=distributed_mesh)

    # Run model
    volume_predictions, surface_predictions = model(sharded_input_dict)

    # Check output
    assert volume_predictions.shape == (1, npoints, 5)
    assert surface_predictions.shape == (1, npoints, 4)

    # The outputs should always match the point sharding:
    assert volume_predictions._spec.placements == point_placements
    assert surface_predictions._spec.placements == point_placements
