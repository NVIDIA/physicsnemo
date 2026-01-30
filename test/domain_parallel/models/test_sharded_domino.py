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

from dataclasses import dataclass
from typing import Sequence

import pytest
import torch
from torch.distributed.tensor import distribute_module
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.models.domino import DoMINO


@dataclass
class model_params:
    @dataclass
    class geometry_rep:
        @dataclass
        class geo_conv:
            base_neurons: int = 32
            base_neurons_in: int = 1
            base_neurons_out: int = 1
            surface_hops: int = 1
            volume_hops: int = 1
            volume_radii: Sequence = (0.1, 0.5, 1.0, 2.5)
            volume_neighbors_in_radius: Sequence = (32, 64, 128, 256)
            surface_radii: Sequence = (0.01, 0.05, 1.0)
            surface_neighbors_in_radius: Sequence = (8, 16, 128)
            activation: str = "gelu"
            fourier_features: bool = False
            num_modes: int = 5

        @dataclass
        class geo_processor:
            base_filters: int = 8
            activation: str = "gelu"
            processor_type: str = "conv"
            self_attention: bool = False
            cross_attention: bool = False
            volume_sdf_scaling_factor: Sequence = (0.04,)
            surface_sdf_scaling_factor: Sequence = (0.01, 0.02, 0.04)

        base_filters: int = 8
        geo_conv = geo_conv
        geo_processor = geo_processor

    @dataclass
    class geometry_local:
        base_layer: int = 512
        volume_neighbors_in_radius: Sequence = (64, 128)
        surface_neighbors_in_radius: Sequence = (32, 128)
        volume_radii: Sequence = (0.1, 0.25)
        surface_radii: Sequence = (0.05, 0.25)

    @dataclass
    class nn_basis_functions:
        base_layer: int = 512
        fourier_features: bool = True
        num_modes: int = 5
        activation: str = "gelu"

    @dataclass
    class local_point_conv:
        activation: str = "gelu"

    @dataclass
    class aggregation_model:
        base_layer: int = 512
        activation: str = "gelu"

    @dataclass
    class position_encoder:
        base_neurons: int = 512
        activation: str = "gelu"
        fourier_features: bool = True
        num_modes: int = 5

    @dataclass
    class parameter_model:
        base_layer: int = 512
        fourier_features: bool = False
        num_modes: int = 5
        activation: str = "gelu"

    model_type: str = "combined"
    activation: str = "gelu"
    interp_res: Sequence = (128, 64, 64)
    use_sdf_in_basis_func: bool = True
    positional_encoding: bool = False
    surface_neighbors: bool = True
    num_neighbors_surface: int = 7
    num_neighbors_volume: int = 10
    use_surface_normals: bool = True
    use_surface_area: bool = True
    encode_parameters: bool = False
    combine_volume_surface: bool = False
    geometry_encoding_type: str = "both"
    solution_calculation_mode: str = "two-loop"
    geometry_rep = geometry_rep
    nn_basis_functions = nn_basis_functions
    aggregation_model = aggregation_model
    position_encoder = position_encoder
    geometry_local = geometry_local


def generate_synthetic_data(shard_grid, shard_points, npoints=100):
    """
    Generate synthetic data for the DoMINO model.
    Args:
        shard_grid: Whether to shard the grid.
        shard_points: Whether to shard the points.
        npoints: Number of points.
    Returns:
        input_dict: Dictionary of input tensors.
    """
    dm = DistributedManager()

    bsize = 1
    nx, ny, nz = 128, 64, 64
    num_neigh = 7
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

    # Construct DoMINO model
    model = DoMINO(
        input_features=3,
        output_features_vol=5,
        output_features_surf=4,
        model_parameters=model_params(),
    ).to(dm.device)

    npoints = 500

    # Create data:
    input_dict = generate_synthetic_data(shard_grid, shard_points, npoints)

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
