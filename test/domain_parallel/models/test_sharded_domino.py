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

"""Domain-parallel test for DoMINO.

DoMINO's parameters are all replicated, so it is distributed with DDP; the grids
and point clouds are ``ShardTensor``s sharded along the point/grid axis (with a
few genuinely global tensors replicated). ShardTensor auto-promotion lets the
plain weights meet the sharded activations. The shared harness runs the
distributed forward/backward against a single-GPU reference.
"""

import copy

import pytest
import torch
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.models.domino import DoMINO
from physicsnemo.models.domino.config import DEFAULT_MODEL_PARAMS
from test.domain_parallel.models.harness import (
    DomainParallelModelCase,
    run_domain_parallel_model_check,
)

# Conv processor for faster tests; same shapes as DEFAULT_MODEL_PARAMS.
_DOMINO_TEST_CONFIG = copy.deepcopy(DEFAULT_MODEL_PARAMS)
_DOMINO_TEST_CONFIG.geometry_rep.geo_processor.processor_type = "conv"

_NPOINTS = 500

# Point/grid tensors are sharded along dim 1; these are genuinely global.
_NON_SHARDED_KEYS = [
    "volume_min_max",
    "surface_min_max",
    "global_params_reference",
    "global_params_values",
]


def generate_synthetic_data(npoints=_NPOINTS, config=None):
    """Generate synthetic (full, plain) input tensors for the DoMINO model."""
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
    """Shard the point clouds and grids; replicate the genuinely global tensors."""
    sharded_dict = {}
    for key, value in input_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key in _NON_SHARDED_KEYS:
            placements = (Replicate(),)
        elif "grid" in key:
            placements = grid_placements
        else:
            placements = point_placements
        sharded_dict[key] = scatter_tensor(
            value,
            0,
            mesh,
            placements,
            global_shape=value.shape,
            dtype=value.dtype,
            requires_grad=value.requires_grad,
        )
    return sharded_dict


def _build_domino(device):
    return DoMINO(
        input_features=3,
        output_features_vol=5,
        output_features_surf=4,
        model_parameters=_DOMINO_TEST_CONFIG,
    ).to(device)


def _build_inputs(device):
    return (generate_synthetic_data(_NPOINTS, config=_DOMINO_TEST_CONFIG),), {}


def _shard_inputs(args, kwargs, mesh):
    (input_dict,) = args
    sharded = convert_input_dict_to_shard_tensor(
        input_dict, (Shard(1),), (Shard(1),), mesh
    )
    return (sharded,), kwargs


def _check_output(d_out):
    volume_predictions, surface_predictions = d_out
    assert volume_predictions.shape == (1, _NPOINTS, 5)
    assert surface_predictions.shape == (1, _NPOINTS, 4)
    # Outputs follow the point sharding.
    assert volume_predictions._spec.placements == (Shard(1),)
    assert surface_predictions._spec.placements == (Shard(1),)


def _loss_fn(output):
    """Scalar loss over both volume and surface predictions."""
    volume_predictions, surface_predictions = output
    return volume_predictions.mean() + surface_predictions.mean()


_CASE = DomainParallelModelCase(
    name="domino",
    build_model=_build_domino,
    build_inputs=_build_inputs,
    shard_inputs=_shard_inputs,
    output_check_fn=_check_output,
    loss_fn=_loss_fn,
)


@pytest.mark.multigpu_static
def test_domino_distributed(distributed_mesh):
    """Distributed forward/backward matches a single-GPU reference."""
    run_domain_parallel_model_check(_CASE, mesh=distributed_mesh)
