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

"""
This file tests the `radius_search` function from physicsnemo when
applied on sharded tensors.

This has a departure from the usual structure of tests.  The radius_search
function does not guarantee any ordering of it's output points nor which
points will be returned, if there are more points inside a radius than requested.

Here, we are manually checking the points agree with two tweaks: first, the
number of input points requested is always more than the number of points
available.  Second, the points are sorted before making a comparison.
"""

import pytest
import torch

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.domino.model import BQWarp
from physicsnemo.utils.version_check import check_module_requirements

try:
    check_module_requirements("physicsnemo.distributed.shard_tensor")

except ImportError:
    pytest.skip(
        "Skipping test because physicsnemo.distributed.shard_tensor is not available",
        allow_module_level=True,
    )


from torch.distributed.tensor import distribute_module  # noqa: E402
from torch.distributed.tensor.placement_types import (  # noqa: E402
    Replicate,
    Shard,
)

from physicsnemo.distributed import (
    scatter_tensor,
)


def convert_input_dict_to_shard_tensor(
    input_dict, point_placements, grid_placements, mesh
):
    # Strategy: convert the point clouds to replicated tensors, and
    # grid objects to sharded tensors
    non_sharded_keys = [
        "surface_min_max",
        "volume_min_max",
        "stream_velocity",
        "air_density",
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


def run_radius_search_module(model, data_dict, reverse_mapping):
    geo_centers = data_dict["geometry_coordinates"]

    # Bounding box grid
    s_grid = data_dict["surf_grid"]

    # Scaling factors
    surf_max = data_dict["surface_min_max"][:, 1]
    surf_min = data_dict["surface_min_max"][:, 0]

    # Normalize based on BBox around surface (car)
    geo_centers_surf = 2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1

    mapping, outputs = model(geo_centers_surf, s_grid, reverse_mapping)

    return mapping, outputs


@pytest.mark.multigpu_static
@pytest.mark.parametrize("shard_points", [True, False])
@pytest.mark.parametrize("shard_grid", [True, False])
@pytest.mark.parametrize("reverse_mapping", [True])
def test_sharded_radius_search_layer_forward(
    distributed_mesh, shard_points, shard_grid, reverse_mapping
):
    dm = DistributedManager()

    device = dm.device

    # Create the input dict:
    bsize = 1
    npoints = 8 * 17
    nx, ny, nz = 8 * 12, 6, 4
    # This is pretty aggressive, it'd never actually be this many.
    # But it enables checking the ring ball query deterministically.
    if reverse_mapping:
        num_neigh = npoints
    else:
        num_neigh = nx * ny * nz
    geom_centers = torch.randn(bsize, npoints, 3, device=device)
    surf_grid = torch.randn(bsize, nx, ny, nz, 3, device=device)
    surf_grid_max_min = torch.randn(bsize, 2, 3, device=device)
    input_dict = {
        "geometry_coordinates": geom_centers,
        "surf_grid": surf_grid,
        "surface_min_max": surf_grid_max_min,
    }

    # Define the sharding placements:
    point_placement = (Shard(1),) if shard_points else (Replicate(),)
    grid_placement = (Shard(1),) if shard_grid else (Replicate(),)

    # Convert the input dict to sharded tensors:
    sharded_input_dict = convert_input_dict_to_shard_tensor(
        input_dict, point_placement, grid_placement, distributed_mesh
    )

    # Get the single_gpu input_dict again, but now it's identical on all GPUs
    input_dict = {key: value.full_tensor() for key, value in sharded_input_dict.items()}

    # Create the model:
    model = BQWarp(
        grid_resolution=[nx, ny, nz],
        radius=1.0,
        neighbors_in_radius=num_neigh,
    ).to(device)

    single_gpu_mapping, single_gpu_outputs = run_radius_search_module(
        model, input_dict, reverse_mapping=reverse_mapping
    )

    # Convert the model to a distributed model:
    # Since the model has no parameters, this might not be necessary.
    model = distribute_module(model, device_mesh=distributed_mesh)

    sharded_mapping, sharded_outputs = run_radius_search_module(
        model, sharded_input_dict, reverse_mapping
    )

    # This ball query function is tricky - we may or may not preserve order.
    # To ensure the mapping is correct, we take the sorted values
    # along the point dimension and compare.

    sorted_single_gpu_mapping, sorted_single_gpu_mapping_indices = torch.sort(
        single_gpu_mapping, dim=-1, descending=True
    )
    sorted_sharded_mapping, sorted_sharded_mapping_indices = torch.sort(
        sharded_mapping.full_tensor(), dim=-1, descending=True
    )

    assert torch.allclose(sorted_single_gpu_mapping, sorted_sharded_mapping)

    # To check the outputs, we apply the sorted indexes into the outputs
    # and validate the sorted version.

    # Apply the sort to the output tensors too:
    single_gpu_output_sort_indices = sorted_single_gpu_mapping_indices.unsqueeze(
        -1
    ).expand(-1, -1, -1, sharded_outputs.shape[-1])
    sorted_single_gpu_outputs = single_gpu_outputs.gather(
        2, index=single_gpu_output_sort_indices
    )

    sharded_output_sort_indices = sorted_sharded_mapping_indices.unsqueeze(-1).expand(
        -1, -1, -1, sharded_outputs.shape[-1]
    )
    sorted_sharded_outputs = sharded_outputs.full_tensor().gather(
        2, index=sharded_output_sort_indices
    )

    assert torch.allclose(sorted_single_gpu_outputs, sorted_sharded_outputs)

    if reverse_mapping:
        correct_placement = grid_placement
    else:
        correct_placement = point_placement

    mapping_placement_correct = sharded_mapping._spec.placements == correct_placement
    sharded_outputs_placement_correct = sharded_outputs.placements == correct_placement

    assert mapping_placement_correct
    assert sharded_outputs_placement_correct
