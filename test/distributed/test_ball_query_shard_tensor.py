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


from pytest_utils import modify_environment  # noqa: E402
from test_shard_tensor_initialization import init_dist
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


def run_ball_query_module(model, data_dict, reverse_mapping):
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


def run_sharded_ball_query_layer_forward(
    rank, num_gpus, shard_points, shard_grid, reverse_mapping
):
    with modify_environment(
        RANK=f"{rank}",
        WORLD_SIZE=f"{num_gpus}",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(13245),
        LOCAL_RANK=f"{rank % torch.cuda.device_count()}",
    ):
        init_dist(rank, num_gpus)
        dm = DistributedManager()

        device = dm.device

        # Create the input dict:
        bsize = 1
        npoints = 37
        nx, ny, nz = 16, 6, 4
        # This is pretty aggressive, it'd never actually be this many.
        # But it enables checking the ring ball query deterministically.
        if reverse_mapping:
            num_neigh = npoints
        else:
            num_neigh = nx * ny * nz
        geom_centers = torch.randn(bsize, npoints, 3).to(device)
        surf_grid = torch.randn(bsize, nx, ny, nz, 3).to(device)
        surf_grid_max_min = torch.randn(bsize, 2, 3).to(device)
        input_dict = {
            "geometry_coordinates": geom_centers,
            "surf_grid": surf_grid,
            "surface_min_max": surf_grid_max_min,
        }

        # To make this work, we need to broadcast the input_dict to all GPUs
        # Easiest to shard it and then pull it together into a full_tensor on each GPU

        global_mesh = dm.initialize_mesh([-1], ["domain"])
        domain_mesh = global_mesh["domain"]

        # Define the sharding placements:
        point_placement = (Shard(1),) if shard_points else (Replicate(),)
        grid_placement = (Shard(1),) if shard_grid else (Replicate(),)

        # Convert the input dict to sharded tensors:
        sharded_input_dict = convert_input_dict_to_shard_tensor(
            input_dict, point_placement, grid_placement, domain_mesh
        )

        # Get the single_gpu input_dict again, but now it's identical on all GPUs
        input_dict = {
            key: value.full_tensor() for key, value in sharded_input_dict.items()
        }

        # Create the model:
        model = BQWarp(
            grid_resolution=[nx, ny, nz],
            radius=1.0,
            neighbors_in_radius=num_neigh,
        ).to(device)

        single_gpu_mapping, single_gpu_outputs = run_ball_query_module(
            model, input_dict, reverse_mapping=reverse_mapping
        )

        # Initialize a mesh:

        # Convert the model to a distributed model:
        # Since the model has no parameters, this might not be necessary.
        model = distribute_module(model, device_mesh=domain_mesh)

        sharded_mapping, sharded_outputs = run_ball_query_module(
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

        sharded_output_sort_indices = sorted_sharded_mapping_indices.unsqueeze(
            -1
        ).expand(-1, -1, -1, sharded_outputs.shape[-1])
        sorted_sharded_outputs = sharded_outputs.full_tensor().gather(
            2, index=sharded_output_sort_indices
        )

        assert torch.allclose(sorted_single_gpu_outputs, sorted_sharded_outputs)

        if reverse_mapping:
            correct_placement = grid_placement
        else:
            correct_placement = point_placement

        mapping_placement_correct = (
            sharded_mapping._spec.placements == correct_placement
        )
        sharded_outputs_placement_correct = (
            sharded_outputs.placements == correct_placement
        )

        assert mapping_placement_correct
        assert sharded_outputs_placement_correct

        DistributedManager().cleanup()


@pytest.mark.multigpu
@pytest.mark.timeout(120)
@pytest.mark.parametrize("shard_points", [True, False])
@pytest.mark.parametrize("shard_grid", [True, False])
@pytest.mark.parametrize("reverse_mapping", [True, False])
def test_shard_tensor_ball_query(shard_points, shard_grid, reverse_mapping):
    """
    This test is meant to ensure ShardTensor can be initialized correctly
    from local data.  Checks the following:

    """
    num_gpus = torch.cuda.device_count()

    if num_gpus < 2:
        pytest.skip("Not enough GPUs available for distributed tests")

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_sharded_ball_query_layer_forward,
        args=(num_gpus, shard_points, shard_grid, reverse_mapping),
        nprocs=num_gpus,
        join=True,
        daemon=True,
    )
