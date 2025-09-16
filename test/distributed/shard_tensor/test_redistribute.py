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
Test redistribution between different sharding schemes on ShardTensor.

One major feature of ShardTensor is that it knows both the global shape
and local layout of every shard, and can seamlessly translate between them.

In many ways, this is an extension of DTensor's utilities, but we're testing
here specifically any uneven shardings, etc.

The tests will test over 1d and 2d meshes of shard tensors, with increasingly
complex resharding requirements.  In all cases, the input tensors are sharded:
(Shard(1),) for 1d, (Shard(1), Shard(2)) for 2d.
"""

import pytest

from physicsnemo.utils.version_check import check_module_requirements

try:
    check_module_requirements("physicsnemo.distributed.shard_tensor")


except ImportError:
    pytest.skip(
        "Skipping test because physicsnemo.distributed.shard_tensor is not available",
        allow_module_level=True,
    )


import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager, ShardTensor


def shard_tensor_factory(mesh, requires_grad=False, uneven=True):
    """
    Generate a shard tensor on the mesh
    """

    dm = DistributedManager()

    local_shape = [
        100,
    ]

    min_size = 4

    if uneven:
        index_stride = 2

        # Using the same size per rank in mesh dimension
        for dim in range(mesh.ndim):
            dim_rank = dist.get_group_rank(mesh.get_group(dim), dm.rank)
            local_shape.append(
                (dim_rank + dim + 1) * min_size + dim_rank * index_stride
            )
    else:
        for dim in range(mesh.ndim):
            local_shape.append(min_size)  # noqa: PERF401

    local_shape.append(100)

    raw_data = torch.randn(
        local_shape,
        device=torch.device(f"cuda:{dm.local_rank}"),
        requires_grad=requires_grad,
    )

    placements = [Shard(1)]
    if mesh.ndim > 1:
        placements.append(Shard(2))

    st = ShardTensor.from_local(
        raw_data,
        device_mesh=mesh,
        placements=placements,
        sharding_shapes="infer",
    )

    return st


@pytest.mark.multigpu_static
@pytest.mark.parametrize(
    "redistribution_case",
    [
        ("S1", [Shard(1)]),  # This ought to be a no op!
        (
            "R",
            [
                Replicate(),
            ],
        ),  # Only triggers redistribution on first tensor dim.  gather_v
        (
            "S2",
            [
                Shard(2),
            ],
        ),  # Trigger sharding on to a *new* dimension all_to_all_v
    ],
)
def test_shard_tensor_redistribute1d(
    distributed_mesh, redistribution_case, verbose=True
):
    """Test redistribution between different sharding schemes"""
    run_shard_tensor_redistribute(
        distributed_mesh, redistribution_case, verbose=verbose
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize(
    "redistribution_case",
    [
        # Test cases for different redistribution scenarios
        ("S1+S2", [Shard(1), Shard(2)]),  # Should be a no op!
        (
            "R+S2",
            [Replicate(), Shard(2)],
        ),  # Only triggers redistribution on first tensor dim.  gather_v
        (
            "S1+R",
            [Shard(1), Replicate()],
        ),  # triggers S2-R on second tensor dim, gather_v
        (
            "R+R",
            [Replicate(), Replicate()],
        ),  # Triggers S2->R, S1-R.  gather_v then gather_v
        (
            "R+S1",
            [Replicate(), Shard(1)],
        ),  # triggers S2->R, S1->R, R->S2.  gather_v then gather_v then scatter_v
        (
            "S2+R",
            [Shard(2), Replicate()],
        ),  # Triggers S2->R, S2/S1 transpose.  gather_v then all_to_all_v
        (
            "S2+S1",
            [Shard(2), Shard(1)],
        ),  # Goes S2 -> R, S1 -> S2, R -> S1.  gather_v, all_to_all_v, scatter_v
        ("S3+R", [Shard(3), Replicate()]),  # Put the sharding on a new axis
    ],
)
def test_shard_tensor_redistribute2d(
    distributed_mesh_2d, redistribution_case, verbose=True
):
    run_shard_tensor_redistribute(
        distributed_mesh_2d, redistribution_case, verbose=verbose
    )


def run_shard_tensor_redistribute(mesh, redistribution_case, verbose=False):
    case_name, dst_placements = redistribution_case

    # Create initial sharded tensor
    shard_tensor = shard_tensor_factory(mesh)
    if verbose:
        print(f"Shard mesh is {shard_tensor._spec.mesh}")
        print(f"shard_tensor placements: {shard_tensor._spec.placements}")
        print(f"Target placements: {dst_placements}")
        print(f"shard_tensor shape: {shard_tensor.shape}")
        print(f"Local tensor shape: {shard_tensor._local_tensor.shape}")

    # Redistribute to new placement
    redistributed = shard_tensor.redistribute(placements=dst_placements)

    # assert False
    if verbose:
        print(f"redistributed placements: {redistributed._spec.placements}")
        dist.barrier()

    # Verify data is preserved after redistribution
    redistributed_data = redistributed.full_tensor()

    # Store original data for validation
    original_data = shard_tensor.full_tensor()

    assert torch.allclose(original_data, redistributed_data)
