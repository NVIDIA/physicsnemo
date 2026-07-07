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

r"""Tests for ShardTensor redistribution between different sharding schemes.

One major feature of ShardTensor is that it knows both the global shape
and local layout of every shard, and can seamlessly translate between them.

In many ways, this is an extension of DTensor's utilities, but we're testing
here specifically any uneven shardings, etc.

The tests cover 1D and 2D meshes of shard tensors with increasingly complex
resharding requirements. In all cases, the input tensors are sharded:
``(Shard(1),)`` for 1D, ``(Shard(1), Shard(2))`` for 2D.

Test cases include:

- No-op redistributions (same source and target placements)
- Shard to Replicate (gather operations)
- Replicate to Shard (scatter operations)
- Shard to Shard on different dimensions (all-to-all transpose)
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor


def shard_tensor_factory(mesh, requires_grad=False, uneven=True):
    r"""Generate a ShardTensor on the mesh for testing.

    Creates a randomly-valued tensor sharded according to the mesh dimensions.
    Can create either even or uneven sharding depending on the ``uneven`` parameter.

    Parameters
    ----------
    mesh : DeviceMesh
        The device mesh to create the ShardTensor on.
    requires_grad : bool, default=False
        Whether the tensor requires gradients.
    uneven : bool, default=True
        If ``True``, creates tensors with different sizes on each rank.
        If ``False``, creates tensors with uniform sizes across ranks.

    Returns
    -------
    ShardTensor
        A ShardTensor with shape ``(100, *, ..., 100)`` where the middle
        dimensions depend on mesh rank if ``uneven=True``.
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
@pytest.mark.parametrize("uneven", [True, False])
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
    distributed_mesh, redistribution_case, uneven, verbose=False
):
    """Test redistribution between different sharding schemes"""
    run_shard_tensor_redistribute(
        distributed_mesh, redistribution_case, uneven=uneven, verbose=verbose
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
    distributed_mesh_2d, redistribution_case, verbose=False
):
    run_shard_tensor_redistribute(
        distributed_mesh_2d, redistribution_case, verbose=verbose
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("uneven", [True, False])
@pytest.mark.parametrize(
    "redistribution_case",
    [
        ("S2+R", [Shard(2), Replicate()]),  # gather_v then all_to_all_v
        ("S2+S1", [Shard(2), Shard(1)]),  # gather_v, all_to_all_v, scatter_v
    ],
)
def test_shard_tensor_redistribute2d_even_uneven(
    distributed_mesh_2d, redistribution_case, uneven, verbose=False
):
    """Multi-hop 2D cases under both even and uneven sharding.

    These paths reach the all_to_all transpose on a later hop, after an
    earlier hop has already mutated the local tensor -- exercising the
    rank-uniform staleness gate on the recv-shape fast path.
    """
    run_shard_tensor_redistribute(
        distributed_mesh_2d, redistribution_case, uneven=uneven, verbose=verbose
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize(
    "redistribution_case",
    [
        ("S2", [Shard(2)]),  # all_to_all_v transpose with an empty send/recv
        ("R", [Replicate()]),  # gather_v with an empty contribution
    ],
)
def test_shard_tensor_redistribute_empty_shard(distributed_mesh, redistribution_case):
    """Redistribution when one rank holds an empty shard.

    Empty shards arise from chunking whenever the global extent does not
    fill every rank (e.g. ``compute_split_shapes(3, 4) == [0, 0, 0, 3]``).
    This is a regression test for the recv-shape fast path: a rank-local
    staleness check can disagree across ranks exactly in this regime (a
    zero-extent shard can coincidentally match a post-gather shape on one
    rank but not another), so the fast-path decision must be rank-uniform.
    """
    case_name, dst_placements = redistribution_case
    dm = DistributedManager()
    mesh = distributed_mesh

    dim_rank = dist.get_group_rank(mesh.get_group(0), dm.rank)
    mesh_size = mesh.size(0)

    # Last rank holds an empty shard on the sharded dimension:
    dim1_extent = 4 if dim_rank < mesh_size - 1 else 0
    raw_data = torch.randn(
        (16, dim1_extent, 32),
        device=torch.device(f"cuda:{dm.local_rank}"),
    )

    shard_tensor = ShardTensor.from_local(
        raw_data,
        device_mesh=mesh,
        placements=[Shard(1)],
        sharding_shapes="infer",
    )

    original_data = shard_tensor.full_tensor()
    redistributed = shard_tensor.redistribute(placements=dst_placements)
    assert torch.allclose(original_data, redistributed.full_tensor())


@pytest.mark.multigpu_static
@pytest.mark.parametrize("uneven", [True, False])
def test_redistribute_fast_path_matches_fallback(distributed_mesh, uneven):
    """The analytic recv-shape fast path must reproduce the negotiated path.

    Runs the same Shard(1) -> Shard(2) transpose twice on identical data:
    once with the spec's recorded sharding shapes available (fast path),
    and once with them wiped (forcing the shape-negotiation all_to_all
    fallback). Results must match exactly.
    """
    dm = DistributedManager()
    mesh = distributed_mesh

    dim_rank = dist.get_group_rank(mesh.get_group(0), dm.rank)
    dim1_extent = (dim_rank + 1) * 4 + dim_rank * 2 if uneven else 4
    raw_data = torch.randn(
        (16, dim1_extent, 32),
        device=torch.device(f"cuda:{dm.local_rank}"),
    )

    st_fast = ShardTensor.from_local(
        raw_data,
        device_mesh=mesh,
        placements=[Shard(1)],
        sharding_shapes="infer",
    )
    st_fallback = ShardTensor.from_local(
        raw_data.clone(),
        device_mesh=mesh,
        placements=[Shard(1)],
        sharding_shapes="infer",
    )
    # Wiping the recorded shapes forces the negotiation fallback:
    st_fallback._spec._sharding_shapes = None

    out_fast = st_fast.redistribute(placements=[Shard(2)])
    out_fallback = st_fallback.redistribute(placements=[Shard(2)])

    assert out_fast._local_tensor.shape == out_fallback._local_tensor.shape
    assert torch.equal(out_fast._local_tensor, out_fallback._local_tensor)


def run_shard_tensor_redistribute(
    mesh, redistribution_case, uneven=True, verbose=False
):
    r"""Run a single redistribution test case.

    Parameters
    ----------
    mesh : DeviceMesh
        The device mesh to test on.
    redistribution_case : Tuple[str, List[Placement]]
        Tuple of (case_name, target_placements) describing the redistribution.
    verbose : bool, default=False
        If ``True``, print debugging information.
    """
    case_name, dst_placements = redistribution_case

    # Create initial sharded tensor
    shard_tensor = shard_tensor_factory(mesh, uneven=uneven)
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
