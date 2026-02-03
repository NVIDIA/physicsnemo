# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

r"""Tests for ShardTensor initialization methods.

This module tests the different supported ways to initialize a ``ShardTensor``:

- Initialization from a single data rank via ``scatter_tensor``
- Initialization from an existing ``DTensor`` via ``ShardTensor.from_dtensor``
- Initialization from local chunks via ``ShardTensor.from_local``

The tests cover both 1D and 2D device meshes, and verify that data is correctly
distributed and can be recovered via ``full_tensor()``.
"""

import random

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel.shard_tensor import ShardTensor, scatter_tensor


def init_global_shape_and_placements(domain_mesh):
    r"""Initialize global shape and placements for test tensors.

    Creates a 4D tensor shape and appropriate placements based on mesh dimensions.
    The shape dimensions are chosen to be divisible by common mesh sizes.

    Parameters
    ----------
    domain_mesh : DeviceMesh
        The device mesh to create placements for.

    Returns
    -------
    Tuple[Tuple[int, ...], List[Shard]]
        Tuple of (global_shape, placements) where global_shape is the tensor
        shape and placements defines how to shard across mesh dimensions.

    Note
    ----
    The shape values (2 * 3 * 4 * 5 * 7 = 840) are chosen to be divisible
    by many common factors, ensuring compatibility with various mesh sizes.
    ShardTensor can handle uneven distributions, but DTensor cannot.
    """
    global_shape = (10, 2 * 3 * 4 * 5 * 7, 2 * 3 * 4 * 5 * 7, 10)

    placements = [Shard(1)]
    # 2D placements if mesh is 2D
    if domain_mesh.ndim > 1:
        placements.append(Shard(2))

    return global_shape, placements


def init_from_data_rank_worker(mesh):
    r"""Worker function to test initialization from a single data rank.

    Emulates loading data on one rank of a mesh and scattering the data to
    the rest of the mesh. Tests the ``scatter_tensor`` function with both
    1D and 2D meshes.

    Parameters
    ----------
    mesh : DeviceMesh
        The device mesh to scatter the tensor across.
    """

    dm = DistributedManager()
    rank = dm.rank

    global_shape, placements = init_global_shape_and_placements(
        mesh,
    )

    # Create the raw data on the first rank of the first dimension of the domain mesh:
    source = 0

    if rank == source:
        raw_data = torch.randn(
            global_shape, device=torch.device(f"cuda:{dm.local_rank}")
        )
    else:
        raw_data = None

    st = scatter_tensor(raw_data, source, mesh, placements)

    # Check that the local shape matches the expected shape:
    local_data = st.to_local()
    # Check the dimensions on the sharded mesh:
    checked_dims = []
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            tensor_dim = placement.dim
            axis_size = dist.get_world_size(group=mesh.get_group(mesh_dim))
            assert global_shape[tensor_dim] == local_data.shape[tensor_dim] * axis_size
            checked_dims.append(tensor_dim)

    # Check the dimensions NOT on the mesh:
    for i, dim in enumerate(global_shape):
        if i in checked_dims:
            continue
        assert dim == local_data.shape[i]


@pytest.mark.timeout(10)
@pytest.mark.multigpu_static
def test_shard_tensor_initialization_from_data_rank_1d(distributed_mesh, verbose=False):
    init_from_data_rank_worker(distributed_mesh)


@pytest.mark.timeout(10)
@pytest.mark.multigpu_static
def test_shard_tensor_initialization_from_data_rank_2d(
    distributed_mesh_2d, verbose=False
):
    init_from_data_rank_worker(distributed_mesh_2d)


def shard_tensor_initialization_from_all_dtensor_worker(mesh):
    r"""Worker function to test initialization from DTensor.

    Creates a DTensor using ``distribute_tensor`` and converts it to a
    ShardTensor using ``ShardTensor.from_dtensor``. Verifies that the
    full tensor is preserved correctly.

    Parameters
    ----------
    mesh : DeviceMesh
        The device mesh to distribute the tensor across.
    """
    dm = DistributedManager()

    global_shape, placements = init_global_shape_and_placements(
        mesh,
    )

    # Create the raw data everywhere, but it will mostly get thrown away
    # only the rank-0 chunks survive
    raw_data = torch.randn(global_shape, device=torch.device(f"cuda:{dm.local_rank}"))

    # DTensor tool to distribute:
    dt = distribute_tensor(raw_data, device_mesh=mesh, placements=placements)

    st = ShardTensor.from_dtensor(dt)

    print(f"Rank {dm.rank} made shard tensors.")

    dt_full = dt.full_tensor()
    st_full = st.full_tensor()

    assert torch.allclose(dt_full, st_full)


@pytest.mark.timeout(10)
@pytest.mark.multigpu_static
def test_shard_tensor_initialization_from_all_dtensor(distributed_mesh, verbose=False):
    shard_tensor_initialization_from_all_dtensor_worker(distributed_mesh)


@pytest.mark.timeout(10)
@pytest.mark.multigpu_static
def test_shard_tensor_initialization_from_all_dtensor_2d(
    distributed_mesh_2d, verbose=False
):
    shard_tensor_initialization_from_all_dtensor_worker(distributed_mesh_2d)


def shard_tensor_initialization_from_local_chunks_worker(mesh):
    r"""Worker function to test initialization from local chunks.

    Creates local shards with randomly varying sizes along the first shard
    axis and combines them into a ShardTensor using ``ShardTensor.from_local``
    with ``sharding_shapes="infer"``. This tests the ability to handle uneven
    sharding.

    Parameters
    ----------
    mesh : DeviceMesh
        The device mesh to create the ShardTensor on.

    Note
    ----
    The local tensor sizes are randomly varied only along the first shard axis.
    2D sharding would break if we varied both dimensions, so the second mesh
    dimension uses a fixed size.
    """

    dm = DistributedManager()

    # Create a mesh right from the inputs:
    global_shape, placements = init_global_shape_and_placements(
        mesh,
    )

    local_shape = list(global_shape)
    first_shard_dim = placements[0].dim
    replacement_size = int(random.uniform(0.5, 1.5) * local_shape[first_shard_dim])
    local_shape[first_shard_dim] = replacement_size
    # Important!  This replaced size is _not_ shared with other ranks.
    # We're specifically testing the utilities to infer that for users.

    # replace the dimension with a new one

    # Create the raw data everywhere, but it will mostly get thrown away
    # only the rank-0 chunks survive
    raw_data = torch.randn(local_shape, device=torch.device(f"cuda:{dm.local_rank}"))

    st = ShardTensor.from_local(
        raw_data,
        device_mesh=mesh,
        placements=placements,
        sharding_shapes="infer",
    )

    # Local data comes back ok:
    assert torch.allclose(st.to_local(), raw_data)

    # Gather the shapes along the random placement and make sure they agree:
    dim_size = mesh.mesh.shape[0]
    shard_dim_sizes = [
        0,
    ] * dim_size
    dist.all_gather_object(shard_dim_sizes, replacement_size, group=mesh.get_group(0))

    shard_dim_size_total = sum(shard_dim_sizes)
    assert st.shape[placements[0].dim] == shard_dim_size_total

    # From the full tensor, use the offset+length to slice it and compare against original:
    offset = st.offsets(mesh_dim=0)
    L = replacement_size

    index = torch.arange(L) + offset
    index = index.to(raw_data.device)

    local_slice = st.full_tensor().index_select(placements[0].dim, index)
    # Slice out what should be the original tensor

    agreement_with_original_data = torch.allclose(local_slice, raw_data)

    assert agreement_with_original_data


@pytest.mark.timeout(10)
@pytest.mark.multigpu_static
def test_shard_tensor_initialization_from_local_chunks(distributed_mesh, verbose=False):
    shard_tensor_initialization_from_local_chunks_worker(distributed_mesh)


# Don't add the 2D version of this test - it's too crudely implemented here
# to work right
