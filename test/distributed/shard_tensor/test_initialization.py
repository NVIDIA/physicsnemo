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

import random

import pytest

from physicsnemo.utils.version_check import check_module_requirements

try:
    check_module_requirements("physicsnemo.distributed.shard_tensor")
    from torch.distributed.tensor import distribute_tensor
    from torch.distributed.tensor.placement_types import Shard

    from physicsnemo.distributed.shard_tensor import ShardTensor, scatter_tensor

except ImportError:
    pytest.skip(
        "Skipping test because physicsnemo.distributed.shard_tensor is not available",
        allow_module_level=True,
    )

import torch
import torch.distributed as dist

from physicsnemo.distributed import DistributedManager


def init_global_shape_and_placements(domain_mesh):
    # Sharding in up to two dimensions (domain_H, domain_W)
    # Why these numbers?  Making sure these axes are actually divisible
    # by these numbers.  ShardTensor can handle the funky distribution
    # but dtensor can not.
    global_shape = (10, 2 * 3 * 4 * 5 * 7, 2 * 3 * 4 * 5 * 7, 10)

    placements = [Shard(1)]
    # 2D placements if mesh is 2D
    if domain_mesh.ndim > 1:
        placements.append(Shard(2))

    return global_shape, placements


def init_from_data_rank_worker(mesh):
    # This test uses a worker function since I want to enable
    # both 1D and 2D meshes

    # It emulates loading the data on one rank of a mesh, and scattering
    # the data to the rest of the mesh.  Testing w/ both 1d and 2d meshes
    # tests scatter_tensor function appropriately.

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

    dt_full = dt.full_tensor()
    st_full = st.full_tensor()

    assert torch.allclose(dt_full, st_full)

    # on the "source" rank of the mesh, we should have agreement with raw data.
    # on the "not-source" rank of the mesh, we shouldn't

    agreement_with_original_data = torch.allclose(st.full_tensor(), raw_data)

    if dm.rank == int(mesh.mesh.min()):
        assert agreement_with_original_data
    else:
        assert not agreement_with_original_data


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
    # Here, we create local shards and combine into a shard tensor.
    # This test is allowed to go a little wild: the shapes for the local tensors
    # are allowed to be randomly generated along the first shard axis.

    # 2D sharding would break if we did that, so it's set to a fixed size
    # on other mesh dims.

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
