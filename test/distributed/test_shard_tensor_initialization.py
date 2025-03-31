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
from distributed_utils_for_testing import modify_environment

from physicsnemo.distributed import DistributedManager


def init_dist(rank, num_gpus):

    DistributedManager.initialize()
    dm = DistributedManager()
    assert dm.is_initialized()


def init_global_shape_and_placements(mesh_names):

    dm = DistributedManager()

    global_mesh = dm.global_mesh

    # Sharding in up to two dimensions (domain_H, domain_W)
    domain_names = mesh_names[1:]
    domain_mesh = global_mesh[tuple(domain_names)]

    global_shape = (10, 64, 64, 10)

    placements = [Shard(1)]
    # 2D placements if mesh is 2D
    if domain_mesh.ndim > 1:
        placements.append(Shard(2))

    return domain_mesh, global_shape, placements


def run_shard_tensor_initialization_from_data_rank(
    rank, num_gpus, mesh_names, mesh_sizes, verbose
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

        # Create a mesh right from the inputs:
        global_mesh = dm.initialize_mesh(mesh_sizes, mesh_names)  # noqa: F841

        domain_mesh, global_shape, placements = init_global_shape_and_placements(
            mesh_names,
        )

        # Create the raw data on the first rank of the first dimension of the domain mesh:
        source = dist.get_global_rank(domain_mesh.get_group(0), 0)
        source = int(domain_mesh.mesh.min())

        if rank == source:
            raw_data = torch.randn(
                global_shape, device=torch.device(f"cuda:{dm.local_rank}")
            )
        else:
            raw_data = torch.empty(0)

        st = scatter_tensor(raw_data, source, domain_mesh, placements)

        # Check that the local shape matches the expected shape:
        local_data = st.to_local()
        print(f"local shape: {local_data.shape}")
        # Check the dimensions on the sharded mesh:
        checked_dims = []
        for mesh_dim, placement in enumerate(placements):
            if isinstance(placement, Shard):
                tensor_dim = placement.dim
                axis_size = dist.get_world_size(group=domain_mesh.get_group(mesh_dim))
                assert (
                    global_shape[tensor_dim] == local_data.shape[tensor_dim] * axis_size
                )
                checked_dims.append(tensor_dim)

        # Check the dimensions NOT on the mesh:
        for i, dim in enumerate(global_shape):
            if i in checked_dims:
                continue
            assert dim == local_data.shape[i]

        dm.cleanup()


def run_shard_tensor_initialization_from_all_dtensor(
    rank, num_gpus, mesh_names, mesh_sizes, verbose
):
    with modify_environment(
        RANK=f"{rank}",
        WORLD_SIZE=f"{num_gpus}",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(13245),
        LOCAL_RANK=f"{rank % torch.cuda.device_count()}",
    ):
        # Here, we manually create a dtensor and convert it to shard tensor.

        # The check is that there are no errors, and basic reductions agree.

        init_dist(rank, num_gpus)

        dm = DistributedManager()

        # Create a mesh right from the inputs:
        global_mesh = dm.initialize_mesh(mesh_sizes, mesh_names)  # noqa: F841
        domain_mesh, global_shape, placements = init_global_shape_and_placements(
            mesh_names,
        )

        # Create the raw data everywhere, but it will mostly get thrown away
        # only the rank-0 chunks survive
        raw_data = torch.randn(
            global_shape, device=torch.device(f"cuda:{dm.local_rank}")
        )

        dt = distribute_tensor(raw_data, device_mesh=domain_mesh, placements=placements)

        st = ShardTensor.from_dtensor(dt)

        assert torch.allclose(dt.full_tensor(), st.full_tensor())

        # on the "source" rank of the mesh, we should have agreement with raw data.
        # on the "not-source" rank of the mesh, we shouldn't

        agreement_with_original_data = torch.allclose(st.full_tensor(), raw_data)

        if dm.rank == int(domain_mesh.mesh.min()):
            assert agreement_with_original_data
        else:
            assert not agreement_with_original_data

        dm.cleanup()


def run_shard_tensor_initialization_from_local_chunks(
    rank, num_gpus, mesh_names, mesh_sizes, verbose
):

    # Here, we create local shards and combine into a shard tensor.
    # This test is allowed to go a little wild: the shapes for the local tensors
    # are allowed to be randomly generated along the first shard axis.

    # 2D sharding would break if we did that, so it's set to a fixed size
    with modify_environment(
        RANK=f"{rank}",
        WORLD_SIZE=f"{num_gpus}",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(13245),
        LOCAL_RANK=f"{rank % torch.cuda.device_count()}",
    ):
        init_dist(rank, num_gpus)

        dm = DistributedManager()

        # Create a mesh right from the inputs:
        global_mesh = dm.initialize_mesh(mesh_sizes, mesh_names)  # noqa: F841
        domain_mesh, global_shape, placements = init_global_shape_and_placements(
            mesh_names,
        )

        local_shape = list(global_shape)
        first_shard_dim = placements[0].dim
        replacement_size = int(random.uniform(0.5, 1.5) * local_shape[first_shard_dim])

        local_shape[first_shard_dim] = replacement_size

        # replace the dimension with a new one

        # Create the raw data everywhere, but it will mostly get thrown away
        # only the rank-0 chunks survive
        raw_data = torch.randn(
            local_shape, device=torch.device(f"cuda:{dm.local_rank}")
        )
        st = ShardTensor.from_local(
            raw_data, device_mesh=domain_mesh, placements=placements, infer_shape=True
        )

        # Data comes back ok:
        assert torch.allclose(st.to_local(), raw_data)

        # Gather the shapes along the random placement and make sure they agree:
        dim_size = domain_mesh.mesh.shape[0]
        shard_dim_sizes = [
            0,
        ] * dim_size
        dist.all_gather_object(
            shard_dim_sizes, replacement_size, group=domain_mesh.get_group(0)
        )

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
        dm.cleanup()


@pytest.mark.multigpu
@pytest.mark.parametrize("data_parallel_size", [-1])
@pytest.mark.parametrize("domain_H", [1, 2])
@pytest.mark.parametrize("domain_W", [2, 4])
def test_shard_tensor_initialization_from_data_rank(
    data_parallel_size, domain_H, domain_W
):
    """
    This test is meant to ensure ShardTensor can be initialized correctly
    from local data.  Checks the following:

    - On one rank on the first axis of the mesh (data parallel), create a tensor
    - Use the scatter_tensor function to turn that into a sharded tensor on the mesh
    - check that the scatter dimensions match

    - sharded dimensions here are hardcoded to be divisible by 2 many times, so we can
      check exact dimensions

    """
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"

    if domain_H == 1 and domain_W == 1:
        pytest.skip("No point testing this without parallelism in the domain axes")

    remaining_gpus = num_gpus
    mesh_names = ["data_parallel"]
    mesh_sizes = [data_parallel_size]

    if int(remaining_gpus / domain_H) != 0:
        mesh_names.append("domain_H")
        mesh_sizes.append(domain_H)
        remaining_gpus = int(remaining_gpus / domain_H)

    if int(remaining_gpus / domain_W) != 0:
        mesh_names.append("domain_W")
        mesh_sizes.append(domain_W)
        remaining_gpus = int(remaining_gpus / domain_W)

    verbose = False  # Change to True for debug

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_shard_tensor_initialization_from_data_rank,
        args=(num_gpus, mesh_names, mesh_sizes, verbose),
        nprocs=num_gpus,
        join=True,
        daemon=True,
    )


@pytest.mark.multigpu
@pytest.mark.parametrize("data_parallel_size", [-1])
@pytest.mark.parametrize("domain_H", [2, 4])
@pytest.mark.parametrize(
    "domain_W",
    [
        1,
    ],
)
def test_shard_tensor_initialization_from_local_chunks(
    data_parallel_size, domain_H, domain_W
):
    """
    This test is meant to ensure ShardTensor can be initialized correctly
    from local data.  It's meant to mimic the situation where a user
    loads data partially to each rank, and combines it into a ShardTensor.

    This test lets one sharding axis float randomly (always the first placement)
    in order to check the inference of size and shapes.

    Only checking uneven sharding on one dimension here.  Two is overly
    complicated for unit tests.

    Checks the following:
        - That the shard tensor correctly infers global size and shape
        - that the shard tensor can correctly return then local tensor
        - That the shard tensor can correctly coalesce the whole tensor.
    """
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"

    if domain_H == 1 and domain_W == 1:
        pytest.skip("No point testing this without parallelism in the domain axes")

    remaining_gpus = num_gpus
    mesh_names = ["data_parallel"]
    mesh_sizes = [data_parallel_size]

    if int(remaining_gpus / domain_H) != 0:
        mesh_names.append("domain_H")
        mesh_sizes.append(domain_H)
        remaining_gpus = int(remaining_gpus / domain_H)

    if int(remaining_gpus / domain_W) != 0:
        mesh_names.append("domain_W")
        mesh_sizes.append(domain_W)
        remaining_gpus = int(remaining_gpus / domain_W)

    verbose = False  # Change to True for debug

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_shard_tensor_initialization_from_local_chunks,
        args=(num_gpus, mesh_names, mesh_sizes, verbose),
        nprocs=num_gpus,
        join=True,
        daemon=True,
    )


@pytest.mark.multigpu
@pytest.mark.parametrize("data_parallel_size", [-1])
@pytest.mark.parametrize("domain_H", [2, 4])
@pytest.mark.parametrize("domain_W", [1, 2])
def test_shard_tensor_initialization_from_all_dtensor(
    data_parallel_size, domain_H, domain_W
):
    """
    This test is meant to ensure ShardTensor can be initialized correctly
    from local data.  Checks the following:

    """
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"

    if domain_H == 1 and domain_W == 1:
        pytest.skip("No point testing this without parallelism in the domain axes")

    remaining_gpus = num_gpus
    mesh_names = ["data_parallel"]
    mesh_sizes = [data_parallel_size]

    if int(remaining_gpus / domain_H) != 0:
        mesh_names.append("domain_H")
        mesh_sizes.append(domain_H)
        remaining_gpus = int(remaining_gpus / domain_H)

    if int(remaining_gpus / domain_W) != 0:
        mesh_names.append("domain_W")
        mesh_sizes.append(domain_W)
        remaining_gpus = int(remaining_gpus / domain_W)

    verbose = False  # Change to True for debug

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_shard_tensor_initialization_from_all_dtensor,
        args=(num_gpus, mesh_names, mesh_sizes, verbose),
        nprocs=num_gpus,
        join=True,
        daemon=True,
    )
