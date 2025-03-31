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

from physicsnemo.utils.version_check import check_module_requirements

try:
    check_module_requirements("physicsnemo.distributed.shard_tensor")
    from test_shard_tensor_initialization import (
        init_dist,
    )
    from test_shard_tensor_redistribute import shard_tensor_factory

    from physicsnemo.distributed import ShardTensor
except ImportError:
    pytest.skip(
        "Skipping test because physicsnemo.distributed.shard_tensor is not available",
        allow_module_level=True,
    )

from distributed_utils_for_testing import modify_environment

from physicsnemo.distributed import DistributedManager


def run_shard_tensor_detach(rank, num_gpus, mesh_names, mesh_sizes, uneven, verbose):
    with modify_environment(
        RANK=f"{rank}",
        WORLD_SIZE=f"{num_gpus}",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(13245),
        LOCAL_RANK=f"{rank % torch.cuda.device_count()}",
    ):
        init_dist(rank, num_gpus)
        shard_tensor = shard_tensor_factory(
            mesh_names, mesh_sizes, requires_grad=True, uneven=uneven
        )
        shard_tensor_detached = shard_tensor.detach()

        print(f"Original spec: {shard_tensor._spec} of type {type(shard_tensor._spec)}")
        print(
            f"Detached spec: {shard_tensor_detached._spec} of type {type(shard_tensor_detached._spec)}"
        )

        print(
            f"Original sharding sizes: {shard_tensor._spec.sharding_sizes()}",
            flush=True,
        )
        print(
            f"Detached sharding sizes: {shard_tensor_detached._spec.sharding_sizes()}",
            flush=True,
        )

        # Detaching should not change the original data nor should it change the spec:
        assert shard_tensor._spec == shard_tensor_detached._spec

        assert torch.allclose(
            shard_tensor.full_tensor(), shard_tensor_detached.full_tensor()
        )

        DistributedManager().cleanup()


@pytest.mark.multigpu
@pytest.mark.parametrize("data_parallel_size", [-1])
@pytest.mark.parametrize("domain_H", [2, 4])
@pytest.mark.parametrize("domain_W", [1, 2])
@pytest.mark.parametrize("uneven", [True, False])
def test_shard_tensor_detach(data_parallel_size, domain_H, domain_W, uneven):
    """
    This test is meant to ensure ShardTensor can be initialized correctly
    from local data.  Checks the following:

    """
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        pytest.skip("Not enough GPUs available for distributed tests")

    if domain_H == 1 and domain_W == 1:
        pytest.skip("No point testing this without parallelism in the domain axes")

    if domain_H * domain_W > num_gpus:
        pytest.skip("Not enough GPUs available for test")

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
        run_shard_tensor_detach,
        args=(num_gpus, mesh_names, mesh_sizes, uneven, verbose),
        nprocs=num_gpus,
        join=True,
        daemon=True,
    )


def run_shard_tensor_input_gradient_full_loss(
    rank, num_gpus, mesh_names, mesh_sizes, uneven, verbose
):

    with modify_environment(
        RANK=f"{rank}",
        WORLD_SIZE=f"{num_gpus}",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(13245),
        LOCAL_RANK=f"{rank % torch.cuda.device_count()}",
    ):
        init_dist(rank, num_gpus)
        shard_tensor = shard_tensor_factory(
            mesh_names, mesh_sizes, requires_grad=True, uneven=uneven
        )
        shard_tensor = (
            shard_tensor.detach()
        )  # Make it a leaf tensor by calling detach andrequires_grad_
        shard_tensor = shard_tensor.detach().requires_grad_(
            True
        )  # Make it a leaf tensor by calling detach andrequires_grad_

        # For this test, we're testing that the gradients of the input tensor work
        # We'll compare them to the local gradients

        # Compute the input gradients on the full_tensor:
        full_local_tensor = shard_tensor.full_tensor().detach()
        full_local_tensor.requires_grad_(True)

        def loss(_input):
            if isinstance(_input, ShardTensor):
                x = _input.full_tensor()
            else:
                x = _input
            x = x**2
            return torch.sum(x)

        computed_local_loss = loss(full_local_tensor)
        computed_local_loss.backward()

        # This should have gradients
        assert full_local_tensor.grad is not None

        # Now compute the sharded gradients with FULL TENSOR LOSS:
        sharded_loss = loss(shard_tensor)

        sharded_loss.backward()
        # Check if shard_tensor requires grad
        assert shard_tensor.requires_grad, "ShardTensor should require grad"
        assert shard_tensor.grad is not None
        assert torch.allclose(shard_tensor.grad.full_tensor(), full_local_tensor.grad)

        DistributedManager().cleanup()


@pytest.mark.multigpu
@pytest.mark.parametrize("data_parallel_size", [-1])
@pytest.mark.parametrize("domain_H", [2, 4])
@pytest.mark.parametrize("domain_W", [1, 2])
@pytest.mark.parametrize("uneven", [True, False])
def test_shard_tensor_input_gradient_full_loss(
    data_parallel_size, domain_H, domain_W, uneven
):
    """
    This test is meant to ensure ShardTensor can be initialized correctly
    from local data.  Checks the following:

    """
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        pytest.skip("Not enough GPUs available for distributed tests")

    if domain_H == 1 and domain_W == 1:
        pytest.skip("No point testing this without parallelism in the domain axes")

    if domain_H * domain_W > num_gpus:
        pytest.skip("Not enough GPUs available for test")

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
        run_shard_tensor_input_gradient_full_loss,
        args=(num_gpus, mesh_names, mesh_sizes, uneven, verbose),
        nprocs=num_gpus,
        join=True,
        daemon=True,
    )


def run_shard_tensor_input_gradient_local_loss(
    rank, num_gpus, mesh_names, mesh_sizes, uneven, verbose
):

    with modify_environment(
        RANK=f"{rank}",
        WORLD_SIZE=f"{num_gpus}",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(13245),
        LOCAL_RANK=f"{rank % torch.cuda.device_count()}",
    ):

        init_dist(rank, num_gpus)
        shard_tensor = shard_tensor_factory(
            mesh_names, mesh_sizes, requires_grad=True, uneven=uneven
        )
        shard_tensor = (
            shard_tensor.detach()
        )  # Make it a leaf tensor by calling detach andrequires_grad_
        shard_tensor = shard_tensor.detach().requires_grad_(
            True
        )  # Make it a leaf tensor by calling detach andrequires_grad_

        # For this test, we're testing that the gradients of the input tensor work
        # We'll compare them to the local gradients

        # Compute the input gradients on the full_tensor:
        full_local_tensor = shard_tensor.full_tensor().detach()
        full_local_tensor.requires_grad_(True)

        def loss(_input):
            # Compute the loss *locally*
            if isinstance(_input, ShardTensor):
                x = _input.to_local()
            else:
                x = _input
            x = x**2
            return torch.sum(x)

        computed_local_loss = loss(full_local_tensor)
        computed_local_loss.backward()

        # This should have gradients
        assert full_local_tensor.grad is not None

        # Now compute the sharded gradients:
        sharded_loss = loss(shard_tensor)

        sharded_loss.backward()

        # Check if shard_tensor requires grad
        assert shard_tensor.requires_grad, "ShardTensor should require grad"
        assert shard_tensor.grad is not None

        assert torch.allclose(shard_tensor.grad.full_tensor(), full_local_tensor.grad)

        DistributedManager().cleanup()


@pytest.mark.multigpu
@pytest.mark.parametrize("data_parallel_size", [-1])
@pytest.mark.parametrize("domain_H", [2, 4])
@pytest.mark.parametrize("domain_W", [1, 2])
@pytest.mark.parametrize("uneven", [True, False])
def test_shard_tensor_input_gradient_local_loss(
    data_parallel_size, domain_H, domain_W, uneven
):
    """
    This test is meant to ensure ShardTensor can be initialized correctly
    from local data.  Checks the following:

    """
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        pytest.skip("Not enough GPUs available for distributed tests")

    if domain_H == 1 and domain_W == 1:
        pytest.skip("No point testing this without parallelism in the domain axes")

    if domain_H * domain_W > num_gpus:
        pytest.skip("Not enough GPUs available for test")

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
        run_shard_tensor_input_gradient_local_loss,
        args=(num_gpus, mesh_names, mesh_sizes, uneven, verbose),
        nprocs=num_gpus,
        join=True,
        daemon=True,
    )


if __name__ == "__main__":

    # test_shard_tensor_detach(-1,2, 1, True)
    test_shard_tensor_input_gradient_local_loss(-1, 2, 1, True)
    test_shard_tensor_input_gradient_full_loss(-1, 2, 1, True)
