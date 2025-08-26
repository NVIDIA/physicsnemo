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

import sys

sys.path.append("../")

import pytest

from physicsnemo.utils.version_check import check_module_requirements

try:
    check_module_requirements("physicsnemo.distributed.shard_tensor")
    from test_shard_tensor_initialization import (
        init_dist,
    )

except ImportError:
    pytest.skip(
        "Skipping test because physicsnemo.distributed.shard_tensor is not available",
        allow_module_level=True,
    )

import torch
from pytest_utils import modify_environment
from test_shard_tensor_redistribute import shard_tensor_factory

from physicsnemo.distributed import DistributedManager


def run_shard_tensor_unsqueeze(rank, num_gpus, mesh_names, mesh_sizes, verbose):
    with modify_environment(
        RANK=f"{rank}",
        WORLD_SIZE=f"{num_gpus}",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(13245),
        LOCAL_RANK=f"{rank % torch.cuda.device_count()}",
    ):
        init_dist(rank, num_gpus)

        shard_tensor = shard_tensor_factory(mesh_names, mesh_sizes)
        if verbose:
            print()

        # For this test, we're testing that the unsqueeze of the tensor works correctly

        full_original_tensor = shard_tensor.full_tensor()

        indexes = list(range(len(full_original_tensor.shape)))

        for i in indexes:
            i_sharded_unsqueeze = shard_tensor.unsqueeze(i)
            i_unsharded_unsqueeze = full_original_tensor.unsqueeze(i)

            assert i_sharded_unsqueeze.shape == i_sharded_unsqueeze.shape
            assert torch.allclose(
                i_sharded_unsqueeze.full_tensor(), i_unsharded_unsqueeze
            )

            ni_sharded_unsqueeze = shard_tensor.unsqueeze(-i)
            ni_unsharded_unsqueeze = full_original_tensor.unsqueeze(-i)

            assert ni_sharded_unsqueeze.shape == ni_sharded_unsqueeze.shape
            assert torch.allclose(
                ni_sharded_unsqueeze.full_tensor(), ni_unsharded_unsqueeze
            )

        DistributedManager().cleanup()


@pytest.mark.multigpu
@pytest.mark.parametrize("data_parallel_size", [-1])
@pytest.mark.parametrize("domain_H", [2, 4])
@pytest.mark.parametrize("domain_W", [1, 2])
def test_shard_tensor_unsqueeze(data_parallel_size, domain_H, domain_W):
    """
    This test is meant to ensure ShardTensor can be initialized correctly
    from local data. Checks that reduction operations work correctly.

    Note: Mean reduction is expected to fail since averaging over uneven tensor shapes
    is not yet supported.
    """
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"

    if domain_H == 1 and domain_W == 1:
        pytest.skip("No point testing this without parallelism in the domain axes")

    # if op == torch.mean:
    # pytest.xfail("Mean reduction not yet supported for uneven tensor shapes")

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
        run_shard_tensor_unsqueeze,
        args=(num_gpus, mesh_names, mesh_sizes, verbose),
        nprocs=num_gpus,
        join=True,
        daemon=True,
    )


if __name__ == "__main__":
    test_shard_tensor_unsqueeze(-1, 4, 1)
