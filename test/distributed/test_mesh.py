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
from distributed_utils_for_testing import modify_environment

from physicsnemo.distributed import (
    DistributedManager,
)
from physicsnemo.utils.version_check import check_module_requirements

try:
    check_module_requirements("device_mesh")
except ImportError:
    pytest.skip(
        "Skipping test because device_mesh is not available",
        allow_module_level=True,
    )


distributed_test = pytest.mark.skipif(
    not torch.distributed.is_available(), reason="PyTorch distributed not available"
)


def run_mesh_creation(rank, num_gpus, mesh_names, mesh_sizes, verbose):

    with modify_environment(
        RANK=f"{rank}",
        WORLD_SIZE=f"{num_gpus}",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(12355),
        LOCAL_RANK=f"{rank % torch.cuda.device_count()}",
    ):

        DistributedManager.initialize()
        dm = DistributedManager()
        assert dm.is_initialized()

        # Create a mesh right from the inputs:
        global_mesh = dm.initialize_mesh(mesh_sizes, mesh_names)

        # Check the dimension matches:
        assert global_mesh.ndim == len(mesh_names)

        # Make sure the number of devices matches the world size:
        for size, name in zip(reversed(mesh_sizes), reversed(mesh_names)):
            if size != -1:
                assert global_mesh[name].size() == size

        # Make sure each dimension of the mesh is orthogonal to other dimensions:
        # (but only if there are at least two names:)
        if len(mesh_names) > 1:
            for i, i_name in enumerate(mesh_names):
                for j, j_name in enumerate(mesh_names[i + 1 :]):

                    mesh_i = global_mesh[i_name].mesh.tolist()
                    mesh_j = global_mesh[j_name].mesh.tolist()
                    intersection = list(set(mesh_i) & set(mesh_j))
                    if verbose:
                        print(
                            f"rank {dm.rank}, i_name {i_name}, j_name {j_name}, mesh_i {mesh_i}, mesh_j {mesh_j}, int {intersection}"
                        )
                    assert len(intersection) == 1
                    assert intersection[0] == dm.rank

        # Cleanup process groups
        DistributedManager.cleanup()


@pytest.mark.multigpu
@pytest.mark.parametrize("data_parallel_size", [-1])
@pytest.mark.parametrize("domain_parallel_size", [2, 1])
@pytest.mark.parametrize("model_parallel_size", [4, 2])
def test_mesh_creation(data_parallel_size, domain_parallel_size, model_parallel_size):
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"

    remaining_gpus = num_gpus
    mesh_names = ["data_parallel"]
    mesh_sizes = [data_parallel_size]

    if int(remaining_gpus / domain_parallel_size) != 0:
        mesh_names.append("domain_parallel")
        mesh_sizes.append(domain_parallel_size)
        remaining_gpus = int(remaining_gpus / domain_parallel_size)

    if int(remaining_gpus / model_parallel_size) != 0:
        mesh_names.append("model_parallel")
        mesh_sizes.append(model_parallel_size)
        remaining_gpus = int(remaining_gpus / model_parallel_size)

    verbose = False  # Change to True for debug

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_mesh_creation,
        args=(num_gpus, mesh_names, mesh_sizes, verbose),
        nprocs=num_gpus,
        join=True,
        daemon=True,
    )


if __name__ == "__main__":
    test_mesh_creation(-1, 2, 2)
