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
from pytest_utils import modify_environment

from physicsnemo.distributed import (
    DistributedManager,
    PhysicsNeMoUndefinedGroupError,
    PhysicsNeMoUninitializedDistributedManagerWarning,
    ProcessGroupConfig,
    ProcessGroupNode,
)

distributed_test = pytest.mark.skipif(
    not torch.distributed.is_available(), reason="PyTorch distributed not available"
)


def test_manager():
    with modify_environment(
        RANK=0,
        WORLD_SIZE=1,
        MASTER_ADDR="localhost",
        MASTER_PORT=str(12355),
        LOCAL_RANK="0",
    ):
        DistributedManager.initialize()
        print(DistributedManager())

        manager = DistributedManager()

        assert manager.is_initialized()
        assert manager.distributed == torch.distributed.is_available(), (
            "Manager should be in serial mode"
        )
        assert manager.rank == 0
        assert manager.world_size == 1
        assert manager.local_rank == 0

        DistributedManager.cleanup()


def test_manager_slurm():
    # Test distributed manager with Slurm variables
    with modify_environment(
        MASTER_ADDR="localhost",
        MASTER_PORT="12345",
        SLURM_PROCID="0",
        SLURM_NPROCS="1",
        SLURM_LOCALID="0",
        SLURM_LAUNCH_NODE_IPADDR="localhost",
    ):
        DistributedManager.initialize()

        manager = DistributedManager()

        assert manager.is_initialized()
        assert manager.rank == 0
        assert manager.world_size == 1
        assert manager.local_rank == 0
        DistributedManager.cleanup()


def test_manager_ompi():
    with modify_environment(
        MASTER_ADDR="localhost",
        MASTER_PORT="12345",
        OMPI_COMM_WORLD_RANK="0",
        OMPI_COMM_WORLD_SIZE="1",
        OMPI_COMM_WORLD_LOCAL_RANK="0",
    ):
        # Test distributed manager with openMPI variables
        DistributedManager.initialize()

        manager = DistributedManager()

        assert manager.is_initialized()
        assert manager.rank == 0
        assert manager.world_size == 1
        assert manager.local_rank == 0
        DistributedManager.cleanup()


def test_manager_specified_initialization():
    # PyTorch env vars
    with modify_environment(
        MASTER_ADDR="localhost",
        MASTER_PORT="12345",
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
    ):
        with modify_environment(
            SLURM_PROCID="0",
            SLURM_NPROCS="1",
            SLURM_LOCALID="0",
            SLURM_LAUNCH_NODE_IPADDR="localhost",
            PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD="SLURM",
        ):
            DistributedManager.initialize()

            # Test SLURM initialization
            # os.environ[""] = "SLURM"
            DistributedManager.initialize()
            manager = DistributedManager()
            assert manager.is_initialized()
            assert manager._initialization_method == "slurm"
            assert manager.distributed == torch.distributed.is_available(), (
                "Manager should be in serial mode"
            )
            assert manager.rank == 0
            assert manager.world_size == 1
            assert manager.local_rank == 0
            DistributedManager.cleanup()

        # Test OpenMPI initialization
        # OpenMPI env vars
        with modify_environment(
            OMPI_COMM_WORLD_RANK="0",
            OMPI_COMM_WORLD_SIZE="1",
            OMPI_COMM_WORLD_LOCAL_RANK="0",
            PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD="OPENMPI",
        ):
            DistributedManager.initialize()
            manager = DistributedManager()
            assert manager.is_initialized()
            assert manager._initialization_method == "openmpi"
            assert manager.distributed == torch.distributed.is_available(), (
                "Manager should be in serial mode"
            )
            assert manager.rank == 0
            assert manager.world_size == 1
            assert manager.local_rank == 0
            DistributedManager.cleanup()


def test_manager_singleton():
    # Test distributed manager singleton functions as expected
    with modify_environment(
        MASTER_ADDR="localhost",
        MASTER_PORT="45678",
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
    ):
        DistributedManager.initialize()

        manager_1 = DistributedManager()
        manager_1.broadcast_buffers = True
        manager_1.find_unused_parameters = True
        manager_2 = DistributedManager()

        # Compare attributes
        assert manager_1.rank == manager_2.rank
        assert manager_1.world_size == manager_2.world_size
        assert manager_1.local_rank == manager_2.local_rank
        assert manager_1.device == manager_2.device
        assert manager_1.distributed == manager_2.distributed
        assert manager_1.cuda == manager_2.cuda
        assert manager_1.group_names == manager_2.group_names
        assert manager_1.group() == manager_2.group()
        assert manager_1.group_size() == manager_2.group_size()
        assert manager_1.group_rank() == manager_2.group_rank()
        assert manager_1.group_name() == manager_2.group_name()
        assert manager_1.broadcast_buffers == manager_2.broadcast_buffers
        assert manager_1.find_unused_parameters == manager_2.find_unused_parameters
        DistributedManager.cleanup()


def test_manager_uninitialized_instantiation():
    with modify_environment(
        MASTER_ADDR="localhost",
        MASTER_PORT="12345",
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
    ):
        assert not DistributedManager.is_initialized()

        with pytest.raises(PhysicsNeMoUninitializedDistributedManagerWarning):
            DistributedManager()

        DistributedManager._shared_state = {}


def test_manager_undefined_group_query():
    with modify_environment(
        MASTER_ADDR="localhost",
        MASTER_PORT="12345",
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
    ):
        DistributedManager.initialize()

        manager = DistributedManager()

        assert manager.is_initialized()

        with pytest.raises(PhysicsNeMoUndefinedGroupError):
            manager.group("undefined_group")
        with pytest.raises(PhysicsNeMoUndefinedGroupError):
            manager.group_size("undefined_group")
        with pytest.raises(PhysicsNeMoUndefinedGroupError):
            manager.group_rank("undefined_group")

        DistributedManager.cleanup()


@pytest.mark.multigpu_dynamic
def test_manager_single_process_subgroups():
    with modify_environment(
        RANK="0",
        WORLD_SIZE="1",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(12375),
        LOCAL_RANK="0",
    ):
        DistributedManager.initialize()

        verbose = False

        # Create model parallel process group
        DistributedManager.create_process_subgroup("model_parallel", 1, verbose=verbose)
        # Create data parallel process group for DDP allreduce
        DistributedManager.create_orthogonal_process_group(
            "data_parallel", "model_parallel", verbose=verbose
        )

        manager = DistributedManager()

        # Test that trivial case of a single GPU still works
        assert manager.rank == 0
        assert manager.group_rank(name="model_parallel") == 0
        assert manager.group_rank(name="data_parallel") == 0
        assert manager.group_size("model_parallel") == 1
        assert manager.group_size("data_parallel") == 1
        DistributedManager.cleanup()


def run_process_groups(rank, model_parallel_size, verbose):
    with modify_environment(
        RANK=f"{rank}",
        WORLD_SIZE=f"{model_parallel_size}",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(12365),
        LOCAL_RANK=f"{rank % torch.cuda.device_count()}",
    ):
        DistributedManager.initialize()

        # Create model parallel process group
        DistributedManager.create_process_subgroup(
            "model_parallel", int(model_parallel_size), verbose=verbose
        )
        # Create data parallel process group for DDP allreduce
        DistributedManager.create_orthogonal_process_group(
            "data_parallel", "model_parallel", verbose=verbose
        )

        manager = DistributedManager()

        assert manager.rank == rank
        assert manager.rank == manager.group_rank(name="model_parallel")
        assert 0 == manager.group_rank(name="data_parallel")
        DistributedManager.cleanup()


@pytest.mark.multigpu_dynamic
def test_process_groups():
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"
    model_parallel_size = num_gpus
    verbose = False  # Change to True for debug

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_process_groups,
        args=(model_parallel_size, verbose),
        nprocs=model_parallel_size,
        join=True,
        daemon=True,
    )


def run_process_groups_from_config(rank, model_parallel_size, verbose):
    with modify_environment(
        RANK=f"{rank}",
        LOCAL_RANK=f"{rank % torch.cuda.device_count()}",
        WORLD_SIZE=f"{model_parallel_size}",
        MASTER_ADDR="localhost",
        MASTER_PORT="13246",
    ):
        DistributedManager.initialize()
        dm = DistributedManager()
        assert dm.is_initialized()

        # Create world group that contains all processes that are part of this job
        world = ProcessGroupNode("world")

        # Create the process group config with the highest level process group
        config = ProcessGroupConfig(world)

        # Create model and data parallel sub-groups
        config.add_node(ProcessGroupNode("model_parallel"), parent="world")
        config.add_node(ProcessGroupNode("data_parallel"), parent="world")

        # Create spatial and channel parallel sub-groups
        config.add_node(ProcessGroupNode("spatial_parallel"), parent="model_parallel")
        config.add_node(ProcessGroupNode("channel_parallel"), parent="model_parallel")

        # Set leaf group sizes
        group_sizes = {
            "channel_parallel": 1,
            "spatial_parallel": model_parallel_size,
            "data_parallel": 1,
        }
        config.set_leaf_group_sizes(group_sizes)  # Updates all parent group sizes too

        assert config.get_node("model_parallel").size == model_parallel_size, (
            "Incorrect size for 'model_parallel' parent node"
        )

        assert config.get_node("world").size == model_parallel_size, (
            "Incorrect size for 'world' parent node"
        )

        # Create model parallel process group
        DistributedManager.create_groups_from_config(config, verbose=verbose)

        manager = DistributedManager()

        assert manager.rank == rank

        # Test that model_parallel and spatial_parallel span all the processes
        assert manager.rank == manager.group_rank(name="model_parallel")
        assert manager.rank == manager.group_rank(name="spatial_parallel")

        # Test orthogonal data_parallel group, only one total model_parallel group so
        # data_parallel rank should always be 0
        assert 0 == manager.group_rank(name="data_parallel")

        # Test channel_parallel group, group with size 1, so rank must be 0
        assert 0 == manager.group_rank(name="channel_parallel")

        # Cleanup process groups
        DistributedManager.cleanup()


@pytest.mark.multigpu_dynamic
def test_process_groups_from_config():
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"
    model_parallel_size = num_gpus
    verbose = False  # Change to True for debug

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_process_groups_from_config,
        args=(model_parallel_size, verbose),
        nprocs=model_parallel_size,
        join=True,
        daemon=True,
    )


if __name__ == "__main__":
    pytest.main([__file__])
