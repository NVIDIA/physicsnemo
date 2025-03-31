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
    ProcessGroupConfig,
    ProcessGroupNode,
)
from physicsnemo.distributed.mappings import reduce_from_parallel_region


def test_config():
    # Create world group that contains all processes that are part of this job
    world = ProcessGroupNode("world")

    # Create the process group config with the highest level process group
    config = ProcessGroupConfig(world)

    # Create model and data parallel sub-groups
    # Sub-groups of a single node are guaranteed to be orthogonal by construction
    config.add_node(ProcessGroupNode("model_parallel"), parent=world)
    config.add_node(ProcessGroupNode("data_parallel"), parent="world")

    # Create spatial and channel parallel sub-groups
    config.add_node(ProcessGroupNode("spatial_parallel"), parent="model_parallel")
    config.add_node(ProcessGroupNode("channel_parallel"), parent="model_parallel")

    # Now check that the leaf nodes are correct
    assert sorted(config.leaf_groups()) == [
        "channel_parallel",
        "data_parallel",
        "spatial_parallel",
    ]

    # Set leaf group sizes
    group_sizes = {"channel_parallel": 3, "spatial_parallel": 2, "data_parallel": 4}
    config.set_leaf_group_sizes(group_sizes)  # Update all parent group sizes too

    assert (
        config.get_node("model_parallel").size == 6
    ), "Incorrect size for 'model_parallel' parent node"

    assert config.get_node("world").size == 24, "Incorrect size for 'world' parent node"


class MockDistributedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.manager = DistributedManager()
        self.alpha = torch.nn.Parameter(data=torch.tensor(0.5), requires_grad=True)
        self.group = "model_parallel"

    def forward(self, x):
        return reduce_from_parallel_region(self.alpha * x, self.group)

    @staticmethod
    def get_process_group_config() -> ProcessGroupConfig:
        world = ProcessGroupNode("world")
        config = ProcessGroupConfig(world)

        # Create model and data parallel sub-groups
        config.add_node(ProcessGroupNode("model_parallel"), parent="world")
        config.add_node(ProcessGroupNode("data_parallel"), parent="world")

        return config


def run_distributed_model_config(rank, model_parallel_size, verbose):
    print(f"Entered function with rank {rank}")
    with modify_environment(
        RANK=f"{rank}",
        WORLD_SIZE=f"{model_parallel_size}",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(12355),
        LOCAL_RANK=f"{rank % torch.cuda.device_count()}",
    ):

        DistributedManager._shared_state = {}

        DistributedManager.initialize()
        print(f"Initialized DistributedManager with rank {DistributedManager().rank}")

        # Query model for the process group config
        config = MockDistributedModel.get_process_group_config()

        # Set leaf group sizes
        group_sizes = {"model_parallel": 2, "data_parallel": 1}
        config.set_leaf_group_sizes(group_sizes)  # Updates all parent group sizes too

        assert (
            config.get_node("model_parallel").size == 2
        ), "Incorrect size for 'model_parallel' parent node"

        assert (
            config.get_node("world").size == 2
        ), "Incorrect size for 'world' parent node"

        # Create model parallel process group
        DistributedManager.create_groups_from_config(config, verbose=verbose)

        manager = DistributedManager()

        assert manager.rank == rank
        assert manager.rank == manager.group_rank(name="model_parallel")
        assert 0 == manager.group_rank(name="data_parallel")

        # Now actually instantiate the model
        model = MockDistributedModel().to(manager.device)
        x = torch.randn(1, device=manager.device)
        y = model(x)
        loss = y.sum()
        loss.backward()

        if verbose:
            print(
                f"{manager.group_rank('model_parallel')}: {[p.grad for p in model.parameters()]}, x: {x}, y: {y}"
            )
        # Test that the output of the model is correct
        y_true = 0.5 * torch.clone(x)
        torch.distributed.all_reduce(y_true)
        assert torch.allclose(y, y_true, rtol=1e-05, atol=1e-08)

        # Check that the backward pass produces the right result
        for p in model.parameters():
            assert torch.allclose(p.grad, x, rtol=1e-05, atol=1e-08)

        # Cleanup process groups
        DistributedManager.cleanup()


@pytest.mark.multigpu
def test_distributed_model_config():
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"
    model_parallel_size = 2
    verbose = False  # Change to True for debug

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_distributed_model_config,
        args=(model_parallel_size, verbose),
        nprocs=model_parallel_size,
        join=True,
        daemon=True,
    )


if __name__ == "__main__":
    test_distributed_model_config()
