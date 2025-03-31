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

import os

import pytest
import torch
import torch.nn as nn
from distributed_utils_for_testing import modify_environment

from physicsnemo.distributed import (
    DistributedManager,
    mark_module_as_shared,
    reduce_loss,
    unmark_module_as_shared,
)
from physicsnemo.distributed.utils import _reduce


def test_modify_environment():

    keys = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    # Set the values to nonsense for testing:
    values = [f"{i}" for i in range(len(keys))]

    key_values = {k: v for k, v in zip(keys, values)}
    print(key_values)

    current_val = {key: os.environ.get(key, "NOT_SET") for key in keys}

    with modify_environment(**key_values):
        for key, value in zip(keys, values):
            assert os.environ[key] == value

    # Make sure the values are restored:
    for key, value in current_val.items():
        if current_val[key] == "NOT_SET":
            assert key not in os.environ
        else:
            assert os.environ[key] == value

    # assert False


def run_test_reduce_loss(rank, world_size):

    with modify_environment(
        RANK=f"{rank}",
        WORLD_SIZE=f"{world_size}",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(12355),
        LOCAL_RANK=f"{rank % torch.cuda.device_count()}",
    ):
        # Reset class state
        DistributedManager._shared_state = {}
        DistributedManager.initialize()

        manager = DistributedManager()
        assert manager.is_initialized()

        loss = reduce_loss(1.0, dst_rank=0, mean=False)
        if manager.local_rank == 0:
            assert loss == 1.0 * world_size, str(loss)
        else:
            assert True

        DistributedManager.cleanup()


def run_test_mark_shared(rank, world_size):
    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin_1 = nn.Linear(4, 2)
            self.lin_2 = nn.Linear(2, 4)

        def forward(self, x):
            return torch.sigmoid(self.lin_2(torch.tanh(self.lin_1(x))))

    with modify_environment(
        RANK=f"{rank}",
        WORLD_SIZE=f"{world_size}",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(12355),
        LOCAL_RANK=f"{rank % torch.cuda.device_count()}",
    ):

        DistributedManager._shared_state = {}
        DistributedManager.initialize()
        DistributedManager.create_process_subgroup(
            name="shared_parallel",
            size=world_size,
        )
        manager = DistributedManager()
        assert manager.is_initialized()

        torch.manual_seed(42 * world_size + rank)
        ref_module = TestModule().to(device=manager.device)
        torch.manual_seed(42 * world_size + rank)
        dist_module = TestModule().to(device=manager.device)
        x = torch.ones(4, device=manager.device)
        ref_out = ref_module(x)
        ref_out.backward(torch.ones_like(ref_out))
        ref_lin_1_weight_grad = _reduce(
            ref_module.lin_1.weight.grad.clone().detach(),
            group=manager.group("shared_parallel"),
            use_fp32=True,
        )
        ref_lin_1_bias_grad = _reduce(
            ref_module.lin_1.bias.grad.clone().detach(),
            group=manager.group("shared_parallel"),
            use_fp32=True,
        )

        # mark lin_1 as shared, lin_2 is not touched
        mark_module_as_shared(dist_module.lin_1, "shared_parallel")
        dist_module.zero_grad()
        dist_out = dist_module(x)
        dist_out.backward(torch.ones_like(dist_out))
        assert torch.allclose(ref_out, dist_out)
        assert torch.allclose(
            ref_module.lin_2.weight.grad, dist_module.lin_2.weight.grad
        )
        assert torch.allclose(ref_module.lin_2.bias.grad, dist_module.lin_2.bias.grad)
        assert torch.allclose(ref_lin_1_weight_grad, dist_module.lin_1.weight.grad)
        assert torch.allclose(ref_lin_1_bias_grad, dist_module.lin_1.bias.grad)

        ref_lin_2_weight_grad = _reduce(
            ref_module.lin_2.weight.grad.clone().detach(),
            group=manager.group("shared_parallel"),
            use_fp32=True,
        )
        ref_lin_2_bias_grad = _reduce(
            ref_module.lin_2.bias.grad.clone().detach(),
            group=manager.group("shared_parallel"),
            use_fp32=True,
        )

        # unmark lin_1 as shared (umarking lin_2 should throw an error)
        with pytest.raises(RuntimeError):
            unmark_module_as_shared(dist_module.lin_2)
        unmark_module_as_shared(dist_module.lin_1)
        dist_module.zero_grad()
        dist_out = dist_module(x)
        dist_out.backward(torch.ones_like(dist_out))
        assert torch.allclose(ref_out, dist_out)
        assert torch.allclose(
            ref_module.lin_2.weight.grad, dist_module.lin_2.weight.grad
        )
        assert torch.allclose(ref_module.lin_2.bias.grad, dist_module.lin_2.bias.grad)
        assert torch.allclose(
            ref_module.lin_1.weight.grad, dist_module.lin_1.weight.grad
        )
        assert torch.allclose(ref_module.lin_1.bias.grad, dist_module.lin_1.bias.grad)

        # mark lin_2 as shared
        mark_module_as_shared(dist_module.lin_2, "shared_parallel")
        dist_module.zero_grad()
        dist_out = dist_module(x)
        dist_out.backward(torch.ones_like(dist_out))
        assert torch.allclose(ref_lin_2_weight_grad, dist_module.lin_2.weight.grad)
        assert torch.allclose(ref_lin_2_bias_grad, dist_module.lin_2.bias.grad)
        assert torch.allclose(
            ref_module.lin_1.weight.grad, dist_module.lin_1.weight.grad
        )
        assert torch.allclose(ref_module.lin_1.bias.grad, dist_module.lin_1.bias.grad)

        # unmark lin_2 again (unmarking lin_1 should throw an error)
        with pytest.raises(RuntimeError):
            unmark_module_as_shared(dist_module.lin_1)

        unmark_module_as_shared(dist_module.lin_2)
        dist_module.zero_grad()
        dist_out = dist_module(x)
        dist_out.backward(torch.ones_like(dist_out))
        assert torch.allclose(ref_out, dist_out)
        assert torch.allclose(
            ref_module.lin_2.weight.grad, dist_module.lin_2.weight.grad
        )
        assert torch.allclose(ref_module.lin_2.bias.grad, dist_module.lin_2.bias.grad)
        assert torch.allclose(
            ref_module.lin_1.weight.grad, dist_module.lin_1.weight.grad
        )
        assert torch.allclose(ref_module.lin_1.bias.grad, dist_module.lin_1.bias.grad)

        # mark whole module as shared, but don't recurse
        # in this set, this should result in parameters behaving
        # as they would not be shared
        mark_module_as_shared(dist_module, "shared_parallel", recurse=False)
        dist_module.zero_grad()
        dist_out = dist_module(x)
        dist_out.backward(torch.ones_like(dist_out))
        assert torch.allclose(ref_out, dist_out)
        assert torch.allclose(
            ref_module.lin_2.weight.grad, dist_module.lin_2.weight.grad
        )
        assert torch.allclose(ref_module.lin_2.bias.grad, dist_module.lin_2.bias.grad)
        assert torch.allclose(
            ref_module.lin_1.weight.grad, dist_module.lin_1.weight.grad
        )
        assert torch.allclose(ref_module.lin_1.bias.grad, dist_module.lin_1.bias.grad)

        # test recurse in unmark and unmark whole model for final test
        with pytest.raises(RuntimeError):
            unmark_module_as_shared(dist_module, recurse=True)
        unmark_module_as_shared(dist_module, recurse=False)

        # mark whole module as shared (both layers now should be shared)
        mark_module_as_shared(dist_module, "shared_parallel", recurse=True)
        dist_module.zero_grad()
        dist_out = dist_module(x)
        dist_out.backward(torch.ones_like(dist_out))
        assert torch.allclose(ref_lin_2_weight_grad, dist_module.lin_2.weight.grad)
        assert torch.allclose(ref_lin_2_bias_grad, dist_module.lin_2.bias.grad)
        assert torch.allclose(ref_lin_1_weight_grad, dist_module.lin_1.weight.grad)
        assert torch.allclose(ref_lin_1_bias_grad, dist_module.lin_1.bias.grad)

        DistributedManager.cleanup()


@pytest.mark.multigpu
def test_reduce_loss():
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 1
    world_size = num_gpus

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_test_reduce_loss,
        args=(world_size,),
        nprocs=world_size,
        join=True,
        daemon=True,
    )


@pytest.mark.multigpu
def test_mark_shared():
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 1
    world_size = num_gpus

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_test_mark_shared,
        args=(world_size,),
        nprocs=world_size,
        join=True,
        daemon=True,
    )
