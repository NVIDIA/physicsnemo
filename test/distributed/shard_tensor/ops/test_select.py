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

"""
Test selection operations on ShardTensor.  This file tests
both torch.select and torch.index_select.  We use a 3D tensor to
do the tests, it has no special significance.  We're not testing
over all possible selection dimensions, especially not along sharded dimensions.

That could be implemented in the future.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.distributed.shard_tensor import scatter_tensor

from .utils import numerical_shard_tensor_check


class SelectWrapper(torch.nn.Module):
    """
    Wrapper class for testing torch.select operation.
    """

    def __init__(self, target_dim: int, index: int):
        super(SelectWrapper, self).__init__()
        self.target_dim = target_dim
        self.index = index

    def forward(self, tensor: torch.Tensor):
        return torch.select(tensor, self.target_dim, self.index)


class IndexSelectWrapper(torch.nn.Module):
    """
    Wrapper class for testing torch.index_select operation.
    """

    def __init__(self, target_dim: int):
        super(IndexSelectWrapper, self).__init__()
        self.target_dim = target_dim

    def forward(self, tensor: torch.Tensor, index: torch.Tensor):
        return torch.index_select(tensor, self.target_dim, index.flatten())


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_select_operation(
    distributed_mesh,
    backward,
):
    """Test basic scaled dot product attention with various configurations"""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (128, 128, 128)
    target_dim = 1
    index = 2

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(2),)

    # Scatter the original tensor and index to all ranks
    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=True,
    )

    module = SelectWrapper(target_dim=target_dim, index=index)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [
            sharded_tensor,
        ],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_index_select_operation(
    distributed_mesh,
    backward,
):
    """Test basic scaled dot product attention with various configurations"""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (128, 128, 128)
    target_dim = 1
    N = 256

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)
    index = torch.randint(
        low=0, high=shape[target_dim] - 1, size=(N,), device=dm.device
    ).reshape(int(N / 2), -1)

    placements = (Shard(2),)

    # Scatter the original tensor and index to all ranks
    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=True,
    )
    sharded_index = scatter_tensor(
        index,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(0),),
        requires_grad=False,
    )

    module = IndexSelectWrapper(target_dim=target_dim)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor, sharded_index],
        {},
        check_grads=backward,
    )
