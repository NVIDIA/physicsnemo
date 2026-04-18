# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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
Test unbind operations on ShardTensor.  We use a 3D tensor sharded along
dim 2 and test unbinding along non-sharded dimensions.  Both forward
correctness and backward gradient flow are verified.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor

from .utils import numerical_shard_tensor_check


class UnbindSelectWrapper(torch.nn.Module):
    """
    Wrapper that unbinds a tensor and returns a single element from the
    result tuple.  This allows reuse of ``numerical_shard_tensor_check``
    which expects a single tensor output.
    """

    def __init__(self, dim: int, index: int):
        super().__init__()
        self.dim = dim
        self.index = index

    def forward(self, tensor: torch.Tensor):
        pieces = torch.unbind(tensor, self.dim)
        return pieces[self.index]


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("unbind_dim,index", [(0, 0), (0, 2), (1, 3), (-3, 0), (-2, 3)])
def test_unbind(distributed_mesh, backward, unbind_dim, index):
    """Verify forward and backward via ``numerical_shard_tensor_check``."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 6, 128)
    placements = (Shard(2),)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=True,
    )

    module = UnbindSelectWrapper(dim=unbind_dim, index=index)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


# -- Error tests --------------------------------------------------------------


@pytest.mark.multigpu_static
def test_unbind_along_sharded_dim(distributed_mesh):
    """Unbinding along the sharded dimension should raise."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 6, 128)
    placements = (Shard(2),)

    original_tensor = torch.rand(shape, device=dm.device)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=False,
    )

    with pytest.raises(RuntimeError, match="unbinding along sharding axis"):
        torch.unbind(sharded_tensor, 2)
