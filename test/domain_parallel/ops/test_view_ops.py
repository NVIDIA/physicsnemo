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
Test view and reshape operations on ShardTensor.

Tests cover tensor.view, tensor.reshape, and torch.reshape with sharding
on various dimensions. The shard dimension is never the one being merged
or split — it is preserved 1:1 through the view, or the view operates
exclusively on non-sharded dimensions.

Backward (gradient) correctness is tested for every configuration.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor

from .utils import numerical_shard_tensor_check


class ViewWrapper(torch.nn.Module):
    """Wrapper class for testing tensor.view operation."""

    def __init__(self, target_shape: tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tensor: torch.Tensor):
        return tensor.view(self.target_shape)


class ReshapeWrapper(torch.nn.Module):
    """Wrapper class for testing tensor.reshape operation."""

    def __init__(self, target_shape: tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tensor: torch.Tensor):
        return tensor.reshape(self.target_shape)


class TorchReshapeWrapper(torch.nn.Module):
    """Wrapper class for testing torch.reshape operation."""

    def __init__(self, target_shape: tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tensor: torch.Tensor):
        return torch.reshape(tensor, self.target_shape)


class ViewRoundTrip(torch.nn.Module):
    """View to merge last two dims, then view back to the original shape.

    Exercises view in a differentiable pipeline so gradients must flow
    back through two consecutive view backward passes.
    """

    def __init__(self, original_shape: tuple[int, ...]):
        super().__init__()
        self.original_shape = original_shape

    def forward(self, tensor: torch.Tensor):
        b, t = tensor.shape[:2]
        merged = tensor.reshape(b, t, -1)
        return merged.view(self.original_shape)


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_view_merge_last_two_dims(
    distributed_mesh,
    backward,
):
    """Test tensor.view merging the last two dims (einops-like 'b t h d -> b t (h d)')."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 128, 8, 4)
    target_shape = (4, 128, 32)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(1),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = ViewWrapper(target_shape=target_shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_view_split_last_dim(
    distributed_mesh,
    backward,
):
    """Test tensor.view splitting the last dim (einops-like 'b t (h d) -> b t h d')."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 128, 32)
    target_shape = (4, 128, 8, 4)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(1),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = ViewWrapper(target_shape=target_shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_view_flatten_to_2d(
    distributed_mesh,
    backward,
):
    """Test tensor.view flattening spatial dims into one."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 128, 8)
    target_shape = (4, -1)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(1),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = ViewWrapper(target_shape=target_shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_view_neg1_infer(
    distributed_mesh,
    backward,
):
    """Test tensor.view with -1 dimension inference."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 128, 8, 4)
    target_shape = (4, -1, 32)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(1),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = ViewWrapper(target_shape=target_shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_reshape_merge_last_two_dims(
    distributed_mesh,
    backward,
):
    """Test tensor.reshape merging the last two dims."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 128, 8, 4)
    target_shape = (4, 128, 32)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(1),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = ReshapeWrapper(target_shape=target_shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_torch_reshape_operation(
    distributed_mesh,
    backward,
):
    """Test torch.reshape(tensor, shape) on a ShardTensor."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 128, 8, 4)
    target_shape = (4, 128, 32)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(1),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = TorchReshapeWrapper(target_shape=target_shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_view_shard_on_non_viewed_dim(
    distributed_mesh,
    backward,
):
    """Test view when shard dim is not involved in the reshape at all."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 128, 8, 4)
    target_shape = (4, 128, 32)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    # Shard on dim 0 (batch) — the view only touches dims 2+3.
    placements = (Shard(0),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = ViewWrapper(target_shape=target_shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_view_round_trip(
    distributed_mesh,
    backward,
):
    """Test that gradients flow through two consecutive views (merge then split)."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (2, 64, 8, 4)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(1),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = ViewRoundTrip(original_shape=shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )
