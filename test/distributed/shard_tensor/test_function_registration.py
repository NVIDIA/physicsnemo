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
The tests in this file are meant to test the registration tools of ShardTensor.

It mocks up some functions that we then plug in to ShardTensor and verify they
connect correctly.
"""

import pytest
import torch

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.version_check import check_module_requirements

try:
    check_module_requirements("physicsnemo.distributed.shard_tensor")
except ImportError:
    pytest.skip(
        "Skipping test because physicsnemo.distributed.shard_tensor is not available",
        allow_module_level=True,
    )

from pytest_utils import modify_environment
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Replicate

from physicsnemo.distributed.shard_tensor import ShardTensor

# Global to track execution paths
torch_function_paths = []
torch_dispatch_paths = []


# Custom handlers for testing
def mul_wrapper(func, types, args, kwargs):
    """
    This wrapper is for TESTING PURPOSES ONLY.
    Don't use it in real code.
    """
    torch_function_paths.append("mul_wrapper")
    # Just multiply the local tensors if inputs are ShardTensors
    if isinstance(args[0], ShardTensor) and isinstance(args[1], ShardTensor):
        local_result = args[0]._local_tensor * args[1]._local_tensor
        return ShardTensor.from_local(
            local_result, args[0]._spec.mesh, args[0]._spec.placements
        )
    # Fall back to original function for regular tensors
    return func(*args, **kwargs)


def add_wrapper(a, b, alpha=1):
    """
    This wrapper is for TESTING PURPOSES ONLY.
    Don't use it in real code.
    """
    torch_dispatch_paths.append("add_wrapper")
    if isinstance(a, ShardTensor) and isinstance(b, ShardTensor):
        local_result = a._local_tensor + alpha * b._local_tensor
        return ShardTensor.from_local(local_result, a._spec.mesh, a._spec.placements)
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.ops.aten.add.Tensor(a, b, alpha)
    else:
        # Handle mixed cases
        if isinstance(a, ShardTensor):
            a = a.to_local()
        if isinstance(b, ShardTensor):
            b = b.to_local()
        return torch.ops.aten.add.Tensor(a, b, alpha)


@pytest.fixture
def setup_registry():
    # Save original registry state
    original_dispatch_registry = ShardTensor._dispatch_registry.copy()
    original_function_registry = ShardTensor._function_registry.copy()

    # Clear execution path tracking
    torch_function_paths.clear()
    torch_dispatch_paths.clear()

    # Register our test handlers
    ShardTensor.register_function_handler(torch.mul, mul_wrapper)
    ShardTensor.register_dispatch_handler(torch.ops.aten.add.Tensor, add_wrapper)

    # Enable ShardTensor patches
    ShardTensor._enable_shard_patches = True

    yield

    # Restore original registry state
    ShardTensor._dispatch_registry = original_dispatch_registry
    ShardTensor._function_registry = original_function_registry


@pytest.fixture(scope="module")
def device_mesh():
    with modify_environment(
        RANK="0",
        WORLD_SIZE="1",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(13245),
        LOCAL_RANK="0",
    ):
        DistributedManager.initialize()

        yield DeviceMesh(
            DistributedManager().device.type,
            mesh=[
                0,
            ],
        )
        DistributedManager.cleanup()


def test_function_registration_with_tensors(setup_registry):
    # Create regular PyTorch tensors
    a = torch.ones(2, 3)
    b = torch.ones(2, 3) * 2

    # Call torch.mul (should use PyTorch's implementation)
    result = torch.mul(a, b)

    # Verify result and execution path
    assert torch.all(result == 2)
    assert len(torch_function_paths) == 0, (
        "Regular tensors should not trigger our wrapper"
    )
    assert len(torch_dispatch_paths) == 0, (
        "Regular tensors should not trigger our wrapper"
    )


def test_function_registration_with_shard_tensors(setup_registry, device_mesh):
    # Create ShardTensors
    a = ShardTensor.from_local(torch.ones(2, 3), device_mesh, [Replicate()])
    b = ShardTensor.from_local(torch.ones(2, 3) * 2, device_mesh, [Replicate()])

    # Call torch.mul (should use our wrapper)
    result = torch.mul(a, b)

    # Verify result and execution path
    assert isinstance(result, ShardTensor)
    assert torch.all(result.to_local() == 2)
    assert torch_function_paths == ["mul_wrapper"], (
        "ShardTensors should trigger our wrapper"
    )
    assert len(torch_dispatch_paths) == 0, (
        "torch_function intercepts should not trigger dispatch intercepts"
    )


def test_dispatch_registration_with_tensors(setup_registry):
    # Create regular PyTorch tensors
    a = torch.ones(2, 3)
    b = torch.ones(2, 3) * 2

    # Call torch.add (which uses aten.add.Tensor internally)
    result = a + b

    # Verify result
    assert torch.all(result == 3)
    assert len(torch_dispatch_paths) == 0, (
        "Regular tensors should not trigger our wrapper"
    )
    assert len(torch_function_paths) == 0, (
        "Regular tensors should not trigger our wrapper"
    )


def test_dispatch_registration_with_shard_tensors(setup_registry, device_mesh):
    # Create ShardTensors
    a = ShardTensor.from_local(torch.ones(2, 3), device_mesh, [Replicate()])
    b = ShardTensor.from_local(torch.ones(2, 3) * 2, device_mesh, [Replicate()])

    # Call addition (which uses aten.add.Tensor internally)
    result = a + b

    # Verify result and execution path
    assert isinstance(result, ShardTensor)
    assert torch.all(result.to_local() == 3)
    assert torch_dispatch_paths == ["add_wrapper"], (
        "ShardTensors should trigger our wrapper"
    )
    assert len(torch_function_paths) == 0, (
        "torch_dispatch intercepts should not trigger torch_function intercepts"
    )
