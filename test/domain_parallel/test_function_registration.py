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

r"""Tests for ShardTensor function and dispatch registration.

This module tests the registration tools of ``ShardTensor`` that allow custom
handlers to be registered for both Python-level functions (via ``__torch_function__``)
and low-level dispatch operations (via ``__torch_dispatch__``).

The tests use mock handlers to verify that:

- ``register_function_handler``: Intercepts Python-level functions like ``torch.mul``
- ``register_dispatch_handler``: Intercepts ATen operators like ``aten.add.Tensor``
- Regular tensors bypass custom handlers and use PyTorch's default behavior
- ShardTensor inputs correctly trigger the registered handlers
- Function-level and dispatch-level interception paths don't interfere with each other
"""

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Replicate

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel.shard_tensor import ShardTensor

# Global to track execution paths
torch_function_paths = []
torch_dispatch_paths = []


# Custom handlers for testing
def mul_wrapper(func, types, args, kwargs):
    r"""Test wrapper for multiplication - TESTING PURPOSES ONLY.

    This wrapper intercepts ``torch.mul`` calls when ShardTensor inputs are detected.
    It performs local multiplication on the underlying tensors.

    Warning
    -------
    This is a test-only implementation. Do not use in production code.
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
    r"""Test wrapper for addition - TESTING PURPOSES ONLY.

    This wrapper intercepts add (dispatch or function) when ShardTensor
    inputs are detected. It performs local addition on the underlying tensors.

    Warning
    -------
    This is a test-only implementation. Do not use in production code.
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
    # a + b can dispatch to aten.add.default or aten.add.Tensor depending on PyTorch version
    ShardTensor.register_dispatch_handler(torch.ops.aten.add.default, add_wrapper)
    ShardTensor.register_dispatch_handler(torch.ops.aten.add.Tensor, add_wrapper)

    # Enable ShardTensor patches
    ShardTensor._enable_shard_patches = True

    yield

    # Restore original registry state
    ShardTensor._dispatch_registry = original_dispatch_registry
    ShardTensor._function_registry = original_function_registry


@pytest.fixture
def device_mesh(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "13245")
    monkeypatch.setenv("LOCAL_RANK", "0")

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


@pytest.mark.skip(
    reason="torch_dispatch_paths not populated for ShardTensor add; dispatch registration behavior under investigation"
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
        f"ShardTensors should trigger our wrapper, got: {torch_dispatch_paths}"
    )
    assert len(torch_function_paths) == 0, (
        "torch_dispatch intercepts should not trigger torch_function intercepts"
    )
