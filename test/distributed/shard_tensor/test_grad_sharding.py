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

from physicsnemo.utils.version_check import check_module_requirements

try:
    check_module_requirements("physicsnemo.distributed.shard_tensor")

except ImportError:
    pytest.skip(
        "Skipping test because physicsnemo.distributed.shard_tensor is not available",
        allow_module_level=True,
    )

from test_shard_tensor_redistribute import shard_tensor_factory

from physicsnemo.distributed import ShardTensor


def run_shard_tensor_detach(mesh, uneven, verbose):
    shard_tensor = shard_tensor_factory(mesh, uneven=uneven)
    shard_tensor_detached = shard_tensor.detach()

    # Detaching should not change the original data nor should it change the spec:
    assert shard_tensor._spec == shard_tensor_detached._spec

    assert torch.allclose(
        shard_tensor.full_tensor(), shard_tensor_detached.full_tensor()
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
@pytest.mark.parametrize("uneven", [True, False])
def test_shard_tensor_detach(distributed_mesh, uneven):
    run_shard_tensor_detach(distributed_mesh, uneven, verbose=False)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
@pytest.mark.parametrize("uneven", [True, False])
def test_shard_tensor_detach_2d(distributed_mesh_2d, uneven):
    run_shard_tensor_detach(distributed_mesh_2d, uneven, verbose=False)


def run_shard_tensor_input_gradient_full_loss(mesh, uneven, verbose):
    shard_tensor = shard_tensor_factory(mesh, uneven)

    shard_tensor = shard_tensor.detach().requires_grad_(
        True
    )  # Make it a leaf tensor by calling detach andrequires_grad_

    # For this test, we're testing that the gradients of the input tensor work
    # We'll compare them to the local gradients

    # Compute the input gradients on the full_tensor:
    full_local_tensor = shard_tensor.full_tensor().detach()
    full_local_tensor.requires_grad_(True)

    def loss(_input):
        if isinstance(_input, ShardTensor):
            x = _input.full_tensor()
        else:
            x = _input
        x = x**2
        return torch.sum(x)

    computed_local_loss = loss(full_local_tensor)
    computed_local_loss.backward()

    # This should have gradients
    assert full_local_tensor.grad is not None

    # Now compute the sharded gradients with FULL TENSOR LOSS:
    sharded_loss = loss(shard_tensor)
    sharded_loss.backward()

    # Check if shard_tensor requires grad
    assert shard_tensor.requires_grad, "ShardTensor should require grad"
    assert shard_tensor.grad is not None
    assert torch.allclose(shard_tensor.grad.full_tensor(), full_local_tensor.grad)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
@pytest.mark.parametrize("uneven", [True, False])
def test_shard_tensor_input_gradient_full_loss(distributed_mesh, uneven):
    run_shard_tensor_input_gradient_full_loss(distributed_mesh, uneven, verbose=False)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
@pytest.mark.parametrize("uneven", [True, False])
def test_shard_tensor_input_gradient_full_loss_2d(distributed_mesh_2d, uneven):
    run_shard_tensor_input_gradient_full_loss(
        distributed_mesh_2d, uneven, verbose=False
    )


def run_shard_tensor_input_gradient_local_loss(mesh, uneven, verbose):
    shard_tensor = shard_tensor_factory(mesh, uneven)

    # shard_tensor = (
    #     shard_tensor.detach()
    # )  # Make it a leaf tensor by calling detach andrequires_grad_
    shard_tensor = shard_tensor.detach().requires_grad_(
        True
    )  # Make it a leaf tensor by calling detach andrequires_grad_

    # For this test, we're testing that the gradients of the input tensor work
    # We'll compare them to the local gradients

    # Compute the input gradients on the full_tensor:
    full_local_tensor = shard_tensor.full_tensor().detach()
    full_local_tensor.requires_grad_(True)

    def loss(_input):
        # Compute the loss *locally*
        if isinstance(_input, ShardTensor):
            x = _input.to_local()
        else:
            x = _input
        x = x**2
        return torch.sum(x)

    computed_local_loss = loss(full_local_tensor)
    computed_local_loss.backward()

    # This should have gradients
    assert full_local_tensor.grad is not None

    # Now compute the sharded gradients:
    sharded_loss = loss(shard_tensor)

    sharded_loss.backward()

    # Check if shard_tensor requires grad
    assert shard_tensor.requires_grad, "ShardTensor should require grad"
    assert shard_tensor.grad is not None

    assert torch.allclose(shard_tensor.grad.full_tensor(), full_local_tensor.grad)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
@pytest.mark.parametrize("uneven", [True, False])
def test_shard_tensor_input_gradient_local_loss(distributed_mesh, uneven):
    run_shard_tensor_input_gradient_local_loss(distributed_mesh, uneven, verbose=False)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
@pytest.mark.parametrize("uneven", [True, False])
def test_shard_tensor_input_gradient_local_loss_2d(distributed_mesh_2d, uneven):
    run_shard_tensor_input_gradient_local_loss(
        distributed_mesh_2d, uneven, verbose=False
    )
