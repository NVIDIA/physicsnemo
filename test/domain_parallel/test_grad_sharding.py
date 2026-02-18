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

r"""Tests for ShardTensor gradient sharding.

This module tests the gradient computation capabilities of ``ShardTensor``.
The tests verify that calling ``backward()`` on a ShardTensor produces
gradients that agree with the equivalent local computations.

Test cases include:

- ``detach()``: Verify that detaching preserves tensor data and spec
- Full tensor loss: Gradients computed using ``full_tensor()`` in the loss
- Local tensor loss: Gradients computed using ``to_local()`` in the loss
- DTensor to ShardTensor (leaf): ``ShardTensor.from_dtensor`` on a leaf
  DTensor; backward through the ShardTensor and compare gradients to a
  local reference.
- DTensor to ShardTensor (non-leaf): ``ShardTensor.from_dtensor`` on a
  non-leaf DTensor (e.g. result of an op); backward and verify gradients
  flow to the original DTensor leaf.

Both 1D and 2D device meshes are tested, with even and uneven sharding
where applicable. DTensor conversion tests use even sharding (DTensor
requirement).
"""

import pytest
import torch
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor
from test.domain_parallel.test_redistribute import shard_tensor_factory


def _even_global_shape_and_placements(mesh):
    r"""Global shape and placements for even-sharded DTensor (compatible with DTensor).

    Returns
    -------
    tuple
        (global_shape, placements) for use with ``distribute_tensor``.
    """
    # Shape divisible by common mesh sizes so DTensor can shard evenly.
    global_shape = (10, 2 * 3 * 4 * 5 * 7, 2 * 3 * 4 * 5 * 7, 10)
    placements = [Shard(1)]
    if mesh.ndim > 1:
        placements.append(Shard(2))
    return global_shape, placements


def run_shard_tensor_detach(mesh, uneven, verbose):
    shard_tensor = shard_tensor_factory(mesh, uneven=uneven)
    shard_tensor_detached = shard_tensor.detach()

    # Detaching should not change the original data nor should it change the spec:
    assert shard_tensor._spec == shard_tensor_detached._spec

    assert torch.allclose(
        shard_tensor.full_tensor(), shard_tensor_detached.full_tensor()
    )

    assert shard_tensor_detached.is_leaf


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


def run_dtensor_to_shard_tensor_leaf_gradient(mesh):
    r"""Verify autograd through ShardTensor.from_dtensor when the DTensor is a leaf.

    Creates a leaf DTensor with ``requires_grad=True``, converts to ShardTensor
    via ``from_dtensor``, computes a loss on the ShardTensor, and runs backward.
    Compares the ShardTensor gradient to the gradient of the same computation
    on a local full tensor.
    """
    dm = DistributedManager()
    global_shape, placements = _even_global_shape_and_placements(mesh)
    raw_data = torch.randn(
        global_shape,
        device=torch.device(f"cuda:{dm.local_rank}"),
        requires_grad=False,
    )
    dt = distribute_tensor(raw_data, device_mesh=mesh, placements=placements)
    dt = dt.detach().requires_grad_(True)

    st = ShardTensor.from_dtensor(dt)
    assert st.requires_grad

    # Reference: same computation on full local tensor
    ref = dt.full_tensor().detach().requires_grad_(True)

    def loss_fn(x):
        return (x**2).sum()

    loss_st = loss_fn(st)
    loss_st.backward()

    loss_ref = loss_fn(ref)
    loss_ref.backward()

    assert st.grad is not None
    assert torch.allclose(st.grad.full_tensor(), ref.grad)


def run_dtensor_to_shard_tensor_non_leaf_gradient(mesh):
    r"""Verify autograd through ShardTensor.from_dtensor when the DTensor is non-leaf.

    Creates a leaf DTensor, applies an op to get a non-leaf DTensor, converts
    that result to ShardTensor via ``from_dtensor``, then backward. Verifies
    gradients flow correctly to the original DTensor leaf (compare to local
    reference).
    """
    dm = DistributedManager()
    global_shape, placements = _even_global_shape_and_placements(mesh)
    raw_data = torch.randn(
        global_shape,
        device=torch.device(f"cuda:{dm.local_rank}"),
        requires_grad=False,
    )
    dt = distribute_tensor(raw_data, device_mesh=mesh, placements=placements)
    dt = dt.detach().requires_grad_(True)

    # Non-leaf DTensor (op result)
    dt2 = dt * 2.0
    st = ShardTensor.from_dtensor(dt2)
    assert st.grad_fn is not None

    loss = st.full_tensor().sum()
    loss.backward()

    # Reference: local full tensor, same ops
    ref = dt.full_tensor().detach().requires_grad_(True)
    ref2 = ref * 2.0
    loss_ref = ref2.sum()
    loss_ref.backward()

    assert dt.grad is not None
    assert isinstance(dt.grad, DTensor)
    assert torch.allclose(dt.grad.full_tensor(), ref.grad)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_dtensor_to_shard_tensor_leaf_gradient(distributed_mesh):
    run_dtensor_to_shard_tensor_leaf_gradient(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_dtensor_to_shard_tensor_leaf_gradient_2d(distributed_mesh_2d):
    run_dtensor_to_shard_tensor_leaf_gradient(distributed_mesh_2d)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_dtensor_to_shard_tensor_non_leaf_gradient(distributed_mesh):
    run_dtensor_to_shard_tensor_non_leaf_gradient(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_dtensor_to_shard_tensor_non_leaf_gradient_2d(distributed_mesh_2d):
    run_dtensor_to_shard_tensor_non_leaf_gradient(distributed_mesh_2d)
