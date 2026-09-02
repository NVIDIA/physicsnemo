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
- Partial gradient boundary: a replicated ``from_local`` weight mixed with a
  sharded activation produces a ``Partial`` gradient that must be resolved to
  ``Replicate`` (all-reduce) before crossing back to the plain leaf.
- Partial gradient identity: distributed leaf gradients retain their own
  pending-reduction placement instead of inheriting the primal placement.
- Explicit ``grad_placements``: requesting ``Partial`` declares rank-local
  gradient contributions that resolve to their summed global value.
- Autograd passthrough: ``register_hook`` / ``retain_grad`` bind to the real
  ShardTensor node and fire in backward (the mechanism FSDP2 depends on).

Both 1D and 2D device meshes are tested, with even and uneven sharding
where applicable. DTensor conversion tests use even sharding (DTensor
requirement).
"""

import pytest
import torch
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

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
    assert isinstance(dt.grad, (ShardTensor, DTensor))
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


def run_from_local_partial_grad_boundary(mesh):
    r"""Verify Partial->Replicate normalization at the from_local grad boundary.

    A plain weight is turned into a *replicated* ShardTensor via ``from_local``
    and multiplied against a sharded activation. In backward, the gradient of a
    replicated tensor used with sharded data is ``Partial`` (each rank holds an
    unreduced local contribution). ``_FromTorchTensor.backward`` must resolve
    that ``Partial`` to ``Replicate`` (an all-reduce) before handing the plain
    gradient back, so the leaf gradient equals the single-process reference.
    Without the reduction the gradient would only carry the local shard's
    contribution and disagree with the reference.
    """
    dm = DistributedManager()

    # Sharded activation (even sharding keeps reference alignment simple).
    shard_x = shard_tensor_factory(mesh, uneven=False)
    x_full = shard_x.full_tensor().detach()

    # Deterministic plain weight, identical on every rank, broadcastable over
    # the (unsharded) last dimension of the activation.
    torch.manual_seed(7)
    w_local = torch.randn(x_full.shape[-1], device=dm.device, requires_grad=True)

    st_w = ShardTensor.from_local(
        w_local, device_mesh=mesh, placements=[Replicate()] * mesh.ndim
    )

    out = shard_x * st_w
    loss = out.full_tensor().sum()
    loss.backward()

    # Single-process reference.
    w_ref = w_local.detach().clone().requires_grad_(True)
    (x_full * w_ref).sum().backward()

    assert w_local.grad is not None
    assert torch.allclose(w_local.grad, w_ref.grad, atol=1e-4)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_from_local_partial_grad_boundary(distributed_mesh):
    run_from_local_partial_grad_boundary(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_from_local_partial_grad_boundary_2d(distributed_mesh_2d):
    run_from_local_partial_grad_boundary(distributed_mesh_2d)


def run_partial_primal_gradients_emit_replicate(mesh):
    dm = DistributedManager()
    local = torch.randn(8, 4, device=dm.device)
    dtensor = DTensor.from_local(
        local,
        mesh,
        [Partial()] * mesh.ndim,
        run_check=False,
    )

    local_view_input = ShardTensor.from_dtensor(dtensor).detach().requires_grad_(True)
    local_view_input.to_local().sum().backward()
    assert local_view_input.grad is not None
    assert local_view_input.grad.placements == tuple(
        Replicate() for _ in range(mesh.ndim)
    )

    reduction_input = ShardTensor.from_dtensor(dtensor).detach().requires_grad_(True)
    reduction_input.sum().backward()
    assert reduction_input.grad is not None
    assert reduction_input.grad.placements == tuple(
        Replicate() for _ in range(mesh.ndim)
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_partial_primal_gradients_emit_replicate_1d(distributed_mesh):
    run_partial_primal_gradients_emit_replicate(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_partial_primal_gradients_emit_replicate_2d(distributed_mesh_2d):
    run_partial_primal_gradients_emit_replicate(distributed_mesh_2d)


def run_leaf_grad_preserves_genuine_partial_placement(mesh):
    dm = DistributedManager()
    shard_x = shard_tensor_factory(mesh, uneven=False)
    x_full = shard_x.full_tensor()
    weight = ShardTensor.from_local(
        torch.randn(x_full.shape[-1], device=dm.device),
        mesh,
        [Replicate()] * mesh.ndim,
    ).detach()
    weight.requires_grad_(True)

    (shard_x * weight).full_tensor().sum().backward()

    assert weight.grad is not None
    assert weight.grad.placements == tuple(Partial() for _ in range(mesh.ndim))
    expected = x_full.sum(dim=tuple(range(x_full.ndim - 1)))
    # The distributed and reference sums use different FP32 accumulation
    # orders across thousands of values. Bound the resulting roundoff while
    # still catching a missing or duplicated mesh reduction.
    torch.testing.assert_close(
        weight.grad.full_tensor(),
        expected,
        atol=1e-4,
        rtol=1e-5,
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_leaf_grad_preserves_genuine_partial_placement_1d(distributed_mesh):
    run_leaf_grad_preserves_genuine_partial_placement(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_leaf_grad_preserves_genuine_partial_placement_2d(distributed_mesh_2d):
    run_leaf_grad_preserves_genuine_partial_placement(distributed_mesh_2d)


def run_explicit_partial_grad_placements_are_pending_reduction(mesh):
    dm = DistributedManager()
    value = ShardTensor.from_local(
        torch.randn(8, 4, device=dm.device),
        mesh,
        [Replicate()] * mesh.ndim,
    ).detach()
    value.requires_grad_(True)

    coordinate = mesh.get_coordinate()
    linear_rank = 0
    mesh_size = 1
    for mesh_dim, mesh_rank in enumerate(coordinate):
        linear_rank = linear_rank * mesh.size(mesh_dim) + mesh_rank
        mesh_size *= mesh.size(mesh_dim)
    local_view = value.to_local(
        grad_placements=tuple(Partial() for _ in range(mesh.ndim))
    )
    (local_view * float(linear_rank + 1)).sum().backward()

    assert value.grad is not None
    assert value.grad.placements == tuple(Partial() for _ in range(mesh.ndim))
    torch.testing.assert_close(
        value.grad._local_tensor,
        torch.full_like(value.grad._local_tensor, float(linear_rank + 1)),
    )
    expected_value = float(mesh_size * (mesh_size + 1) // 2)
    torch.testing.assert_close(
        value.grad.full_tensor(),
        torch.full_like(value.grad._local_tensor, expected_value),
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_explicit_partial_grad_placements_are_pending_reduction_1d(distributed_mesh):
    run_explicit_partial_grad_placements_are_pending_reduction(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_explicit_partial_grad_placements_are_pending_reduction_2d(
    distributed_mesh_2d,
):
    run_explicit_partial_grad_placements_are_pending_reduction(distributed_mesh_2d)


def run_shard_tensor_register_hook_fires(mesh):
    r"""A backward hook registered on a ShardTensor must fire in backward.

    ``register_hook`` is routed through ``_autograd_passthrough_functions`` so it
    binds to the real ShardTensor autograd node rather than a throwaway DTensor.
    FSDP2 relies on exactly this behavior (it gates its reduce-scatter on a hook
    firing on the forward output), so the hook not firing would silently break
    FSDP2.
    """
    shard_x = shard_tensor_factory(mesh, uneven=False).detach().requires_grad_(True)

    # Non-leaf ShardTensor, mirroring an FSDP-wrapped module's forward output.
    y = shard_x * 2.0

    fired = []
    handle = y.register_hook(lambda grad: fired.append(grad))
    assert handle is not None

    loss = y.full_tensor().sum()
    loss.backward()

    assert len(fired) == 1


def run_shard_tensor_retain_grad(mesh):
    r"""``retain_grad`` on a non-leaf ShardTensor must populate ``.grad``."""
    shard_x = shard_tensor_factory(mesh, uneven=False).detach().requires_grad_(True)

    y = shard_x * 2.0
    y.retain_grad()

    loss = y.full_tensor().sum()
    loss.backward()

    assert y.grad is not None


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_shard_tensor_register_hook_fires(distributed_mesh):
    run_shard_tensor_register_hook_fires(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_shard_tensor_register_hook_fires_2d(distributed_mesh_2d):
    run_shard_tensor_register_hook_fires(distributed_mesh_2d)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_shard_tensor_retain_grad(distributed_mesh):
    run_shard_tensor_retain_grad(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_shard_tensor_retain_grad_2d(distributed_mesh_2d):
    run_shard_tensor_retain_grad(distributed_mesh_2d)


def run_eager_autograd_grad_returns_shard_tensor(mesh):
    r"""Eager ``torch.autograd.grad(loss, [st])`` must return ShardTensor grads.

    ``torch.autograd.grad`` is routed through ``_autograd_passthrough_functions``
    (it queries gradients for these exact tensor objects, so the DTensor
    fallback would ask about tensors that aren't in the graph). The passthrough
    runs it under ``DisableTorchFunctionSubclass``, which only suppresses
    ``__torch_function__`` re-entry: the backward graph was recorded on the
    subclass and executes via ``__torch_dispatch__``, so the returned grads
    must still be ShardTensor-typed — in eager just as inside AOTAutograd's
    joint trace.
    """
    shard_x = shard_tensor_factory(mesh, uneven=False).detach().requires_grad_(True)

    # Leaf grad, with a local reference for the values.
    full_ref = shard_x.full_tensor().detach().requires_grad_(True)
    (ref_grad,) = torch.autograd.grad((full_ref**2).sum(), [full_ref])

    y = shard_x**2
    loss = y.full_tensor().sum()
    (grad_leaf,) = torch.autograd.grad(loss, [shard_x], create_graph=False)

    assert isinstance(grad_leaf, ShardTensor), (
        f"eager autograd.grad returned {type(grad_leaf)} for a ShardTensor leaf"
    )
    assert torch.allclose(grad_leaf.full_tensor(), ref_grad)

    # Non-leaf grad must also stay ShardTensor-typed.
    y2 = shard_x * 3.0
    loss2 = y2.full_tensor().sum()
    (grad_nonleaf,) = torch.autograd.grad(loss2, [y2])
    assert isinstance(grad_nonleaf, ShardTensor), (
        f"eager autograd.grad returned {type(grad_nonleaf)} for a non-leaf ShardTensor"
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_eager_autograd_grad_returns_shard_tensor(distributed_mesh):
    run_eager_autograd_grad_returns_shard_tensor(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_eager_autograd_grad_returns_shard_tensor_2d(distributed_mesh_2d):
    run_eager_autograd_grad_returns_shard_tensor(distributed_mesh_2d)
