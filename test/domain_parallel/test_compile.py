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

r"""Tests for ShardTensor integration with ``torch.compile`` / AOTAutograd.

The focus is on the runtime tangent-coercion hook
``ShardTensor.__coerce_same_metadata_as_tangent__``, which AOTAutograd
invokes during the compiled backward when the runtime tangent's spec
doesn't match the recorded one. The tests cover uneven sharding, which
DTensor does not have to handle and which earlier coerce implementations
silently dropped (defaulting back to even chunking).
"""

import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, scatter_tensor
from physicsnemo.domain_parallel._shard_tensor_spec import ShardTensorSpec
from test.domain_parallel.test_redistribute import shard_tensor_factory


def _replicate_placements(mesh):
    return [Replicate()] * mesh.ndim


def run_coerce_replicate_to_uneven_shard(mesh):
    # Round-trip: uneven Shard -> Replicate -> coerce back to recorded uneven Shard.
    st_uneven = shard_tensor_factory(mesh, uneven=True)
    recorded_spec = st_uneven._spec
    expected_local_shape = tuple(st_uneven._local_tensor.shape)
    expected_full = st_uneven.full_tensor().clone()

    st_replicated = st_uneven.redistribute(placements=_replicate_placements(mesh))

    coerced = st_replicated.__coerce_same_metadata_as_tangent__((recorded_spec, False))

    assert isinstance(coerced, ShardTensor)
    assert coerced._spec.placements == recorded_spec.placements
    assert tuple(coerced._local_tensor.shape) == expected_local_shape
    assert torch.allclose(coerced.full_tensor(), expected_full)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_replicate_to_uneven_shard_1d(distributed_mesh):
    run_coerce_replicate_to_uneven_shard(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_replicate_to_uneven_shard_2d(distributed_mesh_2d):
    run_coerce_replicate_to_uneven_shard(distributed_mesh_2d)


def run_coerce_same_placements_unknown_shapes(mesh):
    # Recorded spec carries the same placements but no _sharding_shapes; the
    # hook must accept it without erroring and preserve local data.
    st = shard_tensor_factory(mesh, uneven=True)
    expected_local_shape = tuple(st._local_tensor.shape)
    expected_full = st.full_tensor().clone()

    modified_spec = ShardTensorSpec(
        mesh=st._spec.mesh,
        placements=st._spec.placements,
        tensor_meta=st._spec.tensor_meta,
        _sharding_shapes=None,
    )

    coerced = st.__coerce_same_metadata_as_tangent__((modified_spec, False))

    assert isinstance(coerced, ShardTensor)
    assert coerced._spec.placements == st._spec.placements
    assert coerced._spec._sharding_shapes is None
    assert tuple(coerced._local_tensor.shape) == expected_local_shape
    assert torch.allclose(coerced.full_tensor(), expected_full)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_same_placements_unknown_shapes_1d(distributed_mesh):
    run_coerce_same_placements_unknown_shapes(distributed_mesh)


def run_coerce_expected_type_returns_none(mesh):
    # Mismatched expected_type must short-circuit to None (DTensor convention).
    st = shard_tensor_factory(mesh, uneven=True)
    out = st.__coerce_same_metadata_as_tangent__(
        (st._spec, False), expected_type=torch.Tensor
    )
    assert out is None


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_expected_type_returns_none_1d(distributed_mesh):
    run_coerce_expected_type_returns_none(distributed_mesh)


def run_stable_hash_distinguishes_spec_metadata(mesh):
    st = shard_tensor_factory(mesh, uneven=True)
    spec = st._spec

    identical_spec = ShardTensorSpec(
        mesh=spec.mesh,
        placements=spec.placements,
        tensor_meta=spec.tensor_meta,
        _local_shape=spec._local_shape,
        _sharding_shapes=dict(spec._sharding_shapes),
    )
    assert identical_spec._stable_hash() == spec._stable_hash()

    # Mapping insertion order is not part of the logical sharding layout.
    reordered_shapes = dict(reversed(tuple(spec._sharding_shapes.items())))
    reordered_spec = ShardTensorSpec(
        mesh=spec.mesh,
        placements=spec.placements,
        tensor_meta=spec.tensor_meta,
        _local_shape=spec._local_shape,
        _sharding_shapes=reordered_shapes,
    )
    assert reordered_spec._stable_hash() == spec._stable_hash()

    # Unknown layouts isolate placement metadata contributed by the parent hash.
    unknown_layout_spec = ShardTensorSpec(
        mesh=spec.mesh,
        placements=spec.placements,
        tensor_meta=spec.tensor_meta,
        _local_shape=spec._local_shape,
        _sharding_shapes=None,
    )
    different_placements_spec = ShardTensorSpec(
        mesh=spec.mesh,
        placements=tuple(Replicate() for _ in range(mesh.ndim)),
        tensor_meta=spec.tensor_meta,
        _local_shape=spec.tensor_meta.shape,
        _sharding_shapes=None,
    )
    assert (
        different_placements_spec._stable_hash() != unknown_layout_spec._stable_hash()
    )

    # Change an uneven layout while preserving its global extent.
    different_shapes = dict(spec._sharding_shapes)
    mesh_dim, shard_shapes = next(iter(different_shapes.items()))
    tensor_dim = spec.placements[mesh_dim].dim
    changed_shapes = [list(shape) for shape in shard_shapes]
    changed_shapes[0][tensor_dim] += 1
    changed_shapes[1][tensor_dim] -= 1
    different_shapes[mesh_dim] = tuple(tuple(shape) for shape in changed_shapes)
    different_shapes_spec = ShardTensorSpec(
        mesh=spec.mesh,
        placements=spec.placements,
        tensor_meta=spec.tensor_meta,
        _local_shape=spec._local_shape,
        _sharding_shapes=different_shapes,
    )
    assert different_shapes_spec._stable_hash() != spec._stable_hash()


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_stable_hash_distinguishes_spec_metadata_2d(distributed_mesh_2d):
    run_stable_hash_distinguishes_spec_metadata(distributed_mesh_2d)


def _sum_squares(x):
    return (x**2).sum()


def run_compile_backward_uneven_shard(mesh):
    # Smoke test: compile + backward over an uneven ShardTensor must not raise
    # AOTAutograd's "guessed metadata incorrectly" tangent error. Gradient values
    # are validated by the direct __coerce_same_metadata_as_tangent__ tests.
    x = shard_tensor_factory(mesh, uneven=True).detach().requires_grad_(True)

    torch._dynamo.reset()
    compiled = torch.compile(_sum_squares, fullgraph=True, backend="aot_eager")

    loss = compiled(x)
    loss.backward()


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_backward_uneven_shard_1d(distributed_mesh):
    run_compile_backward_uneven_shard(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_backward_uneven_shard_2d(distributed_mesh_2d):
    run_compile_backward_uneven_shard(distributed_mesh_2d)


# --- Regression: grads for ShardTensor *inputs* of a compiled region ---------
#
# AOTAutograd's joint trace computes grad_inputs by calling
# ``torch.autograd.grad`` on the wrapped subclass primals, with no
# DisableTorchFunctionSubclass guard. ShardTensor's ``__torch_function__``
# used to route that call through the DTensor fallback, which re-issued the
# graph query on freshly converted tensors (not in the graph); with
# ``allow_unused=True`` this silently produced all-None grads at trace time,
# so ``grad_input_metas`` was stamped plain and every compiled region
# returned plain-tensor gradients for its ShardTensor inputs -- crashing the
# first eager backward upstream that touched ``._local_tensor``.
# ``torch.autograd.grad`` is now in ``_autograd_passthrough_functions``.

_DIM = 64


def run_compiled_grad_input_stays_shard_tensor(mesh, partial_input):
    # Eager producer -> compiled consumer. The gradient the compiled region
    # returns for its ShardTensor input must arrive at the eager producer's
    # backward as a ShardTensor, with the same values as a fully-eager run.
    dm = DistributedManager()
    device = dm.device
    torch.manual_seed(7)

    x_full = torch.randn(1, 32, _DIM, device=device)
    x = scatter_tensor(x_full, 0, mesh, (Shard(1),))
    w0 = torch.randn(_DIM, _DIM, device=device, requires_grad=True)
    consumer = torch.nn.Linear(_DIM, _DIM).to(device)

    def run_once(consumer_fn):
        m = torch.nn.functional.linear(x, w0)
        if partial_input:
            # Mean over the sharded dim: Partial placement from the eager
            # custom reduction op -- the configuration that crashed first.
            m = m.mean(dim=(1,))
        grad_types = []
        m.register_hook(lambda g: grad_types.append(type(g)))
        consumer_fn(m).sum().backward()
        grad, w0.grad = w0.grad, None
        return grad, grad_types

    eager_grad, eager_types = run_once(consumer)
    assert issubclass(eager_types[0], ShardTensor)

    torch._dynamo.reset()
    compiled = torch.compile(
        consumer, fullgraph=True, backend="aot_eager", dynamic=False
    )
    # Second iteration exercises the cached compiled backward path.
    for _ in range(2):
        compiled_grad, compiled_types = run_once(compiled)
        assert compiled_types and issubclass(compiled_types[0], ShardTensor), (
            f"compiled region delivered grad of type {compiled_types} "
            "for its ShardTensor input"
        )
        torch.testing.assert_close(compiled_grad, eager_grad)


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
@pytest.mark.parametrize("partial_input", [False, True])
def test_compiled_grad_input_stays_shard_tensor_1d(distributed_mesh, partial_input):
    run_compiled_grad_input_stays_shard_tensor(distributed_mesh, partial_input)


def _partial_shard_tensor(mesh, local):
    # Partial local shape == global shape; build through DTensor.from_local
    # (accepts Partial directly) + communication-free spec inference.
    dt = DTensor.from_local(local, mesh, [Partial()] * mesh.ndim, run_check=False)
    return ShardTensor.from_dtensor(dt)


def run_coerce_partial_to_replicate_relabels(mesh):
    # Grad-direction convention: a Partial-labeled tangent is replicate-valued.
    # Coercing it to a recorded Replicate spec must RELABEL, not all-reduce —
    # a real reduction multiplies the tangent by the mesh size. Regression
    # test for the 4x compiled-grad bug with Partial inputs.
    dm = DistributedManager()
    local = torch.full((8, 4), 3.0, device=dm.device)
    st = _partial_shard_tensor(mesh, local)

    recorded = ShardTensorSpec(
        mesh=st._spec.mesh,
        placements=tuple(Replicate() for _ in range(mesh.ndim)),
        tensor_meta=st._spec.tensor_meta,
        _sharding_shapes=None,
    )
    coerced = st.__coerce_same_metadata_as_tangent__((recorded, False))

    assert isinstance(coerced, ShardTensor)
    assert coerced._spec.placements == recorded.placements
    assert torch.allclose(coerced._local_tensor, local), (
        "Partial->Replicate tangent coercion changed values (all-reduced "
        "an already-full tangent?)"
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_partial_to_replicate_relabels_1d(distributed_mesh):
    run_coerce_partial_to_replicate_relabels(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_partial_to_replicate_relabels_2d(distributed_mesh_2d):
    run_coerce_partial_to_replicate_relabels(distributed_mesh_2d)


def run_coerce_replicate_to_partial_relabels(mesh):
    # Reverse direction: recorded spec says Partial, runtime tangent arrived
    # Replicate. Must relabel (values unchanged), not partition (divide by
    # mesh size) — the latent inverse of the 4x bug.
    dm = DistributedManager()
    local = torch.full((8, 4), 3.0, device=dm.device)
    st = ShardTensor.from_local(
        local, mesh, tuple(Replicate() for _ in range(mesh.ndim))
    )

    recorded = ShardTensorSpec(
        mesh=st._spec.mesh,
        placements=tuple(Partial() for _ in range(mesh.ndim)),
        tensor_meta=st._spec.tensor_meta,
        _sharding_shapes=None,
    )
    coerced = st.__coerce_same_metadata_as_tangent__((recorded, False))

    assert isinstance(coerced, ShardTensor)
    assert coerced._spec.placements == recorded.placements
    assert torch.allclose(coerced._local_tensor, local), (
        "Replicate->Partial tangent coercion changed values (partitioned the tangent?)"
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_replicate_to_partial_relabels_1d(distributed_mesh):
    run_coerce_replicate_to_partial_relabels(distributed_mesh)
