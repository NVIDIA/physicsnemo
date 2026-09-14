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

import dataclasses
import math

import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, scatter_tensor
from physicsnemo.domain_parallel._shard_tensor_spec import (
    ShardTensorSpec,
    compute_sharding_shapes_from_chunking_global_shape,
)
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


def run_coerce_empty_and_none_shapes_are_identity(mesh):
    # ``{}`` (fully-replicated / op-result rewrap) vs ``None`` (fresh spec)
    # both mean "no per-shard shape info": with matching placements the hook
    # must early-return self rather than redistribute, in both directions.
    st = shard_tensor_factory(mesh, uneven=False).redistribute(
        placements=_replicate_placements(mesh)
    )

    st_empty = ShardTensor.__new__(
        ShardTensor,
        local_tensor=st._local_tensor,
        spec=dataclasses.replace(st._spec, _sharding_shapes={}),
        requires_grad=False,
    )
    recorded_none = dataclasses.replace(st._spec, _sharding_shapes=None)
    assert st_empty.__coerce_same_metadata_as_tangent__((recorded_none, False)) is (
        st_empty
    )

    st_none = ShardTensor.__new__(
        ShardTensor,
        local_tensor=st._local_tensor,
        spec=dataclasses.replace(st._spec, _sharding_shapes=None),
        requires_grad=False,
    )
    recorded_empty = dataclasses.replace(st._spec, _sharding_shapes={})
    assert st_none.__coerce_same_metadata_as_tangent__((recorded_empty, False)) is (
        st_none
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_empty_and_none_shapes_are_identity_1d(distributed_mesh):
    run_coerce_empty_and_none_shapes_are_identity(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_empty_and_none_shapes_are_identity_2d(distributed_mesh_2d):
    run_coerce_empty_and_none_shapes_are_identity(distributed_mesh_2d)


def run_coerce_expected_type_returns_none(mesh):
    # A *foreign* expected_type (a plain tensor / DTensor, i.e. not a ShardTensor
    # subclass) must short-circuit to None -- the DTensor cross-type convention,
    # which the subclass-friendly path below deliberately preserves.
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
    spec = shard_tensor_factory(mesh, uneven=True)._spec
    identical_spec = dataclasses.replace(
        spec, _sharding_shapes=dict(spec._sharding_shapes)
    )
    assert identical_spec._stable_hash() == spec._stable_hash()

    replicated_spec = dataclasses.replace(
        spec,
        placements=tuple(Replicate() for _ in range(mesh.ndim)),
        shard_order=None,
        _sharding_shapes=None,
    )
    assert replicated_spec._stable_hash() != spec._stable_hash()

    different_shapes = dict(spec._sharding_shapes)
    mesh_dim, shard_shapes = next(
        (dim, shapes) for dim, shapes in different_shapes.items() if len(shapes) > 1
    )
    tensor_dim = spec.placements[mesh_dim].dim
    changed_shapes = [list(shape) for shape in shard_shapes]
    changed_shapes[0][tensor_dim] += 1
    changed_shapes[1][tensor_dim] -= 1
    different_shapes[mesh_dim] = tuple(tuple(shape) for shape in changed_shapes)
    different_layout_spec = dataclasses.replace(spec, _sharding_shapes=different_shapes)
    assert different_layout_spec._stable_hash() != spec._stable_hash()

    # With _sharding_shapes=None the spec carries nothing else that
    # distinguishes uneven layouts, so _local_shape must be folded in:
    # same global shape + placements but a different local slice must hash
    # differently (AOT-cache key discrimination on each rank).
    none_spec = dataclasses.replace(spec, _sharding_shapes=None)
    other_local = torch.Size(tuple(d + 1 for d in none_spec._local_shape))
    none_spec_other_local = dataclasses.replace(none_spec, _local_shape=other_local)
    assert none_spec._stable_hash() != none_spec_other_local._stable_hash()


class _MarkerShardTensor(ShardTensor):
    """A minimal ``ShardTensor`` subclass used to exercise the subclass-friendly
    cross-type tangent-coercion path. It adds no new inner tensors or metadata;
    its own ``__tensor_unflatten__`` just reclasses a base reconstruction so the
    result carries the subclass type (as a real subclass with per-instance
    routing metadata would)."""

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        st = ShardTensor.__tensor_unflatten__(
            inner_tensors, flatten_spec, outer_size, outer_stride
        )
        st.__class__ = _MarkerShardTensor
        return st


def run_coerce_accepts_nested_flatten_context(mesh):
    # A ShardTensor subclass emits a nested ``(base_ctx, subclass_ctx)`` flatten
    # context. The coerce hook must unwrap it to the base ``(spec, requires_grad)``
    # and behave exactly as it does for the base's flat context.
    st = shard_tensor_factory(mesh, uneven=True)
    recorded_spec = st._spec
    expected_local_shape = tuple(st._local_tensor.shape)
    expected_full = st.full_tensor().clone()

    nested_ctx = ((recorded_spec, False), ("subclass-metadata-placeholder",))
    coerced = st.__coerce_same_metadata_as_tangent__(nested_ctx)

    assert isinstance(coerced, ShardTensor)
    assert coerced._spec.placements == recorded_spec.placements
    assert tuple(coerced._local_tensor.shape) == expected_local_shape
    assert torch.allclose(coerced.full_tensor(), expected_full)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_accepts_nested_flatten_context_1d(distributed_mesh):
    run_coerce_accepts_nested_flatten_context(distributed_mesh)


def run_coerce_empty_and_none_sharding_shapes_equal(mesh):
    # ``{}`` (an op-result re-wrap) and ``None`` (a fresh spec) both mean "no
    # explicit per-rank sharding shapes". With identical placements the hook must
    # treat them as equal and short-circuit to ``self`` -- no redistribute.
    st = shard_tensor_factory(mesh, uneven=True)

    spec_empty = ShardTensorSpec(
        mesh=st._spec.mesh,
        placements=st._spec.placements,
        tensor_meta=st._spec.tensor_meta,
        _sharding_shapes={},
    )
    st_empty = ShardTensor(st._local_tensor, spec_empty, requires_grad=False)

    recorded_spec_none = ShardTensorSpec(
        mesh=st._spec.mesh,
        placements=st._spec.placements,
        tensor_meta=st._spec.tensor_meta,
        _sharding_shapes=None,
    )

    coerced = st_empty.__coerce_same_metadata_as_tangent__((recorded_spec_none, False))
    assert coerced is st_empty


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_empty_and_none_sharding_shapes_equal_1d(distributed_mesh):
    run_coerce_empty_and_none_sharding_shapes_equal(distributed_mesh)


def run_coerce_expected_subclass_rebuilds(mesh):
    # A differing expected_type that *is* a ShardTensor subclass must be rebuilt
    # via that subclass's own ``__tensor_unflatten__`` (reclass + reattach), not
    # dropped to None -- this is what lets a subclass accept a base-typed tangent.
    st = shard_tensor_factory(mesh, uneven=True)
    expected_full = st.full_tensor().clone()

    out = st.__coerce_same_metadata_as_tangent__(
        (st._spec, False), expected_type=_MarkerShardTensor
    )

    assert isinstance(out, _MarkerShardTensor)
    assert out._spec.placements == st._spec.placements
    assert torch.allclose(out.full_tensor(), expected_full)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_expected_subclass_rebuilds_1d(distributed_mesh):
    run_coerce_expected_subclass_rebuilds(distributed_mesh)


def run_unflatten_does_not_force_inner_requires_grad(mesh):
    # ``__tensor_unflatten__`` must not force ``requires_grad`` on the inner: the wrapper
    # carries it (via grad_fn) while a detached op-result local does not, and forcing them to
    # agree trips ``assert_metadata_eq`` on a graph-break re-fake. Matches DTensor.
    st = shard_tensor_factory(mesh, uneven=False)
    local = st._local_tensor.detach()
    assert local.requires_grad is False

    _inner_names, ctx = st.__tensor_flatten__()
    spec, _ = ctx

    rebuilt = ShardTensor.__tensor_unflatten__(
        {"_local_tensor": local}, (spec, True), st.shape, st.stride()
    )

    # Wrapper honors the requested requires_grad; the inner is left untouched
    # (still False) and the passed-in local is not mutated in place.
    assert rebuilt.requires_grad is True
    assert rebuilt._local_tensor.requires_grad is False
    assert local.requires_grad is False


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_unflatten_does_not_force_inner_requires_grad_1d(distributed_mesh):
    run_unflatten_does_not_force_inner_requires_grad(distributed_mesh)


def _sum_squares(x):
    return (x**2).sum()


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_stable_hash_distinguishes_spec_metadata_2d(distributed_mesh_2d):
    run_stable_hash_distinguishes_spec_metadata(distributed_mesh_2d)


def run_unflatten_checks_even_chunk_assumption(mesh):
    r"""``__tensor_unflatten__`` with a shapeless spec must not silently
    assume even chunking for an uneven shard.

    When ``_sharding_shapes`` is ``None``, unflatten derives shapes by
    chunking the global shape. For an unevenly sharded tensor that
    derivation is wrong, bakes wrong collective sizes into compiled graphs,
    and Dynamo guards can't discriminate (None == None): ranks whose local
    shard contradicts the chunk assumption must raise instead.
    """
    st = shard_tensor_factory(mesh, uneven=True)
    stripped = dataclasses.replace(st._spec, _sharding_shapes=None)
    global_shape = tuple(st._spec.tensor_meta.shape)
    stride = st._spec.tensor_meta.stride

    chunk = compute_sharding_shapes_from_chunking_global_shape(
        mesh, st._spec.placements, global_shape
    )
    coords = mesh.get_coordinate()
    local_shape = tuple(st._local_tensor.shape)
    mismatch = any(tuple(chunk[m][coords[m]]) != local_shape for m in chunk)

    if mismatch:
        with pytest.raises(RuntimeError, match="unevenly sharded"):
            ShardTensor.__tensor_unflatten__(
                {"_local_tensor": st._local_tensor},
                (stripped, False),
                global_shape,
                stride,
            )
    else:
        out = ShardTensor.__tensor_unflatten__(
            {"_local_tensor": st._local_tensor},
            (stripped, False),
            global_shape,
            stride,
        )
        assert isinstance(out, ShardTensor)

    # An evenly sharded tensor must still unflatten fine from a shapeless
    # spec (the chunk assumption holds).
    st_even = shard_tensor_factory(mesh, uneven=False)
    stripped_even = dataclasses.replace(st_even._spec, _sharding_shapes=None)
    out_even = ShardTensor.__tensor_unflatten__(
        {"_local_tensor": st_even._local_tensor},
        (stripped_even, False),
        tuple(st_even._spec.tensor_meta.shape),
        st_even._spec.tensor_meta.stride,
    )
    assert isinstance(out_even, ShardTensor)
    assert tuple(out_even._local_tensor.shape) == tuple(st_even._local_tensor.shape)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_unflatten_checks_even_chunk_assumption_1d(distributed_mesh):
    run_unflatten_checks_even_chunk_assumption(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_unflatten_checks_even_chunk_assumption_2d(distributed_mesh_2d):
    run_unflatten_checks_even_chunk_assumption(distributed_mesh_2d)


def run_compile_backward_uneven_shard(mesh, op):
    r"""Numerical grad-equivalence for reductions under compile, uneven shards.

    The gradient of ``(x**2).sum()`` is ``2*x`` and of ``(x**2).mean()`` is
    ``2*x / N`` (``N`` = global element count) -- closed forms that make the
    compiled backward's values checkable per-rank without a second autograd
    graph.
    """
    x = shard_tensor_factory(mesh, uneven=True).detach().requires_grad_(True)

    def f(t):
        y = t**2
        return y.sum() if op == "sum" else y.mean()

    torch._dynamo.reset()
    compiled = torch.compile(f, fullgraph=True, backend="aot_eager")

    loss = compiled(x)
    loss.backward()

    grad = x.grad
    assert isinstance(grad, ShardTensor)
    assert grad._spec.placements == x._spec.placements
    expected = 2.0 * x._local_tensor.detach()
    if op == "mean":
        expected = expected / math.prod(x._spec.tensor_meta.shape)
    torch.testing.assert_close(grad.to_local(), expected)


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
@pytest.mark.parametrize("op", ["sum", "mean"])
def test_compile_backward_uneven_shard_1d(distributed_mesh, op):
    run_compile_backward_uneven_shard(distributed_mesh, op)


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
@pytest.mark.parametrize("op", ["sum", "mean"])
def test_compile_backward_uneven_shard_2d(distributed_mesh_2d, op):
    run_compile_backward_uneven_shard(distributed_mesh_2d, op)


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
    # Partial local shape equals global shape. DTensor.from_local accepts the
    # pending-reduction placement directly and avoids an eager collective.
    dt = DTensor.from_local(local, mesh, [Partial()] * mesh.ndim, run_check=False)
    return ShardTensor.from_dtensor(dt)


def _rank_distinct_partial_shard_tensor(mesh):
    coordinate = mesh.get_coordinate()
    linear_rank = 0
    mesh_size = 1
    for mesh_dim, mesh_rank in enumerate(coordinate):
        linear_rank = linear_rank * mesh.size(mesh_dim) + mesh_rank
        mesh_size *= mesh.size(mesh_dim)

    dm = DistributedManager()
    local = torch.full((8, 4), float(linear_rank + 1), device=dm.device)
    st = _partial_shard_tensor(mesh, local)
    expected = torch.full(
        local.shape,
        float(mesh_size * (mesh_size + 1) // 2),
        device=dm.device,
    )
    return st, expected


def run_trace_tangent_coercion_reduces_partial(mesh):
    st, expected = _rank_distinct_partial_shard_tensor(mesh)

    coerced = st.__coerce_tangent_metadata__()

    assert isinstance(coerced, ShardTensor)
    assert coerced.placements == tuple(Replicate() for _ in range(mesh.ndim))
    torch.testing.assert_close(coerced._local_tensor, expected)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_trace_tangent_coercion_reduces_partial_1d(distributed_mesh):
    run_trace_tangent_coercion_reduces_partial(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_trace_tangent_coercion_reduces_partial_2d(distributed_mesh_2d):
    run_trace_tangent_coercion_reduces_partial(distributed_mesh_2d)


def run_runtime_tangent_coercion_reduces_partial(mesh):
    st, expected = _rank_distinct_partial_shard_tensor(mesh)

    recorded = ShardTensorSpec(
        mesh=st._spec.mesh,
        placements=tuple(Replicate() for _ in range(mesh.ndim)),
        tensor_meta=st._spec.tensor_meta,
        _sharding_shapes=None,
    )
    coerced = st.__coerce_same_metadata_as_tangent__((recorded, False))

    assert isinstance(coerced, ShardTensor)
    assert coerced._spec.placements == recorded.placements
    torch.testing.assert_close(coerced._local_tensor, expected)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_runtime_tangent_coercion_reduces_partial_1d(distributed_mesh):
    run_runtime_tangent_coercion_reduces_partial(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_runtime_tangent_coercion_reduces_partial_2d(distributed_mesh_2d):
    run_runtime_tangent_coercion_reduces_partial(distributed_mesh_2d)


def run_runtime_tangent_coercion_rejects_partial_target(mesh):
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
    with pytest.raises(
        RuntimeError, match="recorded a Partial tangent target for ShardTensor"
    ):
        st.__coerce_same_metadata_as_tangent__((recorded, False))

    partial = _partial_shard_tensor(mesh, local)
    with pytest.raises(
        RuntimeError, match="recorded a Partial tangent target for ShardTensor"
    ):
        partial.__coerce_same_metadata_as_tangent__((partial._spec, False))


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_runtime_tangent_coercion_rejects_partial_target_1d(distributed_mesh):
    run_runtime_tangent_coercion_rejects_partial_target(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_runtime_tangent_coercion_rejects_partial_target_2d(distributed_mesh_2d):
    run_runtime_tangent_coercion_rejects_partial_target(distributed_mesh_2d)


def run_genuine_partial_cotangent_enters_compiled_backward(mesh, op):
    dm = DistributedManager()
    device = dm.device
    torch.manual_seed(23)

    feature_size = 16
    batch_shape = (8,) * mesh.ndim
    full_activation = torch.arange(
        torch.tensor(batch_shape).prod().item() * feature_size,
        dtype=torch.float32,
        device=device,
    ).reshape(*batch_shape, feature_size)
    full_activation = full_activation / full_activation.numel()
    # Only the first mesh dimension partitions the batch. Replicating any
    # remaining mesh dimensions keeps this a supported scatter layout while
    # still producing a genuine Partial parameter cotangent.
    activation_placements = (Shard(0),) + tuple(
        Replicate() for _ in range(mesh.ndim - 1)
    )
    expected_tangent_placements = (Partial(),) + tuple(
        Replicate() for _ in range(mesh.ndim - 1)
    )
    activation = scatter_tensor(
        full_activation,
        0,
        mesh,
        activation_placements,
        global_shape=full_activation.shape,
        dtype=full_activation.dtype,
        requires_grad=False,
    )
    replicate_placements = tuple(Replicate() for _ in range(mesh.ndim))

    if op == "pointwise":
        parameter_data = torch.linspace(-0.7, 0.9, feature_size, device=device)

        def producer(value):
            return torch.sin(value) * 2.0

        def consume(distributed_activation, produced):
            return distributed_activation * produced

    elif op == "linear":
        parameter_data = torch.linspace(
            -0.5,
            0.8,
            4 * feature_size,
            device=device,
        ).reshape(4, feature_size)

        def producer(value):
            return torch.tanh(value) * 1.5

        def consume(distributed_activation, produced):
            return torch.nn.functional.linear(distributed_activation, produced)

    else:
        raise ValueError(f"Unsupported op: {op}")

    def run_once(producer_fn):
        parameter_local = parameter_data.detach().clone().requires_grad_(True)
        parameter = ShardTensor.from_local(
            parameter_local,
            mesh,
            replicate_placements,
        )
        produced = producer_fn(parameter)
        tangent_placements = []
        produced.register_hook(lambda grad: tangent_placements.append(grad.placements))
        consume(activation, produced).full_tensor().sum().backward()
        assert parameter_local.grad is not None
        return parameter_local.grad.detach().clone(), tangent_placements

    reference_parameter = parameter_data.detach().clone().requires_grad_(True)
    consume(full_activation, producer(reference_parameter)).sum().backward()
    reference_grad = reference_parameter.grad
    assert reference_grad is not None

    eager_grad, eager_tangents = run_once(producer)
    assert eager_tangents
    assert eager_tangents[0] == expected_tangent_placements
    torch.testing.assert_close(eager_grad, reference_grad)

    torch._dynamo.reset()
    compiled = torch.compile(
        producer,
        fullgraph=True,
        backend="aot_eager",
        dynamic=False,
    )
    for _ in range(2):
        compiled_grad, compiled_tangents = run_once(compiled)
        assert compiled_tangents
        assert compiled_tangents[0] == expected_tangent_placements
        torch.testing.assert_close(compiled_grad, reference_grad)


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
@pytest.mark.parametrize("op", ["pointwise", "linear"])
def test_genuine_partial_cotangent_enters_compiled_backward_1d(distributed_mesh, op):
    run_genuine_partial_cotangent_enters_compiled_backward(distributed_mesh, op)


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
@pytest.mark.parametrize("op", ["pointwise", "linear"])
def test_genuine_partial_cotangent_enters_compiled_backward_2d(distributed_mesh_2d, op):
    run_genuine_partial_cotangent_enters_compiled_backward(distributed_mesh_2d, op)


def run_unbind_compiles_fullgraph(mesh):
    r"""``torch.unbind`` on ShardTensor must survive compile(fullgraph=True).

    The function-level unbind handler routes through ``to_local`` /
    ``from_local`` (``_FromTorchTensor.apply``). This locks in that the
    bridge stays dynamo-traceable end to end — forward values, backward
    values, and ShardTensor-typed grads, compiled vs. a local reference.
    """
    if mesh.ndim != 1:
        pytest.skip("unbind probe is written for 1d meshes")
    dm = DistributedManager()
    world = mesh.size(0)
    rank = mesh.get_local_rank(0)

    torch.manual_seed(1234)
    full = torch.randn(3, 4 * world, 4, device=dm.device)
    local = full[:, rank * 4 : (rank + 1) * 4].clone()
    st = ShardTensor.from_local(local, mesh, (Shard(1),)).detach().requires_grad_(True)

    def f(x):
        a, b, c = torch.unbind(x, 0)
        return (a * 2 + b * b + c).sum()

    ref = full.clone().requires_grad_(True)
    ref_loss = f(ref)
    (ref_grad,) = torch.autograd.grad(ref_loss, [ref])

    torch._dynamo.reset()
    compiled = torch.compile(f, fullgraph=True, backend="aot_eager", dynamic=False)
    loss = compiled(st)
    loss_plain = loss.full_tensor() if isinstance(loss, ShardTensor) else loss
    (grad,) = torch.autograd.grad(loss_plain, [st])

    torch.testing.assert_close(loss_plain, ref_loss)
    assert isinstance(grad, ShardTensor)
    torch.testing.assert_close(grad.full_tensor(), ref_grad)


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_unbind_compiles_fullgraph_1d(distributed_mesh):
    run_unbind_compiles_fullgraph(distributed_mesh)


def run_unbind_dispatch_function_consistency(mesh):
    r"""The two unbind handlers must agree on values and metadata.

    ``torch.unbind`` routes through the ``__torch_function__``-level handler
    (``unbind_wrapper``, to_local/from_local bridge); the same call under
    ``DisableTorchFunctionSubclass`` reaches the dispatcher and lands in the
    ``aten.unbind.int`` handler (``_unbind_dispatch``, direct construction).
    Values, placements, and sharding shapes must match between the two.
    """
    if mesh.ndim != 1:
        pytest.skip("unbind consistency check is written for 1d meshes")
    dm = DistributedManager()
    world = mesh.size(0)
    rank = mesh.get_local_rank(0)

    torch.manual_seed(7)
    full = torch.randn(3, 4 * world, 4, device=dm.device)
    local = full[:, rank * 4 : (rank + 1) * 4].clone()
    st = ShardTensor.from_local(local, mesh, (Shard(1),))

    outs_function = torch.unbind(st, 0)
    with torch._C.DisableTorchFunctionSubclass():
        outs_dispatch = torch.unbind(st, 0)

    assert len(outs_function) == len(outs_dispatch) == 3

    def _norm_shapes(spec):
        return {
            k: tuple(tuple(s) for s in v) for k, v in spec.sharding_shapes().items()
        }

    for out_fn, out_dp in zip(outs_function, outs_dispatch):
        assert isinstance(out_fn, ShardTensor)
        assert isinstance(out_dp, ShardTensor)
        assert out_fn._spec.placements == out_dp._spec.placements
        assert _norm_shapes(out_fn._spec) == _norm_shapes(out_dp._spec)
        torch.testing.assert_close(out_fn.to_local(), out_dp.to_local())


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_unbind_dispatch_function_consistency_1d(distributed_mesh):
    run_unbind_dispatch_function_consistency(distributed_mesh)


def run_compile_backward_redistribute_uneven(mesh):
    r"""A compiled backward must route an uneven grad through a shape-reading op.

    Forward redistributes uneven Shard -> Replicate, so the compiled backward
    redistributes the grad Replicate -> uneven Shard. That adjoint reads the
    recorded per-rank shard shapes (the funcol gather-v/scatter path in
    ``_shard_redistribute``) inside the compiled backward, with a
    closed-form value check (grad of ``(r**2).sum()`` is ``2*x`` locally).
    """
    x = shard_tensor_factory(mesh, uneven=True).detach().requires_grad_(True)
    replicate = [Replicate()] * mesh.ndim

    def f(t):
        r = t.redistribute(placements=replicate)
        return (r**2).sum()

    torch._dynamo.reset()
    compiled = torch.compile(f, fullgraph=True, backend="aot_eager")
    compiled(x).backward()

    grad = x.grad
    assert isinstance(grad, ShardTensor)
    assert grad._spec.placements == x._spec.placements
    assert tuple(grad.to_local().shape) == tuple(x._local_tensor.shape)
    torch.testing.assert_close(grad.to_local(), 2.0 * x._local_tensor.detach())


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_backward_redistribute_uneven_1d(distributed_mesh):
    run_compile_backward_redistribute_uneven(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_backward_redistribute_uneven_2d(distributed_mesh_2d):
    run_compile_backward_redistribute_uneven(distributed_mesh_2d)


# to_local() differentiability under torch.compile: ShardTensor's eager fallback converts to
# DTensor via autograd.Functions that AOT traces through, dropping the compiled backward; the
# __torch_function__ compile-passthrough restores it. These tests catch a zero/missing gradient.
def _to_local_sq_sum(x):
    return (x.to_local() ** 2).sum()


def _to_local_plain_math(x):
    return (x.to_local() * x.to_local()).sum()


def _op_result_to_local(x):
    # Forces an intermediate op-result ShardTensor before to_local().
    return ((x * 2.0 + x).to_local() ** 2).sum()


# Plain-torch references on the local tensor: an independent gradient oracle.
def _ref_sq_sum(local):
    return (local**2).sum()


def _ref_plain_math(local):
    return (local * local).sum()


def _ref_op_result(local):
    return ((local * 2.0 + local) ** 2).sum()


def run_to_local_grad_under_compile(mesh, fn, ref_fn, backend):
    placements = [Shard(0)] + [Replicate()] * (mesh.ndim - 1)

    base = shard_tensor_factory(mesh, uneven=True).full_tensor().detach()

    # Independent ground truth: plain-torch autograd on the local tensor.
    lr = base.clone().requires_grad_(True)
    grad_ref = torch.autograd.grad(ref_fn(lr), lr)[0]

    # ShardTensor eager path.
    le = base.clone().requires_grad_(True)
    ste = ShardTensor.from_local(le, mesh, placements)
    grad_eager = torch.autograd.grad(fn(ste), le)[0]

    # ShardTensor under torch.compile.
    torch._dynamo.reset()
    lc = base.clone().requires_grad_(True)
    stc = ShardTensor.from_local(lc, mesh, placements)
    compiled = torch.compile(fn, fullgraph=True, backend=backend)
    grad_compiled = torch.autograd.grad(compiled(stc), lc)[0]

    assert grad_compiled is not None, "compiled backward dropped the gradient"
    assert grad_compiled.abs().sum() > 0, "compiled gradient is zero"

    # Eager must match the independent reference (guards the reference itself),
    # and the compiled gradient must match both -- i.e. be numerically correct,
    # not merely nonzero or merely equal to a possibly-wrong eager value.
    torch.testing.assert_close(grad_eager, grad_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_compiled, grad_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize(
    "fn,ref_fn",
    [
        (_to_local_sq_sum, _ref_sq_sum),
        (_to_local_plain_math, _ref_plain_math),
        (_op_result_to_local, _ref_op_result),
    ],
    ids=["leaf", "plain_math", "op_result"],
)
def test_to_local_grad_under_compile_1d(distributed_mesh, fn, ref_fn, backend):
    run_to_local_grad_under_compile(distributed_mesh, fn, ref_fn, backend)


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "fn,ref_fn",
    [(_to_local_sq_sum, _ref_sq_sum), (_op_result_to_local, _ref_op_result)],
    ids=["leaf", "op_result"],
)
def test_to_local_grad_under_compile_2d(distributed_mesh_2d, fn, ref_fn):
    run_to_local_grad_under_compile(distributed_mesh_2d, fn, ref_fn, "aot_eager")
