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

r"""Tests for :func:`physicsnemo.distributed.fused_all_reduce`.

The file is collected by both test jobs (cf. ``test/distributed/test_autograd.py``):

- **Serial / no-op path** (unmarked): runs in the normal CPU suite. Distributed
  is not initialized, so ``fused_all_reduce`` returns detached clones. These
  tests pin the structure-preserving contract (dict / sequence / TensorDict
  round-trips, shape/dtype/device passthrough, detached & independent outputs)
  and the integer/float dtype guard.
- **Collective path** (``@pytest.mark.multigpu_static``): runs only under
  ``torchrun --nproc-per-node N -m pytest --multigpu-static`` (the conftest
  initializes ``DistributedManager`` and these are *all-collective* tests). They
  pin the actual reduction math: SUM across ranks, heterogeneous shapes/dtypes,
  key-order determinism, the uneven-shard weighted-mean idiom, nested
  TensorDicts, and MAX/MIN.
"""

import pytest
import torch
import torch.distributed as dist
from tensordict import TensorDict, TensorDictBase

from physicsnemo.distributed import DistributedManager, fused_all_reduce

# -----------------------------------------------------------------------------
# Serial / no-op path (unmarked -> CPU suite)
# -----------------------------------------------------------------------------


def test_mapping_structure_preserved():
    """A dict round-trips to a dict with the same keys (original order)."""
    inputs = {"b": torch.tensor([1.0, 2.0]), "a": torch.tensor(3.0)}
    reduced = fused_all_reduce(inputs)

    assert type(reduced) is dict
    assert list(reduced) == ["b", "a"]  # input order, not sorted
    assert torch.equal(reduced["b"], inputs["b"])
    assert torch.equal(reduced["a"], inputs["a"])


def test_sequence_structure_preserved():
    """A sequence round-trips to a list in the same order."""
    inputs = (torch.tensor(1.0), torch.tensor([2.0, 3.0]))
    reduced = fused_all_reduce(inputs)

    assert type(reduced) is list
    assert len(reduced) == 2
    assert torch.equal(reduced[0], inputs[0])
    assert torch.equal(reduced[1], inputs[1])


def test_bare_tensor_returns_tensor():
    """A bare tensor reduces to a bare tensor (no container), value preserved on
    the no-op path, detached and independent of the input."""
    grad_tensor = torch.tensor(2.0, requires_grad=True)
    reduced = fused_all_reduce(grad_tensor)

    assert isinstance(reduced, torch.Tensor)
    assert not isinstance(reduced, TensorDictBase)
    assert reduced.item() == 2.0
    assert not reduced.requires_grad
    assert reduced.data_ptr() != grad_tensor.data_ptr()
    # Mutating the output must not touch the input.
    reduced.add_(100.0)
    assert grad_tensor.item() == 2.0


def test_bare_tensor_integer_is_exact():
    """A homogeneous integer tensor reduces in its own dtype - no float cast."""
    big = 2**40  # well beyond float32's 24-bit mantissa
    reduced = fused_all_reduce(torch.tensor(big, dtype=torch.int64))

    assert reduced.dtype == torch.int64
    assert reduced.item() == big


def test_bare_tensor_async_noop_returns_work_and_value():
    """``async_op=True`` on a bare tensor returns ``(Tensor, work)``; the no-op
    handle starts completed and the local value holds."""
    result, work = fused_all_reduce(torch.tensor(5.0), async_op=True)

    assert isinstance(result, torch.Tensor)
    assert work.is_completed()  # no collective => already done
    assert work.wait() is True
    assert result.item() == 5.0


def test_outputs_are_detached_independent_clones():
    """No-op outputs are detached and do not alias the inputs."""
    grad_tensor = torch.tensor(2.0, requires_grad=True)
    inputs = {"x": grad_tensor}
    reduced = fused_all_reduce(inputs)

    assert not reduced["x"].requires_grad
    assert reduced["x"].data_ptr() != grad_tensor.data_ptr()
    # Mutating the output must not touch the input.
    reduced["x"].add_(100.0)
    assert grad_tensor.item() == 2.0


def test_empty_inputs():
    """Empty containers reduce to empty containers of the same type."""
    assert fused_all_reduce({}) == {}
    assert fused_all_reduce([]) == []
    empty_td = fused_all_reduce(TensorDict({}, batch_size=[]))
    assert isinstance(empty_td, TensorDictBase)
    assert len(list(empty_td.keys(include_nested=True, leaves_only=True))) == 0


def test_shape_dtype_device_passthrough(device):
    """Heterogeneous shapes/dtypes/devices round-trip unchanged (no-op path)."""
    inputs = {
        "scalar": torch.tensor(1.0, device=device),
        "vec": torch.arange(3, dtype=torch.float64, device=device),
        "mat": torch.ones(2, 2, dtype=torch.float16, device=device),
    }
    reduced = fused_all_reduce(inputs)

    for key, original in inputs.items():
        assert reduced[key].shape == original.shape
        assert reduced[key].dtype == original.dtype
        assert reduced[key].device == original.device
        assert torch.equal(reduced[key], original)


def test_flat_tensordict_returns_tensordict():
    """A flat TensorDict returns a TensorDict, never a degraded plain dict."""
    flat = TensorDict({"x": torch.tensor(1.0), "y": torch.tensor(2.0)}, batch_size=[])
    reduced = fused_all_reduce(flat)

    assert isinstance(reduced, TensorDictBase)
    assert set(reduced.keys()) == {"x", "y"}
    assert reduced["x"].item() == 1.0


def test_nested_tensordict_structure_preserved():
    """A nested TensorDict round-trips with its (nested) structure intact."""
    nested = TensorDict(
        {
            "loss": torch.tensor(1.0),
            "sub": TensorDict(
                {"x": torch.tensor(2.0), "y": torch.tensor(3.0)}, batch_size=[]
            ),
        },
        batch_size=[],
    )
    reduced = fused_all_reduce(nested)

    assert isinstance(reduced, TensorDictBase)
    leaf_keys = set(reduced.keys(include_nested=True, leaves_only=True))
    assert leaf_keys == {"loss", ("sub", "x"), ("sub", "y")}
    assert reduced["loss"].item() == 1.0
    assert reduced["sub"]["x"].item() == 2.0
    assert reduced["sub"]["y"].item() == 3.0


@pytest.mark.parametrize("bad", [42, 3.14, None, object()])
def test_invalid_type_raises(bad):
    """A non-tensor, non-container input is a TypeError (a bare Tensor is now
    supported; str/bytes are covered by test_string_like_raises_typeerror)."""
    with pytest.raises(TypeError):
        fused_all_reduce(bad)


@pytest.mark.parametrize("bad", ["loss", b"loss"])
def test_string_like_raises_typeerror(bad):
    """str/bytes satisfy Sequence but are not tensor containers -> TypeError
    (rather than a confusing AttributeError from the per-leaf ``.detach()``)."""
    with pytest.raises(TypeError):
        fused_all_reduce(bad)


def test_homogeneous_integer_bundle_is_exact():
    """Integer leaves reduce in their own dtype - no lossy float round-trip."""
    big = 2**40  # well beyond float32's 24-bit mantissa
    inputs = {"count": torch.tensor([5, 3]), "big": torch.tensor(big)}
    reduced = fused_all_reduce(inputs)

    assert reduced["count"].dtype == torch.int64
    assert reduced["count"].tolist() == [5, 3]
    assert reduced["big"].item() == big


def test_mixed_int_float_without_buffer_dtype_raises():
    """Implicitly casting integer leaves into a float buffer is refused."""
    with pytest.raises(ValueError, match="integer"):
        fused_all_reduce({"idx": torch.tensor(2**40), "loss": torch.tensor(1.0)})


def test_mixed_int_float_allowed_with_explicit_buffer_dtype():
    """An explicit ``buffer_dtype`` is the caller opting in to the cast."""
    reduced = fused_all_reduce(
        {"idx": torch.tensor(5), "loss": torch.tensor(1.5)},
        buffer_dtype=torch.float64,
    )
    assert reduced["idx"].item() == 5
    assert reduced["loss"].item() == pytest.approx(1.5)


def test_async_op_noop_returns_work_and_values():
    """``async_op=True`` returns ``(result, work)``; the no-op handle starts
    completed, ``wait()`` is ``True`` and idempotent, and the local values hold."""
    inputs = {"a": torch.tensor(3.0), "b": torch.tensor([1.0, 2.0])}
    result, work = fused_all_reduce(inputs, async_op=True)

    assert type(result) is dict
    assert list(result) == ["a", "b"]
    assert work.is_completed()  # no collective => already done
    assert work.wait() is True
    assert work.wait() is True  # idempotent
    assert result["a"].item() == 3.0
    assert torch.equal(result["b"], inputs["b"])


def test_async_op_preserves_container_types():
    """The ``(result, work)`` tuple round-trips dict / list / TensorDict."""
    res_dict, work_dict = fused_all_reduce({"x": torch.tensor(1.0)}, async_op=True)
    res_list, work_list = fused_all_reduce([torch.tensor(1.0)], async_op=True)
    res_td, work_td = fused_all_reduce(
        TensorDict({"x": torch.tensor(1.0)}), async_op=True
    )

    assert type(res_dict) is dict
    assert type(res_list) is list
    assert isinstance(res_td, TensorDictBase)
    for work in (work_dict, work_list, work_td):
        assert work.wait() is True
    assert res_dict["x"].item() == 1.0
    assert res_list[0].item() == 1.0
    assert res_td["x"].item() == 1.0


def test_compiles_fullgraph_without_graph_break():
    """``fused_all_reduce`` traces under ``torch.compile(fullgraph=True)`` with no
    graph breaks: Dynamo rewrites the in-place ``dist.all_reduce`` to a functional
    collective, so no functional-collective migration is needed. A fake process
    group makes the real collective path (cat + all_reduce + unpack) traceable
    without launching multiple processes.
    """
    import torch._dynamo as dynamo
    from torch.testing._internal.distributed.fake_pg import FakeStore

    assert not dist.is_initialized()
    dist.init_process_group(backend="fake", store=FakeStore(), rank=0, world_size=4)
    try:

        def reduce_dict(a, b):
            out = fused_all_reduce({"a": a, "b": b})
            return out["a"].sum() + out["b"].sum()

        inputs = (torch.ones(4), torch.ones(3))
        assert dynamo.explain(reduce_dict)(*inputs).graph_break_count == 0
        torch.compile(reduce_dict, fullgraph=True)(*inputs)
    finally:
        # Reset Dynamo state and tear down the fake group so it cannot leak into
        # other serial tests (which assume an uninitialized default group).
        dynamo.reset()
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# Collective path (@pytest.mark.multigpu_static)
# -----------------------------------------------------------------------------


def _arithmetic_series(world_size: int) -> float:
    """Sum over ranks of the per-rank contribution ``rank + 1``."""
    return float(sum(k + 1 for k in range(world_size)))


@pytest.mark.multigpu_static
def test_sum_across_ranks():
    """Default SUM adds each rank's contribution element-wise."""
    dm = DistributedManager()
    assert dm.is_initialized()
    rank, world_size = dm.rank, dm.world_size
    expected = _arithmetic_series(world_size)

    reduced = fused_all_reduce(
        {
            "scalar": torch.tensor(float(rank + 1), device=dm.device),
            "vec": torch.full((3,), float(rank + 1), device=dm.device),
        }
    )

    assert reduced["scalar"].item() == pytest.approx(expected)
    assert torch.allclose(reduced["vec"], torch.full((3,), expected, device=dm.device))


@pytest.mark.multigpu_static
def test_bare_tensor_sum_across_ranks():
    """A bare tensor reduces to (and returns) a bare tensor: default SUM adds
    each rank's contribution, the one-leaf analogue of the keyed SUM."""
    dm = DistributedManager()
    rank, world_size = dm.rank, dm.world_size
    expected = _arithmetic_series(world_size)

    reduced = fused_all_reduce(torch.tensor(float(rank + 1), device=dm.device))

    assert isinstance(reduced, torch.Tensor)
    assert reduced.item() == pytest.approx(expected)


@pytest.mark.multigpu_static
def test_bare_tensor_async_sum_across_ranks():
    """``async_op=True`` on a bare tensor: the result holds this rank's local
    value until ``wait()``, after which the fused SUM is filled in place."""
    dm = DistributedManager()
    rank, world_size = dm.rank, dm.world_size
    expected = _arithmetic_series(world_size)

    result, work = fused_all_reduce(
        torch.tensor(float(rank + 1), device=dm.device), async_op=True
    )

    # Before wait(): the lone leaf still holds this rank's own (un-reduced) value.
    assert result.item() == pytest.approx(float(rank + 1))

    assert work.wait() is True
    # After wait(): the fused SUM is filled into the output in place.
    assert result.item() == pytest.approx(expected)
    assert work.is_completed()


@pytest.mark.multigpu_static
def test_heterogeneous_shapes_roundtrip():
    """0-D / 1-D / 2-D leaves reduce together, with shape/dtype/device restored."""
    dm = DistributedManager()
    rank, world_size = dm.rank, dm.world_size
    expected = _arithmetic_series(world_size)

    inputs = {
        "scalar": torch.tensor(float(rank + 1), device=dm.device),
        "vec": torch.full((3,), float(rank + 1), device=dm.device),
        "mat": torch.full((2, 2), float(rank + 1), device=dm.device),
    }
    reduced = fused_all_reduce(inputs)

    for key, original in inputs.items():
        assert reduced[key].shape == original.shape
        assert reduced[key].dtype == original.dtype
        assert reduced[key].device == original.device
        assert torch.allclose(reduced[key], torch.full_like(original, expected))


@pytest.mark.multigpu_static
def test_mixed_dtype_fp32_promotion():
    """Mixed-precision leaves reduce correctly; each leaf keeps its own dtype."""
    dm = DistributedManager()
    rank, world_size = dm.rank, dm.world_size
    expected = _arithmetic_series(world_size)

    inputs = {
        "f16": torch.tensor(float(rank + 1), dtype=torch.float16, device=dm.device),
        "f32": torch.tensor(float(rank + 1), dtype=torch.float32, device=dm.device),
        "f64": torch.tensor(float(rank + 1), dtype=torch.float64, device=dm.device),
    }
    reduced = fused_all_reduce(inputs)

    assert reduced["f16"].dtype == torch.float16
    assert reduced["f32"].dtype == torch.float32
    assert reduced["f64"].dtype == torch.float64
    # f64 is present, so the buffer accumulates in f64: f32/f64 are exact.
    assert reduced["f32"].item() == pytest.approx(expected)
    assert reduced["f64"].item() == pytest.approx(expected)
    assert reduced["f16"].item() == pytest.approx(expected, abs=1e-1)


@pytest.mark.multigpu_static
def test_key_order_determinism():
    """Differing per-rank dict insertion orders still reduce the right keys.

    Each rank inserts the same keys in a rank-dependent order. Sorted-key
    packing makes the wire layout identical across ranks, so element ``k`` is
    summed with element ``k`` (not a transposed neighbor).
    """
    dm = DistributedManager()
    rank, world_size = dm.rank, dm.world_size
    series = _arithmetic_series(world_size)

    keys = ["a", "b", "c", "d"]
    ordered_keys = keys if rank % 2 == 0 else list(reversed(keys))
    inputs = {
        key: torch.tensor(float((rank + 1) * (keys.index(key) + 1)), device=dm.device)
        for key in ordered_keys
    }
    reduced = fused_all_reduce(inputs)

    for j, key in enumerate(keys):
        # sum_r (r+1)*(j+1) = (j+1) * series
        assert reduced[key].item() == pytest.approx((j + 1) * series)


@pytest.mark.multigpu_static
def test_weighted_mean_uneven_shards():
    """The sum/count idiom yields a sample-weighted mean, not a mean-of-means."""
    dm = DistributedManager()
    rank, world_size = dm.rank, dm.world_size

    count = float(rank + 1)  # rank r holds (r+1) samples
    total = count * (rank + 1)  # whose local mean is (r+1)
    reduced = fused_all_reduce(
        {
            "sum": torch.tensor(total, device=dm.device),
            "count": torch.tensor(count, device=dm.device),
        }
    )
    weighted_mean = (reduced["sum"] / reduced["count"]).item()

    expected = sum((k + 1) ** 2 for k in range(world_size)) / sum(
        k + 1 for k in range(world_size)
    )
    assert weighted_mean == pytest.approx(expected)
    if world_size > 1:
        # The (wrong) mean-of-means would be the plain average of (r+1).
        mean_of_means = sum(k + 1 for k in range(world_size)) / world_size
        assert weighted_mean != pytest.approx(mean_of_means)


@pytest.mark.multigpu_static
def test_nested_tensordict_collective_roundtrip():
    """A nested TensorDict reduces its leaves and keeps its structure."""
    dm = DistributedManager()
    rank, world_size = dm.rank, dm.world_size
    expected = _arithmetic_series(world_size)

    nested = TensorDict(
        {
            "loss": torch.tensor(float(rank + 1), device=dm.device),
            "sub": TensorDict(
                {"x": torch.tensor(float(rank + 1), device=dm.device)}, batch_size=[]
            ),
        },
        batch_size=[],
    )
    reduced = fused_all_reduce(nested)

    assert isinstance(reduced, TensorDictBase)
    assert set(reduced.keys(include_nested=True, leaves_only=True)) == {
        "loss",
        ("sub", "x"),
    }
    assert reduced["loss"].item() == pytest.approx(expected)
    assert reduced["sub"]["x"].item() == pytest.approx(expected)


@pytest.mark.multigpu_static
@pytest.mark.parametrize(
    "op, reference", [(dist.ReduceOp.MAX, max), (dist.ReduceOp.MIN, min)]
)
def test_max_min(op, reference):
    """MAX / MIN reduce element-wise across ranks."""
    dm = DistributedManager()
    rank, world_size = dm.rank, dm.world_size

    reduced = fused_all_reduce(
        {"v": torch.tensor(float(rank + 1), device=dm.device)}, op=op
    )
    expected = float(reference(k + 1 for k in range(world_size)))
    assert reduced["v"].item() == pytest.approx(expected)


@pytest.mark.multigpu_static
def test_avg_across_ranks():
    """AVG averages each rank's contribution element-wise (mean-of-means).

    This pins the path the recipe metric reducers now depend on: each rank
    passes its rank-local mean and AVG returns the across-rank average, which
    equals the true global mean when the shards are equal-weight.
    """
    dm = DistributedManager()
    rank, world_size = dm.rank, dm.world_size
    expected = _arithmetic_series(world_size) / world_size

    reduced = fused_all_reduce(
        {
            "scalar": torch.tensor(float(rank + 1), device=dm.device),
            "vec": torch.full((3,), float(rank + 1), device=dm.device),
        },
        op=dist.ReduceOp.AVG,
    )

    assert reduced["scalar"].item() == pytest.approx(expected)
    assert torch.allclose(reduced["vec"], torch.full((3,), expected, device=dm.device))


@pytest.mark.multigpu_static
def test_integer_sum_is_exact():
    """A homogeneous int64 bundle sums exactly across ranks (no float cast)."""
    dm = DistributedManager()
    rank, world_size = dm.rank, dm.world_size
    expected = sum(k + 1 for k in range(world_size))

    reduced = fused_all_reduce(
        {"count": torch.tensor(rank + 1, dtype=torch.int64, device=dm.device)}
    )
    assert reduced["count"].dtype == torch.int64
    assert reduced["count"].item() == expected


@pytest.mark.multigpu_static
def test_async_op_sum_across_ranks():
    """``async_op=True`` matches the sync SUM once ``wait()`` lands; before
    ``wait()`` the result holds this rank's own (un-reduced) values."""
    dm = DistributedManager()
    assert dm.is_initialized()
    rank, world_size = dm.rank, dm.world_size
    expected = _arithmetic_series(world_size)

    result, work = fused_all_reduce(
        {
            "scalar": torch.tensor(float(rank + 1), device=dm.device),
            "vec": torch.full((3,), float(rank + 1), device=dm.device),
        },
        async_op=True,
    )

    # Deferred unpack: until wait(), the outputs are this rank's local clones.
    assert result["scalar"].item() == pytest.approx(float(rank + 1))

    assert work.wait() is True
    # After wait(), the fused SUM is filled into the outputs in place.
    assert result["scalar"].item() == pytest.approx(expected)
    assert torch.allclose(result["vec"], torch.full((3,), expected, device=dm.device))
    assert work.is_completed()


@pytest.mark.multigpu_static
def test_async_op_nested_tensordict_sum_across_ranks():
    """``async_op=True`` with a (nested) TensorDict: the leaves hold this rank's
    local values until ``wait()``, after which the fused SUM is filled in place.

    This pins the aliasing contract the deferred unpack relies on - that
    ``TensorDict.__setitem__`` keeps the assigned tensor object, so ``wait()``'s
    in-place ``copy_`` reaches the caller's TensorDict leaves - under a real
    collective, which the serial no-op tests cannot exercise.
    """
    dm = DistributedManager()
    assert dm.is_initialized()
    rank, world_size = dm.rank, dm.world_size
    expected = _arithmetic_series(world_size)

    nested = TensorDict(
        {
            "loss": torch.tensor(float(rank + 1), device=dm.device),
            "sub": TensorDict({"x": torch.tensor(float(rank + 1), device=dm.device)}),
        },
    )
    result, work = fused_all_reduce(nested, async_op=True)
    assert isinstance(result, TensorDictBase)

    # Before wait(): each leaf still holds this rank's own (un-reduced) value.
    assert result["loss"].item() == pytest.approx(float(rank + 1))
    assert result["sub", "x"].item() == pytest.approx(float(rank + 1))

    assert work.wait() is True
    # After wait(): the fused SUM lands in place, including the nested leaf.
    assert result["loss"].item() == pytest.approx(expected)
    assert result["sub", "x"].item() == pytest.approx(expected)
    assert work.is_completed()
