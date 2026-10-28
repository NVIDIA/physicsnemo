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

"""Public rule extension and generic accumulator behavior."""

from dataclasses import dataclass

import pytest
import torch

from physicsnemo.metrics.general.ensemble_metrics import Mean, Variance
from physicsnemo.metrics.general.moment_rules import MeanRule, VarianceRule
from physicsnemo.metrics.general.tensor_accumulator import TensorAccumulator
from test._weighted_mean_rule import WeightedMeanRule


@pytest.mark.parametrize("dim", [0, 1, -1, None])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_weighted_mean_extension(device, dim, dtype):
    """Unequal batches and independently accumulated states match a dense oracle."""
    values = torch.arange(30, device=device, dtype=dtype).reshape(10, 3) / 7
    weights = torch.linspace(0, 2, 30, device=device, dtype=dtype).reshape_as(values)
    values.requires_grad_()
    weights.requires_grad_()
    promoted = torch.float64 if dtype == torch.float64 else torch.float32
    state = TensorAccumulator(WeightedMeanRule())
    tail = TensorAccumulator(WeightedMeanRule())
    if dim is None:
        state.update(values, dim=None, weights=weights + 1)
        tail.update(values + 3, dim=None, weights=weights + 2)
        w1, w2 = (weights + 1).to(promoted), (weights + 2).to(promoted)
        expected = (values.to(promoted) * w1 + (values + 3).to(promoted) * w2) / (
            w1 + w2
        )
        count = 2
    else:
        data_chunks = values.split(2, dim)
        weight_chunks = weights.split(2, dim)
        for data, weight in zip(data_chunks[:-1], weight_chunks[:-1]):
            state.update(data, dim=dim, weights=weight)
        tail.update(data_chunks[-1], dim=dim, weights=weight_chunks[-1])
        expected = (values.to(promoted) * weights.to(promoted)).sum(dim) / weights.to(
            promoted
        ).sum(dim)
        count = values.shape[dim]
    before = tail.finalize()
    state.combine(tail)
    torch.testing.assert_close(state.finalize(), expected)
    torch.testing.assert_close(tail.finalize(), before)
    assert state.count == count
    assert state.dtype == promoted
    assert not state.finalize().requires_grad
    state.finalize().zero_()
    torch.testing.assert_close(state.finalize(), expected)
    tail.reset()
    torch.testing.assert_close(state.finalize(), expected)


@pytest.mark.parametrize("rule", [MeanRule(), VarianceRule()])
def test_builtin_rule_matches_facade(device, rule):
    """The public facades and direct rule use follow the same lifecycle."""
    values = 1e9 + torch.randn(13, 4, device=device, dtype=torch.float64)
    direct = TensorAccumulator(rule)
    facade = (
        Mean(mode="tensor") if isinstance(rule, MeanRule) else Variance(mode="tensor")
    )
    for batch in values.split(3):
        direct.update(batch)
        facade.update(batch)
    torch.testing.assert_close(direct.finalize(), facade.finalize())
    expected = values.mean(0) if isinstance(rule, MeanRule) else values.var(0)
    torch.testing.assert_close(direct.finalize(), expected, rtol=1e-6, atol=1e-7)


def test_weighted_validation_and_transactional_dtype(device):
    """A rejected first update must not commit precision, count or state."""
    state = TensorAccumulator(WeightedMeanRule())
    x = torch.ones(3, 2, device=device)
    with pytest.raises(IndexError):
        state.update(x, weights=x, dim=3)
    with pytest.raises(ValueError, match="shape"):
        state.update(x, weights=x[:, :1])
    with pytest.raises(ValueError, match="nonnegative"):
        state.update(x, weights=-x)
    assert state.dtype is None and state.count == 0
    state.update(x, weights=x.double())
    assert state.dtype == state.finalize().dtype == torch.float64
    expected = state.finalize()
    for invalid in [float("nan"), float("inf"), -1.0]:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            state.update(x, weights=torch.full_like(x, invalid))
        assert state.count == 3
        torch.testing.assert_close(state.finalize(), expected)
    with pytest.raises(TypeError, match="floating-point"):
        state.update(x, weights=x.long())


def test_weighted_zero_weights_reset_and_empty_updates(device):
    """Zero weights can be followed by valid weights; empty batches add no count."""
    x = torch.ones(3, 2, device=device)
    state = TensorAccumulator(WeightedMeanRule())
    state.update(x[:0], weights=x[:0])
    with pytest.raises(ValueError, match="empty"):
        state.finalize()
    state.update(x, weights=torch.zeros_like(x))
    with pytest.raises(ValueError, match="positive finite"):
        state.finalize()
    state.update(x[:1] * 7, weights=x[:1])
    torch.testing.assert_close(state.finalize(), x[0] * 7)
    assert state.count == 4
    state.reset()
    assert state.dtype == torch.float32 and state.count == 0
    with pytest.raises(ValueError, match="shape"):
        state.update(x[:, :1], weights=x[:, :1])
    state.update(x, weights=x)
    torch.testing.assert_close(state.finalize(), x[0])


@dataclass(frozen=True)
class PowerSumRule:
    """Example configured rule with multiple tensor shapes and no moment fields."""

    power: int = 2

    def summarize(self, values, *, dim, reduce_sum):
        total = reduce_sum(values.pow(self.power), dim)
        checksum = total
        for _ in range(checksum.ndim):
            checksum = reduce_sum(checksum, 0)
        return {"total": total, "checksum": checksum, "batches": 1}

    def merge(self, left, right):
        return {key: left[key] + right[key] for key in left}

    def result(self, state, *, reduce_sum, checksum=False):
        return state["checksum" if checksum else "total"]


def test_configured_rules_and_differently_shaped_state(device):
    """Configuration, field shapes and result options belong to the rule."""
    x = torch.arange(15, device=device, dtype=torch.float64).reshape(5, 3)
    state = TensorAccumulator(PowerSumRule(3))
    other = TensorAccumulator(PowerSumRule(3))
    state.update(x[:2])
    other.update(x[2:])
    state.combine(other)
    torch.testing.assert_close(state.finalize(), x.pow(3).sum(0))
    torch.testing.assert_close(state.finalize(checksum=True), x.pow(3).sum())
    with pytest.raises(ValueError, match="matching rules"):
        state.combine(TensorAccumulator(PowerSumRule(2)))
    with pytest.raises(ValueError, match="matching rules"):
        state.combine(TensorAccumulator(MeanRule()))
    assert state.count == 5


def test_failed_combine_preserves_unsettled_dtype(device):
    """An incompatible rule must not transfer its dtype or observations."""
    empty = TensorAccumulator(MeanRule())
    other = TensorAccumulator(VarianceRule())
    other.update(torch.ones(2, 3, device=device, dtype=torch.float32))
    with pytest.raises(ValueError, match="matching rules"):
        empty.combine(other)
    assert empty.dtype is None and empty.count == 0
    empty.update(torch.ones(2, 3, device=device, dtype=torch.float64))
    assert empty.dtype == torch.float64


@dataclass(frozen=True)
class InvalidMergeRule(MeanRule):
    """A faulty extension must not corrupt a previously valid state."""

    def merge(self, left, right):
        state = super().merge(left, right)
        state["mean"] = state["mean"][:1]
        return state


def test_invalid_merge_does_not_commit(device):
    state = TensorAccumulator(InvalidMergeRule())
    x = torch.randn(3, 4, device=device)
    state.update(x)
    before = state.finalize()
    with pytest.raises(ValueError, match="shape"):
        state.update(x)
    assert state.count == 3
    torch.testing.assert_close(state.finalize(), before)
    other = TensorAccumulator(InvalidMergeRule())
    other.update(x)
    with pytest.raises(ValueError, match="shape"):
        state.combine(other)
    assert state.count == 3
    torch.testing.assert_close(state.finalize(), before)


@pytest.mark.parametrize("width", [0, 4])
def test_fresh_variance_result_is_not_copied(device, monkeypatch, width):
    """Fresh results, including empty fields, pass through without another copy."""
    original = VarianceRule.result
    returned = []

    def capture(self, state, **options):
        result = original(self, state, **options)
        returned.append(result)
        return result

    monkeypatch.setattr(VarianceRule, "result", capture)
    values = torch.arange(5 * width, device=device, dtype=torch.float64).reshape(
        5, width
    )
    state = TensorAccumulator(VarianceRule())
    state.update(values)
    result = state.finalize()
    assert result is returned[-1]
    torch.testing.assert_close(result, values.var(0))
    result.zero_()
    torch.testing.assert_close(state.finalize(), values.var(0))


@pytest.mark.parametrize("alias", ["direct", "detached", "offset", "other_field"])
def test_result_aliases_cannot_mutate_state(device, monkeypatch, alias):
    """Protect every tensor field, including views with a different data pointer."""
    returned = []

    def result(self, state, **options):
        value = state["checksum" if alias == "other_field" else "total"]
        if alias == "detached":
            value = value.detach()
        elif alias == "offset":
            value = value[1:]
        returned.append(value)
        return value

    monkeypatch.setattr(PowerSumRule, "result", result)
    state = TensorAccumulator(PowerSumRule())
    state.update(torch.arange(20, device=device, dtype=torch.float64).reshape(5, 4))
    before = {
        key: value.clone()
        for key, value in state._state.items()
        if isinstance(value, torch.Tensor)
    }
    output = state.finalize()
    assert output is not returned[-1]
    torch.testing.assert_close(output, returned[-1])
    output.fill_(-999)
    for key, value in before.items():
        torch.testing.assert_close(state._state[key], value)


@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("kind", ["mean_rule", "variance_rule", "mean", "variance"])
def test_self_combine_is_rejected_without_mutation(device, populated, kind):
    """Both the engine and public facades reject duplicate self-observations."""
    if kind.endswith("_rule"):
        state = TensorAccumulator(MeanRule() if kind == "mean_rule" else VarianceRule())
    else:
        state = (Mean if kind == "mean" else Variance)(mode="tensor")
    if populated:
        state.update(torch.arange(12, device=device, dtype=torch.float64).reshape(3, 4))
    before = state.finalize() if populated else None
    count, dtype = state.count, state.dtype
    with pytest.raises(ValueError, match="itself"):
        state.combine(state)
    assert state.count == count and state.dtype == dtype
    if populated:
        torch.testing.assert_close(state.finalize(), before)
