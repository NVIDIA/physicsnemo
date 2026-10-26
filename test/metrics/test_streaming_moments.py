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

"""Independent reference checks for tensor-aware streaming statistics."""

import pickle

import pytest
import torch

from physicsnemo.metrics.general.ensemble_metrics import Mean, Variance


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_stream_and_combine(device, dtype, axis):
    torch.manual_seed(19)
    source = torch.randn(17, 5, 3, device=device, dtype=dtype)
    data = source.movedim(0, axis).requires_grad_()
    stats = Variance(mode="tensor", correction=0)
    average = Mean(mode="tensor")
    left = Variance(mode="tensor", correction=0)
    right = Variance(mode="tensor", correction=1)
    for chunk in data.split([2, 7, 8], dim=axis):
        stats.update(chunk, dim=axis)
        average.update(chunk, dim=axis)
    left.update(data.narrow(axis, 0, 2), dim=axis)
    right.update(data.narrow(axis, 2, 15), dim=axis)
    right_before = right.finalize().clone()
    left.combine(right)
    reference = source.to(torch.float64 if dtype == torch.float64 else torch.float32)
    for actual in (stats.finalize(), left.finalize()):
        torch.testing.assert_close(actual, reference.var(0, correction=0))
        assert not actual.requires_grad
    torch.testing.assert_close(stats.mean, reference.mean(0))
    torch.testing.assert_close(average.finalize(), reference.mean(0))
    torch.testing.assert_close(right.finalize(), right_before)
    torch.testing.assert_close(
        stats.finalize(correction=1), reference.var(0, correction=1)
    )
    torch.testing.assert_close(stats.finalize(std=True), reference.std(0, correction=0))
    assert stats.count == 17
    # Reading cannot expose mutable accumulator storage.
    stats.mean.zero_()
    torch.testing.assert_close(stats.mean, reference.mean(0))


def test_single_observations_and_large_offset(device):
    values = (
        1.0e9
        + torch.arange(120, dtype=torch.float64, device=device).reshape(20, 6) / 17
    )
    stats = Variance(mode="tensor", correction=0, like=values[0])
    average = Mean(mode="tensor", like=values[0])
    for value in values:
        stats.update(value, dim=None)
        average.update(value, dim=None)
    torch.testing.assert_close(
        stats.finalize(), values.var(0, correction=0), rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(average.finalize(), values.mean(0))
    assert stats.count == 20


def test_empty_correction_reset_and_validation(device):
    data = torch.arange(12, device=device, dtype=torch.float32).reshape(4, 3)
    stats = Variance(mode="tensor")
    with pytest.raises(ValueError, match="Insufficient"):
        stats.finalize()
    with pytest.raises(ValueError, match="empty"):
        _ = stats.mean
    assert stats.update(data[:0]) is None
    assert stats.update(data[:1]) is None
    with pytest.raises(ValueError, match="Insufficient"):
        stats.finalize()
    stats.update(data[1:])
    torch.testing.assert_close(stats.finalize(), data.var(0))
    snapshot = stats.finalize()
    with pytest.raises(ValueError, match="shape"):
        stats.update(torch.ones(2, 4, device=device))
    torch.testing.assert_close(stats.finalize(), snapshot)
    stats.reset()
    assert stats.count == 0
    torch.testing.assert_close(stats(data), data.var(0))
    stats.combine(Variance(mode="tensor"))
    assert stats.count == 4
    with pytest.raises(ValueError):
        stats.combine(Mean(mode="tensor"))
    with pytest.raises(TypeError):
        stats.update(data.long())
    with pytest.raises(IndexError):
        stats.update(data, dim=3)
    for correction in (-1, 0.5, True):
        with pytest.raises(ValueError):
            stats.finalize(correction=correction)
    with pytest.raises(ValueError):
        Variance(mode="tensor", dtype=torch.float16)
    with pytest.raises(ValueError):
        Mean(mode="tensor", input_shape=(3,))
    with pytest.raises(ValueError):
        Mean(mode="typo")


def test_gp_total_variance(device):
    means = torch.tensor([[1.0, 3.0], [4.0, 5.0], [7.0, 1.0]], device=device)
    variances = torch.tensor([[2.0, 1.0], [3.0, 8.0], [4.0, 3.0]], device=device)
    between = Variance(mode="tensor", correction=0)
    within = Mean(mode="tensor")
    for mean, variance in zip(means, variances):
        between.update(mean, dim=None)
        within.update(variance, dim=None)
    total = within.finalize() + between.finalize()
    # Independent E[X^2] - E[X]^2 calculation for this well-conditioned mixture.
    expected = (variances + means.square()).mean(0) - means.mean(0).square()
    torch.testing.assert_close(total, expected)


def test_precision_and_ownership(device):
    source = torch.randn(5, 3, device=device)
    reference = source.double().clone()
    stats = Variance(mode="tensor", dtype=torch.float64, correction=0)
    stats.update(source)
    source.zero_()
    torch.testing.assert_close(stats.mean, reference.mean(0))
    torch.testing.assert_close(stats.finalize(), reference.var(0, correction=0))


@pytest.mark.parametrize("statistic", [Mean, Variance])
def test_empty_combinations_and_input_ownership(device, statistic):
    """Empty states preserve identity and shape validation without adding observations."""
    value = torch.randn(5, 3, device=device, dtype=torch.float64)
    expected = value.mean(0) if statistic is Mean else value.var(0)
    populated, empty = statistic(mode="tensor"), statistic(mode="tensor")
    populated.update(value)
    empty.combine(populated)
    with torch.no_grad():
        value.fill_(99)
    populated.reset()
    torch.testing.assert_close(empty.finalize(), expected)
    empty.combine(populated)
    torch.testing.assert_close(empty.finalize(), expected)
    assert empty.count == 5
    empty.update(torch.zeros(2, 3, dtype=torch.float32, device=device))
    assert empty.dtype == torch.float64
    with pytest.raises(ValueError):
        empty.combine(
            statistic(
                mode="tensor", like=torch.zeros(4, device=device, dtype=torch.float64)
            )
        )
    assert empty.count == 7


def test_mean_does_not_retain_second_moment(device):
    """Mean-only workflows do not allocate an unused variance field."""
    value = torch.arange(12, device=device, dtype=torch.float32).reshape(4, 3)
    stats = Mean(mode="tensor")
    stats.update(value)
    assert stats._moments.m2 is None
    snapshot = stats.finalize()
    snapshot.fill_(99)
    torch.testing.assert_close(stats.finalize(), value.mean(0))


@pytest.mark.parametrize("statistic", [Mean, Variance])
def test_legacy_pickle_and_optional_attributes(statistic):
    """Pre-tensor-mode pickles and attribute discovery retain their old behavior."""
    data = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    state = statistic((3,))
    state(data[:2])
    state.__dict__.pop("_moments", None)
    restored = pickle.loads(pickle.dumps(state))  # noqa: S301 - locally constructed state
    assert not hasattr(restored, "count")
    restored.update(data[2:])
    expected = data.mean(0) if statistic is Mean else data.var(0)
    torch.testing.assert_close(restored.finalize(), expected)


def test_base_subclasses_keep_their_own_count():
    """New opt-in methods must not change the API of unrelated statistics."""
    from physicsnemo.metrics.general import ensemble_metrics as em
    from physicsnemo.metrics.general.histogram import Histogram

    # An unrelated subclass can continue managing its own count.
    class CustomStatistic(em.StreamingStatistic):
        def __init__(self):
            super().__init__((3,))
            self.count = 0

    assert CustomStatistic().count == 0
    for name in ("count", "combine", "reset"):
        assert not hasattr(Histogram, name)


@pytest.mark.parametrize("statistic", [Mean, Variance])
def test_settled_dtype_and_update_return(device, statistic):
    """Inference accepts safe widening, rejects silent narrowing, and survives reset."""
    data = torch.arange(12, device=device, dtype=torch.float64).reshape(4, 3)
    state = statistic(mode="tensor")
    assert state.device is None and state.input_shape is None and state.dtype is None
    assert state.update(data[:2]) is None
    assert state.dtype == torch.float64
    assert state.update(data[2:].float()) is None
    expected = data.mean(0) if statistic is Mean else data.var(0)
    torch.testing.assert_close(state.finalize(), expected)
    for kwargs in ({}, {"like": data[0].float()}):
        inferred = statistic(mode="tensor", **kwargs)
        inferred.update(data.float())
        for reset in (False, True):
            if reset:
                inferred.reset()
            with pytest.raises(ValueError, match="dtype=torch.float64"):
                inferred.update(data)
    explicit = statistic(mode="tensor", dtype=torch.float32)
    explicit.update(data)
    assert explicit.dtype == torch.float32
    explicit.reset().update(data)
    torch.testing.assert_close(explicit.finalize(), expected.float())
    with pytest.raises(TypeError, match=statistic.__name__):
        statistic(mode="tensor", unexpected=True)


@pytest.mark.parametrize("statistic", [Mean, Variance])
def test_empty_updates_do_not_retain_extra_field(device, statistic, monkeypatch):
    """Padding after real data needs no persistent extra output-sized template."""
    data = torch.randn(4, 3, device=device)
    state = statistic(mode="tensor")
    state.update(data)
    expected = state.finalize()

    # Updates should not materialize a result the caller did not request.
    def forbidden_read(*args, **kwargs):
        raise AssertionError("update must not finalize")

    monkeypatch.setattr(state, "finalize", forbidden_read)
    assert state.update(data[:0]) is None
    assert state._moments.template is None
    state.combine(statistic(mode="tensor", like=data[0]))
    assert state._moments.template is None
    actual = (
        state._moments.read_mean()
        if statistic is Mean
        else state._moments.read_variance(1)
    )
    torch.testing.assert_close(actual, expected)


def test_deprecated_base_aliases_and_silent_module_imports():
    """Both old exports resolve to the same base without warning on module import."""
    import importlib
    import warnings

    from physicsnemo.core.warnings import LegacyFeatureWarning
    from physicsnemo.metrics.general import ensemble_metrics as em

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", LegacyFeatureWarning)
        diffusion = importlib.import_module("physicsnemo.diffusion.metrics")
        importlib.reload(diffusion)
    assert not any("EnsembleMetrics" in str(w.message) for w in recorded)
    for module in (em, diffusion):
        with pytest.warns(LegacyFeatureWarning):
            alias = module.EnsembleMetrics
        assert alias is em.StreamingStatistic

        class LegacySubclass(alias):
            pass

        assert isinstance(LegacySubclass((3,)), em.StreamingStatistic)


@pytest.mark.parametrize("statistic", [Mean, Variance])
def test_combine_settled_dtype_independent_of_configuration(device, statistic):
    """Explicit and inferred states can combine when their actual precisions match."""
    data = torch.arange(12, device=device, dtype=torch.float64).reshape(4, 3)
    left = statistic(mode="tensor", dtype=torch.float64)
    right = statistic(mode="tensor")
    left.update(data[:2])
    right.update(data[2:])
    left.combine(right)
    expected = data.mean(0) if statistic is Mean else data.var(0)
    torch.testing.assert_close(left.finalize(), expected)
    with pytest.raises(ValueError, match="dtype"):
        left.combine(statistic(mode="tensor", dtype=torch.float32))


@pytest.mark.parametrize("statistic", [Mean, Variance])
@pytest.mark.parametrize("dim,error", [(2, IndexError), (True, TypeError)])
def test_failed_first_update_does_not_settle_dtype(device, statistic, dim, error):
    """Rejected inputs cannot lock an empty accumulator to their precision."""
    stats = statistic(mode="tensor")
    data = torch.arange(12, device=device, dtype=torch.float32).reshape(4, 3)
    with pytest.raises(error):
        stats.update(data, dim=dim)
    assert stats.count == 0
    assert stats.dtype is stats._moments.dtype is None

    data = data.double()
    stats.update(data)
    expected = data.mean(0) if statistic is Mean else data.var(0)
    torch.testing.assert_close(stats.finalize(), expected)
    assert stats.count == data.shape[0]
    assert (
        stats.dtype == stats._moments.dtype == stats.finalize().dtype == torch.float64
    )


@pytest.mark.parametrize(
    "module_name,other_exports",
    [
        (
            "physicsnemo.metrics.general.ensemble_metrics",
            ("Mean", "Variance", "Tensor"),
        ),
        (
            "physicsnemo.diffusion.metrics",
            ("Mean", "Variance", "Histogram", "crps", "EDMLoss", "MSEDSMLoss"),
        ),
    ],
)
def test_deprecated_base_wildcard_import(module_name, other_exports):
    """Legacy wildcard imports can still define subclasses of EnsembleMetrics."""
    import importlib

    from physicsnemo.core.warnings import LegacyFeatureWarning
    from physicsnemo.metrics.general.ensemble_metrics import StreamingStatistic

    module = importlib.import_module(module_name)
    namespace = {}
    with pytest.warns(LegacyFeatureWarning, match="EnsembleMetrics is deprecated"):
        exec(  # noqa: S102 - exercise wildcard syntax with fixed test module names
            f"from {module_name} import *\n"
            "class LegacyStatistic(EnsembleMetrics):\n"
            "    pass\n",
            namespace,
        )
    assert namespace["EnsembleMetrics"] is StreamingStatistic
    assert namespace["StreamingStatistic"] is StreamingStatistic
    assert isinstance(namespace["LegacyStatistic"]((3,)), StreamingStatistic)
    for name in other_exports:
        assert namespace[name] is getattr(module, name)
