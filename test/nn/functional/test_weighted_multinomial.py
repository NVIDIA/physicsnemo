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

"""Tests for weighted multinomial index sampling."""

import importlib
import inspect

import pytest
import torch

from physicsnemo.nn.functional import weighted_multinomial
from physicsnemo.nn.functional.weighted_multinomial import WeightedMultinomial

sampling_module = importlib.import_module(
    "physicsnemo.nn.functional.weighted_multinomial"
)


def test_public_signature_follows_torch_multinomial():
    signature = inspect.signature(weighted_multinomial)

    assert list(signature.parameters) == [
        "input",
        "num_samples",
        "replacement",
        "generator",
        "strategy",
        "device",
        "implementation",
    ]
    assert signature.parameters["replacement"].default is False
    assert signature.parameters["generator"].kind is inspect.Parameter.KEYWORD_ONLY


def test_exact_uniform_sampling_is_unique_and_deterministic(monkeypatch):
    monkeypatch.setattr(sampling_module, "_SAMPLE_CHUNK_SIZE", 2)
    monkeypatch.setattr(sampling_module, "_RANDPERM_POPULATION_LIMIT", 2)

    first = weighted_multinomial(
        7,
        4,
        generator=torch.Generator().manual_seed(0),
    )
    second = weighted_multinomial(
        7,
        4,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(first, second)
    assert torch.unique(first).numel() == 4
    assert first.min() >= 0
    assert first.max() < 7


def test_exact_uniform_small_population_matches_randperm():
    expected = torch.randperm(8, generator=torch.Generator().manual_seed(0))

    indices = weighted_multinomial(
        8,
        8,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(indices, expected)


def test_exact_weighted_sampling_avoids_torch_category_limit(monkeypatch):
    def reject_multinomial(*args, **kwargs):
        raise AssertionError(
            "torch.multinomial rejects inputs with more than 2**24 categories"
        )

    monkeypatch.setattr(torch, "multinomial", reject_multinomial)
    monkeypatch.setattr(sampling_module, "_SAMPLE_CHUNK_SIZE", 2)
    weights = torch.tensor([1.0, 0.0, 4.0, 2.0, 3.0])

    indices = weighted_multinomial(
        weights,
        3,
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.unique(indices).numel() == 3
    assert 1 not in indices


def test_exact_sampling_orders_full_population_by_race_keys(monkeypatch):
    monkeypatch.setattr(sampling_module, "_SAMPLE_CHUNK_SIZE", 2)
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    expected_generator = torch.Generator().manual_seed(0)
    expected_keys = torch.cat(
        [
            torch.empty_like(chunk)
            .exponential_(generator=expected_generator)
            .div_(chunk)
            for chunk in weights.split(2)
        ]
    )
    expected = expected_keys.argsort()

    indices = weighted_multinomial(
        weights,
        weights.numel(),
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(indices, expected)


def test_exact_weighted_sampling_applies_chunk_offsets(monkeypatch):
    monkeypatch.setattr(sampling_module, "_SAMPLE_CHUNK_SIZE", 4)
    weights = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    indices = weighted_multinomial(
        weights,
        2,
        generator=torch.Generator().manual_seed(0),
    )

    assert set(indices.tolist()) == {4, 5}


def test_uniform_sampling_with_replacement_matches_randint():
    expected = torch.randint(4, (20,), generator=torch.Generator().manual_seed(0))

    indices = weighted_multinomial(
        4,
        20,
        replacement=True,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(indices, expected)
    assert torch.unique(indices).numel() < indices.numel()


def test_weighted_sampling_with_replacement_matches_torch_multinomial():
    weights = torch.tensor([0.0, 1.0, 3.0])
    expected = torch.multinomial(
        weights,
        20,
        replacement=True,
        generator=torch.Generator().manual_seed(0),
    )

    indices = weighted_multinomial(
        weights,
        20,
        replacement=True,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(indices, expected)
    assert 0 not in indices


@pytest.mark.parametrize("population_size", [10_000, 100_000_000])
def test_poisson_gap_sampling_is_ordered_and_unique(population_size):
    num_samples = 1_000

    indices = weighted_multinomial(
        population_size,
        num_samples,
        strategy="poisson_gap",
        generator=torch.Generator().manual_seed(0),
    )

    assert indices.shape == (num_samples,)
    assert indices.dtype == torch.long
    assert indices.min() >= 0
    assert indices.max() < population_size
    assert torch.all(indices[1:] - indices[:-1] >= 1)


def test_poisson_gap_sampling_full_population():
    indices = weighted_multinomial(5, 5, strategy="poisson_gap")

    torch.testing.assert_close(indices, torch.arange(5))


@pytest.mark.parametrize("input", [0, torch.empty(0)])
def test_empty_population(input):
    indices = weighted_multinomial(input, 0)

    assert indices.dtype == torch.long
    assert indices.shape == (0,)


@pytest.mark.parametrize(
    ("input", "num_samples", "replacement"),
    [(-1, 0, False), (3, -1, False), (3, 4, False), (0, 1, True)],
)
def test_rejects_invalid_sizes(input, num_samples, replacement):
    with pytest.raises(ValueError):
        weighted_multinomial(input, num_samples, replacement)


@pytest.mark.parametrize("input", [True, 1.5, [1.0, 2.0]])
def test_rejects_invalid_input_type(input):
    with pytest.raises(TypeError, match="input"):
        weighted_multinomial(input, 1)


@pytest.mark.parametrize("num_samples", [True, 1.5])
def test_rejects_invalid_num_samples_type(num_samples):
    with pytest.raises(TypeError, match="num_samples"):
        weighted_multinomial(3, num_samples)


def test_rejects_invalid_strategy():
    with pytest.raises(ValueError, match="strategy"):
        weighted_multinomial(3, 2, strategy="unknown")


def test_rejects_poisson_gap_for_weights():
    with pytest.raises(ValueError, match="integer uniform population"):
        weighted_multinomial(
            torch.ones(3),
            2,
            strategy="poisson_gap",
        )


def test_rejects_poisson_gap_with_replacement():
    with pytest.raises(ValueError, match="does not support replacement"):
        weighted_multinomial(
            3,
            2,
            replacement=True,
            strategy="poisson_gap",
        )


def test_rejects_invalid_weights():
    with pytest.raises(ValueError, match="1D"):
        weighted_multinomial(torch.ones(2, 2), 2)
    with pytest.raises(TypeError, match="floating point"):
        weighted_multinomial(torch.ones(3, dtype=torch.long), 2)


@pytest.mark.parametrize(
    "weights",
    [
        torch.tensor([1.0, -1.0]),
        torch.tensor([1.0, torch.nan]),
        torch.tensor([1.0, torch.inf]),
        torch.zeros(2),
    ],
)
def test_rejects_invalid_weight_values(weights):
    with pytest.raises(RuntimeError, match="weights"):
        weighted_multinomial(weights, 1)


def test_rejects_too_few_positive_weights_without_replacement():
    with pytest.raises(RuntimeError, match="positive input weights"):
        weighted_multinomial(torch.tensor([0.0, 1.0, 0.0]), 2)


def test_make_inputs_forward(device: str):
    label, args, kwargs = next(
        iter(WeightedMultinomial.make_inputs_forward(device=device))
    )

    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    indices = WeightedMultinomial.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    assert indices.shape == (args[1],)
    assert indices.device.type == torch.device(device).type
