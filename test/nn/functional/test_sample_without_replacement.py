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

"""Tests for sampling indices without replacement."""

import importlib

import pytest
import torch

from physicsnemo.nn.functional import sample_without_replacement
from physicsnemo.nn.functional.sample_without_replacement import (
    SampleWithoutReplacement,
)

sampling_module = importlib.import_module(
    "physicsnemo.nn.functional.sample_without_replacement"
)


def test_exact_uniform_sampling_is_unique_and_deterministic(monkeypatch):
    monkeypatch.setattr(sampling_module, "_SAMPLE_CHUNK_SIZE", 2)
    monkeypatch.setattr(sampling_module, "_RANDPERM_POPULATION_LIMIT", 2)

    first = sample_without_replacement(
        7,
        4,
        generator=torch.Generator().manual_seed(0),
    )
    second = sample_without_replacement(
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

    indices = sample_without_replacement(
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

    indices = sample_without_replacement(
        weights.numel(),
        3,
        weights=weights,
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

    indices = sample_without_replacement(
        weights.numel(),
        weights.numel(),
        weights=weights,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(indices, expected)


def test_exact_weighted_sampling_applies_chunk_offsets(monkeypatch):
    monkeypatch.setattr(sampling_module, "_SAMPLE_CHUNK_SIZE", 4)
    weights = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    indices = sample_without_replacement(
        weights.numel(),
        2,
        weights=weights,
        generator=torch.Generator().manual_seed(0),
    )

    assert set(indices.tolist()) == {4, 5}


@pytest.mark.parametrize("population_size", [10_000, 100_000_000])
def test_poisson_gap_sampling_is_ordered_and_unique(population_size):
    count = 1_000

    indices = sample_without_replacement(
        population_size,
        count,
        strategy="poisson_gap",
        generator=torch.Generator().manual_seed(0),
    )

    assert indices.shape == (count,)
    assert indices.dtype == torch.long
    assert indices.min() >= 0
    assert indices.max() < population_size
    assert torch.all(indices[1:] - indices[:-1] >= 1)


def test_poisson_gap_sampling_full_population():
    indices = sample_without_replacement(5, 5, strategy="poisson_gap")

    torch.testing.assert_close(indices, torch.arange(5))


def test_empty_population():
    indices = sample_without_replacement(0, 0)

    assert indices.dtype == torch.long
    assert indices.shape == (0,)


@pytest.mark.parametrize(
    ("population_size", "count"),
    [(-1, 0), (3, -1), (3, 4)],
)
def test_rejects_invalid_sizes(population_size, count):
    with pytest.raises(ValueError):
        sample_without_replacement(population_size, count)


def test_rejects_invalid_strategy():
    with pytest.raises(ValueError, match="strategy"):
        sample_without_replacement(3, 2, strategy="unknown")


def test_rejects_weights_for_poisson_gap():
    with pytest.raises(ValueError, match="does not support weights"):
        sample_without_replacement(
            3,
            2,
            weights=torch.ones(3),
            strategy="poisson_gap",
        )


def test_rejects_invalid_weights():
    with pytest.raises(ValueError, match="1D"):
        sample_without_replacement(4, 2, weights=torch.ones(2, 2))
    with pytest.raises(ValueError, match="population_size"):
        sample_without_replacement(4, 2, weights=torch.ones(3))
    with pytest.raises(TypeError, match="floating point"):
        sample_without_replacement(3, 2, weights=torch.ones(3, dtype=torch.long))


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
        sample_without_replacement(weights.numel(), 1, weights=weights)


def test_make_inputs_forward(device: str):
    label, args, kwargs = next(
        iter(SampleWithoutReplacement.make_inputs_forward(device=device))
    )

    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    indices = SampleWithoutReplacement.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    assert indices.shape == (args[1],)
    assert indices.device.type == torch.device(device).type
