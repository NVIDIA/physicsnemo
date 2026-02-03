# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Tests for collation utilities."""

import pytest
import torch
from tensordict import TensorDict

import physicsnemo.datapipes as dp

# ============================================================================
# DefaultCollator (stack-based)
# ============================================================================


def test_default_collate_basic(batch_of_samples):
    collator = dp.DefaultCollator(collate_metadata=False)
    batched_data = collator(batch_of_samples)

    # 4 samples, each with shape (10, 3) -> (4, 10, 3)
    assert batched_data["x"].shape == (4, 10, 3)
    assert batched_data["y"].shape == (4, 10)


def test_default_collate_metadata(batch_of_samples):
    collator = dp.DefaultCollator(collate_metadata=True)
    batched_data, metadata_list = collator(batch_of_samples)

    # Metadata should be collected into lists
    assert isinstance(metadata_list, list)
    assert len(metadata_list) == 4
    assert [m["idx"] for m in metadata_list] == [0, 1, 2, 3]


def test_default_collate_empty_raises():
    collator = dp.DefaultCollator()

    with pytest.raises(ValueError, match="empty sequence"):
        collator([])


def test_default_collate_mismatched_keys_raises():
    samples = [
        (TensorDict({"x": torch.randn(10)}), {}),
        (TensorDict({"y": torch.randn(10)}), {}),  # Different key!
    ]
    collator = dp.DefaultCollator()

    with pytest.raises(
        RuntimeError, match="sets of keys in the tensordicts to stack are exclusive"
    ):
        collator(samples)


def test_default_collate_mismatched_shapes_raises():
    samples = [
        (TensorDict({"x": torch.randn(10, 3)}), {}),
        (TensorDict({"x": torch.randn(20, 3)}), {}),  # Different shape!
    ]
    collator = dp.DefaultCollator()

    with pytest.raises(
        RuntimeError, match="shapes of the tensors to stack is incompatible"
    ):
        collator(samples)


def test_default_collate_specific_keys(batch_of_samples):
    collator = dp.DefaultCollator(keys=["x"], collate_metadata=False)
    batched_data = collator(batch_of_samples)

    assert "x" in batched_data
    assert "y" not in batched_data


def test_default_collate_different_stack_dim():
    samples = [
        (TensorDict({"x": torch.randn(3, 10)}), {}),
        (TensorDict({"x": torch.randn(3, 10)}), {}),
    ]
    collator = dp.DefaultCollator(stack_dim=1, collate_metadata=False)
    batched_data = collator(samples)

    # Stack along dim 1: (3, 10) -> (3, 2, 10)
    assert batched_data["x"].shape == (3, 2, 10)


def test_default_collate_disable_metadata():
    samples = [
        (TensorDict({"x": torch.randn(10)}), {"idx": 0}),
        (TensorDict({"x": torch.randn(10)}), {"idx": 1}),
    ]
    collator = dp.DefaultCollator(collate_metadata=False)
    _ = collator(samples)


# ============================================================================
# ConcatCollator (concat-based)
# ============================================================================


def test_concat_collate_ragged(ragged_samples):
    collator = dp.ConcatCollator(dim=0, add_batch_idx=True)
    batched_data = collator(ragged_samples)

    # 100 + 150 + 80 = 330 points
    assert batched_data["points"].shape == (330, 3)
    assert batched_data["batch_idx"].shape == (330,)


def test_concat_batch_idx_values(ragged_samples):
    collator = dp.ConcatCollator(dim=0, add_batch_idx=True)
    batched_data = collator(ragged_samples)

    # First 100 should be 0, next 150 should be 1, last 80 should be 2
    assert (batched_data["batch_idx"][:100] == 0).all()
    assert (batched_data["batch_idx"][100:250] == 1).all()
    assert (batched_data["batch_idx"][250:] == 2).all()


def test_concat_collate_no_batch_idx(ragged_samples):
    collator = dp.ConcatCollator(dim=0, add_batch_idx=False)
    batched_data = collator(ragged_samples)

    assert "batch_idx" not in batched_data


def test_concat_collate_custom_batch_idx_key(ragged_samples):
    collator = dp.ConcatCollator(
        dim=0,
        add_batch_idx=True,
        batch_idx_key="sample_id",
    )
    batched_data = collator(ragged_samples)

    assert "sample_id" in batched_data
    assert "batch_idx" not in batched_data


def test_concat_collate_metadata(ragged_samples):
    collator = dp.ConcatCollator(dim=0, collate_metadata=True)
    batched_data, metadata_list = collator(ragged_samples)

    assert len(metadata_list) == 3
    assert [m["idx"] for m in metadata_list] == [0, 1, 2]


def test_concat_collate_empty_raises():
    collator = dp.ConcatCollator()

    with pytest.raises(ValueError, match="empty sequence"):
        collator([])


# ============================================================================
# FunctionCollator
# ============================================================================


def test_function_collator():
    def my_collate(samples):
        # Just sum all tensors
        data_list = [data for data, _ in samples]
        total = sum(d["x"].sum() for d in data_list)
        return TensorDict({"total": total.unsqueeze(0)})

    samples = [
        (TensorDict({"x": torch.ones(10)}), {}),
        (TensorDict({"x": torch.ones(10) * 2}), {}),
    ]

    collator = dp.FunctionCollator(my_collate)
    batched_data = collator(samples)
    print(type(batched_data))

    # 10*1 + 10*2 = 30
    assert batched_data["total"].item() == 30.0


# ============================================================================
# Collation convenience functions
# ============================================================================


def test_default_collate_function(batch_of_samples):
    batched_data = dp.default_collate(batch_of_samples)

    assert batched_data["x"].shape == (4, 10, 3)


def test_concat_collate_function(ragged_samples):
    batched_data = dp.concat_collate(ragged_samples, dim=0, add_batch_idx=True)

    assert batched_data["points"].shape == (330, 3)
    assert "batch_idx" in batched_data


def test_get_collator_none():
    collator = dp.get_collator(None)
    assert isinstance(collator, dp.DefaultCollator)


def test_get_collator_instance():
    original = dp.ConcatCollator()
    collator = dp.get_collator(original)
    assert collator is original


def test_get_collator_function():
    def my_fn(samples):
        return samples[0]

    collator = dp.get_collator(my_fn)
    assert isinstance(collator, dp.FunctionCollator)


def test_get_collator_invalid():
    with pytest.raises(TypeError):
        dp.get_collator("not a collator")


# ============================================================================
# Collation with device
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_collate_cuda_samples():
    samples = [
        (TensorDict({"x": torch.randn(10, device="cuda")}), {}),
        (TensorDict({"x": torch.randn(10, device="cuda")}), {}),
    ]

    batched_data, metadata_list = dp.default_collate(samples)
    assert batched_data["x"].device.type == "cuda"
