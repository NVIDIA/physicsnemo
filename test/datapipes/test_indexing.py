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

"""Tests for shared datapipe indexing helpers."""

import pytest
import torch

from physicsnemo.datapipes._indexing import _cyclic_block_indices


class TestCyclicBlockIndices:
    def test_no_wrap(self):
        indices = _cyclic_block_indices(10, 4, start=3)
        assert indices.tolist() == [3, 4, 5, 6]

    def test_wraps_past_end(self):
        indices = _cyclic_block_indices(10, 4, start=8)
        assert indices.tolist() == [8, 9, 0, 1]

    def test_full_range_keeps_natural_order(self):
        indices = _cyclic_block_indices(
            5,
            5,
            generator=torch.Generator().manual_seed(0),
        )
        assert indices.tolist() == [0, 1, 2, 3, 4]

    def test_random_start_is_deterministic(self):
        generator_a = torch.Generator().manual_seed(42)
        generator_b = torch.Generator().manual_seed(42)
        for _ in range(20):
            indices_a = _cyclic_block_indices(64, 8, generator=generator_a)
            indices_b = _cyclic_block_indices(64, 8, generator=generator_b)
            assert torch.equal(indices_a, indices_b)

    def test_random_start_avoids_scalar_readback(self, monkeypatch):
        def fail_item(_):
            raise AssertionError("Tensor.item() must not be called")

        with monkeypatch.context() as context:
            context.setattr(torch.Tensor, "item", fail_item)
            indices = _cyclic_block_indices(
                10,
                4,
                generator=torch.Generator().manual_seed(2),
            )

        assert indices.tolist() == [8, 9, 0, 1]

    def test_inclusion_probability_is_exactly_uniform(self):
        # Over all N starts, every element appears in exactly k blocks, so
        # each inclusion probability is exactly k/N.
        total, k = 11, 4
        counts = torch.zeros(total, dtype=torch.long)
        for start in range(total):
            counts[_cyclic_block_indices(total, k, start=start)] += 1
        assert (counts == k).all()

    @pytest.mark.parametrize(
        ("total", "k"),
        [(-1, 0), (5, -1), (5, 6)],
    )
    def test_invalid_sizes_raise(self, total, k):
        with pytest.raises(ValueError):
            _cyclic_block_indices(total, k)

    def test_start_and_generator_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _cyclic_block_indices(
                10,
                4,
                start=3,
                generator=torch.Generator().manual_seed(0),
            )
