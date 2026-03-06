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

"""Tests for utility functions."""

import torch

from physicsnemo.mesh.utilities._padding import _pad_by_tiling_last, _pad_with_value


class TestPadding:
    """Tests for padding utilities used in mesh operations."""

    def test_pad_by_tiling_last_simple(self):
        """Test padding by tiling last element."""
        tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])

        padded = _pad_by_tiling_last(tensor, 5)

        expected = torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [5, 6],  # Tiled last
                [5, 6],  # Tiled last
            ]
        )

        assert torch.equal(padded, expected)
        assert padded.shape == (5, 2)

    def test_pad_by_tiling_last_no_padding_needed(self):
        """Test that tiling doesn't occur when size matches."""
        tensor = torch.tensor([[1, 2], [3, 4]])

        padded = _pad_by_tiling_last(tensor, 2)

        assert torch.equal(padded, tensor)

    def test_pad_by_tiling_last_different_dtypes(self):
        """Test padding with different dtypes."""
        tensor = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float64)

        padded = _pad_by_tiling_last(tensor, 4)

        assert padded.dtype == torch.float64
        assert padded.shape == (4, 2)
        assert torch.equal(padded[-2:], tensor[-1:].expand(2, -1))

    def test_pad_with_value_simple(self):
        """Test padding with constant value."""
        tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)

        padded = _pad_with_value(tensor, 5, value=99)

        expected = torch.tensor(
            [
                [1, 2],
                [3, 4],
                [99, 99],
                [99, 99],
                [99, 99],
            ],
            dtype=torch.long,
        )

        assert torch.equal(padded, expected)
        assert padded.shape == (5, 2)

    def test_pad_with_value_float(self):
        """Test padding with float value."""
        tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        padded = _pad_with_value(tensor, 3, value=-1.5)

        assert padded.shape == (3, 2)
        assert padded[0, 0] == 1.0
        assert padded[1, 0] == -1.5
        assert padded[2, 1] == -1.5

    def test_pad_with_value_no_padding_needed(self):
        """Test that padding doesn't occur when size matches."""
        tensor = torch.tensor([[1, 2], [3, 4]])

        padded = _pad_with_value(tensor, 2, value=0)

        assert torch.equal(padded, tensor)

    def test_pad_with_value_preserves_dtype(self):
        """Test that padding preserves dtype."""
        tensor = torch.tensor([[1, 2]], dtype=torch.int32)

        padded = _pad_with_value(tensor, 3, value=0)

        assert padded.dtype == torch.int32

    def test_pad_with_value_preserves_device(self):
        """Test that padding preserves device."""
        tensor = torch.tensor([[1, 2]], dtype=torch.float32)
        device = tensor.device

        padded = _pad_with_value(tensor, 3, value=0.0)

        assert padded.device == device

    def test_pad_with_value_higher_dim(self):
        """Test padding with higher dimensional tensors."""
        tensor = torch.randn(2, 3, 4)

        padded = _pad_with_value(tensor, 5, value=0.0)

        assert padded.shape == (5, 3, 4)
        assert torch.equal(padded[:2], tensor)
        assert torch.allclose(padded[2:], torch.zeros(3, 3, 4))

    def test_pad_by_tiling_last_single_element(self):
        """Test tiling from single element tensor."""
        tensor = torch.tensor([[42, 43]])

        padded = _pad_by_tiling_last(tensor, 3)

        assert padded.shape == (3, 2)
        assert torch.equal(padded[0], tensor[0])
        assert torch.equal(padded[1], tensor[0])
        assert torch.equal(padded[2], tensor[0])

    def test_pad_with_value_zero_padding(self):
        """Test padding with value 0."""
        tensor = torch.tensor([[1, 2], [3, 4]])

        padded = _pad_with_value(tensor, 4, value=0)

        assert torch.equal(padded[:2], tensor)
        assert torch.equal(padded[2:], torch.zeros(2, 2, dtype=tensor.dtype))
