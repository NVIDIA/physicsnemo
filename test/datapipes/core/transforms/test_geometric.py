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

"""Tests for geometric transforms."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.datapipes.core.transforms.geometric import ReScale, Scale, Translate


class TestTranslate:
    """Tests for Translate transform."""

    def test_translate_add_mode_default(self):
        """Test that add mode is the default (subtract=False)."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value=torch.tensor([1.0, 2.0, 3.0]),
        )
        assert transform.subtract is False

    def test_translate_add_mode_with_tensor(self):
        """Test add mode with a fixed tensor value."""
        offset = torch.tensor([1.0, 2.0, 3.0])
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value=offset,
            subtract=False,
        )

        data = TensorDict(
            {"positions": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        assert torch.allclose(result["positions"], expected)

    def test_translate_subtract_mode_with_tensor(self):
        """Test subtract mode with a fixed tensor value."""
        offset = torch.tensor([1.0, 2.0, 3.0])
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value=offset,
            subtract=True,
        )

        data = TensorDict(
            {"positions": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[-1.0, -2.0, -3.0], [0.0, -1.0, -2.0]])
        assert torch.allclose(result["positions"], expected)

    def test_translate_add_mode_with_key(self):
        """Test add mode with a key reference."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value="offset",
            subtract=False,
        )

        data = TensorDict(
            {
                "positions": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
                "offset": torch.tensor([5.0, 10.0, 15.0]),
            },
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[5.0, 10.0, 15.0], [6.0, 11.0, 16.0]])
        assert torch.allclose(result["positions"], expected)

    def test_translate_subtract_mode_with_key(self):
        """Test subtract mode with a key reference (centering use case)."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value="center_of_mass",
            subtract=True,
        )

        data = TensorDict(
            {
                "positions": torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
                "center_of_mass": torch.tensor([1.0, 1.0, 1.0]),
            },
            batch_size=[],
        )

        result = transform(data)

        # Points should be centered: original - center_of_mass
        expected = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        assert torch.allclose(result["positions"], expected)

    def test_translate_multiple_keys(self):
        """Test translation applied to multiple keys."""
        offset = torch.tensor([1.0, 1.0, 1.0])
        transform = Translate(
            input_keys=["positions", "surface_points"],
            center_key_or_value=offset,
            subtract=False,
        )

        data = TensorDict(
            {
                "positions": torch.tensor([[0.0, 0.0, 0.0]]),
                "surface_points": torch.tensor([[5.0, 5.0, 5.0]]),
            },
            batch_size=[],
        )

        result = transform(data)

        assert torch.allclose(result["positions"], torch.tensor([[1.0, 1.0, 1.0]]))
        assert torch.allclose(result["surface_points"], torch.tensor([[6.0, 6.0, 6.0]]))

    def test_translate_preserves_other_fields(self):
        """Test that translation preserves fields not in input_keys."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value=torch.tensor([1.0, 1.0, 1.0]),
        )

        data = TensorDict(
            {
                "positions": torch.tensor([[0.0, 0.0, 0.0]]),
                "velocities": torch.tensor([[1.0, 2.0, 3.0]]),
            },
            batch_size=[],
        )

        result = transform(data)

        # Velocities should be unchanged
        assert torch.allclose(result["velocities"], torch.tensor([[1.0, 2.0, 3.0]]))

    def test_translate_missing_key_raises(self):
        """Test that missing center key raises KeyError."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value="nonexistent_key",
        )

        data = TensorDict(
            {"positions": torch.tensor([[0.0, 0.0, 0.0]])},
            batch_size=[],
        )

        with pytest.raises(KeyError, match="nonexistent_key"):
            transform(data)

    def test_translate_1d_center_broadcasted(self):
        """Test that 1D center tensor is properly broadcasted."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value=torch.tensor([1.0, 2.0, 3.0]),  # 1D tensor
            subtract=False,
        )

        data = TensorDict(
            {"positions": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        assert torch.allclose(result["positions"], expected)

    def test_translate_2d_center(self):
        """Test that 2D center tensor works correctly."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value=torch.tensor([[1.0, 2.0, 3.0]]),  # 2D tensor (1, 3)
            subtract=False,
        )

        data = TensorDict(
            {"positions": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        assert torch.allclose(result["positions"], expected)

    def test_translate_repr_add_mode(self):
        """Test repr shows add mode."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value="offset",
            subtract=False,
        )

        repr_str = repr(transform)
        assert "Translate" in repr_str
        assert "mode=add" in repr_str
        assert "positions" in repr_str

    def test_translate_repr_subtract_mode(self):
        """Test repr shows subtract mode."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value="center_of_mass",
            subtract=True,
        )

        repr_str = repr(transform)
        assert "Translate" in repr_str
        assert "mode=subtract" in repr_str
        assert "center_of_mass" in repr_str

    def test_translate_skips_missing_input_keys(self):
        """Test that missing input keys are silently skipped."""
        transform = Translate(
            input_keys=["positions", "nonexistent"],
            center_key_or_value=torch.tensor([1.0, 1.0, 1.0]),
        )

        data = TensorDict(
            {"positions": torch.tensor([[0.0, 0.0, 0.0]])},
            batch_size=[],
        )

        # Should not raise, just skip the missing key
        result = transform(data)
        assert torch.allclose(result["positions"], torch.tensor([[1.0, 1.0, 1.0]]))

    def test_translate_add_then_subtract_roundtrip(self):
        """Test that add followed by subtract returns to original."""
        offset = torch.tensor([5.0, 10.0, 15.0])
        add_transform = Translate(
            input_keys=["positions"],
            center_key_or_value=offset,
            subtract=False,
        )
        subtract_transform = Translate(
            input_keys=["positions"],
            center_key_or_value=offset,
            subtract=True,
        )

        original = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        data = TensorDict({"positions": original.clone()}, batch_size=[])

        # Add then subtract should return to original
        result = add_transform(data)
        result = subtract_transform(result)

        assert torch.allclose(result["positions"], original)


class TestScale:
    """Tests for Scale transform."""

    def test_scale_multiply_mode_default(self):
        """Test that multiply mode is the default (divide=False)."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([2.0, 2.0, 2.0]),
        )
        assert transform.divide is False

    def test_scale_multiply_mode_with_tensor(self):
        """Test multiply mode with a fixed tensor value."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([2.0, 2.0, 2.0]),
            divide=False,
        )

        data = TensorDict(
            {"positions": torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
        assert torch.allclose(result["positions"], expected)

    def test_scale_divide_mode_with_tensor(self):
        """Test divide mode with a fixed tensor value."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([2.0, 2.0, 2.0]),
            divide=True,
        )

        data = TensorDict(
            {"positions": torch.tensor([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        assert torch.allclose(result["positions"], expected)

    def test_scale_multiple_keys(self):
        """Test scaling multiple keys."""
        transform = Scale(
            input_keys=["positions", "velocities"],
            scale=torch.tensor([2.0, 2.0, 2.0]),
            divide=False,
        )

        data = TensorDict(
            {
                "positions": torch.tensor([[1.0, 2.0, 3.0]]),
                "velocities": torch.tensor([[2.0, 4.0, 6.0]]),
            },
            batch_size=[],
        )

        result = transform(data)

        assert torch.allclose(result["positions"], torch.tensor([[2.0, 4.0, 6.0]]))
        assert torch.allclose(result["velocities"], torch.tensor([[4.0, 8.0, 12.0]]))

    def test_scale_preserves_other_fields(self):
        """Test that scaling preserves fields not in input_keys."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([2.0, 2.0, 2.0]),
        )

        data = TensorDict(
            {
                "positions": torch.tensor([[1.0, 2.0, 3.0]]),
                "labels": torch.tensor([1, 2, 3]),
            },
            batch_size=[],
        )

        result = transform(data)

        assert torch.equal(result["labels"], torch.tensor([1, 2, 3]))

    def test_scale_nonuniform_multiply(self):
        """Test scaling with non-uniform scale factors in multiply mode."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([1.0, 2.0, 4.0]),
            divide=False,
        )

        data = TensorDict(
            {"positions": torch.tensor([[1.0, 1.0, 1.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[1.0, 2.0, 4.0]])
        assert torch.allclose(result["positions"], expected)

    def test_scale_nonuniform_divide(self):
        """Test scaling with non-uniform scale factors in divide mode."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([1.0, 2.0, 4.0]),
            divide=True,
        )

        data = TensorDict(
            {"positions": torch.tensor([[1.0, 2.0, 4.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[1.0, 1.0, 1.0]])
        assert torch.allclose(result["positions"], expected)

    def test_scale_repr_multiply_mode(self):
        """Test repr shows multiply mode."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([1.0, 2.0, 3.0]),
            divide=False,
        )

        repr_str = repr(transform)
        assert "Scale" in repr_str
        assert "mode=multiply" in repr_str
        assert "positions" in repr_str

    def test_scale_repr_divide_mode(self):
        """Test repr shows divide mode."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([1.0, 2.0, 3.0]),
            divide=True,
        )

        repr_str = repr(transform)
        assert "Scale" in repr_str
        assert "mode=divide" in repr_str
        assert "positions" in repr_str

    def test_scale_1d_scale_broadcasted(self):
        """Test that 1D scale tensor is properly broadcasted."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([2.0, 2.0, 2.0]),  # 1D tensor
            divide=False,
        )

        data = TensorDict(
            {"positions": torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
        assert torch.allclose(result["positions"], expected)

    def test_scale_2d_scale(self):
        """Test that 2D scale tensor works correctly."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([[2.0, 2.0, 2.0]]),  # 2D tensor (1, 3)
            divide=False,
        )

        data = TensorDict(
            {"positions": torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
        assert torch.allclose(result["positions"], expected)

    def test_scale_skips_missing_input_keys(self):
        """Test that missing input keys are silently skipped."""
        transform = Scale(
            input_keys=["positions", "nonexistent"],
            scale=torch.tensor([2.0, 2.0, 2.0]),
        )

        data = TensorDict(
            {"positions": torch.tensor([[1.0, 1.0, 1.0]])},
            batch_size=[],
        )

        # Should not raise, just skip the missing key
        result = transform(data)
        assert torch.allclose(result["positions"], torch.tensor([[2.0, 2.0, 2.0]]))

    def test_scale_multiply_then_divide_roundtrip(self):
        """Test that multiply followed by divide returns to original."""
        scale_factor = torch.tensor([2.0, 3.0, 4.0])
        multiply_transform = Scale(
            input_keys=["positions"],
            scale=scale_factor,
            divide=False,
        )
        divide_transform = Scale(
            input_keys=["positions"],
            scale=scale_factor,
            divide=True,
        )

        original = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        data = TensorDict({"positions": original.clone()}, batch_size=[])

        # Multiply then divide should return to original
        result = multiply_transform(data)
        result = divide_transform(result)

        assert torch.allclose(result["positions"], original)


class TestReScaleBackwardsCompatibility:
    """Tests for ReScale backwards compatibility alias."""

    def test_rescale_is_scale(self):
        """Test that ReScale is an alias for Scale."""
        assert ReScale is Scale

    def test_rescale_divide_mode_default_for_backwards_compat(self):
        """Test that ReScale can be used with the new API.

        Note: The old ReScale always divided. With the new Scale class,
        users need to explicitly set divide=True for the same behavior.
        """
        # Old behavior equivalent: dividing by scale
        transform = ReScale(
            input_keys=["positions"],
            scale=torch.tensor([2.0, 2.0, 2.0]),
            divide=True,
        )

        data = TensorDict(
            {"positions": torch.tensor([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        assert torch.allclose(result["positions"], expected)

    def test_rescale_import_from_module(self):
        """Test that ReScale can be imported from the transforms module."""
        from physicsnemo.datapipes.core.transforms import ReScale as ImportedReScale

        assert ImportedReScale is Scale
