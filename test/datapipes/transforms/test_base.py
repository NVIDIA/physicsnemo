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

"""Tests for Transform base class."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.datapipes.transforms.base import Transform

# ============================================================================
# Concrete Transform Implementation for Testing
# ============================================================================


class SimpleScaleTransform(Transform):
    """A simple transform that scales all tensors by a factor."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.scale_tensor = torch.tensor(scale)

    def __call__(self, data: TensorDict) -> TensorDict:
        return data.apply(lambda x: x * self.scale)

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


class TransformWithState(Transform):
    """A transform with state that can be saved/loaded."""

    def __init__(self, offset: torch.Tensor):
        super().__init__()
        self.offset = offset

    def __call__(self, data: TensorDict) -> TensorDict:
        return data.apply(lambda x: x + self.offset)

    def state_dict(self) -> dict:
        return {"offset": self.offset.cpu()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.offset = state_dict["offset"].clone()


# ============================================================================
# Transform Base Class Tests
# ============================================================================


class TestTransformBase:
    """Tests for Transform base class functionality."""

    def test_abstract_call_not_implemented(self):
        """Test that Transform cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            Transform()

    def test_concrete_transform_instantiation(self):
        """Test that concrete transforms can be instantiated."""
        transform = SimpleScaleTransform(scale=2.0)
        assert transform.scale == 2.0

    def test_call_method(self):
        """Test that __call__ works on concrete transform."""
        transform = SimpleScaleTransform(scale=3.0)

        data = TensorDict({"x": torch.tensor([1.0, 2.0, 3.0])})
        result = transform(data)

        expected = torch.tensor([3.0, 6.0, 9.0])
        torch.testing.assert_close(result["x"], expected)


class TestTransformDevice:
    """Tests for Transform device handling."""

    def test_device_initially_none(self):
        """Test that device is None by default."""
        transform = SimpleScaleTransform(scale=1.0)
        assert transform.device is None

    def test_to_device_string(self):
        """Test moving transform to device with string."""
        transform = SimpleScaleTransform(scale=2.0)

        result = transform.to("cpu")

        assert result is transform  # Returns self for chaining
        assert transform.device == torch.device("cpu")

    def test_to_device_torch_device(self):
        """Test moving transform to device with torch.device."""
        transform = SimpleScaleTransform(scale=2.0)

        device = torch.device("cpu")
        transform.to(device)

        assert transform.device == device

    def test_to_moves_tensor_attributes(self):
        """Test that .to() moves tensor attributes."""
        transform = SimpleScaleTransform(scale=2.0)

        transform.to("cpu")

        assert transform.scale_tensor.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_cuda(self):
        """Test moving transform to CUDA."""
        transform = SimpleScaleTransform(scale=2.0)

        transform.to("cuda:0")

        assert transform.device == torch.device("cuda:0")
        assert transform.scale_tensor.device == torch.device("cuda:0")

    def test_chaining_to_calls(self):
        """Test that .to() calls can be chained."""
        transform = SimpleScaleTransform(scale=2.0)

        # Chain calls
        result = transform.to("cpu")

        assert result is transform


class TestTransformRepr:
    """Tests for Transform string representation."""

    def test_default_repr(self):
        """Test default repr without extra_repr."""

        class MinimalTransform(Transform):
            def __call__(self, data: TensorDict) -> TensorDict:
                return data

        transform = MinimalTransform()
        repr_str = repr(transform)

        assert "MinimalTransform" in repr_str
        assert "()" in repr_str

    def test_repr_with_extra_repr(self):
        """Test repr with custom extra_repr."""
        transform = SimpleScaleTransform(scale=2.5)

        repr_str = repr(transform)

        assert "SimpleScaleTransform" in repr_str
        assert "scale=2.5" in repr_str

    def test_extra_repr_default_empty(self):
        """Test that default extra_repr returns empty string."""

        class MinimalTransform(Transform):
            def __call__(self, data: TensorDict) -> TensorDict:
                return data

        transform = MinimalTransform()
        assert transform.extra_repr() == ""


class TestTransformStateDict:
    """Tests for Transform state serialization."""

    def test_default_state_dict_empty(self):
        """Test that default state_dict returns empty dict."""
        transform = SimpleScaleTransform(scale=2.0)

        state = transform.state_dict()

        assert state == {}

    def test_default_load_state_dict_no_op(self):
        """Test that default load_state_dict is a no-op."""
        transform = SimpleScaleTransform(scale=2.0)

        # Should not raise
        transform.load_state_dict({"some_key": "some_value"})

        # Scale should be unchanged
        assert transform.scale == 2.0

    def test_custom_state_dict(self):
        """Test custom state_dict implementation."""
        offset = torch.tensor([1.0, 2.0, 3.0])
        transform = TransformWithState(offset=offset)

        state = transform.state_dict()

        assert "offset" in state
        torch.testing.assert_close(state["offset"], offset)

    def test_custom_load_state_dict(self):
        """Test custom load_state_dict implementation."""
        transform = TransformWithState(offset=torch.zeros(3))

        new_offset = torch.tensor([5.0, 6.0, 7.0])
        transform.load_state_dict({"offset": new_offset})

        torch.testing.assert_close(transform.offset, new_offset)

    def test_state_dict_round_trip(self):
        """Test saving and loading state."""
        original = TransformWithState(offset=torch.tensor([1.0, 2.0, 3.0]))

        state = original.state_dict()

        restored = TransformWithState(offset=torch.zeros(3))
        restored.load_state_dict(state)

        torch.testing.assert_close(original.offset, restored.offset)


class TestTransformIntegration:
    """Integration tests for Transform class."""

    def test_transform_with_tensordict(self):
        """Test transform applied to TensorDict."""
        transform = SimpleScaleTransform(scale=2.0)

        data = TensorDict(
            {
                "positions": torch.randn(100, 3),
                "features": torch.randn(100, 8),
            }
        )

        original_positions = data["positions"].clone()
        original_features = data["features"].clone()

        result = transform(data)

        torch.testing.assert_close(result["positions"], original_positions * 2.0)
        torch.testing.assert_close(result["features"], original_features * 2.0)

    def test_transform_preserves_device(self):
        """Test that transform preserves tensor devices."""
        transform = SimpleScaleTransform(scale=2.0)

        data = TensorDict({"x": torch.tensor([1.0, 2.0, 3.0], device="cpu")})

        result = transform(data)

        assert result["x"].device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_transform_on_cuda_data(self):
        """Test transform with CUDA tensors."""
        transform = SimpleScaleTransform(scale=2.0)
        transform.to("cuda:0")

        data = TensorDict({"x": torch.tensor([1.0, 2.0, 3.0], device="cuda:0")})

        result = transform(data)

        assert result["x"].device == torch.device("cuda:0")
        expected = torch.tensor([2.0, 4.0, 6.0], device="cuda:0")
        torch.testing.assert_close(result["x"], expected)

    def test_multiple_transforms_sequential(self):
        """Test applying multiple transforms sequentially."""
        scale_transform = SimpleScaleTransform(scale=2.0)
        offset_transform = TransformWithState(offset=torch.tensor(10.0))

        data = TensorDict({"x": torch.tensor([1.0, 2.0, 3.0])})

        # Apply sequentially
        data = scale_transform(data)
        data = offset_transform(data)

        expected = torch.tensor([12.0, 14.0, 16.0])
        torch.testing.assert_close(data["x"], expected)
