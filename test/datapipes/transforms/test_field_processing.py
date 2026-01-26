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

"""Tests for field processing transforms (ConcatFields, NormalizeVectors, BroadcastGlobalFeatures)."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.datapipes.transforms.concat_fields import (
    ConcatFields,
    NormalizeVectors,
)
from physicsnemo.datapipes.transforms.field_processing import (
    BroadcastGlobalFeatures,
)

# ============================================================================
# ConcatFields Tests
# ============================================================================


class TestConcatFields:
    """Tests for ConcatFields transform."""

    def test_basic_concatenation(self):
        """Test basic field concatenation along last dimension."""
        transform = ConcatFields(
            input_keys=["a", "b"],
            output_key="concat",
        )

        a = torch.randn(100, 3)
        b = torch.randn(100, 5)
        data = TensorDict({"a": a, "b": b})

        result = transform(data)

        assert "concat" in result
        assert result["concat"].shape == (100, 8)

        # Check values match
        torch.testing.assert_close(result["concat"][:, :3], a)
        torch.testing.assert_close(result["concat"][:, 3:], b)

    def test_multiple_fields(self):
        """Test concatenating more than two fields."""
        transform = ConcatFields(
            input_keys=["positions", "sdf", "normals", "features"],
            output_key="embeddings",
        )

        data = TensorDict(
            {
                "positions": torch.randn(50, 3),
                "sdf": torch.randn(50, 1),
                "normals": torch.randn(50, 3),
                "features": torch.randn(50, 8),
            }
        )

        result = transform(data)

        assert result["embeddings"].shape == (50, 15)  # 3 + 1 + 3 + 8

    def test_concat_along_different_dim(self):
        """Test concatenation along a different dimension."""
        transform = ConcatFields(
            input_keys=["a", "b"],
            output_key="concat",
            dim=0,
        )

        a = torch.randn(10, 5)
        b = torch.randn(20, 5)
        data = TensorDict({"a": a, "b": b})

        result = transform(data)

        assert result["concat"].shape == (30, 5)

    def test_preserves_other_fields(self):
        """Test that other fields are preserved."""
        transform = ConcatFields(
            input_keys=["a", "b"],
            output_key="concat",
        )

        a = torch.randn(10, 3)
        b = torch.randn(10, 5)
        c = torch.randn(20, 7)
        data = TensorDict({"a": a, "b": b, "other": c})

        result = transform(data)

        assert "other" in result
        torch.testing.assert_close(result["other"], c)

    def test_missing_key_raises(self):
        """Test that missing key raises KeyError."""
        transform = ConcatFields(
            input_keys=["a", "b", "c"],
            output_key="concat",
        )

        data = TensorDict({"a": torch.randn(10, 3), "b": torch.randn(10, 5)})

        with pytest.raises(KeyError, match="Input key 'c' not found"):
            transform(data)

    def test_skip_missing_flag(self):
        """Test skip_missing option."""
        transform = ConcatFields(
            input_keys=["a", "b", "c"],
            output_key="concat",
            skip_missing=True,
        )

        a = torch.randn(10, 3)
        b = torch.randn(10, 5)
        data = TensorDict({"a": a, "b": b})

        result = transform(data)

        # Should only concat a and b
        assert result["concat"].shape == (10, 8)

    def test_skip_missing_all_missing_raises(self):
        """Test that skip_missing raises when all fields are missing."""
        transform = ConcatFields(
            input_keys=["x", "y", "z"],
            output_key="concat",
            skip_missing=True,
        )

        data = TensorDict({"a": torch.randn(10, 3)})

        with pytest.raises(ValueError, match="No tensors found to concatenate"):
            transform(data)

    def test_single_field(self):
        """Test concatenating a single field (identity-like)."""
        transform = ConcatFields(
            input_keys=["a"],
            output_key="concat",
        )

        a = torch.randn(10, 3)
        data = TensorDict({"a": a})

        result = transform(data)

        assert result["concat"].shape == (10, 3)
        torch.testing.assert_close(result["concat"], a)

    def test_incompatible_shapes_raises(self):
        """Test that incompatible shapes raise RuntimeError."""
        transform = ConcatFields(
            input_keys=["a", "b"],
            output_key="concat",
            dim=-1,
        )

        # Different first dimensions - can't concat on last dim
        a = torch.randn(10, 3)
        b = torch.randn(20, 5)
        data = TensorDict({"a": a, "b": b})

        with pytest.raises(RuntimeError):
            transform(data)

    def test_extra_repr(self):
        """Test extra_repr method."""
        transform = ConcatFields(
            input_keys=["a", "b"],
            output_key="concat",
            dim=1,
        )

        repr_str = transform.extra_repr()
        assert "a" in repr_str
        assert "b" in repr_str
        assert "concat" in repr_str
        assert "dim=1" in repr_str


# ============================================================================
# NormalizeVectors Tests
# ============================================================================


class TestNormalizeVectors:
    """Tests for NormalizeVectors transform."""

    def test_basic_normalization(self):
        """Test basic vector normalization."""
        transform = NormalizeVectors(input_keys=["normals"])

        normals = torch.randn(100, 3) * 10  # Scale up to ensure non-unit
        data = TensorDict({"normals": normals})

        result = transform(data)

        # Check all vectors are unit length
        norms = torch.norm(result["normals"], dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    def test_multiple_fields(self):
        """Test normalizing multiple fields."""
        transform = NormalizeVectors(input_keys=["normals", "directions"])

        data = TensorDict(
            {
                "normals": torch.randn(50, 3) * 5,
                "directions": torch.randn(50, 3) * 3,
            }
        )

        result = transform(data)

        # Check both fields are normalized
        norm_lengths = torch.norm(result["normals"], dim=-1)
        dir_lengths = torch.norm(result["directions"], dim=-1)

        torch.testing.assert_close(
            norm_lengths, torch.ones_like(norm_lengths), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            dir_lengths, torch.ones_like(dir_lengths), atol=1e-5, rtol=1e-5
        )

    def test_preserves_direction(self):
        """Test that normalization preserves vector direction."""
        transform = NormalizeVectors(input_keys=["v"])

        v = torch.tensor([[3.0, 4.0, 0.0]])  # Length 5
        data = TensorDict({"v": v})

        result = transform(data)

        expected = torch.tensor([[0.6, 0.8, 0.0]])  # Normalized
        torch.testing.assert_close(result["v"], expected, atol=1e-5, rtol=1e-5)

    def test_handles_near_zero_vectors(self):
        """Test handling of near-zero length vectors."""
        transform = NormalizeVectors(input_keys=["v"], eps=1e-6)

        v = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],  # Zero vector
                [0.0, 1.0, 0.0],
            ]
        )
        data = TensorDict({"v": v})

        # Should not raise or produce NaN
        result = transform(data)
        assert not torch.isnan(result["v"]).any()

    def test_normalize_along_different_dim(self):
        """Test normalization along a different dimension."""
        transform = NormalizeVectors(input_keys=["v"], dim=0)

        v = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        data = TensorDict({"v": v})

        result = transform(data)

        # Normalize along dim 0
        expected_norms = torch.norm(result["v"], dim=0)
        torch.testing.assert_close(
            expected_norms, torch.ones_like(expected_norms), atol=1e-5, rtol=1e-5
        )

    def test_missing_key_raises(self):
        """Test that missing key raises KeyError."""
        transform = NormalizeVectors(input_keys=["normals"])

        data = TensorDict({"other": torch.randn(10, 3)})

        with pytest.raises(KeyError, match="Input key 'normals' not found"):
            transform(data)

    def test_preserves_other_fields(self):
        """Test that other fields are preserved."""
        transform = NormalizeVectors(input_keys=["normals"])

        normals = torch.randn(10, 3)
        positions = torch.randn(10, 3)
        data = TensorDict({"normals": normals, "positions": positions})

        result = transform(data)

        torch.testing.assert_close(result["positions"], positions)

    def test_extra_repr(self):
        """Test extra_repr method."""
        transform = NormalizeVectors(input_keys=["a", "b"], dim=-1, eps=1e-8)

        repr_str = transform.extra_repr()
        assert "a" in repr_str
        assert "b" in repr_str
        assert "dim=-1" in repr_str


# ============================================================================
# BroadcastGlobalFeatures Tests
# ============================================================================


class TestBroadcastGlobalFeatures:
    """Tests for BroadcastGlobalFeatures transform."""

    def test_basic_broadcast(self):
        """Test basic broadcasting of global features."""
        transform = BroadcastGlobalFeatures(
            input_keys=["density", "velocity"],
            n_points_key="positions",
            output_key="fx",
        )

        data = TensorDict(
            {
                "density": torch.tensor(1.225),
                "velocity": torch.tensor(30.0),
                "positions": torch.randn(1000, 3),
            }
        )

        result = transform(data)

        assert "fx" in result
        assert result["fx"].shape == (1000, 2)

        # All rows should be the same
        assert (result["fx"][:, 0] == 1.225).all()
        assert (result["fx"][:, 1] == 30.0).all()

    def test_broadcast_multiple_1d_features(self):
        """Test broadcasting multiple 1D tensor features."""
        transform = BroadcastGlobalFeatures(
            input_keys=["vx", "vy", "vz"],
            n_points_key="positions",
            output_key="fx",
        )

        data = TensorDict(
            {
                "vx": torch.tensor([1.0]),
                "vy": torch.tensor([2.0]),
                "vz": torch.tensor([3.0]),
                "positions": torch.randn(500, 3),
            }
        )

        result = transform(data)

        assert result["fx"].shape == (500, 3)
        torch.testing.assert_close(result["fx"][0], torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(result["fx"][-1], torch.tensor([1.0, 2.0, 3.0]))

    def test_multiple_scalar_features(self):
        """Test broadcasting multiple scalar features."""
        transform = BroadcastGlobalFeatures(
            input_keys=["a", "b", "c", "d"],
            n_points_key="points",
            output_key="global_feats",
        )

        data = TensorDict(
            {
                "a": torch.tensor(1.0),
                "b": torch.tensor(2.0),
                "c": torch.tensor(3.0),
                "d": torch.tensor(4.0),
                "points": torch.randn(200, 3),
            }
        )

        result = transform(data)

        assert result["global_feats"].shape == (200, 4)

    def test_missing_reference_key_raises(self):
        """Test that missing reference key raises KeyError."""
        transform = BroadcastGlobalFeatures(
            input_keys=["density"],
            n_points_key="positions",
            output_key="fx",
        )

        data = TensorDict({"density": torch.tensor(1.0)})

        with pytest.raises(KeyError, match="Reference key 'positions' not found"):
            transform(data)

    def test_missing_feature_key_raises(self):
        """Test that missing feature key raises KeyError."""
        transform = BroadcastGlobalFeatures(
            input_keys=["density", "velocity"],
            n_points_key="positions",
            output_key="fx",
        )

        data = TensorDict(
            {
                "density": torch.tensor(1.0),
                "positions": torch.randn(100, 3),
            }
        )

        with pytest.raises(KeyError, match="Feature key 'velocity' not found"):
            transform(data)

    def test_preserves_other_fields(self):
        """Test that other fields are preserved."""
        transform = BroadcastGlobalFeatures(
            input_keys=["density"],
            n_points_key="positions",
            output_key="fx",
        )

        positions = torch.randn(100, 3)
        normals = torch.randn(100, 3)
        data = TensorDict(
            {
                "density": torch.tensor(1.0),
                "positions": positions,
                "normals": normals,
            }
        )

        result = transform(data)

        torch.testing.assert_close(result["positions"], positions)
        torch.testing.assert_close(result["normals"], normals)

    def test_repr(self):
        """Test string representation."""
        transform = BroadcastGlobalFeatures(
            input_keys=["a", "b"],
            n_points_key="points",
            output_key="fx",
        )

        repr_str = repr(transform)
        assert "BroadcastGlobalFeatures" in repr_str
        assert "fx" in repr_str


# ============================================================================
# Integration Tests
# ============================================================================


class TestFieldProcessingIntegration:
    """Integration tests combining field processing transforms."""

    def test_concat_then_normalize(self):
        """Test concatenating fields then normalizing vectors."""
        concat_transform = ConcatFields(
            input_keys=["a", "b"],
            output_key="combined",
        )

        # Assume combined vectors should be normalized
        normalize_transform = NormalizeVectors(
            input_keys=["combined"],
            dim=-1,
        )

        data = TensorDict(
            {
                "a": torch.randn(50, 2),
                "b": torch.randn(50, 2),
            }
        )

        data = concat_transform(data)
        data = normalize_transform(data)

        # Combined should be unit vectors
        norms = torch.norm(data["combined"], dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    def test_broadcast_then_concat(self):
        """Test broadcasting global features then concatenating with local."""
        broadcast_transform = BroadcastGlobalFeatures(
            input_keys=["density"],
            n_points_key="positions",
            output_key="global_fx",
        )

        concat_transform = ConcatFields(
            input_keys=["positions", "global_fx"],
            output_key="embeddings",
        )

        data = TensorDict(
            {
                "density": torch.tensor(1.225),
                "positions": torch.randn(100, 3),
            }
        )

        data = broadcast_transform(data)
        data = concat_transform(data)

        # Embeddings should be positions (3) + global (1) = 4
        assert data["embeddings"].shape == (100, 4)

    def test_full_feature_engineering_pipeline(self):
        """Test a realistic feature engineering pipeline."""
        from physicsnemo.datapipes.transforms.compose import Compose

        pipeline = Compose(
            [
                # Normalize surface normals
                NormalizeVectors(input_keys=["normals"]),
                # Broadcast global parameters
                BroadcastGlobalFeatures(
                    input_keys=["reynolds_number", "mach_number"],
                    n_points_key="positions",
                    output_key="global_params",
                ),
                # Concatenate all features
                ConcatFields(
                    input_keys=["positions", "normals", "sdf", "global_params"],
                    output_key="node_features",
                ),
            ]
        )

        data = TensorDict(
            {
                "positions": torch.randn(500, 3),
                "normals": torch.randn(500, 3) * 10,  # Not unit length
                "sdf": torch.randn(500, 1),
                "reynolds_number": torch.tensor(1e6),
                "mach_number": torch.tensor(0.3),
            }
        )

        result = pipeline(data)

        # Final features: positions(3) + normals(3) + sdf(1) + global(2) = 9
        assert result["node_features"].shape == (500, 9)

        # Normals should be normalized
        norms = torch.norm(result["normals"], dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)
