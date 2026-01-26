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

"""Tests for transforms."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import physicsnemo.datapipes as dp

# ============================================================================
# Normalize transform
# ============================================================================


def test_normalize_single_field():
    sample = TensorDict({"x": torch.tensor([10.0, 20.0, 30.0])})
    norm = dp.Normalize(
        input_keys=["x"],
        method="mean_std",
        means={"x": 20.0},
        stds={"x": 10.0},
    )

    result = norm(sample)
    expected = torch.tensor([-1.0, 0.0, 1.0])
    torch.testing.assert_close(result["x"], expected, atol=1e-6, rtol=1e-6)


def test_normalize_multiple_fields():
    sample = TensorDict(
        {
            "a": torch.tensor([100.0]),
            "b": torch.tensor([50.0]),
        }
    )
    norm = dp.Normalize(
        input_keys=["a", "b"],
        method="mean_std",
        means={"a": 100.0, "b": 0.0},
        stds={"a": 10.0, "b": 50.0},
    )

    result = norm(sample)
    torch.testing.assert_close(result["a"], torch.tensor([0.0]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result["b"], torch.tensor([1.0]), atol=1e-6, rtol=1e-6)


def test_normalize_preserves_other_fields():
    sample = TensorDict(
        {
            "x": torch.tensor([10.0]),
            "y": torch.tensor([999.0]),
        }
    )
    norm = dp.Normalize(
        input_keys=["x"], method="mean_std", means={"x": 0.0}, stds={"x": 1.0}
    )

    result = norm(sample)
    assert "y" in result
    torch.testing.assert_close(result["y"], torch.tensor([999.0]))


def test_normalize_inverse():
    sample = TensorDict({"x": torch.tensor([1.0, 2.0, 3.0])})
    norm = dp.Normalize(
        input_keys=["x"], method="mean_std", means={"x": 10.0}, stds={"x": 2.0}
    )

    normalized = norm(sample)
    denormalized = norm.inverse(normalized)

    torch.testing.assert_close(denormalized["x"], sample["x"], atol=1e-5, rtol=1e-5)


def test_normalize_scalar_mean_std():
    sample = TensorDict(
        {
            "a": torch.tensor([10.0]),
            "b": torch.tensor([20.0]),
        }
    )
    norm = dp.Normalize(input_keys=["a", "b"], method="mean_std", means=0.0, stds=10.0)

    result = norm(sample)
    torch.testing.assert_close(result["a"], torch.tensor([1.0]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result["b"], torch.tensor([2.0]), atol=1e-6, rtol=1e-6)


def test_normalize_missing_field_raises():
    sample = TensorDict({"x": torch.randn(10)})
    norm = dp.Normalize(
        input_keys=["y"], method="mean_std", means={"y": 0.0}, stds={"y": 1.0}
    )

    with pytest.raises(KeyError):
        norm(sample)


def test_normalize_empty_fields_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        dp.Normalize(input_keys=[], method="mean_std", means={}, stds={})


def test_normalize_missing_mean_raises():
    with pytest.raises(ValueError, match="Mean not provided"):
        dp.Normalize(
            input_keys=["x", "y"],
            method="mean_std",
            means={"x": 0.0},
            stds={"x": 1.0, "y": 1.0},
        )


def test_normalize_state_dict():
    norm = dp.Normalize(
        input_keys=["x"],
        method="mean_std",
        means={"x": 5.0},
        stds={"x": 2.0},
    )
    state = norm.state_dict()

    assert state["input_keys"] == ["x"]
    assert state["method"] == "mean_std"
    assert "x" in state["means"]
    assert "x" in state["stds"]


# ============================================================================
# Min-Max Scaling Tests
# ============================================================================


def test_normalize_minmax_single_field():
    """Test min-max normalization normalizes to [-1, 1]."""
    sample = TensorDict({"x": torch.tensor([0.0, 50.0, 100.0])})
    norm = dp.Normalize(
        input_keys=["x"],
        method="min_max",
        mins={"x": 0.0},
        maxs={"x": 100.0},
    )

    result = norm(sample)
    # min=0, max=100 -> center=50, half_range=50
    # Values: (0-50)/50=-1, (50-50)/50=0, (100-50)/50=1
    expected = torch.tensor([-1.0, 0.0, 1.0])
    torch.testing.assert_close(result["x"], expected, atol=1e-6, rtol=1e-6)


def test_normalize_minmax_multiple_fields():
    """Test min-max normalization with multiple fields."""
    sample = TensorDict(
        {
            "pressure": torch.tensor([100000.0]),
            "velocity": torch.tensor([0.0]),
        }
    )
    norm = dp.Normalize(
        input_keys=["pressure", "velocity"],
        method="min_max",
        mins={"pressure": 90000.0, "velocity": -50.0},
        maxs={"pressure": 110000.0, "velocity": 50.0},
    )

    result = norm(sample)
    # pressure: center=100000, half_range=10000 -> (100000-100000)/10000 = 0
    # velocity: center=0, half_range=50 -> (0-0)/50 = 0
    torch.testing.assert_close(
        result["pressure"], torch.tensor([0.0]), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        result["velocity"], torch.tensor([0.0]), atol=1e-6, rtol=1e-6
    )


def test_normalize_minmax_inverse():
    """Test inverse min-max normalization."""
    sample = TensorDict({"x": torch.tensor([25.0, 50.0, 75.0])})
    norm = dp.Normalize(
        input_keys=["x"],
        method="min_max",
        mins={"x": 0.0},
        maxs={"x": 100.0},
    )

    normalized = norm(sample)
    denormalized = norm.inverse(normalized)

    torch.testing.assert_close(denormalized["x"], sample["x"], atol=1e-5, rtol=1e-5)


def test_normalize_minmax_scalar_values():
    """Test min-max with scalar min/max applied to all fields."""
    sample = TensorDict(
        {
            "a": torch.tensor([0.0]),
            "b": torch.tensor([100.0]),
        }
    )
    norm = dp.Normalize(input_keys=["a", "b"], method="min_max", mins=0.0, maxs=100.0)

    result = norm(sample)
    # Both use center=50, half_range=50
    torch.testing.assert_close(result["a"], torch.tensor([-1.0]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result["b"], torch.tensor([1.0]), atol=1e-6, rtol=1e-6)


def test_normalize_minmax_edge_case_same_min_max():
    """Test min-max when min == max (should use eps to avoid division by zero)."""
    sample = TensorDict({"x": torch.tensor([50.0])})
    norm = dp.Normalize(
        input_keys=["x"],
        method="min_max",
        mins={"x": 50.0},
        maxs={"x": 50.0},
        eps=1e-8,
    )

    result = norm(sample)
    # center=50, half_range=0 -> (50-50)/(0+eps) ≈ 0
    torch.testing.assert_close(result["x"], torch.tensor([0.0]), atol=1e-5, rtol=1e-5)


def test_normalize_minmax_state_dict():
    """Test state_dict for min-max normalization."""
    norm = dp.Normalize(
        input_keys=["x"],
        method="min_max",
        mins={"x": 0.0},
        maxs={"x": 100.0},
    )
    state = norm.state_dict()

    assert state["input_keys"] == ["x"]
    assert state["method"] == "min_max"
    assert "x" in state["mins"]
    assert "x" in state["maxs"]
    assert "means" not in state
    assert "stds" not in state


def test_normalize_minmax_load_state_dict():
    """Test loading state_dict for min-max normalization."""
    state = {
        "input_keys": ["x"],
        "method": "min_max",
        "mins": {"x": torch.tensor(0.0)},
        "maxs": {"x": torch.tensor(100.0)},
        "eps": 1e-8,
    }

    norm = dp.Normalize(
        input_keys=["x"], method="min_max", mins={"x": 50.0}, maxs={"x": 150.0}
    )
    norm.load_state_dict(state)

    sample = TensorDict({"x": torch.tensor([50.0])})
    result = norm(sample)
    # Should use loaded mins/maxs: center=50, half_range=50
    expected = torch.tensor([0.0])
    torch.testing.assert_close(result["x"], expected, atol=1e-6, rtol=1e-6)


# ============================================================================
# File Loading Tests
# ============================================================================


def test_normalize_load_from_npz_mean_std():
    """Test loading mean_std normalization from .npz file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test npz file
        npz_path = Path(tmpdir) / "stats.npz"
        stats_data = {
            "pressure": {"mean": np.array(100000.0), "std": np.array(10000.0)},
            "velocity": {"mean": np.array(0.0), "std": np.array(10.0)},
        }
        np.savez(npz_path, **stats_data)

        # Load normalizer
        norm = dp.Normalize(
            input_keys=["pressure", "velocity"],
            method="mean_std",
            stats_file=npz_path,
        )

        sample = TensorDict(
            {
                "pressure": torch.tensor([110000.0]),
                "velocity": torch.tensor([10.0]),
            }
        )
        result = norm(sample)

        # pressure: (110000 - 100000) / 10000 = 1.0
        # velocity: (10 - 0) / 10 = 1.0
        torch.testing.assert_close(
            result["pressure"], torch.tensor([1.0]), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            result["velocity"], torch.tensor([1.0]), atol=1e-6, rtol=1e-6
        )


def test_normalize_load_from_npz_min_max():
    """Test loading min_max normalization from .npz file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test npz file
        npz_path = Path(tmpdir) / "stats.npz"
        stats_data = {
            "x": {"min": np.array(0.0), "max": np.array(100.0)},
            "y": {"min": np.array(-50.0), "max": np.array(50.0)},
        }
        np.savez(npz_path, **stats_data)

        # Load normalizer
        norm = dp.Normalize(
            input_keys=["x", "y"],
            method="min_max",
            stats_file=npz_path,
        )

        sample = TensorDict(
            {
                "x": torch.tensor([50.0]),
                "y": torch.tensor([0.0]),
            }
        )
        result = norm(sample)

        # x: center=50, half_range=50 -> (50-50)/50 = 0
        # y: center=0, half_range=50 -> (0-0)/50 = 0
        torch.testing.assert_close(
            result["x"], torch.tensor([0.0]), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            result["y"], torch.tensor([0.0]), atol=1e-6, rtol=1e-6
        )


def test_normalize_load_file_not_found():
    """Test error handling when stats file doesn't exist."""
    with pytest.raises(FileNotFoundError, match="not found"):
        dp.Normalize(
            input_keys=["x"],
            method="mean_std",
            stats_file="nonexistent_file.npz",
        )


def test_normalize_load_missing_field_in_file():
    """Test error handling when required field is missing in stats file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create npz file with only one field
        npz_path = Path(tmpdir) / "stats.npz"
        stats_data = {
            "x": {"mean": np.array(0.0), "std": np.array(1.0)},
        }
        np.savez(npz_path, **stats_data)

        # Try to load normalizer expecting two fields
        with pytest.raises(ValueError, match="not found in stats file"):
            dp.Normalize(
                input_keys=["x", "y"],
                method="mean_std",
                stats_file=npz_path,
            )


def test_normalize_file_override_with_direct_params():
    """Test that direct parameters override file parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test npz file with some values
        npz_path = Path(tmpdir) / "stats.npz"
        stats_data = {
            "x": {"mean": np.array(100.0), "std": np.array(10.0)},
        }
        np.savez(npz_path, **stats_data)

        # Load with direct override of mean
        norm = dp.Normalize(
            input_keys=["x"],
            method="mean_std",
            stats_file=npz_path,
            means={"x": 50.0},  # Override mean from file
        )

        sample = TensorDict({"x": torch.tensor([60.0])})
        result = norm(sample)

        # Should use direct mean (50), but file std (10)
        # (60 - 50) / 10 = 1.0
        torch.testing.assert_close(
            result["x"], torch.tensor([1.0]), atol=1e-6, rtol=1e-6
        )


# ============================================================================
# State Dict Tests
# ============================================================================


def test_normalize_load_state_dict_mean_std():
    """Test loading state_dict for mean_std method."""
    state = {
        "input_keys": ["x"],
        "method": "mean_std",
        "means": {"x": torch.tensor(5.0)},
        "stds": {"x": torch.tensor(2.0)},
        "eps": 1e-8,
    }

    norm = dp.Normalize(
        input_keys=["x"], method="mean_std", means={"x": 0.0}, stds={"x": 1.0}
    )
    norm.load_state_dict(state)

    assert norm.method == "mean_std"

    sample = TensorDict({"x": torch.tensor([7.0])})
    result = norm(sample)
    # (7 - 5) / 2 = 1.0
    torch.testing.assert_close(result["x"], torch.tensor([1.0]), atol=1e-6, rtol=1e-6)


# ============================================================================
# Method Validation Tests
# ============================================================================


def test_normalize_invalid_method_raises():
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError, match="must be 'mean_std' or 'min_max'"):
        dp.Normalize(
            input_keys=["x"],
            method="invalid_method",
            means={"x": 0.0},
            stds={"x": 1.0},
        )


def test_normalize_minmax_missing_mins_raises():
    """Test that min_max method without mins raises ValueError."""
    with pytest.raises(ValueError, match="'mins' and 'maxs' must be provided"):
        dp.Normalize(
            input_keys=["x"],
            method="min_max",
            maxs={"x": 100.0},
        )


def test_normalize_minmax_missing_maxs_raises():
    """Test that min_max method without maxs raises ValueError."""
    with pytest.raises(ValueError, match="'mins' and 'maxs' must be provided"):
        dp.Normalize(
            input_keys=["x"],
            method="min_max",
            mins={"x": 0.0},
        )


def test_normalize_mean_std_missing_stds_raises():
    """Test that mean_std method without stds raises ValueError."""
    with pytest.raises(ValueError, match="'means' and 'stds' must be provided"):
        dp.Normalize(
            input_keys=["x"],
            method="mean_std",
            means={"x": 0.0},
        )


# ============================================================================
# Downsample transform - NOT YET IMPLEMENTED
# ============================================================================


def test_subsample_basic():
    sample = TensorDict({"points": torch.randn(1000, 3)})
    ds = dp.SubsamplePoints(input_keys=["points"], n_points=100)

    result = ds(sample)
    assert result["points"].shape == (100, 3)


def test_subsample_multiple_fields():
    sample = TensorDict(
        {
            "positions": torch.randn(500, 3),
            "features": torch.randn(500, 8),
        }
    )
    ds = dp.SubsamplePoints(input_keys=["positions", "features"], n_points=100)

    result = ds(sample)
    assert result["positions"].shape == (100, 3)
    assert result["features"].shape == (100, 8)


def test_subsample_preserves_other_fields():
    sample = TensorDict(
        {
            "points": torch.randn(500, 3),
            "label": torch.tensor([1]),
        }
    )
    ds = dp.SubsamplePoints(input_keys=["points"], n_points=100)

    result = ds(sample)
    assert result["points"].shape == (100, 3)
    torch.testing.assert_close(result["label"], torch.tensor([1]))


def test_subsample_no_op_when_smaller():
    sample = TensorDict({"x": torch.randn(50, 3)})
    ds = dp.SubsamplePoints(input_keys=["x"], n_points=100)

    result = ds(sample)
    # Should return original since 50 < 100
    assert result["x"].shape == (50, 3)


def test_subsample_inconsistent_sizes_raises():
    sample = TensorDict(
        {
            "a": torch.randn(100, 3),
            "b": torch.randn(200, 3),  # Different size!
        }
    )
    ds = dp.SubsamplePoints(input_keys=["a", "b"], n_points=50)

    with pytest.raises(ValueError, match="same first dimension"):
        ds(sample)


def test_subsample_missing_key_raises():
    sample = TensorDict({"x": torch.randn(100, 3)})
    ds = dp.SubsamplePoints(input_keys=["y"], n_points=50)

    with pytest.raises(KeyError):
        ds(sample)


def test_subsample_weighted():
    sample = TensorDict(
        {
            "points": torch.randn(1000, 3),
            "weights": torch.rand(1000),
        }
    )
    ds = dp.SubsamplePoints(input_keys=["points"], n_points=100, weights_key="weights")

    result = ds(sample)
    assert result["points"].shape == (100, 3)


def test_subsample_weighted_missing_weights_raises():
    sample = TensorDict({"points": torch.randn(1000, 3)})
    ds = dp.SubsamplePoints(
        input_keys=["points"], n_points=100, weights_key="missing_weights"
    )

    with pytest.raises(KeyError, match="missing_weights"):
        ds(sample)


def test_subsample_poisson_algorithm():
    sample = TensorDict({"points": torch.randn(1000, 3)})
    ds = dp.SubsamplePoints(
        input_keys=["points"], n_points=100, algorithm="poisson_fixed"
    )

    result = ds(sample)
    assert result["points"].shape == (100, 3)


def test_subsample_uniform_algorithm():
    sample = TensorDict({"points": torch.randn(1000, 3)})
    ds = dp.SubsamplePoints(input_keys=["points"], n_points=100, algorithm="uniform")

    result = ds(sample)
    assert result["points"].shape == (100, 3)


def test_subsample_repr():
    ds = dp.SubsamplePoints(input_keys=["x"], n_points=100)
    assert "SubsamplePoints" in repr(ds)
    assert "100" in repr(ds)


def test_subsample_repr_with_weights():
    ds = dp.SubsamplePoints(input_keys=["x"], n_points=100, weights_key="areas")
    assert "SubsamplePoints" in repr(ds)
    assert "weights_key=areas" in repr(ds)


# TODO: Implement Downsample transform
# def test_downsample_basic():
#     sample = Sample({"points": torch.randn(1000, 3)})
#     ds = dp.Downsample(input_keys=["points"], n=100)
#
#     result = ds(sample)
#     assert result["points"].shape == (100, 3)
#
#
# def test_downsample_multiple_fields():
#     sample = Sample(
#         {
#             "positions": torch.randn(500, 3),
#             "features": torch.randn(500, 8),
#         }
#     )
#     ds = dp.Downsample(input_keys=["positions", "features"], n=100)
#
#     result = ds(sample)
#     assert result["positions"].shape == (100, 3)
#     assert result["features"].shape == (100, 8)
#
#
# def test_downsample_preserves_other_fields():
#     sample = Sample(
#         {
#             "points": torch.randn(500, 3),
#             "label": torch.tensor([1]),
#         }
#     )
#     ds = dp.Downsample(input_keys=["points"], n=100)
#
#     result = ds(sample)
#     assert result["points"].shape == (100, 3)
#     torch.testing.assert_close(result["label"], torch.tensor([1]))
#
#
# def test_downsample_seed_reproducibility():
#     sample = Sample({"x": torch.randn(1000)})
#     ds1 = dp.Downsample(input_keys=["x"], n=100, seed=42)
#     ds2 = dp.Downsample(input_keys=["x"], n=100, seed=42)
#
#     result1 = ds1(sample)
#     result2 = ds2(sample)
#
#     torch.testing.assert_close(result1["x"], result2["x"])
#
#
# def test_downsample_no_op_when_smaller():
#     sample = Sample({"x": torch.randn(50, 3)})
#     ds = dp.Downsample(input_keys=["x"], n=100, replacement=False)
#
#     result = ds(sample)
#     # Should return original since 50 < 100 and no replacement
#     assert result["x"].shape == (50, 3)
#
#
# def test_downsample_with_replacement():
#     sample = Sample({"x": torch.randn(50, 3)})
#     ds = dp.Downsample(input_keys=["x"], n=100, replacement=True)
#
#     result = ds(sample)
#     # With replacement, can upsample
#     assert result["x"].shape == (100, 3)
#
#
# def test_downsample_different_axis():
#     # Shape: (3, 1000) - downsample along axis 1
#     sample = Sample({"x": torch.randn(3, 1000)})
#     ds = dp.Downsample(input_keys=["x"], n=100, axis=1)
#
#     result = ds(sample)
#     assert result["x"].shape == (3, 100)
#
#
# def test_downsample_inconsistent_sizes_raises():
#     sample = Sample(
#         {
#             "a": torch.randn(100, 3),
#             "b": torch.randn(200, 3),  # Different size!
#         }
#     )
#     ds = dp.Downsample(input_keys=["a", "b"], n=50)
#
#     with pytest.raises(ValueError, match="has size"):
#         ds(sample)
#
#
# def test_downsample_empty_fields_raises():
#     with pytest.raises(ValueError, match="cannot be empty"):
#         dp.Downsample(input_keys=[], n=100)
#
#
# def test_downsample_invalid_n_raises():
#     with pytest.raises(ValueError, match="must be >= 1"):
#         dp.Downsample(input_keys=["x"], n=0)


# ============================================================================
# Compose transform
# ============================================================================


def test_compose_single_transform():
    sample = TensorDict({"x": torch.tensor([10.0])})
    norm = dp.Normalize(
        input_keys=["x"], method="mean_std", means={"x": 10.0}, stds={"x": 1.0}
    )
    pipeline = dp.Compose([norm])

    result = pipeline(sample)
    torch.testing.assert_close(result["x"], torch.tensor([0.0]), atol=1e-6, rtol=1e-6)


def test_compose_order_matters():
    sample = TensorDict({"x": torch.tensor([100.0, 200.0, 300.0])})

    # Normalize then check values
    norm = dp.Normalize(
        input_keys=["x"], method="mean_std", means={"x": 200.0}, stds={"x": 100.0}
    )
    pipeline = dp.Compose([norm])

    result = pipeline(sample)
    expected = torch.tensor([-1.0, 0.0, 1.0])
    torch.testing.assert_close(result["x"], expected, atol=1e-6, rtol=1e-6)


def test_compose_len():
    pipeline = dp.Compose(
        [
            dp.Normalize(
                input_keys=["x"], method="mean_std", means={"x": 0.0}, stds={"x": 1.0}
            ),
            # dp.Downsample(input_keys=["x"], n=10),  # Not implemented yet
        ]
    )
    assert len(pipeline) == 1


def test_compose_getitem():
    norm = dp.Normalize(
        input_keys=["x"], method="mean_std", means={"x": 0.0}, stds={"x": 1.0}
    )
    # ds = dp.Downsample(input_keys=["x"], n=10)  # Not implemented yet
    pipeline = dp.Compose([norm])

    assert pipeline[0] is norm
    # assert pipeline[1] is ds


def test_compose_iteration():
    transforms = [
        dp.Normalize(
            input_keys=["x"], method="mean_std", means={"x": 0.0}, stds={"x": 1.0}
        ),
        # dp.Downsample(input_keys=["x"], n=10),  # Not implemented yet
    ]
    pipeline = dp.Compose(transforms)

    for i, t in enumerate(pipeline):
        assert t is transforms[i]


def test_compose_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        dp.Compose([])


def test_compose_non_transform_raises():
    with pytest.raises(TypeError, match="must be Transform"):
        dp.Compose([lambda x: x])


# ============================================================================
# Transform repr
# ============================================================================


def test_normalize_repr():
    norm = dp.Normalize(
        input_keys=["x"], method="mean_std", means={"x": 0.0}, stds={"x": 1.0}
    )
    assert "Normalize" in repr(norm)
    assert "mean_std" in repr(norm)


# def test_downsample_repr():
#     ds = dp.Downsample(input_keys=["x"], n=100)
#     assert "Downsample" in repr(ds)
#     assert "100" in repr(ds)


def test_compose_repr():
    pipeline = dp.Compose(
        [
            dp.Normalize(
                input_keys=["x"], method="mean_std", means={"x": 0.0}, stds={"x": 1.0}
            ),
        ]
    )
    assert "Compose" in repr(pipeline)
    assert "Normalize" in repr(pipeline)
