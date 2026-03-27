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

"""
Unit tests for FloodForecaster inference module.

This module tests the inference pipeline including checkpoint loading,
rollout prediction, and metric computation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import shutil

import pytest
import torch
import torch.nn as nn
import numpy as np

import physicsnemo

# Conditionally include CUDA in device parametrization only if available
_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda:0")

# Add the FloodForecaster example to the path
_examples_dir = Path(__file__).parent.parent.parent / "examples" / "weather" / "flood_modeling" / "flood_forecaster"
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

# Import rollout functions - need to set up module structure first
import importlib.util

# Set up inference package
if "inference" not in sys.modules:
    import types
    inference_pkg = types.ModuleType("inference")
    sys.modules["inference"] = inference_pkg

# Import rollout module
spec = importlib.util.spec_from_file_location("inference.rollout", _examples_dir / "inference" / "rollout.py")
rollout_module = importlib.util.module_from_spec(spec)
sys.modules["inference.rollout"] = rollout_module
spec.loader.exec_module(rollout_module)

# Make rollout available as inference.rollout attribute
sys.modules["inference"].rollout = rollout_module

from inference.rollout import compute_csi, rollout_prediction
from data_processing import GINOWrapper


@pytest.fixture
def mock_gino_model():
    """Create a mock GINO model."""
    model = MagicMock(spec=nn.Module)
    model.fno_hidden_channels = 64
    model.out_channels = 3
    model.gno_coord_dim = 2
    model.in_coord_dim_reverse_order = [2, 3]  # For 2D: permute dims 2,3 (H, W)
    model.out_gno_tanh = None  # No tanh activation
    
    # Mock latent_embedding (required by GINOWrapper.forward)
    # Use Identity which just returns the input
    model.latent_embedding = nn.Identity()
    
    # Mock gno_in method (required by GINOWrapper.forward)
    def mock_gno_in(y, x, f_y=None):
        # Return tensor with shape (n_points, channels) for flattened queries
        # GINOWrapper expects gno_in to return (n_points, channels) which gets reshaped to (batch_size, H, W, channels)
        n_points = x.shape[0] if x.ndim == 2 else x.shape[1]
        return torch.rand(n_points, 64)  # (n_points, channels)
    
    model.gno_in = MagicMock(side_effect=mock_gno_in)
    
    # Mock gno_out method (required by GINOWrapper.forward)
    def mock_gno_out(y, x, f_y):
        # f_y is (B, n_latent, channels), x is output queries (n_out, coord_dim)
        # Return (B, channels, n_out) - will be permuted to (B, n_out, channels) in GINOWrapper
        batch_size = f_y.shape[0]
        n_out = x.shape[0]
        return torch.rand(batch_size, 64, n_out)  # (B, channels, n_out)
    
    model.gno_out = MagicMock(side_effect=mock_gno_out)
    
    # Mock projection method (required by GINOWrapper.forward)
    def mock_projection(x):
        # x is (B, n_out, channels) after permute in GINOWrapper
        batch_size = x.shape[0]
        n_out = x.shape[1]
        return torch.rand(batch_size, n_out, 3)  # (B, n_out, out_channels)
    
    model.projection = MagicMock(side_effect=mock_projection)
    
    # Mock forward to return predictions on correct device
    def mock_forward(**kwargs):
        batch_size = kwargs.get("x", torch.rand(1, 100, 10)).shape[0]
        n_out = kwargs.get("output_queries", torch.rand(100, 2)).shape[0]
        # Get device from x if available, otherwise use CPU
        x_tensor = kwargs.get("x")
        if x_tensor is not None:
            device = x_tensor.device
        else:
            device = torch.device("cpu")
        return torch.rand(batch_size, n_out, 3, device=device)
    
    model.forward = MagicMock(side_effect=mock_forward)
    return model


@pytest.fixture
def mock_normalizer():
    """Create a mock normalizer."""
    norm = MagicMock()
    norm.inverse_transform = MagicMock(side_effect=lambda x: x * 2.0)  # Simple transform
    norm.to = MagicMock(return_value=norm)
    return norm


@pytest.fixture
def mock_rollout_dataset():
    """Create a mock rollout dataset."""
    class MockRolloutDataset:
        def __init__(self):
            self.valid_run_ids = ["run_001", "run_002"]
        
        def __len__(self):
            return 2
        
        def __getitem__(self, idx):
            return {
                "run_id": self.valid_run_ids[idx],
                "geometry": torch.rand(100, 2),
                "static": torch.rand(100, 7),
                "boundary": torch.rand(20, 100, 1),
                "dynamic": torch.rand(20, 100, 3),
                "cell_area": torch.rand(100),
            }
    
    return MockRolloutDataset()


@pytest.mark.parametrize("device", _DEVICES)
def test_compute_csi_perfect_match(device):
    """Test CSI computation with perfect match."""
    pred = np.array([1.0, 1.0, 0.0, 0.0])
    gt = np.array([1.0, 1.0, 0.0, 0.0])
    threshold = 0.5
    
    csi = compute_csi(threshold, pred, gt)
    assert csi == 1.0


@pytest.mark.parametrize("device", _DEVICES)
def test_compute_csi_no_match(device):
    """Test CSI computation with no match."""
    pred = np.array([1.0, 1.0, 0.0, 0.0])
    gt = np.array([0.0, 0.0, 1.0, 1.0])
    threshold = 0.5
    
    csi = compute_csi(threshold, pred, gt)
    assert csi == 0.0


@pytest.mark.parametrize("device", _DEVICES)
def test_compute_csi_partial_match(device):
    """Test CSI computation with partial match."""
    pred = np.array([1.0, 1.0, 1.0, 0.0])
    gt = np.array([1.0, 1.0, 0.0, 0.0])
    threshold = 0.5
    
    csi = compute_csi(threshold, pred, gt)
    # TP=2, FP=1, FN=0, CSI = 2/(2+1+0) = 2/3
    assert abs(csi - 2.0/3.0) < 1e-6


@pytest.mark.parametrize("device", _DEVICES)
def test_compute_csi_all_zeros(device):
    """Test CSI computation with all zeros (no events)."""
    pred = np.array([0.0, 0.0, 0.0, 0.0])
    gt = np.array([0.0, 0.0, 0.0, 0.0])
    threshold = 0.5
    
    csi = compute_csi(threshold, pred, gt)
    # When TP+FP+FN=0, CSI should be 1.0 (perfect match, no events)
    assert csi == 1.0


@pytest.mark.parametrize("device", _DEVICES)
def test_compute_csi_different_thresholds(device):
    """Test CSI computation with different thresholds."""
    pred = np.array([0.3, 0.6, 0.2, 0.8])
    gt = np.array([0.1, 0.7, 0.4, 0.9])
    
    # Low threshold
    csi_low = compute_csi(0.05, pred, gt)
    # High threshold
    csi_high = compute_csi(0.3, pred, gt)
    
    # Both should be valid (0-1 range)
    assert 0.0 <= csi_low <= 1.0
    assert 0.0 <= csi_high <= 1.0
    # Different thresholds should give different results
    assert csi_low != csi_high or (csi_low == 1.0 and csi_high == 1.0)


@pytest.mark.parametrize("device", _DEVICES)
def test_compute_csi_edge_cases(device):
    """Test CSI computation with various edge cases."""
    # Single element
    csi = compute_csi(0.5, np.array([1.0]), np.array([1.0]))
    assert csi == 1.0
    
    # All predictions above threshold, no ground truth
    csi = compute_csi(0.5, np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    assert csi == 0.0
    
    # All ground truth above threshold, no predictions
    csi = compute_csi(0.5, np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    assert csi == 0.0


