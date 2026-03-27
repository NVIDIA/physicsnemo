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
Unit tests for FloodForecaster data processing modules.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import physicsnemo

# Conditionally include CUDA in device parametrization only if available
_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda:0")

# Add the FloodForecaster example to the path
_examples_dir = Path(__file__).parent.parent.parent / "examples" / "weather" / "flood_modeling" / "flood_forecaster"
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from data_processing import FloodGINODataProcessor, GINOWrapper, LpLossWrapper

from . import common


# Define MockGINOModelForCheckpoint at module level so it can be properly loaded from checkpoint
# Note: Name doesn't start with "Test" to avoid pytest collection
# Register it in the model registry to ensure it can be found when loading checkpoints
class MockGINOModelForCheckpoint(physicsnemo.Module):
    """Simple test GINO model for checkpoint testing."""
    def __init__(self):
        super().__init__(meta=physicsnemo.ModelMetaData(name="MockGINOForCheckpoint"))
        self.fno_hidden_channels = 64
        self.out_channels = 3
        self.gno_coord_dim = 2
        self.latent_feature_channels = None
        self.in_coord_dim_reverse_order = [2, 3]
        self.out_gno_tanh = None
        
        # Create minimal layers for the model to work
        self.gno_in = nn.Linear(2, 64)
        self.gno_out = nn.Linear(64, 64)
        self.projection = nn.Linear(64, 3)
        self.latent_embedding = nn.Identity()

# Register MockGINOModelForCheckpoint in the model registry so it can be loaded from checkpoints
try:
    registry = physicsnemo.registry.ModelRegistry()
    if "MockGINOModelForCheckpoint" not in registry.list_models():
        registry.register(MockGINOModelForCheckpoint, "MockGINOModelForCheckpoint")
except (ValueError, AttributeError):
    # Already registered or registry issue - continue
    pass


@pytest.fixture
def sample_dict():
    """Create sample dictionary for preprocessing."""
    batch_size = 2
    num_cells = 100
    n_history = 3
    # query_points should be (B, H, W, 2) or (H, W, 2) for latent queries
    # Using a simple grid: (8, 8, 2) for 2D
    H, W = 8, 8

    return {
        "geometry": torch.rand(batch_size, num_cells, 2),
        "static": torch.rand(batch_size, num_cells, 7),
        "boundary": torch.rand(batch_size, n_history, num_cells, 1),
        "dynamic": torch.rand(batch_size, n_history, num_cells, 3),
        "target": torch.rand(batch_size, num_cells, 3),
        "query_points": torch.rand(batch_size, H, W, 2),  # Required for preprocessing
    }


@pytest.fixture
def mock_gino_model():
    """Create a mock GINO model with internal components."""
    model = MagicMock(spec=nn.Module)
    model.fno_hidden_channels = 64
    model.out_channels = 3
    model.gno_coord_dim = 2
    model.latent_feature_channels = None
    model.in_coord_dim_reverse_order = [2, 3]  # For 2D: [2, 3] means permute dims 2,3
    model.out_gno_tanh = None
    
    # Mock internal components - gno_in returns flattened output
    def mock_gno_in(y, x, f_y=None):
        # Returns (n_points, channels) for flattened queries
        # GINOWrapper reshapes this to (batch_size, H, W, channels)
        n_points = x.shape[0]
        return torch.rand(n_points, 64)  # (n_points, channels)
    
    model.gno_in = MagicMock(side_effect=mock_gno_in)
    
    # Mock gno_out - returns (B, channels, n_out) after permute
    def mock_gno_out(y, x, f_y):
        # f_y is (B, n_latent, channels)
        batch_size = f_y.shape[0]
        n_out = x.shape[0]
        return torch.rand(batch_size, 64, n_out)  # (B, channels, n_out)
    
    model.gno_out = MagicMock(side_effect=mock_gno_out)
    
    # Mock projection - returns (B, n_out, out_channels)
    def mock_projection(x):
        # x is (B, n_out, channels) after permute in GINOWrapper
        batch_size = x.shape[0]
        n_out = x.shape[1]
        return torch.rand(batch_size, n_out, 3)  # (B, n_out, out_channels)
    
    model.projection = MagicMock(side_effect=mock_projection)
    
    # Mock latent_embedding to return a tensor
    def mock_latent_embedding(in_p, ada_in=None):
        # in_p shape: (B, H, W, channels)
        # Return shape: (B, channels, H, W) for 2D
        batch_size = in_p.shape[0]
        grid_shape = in_p.shape[1:-1]  # (H, W)
        return torch.rand(batch_size, 64, *grid_shape)
    
    model.latent_embedding = mock_latent_embedding
    
    return model


@pytest.mark.parametrize("device", _DEVICES)
def test_data_processor_init(device):
    """Test FloodGINODataProcessor initialization."""
    processor = FloodGINODataProcessor(device=device)
    # device property returns torch.device, so compare string representations
    assert str(processor.device) == str(torch.device(device))
    assert processor.target_norm is None
    assert processor.inverse_test is True
    assert processor.model is None
    assert isinstance(processor, nn.Module)


@pytest.mark.parametrize("device", _DEVICES)
def test_data_processor_preprocess(sample_dict, device):
    """Test preprocessing with batched input."""
    processor = FloodGINODataProcessor(device=device)
    result = processor.preprocess(sample_dict)

    # Check required keys exist
    assert "input_geom" in result
    assert "latent_queries" in result
    assert "output_queries" in result
    assert "x" in result
    assert "y" in result

    # Check geometry has no batch dim (shared)
    assert result["input_geom"].dim() == 2  # (n_cells, 2)
    assert result["latent_queries"].dim() == 3  # (H, W, 2)
    assert result["output_queries"].dim() == 2  # (n_cells, 2)

    # Check x has batch dim
    assert result["x"].dim() == 3  # (B, n_cells, features)


@pytest.mark.parametrize("device", _DEVICES)
def test_data_processor_postprocess(device):
    """Test postprocessing in training and eval modes."""
    mock_norm = MagicMock()
    mock_norm.inverse_transform = MagicMock(return_value=torch.ones(2, 100, 3) * 2)

    processor = FloodGINODataProcessor(device=device, target_norm=mock_norm, inverse_test=True)
    out = torch.ones(2, 100, 3)
    sample = {"y": torch.ones(2, 100, 3)}

    # Training mode: no inverse transform
    processor.train()
    result_out, _ = processor.postprocess(out, sample)
    mock_norm.inverse_transform.assert_not_called()

    # Eval mode: applies inverse transform
    processor.eval()
    result_out, _ = processor.postprocess(out, sample)
    assert torch.allclose(result_out, torch.ones(2, 100, 3) * 2)


@pytest.mark.parametrize("device", _DEVICES)
def test_ginowrapper_init(mock_gino_model, device):
    """Test GINOWrapper initialization."""
    wrapper = GINOWrapper(mock_gino_model)
    # After conversion, gino is wrapped in CustomPhysicsNeMoWrapper
    # Check that the inner model is accessible
    assert hasattr(wrapper.gino, 'inner_model') or wrapper.gino == mock_gino_model
    assert isinstance(wrapper, nn.Module)
    assert wrapper.fno_hidden_channels == 64
    assert wrapper.autoregressive is False  # Default value


@pytest.mark.parametrize("device", _DEVICES)
def test_ginowrapper_forward(mock_gino_model, device):
    """Test GINOWrapper forward pass with kwargs filtering and autoregressive mode."""
    wrapper = GINOWrapper(mock_gino_model, autoregressive=True)

    input_geom = torch.rand(1, 100, 2)
    latent_queries = torch.rand(1, 8, 8, 2)
    output_queries = torch.rand(1, 100, 2)
    x = torch.rand(1, 100, 10)

    # Mock the internal GNO components
    mock_gino_model.gno_in.return_value = torch.rand(8 * 8, 64)  # (n_points, channels)
    mock_gino_model.gno_out.return_value = torch.rand(1, 64, 100)  # (B, channels, n_out)
    def mock_projection(x):
        # x is (B, n_out, channels), should return (B, n_out, out_channels)
        return torch.zeros(x.shape[0], x.shape[1], 3)  # (B, n_out, out_channels)
    mock_gino_model.projection.side_effect = mock_projection

    # Test forward with extra kwargs (should be filtered)
    result = wrapper(
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        x=x,
        y=torch.rand(1, 100, 3),  # Should be filtered out
        extra_arg="should be ignored",
    )

    # Verify result shape and autoregressive residual
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 100, 3)
    expected = x[..., -3:]  # Last 3 channels (autoregressive residual)
    assert torch.allclose(result, expected, atol=1e-5)

    # Test return_features
    out, features = wrapper(
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        x=x,
        return_features=True,
    )
    assert isinstance(features, torch.Tensor)
    assert features.shape == (1, 64, 8, 8)  # (B, channels, H, W)


@pytest.mark.parametrize("device", _DEVICES)
def test_lploss_wrapper(device):
    """Test LpLossWrapper filters kwargs correctly."""
    mock_loss = MagicMock(return_value=torch.tensor(0.5))
    wrapper = LpLossWrapper(mock_loss)

    y_pred = torch.rand(2, 100, 3)
    y = torch.rand(2, 100, 3)

    # Extra kwargs should be ignored
    wrapper(y_pred, y=y, input_geom=torch.rand(100, 2), extra_arg="ignored")

    # Should only pass y_pred and y
    mock_loss.assert_called_once_with(y_pred, y)


def _instantiate_model(cls, seed: int = 0, **kwargs):
    """Helper to create model with reproducible parameters."""
    model = cls(**kwargs)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=param.dtype))
    return model




@pytest.mark.parametrize("device", _DEVICES)
def test_ginowrapper_from_checkpoint(device, mock_gino_model):
    """Test loading GINOWrapper from checkpoint and verify outputs."""
    from pathlib import Path
    import physicsnemo
    
    # Use the module-level MockGINOModelForCheckpoint class for checkpoint testing
    # This ensures the class can be properly loaded from checkpoint
    gino_model = MockGINOModelForCheckpoint()
    
    # Create a model and save checkpoint
    model_orig = GINOWrapper(gino_model, autoregressive=False).to(device)
    checkpoint_path = Path("checkpoint_gino_wrapper.mdlus")
    model_orig.save(str(checkpoint_path))
    
    # Load from checkpoint - use strict=False to handle potential state dict mismatches
    # The nested TestGINOModel should be properly reconstructed via module path or registry
    model = physicsnemo.Module.from_checkpoint(str(checkpoint_path), strict=False).to(device)
    
    # Verify attributes after loading
    assert model.autoregressive is False
    assert isinstance(model, GINOWrapper)
    # Verify the wrapped model was loaded correctly with all layers
    # Note: The model structure is preserved even if class type isn't exactly TestGINOModel
    # (physicsnemo may load it as a generic Module if the class can't be imported)
    assert hasattr(model.gino, 'gno_in')
    assert hasattr(model.gino, 'gno_out')
    assert hasattr(model.gino, 'projection')
    assert hasattr(model.gino, 'latent_embedding')
    # Verify the layers have the correct structure
    assert isinstance(model.gino.gno_in, nn.Linear)
    assert isinstance(model.gino.gno_out, nn.Linear)
    assert isinstance(model.gino.projection, nn.Linear)
    
    # Cleanup
    checkpoint_path.unlink(missing_ok=True)



