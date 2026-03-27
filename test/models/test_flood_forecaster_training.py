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
Unit tests for FloodForecaster training modules.
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

# Import modules explicitly to avoid conflicts with other utils modules
import importlib.util

# First, set up the training package structure
if "training" not in sys.modules:
    import types
    training_pkg = types.ModuleType("training")
    sys.modules["training"] = training_pkg

# Import trainer module FIRST (pretraining depends on it)
spec = importlib.util.spec_from_file_location("training.trainer", _examples_dir / "training" / "trainer.py")
trainer_module = importlib.util.module_from_spec(spec)
sys.modules["training.trainer"] = trainer_module
spec.loader.exec_module(trainer_module)

# Import pretraining (depends on trainer)
spec = importlib.util.spec_from_file_location("training.pretraining", _examples_dir / "training" / "pretraining.py")
pretraining_module = importlib.util.module_from_spec(spec)
sys.modules["training.pretraining"] = pretraining_module
spec.loader.exec_module(pretraining_module)

# Import domain_adaptation (depends on pretraining, but not on __init__)
# Use full module path to ensure __module__ attribute is set correctly for checkpoint loading
spec = importlib.util.spec_from_file_location("training.domain_adaptation", _examples_dir / "training" / "domain_adaptation.py")
domain_adaptation_module = importlib.util.module_from_spec(spec)
sys.modules["training.domain_adaptation"] = domain_adaptation_module
spec.loader.exec_module(domain_adaptation_module)

# Fix __module__ attributes for classes to ensure checkpoint loading works
for name in dir(domain_adaptation_module):
    obj = getattr(domain_adaptation_module, name)
    if isinstance(obj, type) and issubclass(obj, (torch.nn.Module, physicsnemo.Module)):
        obj.__module__ = "training.domain_adaptation"

# Now import __init__ (it can safely import from domain_adaptation and pretraining)
spec = importlib.util.spec_from_file_location("training", _examples_dir / "training" / "__init__.py")
training_init = importlib.util.module_from_spec(spec)
sys.modules["training"].__dict__.update(training_init.__dict__)
spec.loader.exec_module(training_init)

from training.domain_adaptation import (
    CNNDomainClassifier,
    DomainAdaptationTrainer,
    GradientReversal,
    GradientReversalFunction,
)
from training.pretraining import create_scheduler


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("scheduler_type", ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"])
def test_create_scheduler(device, scheduler_type):
    """Test scheduler creation for different types."""
    model = nn.Linear(10, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    config = MagicMock()
    config.training = MagicMock()
    if scheduler_type == "StepLR":
        config.training.get = lambda key, default=None: {
            "scheduler": "StepLR",
            "step_size": 10,
            "gamma": 0.5,
        }.get(key, default)
        expected_type = torch.optim.lr_scheduler.StepLR
    elif scheduler_type == "CosineAnnealingLR":
        config.training.get = lambda key, default=None: {
            "scheduler": "CosineAnnealingLR",
            "scheduler_T_max": 100,
        }.get(key, default)
        expected_type = torch.optim.lr_scheduler.CosineAnnealingLR
    else:  # ReduceLROnPlateau
        config.training.get = lambda key, default=None: {
            "scheduler": "ReduceLROnPlateau",
            "gamma": 0.5,
            "scheduler_patience": 5,
        }.get(key, default)
        expected_type = torch.optim.lr_scheduler.ReduceLROnPlateau

    scheduler = create_scheduler(optimizer, config)
    assert isinstance(scheduler, expected_type)


@pytest.mark.parametrize("device", _DEVICES)
def test_gradient_reversal(device):
    """Test gradient reversal forward and backward passes."""
    grl = GradientReversal(lambda_max=1.0)
    x = torch.rand(4, 10, requires_grad=True).to(device).detach().requires_grad_(True)

    # Forward: identity
    y = grl(x)
    assert torch.allclose(x, y)

    # Backward: negates gradient
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, -torch.ones_like(x.grad))


@pytest.fixture
def da_config():
    """Create mock DA config that supports .get() method, 'in' operator, and [] access."""
    # Use a dict-like object that supports both attribute access and dict-like access
    class DictLike:
        def __init__(self):
            self.conv_layers = [
                {"out_channels": 16, "kernel_size": 3, "pool_size": 2},
                {"out_channels": 32, "kernel_size": 3, "pool_size": 2},
            ]
            self.fc_dim = 1
        
        def get(self, key, default=None):
            """Support .get() method for dict-like access."""
            return getattr(self, key, default)
        
        def __contains__(self, key):
            """Support 'in' operator for dict-like access."""
            return hasattr(self, key)
        
        def __getitem__(self, key):
            """Support [] subscript access for dict-like access."""
            return getattr(self, key)
    
    return DictLike()


@pytest.mark.parametrize("device", _DEVICES)
def test_cnn_domain_classifier_init(da_config, device):
    """Test CNN domain classifier initialization."""
    classifier = CNNDomainClassifier(in_channels=64, lambda_max=1.0, da_cfg=da_config).to(device)

    assert isinstance(classifier, nn.Module)
    assert hasattr(classifier, "grl")
    assert isinstance(classifier.grl, GradientReversal)


@pytest.mark.parametrize("device", _DEVICES)
def test_cnn_domain_classifier_forward(da_config, device):
    """Test CNN domain classifier forward pass."""
    classifier = CNNDomainClassifier(in_channels=64, lambda_max=1.0, da_cfg=da_config).to(device)

    x = torch.rand(4, 64, 16, 16).to(device)  # (B, C, H, W)
    y = classifier(x)

    assert y.shape == (4, 1)  # (B, fc_dim)


@pytest.mark.parametrize("device", _DEVICES)
def test_domain_adaptation_trainer_init(device):
    """Test DomainAdaptationTrainer initialization."""
    mock_model = MagicMock(spec=nn.Module)
    mock_model.to = MagicMock(return_value=mock_model)
    mock_classifier = MagicMock(spec=nn.Module)
    mock_classifier.to = MagicMock(return_value=mock_classifier)
    mock_processor = MagicMock()
    mock_processor.to = MagicMock(return_value=mock_processor)

    trainer = DomainAdaptationTrainer(
        model=mock_model,
        data_processor=mock_processor,
        domain_classifier=mock_classifier,
        device=device,
    )

    assert trainer.model == mock_model
    assert trainer.domain_classifier == mock_classifier
    assert trainer.data_processor == mock_processor
    assert trainer.device == device


@pytest.mark.parametrize("device", _DEVICES)
def test_gradient_reversal_lambda(device):
    """Test GradientReversal lambda setting and scaling."""
    grl = GradientReversal(lambda_max=1.0)
    grl.set_lambda(0.5)
    assert grl.lambda_ == 0.5

    # Test lambda scales gradient
    grl2 = GradientReversal(lambda_max=0.5)
    x2 = torch.ones(4, 10, requires_grad=True).to(device).detach().requires_grad_(True)
    y2 = grl2(x2)
    loss2 = y2.sum()
    loss2.backward()
    assert x2.grad is not None
    assert torch.allclose(x2.grad, -0.5 * torch.ones_like(x2.grad))


@pytest.mark.parametrize("device", _DEVICES)
def test_create_scheduler_unknown_raises(device):
    """Test that unknown scheduler raises ValueError."""
    model = nn.Linear(10, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    config = MagicMock()
    config.training = MagicMock()
    config.training.get = lambda key, default=None: {
        "scheduler": "UnknownScheduler",
    }.get(key, default)

    with pytest.raises(ValueError, match="Unknown scheduler"):
        create_scheduler(optimizer, config)




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
def test_checkpoint_save_load(device):
    """Test checkpoint save/load for training components."""
    import physicsnemo
    from pathlib import Path
    
    # Test GradientReversal checkpoint
    grl_orig = GradientReversal(lambda_max=1.0).to(device)
    grl_path = Path("checkpoint_grl.mdlus")
    grl_orig.save(str(grl_path))
    grl_loaded = physicsnemo.Module.from_checkpoint(str(grl_path)).to(device)
    assert grl_loaded.lambda_ == 1.0
    assert isinstance(grl_loaded, GradientReversal)
    grl_path.unlink(missing_ok=True)
    
    # Test CNNDomainClassifier checkpoint
    da_config = {
        "conv_layers": [
            {"out_channels": 16, "kernel_size": 3, "pool_size": 2},
            {"out_channels": 32, "kernel_size": 3, "pool_size": 2},
        ],
        "fc_dim": 1
    }
    classifier_orig = CNNDomainClassifier(in_channels=64, lambda_max=1.0, da_cfg=da_config).to(device)
    classifier_path = Path("checkpoint_classifier.mdlus")
    classifier_orig.save(str(classifier_path))
    classifier_loaded = physicsnemo.Module.from_checkpoint(str(classifier_path)).to(device)
    assert classifier_loaded.grl.lambda_ == 1.0
    assert isinstance(classifier_loaded, CNNDomainClassifier)
    classifier_path.unlink(missing_ok=True)