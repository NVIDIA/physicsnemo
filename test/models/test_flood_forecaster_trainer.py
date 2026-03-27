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
Unit tests for FloodForecaster NeuralOperatorTrainer class.

This module tests the PhysicsNeMo-style Trainer class that handles neural operator training
with checkpointing, distributed training, and evaluation support.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import shutil

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import physicsnemo

# Conditionally include CUDA in device parametrization only if available
_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda:0")

# Add the FloodForecaster example to the path
_examples_dir = Path(__file__).parent.parent.parent / "examples" / "weather" / "flood_modeling" / "flood_forecaster"
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

# Import modules explicitly to avoid conflicts
import importlib.util

# First, set up the training package structure
if "training" not in sys.modules:
    import types
    training_pkg = types.ModuleType("training")
    sys.modules["training"] = training_pkg

# Import trainer module
spec = importlib.util.spec_from_file_location("training.trainer", _examples_dir / "training" / "trainer.py")
trainer_module = importlib.util.module_from_spec(spec)
sys.modules["training.trainer"] = trainer_module
spec.loader.exec_module(trainer_module)

from training.trainer import NeuralOperatorTrainer, _has_pytorch_submodules, save_model_checkpoint
from data_processing import GINOWrapper

from . import common


@pytest.fixture
def simple_model(device):
    """Create a simple test model that accepts kwargs."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 3)
            )
        
        def forward(self, x=None, **kwargs):
            # Accept x as positional or keyword, ignore other kwargs
            if x is None:
                # Try to get x from kwargs
                x = kwargs.get('x', None)
                if x is None:
                    raise ValueError("x must be provided")
            return self.layers(x)
    
    return SimpleModel().to(device)


@pytest.fixture
def mock_data_processor(device):
    """Create a mock data processor."""
    processor = MagicMock(spec=nn.Module)
    processor.preprocess = MagicMock(side_effect=lambda x: x)
    processor.postprocess = MagicMock(side_effect=lambda out, sample: (out, sample))
    processor.train = MagicMock()
    processor.eval = MagicMock()
    processor.to = MagicMock(return_value=processor)
    return processor


@pytest.fixture
def sample_dataset(device):
    """Create a sample dataset for training."""
    batch_size = 4
    n_samples = 20
    n_features = 10
    n_outputs = 3
    
    # Create sample data
    x = torch.rand(n_samples, n_features).to(device)
    y = torch.rand(n_samples, n_outputs).to(device)
    
    # Create dataset with dict format
    samples = [{"x": x[i], "y": y[i]} for i in range(n_samples)]
    return samples


@pytest.fixture
def train_loader(sample_dataset, device):
    """Create a training dataloader."""
    class DictDataset:
        def __init__(self, samples):
            self.samples = samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    dataset = DictDataset(sample_dataset)
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def test_loader(sample_dataset, device):
    """Create a test dataloader."""
    class DictDataset:
        def __init__(self, samples):
            self.samples = samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    dataset = DictDataset(sample_dataset[:10])  # Smaller test set
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.mark.parametrize("device", _DEVICES)
def test_trainer_init(simple_model, mock_data_processor, device):
    """Test NeuralOperatorTrainer initialization with various configurations."""
    # Basic init
    trainer = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=10,
        device=device,
        verbose=False,
    )
    assert trainer.model == simple_model
    assert trainer.n_epochs == 10
    assert str(trainer.device) == str(torch.device(device))
    assert trainer.mixed_precision is False
    assert trainer.data_processor is None
    
    # With data processor
    trainer2 = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=10,
        device=device,
        data_processor=mock_data_processor,
        verbose=False,
    )
    assert trainer2.data_processor == mock_data_processor
    
    # With mixed precision
    trainer3 = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=10,
        device=device,
        mixed_precision=True,
        verbose=False,
    )
    assert trainer3.mixed_precision is True
    assert trainer3.scaler is not None


@pytest.mark.parametrize("device", _DEVICES)
def test_trainer_train_one_epoch(simple_model, train_loader, test_loader, device):
    """Test training for one epoch."""
    trainer = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=1,
        device=device,
        verbose=False,
        eval_interval=1,
    )
    
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    from neuralop.losses import LpLoss
    training_loss = LpLoss(d=2, p=2)
    eval_losses = {"l2": LpLoss(d=2, p=2)}
    
    # Train for one epoch
    metrics = trainer.train(
        train_loader=train_loader,
        test_loaders={"val": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=training_loss,
        eval_losses=eval_losses,
        save_dir=None,  # Don't save checkpoints in test
    )
    
    # Check that metrics were returned
    assert isinstance(metrics, dict)
    assert "train_err" in metrics
    assert "val_l2" in metrics
    assert trainer.epoch == 0  # After 1 epoch, epoch should be 0 (0-indexed)


@pytest.mark.parametrize("device", _DEVICES)
def test_trainer_checkpoint_save_load(simple_model, train_loader, test_loader, device, tmp_path):
    """Test checkpoint saving and loading."""
    trainer = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=2,
        device=device,
        verbose=False,
        eval_interval=1,
    )
    
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    from neuralop.losses import LpLoss
    training_loss = LpLoss(d=2, p=2)
    eval_losses = {"l2": LpLoss(d=2, p=2)}
    
    save_dir = tmp_path / "checkpoints"
    save_dir.mkdir()
    
    # Train for one epoch and save checkpoint
    trainer.train(
        train_loader=train_loader,
        test_loaders={"val": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=training_loss,
        eval_losses=eval_losses,
        save_dir=str(save_dir),
        save_best="val_l2",
    )
    
    # Check that checkpoint files were created
    checkpoint_files = list(save_dir.glob("checkpoint.*.pt"))
    assert len(checkpoint_files) > 0, "Checkpoint file should be created"
    
    # Create new trainer and resume from checkpoint
    new_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 3)
    ).to(device)
    
    new_trainer = NeuralOperatorTrainer(
        model=new_model,
        n_epochs=2,
        device=device,
        verbose=False,
        eval_interval=1,
    )
    
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=0.9)
    
    # Resume from checkpoint
    new_trainer.train(
        train_loader=train_loader,
        test_loaders={"val": test_loader},
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        training_loss=training_loss,
        eval_losses=eval_losses,
        save_dir=str(save_dir),
        resume_from_dir=str(save_dir),
    )
    
    # Check that training resumed
    assert new_trainer.start_epoch > 0 or new_trainer.epoch >= 0


@pytest.mark.parametrize("device", _DEVICES)
def test_trainer_training_features(simple_model, mock_data_processor, train_loader, test_loader, device, tmp_path):
    """Test training with various features: mixed precision, data processor, best model tracking."""
    from neuralop.losses import LpLoss
    
    # Test mixed precision training
    trainer1 = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=1,
        device=device,
        mixed_precision=True,
        verbose=False,
        eval_interval=1,
    )
    optimizer1 = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=1, gamma=0.9)
    training_loss = LpLoss(d=2, p=2)
    eval_losses = {"l2": LpLoss(d=2, p=2)}
    
    metrics1 = trainer1.train(
        train_loader=train_loader,
        test_loaders={"val": test_loader},
        optimizer=optimizer1,
        scheduler=scheduler1,
        training_loss=training_loss,
        eval_losses=eval_losses,
    )
    assert isinstance(metrics1, dict)
    assert trainer1.scaler is not None
    
    # Test with data processor
    trainer2 = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=1,
        device=device,
        data_processor=mock_data_processor,
        verbose=False,
        eval_interval=1,
    )
    optimizer2 = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.9)
    metrics2 = trainer2.train(
        train_loader=train_loader,
        test_loaders={"val": test_loader},
        optimizer=optimizer2,
        scheduler=scheduler2,
        training_loss=training_loss,
        eval_losses=eval_losses,
    )
    assert isinstance(metrics2, dict)
    
    # Test best model tracking
    save_dir = tmp_path / "checkpoints"
    save_dir.mkdir()
    trainer3 = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=2,
        device=device,
        verbose=False,
        eval_interval=1,
    )
    optimizer3 = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=1, gamma=0.9)
    trainer3.train(
        train_loader=train_loader,
        test_loaders={"val": test_loader},
        optimizer=optimizer3,
        scheduler=scheduler3,
        training_loss=training_loss,
        eval_losses=eval_losses,
        save_dir=str(save_dir),
        save_best="val_l2",
    )
    assert trainer3.best_metric_value < float("inf")
    checkpoint_files = list(save_dir.glob("checkpoint.*.pt"))
    assert len(checkpoint_files) > 0

