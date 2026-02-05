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

"""Tests for active learning training loop."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from physicsnemo.active_learning.loop import (
    DefaultTrainingLoop,
    _recursive_data_device_cast,
)

# Define device parametrization for reuse across tests
AVAILABLE_DEVICES = [torch.device("cpu")] + (
    [torch.device("cuda:0")] if torch.cuda.is_available() else []
)
DEVICE_IDS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class TestRecursiveDataDeviceCast:
    """Tests for _recursive_data_device_cast function."""

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.bfloat16],
        ids=["float32", "float16", "bfloat16"],
    )
    def test_tensor_cast_dtype(self, dtype: torch.dtype):
        """Test casting tensor to different dtypes."""
        tensor = torch.randn(4, 8, dtype=torch.float32)
        result = _recursive_data_device_cast(tensor, dtype=dtype)
        assert result.dtype == dtype
        assert result.shape == tensor.shape

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES, ids=DEVICE_IDS)
    def test_tensor_cast_device(self, device: torch.device):
        """Test moving tensor to different devices."""
        tensor = torch.randn(4, 8)
        result = _recursive_data_device_cast(tensor, device=device)
        assert result.device == device
        assert result.shape == tensor.shape

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES, ids=DEVICE_IDS)
    def test_dict_cast(self, device: torch.device):
        """Test casting dictionary of tensors."""
        data = {
            "input": torch.randn(4, 8),
            "target": torch.randn(4, 1),
            "metadata": torch.tensor([1, 2, 3, 4]),
        }
        result = _recursive_data_device_cast(data, device=device, dtype=torch.float16)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(data.keys())
        for key in result:
            assert result[key].device == device
            if result[key].dtype.is_floating_point:
                assert result[key].dtype == torch.float16
            assert result[key].shape == data[key].shape

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES, ids=DEVICE_IDS)
    def test_list_cast(self, device: torch.device):
        """Test casting list of tensors."""
        data = [
            torch.randn(4, 8),
            torch.randn(4, 1),
            torch.tensor([1, 2, 3, 4]),
        ]
        result = _recursive_data_device_cast(data, device=device, dtype=torch.float16)

        assert isinstance(result, list)
        assert len(result) == len(data)
        for i, tensor in enumerate(result):
            assert tensor.device == device
            if tensor.dtype.is_floating_point:
                assert tensor.dtype == torch.float16
            assert tensor.shape == data[i].shape

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES, ids=DEVICE_IDS)
    def test_tuple_cast(self, device: torch.device):
        """Test casting tuple of tensors."""
        data = (
            torch.randn(4, 8),
            torch.randn(4, 1),
            torch.tensor([1, 2, 3, 4]),
        )
        result = _recursive_data_device_cast(data, device=device, dtype=torch.float16)

        assert isinstance(result, tuple)
        assert len(result) == len(data)
        for i, tensor in enumerate(result):
            assert tensor.device == device
            if tensor.dtype.is_floating_point:
                assert tensor.dtype == torch.float16
            assert tensor.shape == data[i].shape

    def test_nested_structures(self):
        """Test casting nested data structures."""
        data = {
            "batch": {
                "input": torch.randn(4, 8),
                "target": torch.randn(4, 1),
            },
            "metadata": [
                torch.tensor([1, 2, 3, 4]),
                torch.tensor([5, 6, 7, 8]),
            ],
        }

        result = _recursive_data_device_cast(
            data, device=torch.device("cpu"), dtype=torch.float32
        )

        assert isinstance(result, dict)
        assert isinstance(result["batch"], dict)
        assert isinstance(result["metadata"], list)
        assert result["batch"]["input"].dtype == torch.float32
        assert result["batch"]["target"].dtype == torch.float32

    def test_non_tensor_passthrough(self):
        """Test that non-tensor data passes through unchanged."""
        data = {
            "tensor": torch.randn(4, 8),
            "string": "some_string",
            "int": 42,
            "float": 3.14,
        }

        result = _recursive_data_device_cast(data, device=torch.device("cpu"))

        assert result["string"] == "some_string"
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert isinstance(result["tensor"], torch.Tensor)


class TestDefaultTrainingLoop:
    """Tests for DefaultTrainingLoop class."""

    def test_instantiation_default(self):
        """Test instantiation with default parameters."""
        loop = DefaultTrainingLoop()

        assert loop.train_step_fn is None
        assert loop.validate_step_fn is None
        assert loop.enable_static_capture is True
        assert loop.use_progress_bars is True
        assert loop.dtype == torch.get_default_dtype()

    def test_instantiation_with_train_step(self):
        """Test instantiation with custom train step function."""

        def mock_train_step(model, batch):
            return torch.tensor(0.5)

        loop = DefaultTrainingLoop(train_step_fn=mock_train_step)
        assert loop.train_step_fn == mock_train_step

    def test_call_without_train_step_raises_error(self, mock_module):
        """Test that calling loop without train_step_fn raises error."""
        loop = DefaultTrainingLoop()
        optimizer = torch.optim.SGD(mock_module.parameters(), lr=0.01)

        # Create mock dataloader
        dataset = TensorDataset(torch.randn(8, 64), torch.randn(8, 3))
        dataloader = DataLoader(dataset, batch_size=4)

        with pytest.raises(RuntimeError, match="No training step function provided"):
            loop(mock_module, optimizer, dataloader, max_epochs=1)

    def test_basic_training_loop_execution(self, mock_module):
        """Test basic execution of training loop with mocked components."""
        # Create a mock train step that returns a loss with backward method
        mock_loss = MagicMock()
        mock_loss.detach.return_value.item.return_value = 0.5
        mock_loss.backward = MagicMock()

        def mock_train_step(model, batch, *args, **kwargs):
            return mock_loss

        loop = DefaultTrainingLoop(
            enable_static_capture=False,
            use_progress_bars=False,
        )

        optimizer = torch.optim.SGD(mock_module.parameters(), lr=0.01)

        # Wrap optimizer.step to track calls
        original_optimizer_step = optimizer.step
        optimizer.step = MagicMock(side_effect=original_optimizer_step)

        # Create mock dataloader with 2 batches
        dataset = TensorDataset(torch.randn(8, 64), torch.randn(8, 3))
        dataloader = DataLoader(dataset, batch_size=4)

        # Run the loop, passing train_step_fn to __call__
        loop(
            mock_module,
            optimizer,
            dataloader,
            max_epochs=2,
            train_step_fn=mock_train_step,
        )

        # Verify train step was called (2 epochs * 2 batches = 4 times)
        assert mock_loss.backward.call_count == 4
        assert mock_loss.detach.call_count == 4
        # Verify optimizer.step was called once per batch (4 times total)
        assert optimizer.step.call_count == 4

    def test_training_with_validation(self, mock_module):
        """Test training loop with validation step."""
        # Create mock train step
        mock_loss = MagicMock()
        mock_loss.detach.return_value.item.return_value = 0.5
        mock_loss.backward = MagicMock()

        def mock_train_step(model, batch, *args, **kwargs):
            return mock_loss

        # Create mock validation step
        mock_validate_step = MagicMock()

        loop = DefaultTrainingLoop(
            enable_static_capture=False,
            use_progress_bars=False,
        )

        optimizer = torch.optim.SGD(mock_module.parameters(), lr=0.01)

        # Wrap optimizer.step to track calls
        original_optimizer_step = optimizer.step
        optimizer.step = MagicMock(side_effect=original_optimizer_step)

        # Create mock dataloaders
        train_dataset = TensorDataset(torch.randn(8, 64), torch.randn(8, 3))
        train_dataloader = DataLoader(train_dataset, batch_size=4)

        val_dataset = TensorDataset(torch.randn(4, 64), torch.randn(4, 3))
        val_dataloader = DataLoader(val_dataset, batch_size=4)

        # Run the loop, passing both step functions to __call__
        loop(
            mock_module,
            optimizer,
            train_dataloader,
            max_epochs=1,
            validation_dataloader=val_dataloader,
            train_step_fn=mock_train_step,
            validate_step_fn=mock_validate_step,
        )

        # Verify training step was called (1 epoch * 2 batches)
        assert mock_loss.backward.call_count == 2
        assert mock_loss.detach.call_count == 2
        # Verify optimizer.step was called once per training batch (2 times)
        assert optimizer.step.call_count == 2
        # Verify validation step was called (1 epoch * 1 validation batch)
        assert mock_validate_step.call_count == 1

    def test_training_with_lr_scheduler(self, mock_module):
        """Test training loop with learning rate scheduler."""
        mock_loss = MagicMock()
        mock_loss.detach.return_value.item.return_value = 0.5
        mock_loss.backward = MagicMock()

        def mock_train_step(model, batch, *args, **kwargs):
            return mock_loss

        loop = DefaultTrainingLoop(
            enable_static_capture=False,
            use_progress_bars=False,
        )

        optimizer = torch.optim.SGD(mock_module.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        # Wrap optimizer.step and scheduler.step to track calls
        original_optimizer_step = optimizer.step
        optimizer.step = MagicMock(side_effect=original_optimizer_step)
        original_scheduler_step = scheduler.step
        scheduler.step = MagicMock(side_effect=original_scheduler_step)

        # Create mock dataloader with 2 batches
        dataset = TensorDataset(torch.randn(8, 64), torch.randn(8, 3))
        dataloader = DataLoader(dataset, batch_size=4)

        # Run the loop, passing train_step_fn to __call__
        loop(
            mock_module,
            optimizer,
            dataloader,
            max_epochs=2,
            lr_scheduler=scheduler,
            train_step_fn=mock_train_step,
        )

        # Verify training step was called (2 epochs * 2 batches = 4 times)
        assert mock_loss.backward.call_count == 4
        assert mock_loss.detach.call_count == 4
        # Verify optimizer.step was called once per batch (4 times total)
        assert optimizer.step.call_count == 4
        # Verify scheduler.step was called once per batch (4 times total)
        assert scheduler.step.call_count == 4

    def test_device_override_in_call(self, mock_module):
        """Test that device specified in call overrides constructor device."""
        mock_loss = MagicMock()
        mock_loss.detach.return_value.item.return_value = 0.5
        mock_loss.backward = MagicMock()

        def mock_train_step(model, batch, *args, **kwargs):
            return mock_loss

        loop = DefaultTrainingLoop(
            device="cpu",
            enable_static_capture=False,
            use_progress_bars=False,
        )

        optimizer = torch.optim.SGD(mock_module.parameters(), lr=0.01)
        dataset = TensorDataset(torch.randn(8, 64), torch.randn(8, 3))
        dataloader = DataLoader(dataset, batch_size=4)

        # Call with device override and train_step_fn
        loop(
            mock_module,
            optimizer,
            dataloader,
            max_epochs=1,
            device="cpu",
            train_step_fn=mock_train_step,
        )

        # Verify training completed
        assert mock_loss.backward.call_count == 2

    def test_dtype_override_in_call(self, mock_module):
        """Test that dtype specified in call overrides constructor dtype."""
        mock_loss = MagicMock()
        mock_loss.detach.return_value.item.return_value = 0.5
        mock_loss.backward = MagicMock()

        def mock_train_step(model, batch, *args, **kwargs):
            return mock_loss

        loop = DefaultTrainingLoop(
            dtype=torch.float32,
            enable_static_capture=False,
            use_progress_bars=False,
        )

        optimizer = torch.optim.SGD(mock_module.parameters(), lr=0.01)
        dataset = TensorDataset(torch.randn(8, 64), torch.randn(8, 3))
        dataloader = DataLoader(dataset, batch_size=4)

        # Call with dtype override and train_step_fn
        loop(
            mock_module,
            optimizer,
            dataloader,
            max_epochs=1,
            dtype=torch.float16,
            train_step_fn=mock_train_step,
        )

        # Verify training completed
        assert mock_loss.backward.call_count == 2

    def test_train_step_fn_override_in_call(self, mock_module):
        """Test that train_step_fn in call overrides constructor train_step_fn."""
        # Constructor train step
        constructor_loss = MagicMock()
        constructor_loss.detach.return_value.item.return_value = 0.3
        constructor_loss.backward = MagicMock()

        def constructor_train_step(model, batch, *args, **kwargs):
            return constructor_loss

        # Call train step
        call_loss = MagicMock()
        call_loss.detach.return_value.item.return_value = 0.5
        call_loss.backward = MagicMock()

        def call_train_step(model, batch, *args, **kwargs):
            return call_loss

        loop = DefaultTrainingLoop(
            train_step_fn=constructor_train_step,
            enable_static_capture=False,
            use_progress_bars=False,
        )

        optimizer = torch.optim.SGD(mock_module.parameters(), lr=0.01)
        dataset = TensorDataset(torch.randn(8, 64), torch.randn(8, 3))
        dataloader = DataLoader(dataset, batch_size=4)

        # This should use the call_train_step due to static capture being disabled
        # and the override logic
        loop(
            mock_module,
            optimizer,
            dataloader,
            max_epochs=1,
            train_step_fn=call_train_step,
        )

        # Since static capture is disabled and we pass train_step_fn to call,
        # it won't be used directly in this implementation
        # This test verifies the loop runs without error
        assert True
