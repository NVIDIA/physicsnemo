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

"""Tests for active learning checkpoint functionality."""

from __future__ import annotations

import queue
import shutil
from typing import Any

import pytest
import torch

from physicsnemo.active_learning import protocols as p
from physicsnemo.active_learning.config import (
    DriverConfig,
    OptimizerConfig,
    StrategiesConfig,
    TrainingConfig,
)
from physicsnemo.active_learning.driver import ActiveLearningCheckpoint, Driver

from .conftest import MockDataStructure, MockModule


class SimpleQueue:
    """Simple queue implementation with serialization support for testing."""

    def __init__(self):
        """Initialize empty queue."""
        self._items = []

    def put(self, item: Any) -> None:
        """Add item to queue."""
        self._items.append(item)

    def get(self) -> Any:
        """Remove and return item from queue."""
        return self._items.pop(0) if self._items else None

    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._items) == 0

    def to_list(self) -> list[Any]:
        """Serialize queue to list."""
        return self._items.copy()

    def from_list(self, items: list[Any]) -> None:
        """Restore queue from list."""
        self._items = items.copy()


@pytest.fixture
def simple_queue():
    """Fixture for a simple queue with serialization support."""
    return SimpleQueue


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Fixture for temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    yield checkpoint_dir
    # Cleanup
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)


@pytest.fixture
def driver_config(temp_checkpoint_dir):
    """Fixture for basic driver config with checkpointing enabled."""
    return DriverConfig(
        batch_size=16,
        max_active_learning_steps=2,
        checkpoint_interval=1,
        checkpoint_on_query=True,
        skip_training=True,  # Skip training for basic checkpoint tests
        skip_labeling=True,  # Skip labeling for basic checkpoint tests
        root_log_dir=temp_checkpoint_dir.parent,
        device=torch.device("cpu"),
    )


@pytest.fixture
def strategies_config(mock_query_strategy, simple_queue):
    """Fixture for strategies config with simple queue."""
    return StrategiesConfig(
        query_strategies=[mock_query_strategy],
        queue_cls=simple_queue,
    )


@pytest.fixture
def training_config(mock_data_pool, mock_training_loop):
    """Fixture for training config."""
    return TrainingConfig(
        train_datapool=mock_data_pool,
        max_training_epochs=2,
        optimizer_config=OptimizerConfig(
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01},
        ),
        train_loop_fn=mock_training_loop,
    )


@pytest.mark.dependency()
def test_checkpoint_basic_save(
    driver_config, mock_module, strategies_config, temp_checkpoint_dir
):
    """Test basic checkpoint saving functionality."""
    driver = Driver(
        config=driver_config,
        learner=mock_module,
        strategies_config=strategies_config,
    )

    # Manually set a phase
    driver.current_phase = p.ActiveLearningPhase.QUERY

    # Save checkpoint
    checkpoint_path = temp_checkpoint_dir / "test_checkpoint"
    driver.save_checkpoint(path=checkpoint_path)

    # Verify checkpoint files exist
    assert checkpoint_path.exists()
    assert (checkpoint_path / "checkpoint.pt").exists()
    assert (checkpoint_path / "MockModule.mdlus").exists()

    # Verify last_checkpoint property
    assert driver.last_checkpoint == checkpoint_path


@pytest.mark.dependency(depends=["test_checkpoint_basic_save"])
def test_checkpoint_contains_correct_metadata(
    driver_config, mock_module, strategies_config, temp_checkpoint_dir
):
    """Test that checkpoint contains all required metadata."""
    driver = Driver(
        config=driver_config,
        learner=mock_module,
        strategies_config=strategies_config,
    )

    driver.active_learning_step_idx = 5
    driver.current_phase = p.ActiveLearningPhase.QUERY

    checkpoint_path = temp_checkpoint_dir / "metadata_test"
    driver.save_checkpoint(path=checkpoint_path)

    # Load and verify checkpoint
    checkpoint_dict = torch.load(checkpoint_path / "checkpoint.pt", weights_only=False)
    checkpoint: ActiveLearningCheckpoint = checkpoint_dict["checkpoint"]

    assert checkpoint.active_learning_step_idx == 5
    assert checkpoint.active_learning_phase == p.ActiveLearningPhase.QUERY
    assert checkpoint.driver_config is not None
    assert checkpoint.strategies_config is not None


@pytest.mark.dependency(depends=["test_checkpoint_basic_save"])
def test_checkpoint_with_training_state(
    mock_module, strategies_config, training_config, temp_checkpoint_dir
):
    """Test that training state is saved separately via training loop."""
    from physicsnemo.active_learning.loop import DefaultTrainingLoop

    # Create config without skip_training for this test
    config = DriverConfig(
        batch_size=16,
        max_active_learning_steps=2,
        checkpoint_interval=1,
        checkpoint_on_query=True,
        skip_labeling=True,  # Skip labeling since we don't have a label strategy
        root_log_dir=temp_checkpoint_dir.parent,
        device=torch.device("cpu"),
    )

    driver = Driver(
        config=config,
        learner=mock_module,
        strategies_config=strategies_config,
        training_config=training_config,
    )

    # Configure optimizer
    driver.configure_optimizer()
    assert driver.optimizer is not None

    checkpoint_path = temp_checkpoint_dir / "training_state_test"

    # Use training loop to save training state
    training_loop = DefaultTrainingLoop()
    training_loop.save_training_checkpoint(
        checkpoint_dir=checkpoint_path,
        model=driver.learner,
        optimizer=driver.optimizer,
        lr_scheduler=driver.lr_scheduler if hasattr(driver, "lr_scheduler") else None,
        training_epoch=5,
    )

    # Verify training state file exists and contains optimizer state
    assert (checkpoint_path / "training_state.pt").exists()
    training_state = torch.load(checkpoint_path / "training_state.pt")
    assert "optimizer_state" in training_state
    assert "param_groups" in training_state["optimizer_state"]
    assert training_state["training_epoch"] == 5


@pytest.mark.dependency(depends=["test_checkpoint_basic_save"])
def test_queue_serialization_with_to_list(
    driver_config, mock_module, strategies_config, temp_checkpoint_dir
):
    """Test queue serialization using to_list/from_list methods."""
    driver = Driver(
        config=driver_config,
        learner=mock_module,
        strategies_config=strategies_config,
    )

    # Add items to queues
    test_data = MockDataStructure(inputs=torch.randn(16, 64))
    driver.query_queue.put(test_data)
    driver.label_queue.put(test_data)

    checkpoint_path = temp_checkpoint_dir / "queue_test"
    driver.save_checkpoint(path=checkpoint_path)

    # Load and verify queue files were created
    checkpoint_dict = torch.load(checkpoint_path / "checkpoint.pt", weights_only=False)
    checkpoint: ActiveLearningCheckpoint = checkpoint_dict["checkpoint"]

    assert checkpoint.has_query_queue is True
    assert checkpoint.has_label_queue is True
    assert (checkpoint_path / "query_queue.pt").exists()
    assert (checkpoint_path / "label_queue.pt").exists()

    # Verify queue contents
    query_queue_data = torch.load(
        checkpoint_path / "query_queue.pt", weights_only=False
    )
    assert query_queue_data["type"] == "list"
    assert len(query_queue_data["data"]) == 1


@pytest.mark.dependency(depends=["test_checkpoint_basic_save"])
def test_queue_serialization_fallback(
    driver_config, mock_module, mock_query_strategy, temp_checkpoint_dir
):
    """Test queue serialization handles unpicklable queues gracefully."""
    # Use standard library queue without to_list method
    # This queue cannot be pickled due to thread locks
    strategies_config_stdlib = StrategiesConfig(
        query_strategies=[mock_query_strategy],
        queue_cls=queue.Queue,
    )

    driver = Driver(
        config=driver_config,
        learner=mock_module,
        strategies_config=strategies_config_stdlib,
    )

    # Add item to queue
    test_data = MockDataStructure(inputs=torch.randn(16, 64))
    driver.query_queue.put(test_data)

    checkpoint_path = temp_checkpoint_dir / "queue_fallback_test"
    driver.save_checkpoint(path=checkpoint_path)

    # Verify checkpoint was created
    assert (checkpoint_path / "checkpoint.pt").exists()

    # Load and verify queue serialization failed (unpicklable)
    checkpoint_dict = torch.load(checkpoint_path / "checkpoint.pt", weights_only=False)
    checkpoint: ActiveLearningCheckpoint = checkpoint_dict["checkpoint"]

    # stdlib queue.Queue cannot be pickled, so has_query_queue should be False
    assert checkpoint.has_query_queue is False
    # Queue file should not exist
    assert not (checkpoint_path / "query_queue.pt").exists()


@pytest.mark.dependency(depends=["test_checkpoint_basic_save"])
def test_checkpoint_load_basic(
    driver_config, mock_module, strategies_config, temp_checkpoint_dir
):
    """Test basic checkpoint loading functionality."""
    # Create and save a checkpoint
    driver = Driver(
        config=driver_config,
        learner=mock_module,
        strategies_config=strategies_config,
    )

    driver.active_learning_step_idx = 3
    driver.current_phase = p.ActiveLearningPhase.QUERY

    checkpoint_path = temp_checkpoint_dir / "load_test"
    driver.save_checkpoint(path=checkpoint_path)

    # Load checkpoint
    loaded_driver = Driver.load_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=MockModule(),
    )

    # Verify state was restored
    assert loaded_driver.active_learning_step_idx == 3
    assert loaded_driver.current_phase == p.ActiveLearningPhase.QUERY
    assert loaded_driver.last_checkpoint == checkpoint_path


@pytest.mark.dependency(depends=["test_checkpoint_basic_save"])
def test_checkpoint_load_with_datapools(
    mock_module,
    strategies_config,
    training_config,
    mock_data_pool,
    temp_checkpoint_dir,
):
    """Test checkpoint loading with datapool restoration."""
    # Create config without skip_training for this test
    config = DriverConfig(
        batch_size=16,
        max_active_learning_steps=2,
        checkpoint_interval=1,
        checkpoint_on_query=True,
        skip_labeling=True,
        root_log_dir=temp_checkpoint_dir.parent,
        device=torch.device("cpu"),
    )

    driver = Driver(
        config=config,
        learner=mock_module,
        strategies_config=strategies_config,
        training_config=training_config,
    )

    checkpoint_path = temp_checkpoint_dir / "datapool_test"
    driver.save_checkpoint(path=checkpoint_path)

    # Load with datapools provided
    loaded_driver = Driver.load_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=MockModule(),
        train_datapool=mock_data_pool,
        val_datapool=mock_data_pool,
    )

    # Verify datapools were set
    assert loaded_driver.train_datapool is not None
    assert loaded_driver.val_datapool is not None


@pytest.mark.dependency(depends=["test_checkpoint_basic_save"])
def test_training_loop_restores_optimizer(
    mock_module,
    strategies_config,
    training_config,
    mock_data_pool,
    temp_checkpoint_dir,
):
    """Test that training loop can restore optimizer state from training_state.pt."""
    from physicsnemo.active_learning.loop import DefaultTrainingLoop

    # Create config without skip_training for this test
    config = DriverConfig(
        batch_size=16,
        max_active_learning_steps=2,
        checkpoint_interval=1,
        checkpoint_on_query=True,
        skip_labeling=True,
        root_log_dir=temp_checkpoint_dir.parent,
        device=torch.device("cpu"),
    )

    driver = Driver(
        config=config,
        learner=mock_module,
        strategies_config=strategies_config,
        training_config=training_config,
    )

    driver.configure_optimizer()
    # Modify optimizer state
    for param_group in driver.optimizer.param_groups:
        param_group["lr"] = 0.1234

    checkpoint_path = temp_checkpoint_dir / "training_state_restore_test"

    # Use training loop to save training state
    training_loop = DefaultTrainingLoop()
    training_loop.save_training_checkpoint(
        checkpoint_dir=checkpoint_path,
        model=driver.learner,
        optimizer=driver.optimizer,
        lr_scheduler=driver.lr_scheduler if hasattr(driver, "lr_scheduler") else None,
        training_epoch=3,
    )

    # Create new driver and optimizer
    new_module = MockModule()
    new_driver = Driver(
        config=config,
        learner=new_module,
        strategies_config=strategies_config,
        training_config=training_config,
    )
    new_driver.configure_optimizer()

    # Load training state using training loop
    epoch = DefaultTrainingLoop.load_training_checkpoint(
        checkpoint_dir=checkpoint_path,
        model=new_driver.learner,
        optimizer=new_driver.optimizer,
        lr_scheduler=new_driver.lr_scheduler
        if hasattr(new_driver, "lr_scheduler")
        else None,
    )

    # Verify optimizer state was restored
    assert new_driver.optimizer.param_groups[0]["lr"] == pytest.approx(0.1234)
    assert epoch == 3


@pytest.mark.dependency(depends=["test_checkpoint_basic_save"])
def test_checkpoint_load_restores_queues(
    driver_config, mock_module, strategies_config, temp_checkpoint_dir
):
    """Test that checkpoint loading restores queue contents."""
    driver = Driver(
        config=driver_config,
        learner=mock_module,
        strategies_config=strategies_config,
    )

    # Add items to queue
    test_data = MockDataStructure(inputs=torch.randn(16, 64))
    driver.query_queue.put(test_data)

    checkpoint_path = temp_checkpoint_dir / "queue_restore_test"
    driver.save_checkpoint(path=checkpoint_path)

    # Load checkpoint
    loaded_driver = Driver.load_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=MockModule(),
    )

    # Verify queue was restored
    assert not loaded_driver.query_queue.empty()
    restored_item = loaded_driver.query_queue.get()
    assert isinstance(restored_item, MockDataStructure)
    assert restored_item.inputs.shape == test_data.inputs.shape


@pytest.mark.dependency(depends=["test_checkpoint_basic_save"])
def test_checkpoint_interval_controls_saving(
    temp_checkpoint_dir, mock_module, mock_query_strategy, simple_queue
):
    """Test that checkpoint_interval controls when checkpoints are saved."""
    # Set checkpoint_interval to 2
    config = DriverConfig(
        batch_size=16,
        max_active_learning_steps=5,
        checkpoint_interval=2,
        checkpoint_on_query=True,
        skip_training=True,
        skip_labeling=True,
        root_log_dir=temp_checkpoint_dir.parent,
        device=torch.device("cpu"),
    )

    strategies_config = StrategiesConfig(
        query_strategies=[mock_query_strategy],
        queue_cls=simple_queue,
    )

    driver = Driver(
        config=config,
        learner=mock_module,
        strategies_config=strategies_config,
    )

    # Test at step 0 (should checkpoint)
    driver.active_learning_step_idx = 0
    assert driver._should_checkpoint_at_step()

    # Test at step 1 (should NOT checkpoint)
    driver.active_learning_step_idx = 1
    assert not driver._should_checkpoint_at_step()

    # Test at step 2 (should checkpoint)
    driver.active_learning_step_idx = 2
    assert driver._should_checkpoint_at_step()

    # Test at step 4 (should checkpoint)
    driver.active_learning_step_idx = 4
    assert driver._should_checkpoint_at_step()


@pytest.mark.dependency(depends=["test_checkpoint_basic_save"])
def test_checkpoint_interval_zero_disables_checkpointing(
    temp_checkpoint_dir, mock_module, mock_query_strategy, simple_queue
):
    """Test that checkpoint_interval=0 disables checkpointing."""
    config = DriverConfig(
        batch_size=16,
        max_active_learning_steps=5,
        checkpoint_interval=0,  # Disabled
        checkpoint_on_query=True,
        skip_training=True,
        skip_labeling=True,
        root_log_dir=temp_checkpoint_dir.parent,
        device=torch.device("cpu"),
    )

    strategies_config = StrategiesConfig(
        query_strategies=[mock_query_strategy],
        queue_cls=simple_queue,
    )

    driver = Driver(
        config=config,
        learner=mock_module,
        strategies_config=strategies_config,
    )

    # Should never checkpoint when interval is 0
    for step in range(5):
        driver.active_learning_step_idx = step
        assert not driver._should_checkpoint_at_step()


@pytest.mark.dependency(depends=["test_checkpoint_basic_save"])
def test_checkpoint_with_training_epoch(
    mock_module,
    strategies_config,
    training_config,
    mock_data_pool,
    temp_checkpoint_dir,
):
    """Test checkpoint saving with training epoch information."""
    # Create config without skip_training for this test
    config = DriverConfig(
        batch_size=16,
        max_active_learning_steps=2,
        checkpoint_interval=1,
        checkpoint_on_query=True,
        skip_labeling=True,
        root_log_dir=temp_checkpoint_dir.parent,
        device=torch.device("cpu"),
    )

    driver = Driver(
        config=config,
        learner=mock_module,
        strategies_config=strategies_config,
        training_config=training_config,
    )

    driver.current_phase = p.ActiveLearningPhase.TRAINING

    # Save checkpoint with epoch number
    checkpoint_path = temp_checkpoint_dir / "epoch_test"
    driver.save_checkpoint(path=checkpoint_path, training_epoch=5)

    # Verify epoch is saved
    checkpoint_dict = torch.load(checkpoint_path / "checkpoint.pt", weights_only=False)
    assert "training_epoch" in checkpoint_dict
    assert checkpoint_dict["training_epoch"] == 5


@pytest.mark.dependency(depends=["test_checkpoint_basic_save"])
def test_checkpoint_auto_path_generation(
    driver_config, mock_module, strategies_config, temp_checkpoint_dir
):
    """Test that checkpoints are saved with auto-generated paths."""
    driver = Driver(
        config=driver_config,
        learner=mock_module,
        strategies_config=strategies_config,
    )

    driver.active_learning_step_idx = 2
    driver.current_phase = p.ActiveLearningPhase.QUERY

    # Save without specifying path
    driver.save_checkpoint()

    # Verify path was auto-generated
    expected_path = driver.log_dir / "checkpoints" / "step_2" / "query"
    assert expected_path.exists()
    assert (expected_path / "checkpoint.pt").exists()


@pytest.mark.dependency(depends=["test_checkpoint_basic_save"])
def test_checkpoint_preserves_model_weights(
    driver_config, mock_module, strategies_config, temp_checkpoint_dir
):
    """Test that model weights are correctly saved and loaded."""
    driver = Driver(
        config=driver_config,
        learner=mock_module,
        strategies_config=strategies_config,
    )

    # Get initial weights
    initial_weights = {
        name: param.clone() for name, param in driver.learner.named_parameters()
    }

    checkpoint_path = temp_checkpoint_dir / "weights_test"
    driver.save_checkpoint(path=checkpoint_path)

    # Create new module and load weights
    new_module = MockModule()
    loaded_driver = Driver.load_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=new_module,
    )

    # Verify weights match
    for name, param in loaded_driver.learner.named_parameters():
        assert torch.allclose(param, initial_weights[name])


def test_checkpoint_phase_specific_flags(
    temp_checkpoint_dir, mock_module, mock_query_strategy, simple_queue
):
    """Test that phase-specific checkpoint flags are respected."""
    config = DriverConfig(
        batch_size=16,
        max_active_learning_steps=2,
        checkpoint_interval=1,
        checkpoint_on_training=True,
        checkpoint_on_metrology=False,
        checkpoint_on_query=True,
        checkpoint_on_labeling=False,
        skip_training=True,
        skip_labeling=True,
        root_log_dir=temp_checkpoint_dir.parent,
        device=torch.device("cpu"),
    )

    strategies_config = StrategiesConfig(
        query_strategies=[mock_query_strategy],
        queue_cls=simple_queue,
    )

    driver = Driver(
        config=config,
        learner=mock_module,
        strategies_config=strategies_config,
    )

    # Verify flags are set correctly
    assert driver.config.checkpoint_on_training is True
    assert driver.config.checkpoint_on_metrology is False
    assert driver.config.checkpoint_on_query is True
    assert driver.config.checkpoint_on_labeling is False


# ============================================================================
# Phase Resumption Tests
# ============================================================================


def test_get_phase_index(driver_config, mock_module, strategies_config):
    """Test _get_phase_index helper method."""
    driver = Driver(
        config=driver_config,
        learner=mock_module,
        strategies_config=strategies_config,
    )

    # Test each phase
    assert driver._get_phase_index(None) == 0
    assert driver._get_phase_index(p.ActiveLearningPhase.TRAINING) == 0
    assert driver._get_phase_index(p.ActiveLearningPhase.METROLOGY) == 1
    assert driver._get_phase_index(p.ActiveLearningPhase.QUERY) == 2
    assert driver._get_phase_index(p.ActiveLearningPhase.LABELING) == 3


def test_build_phase_queue_from_fresh_start(
    driver_config, mock_module, strategies_config
):
    """Test phase queue includes only non-skipped phases when current_phase is None."""
    driver = Driver(
        config=driver_config,  # skip_training=True, skip_labeling=True, skip_metrology=False
        learner=mock_module,
        strategies_config=strategies_config,
    )

    # current_phase is None (fresh start)
    assert driver.current_phase is None

    # Build phase queue
    phase_queue = driver._build_phase_queue(None, None, (), {})

    # Verify: metrology and query phases (training and labeling skipped)
    assert len(phase_queue) == 2  # metrology + query


def test_build_phase_queue_from_query_phase(
    driver_config, mock_module, strategies_config
):
    """Test phase queue starts from query when current_phase=QUERY."""
    driver = Driver(
        config=driver_config,
        learner=mock_module,
        strategies_config=strategies_config,
    )

    # Set current_phase to QUERY (as if loaded from checkpoint)
    driver.current_phase = p.ActiveLearningPhase.QUERY

    # Build phase queue
    phase_queue = driver._build_phase_queue(None, None, (), {})

    # With skip_training=True, skip_labeling=True
    # Queue should include: [query] (labeling skipped by config)
    assert len(phase_queue) == 1


def test_resume_from_query_phase_skips_earlier_phases(
    driver_config, mock_module, strategies_config, temp_checkpoint_dir
):
    """Test that resuming from query phase skips training and metrology."""
    driver = Driver(
        config=driver_config,
        learner=mock_module,
        strategies_config=strategies_config,
    )

    # Create checkpoint at query phase
    driver.active_learning_step_idx = 1
    driver.current_phase = p.ActiveLearningPhase.QUERY
    driver.query_queue.put(MockDataStructure(inputs=torch.randn(16, 64)))

    checkpoint_path = temp_checkpoint_dir / "resume_query_test"
    driver.save_checkpoint(path=checkpoint_path)

    # Load checkpoint
    loaded_driver = Driver.load_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=MockModule(),
    )

    # Track which phases execute
    executed_phases = []

    loaded_driver._training_phase = lambda *a, **k: executed_phases.append("training")
    loaded_driver._metrology_phase = lambda *a, **k: executed_phases.append("metrology")
    loaded_driver._query_phase = lambda *a, **k: executed_phases.append("query")
    loaded_driver._labeling_phase = lambda *a, **k: executed_phases.append("labeling")

    # Execute one AL step
    loaded_driver.active_learning_step()

    # Verify: only query executed (training/metrology skipped, labeling skipped by config)
    assert "training" not in executed_phases
    assert "metrology" not in executed_phases
    assert "query" in executed_phases
    assert "labeling" not in executed_phases

    # Verify: current_phase reset after step completion
    assert loaded_driver.current_phase is None
    assert loaded_driver.active_learning_step_idx == 2


def test_current_phase_resets_after_step_completion(
    driver_config, mock_module, strategies_config, temp_checkpoint_dir
):
    """Test that current_phase is reset to None after completing an AL step."""
    driver = Driver(
        config=driver_config,
        learner=mock_module,
        strategies_config=strategies_config,
    )

    # Save checkpoint at query phase
    driver.active_learning_step_idx = 0
    driver.current_phase = p.ActiveLearningPhase.QUERY
    checkpoint_path = temp_checkpoint_dir / "phase_reset_test"
    driver.save_checkpoint(path=checkpoint_path)

    # Load checkpoint
    loaded_driver = Driver.load_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=MockModule(),
    )

    # Verify loaded state
    assert loaded_driver.current_phase == p.ActiveLearningPhase.QUERY

    # Mock all phase methods as no-ops
    loaded_driver._training_phase = lambda *a, **k: None
    loaded_driver._metrology_phase = lambda *a, **k: None
    loaded_driver._query_phase = lambda *a, **k: None
    loaded_driver._labeling_phase = lambda *a, **k: None

    # Execute one AL step
    loaded_driver.active_learning_step()

    # Verify: current_phase reset to None after step completion
    assert loaded_driver.current_phase is None
    assert loaded_driver.active_learning_step_idx == 1


def test_resume_continues_to_next_al_step(
    driver_config, mock_module, strategies_config, temp_checkpoint_dir
):
    """Test that after resuming and completing one step, next step starts fresh."""
    driver = Driver(
        config=driver_config,
        learner=mock_module,
        strategies_config=strategies_config,
    )

    # Checkpoint at step 1, query phase
    driver.active_learning_step_idx = 1
    driver.current_phase = p.ActiveLearningPhase.QUERY

    checkpoint_path = temp_checkpoint_dir / "multi_step_test"
    driver.save_checkpoint(path=checkpoint_path)

    # Load checkpoint
    loaded_driver = Driver.load_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=MockModule(),
    )

    all_executions = []

    def track_query():
        all_executions.append(f"step_{loaded_driver.active_learning_step_idx}_query")

    loaded_driver._training_phase = lambda *a, **k: all_executions.append(
        f"step_{loaded_driver.active_learning_step_idx}_training"
    )
    loaded_driver._metrology_phase = lambda *a, **k: all_executions.append(
        f"step_{loaded_driver.active_learning_step_idx}_metrology"
    )
    loaded_driver._query_phase = lambda *a, **k: track_query()
    loaded_driver._labeling_phase = lambda *a, **k: all_executions.append(
        f"step_{loaded_driver.active_learning_step_idx}_labeling"
    )

    # Execute step 1 (resume from query)
    loaded_driver.active_learning_step()

    # Verify step 1 completed, current_phase reset
    assert loaded_driver.current_phase is None
    assert loaded_driver.active_learning_step_idx == 2

    # Execute step 2 (fresh start, should build full queue)
    loaded_driver.active_learning_step()

    # Verify execution pattern:
    # Step 1: query only (resumed from QUERY, training/labeling skipped by config)
    # Step 2: metrology + query (fresh start, current_phase=None, training/labeling skipped by config)
    assert all_executions == ["step_1_query", "step_2_metrology", "step_2_query"]
    assert loaded_driver.current_phase is None
    assert loaded_driver.active_learning_step_idx == 3
