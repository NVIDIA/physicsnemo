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

"""Tests for active learning Driver class."""

from queue import Queue
from uuid import UUID

import pytest
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from physicsnemo.active_learning.config import (
    DriverConfig,
    OptimizerConfig,
    StrategiesConfig,
    TrainingConfig,
)
from physicsnemo.active_learning.driver import Driver


@pytest.fixture
def minimal_config(tmp_path) -> DriverConfig:
    """A minimal functioning configuration"""
    return DriverConfig(batch_size=4, root_log_dir=tmp_path)


def test_minimal_driver_init(
    minimal_config: DriverConfig, mock_module, mock_query_strategy
):
    """Test a minimal driver initialization"""
    # Skip training and metrology for minimal initialization
    minimal_config.skip_training = True
    minimal_config.skip_metrology = True
    minimal_config.skip_labeling = True
    strategies_config = StrategiesConfig(
        query_strategies=[mock_query_strategy],
        queue_cls=Queue,
    )
    driver = Driver(
        config=minimal_config,
        learner=mock_module,
        strategies_config=strategies_config,
    )
    assert driver.config == minimal_config
    assert driver.strategies_config == strategies_config
    assert hasattr(driver, "logger")
    assert driver.active_learning_step_idx == 0
    # assert strategies were attached
    mock_query_strategy.attach.assert_called_once_with(driver)
    # check that queue was initialized
    assert isinstance(driver.query_queue, Queue)
    # make sure run ID is assigned something valid
    assert isinstance(driver.run_id, str)
    assert UUID(driver.run_id, version=4)
    assert isinstance(driver.short_run_id, str)
    # make sure the log file exists
    assert (driver.log_dir / f"{driver.run_id}.log").exists()


def test_driver_configure_optimizer(
    minimal_config: DriverConfig,
    mock_module,
    mock_query_strategy,
    mock_data_pool,
    mock_training_loop,
):
    """Test the driver's optimizer configuration"""
    # Skip metrology and labeling for this test
    minimal_config.skip_metrology = True
    minimal_config.skip_labeling = True
    strategies_config = StrategiesConfig(
        query_strategies=[mock_query_strategy],
        queue_cls=Queue,
    )
    training_config = TrainingConfig(
        train_datapool=mock_data_pool,
        train_loop_fn=mock_training_loop,
        max_training_epochs=10,
    )
    driver = Driver(
        config=minimal_config,
        learner=mock_module,
        strategies_config=strategies_config,
        training_config=training_config,
    )
    driver.configure_optimizer()
    assert driver.optimizer is not None
    assert driver.is_optimizer_configured

    # now try with a non-default optimizer and scheduler
    optimizer_config = OptimizerConfig(
        optimizer_cls=SGD,
        optimizer_kwargs={"lr": 1e-3},
        scheduler_cls=StepLR,
        scheduler_kwargs={"step_size": 10},
    )
    training_config_custom = TrainingConfig(
        train_datapool=mock_data_pool,
        train_loop_fn=mock_training_loop,
        max_training_epochs=10,
        optimizer_config=optimizer_config,
    )
    new_driver = Driver(
        config=minimal_config,
        learner=mock_module,
        strategies_config=strategies_config,
        training_config=training_config_custom,
    )
    new_driver.configure_optimizer()
    assert new_driver.is_optimizer_configured
    assert new_driver.is_lr_scheduler_configured
    # this should work without fail
    new_driver.lr_scheduler.step()


def test_construct_loader(
    minimal_config: DriverConfig, mock_data_pool, mock_module, mock_query_strategy
):
    """Test the driver's loader construction"""
    # Skip all phases for this construction test
    minimal_config.skip_training = True
    minimal_config.skip_metrology = True
    minimal_config.skip_labeling = True
    strategies_config = StrategiesConfig(
        query_strategies=[mock_query_strategy],
        queue_cls=Queue,
    )
    driver = Driver(
        config=minimal_config,
        learner=mock_module,
        strategies_config=strategies_config,
    )
    loader = driver._construct_dataloader(mock_data_pool, shuffle=False)
    assert isinstance(loader, DataLoader)


def test_minimal_learning_step(
    minimal_config: DriverConfig,
    mock_module,
    mock_query_strategy,
    mock_training_loop,
    mock_data_pool,
):
    """Test the driver's learning step"""
    # disable the extra steps
    minimal_config.skip_labeling = True
    minimal_config.skip_metrology = True
    strategies_config = StrategiesConfig(
        query_strategies=[mock_query_strategy],
        queue_cls=Queue,
    )

    # Test that without skip_training=True and no training_config, initialization fails
    with pytest.raises(ValueError, match="`training_config` must be provided"):
        driver = Driver(
            config=minimal_config,
            learner=mock_module,
            strategies_config=strategies_config,
        )

    # Create training config
    training_config = TrainingConfig(
        train_datapool=mock_data_pool,
        train_loop_fn=mock_training_loop,
        max_training_epochs=10,
    )
    driver = Driver(
        config=minimal_config,
        learner=mock_module,
        strategies_config=strategies_config,
        training_config=training_config,
    )
    # should fail without passing a train_step_fn when not using a learner
    with pytest.raises(ValueError, match="`train_step_fn` must be provided"):
        driver.active_learning_step()
    # this should work, despite that the train_step_fn is a dummy one
    driver.active_learning_step(train_step_fn=lambda x: None)
    assert driver.active_learning_step_idx == 1


def test_query_label_pipeline(
    minimal_config: DriverConfig,
    mock_module,
    mock_query_strategy,
    mock_data_pool,
    mock_label_strategy,
    mock_queue,
    mock_training_loop,
):
    """Test the driver's query and label pipeline without training"""
    minimal_config.skip_training = True
    minimal_config.skip_metrology = True
    strategies_config = StrategiesConfig(
        query_strategies=[mock_query_strategy],
        queue_cls=Queue,
        label_strategy=mock_label_strategy,
    )
    # Create a minimal training config with just train_datapool for labeling
    training_config = TrainingConfig(
        train_datapool=mock_data_pool,
        train_loop_fn=mock_training_loop,
        max_training_epochs=10,
    )
    driver = Driver(
        config=minimal_config,
        learner=mock_module,
        strategies_config=strategies_config,
        training_config=training_config,
    )
    # this checks to make sure the query strategy was called
    driver.active_learning_step()
    assert driver.active_learning_step_idx == 1
    driver.query_strategies[0].assert_called_with(driver.query_queue)
    # do it again, but this time we subsitute the queue with a populated one
    driver.query_queue = mock_queue
    driver.label_queue.put("literally anything")
    driver.active_learning_step()
    assert driver.active_learning_step_idx == 2
    driver.label_strategy.assert_called_with(driver.query_queue, driver.label_queue)
    # check that the data pool was theoretically updated
    assert len(driver.train_datapool) == 1
    mock_data_pool.append.assert_called_once()


def test_train_metrology_pipeline(
    minimal_config: DriverConfig,
    mock_module,
    mock_query_strategy,
    mock_data_pool,
    mock_metrology_strategy,
    mock_training_loop,
):
    """Test the driver's train, metrology, and query pipeline"""
    minimal_config.skip_labeling = True
    # do not do any batching
    minimal_config.collate_fn = lambda x: x
    strategies_config = StrategiesConfig(
        query_strategies=[mock_query_strategy],
        queue_cls=Queue,
        metrology_strategies=[mock_metrology_strategy],
    )
    training_config = TrainingConfig(
        train_datapool=mock_data_pool,
        val_datapool=mock_data_pool,
        train_loop_fn=mock_training_loop,
        max_training_epochs=10,
    )
    driver = Driver(
        config=minimal_config,
        learner=mock_module,
        strategies_config=strategies_config,
        training_config=training_config,
    )
    # patch the dataloader to return a dummy batch
    driver.active_learning_step(
        train_step_fn=lambda x: None, validate_step_fn=lambda x: None
    )
    assert driver.active_learning_step_idx == 1
    driver.query_strategies[0].assert_called_with(driver.query_queue)
    assert driver.train_loop_fn.call_count == 1
    driver.metrology_strategies[0].assert_called_once()


def test_full_step_pipeline(
    minimal_config: DriverConfig,
    mock_module,
    mock_query_strategy,
    mock_data_pool,
    mock_label_strategy,
    mock_metrology_strategy,
    mock_training_loop,
):
    """Test the fully specified active learning pipeline"""
    minimal_config.collate_fn = lambda x: x
    strategies_config = StrategiesConfig(
        query_strategies=[mock_query_strategy],
        queue_cls=Queue,
        label_strategy=mock_label_strategy,
        metrology_strategies=[mock_metrology_strategy],
    )
    training_config = TrainingConfig(
        train_datapool=mock_data_pool,
        val_datapool=mock_data_pool,
        train_loop_fn=mock_training_loop,
        max_training_epochs=10,
    )
    driver = Driver(
        config=minimal_config,
        learner=mock_module,
        strategies_config=strategies_config,
        training_config=training_config,
    )
    driver.query_queue.put("something from querying")
    driver.label_queue.put("something from labeling")
    driver.active_learning_step(
        train_step_fn=lambda x: None, validate_step_fn=lambda x: None
    )
    # every component should be called at least once
    assert driver.active_learning_step_idx == 1
    driver.query_strategies[0].assert_called_with(driver.query_queue)
    driver.metrology_strategies[0].assert_called_once()
    # make sure the label strategy was called and new data added
    driver.label_strategy.assert_called_with(driver.query_queue, driver.label_queue)
    assert driver.train_datapool.append.call_count == 1


def test_run_loop(
    minimal_config: DriverConfig,
    mock_module,
    mock_query_strategy,
    mock_data_pool,
    mock_label_strategy,
    mock_metrology_strategy,
    mock_training_loop,
):
    """Test a minimal configuration running the loop a few times"""
    minimal_config.max_active_learning_steps = 5
    minimal_config.collate_fn = lambda x: x
    minimal_config.fine_tuning_lr = 50.0
    strategies_config = StrategiesConfig(
        query_strategies=[mock_query_strategy],
        queue_cls=Queue,
        label_strategy=mock_label_strategy,
        metrology_strategies=[mock_metrology_strategy],
    )
    training_config = TrainingConfig(
        train_datapool=mock_data_pool,
        val_datapool=mock_data_pool,
        train_loop_fn=mock_training_loop,
        max_training_epochs=10,
    )
    driver = Driver(
        config=minimal_config,
        learner=mock_module,
        strategies_config=strategies_config,
        training_config=training_config,
    )
    # artificially populate the queues
    driver.query_queue.put("something from querying")
    driver.label_queue.put("something from labeling")
    driver(
        train_step_fn=lambda x: None,
        validate_step_fn=lambda x: None,
    )
    assert driver.active_learning_step_idx == 5
    assert driver.label_strategy.call_count == 5
    assert driver.metrology_strategies[0].call_count == 5
    assert driver.query_strategies[0].call_count == 5
    assert driver.train_loop_fn.call_count == 5
    assert driver.optimizer.param_groups[0]["lr"] == 50.0
