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

"""Tests for active learning configuration classes."""

import json
from math import nan
from queue import Queue

import pytest
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR

from physicsnemo.active_learning.config import (
    DriverConfig,
    OptimizerConfig,
    StrategiesConfig,
    TrainingConfig,
)


class TestOptimizerConfig:
    """Tests for OptimizerConfig."""

    def test_default_config(self):
        """Test default optimizer configuration."""
        config = OptimizerConfig()
        assert config.optimizer_cls == AdamW
        assert config.optimizer_kwargs == {"lr": 1e-4}
        assert config.scheduler_cls is None
        assert config.scheduler_kwargs == {}

    def test_custom_optimizer(self):
        """Test custom optimizer configuration."""
        config = OptimizerConfig(
            optimizer_cls=SGD,
            optimizer_kwargs={"lr": 0.01, "momentum": 0.9},
        )
        assert config.optimizer_cls == SGD
        assert config.optimizer_kwargs["lr"] == 0.01
        assert config.optimizer_kwargs["momentum"] == 0.9

    def test_with_scheduler(self):
        """Test optimizer config with scheduler."""
        config = OptimizerConfig(
            scheduler_cls=StepLR,
            scheduler_kwargs={"step_size": 10, "gamma": 0.1},
        )
        assert config.scheduler_cls == StepLR
        assert config.scheduler_kwargs["step_size"] == 10

    def test_invalid_learning_rate(self):
        """Test that invalid learning rates are rejected."""
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            OptimizerConfig(optimizer_kwargs={"lr": -0.01})

        with pytest.raises(ValueError, match="Learning rate must be positive"):
            OptimizerConfig(optimizer_kwargs={"lr": 0})

    def test_scheduler_kwargs_without_scheduler(self):
        """Test that scheduler_kwargs without scheduler_cls raises error."""
        with pytest.raises(
            ValueError, match="scheduler_kwargs provided but scheduler_cls is None"
        ):
            OptimizerConfig(scheduler_kwargs={"step_size": 10})

    def test_to_dict_from_dict_round_trip(self):
        """Test that OptimizerConfig can be serialized and deserialized."""
        config = OptimizerConfig(
            optimizer_cls=SGD,
            optimizer_kwargs={"lr": 0.01, "momentum": 0.9},
            scheduler_cls=StepLR,
            scheduler_kwargs={"step_size": 10, "gamma": 0.1},
        )

        # Serialize
        config_dict = config.to_dict()

        # Deserialize
        restored_config = OptimizerConfig.from_dict(config_dict)

        # Verify
        assert restored_config.optimizer_cls == SGD
        assert restored_config.optimizer_kwargs == {"lr": 0.01, "momentum": 0.9}
        assert restored_config.scheduler_cls == StepLR
        assert restored_config.scheduler_kwargs == {"step_size": 10, "gamma": 0.1}

    def test_to_dict_from_dict_no_scheduler(self):
        """Test serialization round-trip without scheduler."""
        config = OptimizerConfig(
            optimizer_cls=AdamW,
            optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-4},
        )

        config_dict = config.to_dict()
        restored_config = OptimizerConfig.from_dict(config_dict)

        assert restored_config.optimizer_cls == AdamW
        assert restored_config.optimizer_kwargs == {"lr": 1e-3, "weight_decay": 1e-4}
        assert restored_config.scheduler_cls is None
        assert restored_config.scheduler_kwargs == {}


class TestStrategiesConfig:
    """Tests for StrategiesConfig."""

    def test_minimal_config(self, mock_query_strategy):
        """Test minimal strategies configuration."""
        config = StrategiesConfig(
            query_strategies=[mock_query_strategy],
            queue_cls=Queue,
        )
        assert len(config.query_strategies) == 1
        assert config.queue_cls == Queue
        assert config.label_strategy is None
        assert config.metrology_strategies is None
        assert config.unlabeled_datapool is None

    def test_full_config(
        self, mock_query_strategy, mock_label_strategy, mock_metrology_strategy
    ):
        """Test fully configured strategies."""
        config = StrategiesConfig(
            query_strategies=[mock_query_strategy],
            queue_cls=Queue,
            label_strategy=mock_label_strategy,
            metrology_strategies=[mock_metrology_strategy],
        )
        assert len(config.query_strategies) == 1
        assert config.label_strategy is not None
        assert len(config.metrology_strategies) == 1

    def test_empty_query_strategies(self):
        """Test that empty query strategies list raises error."""
        with pytest.raises(ValueError, match="At least one query strategy"):
            StrategiesConfig(query_strategies=[], queue_cls=Queue)

    def test_empty_metrology_strategies(self, mock_query_strategy):
        """Test that empty metrology strategies list raises error."""
        with pytest.raises(ValueError, match="metrology_strategies is an empty list"):
            StrategiesConfig(
                query_strategies=[mock_query_strategy],
                queue_cls=Queue,
                metrology_strategies=[],
            )

    def test_non_callable_query_strategy(self):
        """Test that non-callable query strategies are rejected."""
        with pytest.raises(ValueError, match="must be callable"):
            StrategiesConfig(
                query_strategies=["not_callable"],
                queue_cls=Queue,
            )

    def test_non_callable_label_strategy(self, mock_query_strategy):
        """Test that non-callable label strategy is rejected."""
        with pytest.raises(ValueError, match="label_strategy must be callable"):
            StrategiesConfig(
                query_strategies=[mock_query_strategy],
                queue_cls=Queue,
                label_strategy="not_callable",
            )

    def test_non_callable_metrology_strategy(self, mock_query_strategy):
        """Test that non-callable metrology strategies are rejected."""
        with pytest.raises(ValueError, match="must be callable"):
            StrategiesConfig(
                query_strategies=[mock_query_strategy],
                queue_cls=Queue,
                metrology_strategies=["not_callable"],
            )

    def test_to_dict_from_dict_round_trip(
        self, mock_query_strategy, mock_label_strategy, mock_metrology_strategy
    ):
        """Test that StrategiesConfig can be serialized and deserialized."""
        config = StrategiesConfig(
            query_strategies=[mock_query_strategy],
            queue_cls=Queue,
            label_strategy=mock_label_strategy,
            metrology_strategies=[mock_metrology_strategy],
        )

        # Serialize (no warning when unlabeled_datapool is None)
        config_dict = config.to_dict()

        # Deserialize (without unlabeled_datapool as it's not serialized)
        restored_config = StrategiesConfig.from_dict(config_dict)

        # Verify
        assert len(restored_config.query_strategies) == 1
        assert restored_config.queue_cls == Queue
        assert restored_config.label_strategy is not None
        assert len(restored_config.metrology_strategies) == 1
        assert restored_config.unlabeled_datapool is None

    def test_to_dict_from_dict_minimal(self, mock_query_strategy):
        """Test serialization round-trip with minimal config."""
        config = StrategiesConfig(
            query_strategies=[mock_query_strategy],
            queue_cls=Queue,
        )

        # Serialize (no warning when unlabeled_datapool is None)
        config_dict = config.to_dict()

        restored_config = StrategiesConfig.from_dict(config_dict)

        assert len(restored_config.query_strategies) == 1
        assert restored_config.queue_cls == Queue
        assert restored_config.label_strategy is None
        assert restored_config.metrology_strategies is None


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_minimal_config(self, mock_data_pool, mock_training_loop):
        """Test minimal training configuration."""
        config = TrainingConfig(
            train_datapool=mock_data_pool,
            train_loop_fn=mock_training_loop,
            max_training_epochs=10,
        )
        assert config.train_datapool is not None
        assert config.train_loop_fn is not None
        assert config.val_datapool is None
        assert isinstance(config.optimizer_config, OptimizerConfig)
        assert config.max_training_epochs == 10
        assert (
            config.max_fine_tuning_epochs == 10
        )  # should default to max_training_epochs

    def test_with_validation(self, mock_data_pool, mock_training_loop):
        """Test training config with validation data."""
        config = TrainingConfig(
            train_datapool=mock_data_pool,
            train_loop_fn=mock_training_loop,
            max_training_epochs=10,
            val_datapool=mock_data_pool,
        )
        assert config.val_datapool is not None

    def test_custom_optimizer_config(self, mock_data_pool, mock_training_loop):
        """Test training config with custom optimizer."""
        opt_config = OptimizerConfig(
            optimizer_cls=SGD,
            optimizer_kwargs={"lr": 0.01},
        )
        config = TrainingConfig(
            train_datapool=mock_data_pool,
            train_loop_fn=mock_training_loop,
            max_training_epochs=10,
            optimizer_config=opt_config,
        )
        assert config.optimizer_config.optimizer_cls == SGD

    def test_non_callable_training_loop(self, mock_data_pool):
        """Test that non-callable training loop is rejected."""
        with pytest.raises(ValueError, match="train_loop_fn must be callable"):
            TrainingConfig(
                train_datapool=mock_data_pool,
                train_loop_fn="not_callable",
                max_training_epochs=10,
            )

    def test_to_dict_from_dict_round_trip(self, mock_data_pool, mock_training_loop):
        """Test that TrainingConfig can be serialized and deserialized."""
        opt_config = OptimizerConfig(
            optimizer_cls=SGD,
            optimizer_kwargs={"lr": 0.05},
        )
        config = TrainingConfig(
            train_datapool=mock_data_pool,
            train_loop_fn=mock_training_loop,
            max_training_epochs=20,
            max_fine_tuning_epochs=5,
            optimizer_config=opt_config,
        )

        # Serialize
        with pytest.warns(UserWarning, match="train_datapool.*not supported"):
            config_dict = config.to_dict()

        # Deserialize (must provide datapools)
        restored_config = TrainingConfig.from_dict(
            config_dict, train_datapool=mock_data_pool
        )

        # Verify
        assert restored_config.max_training_epochs == 20
        assert restored_config.max_fine_tuning_epochs == 5
        assert restored_config.optimizer_config.optimizer_cls == SGD
        assert restored_config.optimizer_config.optimizer_kwargs == {"lr": 0.05}
        assert restored_config.train_datapool is mock_data_pool
        assert restored_config.val_datapool is None

    def test_from_dict_requires_train_datapool(self, mock_training_loop):
        """Test that from_dict requires train_datapool in kwargs."""
        config_dict = {
            "max_training_epochs": 10,
            "max_fine_tuning_epochs": 10,
            "optimizer_config": OptimizerConfig().to_dict(),
            "train_loop_fn": mock_training_loop._args,
        }

        with pytest.raises(ValueError, match="train_datapool.*must be provided"):
            TrainingConfig.from_dict(config_dict)

    def test_to_dict_from_dict_with_validation(
        self, mock_data_pool, mock_training_loop
    ):
        """Test serialization round-trip with validation datapool."""
        config = TrainingConfig(
            train_datapool=mock_data_pool,
            val_datapool=mock_data_pool,
            train_loop_fn=mock_training_loop,
            max_training_epochs=15,
        )

        with pytest.warns(UserWarning, match="train_datapool.*not supported"):
            config_dict = config.to_dict()

        # Provide both datapools during deserialization
        restored_config = TrainingConfig.from_dict(
            config_dict,
            train_datapool=mock_data_pool,
            val_datapool=mock_data_pool,
        )

        assert restored_config.train_datapool is mock_data_pool
        assert restored_config.val_datapool is mock_data_pool
        assert restored_config.max_training_epochs == 15


class TestDriverConfig:
    """Tests for DriverConfig."""

    def test_minimal_config(self):
        """Test minimal driver configuration."""
        config = DriverConfig(batch_size=4)
        assert config.batch_size == 4
        assert config.max_active_learning_steps == float("inf")
        assert config.skip_training is False
        assert config.skip_metrology is False
        assert config.skip_labeling is False

    def test_to_json(self):
        """Test JSON serialization."""
        config = DriverConfig(batch_size=8)
        json_str = config.to_json()
        json_dict = json.loads(json_str)
        assert json_dict["batch_size"] == 8
        assert "run_id" in json_dict
        assert "world_size" in json_dict

    @pytest.mark.parametrize("bad_number", [-1, float("inf"), nan])
    def test_invalid_batch_size(self, bad_number):
        """Test that invalid batch sizes are rejected."""
        with pytest.raises(ValueError, match="`batch_size` must be a positive integer"):
            DriverConfig(batch_size=bad_number)

    @pytest.mark.parametrize("bad_number", [-1, float("inf"), nan])
    def test_invalid_checkpoint_interval(self, bad_number):
        """Test that invalid checkpoint intervals are rejected."""
        with pytest.raises(
            ValueError, match="`checkpoint_interval` must be a non-negative integer"
        ):
            DriverConfig(batch_size=1, checkpoint_interval=bad_number)

    def test_zero_checkpoint_interval(self):
        """Test that checkpoint_interval=0 is valid (disables checkpointing)."""
        config = DriverConfig(batch_size=1, checkpoint_interval=0)
        assert config.checkpoint_interval == 0

    def test_invalid_fine_tuning_lr(self):
        """Test that invalid fine-tuning learning rates are rejected."""
        with pytest.raises(ValueError, match="`fine_tuning_lr` must be positive"):
            DriverConfig(batch_size=1, fine_tuning_lr=-0.01)

        with pytest.raises(ValueError, match="`fine_tuning_lr` must be positive"):
            DriverConfig(batch_size=1, fine_tuning_lr=0)

    def test_invalid_num_workers(self):
        """Test that negative num_workers is rejected."""
        with pytest.raises(
            ValueError, match="`num_dataloader_workers` must be non-negative"
        ):
            DriverConfig(batch_size=1, num_dataloader_workers=-1)

    def test_invalid_collate_fn(self):
        """Test that non-callable collate_fn is rejected."""
        with pytest.raises(ValueError, match="`collate_fn` must be callable"):
            DriverConfig(batch_size=1, collate_fn="not_callable")

    def test_max_steps_invalid(self):
        """Test that invalid max_active_learning_steps is rejected."""
        with pytest.raises(
            ValueError, match="`max_active_learning_steps` must be a positive integer"
        ):
            DriverConfig(batch_size=1, max_active_learning_steps=0)

        with pytest.raises(
            ValueError, match="`max_active_learning_steps` must be a positive integer"
        ):
            DriverConfig(batch_size=1, max_active_learning_steps=-5)

    def test_to_json_from_json_round_trip(self):
        """Test that DriverConfig can be serialized and deserialized."""
        import torch

        config = DriverConfig(
            batch_size=16,
            max_active_learning_steps=100,
            fine_tuning_lr=1e-5,
            reset_optim_states=False,
            skip_training=True,
            checkpoint_interval=5,
            num_dataloader_workers=4,
            device="cpu",
            dtype=torch.float32,
        )

        # Serialize
        json_str = config.to_json()

        # Verify it's valid JSON
        json_dict = json.loads(json_str)
        assert json_dict["batch_size"] == 16
        assert json_dict["max_active_learning_steps"] == 100

        # Deserialize
        restored_config = DriverConfig.from_json(json_str)

        # Verify
        assert restored_config.batch_size == 16
        assert restored_config.max_active_learning_steps == 100
        assert restored_config.fine_tuning_lr == 1e-5
        assert restored_config.reset_optim_states is False
        assert restored_config.skip_training is True
        assert restored_config.checkpoint_interval == 5
        assert restored_config.num_dataloader_workers == 4
        assert restored_config.device == torch.device("cpu")
        assert restored_config.dtype == torch.float32

    def test_to_json_from_json_minimal(self):
        """Test serialization round-trip with minimal config."""
        config = DriverConfig(batch_size=8)

        json_str = config.to_json()
        restored_config = DriverConfig.from_json(json_str)

        assert restored_config.batch_size == 8
        assert restored_config.max_active_learning_steps == float("inf")
        assert restored_config.skip_training is False

    def test_from_json_with_kwargs_override(self):
        """Test that kwargs can override deserialized values."""
        config = DriverConfig(batch_size=4, checkpoint_interval=10)
        json_str = config.to_json()

        # Override batch_size during deserialization
        restored_config = DriverConfig.from_json(json_str, batch_size=32)

        assert restored_config.batch_size == 32
        assert restored_config.checkpoint_interval == 10

    def test_from_json_with_collate_fn(self):
        """Test providing non-serializable collate_fn via kwargs."""

        def custom_collate(batch):
            return batch

        config = DriverConfig(batch_size=8)
        json_str = config.to_json()

        # Provide collate_fn during deserialization
        restored_config = DriverConfig.from_json(json_str, collate_fn=custom_collate)

        assert restored_config.collate_fn is custom_collate
        assert restored_config.batch_size == 8

    def test_to_json_from_json_different_dtypes(self):
        """Test serialization with different dtype values."""
        import torch

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            config = DriverConfig(batch_size=4, dtype=dtype)
            json_str = config.to_json()
            restored_config = DriverConfig.from_json(json_str)
            assert restored_config.dtype == dtype
