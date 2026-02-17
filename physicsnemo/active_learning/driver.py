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

"""
This module contains the definition for an active learning driver
class, which is responsible for orchestration and automation of
the end-to-end active learning process.
"""

from __future__ import annotations

import inspect
import pickle
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from physicsnemo import __version__ as physicsnemo_version
from physicsnemo.active_learning import protocols as p
from physicsnemo.active_learning.config import (
    DriverConfig,
    StrategiesConfig,
    TrainingConfig,
)
from physicsnemo.active_learning.logger import (
    ActiveLearningLoggerAdapter,
    setup_active_learning_logger,
)
from physicsnemo.core import Module
from physicsnemo.distributed import DistributedManager


@dataclass
class ActiveLearningCheckpoint:
    """
    Metadata associated with an ongoing (or completed) active
    learning experiment.

    The information contained in this metadata should be sufficient
    to restart the active learning experiment at the nearest point:
    for example, training should be able to continue from an epoch,
    while for querying/sampling, etc. we continue from a pre-existing
    queue.

    Attributes
    ----------
    driver_config: :class:`DriverConfig`
        Infrastructure and orchestration configuration.
    strategies_config: :class:`StrategiesConfig`
        Active learning strategies configuration.
    active_learning_step_idx: int
        Current iteration index of the active learning loop.
    active_learning_phase: :class:`~physicsnemo.active_learning.protocols.ActiveLearningPhase`
        Current phase of the active learning workflow.
    physicsnemo_version: str
        Version of PhysicsNeMo used to create the checkpoint.
    training_config: :class:`TrainingConfig` or None
        Training components configuration, if training is used.
    optimizer_state: dict or None
        Optimizer state dictionary for checkpointing.
    lr_scheduler_state: dict or None
        Learning rate scheduler state dictionary for checkpointing.
    has_query_queue: bool
        Whether the checkpoint includes a query queue.
    has_label_queue: bool
        Whether the checkpoint includes a label queue.

    See Also
    --------
    Driver : Uses this dataclass for checkpointing
    DriverConfig : Driver configuration
    StrategiesConfig : Strategies configuration
    TrainingConfig : Training configuration
    """

    driver_config: DriverConfig
    strategies_config: StrategiesConfig
    active_learning_step_idx: int
    active_learning_phase: p.ActiveLearningPhase
    physicsnemo_version: str = physicsnemo_version
    training_config: TrainingConfig | None = None
    optimizer_state: dict[str, Any] | None = None
    lr_scheduler_state: dict[str, Any] | None = None
    has_query_queue: bool = False
    has_label_queue: bool = False


class Driver(p.DriverProtocol):
    """
    Provides a simple implementation of the :class:`~physicsnemo.active_learning.protocols.DriverProtocol` used to
    orchestrate an active learning process within PhysicsNeMo.

    At a high level, the active learning process is broken down into four
    phases: training, metrology, query, and labeling.

    To understand the orchestration, start by inspecting the
    :meth:`active_learning_step` method, which defines a single iteration of
    the active learning loop, which is dispatched by the :meth:`run` method.
    From there, it should be relatively straightforward to trace the
    remaining components.

    Attributes
    ----------
    config: :class:`DriverConfig`
        Infrastructure and orchestration configuration.
    learner: :class:`~physicsnemo.Module` or :class:`~physicsnemo.active_learning.protocols.LearnerProtocol`
        The learner module for the active learning process.
    strategies_config: :class:`StrategiesConfig`
        Active learning strategies (query, label, metrology).
    training_config: :class:`TrainingConfig` or None
        Training components. None if training is skipped.
    inference_fn: :class:`~physicsnemo.active_learning.protocols.InferenceProtocol` or None
        Custom inference function.
    active_learning_step_idx: int
        Current iteration index of the active learning loop.
    query_queue: :class:`~physicsnemo.active_learning.protocols.AbstractQueue`
        Queue populated with data by query strategies.
    label_queue: :class:`~physicsnemo.active_learning.protocols.AbstractQueue`
        Queue populated with labeled data by the label strategy.
    optimizer: `torch.optim.Optimizer` or None
        Configured optimizer (set after configure_optimizer is called).
    lr_scheduler: `torch.optim.lr_scheduler._LRScheduler` or None
        Configured learning rate scheduler.
    logger: :class:`logging.Logger`
        Persistent logger for the active learning process.

    See Also
    --------
    DriverProtocol : Protocol specification for active learning drivers
    DriverConfig : Configuration for the driver
    StrategiesConfig : Configuration for active learning strategies
    TrainingConfig : Configuration for training
    """

    # Phase execution order for active learning step (immutable)
    _PHASE_ORDER = [
        p.ActiveLearningPhase.TRAINING,
        p.ActiveLearningPhase.METROLOGY,
        p.ActiveLearningPhase.QUERY,
        p.ActiveLearningPhase.LABELING,
    ]

    def __init__(
        self,
        config: DriverConfig,
        learner: Module | p.LearnerProtocol,
        strategies_config: StrategiesConfig,
        training_config: TrainingConfig | None = None,
        inference_fn: p.InferenceProtocol | None = None,
    ) -> None:
        """
        Initializes the active learning driver.

        At the bare minimum, the driver requires a config, learner, and
        strategies config to be used in a purely querying loop. Additional
        arguments can be provided to enable training and other workflows.

        Parameters
        ----------
        config: :class:`DriverConfig`
            Orchestration and infrastructure configuration, for example
            the batch size, the log directory, the distributed manager, etc.
        learner: :class:`~physicsnemo.Module` or :class:`~physicsnemo.active_learning.protocols.LearnerProtocol`
            The model to use for active learning.
        strategies_config: :class:`StrategiesConfig`
            Container for active learning strategies (query, label, metrology).
        training_config: :class:`TrainingConfig` or None
            Training components. Required if ``skip_training`` is False in
            the :class:`DriverConfig`.
        inference_fn: :class:`~physicsnemo.active_learning.protocols.InferenceProtocol` or None
            Custom inference function. If None, uses ``learner.__call__``.
            This is not actually called by the driver, but is stored as an
            attribute for attached strategies to use as needed.
        """
        # Configs have already validated themselves in __post_init__
        self.config = config
        self.learner = learner
        self.strategies_config = strategies_config
        self.training_config = training_config
        self.inference_fn = inference_fn
        self.active_learning_step_idx = 0
        self.current_phase: p.ActiveLearningPhase | None = (
            None  # Track current phase for logging context
        )
        self._last_checkpoint_path: Path | None = None

        # Validate cross-config constraints
        self._validate_config_consistency()

        self._setup_logger()
        self.attach_strategies()

        # Initialize queues from strategies_config
        self.query_queue = strategies_config.queue_cls()
        self.label_queue = strategies_config.queue_cls()

    def _validate_config_consistency(self) -> None:
        """
        Validate consistency across configs.

        Each config validates itself, but this method checks relationships
        between configs that can only be validated when composed together.
        """
        # If training is not skipped, training_config must be provided
        if not self.config.skip_training and self.training_config is None:
            raise ValueError(
                "`training_config` must be provided when `skip_training` is False."
            )

        # If labeling is not skipped, must have label strategy and train datapool
        if not self.config.skip_labeling:
            if self.strategies_config.label_strategy is None:
                raise ValueError(
                    "`label_strategy` must be provided in strategies_config "
                    "when `skip_labeling` is False."
                )
            if (
                self.training_config is None
                or self.training_config.train_datapool is None
            ):
                raise ValueError(
                    "`train_datapool` must be provided in training_config "
                    "when `skip_labeling` is False (labeled data is appended to it)."
                )

        # If fine-tuning lr is set, must have training enabled
        if self.config.fine_tuning_lr is not None and self.config.skip_training:
            raise ValueError(
                "`fine_tuning_lr` has no effect when `skip_training` is True."
            )

    @property
    def query_strategies(self) -> list[p.QueryStrategy]:
        """Returns the query strategies from strategies_config."""
        return self.strategies_config.query_strategies

    @property
    def label_strategy(self) -> p.LabelStrategy | None:
        """Returns the label strategy from strategies_config."""
        return self.strategies_config.label_strategy

    @property
    def metrology_strategies(self) -> list[p.MetrologyStrategy] | None:
        """Returns the metrology strategies from strategies_config."""
        return self.strategies_config.metrology_strategies

    @property
    def unlabeled_datapool(self) -> p.DataPool | None:
        """Returns the unlabeled datapool from strategies_config."""
        return self.strategies_config.unlabeled_datapool

    @property
    def train_datapool(self) -> p.DataPool | None:
        """Returns the training datapool from training_config."""
        return self.training_config.train_datapool if self.training_config else None

    @property
    def val_datapool(self) -> p.DataPool | None:
        """Returns the validation datapool from training_config."""
        return self.training_config.val_datapool if self.training_config else None

    @property
    def train_loop_fn(self) -> p.TrainingLoop | None:
        """Returns the training loop function from training_config."""
        return self.training_config.train_loop_fn if self.training_config else None

    @property
    def device(self) -> torch.device:
        """Return a consistent device interface to use across the driver."""
        if self.dist_manager is not None and self.dist_manager.is_initialized():
            return self.dist_manager.device
        else:
            return torch.get_default_device()

    @property
    def run_id(self) -> str:
        """Returns the run id from the ``DriverConfig``.

        Returns
        -------
        str
            The run id.
        """
        return self.config.run_id

    @property
    def log_dir(self) -> Path:
        """Returns the log directory.

        Note that this is the ``DriverConfig.root_log_dir`` combined
        with the shortened run ID for the current run.

        Effectively, this means that each run will have its own
        directory for logs, checkpoints, etc.

        Returns
        -------
        Path
            The log directory.
        """
        return self.config.root_log_dir / self.short_run_id

    @property
    def short_run_id(self) -> str:
        """Returns the first 8 characters of the run id.

        The 8 character limit assumes that the run ID is a UUID4.
        This is particularly useful for user-facing interfaces,
        where you do not necessarily want to reference the full UUID.

        Returns
        -------
        str
            The first 8 characters of the run id.
        """
        return self.run_id[:8]

    @property
    def last_checkpoint(self) -> Path | None:
        """
        Returns path to the most recently saved checkpoint.

        Returns
        -------
        Path | None
            Path to the last checkpoint directory, or None if no checkpoint
            has been saved yet.
        """
        return self._last_checkpoint_path

    @property
    def active_learning_step_idx(self) -> int:
        """
        Returns the current active learning step index.

        This represents the number of times the active learning step
        has been called, i.e. the number of iterations of the loop.

        Returns
        -------
        int
            The current active learning step index.
        """
        return self._active_learning_step_idx

    @active_learning_step_idx.setter
    def active_learning_step_idx(self, value: int) -> None:
        """
        Sets the current active learning step index.

        Parameters
        ----------
        value: int
            The new active learning step index.

        Raises
        ------
        ValueError
            If the new active learning step index is negative.
        """
        if value < 0:
            raise ValueError("Active learning step index must be non-negative.")
        self._active_learning_step_idx = value

    @property
    def dist_manager(self) -> DistributedManager | None:
        """Returns the distributed manager, if it was specified as part
        of the `DriverConfig` configuration.

        Returns
        -------
        DistributedManager | None
            The distributed manager.
        """
        return self.config.dist_manager

    def configure_optimizer(self) -> None:
        """Setup optimizer and LR schedulers from training_config."""
        if self.training_config is None:
            self.optimizer = None
            self.lr_scheduler = None
            return

        opt_cfg = self.training_config.optimizer_config

        if opt_cfg.optimizer_cls is not None:
            try:
                _ = inspect.signature(opt_cfg.optimizer_cls).bind(
                    self.learner.parameters(), **opt_cfg.optimizer_kwargs
                )
            except TypeError as e:
                raise ValueError(
                    f"Invalid optimizer kwargs for {opt_cfg.optimizer_cls}; {e}"
                )
            self.optimizer = opt_cfg.optimizer_cls(
                self.learner.parameters(), **opt_cfg.optimizer_kwargs
            )
        else:
            self.optimizer = None
            return

        if opt_cfg.scheduler_cls is not None and self.optimizer is not None:
            try:
                _ = inspect.signature(opt_cfg.scheduler_cls).bind(
                    self.optimizer, **opt_cfg.scheduler_kwargs
                )
            except TypeError as e:
                raise ValueError(
                    f"Invalid LR scheduler kwargs for {opt_cfg.scheduler_cls}; {e}"
                )
            self.lr_scheduler = opt_cfg.scheduler_cls(
                self.optimizer, **opt_cfg.scheduler_kwargs
            )
        else:
            self.lr_scheduler = None
        # in the case where we want to reset optimizer states between active learning steps
        if self.config.reset_optim_states and self.is_optimizer_configured:
            self._original_optim_state = deepcopy(self.optimizer.state_dict())

    @property
    def is_optimizer_configured(self) -> bool:
        """Returns whether the optimizer is configured."""
        return getattr(self, "optimizer", None) is not None

    @property
    def is_lr_scheduler_configured(self) -> bool:
        """Returns whether the LR scheduler is configured."""
        return getattr(self, "lr_scheduler", None) is not None

    def attach_strategies(self) -> None:
        """Calls ``strategy.attach`` for all available strategies."""
        super().attach_strategies()

    def _setup_logger(self) -> None:
        """
        Sets up a persistent logger for the driver.

        This logger is specialized in that it provides additional context
        information depending on the part of the active learning cycle.
        """
        base_logger = setup_active_learning_logger(
            "core.active_learning",
            run_id=self.run_id,
            log_dir=self.log_dir,
        )
        # Wrap with adapter to automatically include iteration context
        self.logger = ActiveLearningLoggerAdapter(base_logger, driver_ref=self)

    def _should_checkpoint_at_step(self) -> bool:
        """
        Determine if a checkpoint should be saved at the current AL step.

        Uses the `checkpoint_interval` from config to decide. If interval is 0,
        checkpointing is disabled. Otherwise, checkpoint at step 0 and every
        N steps thereafter.

        Returns
        -------
        bool
            True if checkpoint should be saved, False otherwise.
        """
        if self.config.checkpoint_interval == 0:
            return False
        # Always checkpoint at step 0, then every checkpoint_interval steps
        return self.active_learning_step_idx % self.config.checkpoint_interval == 0

    def _serialize_queue(self, queue: p.AbstractQueue, file_path: Path) -> bool:
        """
        Serialize queue to a file.

        If queue implements `to_list()`, serialize the list. Otherwise, use
        torch.save to serialize the entire queue object.

        Parameters
        ----------
        queue: p.AbstractQueue
            The queue to serialize.
        file_path: Path
            Path where the queue should be saved.

        Returns
        -------
        bool
            True if serialization succeeded, False otherwise.
        """
        try:
            if hasattr(queue, "to_list") and callable(getattr(queue, "to_list")):
                # Use custom serialization method
                queue_data = {"type": "list", "data": queue.to_list()}
            else:
                # Fallback to torch.save for the entire queue
                queue_data = {"type": "torch", "data": queue}

            torch.save(queue_data, file_path)
            return True
        except (TypeError, AttributeError, pickle.PicklingError, RuntimeError) as e:
            # Some queues cannot be pickled, e.g. stdlib queue.Queue with thread locks
            # Clean up any partially written file
            if file_path.exists():
                file_path.unlink()

            self.logger.warning(
                f"Failed to serialize queue to {file_path}: {e}. Queue state will not be saved. "
                f"Consider implementing to_list()/from_list() methods for custom serialization."
            )
            return False

    def _deserialize_queue(self, queue: p.AbstractQueue, file_path: Path) -> None:
        """
        Restore queue from a file.

        Parameters
        ----------
        queue: p.AbstractQueue
            The queue to restore data into.
        file_path: Path
            Path to the saved queue file.
        """
        if not file_path.exists():
            return

        try:
            queue_data = torch.load(file_path, map_location="cpu", weights_only=False)

            if queue_data["type"] == "list":
                if hasattr(queue, "from_list") and callable(
                    getattr(queue, "from_list")
                ):
                    queue.from_list(queue_data["data"])
                else:
                    # Manually populate queue from list
                    for item in queue_data["data"]:
                        queue.put(item)
            elif queue_data["type"] == "torch":
                # Restore from torch-saved queue - copy items to current queue
                restored_queue = queue_data["data"]
                # Copy items from restored queue to current queue
                while not restored_queue.empty():
                    queue.put(restored_queue.get())
        except Exception as e:
            self.logger.warning(
                f"Failed to deserialize queue from {file_path}: {e}. "
                f"Queue will be empty."
            )

    def save_checkpoint(
        self, path: str | Path | None = None, training_epoch: int | None = None
    ) -> Path | None:
        """
        Save a checkpoint of the active learning experiment.

        Saves AL orchestration state (configs, queues, step index, phase) and model weights.
        Training-specific state (optimizer, scheduler) is handled by DefaultTrainingLoop
        and saved to training_state.pt during training.

        Parameters
        ----------
        path: str | Path | None
            Path to save checkpoint. If None, creates path based on current
            AL step index and phase: log_dir/checkpoints/step_{idx}/{phase}/
        training_epoch: int | None
            Optional epoch number for mid-training checkpoints.

        Returns
        -------
        Path | None
            Checkpoint directory path, or None if checkpoint not saved (non-rank-0 in distributed).
        """
        # Determine checkpoint directory
        if path is None:
            phase_name = self.current_phase if self.current_phase else "init"
            checkpoint_dir = (
                self.log_dir
                / "checkpoints"
                / f"step_{self.active_learning_step_idx}"
                / phase_name
            )
            if training_epoch is not None:
                checkpoint_dir = checkpoint_dir / f"epoch_{training_epoch}"
        else:
            checkpoint_dir = Path(path)

        # Create checkpoint directory
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Only rank 0 saves checkpoint in distributed setting
        if self.dist_manager is not None and self.dist_manager.is_initialized():
            if self.dist_manager.rank != 0:
                return None

        # Serialize configurations
        driver_config_json = self.config.to_json()
        strategies_config_dict = self.strategies_config.to_dict()
        training_config_dict = (
            self.training_config.to_dict() if self.training_config else None
        )

        # Serialize queue states to separate files
        query_queue_file = checkpoint_dir / "query_queue.pt"
        label_queue_file = checkpoint_dir / "label_queue.pt"
        has_query_queue = self._serialize_queue(self.query_queue, query_queue_file)
        has_label_queue = self._serialize_queue(self.label_queue, label_queue_file)

        # Create checkpoint dataclass (only AL orchestration state)
        checkpoint = ActiveLearningCheckpoint(
            driver_config=driver_config_json,
            strategies_config=strategies_config_dict,
            active_learning_step_idx=self.active_learning_step_idx,
            active_learning_phase=self.current_phase or p.ActiveLearningPhase.TRAINING,
            physicsnemo_version=physicsnemo_version,
            training_config=training_config_dict,
            optimizer_state=None,  # Training loop handles this
            lr_scheduler_state=None,  # Training loop handles this
            has_query_queue=has_query_queue,
            has_label_queue=has_label_queue,
        )

        # Add training epoch if in mid-training checkpoint
        checkpoint_dict = {
            "checkpoint": checkpoint,
        }
        if training_epoch is not None:
            checkpoint_dict["training_epoch"] = training_epoch

        # Save checkpoint metadata
        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        torch.save(checkpoint_dict, checkpoint_path)

        # Save model weights (separate from training state)
        if isinstance(self.learner, Module):
            model_name = self.learner.__class__.__name__
            model_path = checkpoint_dir / f"{model_name}.mdlus"
            self.learner.save(str(model_path))
        elif hasattr(self.learner, "module") and isinstance(
            self.learner.module, Module
        ):
            # Unwrap DDP
            model_name = self.learner.module.__class__.__name__
            model_path = checkpoint_dir / f"{model_name}.mdlus"
            self.learner.module.save(str(model_path))
        else:
            model_name = self.learner.__class__.__name__
            model_path = checkpoint_dir / f"{model_name}.pt"
            torch.save(self.learner.state_dict(), model_path)

        # Update last checkpoint path
        self._last_checkpoint_path = checkpoint_dir

        # Log successful checkpoint save
        self.logger.info(
            f"Saved checkpoint at step {self.active_learning_step_idx}, "
            f"phase {self.current_phase}: {checkpoint_dir}"
        )

        return checkpoint_dir

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        learner: Module | p.LearnerProtocol | None = None,
        train_datapool: p.DataPool | None = None,
        val_datapool: p.DataPool | None = None,
        unlabeled_datapool: p.DataPool | None = None,
        **kwargs: Any,
    ) -> Driver:
        """
        Load a Driver instance from a checkpoint.

        Given a checkpoint directory, this method will attempt to reconstruct
        the driver and its associated components from the checkpoint. The
        checkpoint path must contain a ``checkpoint.pt`` file, which contains
        the metadata associated with the experiment.

        Additional parameters that might not be serialized with the checkpointing
        mechanism can/need to be provided to this method; for example when
        using non-`physicsnemo.Module` learners, and any data pools associated
        with the workflow.

        .. important::

            Currently, the strategy states are not reloaded from the checkpoint.
            This will be addressed in a future patch, but for now it is recommended
            to back up your strategy states (e.g. metrology records) manually
            before restarting experiments.

        Parameters
        ----------
        checkpoint_path: str | Path
            Path to checkpoint directory containing checkpoint.pt and model weights.
        learner: Module | p.LearnerProtocol | None
            Learner model to load weights into. If None, will attempt to
            reconstruct from checkpoint (only works for physicsnemo.Module).
        train_datapool: p.DataPool | None
            Training datapool. Required if training_config exists in checkpoint.
        val_datapool: p.DataPool | None
            Validation datapool. Optional.
        unlabeled_datapool: p.DataPool | None
            Unlabeled datapool for query strategies. Optional.
        **kwargs: Any
            Additional keyword arguments to override config values.

        Returns
        -------
        Driver
            Reconstructed Driver instance ready to resume execution.
        """
        checkpoint_path = Path(checkpoint_path)

        # Load checkpoint file
        checkpoint_file = checkpoint_path / "checkpoint.pt"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        checkpoint_dict = torch.load(
            checkpoint_file, map_location="cpu", weights_only=False
        )
        checkpoint: ActiveLearningCheckpoint = checkpoint_dict["checkpoint"]
        training_epoch = checkpoint_dict.get("training_epoch", None)

        # Reconstruct configs
        driver_config = DriverConfig.from_json(
            checkpoint.driver_config, **kwargs.get("driver_config_overrides", {})
        )

        # TODO add strategy state loading from checkpoint
        strategies_config = StrategiesConfig.from_dict(
            checkpoint.strategies_config,
            unlabeled_datapool=unlabeled_datapool,
            **kwargs.get("strategies_config_overrides", {}),
        )

        training_config = None
        if checkpoint.training_config is not None:
            training_config = TrainingConfig.from_dict(
                checkpoint.training_config,
                train_datapool=train_datapool,
                val_datapool=val_datapool,
                **kwargs.get("training_config_overrides", {}),
            )

        # Load or reconstruct learner
        if learner is None:
            # Attempt to reconstruct from checkpoint (only for Module)
            # Try to find any .mdlus file in the checkpoint directory
            mdlus_files = list(checkpoint_path.glob("*.mdlus"))
            if mdlus_files:
                # Use the first .mdlus file found
                model_path = mdlus_files[0]
                learner = Module.from_checkpoint(str(model_path))
            else:
                raise ValueError(
                    "No learner provided and unable to reconstruct from checkpoint. "
                    "Please provide a learner instance."
                )
        else:
            # Load model weights into provided learner
            # Determine expected model filename based on learner type
            if isinstance(learner, Module):
                model_name = learner.__class__.__name__
                model_path = checkpoint_path / f"{model_name}.mdlus"
                if model_path.exists():
                    learner.load(str(model_path))
                else:
                    # Fallback: try to find any .mdlus file
                    mdlus_files = list(checkpoint_path.glob("*.mdlus"))
                    if mdlus_files:
                        learner.load(str(mdlus_files[0]))
            elif hasattr(learner, "module") and isinstance(learner.module, Module):
                # Unwrap DDP
                model_name = learner.module.__class__.__name__
                model_path = checkpoint_path / f"{model_name}.mdlus"
                if model_path.exists():
                    learner.module.load(str(model_path))
                else:
                    # Fallback: try to find any .mdlus file
                    mdlus_files = list(checkpoint_path.glob("*.mdlus"))
                    if mdlus_files:
                        learner.module.load(str(mdlus_files[0]))
            else:
                # Non-Module learner: look for .pt file with class name
                model_name = learner.__class__.__name__
                model_path = checkpoint_path / f"{model_name}.pt"
                if model_path.exists():
                    state_dict = torch.load(model_path, map_location="cpu")
                    learner.load_state_dict(state_dict)
                else:
                    # Fallback: try to find any .pt file
                    pt_files = list(checkpoint_path.glob("*.pt"))
                    # Filter out checkpoint.pt and queue files
                    model_pt_files = [
                        f
                        for f in pt_files
                        if f.name
                        not in [
                            "checkpoint.pt",
                            "query_queue.pt",
                            "label_queue.pt",
                            "training_state.pt",
                        ]
                    ]
                    if model_pt_files:
                        state_dict = torch.load(model_pt_files[0], map_location="cpu")
                        learner.load_state_dict(state_dict)

        # Instantiate Driver
        driver = cls(
            config=driver_config,
            learner=learner,
            strategies_config=strategies_config,
            training_config=training_config,
            inference_fn=kwargs.get("inference_fn", None),
        )

        # Restore active learning state
        driver.active_learning_step_idx = checkpoint.active_learning_step_idx
        driver.current_phase = checkpoint.active_learning_phase
        driver._last_checkpoint_path = checkpoint_path

        # Load training state (optimizer, scheduler) if training_config exists
        # This delegates to the training loop's checkpoint loading logic
        if driver.training_config is not None:
            driver.configure_optimizer()

            # Use training loop to load training state (including model weights again if needed)
            from physicsnemo.active_learning.loop import DefaultTrainingLoop

            DefaultTrainingLoop.load_training_checkpoint(
                checkpoint_dir=checkpoint_path,
                model=driver.learner,
                optimizer=driver.optimizer,
                lr_scheduler=driver.lr_scheduler
                if hasattr(driver, "lr_scheduler")
                else None,
            )

        # Restore queue states from separate files
        if checkpoint.has_query_queue:
            query_queue_file = checkpoint_path / "query_queue.pt"
            driver._deserialize_queue(driver.query_queue, query_queue_file)

        if checkpoint.has_label_queue:
            label_queue_file = checkpoint_path / "label_queue.pt"
            driver._deserialize_queue(driver.label_queue, label_queue_file)

        driver.logger.info(
            f"Loaded checkpoint from {checkpoint_path} at step "
            f"{checkpoint.active_learning_step_idx}, phase {checkpoint.active_learning_phase}"
        )
        if training_epoch is not None:
            driver.logger.info(f"Resuming from training epoch {training_epoch}")

        return driver

    def barrier(self) -> None:
        """
        Wrapper to call barrier on the correct device.

        Becomes a no-op if distributed is not initialized, otherwise
        will attempt to read the local device ID from either the distributed manager
        or the default device.
        """
        if dist.is_initialized():
            if (
                self.dist_manager is not None
                and self.dist_manager.device.type == "cuda"
            ):
                dist.barrier(device_ids=[self.dist_manager.local_rank])
            elif torch.get_default_device().type == "cuda":
                # this might occur if distributed manager is not used
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()

    def _configure_model(self) -> None:
        """
        Method that encapsulates all the logic for preparing the model
        ahead of time.

        If the distributed manager has been configured and initialized
        with a world size greater than 1, then we wrap the model in DDP.
        Otherwise, we simply move the model to the correct device.

        After the model has been moved to device, we configure the optimizer
        and learning rate scheduler if training is enabled.
        """
        if self.dist_manager is not None and self.dist_manager.is_initialized():
            if self.dist_manager.world_size > 1 and not isinstance(
                self.learner, DistributedDataParallel
            ):
                # wrap the model in DDP
                self.learner = torch.nn.parallel.DistributedDataParallel(
                    self.learner,
                    device_ids=[self.dist_manager.local_rank],
                    output_device=self.dist_manager.device,
                    broadcast_buffers=self.dist_manager.broadcast_buffers,
                    find_unused_parameters=self.dist_manager.find_unused_parameters,
                )
        else:
            if self.config.device is not None:
                self.learner = self.learner.to(self.config.device, self.config.dtype)
        # assume all device management is done via the dist_manager, so at this
        # point the model is on the correct device and we can set up the optimizer
        # if we intend to train
        if not self.config.skip_training and not self.is_optimizer_configured:
            self.configure_optimizer()
        if self.is_optimizer_configured and self.config.reset_optim_states:
            self.optimizer.load_state_dict(self._original_optim_state)

    def _get_phase_index(self, phase: p.ActiveLearningPhase | None) -> int:
        """
        Get index of phase in execution order.

        Parameters
        ----------
        phase: p.ActiveLearningPhase | None
            Phase to find index for. If None, returns 0 (start from beginning).

        Returns
        -------
        int
            Index in _PHASE_ORDER (0-3).
        """
        if phase is None:
            return 0
        try:
            return self._PHASE_ORDER.index(phase)
        except ValueError:
            self.logger.warning(
                f"Unknown phase {phase}, defaulting to start from beginning"
            )
            return 0

    def _build_phase_queue(
        self,
        train_step_fn: p.TrainingProtocol | None,
        validate_step_fn: p.ValidationProtocol | None,
        args: tuple,
        kwargs: dict,
    ) -> list[Any]:
        """
        Build list of phase functions to execute for this AL step.

        If current_phase is set (e.g., from checkpoint), only phases at or after
        current_phase are included. Otherwise, all non-skipped phases are included.

        Parameters
        ----------
        train_step_fn: p.TrainingProtocol | None
            Training function to pass to training phase.
        validate_step_fn: p.ValidationProtocol | None
            Validation function to pass to training phase.
        args: tuple
            Additional arguments to pass to phase methods.
        kwargs: dict
            Additional keyword arguments to pass to phase methods.

        Returns
        -------
        list[Callable]
            Queue of phase functions to execute in order.
        """
        # Define all possible phases with their execution conditions
        all_phases = [
            (
                p.ActiveLearningPhase.TRAINING,
                lambda: self._training_phase(
                    train_step_fn, validate_step_fn, *args, **kwargs
                ),
                not self.config.skip_training,
            ),
            (
                p.ActiveLearningPhase.METROLOGY,
                lambda: self._metrology_phase(*args, **kwargs),
                not self.config.skip_metrology,
            ),
            (
                p.ActiveLearningPhase.QUERY,
                lambda: self._query_phase(*args, **kwargs),
                True,  # Query phase always runs
            ),
            (
                p.ActiveLearningPhase.LABELING,
                lambda: self._labeling_phase(*args, **kwargs),
                not self.config.skip_labeling,
            ),
        ]

        # Find starting index based on current_phase (resume point)
        start_idx = self._get_phase_index(self.current_phase)

        if start_idx > 0:
            self.logger.info(
                f"Resuming AL step {self.active_learning_step_idx} from "
                f"{self.current_phase}"
            )

        # Build queue: only phases from start_idx onwards that should run
        phase_queue = []
        for idx, (phase, phase_fn, should_run) in enumerate(all_phases):
            # Skip phases before current_phase
            if idx < start_idx:
                self.logger.debug(
                    f"Skipping {phase} (already completed in this AL step)"
                )
                continue

            # Add phase to queue if not skipped by config
            if should_run:
                phase_queue.append(phase_fn)
            else:
                self.logger.debug(f"Skipping {phase} (disabled in config)")

        return phase_queue

    def _construct_dataloader(
        self, pool: p.DataPool, shuffle: bool = False, drop_last: bool = False
    ) -> DataLoader:
        """
        Helper method to construct a data loader for a given data pool.

        In the case that a distributed manager was provided, then a distributed
        sampler will be used, which will be bound to the current rank.
        Otherwise, a regular sampler will be used. Similarly, if your data
        structure requires a specialized function to construct batches,
        then this function can be provided via the `collate_fn` argument.

        Parameters
        ----------
        pool: p.DataPool
            The data pool to construct a data loader for.
        shuffle: bool = False
            Whether to shuffle the data.
        drop_last: bool = False
            Whether to drop the last batch if it is not complete.

        Returns
        -------
        DataLoader
            The constructed data loader.
        """
        # if a distributed manager was omitted, then we assume single process
        if self.dist_manager is not None and self.dist_manager.is_initialized():
            sampler = DistributedSampler(
                pool,
                num_replicas=self.dist_manager.world_size,
                rank=self.dist_manager.rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            # set to None, because sampler will handle instead
            shuffle = None
        else:
            sampler = None
        # fully spec out the data loader
        pin_memory = False
        if self.dist_manager is not None and self.dist_manager.is_initialized():
            if self.dist_manager.device.type == "cuda":
                pin_memory = True
        loader = DataLoader(
            pool,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self.config.collate_fn,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_dataloader_workers,
            persistent_workers=self.config.num_dataloader_workers > 0,
            pin_memory=pin_memory,
        )
        return loader

    def active_learning_step(
        self,
        train_step_fn: p.TrainingProtocol | None = None,
        validate_step_fn: p.ValidationProtocol | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Performs a single active learning iteration.

        This method will perform the following sequence of steps:
        1. Train the model stored in ``Driver.learner`` by creating data loaders
        with ``Driver.train_datapool`` and ``Driver.val_datapool``.
        2. Run the metrology strategies stored in ``Driver.metrology_strategies``.
        3. Run the query strategies stored in ``Driver.query_strategies``, if available.
        4. Run the labeling strategy stored in ``Driver.label_strategy``, if available.

        When entering each stage, we check to ensure all components necessary for the
        minimum function for that stage are available before proceeding.

        If current_phase is set (e.g., from checkpoint resumption), only phases at
        or after current_phase will be executed. After completing all phases,
        current_phase is reset to None for the next AL step.

        Parameters
        ----------
        train_step_fn: p.TrainingProtocol | None = None
            The training function to use for training. If not provided, then the
            ``Driver.train_loop_fn`` will be used.
        validate_step_fn: p.ValidationProtocol | None = None
            The validation function to use for validation. If not provided, then
            validation will not be performed.
        args: Any
            Additional arguments to pass to the method. These will be passed to the
            training loop, metrology strategies, query strategies, and labeling strategies.
        kwargs: Any
            Additional keyword arguments to pass to the method. These will be passed to the
            training loop, metrology strategies, query strategies, and labeling strategies.

        Raises
        ------
        ValueError
            If any of the required components for a stage are not available.
        """
        self._setup_active_learning_step()

        # Build queue of phase functions based on current_phase
        phase_queue = self._build_phase_queue(
            train_step_fn, validate_step_fn, args, kwargs
        )

        # Execute each phase in order (de-populate queue)
        for phase_fn in phase_queue:
            phase_fn()

        # Reset current_phase after completing all phases in this AL step
        self.current_phase = None

        self.logger.debug("Entering barrier for synchronization.")
        self.barrier()
        self.active_learning_step_idx += 1
        self.logger.info(
            f"Completed active learning step {self.active_learning_step_idx}"
        )

    def _setup_active_learning_step(self) -> None:
        """Initialize distributed manager and configure model for the active learning step."""
        if self.dist_manager is not None and not self.dist_manager.is_initialized():
            self.logger.info(
                "Distributed manager configured but not initialized; initializing."
            )
            self.dist_manager.initialize()
        self._configure_model()
        self.logger.info(
            f"Starting active learning step {self.active_learning_step_idx}"
        )

    def _training_phase(
        self,
        train_step_fn: p.TrainingProtocol | None,
        validate_step_fn: p.ValidationProtocol | None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Execute the training phase of the active learning step."""
        self._validate_training_requirements(train_step_fn, validate_step_fn)

        # don't need to barrier because it'll be done at the end of training anyway
        with self._phase_context("training", call_barrier=False):
            # Note: Training phase checkpointing is handled by the training loop itself
            # during epoch execution based on model_checkpoint_frequency

            train_loader = self._construct_dataloader(self.train_datapool, shuffle=True)
            self.logger.info(
                f"There are {len(train_loader)} batches in the training loader."
            )
            val_loader = None
            if self.val_datapool is not None:
                if validate_step_fn or hasattr(self.learner, "validation_step"):
                    val_loader = self._construct_dataloader(
                        self.val_datapool, shuffle=False
                    )
                else:
                    self.logger.warning(
                        "Validation data is available, but no `validate_step_fn` "
                        "or `validation_step` method in Learner is provided."
                    )
            # if a fine-tuning lr is provided, adjust it after the first iteration
            if (
                self.config.fine_tuning_lr is not None
                and self.active_learning_step_idx > 0
            ):
                self.optimizer.param_groups[0]["lr"] = self.config.fine_tuning_lr

            # Determine max epochs to train for this AL step
            if self.active_learning_step_idx > 0:
                target_max_epochs = self.training_config.max_fine_tuning_epochs
            else:
                target_max_epochs = self.training_config.max_training_epochs

            # Check if resuming from mid-training checkpoint
            start_epoch = 1
            epochs_to_train = target_max_epochs

            if self._last_checkpoint_path and self._last_checkpoint_path.exists():
                training_state_path = self._last_checkpoint_path / "training_state.pt"
                if training_state_path.exists():
                    training_state = torch.load(
                        training_state_path, map_location="cpu", weights_only=False
                    )
                    last_completed_epoch = training_state.get("training_epoch", 0)
                    if last_completed_epoch > 0:
                        start_epoch = last_completed_epoch + 1
                        epochs_to_train = target_max_epochs - last_completed_epoch
                        self.logger.info(
                            f"Resuming training from epoch {start_epoch} "
                            f"({epochs_to_train} epochs remaining)"
                        )

            # Skip training if all epochs already completed
            if epochs_to_train <= 0:
                self.logger.info(
                    f"Training already complete ({target_max_epochs} epochs), "
                    f"skipping training phase"
                )
                return

            device = (
                self.dist_manager.device
                if self.dist_manager is not None
                else self.config.device
            )
            dtype = self.config.dtype

            # Set checkpoint directory and frequency on training loop
            # This allows the training loop to handle training state checkpointing internally
            if hasattr(self.train_loop_fn, "checkpoint_base_dir") and hasattr(
                self.train_loop_fn, "checkpoint_frequency"
            ):
                # Checkpoint base is the current AL step's training directory
                checkpoint_base = (
                    self.log_dir
                    / "checkpoints"
                    / f"step_{self.active_learning_step_idx}"
                    / "training"
                )
                self.train_loop_fn.checkpoint_base_dir = checkpoint_base
                self.train_loop_fn.checkpoint_frequency = (
                    self.config.model_checkpoint_frequency
                )

            self.train_loop_fn(
                self.learner,
                self.optimizer,
                train_step_fn=train_step_fn,
                validate_step_fn=validate_step_fn,
                train_dataloader=train_loader,
                validation_dataloader=val_loader,
                lr_scheduler=self.lr_scheduler,
                max_epochs=epochs_to_train,  # Only remaining epochs
                device=device,
                dtype=dtype,
                **kwargs,
            )

    def _metrology_phase(self, *args: Any, **kwargs: Any) -> None:
        """Execute the metrology phase of the active learning step."""

        with self._phase_context("metrology"):
            for strategy in self.metrology_strategies:
                self.logger.info(
                    f"Running metrology strategy: {strategy.__class__.__name__}"
                )
                strategy(*args, **kwargs)
                self.logger.info(
                    f"Completed metrics for strategy: {strategy.__class__.__name__}"
                )
                strategy.serialize_records(*args, **kwargs)

    def _query_phase(self, *args: Any, **kwargs: Any) -> None:
        """Execute the query phase of the active learning step."""
        with self._phase_context("query"):
            for strategy in self.query_strategies:
                self.logger.info(
                    f"Running query strategy: {strategy.__class__.__name__}"
                )
                strategy(self.query_queue, *args, **kwargs)

            if self.query_queue.empty():
                self.logger.warning(
                    "Querying strategies produced no samples this iteration."
                )

    def _labeling_phase(self, *args: Any, **kwargs: Any) -> None:
        """Execute the labeling phase of the active learning step."""
        self._validate_labeling_requirements()

        if self.query_queue.empty():
            self.logger.warning("No samples to label. Skipping labeling phase.")
            return

        with self._phase_context("labeling"):
            try:
                self.label_strategy(self.query_queue, self.label_queue, *args, **kwargs)
            except Exception as e:
                self.logger.error(f"Exception encountered during labeling: {e}")
            self.logger.info("Labeling completed. Now appending to training pool.")

            # TODO this is done serially, could be improved with batched writes
            sample_counter = 0
            while not self.label_queue.empty():
                self.train_datapool.append(self.label_queue.get())
                sample_counter += 1
            self.logger.info(f"Appended {sample_counter} samples to training pool.")

    def _validate_training_requirements(
        self,
        train_step_fn: p.TrainingProtocol | None,
        validate_step_fn: p.ValidationProtocol | None,
    ) -> None:
        """Validate that all required components for training are available."""
        if self.training_config is None:
            raise ValueError(
                "`training_config` must be provided if `skip_training` is False."
            )
        if self.train_loop_fn is None:
            raise ValueError("`train_loop_fn` must be provided in training_config.")
        if self.train_datapool is None:
            raise ValueError("`train_datapool` must be provided in training_config.")
        if not train_step_fn and not hasattr(self.learner, "training_step"):
            raise ValueError(
                "`train_step_fn` must be provided if the model does not implement "
                "the `training_step` method."
            )
        if validate_step_fn and self.val_datapool is None:
            raise ValueError(
                "`val_datapool` must be provided in training_config if "
                "`validate_step_fn` is provided."
            )

    def _validate_labeling_requirements(self) -> None:
        """Validate that all required components for labeling are available."""
        if self.label_strategy is None:
            raise ValueError(
                "`label_strategy` must be provided in strategies_config if "
                "`skip_labeling` is False."
            )
        if self.training_config is None or self.train_datapool is None:
            raise ValueError(
                "`train_datapool` must be provided in training_config for "
                "labeling, as data will be appended to it."
            )

    @contextmanager
    def _phase_context(
        self, phase_name: p.ActiveLearningPhase, call_barrier: bool = True
    ) -> Generator[None, Any, None]:
        """
        Context manager for consistent phase tracking, error handling, and synchronization.

        Sets the current phase for logging context, handles exceptions,
        and synchronizes distributed workers with a barrier. Also triggers
        checkpoint saves at the start of each phase if configured.

        Parameters
        ----------
        phase_name: p.ActiveLearningPhase
            A discrete phase of the active learning workflow.
        call_barrier: bool
            Whether to call barrier for synchronization at the end.
        """
        self.current_phase = phase_name

        # Save checkpoint at START of phase if configured
        # Exception: training phase handles checkpointing internally
        if phase_name != p.ActiveLearningPhase.TRAINING:
            should_checkpoint = getattr(
                self.config, f"checkpoint_on_{phase_name}", False
            )
            # Check if we should checkpoint based on interval
            if should_checkpoint and self._should_checkpoint_at_step():
                self.save_checkpoint()

        try:
            yield
        except Exception as e:
            self.logger.error(f"Exception encountered during {phase_name}: {e}")
            raise
        finally:
            if call_barrier:
                self.logger.debug("Entering barrier for synchronization.")
                self.barrier()

    def run(
        self,
        train_step_fn: p.TrainingProtocol | None = None,
        validate_step_fn: p.ValidationProtocol | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Runs the active learning loop until the maximum number of
        active learning steps is reached.

        Parameters
        ----------
        train_step_fn: p.TrainingProtocol | None = None
            The training function to use for training. If not provided, then the
            ``Driver.train_loop_fn`` will be used.
        validate_step_fn: p.ValidationProtocol | None = None
            The validation function to use for validation. If not provided, then
            validation will not be performed.
        args: Any
            Additional arguments to pass to the method. These will be passed to the
            training loop, metrology strategies, query strategies, and labeling strategies.
        kwargs: Any
            Additional keyword arguments to pass to the method. These will be passed to the
            training loop, metrology strategies, query strategies, and labeling strategies.
        """
        # TODO: refactor initialization logic here instead of inside the step
        while self.active_learning_step_idx < self.config.max_active_learning_steps:
            self.active_learning_step(
                train_step_fn=train_step_fn,
                validate_step_fn=validate_step_fn,
                *args,
                **kwargs,
            )

    def __call__(
        self,
        train_step_fn: p.TrainingProtocol | None = None,
        validate_step_fn: p.ValidationProtocol | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Provides syntactic sugar for running the active learning loop.

        Calls ``Driver.run`` internally.

        Parameters
        ----------
        train_step_fn: p.TrainingProtocol | None = None
            The training function to use for training. If not provided, then the
            ``Driver.train_loop_fn`` will be used.
        validate_step_fn: p.ValidationProtocol | None = None
            The validation function to use for validation. If not provided, then
            validation will not be performed.
        args: Any
            Additional arguments to pass to the method. These will be passed to the
            training loop, metrology strategies, query strategies, and labeling strategies.
        kwargs: Any
            Additional keyword arguments to pass to the method. These will be passed to the
            training loop, metrology strategies, query strategies, and labeling strategies.
        """
        self.run(
            train_step_fn=train_step_fn,
            validate_step_fn=validate_step_fn,
            *args,
            **kwargs,
        )
