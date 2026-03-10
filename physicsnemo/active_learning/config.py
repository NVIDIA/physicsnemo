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
Configuration dataclasses for the active learning driver.

This module provides structured configuration classes that separate different
concerns in the active learning workflow: optimization, training, strategies,
and driver orchestration.
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from json import dumps
from pathlib import Path
from typing import Any
from warnings import warn

import torch
from torch import distributed as dist
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from physicsnemo.active_learning import protocols as p
from physicsnemo.active_learning._registry import registry
from physicsnemo.active_learning.loop import DefaultTrainingLoop
from physicsnemo.distributed import DistributedManager


@dataclass
class OptimizerConfig:
    """
    Configuration for optimizer and learning rate scheduler.

    This encapsulates all training optimization parameters, keeping
    them separate from the active learning orchestration logic.

    Attributes
    ----------
    optimizer_cls: type
        The optimizer class to use. Defaults to `AdamW`.
    optimizer_kwargs: dict
        Keyword arguments to pass to the optimizer constructor.
        Defaults to {"lr": 1e-4}.
    scheduler_cls: type or None
        The learning rate scheduler class to use. If None, no
        scheduler will be configured.
    scheduler_kwargs: dict
        Keyword arguments to pass to the scheduler constructor.

    See Also
    --------
    TrainingConfig : Uses this config for optimizer setup
    Driver : Configures optimizer using this config
    """

    optimizer_cls: type[Optimizer] = AdamW
    optimizer_kwargs: dict[str, Any] = field(default_factory=lambda: {"lr": 1e-4})
    scheduler_cls: type[_LRScheduler] | None = None
    scheduler_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate optimizer configuration."""
        # Validate learning rate if present
        if "lr" in self.optimizer_kwargs:
            lr = self.optimizer_kwargs["lr"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                raise ValueError(f"Learning rate must be positive, got {lr}")

        # Validate that scheduler_kwargs is only set if scheduler_cls is provided
        if self.scheduler_kwargs and self.scheduler_cls is None:
            raise ValueError(
                "scheduler_kwargs provided but scheduler_cls is None. "
                "Provide a scheduler_cls or remove scheduler_kwargs."
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Returns a JSON-serializable dictionary representation of the OptimizerConfig.

        For round-tripping, the registry is used to de-serialize the optimizer and scheduler
        classes.

        Returns
        -------
        dict[str, Any]
            A dictionary that can be JSON serialized.
        """
        opt = {
            "__name__": self.optimizer_cls.__name__,
            "__module__": self.optimizer_cls.__module__,
        }
        if self.scheduler_cls:
            scheduler = {
                "__name__": self.scheduler_cls.__name__,
                "__module__": self.scheduler_cls.__module__,
            }
        else:
            scheduler = None
        return {
            "optimizer_cls": opt,
            "optimizer_kwargs": self.optimizer_kwargs,
            "scheduler_cls": scheduler,
            "scheduler_kwargs": self.scheduler_kwargs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizerConfig:
        """
        Creates an OptimizerConfig instance from a dictionary.

        This method assumes that the optimizer and scheduler classes are
        included in the ``physicsnemo.active_learning.registry``, or
        a module path is specified to import the class from.

        Parameters
        ----------
        data: dict[str, Any]
            A dictionary that was previously serialized using the ``to_dict`` method.

        Returns
        -------
        OptimizerConfig
            A new ``OptimizerConfig`` instance.
        """
        optimizer_cls = registry.get_class(
            data["optimizer_cls"]["__name__"], data["optimizer_cls"]["__module__"]
        )
        if (s := data.get("scheduler_cls")) is not None:
            scheduler_cls = registry.get_class(s["__name__"], s["__module__"])
        else:
            scheduler_cls = None
        return cls(
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=data["optimizer_kwargs"],
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=data["scheduler_kwargs"],
        )


@dataclass
class TrainingConfig:
    """
    Configuration for the training phase of active learning.

    This groups all training-related components together, making it
    clear when training is or isn't being used in the AL workflow.

    Attributes
    ----------
    train_datapool: :class:`~physicsnemo.active_learning.protocols.DataPool`
        The pool of labeled data to use for training.
    max_training_epochs: int
        The maximum number of epochs to train for. If ``max_fine_tuning_epochs``
        isn't specified, this value is used for all active learning steps.
    val_datapool: :class:`~physicsnemo.active_learning.protocols.DataPool` or None
        Optional pool of data to use for validation during training.
    optimizer_config: :class:`OptimizerConfig`
        Configuration for the optimizer and scheduler. Defaults to
        AdamW with lr=1e-4, no scheduler.
    max_fine_tuning_epochs: int or None
        The maximum number of epochs used during fine-tuning steps, i.e. after
        the first active learning step. If None, then the fine-tuning will
        be performed for the duration of the active learning loop.
    train_loop_fn: :class:`~physicsnemo.active_learning.protocols.TrainingLoop`
        The training loop function that orchestrates the training process.
        This defaults to a concrete implementation, :class:`~physicsnemo.active_learning.loop.DefaultTrainingLoop`,
        which provides a very typical loop that includes the use of static
        capture, etc.

    See Also
    --------
    Driver : Uses this config for training
    OptimizerConfig : Optimizer configuration
    StrategiesConfig : Strategies configuration
    DefaultTrainingLoop : Default training loop implementation
    """

    train_datapool: p.DataPool
    max_training_epochs: int
    val_datapool: p.DataPool | None = None
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    max_fine_tuning_epochs: int | None = None
    train_loop_fn: p.TrainingLoop = field(default_factory=DefaultTrainingLoop)

    def __post_init__(self) -> None:
        """Validate training configuration."""
        # Validate datapools have consistent interface
        if not hasattr(self.train_datapool, "__len__"):
            raise ValueError("train_datapool must implement __len__")
        if self.val_datapool is not None and not hasattr(self.val_datapool, "__len__"):
            raise ValueError("val_datapool must implement __len__")

        # Validate training loop is callable
        if not callable(self.train_loop_fn):
            raise ValueError("train_loop_fn must be callable")

        # set the same value for fine tuning epochs if not provided
        if self.max_fine_tuning_epochs is None:
            self.max_fine_tuning_epochs = self.max_training_epochs

    def to_dict(self) -> dict[str, Any]:
        """
        Returns a JSON-serializable dictionary representation of the TrainingConfig.

        For round-tripping, the registry is used to de-serialize the training loop
        and optimizer configuration. Note that datapools (train_datapool and val_datapool)
        are NOT serialized as they typically contain large datasets, file handles, or other
        non-serializable state.

        Returns
        -------
        dict[str, Any]
            A dictionary that can be JSON serialized. Excludes datapools.

        Warnings
        --------
        This method will issue a warning about the exclusion of datapools.
        """
        # Warn about datapool exclusion
        warn(
            "The `train_datapool` and `val_datapool` attributes are not supported for "
            "serialization and will be excluded from the ``TrainingConfig`` dictionary. "
            "You must re-provide these datapools when deserializing."
        )

        # Serialize optimizer config
        optimizer_dict = self.optimizer_config.to_dict()

        # Serialize training loop function
        if not hasattr(self.train_loop_fn, "_args"):
            raise ValueError(
                f"Training loop {self.train_loop_fn} does not have an `_args` attribute "
                "which is required for serialization. Make sure your training loop "
                "either subclasses `ActiveLearningProtocol` or implements the `__new__` "
                "method to capture object arguments."
            )

        train_loop_dict = self.train_loop_fn._args

        return {
            "max_training_epochs": self.max_training_epochs,
            "max_fine_tuning_epochs": self.max_fine_tuning_epochs,
            "optimizer_config": optimizer_dict,
            "train_loop_fn": train_loop_dict,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> TrainingConfig:
        """
        Creates a TrainingConfig instance from a dictionary.

        This method assumes that the training loop class is included in the
        ``physicsnemo.active_learning.registry``, or a module path is specified
        to import the class from. Note that datapools must be provided via
        kwargs as they are not serialized.

        Parameters
        ----------
        data: dict[str, Any]
            A dictionary that was previously serialized using the ``to_dict`` method.
        **kwargs: Any
            Additional keyword arguments to pass to the constructor. This is where
            you must provide ``train_datapool`` and optionally ``val_datapool``.

        Returns
        -------
        TrainingConfig
            A new ``TrainingConfig`` instance.

        Raises
        ------
        ValueError
            If required datapools are not provided in kwargs, if the data contains
            unexpected keys, or if object construction fails.
        """
        # Ensure required datapools are provided
        if "train_datapool" not in kwargs:
            raise ValueError(
                "``train_datapool`` must be provided in kwargs when deserializing "
                "TrainingConfig, as datapools are not serialized."
            )

        # Reconstruct optimizer config
        optimizer_config = OptimizerConfig.from_dict(data["optimizer_config"])

        # Reconstruct training loop function
        train_loop_data = data["train_loop_fn"]
        train_loop_fn = registry.construct(
            train_loop_data["__name__"],
            module_path=train_loop_data["__module__"],
            **train_loop_data["__args__"],
        )

        # Build the config
        try:
            config = cls(
                max_training_epochs=data["max_training_epochs"],
                max_fine_tuning_epochs=data.get("max_fine_tuning_epochs"),
                optimizer_config=optimizer_config,
                train_loop_fn=train_loop_fn,
                **kwargs,
            )
        except Exception as e:
            raise ValueError(
                "Failed to construct ``TrainingConfig`` from dictionary."
            ) from e

        return config


@dataclass
class StrategiesConfig:
    """
    Configuration for active learning strategies and data acquisition.

    This encapsulates the query-label-metrology cycle that is at the
    heart of active learning: strategies for selecting data, labeling it,
    and measuring model uncertainty/performance.

    Attributes
    ----------
    query_strategies: list
        The query strategies to use for selecting data to label. Each element should be a
        :class:`~physicsnemo.active_learning.protocols.QueryStrategy` instance.
    queue_cls: type
        The queue implementation to use for passing data between
        query and labeling phases. Should implement :class:`~physicsnemo.active_learning.protocols.AbstractQueue`.
    label_strategy: :class:`~physicsnemo.active_learning.protocols.LabelStrategy` or None
        The strategy to use for labeling queried data. If None,
        labeling will be skipped.
    metrology_strategies: list or None
        Strategies for measuring model performance and uncertainty. Each element should be a
        :class:`~physicsnemo.active_learning.protocols.MetrologyStrategy` instance.
        If None, metrology will be skipped.
    unlabeled_datapool: :class:`~physicsnemo.active_learning.protocols.DataPool` or None
        Pool of unlabeled data that query strategies can sample from.
        Not all strategies require this (some may generate synthetic data).

    See Also
    --------
    Driver : Uses this config for strategy orchestration
    QueryStrategy : Query strategy protocol
    LabelStrategy : Label strategy protocol
    MetrologyStrategy : Metrology strategy protocol
    """

    query_strategies: list[p.QueryStrategy]
    queue_cls: type[p.AbstractQueue]
    label_strategy: p.LabelStrategy | None = None
    metrology_strategies: list[p.MetrologyStrategy] | None = None
    unlabeled_datapool: p.DataPool | None = None

    def __post_init__(self) -> None:
        """Validate strategies configuration."""
        # Must have at least one query strategy
        if not self.query_strategies:
            raise ValueError(
                "At least one query strategy must be provided. "
                "Active learning requires a mechanism to select data."
            )

        # All query strategies must be callable
        for strategy in self.query_strategies:
            if not callable(strategy):
                raise ValueError(f"Query strategy {strategy} must be callable")

        # Label strategy must be callable if provided
        if self.label_strategy is not None and not callable(self.label_strategy):
            raise ValueError("label_strategy must be callable")

        # Metrology strategies must be callable if provided
        if self.metrology_strategies is not None:
            if not self.metrology_strategies:
                raise ValueError(
                    "metrology_strategies is an empty list. "
                    "Either provide strategies or set to None to skip metrology."
                )
            for strategy in self.metrology_strategies:
                if not callable(strategy):
                    raise ValueError(f"Metrology strategy {strategy} must be callable")

        # Validate queue class has basic queue interface
        if not hasattr(self.queue_cls, "__call__"):
            raise ValueError("queue_cls must be a callable class")

    def to_dict(self) -> dict[str, Any]:
        """
        Method that converts the present ``StrategiesConfig`` instance into a dictionary
        that can be JSON serialized.

        This method, for the most part, assumes that strategies are subclasses of
        ``ActiveLearningProtocol`` and/or they have an ``_args`` attribute that
        captures the arguments to the constructor.

        One issue is the inability to reliably serialize the ``unlabeled_datapool``,
        which for the most part, likely does not need serialization as a dataset.
        Regardless, this method will trigger a warning if ``unlabeled_datapool`` is
        not None.

        Returns
        -------
        dict[str, Any]
            A dictionary that can be JSON serialized.
        """
        output = defaultdict(list)
        for strategy in self.query_strategies:
            if not hasattr(strategy, "_args"):
                raise ValueError(
                    f"Query strategy {strategy} does not have an `_args` attribute"
                    " which is required for serialization. Make sure your strategy"
                    " either subclasses `ActiveLearningProtocol` or implements"
                    " the `__new__` method to capture object arguments."
                )
            output["query_strategies"].append(strategy._args)
        if self.label_strategy is not None:
            if not hasattr(self.label_strategy, "_args"):
                raise ValueError(
                    f"Label strategy {self.label_strategy} does not have an `_args` attribute"
                    " which is required for serialization. Make sure your strategy"
                    " either subclasses `ActiveLearningProtocol` or implements"
                    " the `__new__` method to capture object arguments."
                )
            output["label_strategy"] = self.label_strategy._args
        output["queue_cls"] = {
            "__name__": self.queue_cls.__name__,
            "__module__": self.queue_cls.__module__,
        }
        if self.metrology_strategies is not None:
            for strategy in self.metrology_strategies:
                if not hasattr(strategy, "_args"):
                    raise ValueError(
                        f"Metrology strategy {strategy} does not have an `_args` attribute"
                        " which is required for serialization. Make sure your strategy"
                        " either subclasses `ActiveLearningProtocol` or implements"
                        " the `__new__` method to capture object arguments."
                    )
                output["metrology_strategies"].append(strategy._args)
        if self.unlabeled_datapool is not None:
            warn(
                "The `unlabeled_datapool` attribute is not supported for serialization"
                " and will be excluded from the ``StrategiesConfig`` dictionary."
            )
        return output

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> StrategiesConfig:
        """
        Create a ``StrategiesConfig`` instance from a dictionary.

        This method heavily relies on classes being added to the
        ``physicsnemo.active_learning.registry``, which is used to instantiate
        all strategies and custom types used in active learning. As a fall
        back, the `registry.construct` method will try and import the class
        from the module path if it is not found in the registry.

        Parameters
        ----------
        data: dict[str, Any]
            A dictionary that was previously serialized using the ``to_dict`` method.
        **kwargs: Any
            Additional keyword arguments to pass to the constructor.

        Returns
        -------
        StrategiesConfig
            A new ``StrategiesConfig`` instance.

        Raises
        ------
        ValueError:
            If the data contains unexpected keys or if the object construction fails.
        NameError:
            If a class is not found in the registry and no module path is provided.
        ModuleNotFoundError:
            If a module is not found with the specified module path.
        """
        # ensure that the data contains no unexpected keys
        data_keys = set(data.keys())
        expected_keys = set(cls.__dataclass_fields__.keys())
        extra_keys = data_keys - expected_keys
        if extra_keys:
            raise ValueError(
                f"Unexpected keys in data: {extra_keys}. Expected keys are {expected_keys}."
            )
        # instantiate objects from the serialized data; general strategy is to
        # use `registry.construct` that will try and resolve the class within
        # the registry first, and if not found, then it will try and import the
        # class from the module path.
        output_dict = defaultdict(list)
        for entry in data["query_strategies"]:
            output_dict["query_strategies"].append(
                registry.construct(
                    entry["__name__"],
                    module_path=entry["__module__"],
                    **entry["__args__"],
                )
            )
        if "metrology_strategies" in data:
            for entry in data["metrology_strategies"]:
                output_dict["metrology_strategies"].append(
                    registry.construct(
                        entry["__name__"],
                        module_path=entry["__module__"],
                        **entry["__args__"],
                    )
                )
        if "label_strategy" in data:
            output_dict["label_strategy"] = registry.construct(
                data["label_strategy"]["__name__"],
                module_path=data["label_strategy"]["__module__"],
                **data["label_strategy"]["__args__"],
            )
        output_dict["queue_cls"] = registry.get_class(
            data["queue_cls"]["__name__"], data["queue_cls"]["__module__"]
        )
        # potentially override with keyword arguments
        output_dict.update(kwargs)
        try:
            config = cls(**output_dict)
        except Exception as e:
            raise ValueError(
                "Failed to construct ``StrategiesConfig`` from dictionary."
            ) from e
        return config


@dataclass
class DriverConfig:
    """
    Configuration for driver orchestration and infrastructure.

    This contains parameters that control the overall loop execution,
    logging, checkpointing, and distributed training setup - orthogonal
    to the specific AL or training logic.

    Attributes
    ----------
    batch_size: int
        The batch size to use for data loaders.
    max_active_learning_steps: int or None
        Maximum number of AL iterations to perform. None means infinite.
    run_id: str
        Unique identifier for this run. Auto-generated if not provided.
    fine_tuning_lr: float or None
        Learning rate to switch to after the first AL step for fine-tuning.
    reset_optim_states: bool
        Whether to reset optimizer states between AL steps. Defaults to True.
    skip_training: bool
        If True, skip the training phase entirely. Defaults to False.
    skip_metrology: bool
        If True, skip the metrology phase entirely. Defaults to False.
    skip_labeling: bool
        If True, skip the labeling phase entirely. Defaults to False.
    checkpoint_interval: int
        Save model checkpoint every N AL steps. 0 disables checkpointing. Defaults to 1.
    checkpoint_on_training: bool
        If True, save checkpoint at the start of the training phase. Defaults to False.
    checkpoint_on_metrology: bool
        If True, save checkpoint at the start of the metrology phase. Defaults to False.
    checkpoint_on_query: bool
        If True, save checkpoint at the start of the query phase. Defaults to False.
    checkpoint_on_labeling: bool
        If True, save checkpoint at the start of the labeling phase. Defaults to True.
    model_checkpoint_frequency: int
        Save model weights every N epochs during training. 0 means only save
        between active learning phases. Useful for mid-training restarts. Defaults to 0.
    root_log_dir: str or :class:`pathlib.Path`
        Directory to save logs and checkpoints to. Defaults to
        an 'active_learning_logs' directory in the current working directory.
    dist_manager: :class:`~physicsnemo.distributed.DistributedManager` or None
        Manager for distributed training configuration.
    collate_fn: callable or None
        Custom collate function for batching data.
    num_dataloader_workers: int
        Number of worker processes for data loading. Defaults to 0.
    device: str or `torch.device` or None
        Device to use for model and data. This is intended for single process
        workflows; for distributed workflows, the device should be set in
        :class:`~physicsnemo.distributed.DistributedManager` instead. If not specified, then the device
        will default to ``torch.get_default_device()``.
    dtype: `torch.dtype` or None
        The dtype to use for model and data, and AMP contexts. If not provided,
        then the dtype will default to ``torch.get_default_dtype()``.

    See Also
    --------
    Driver : Uses this config for orchestration
    TrainingConfig : Training configuration
    StrategiesConfig : Strategies configuration
    DataPool : Data pool protocol
    AbstractQueue : Queue protocol
    """

    batch_size: int
    max_active_learning_steps: int | None = None
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fine_tuning_lr: float | None = None  # TODO: move to TrainingConfig
    reset_optim_states: bool = True
    skip_training: bool = False
    skip_metrology: bool = False
    skip_labeling: bool = False
    checkpoint_interval: int = 1
    checkpoint_on_training: bool = False
    checkpoint_on_metrology: bool = False
    checkpoint_on_query: bool = False
    checkpoint_on_labeling: bool = True
    model_checkpoint_frequency: int = 0
    root_log_dir: str | Path = field(default=Path.cwd() / "active_learning_logs")
    dist_manager: DistributedManager | None = None
    collate_fn: callable | None = None
    num_dataloader_workers: int = 0
    device: str | torch.device | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        """Validate driver configuration."""
        if self.max_active_learning_steps is None:
            self.max_active_learning_steps = float("inf")

        if (
            self.max_active_learning_steps is not None
            and self.max_active_learning_steps <= 0
        ):
            raise ValueError(
                "`max_active_learning_steps` must be a positive integer or None."
            )

        if not math.isfinite(self.batch_size) or self.batch_size <= 0:
            raise ValueError("`batch_size` must be a positive integer.")

        if not math.isfinite(self.checkpoint_interval) or self.checkpoint_interval < 0:
            raise ValueError(
                "`checkpoint_interval` must be a non-negative integer. "
                "Use 0 to disable checkpointing."
            )

        if self.fine_tuning_lr is not None and self.fine_tuning_lr <= 0:
            raise ValueError("`fine_tuning_lr` must be positive if provided.")

        if self.num_dataloader_workers < 0:
            raise ValueError("`num_dataloader_workers` must be non-negative.")

        if self.model_checkpoint_frequency < 0:
            raise ValueError("`model_checkpoint_frequency` must be non-negative.")

        if isinstance(self.root_log_dir, str):
            self.root_log_dir = Path(self.root_log_dir)

        # Validate collate_fn if provided
        if self.collate_fn is not None and not callable(self.collate_fn):
            raise ValueError("`collate_fn` must be callable if provided.")

        # device and dtype setup when not using DistributedManager
        if self.device is None and not self.dist_manager:
            self.device = torch.get_default_device()
        if self.dtype is None:
            self.dtype = torch.get_default_dtype()

    def to_json(self) -> str:
        """
        Returns a JSON string representation of the ``DriverConfig``.

        Note that certain fields are not serialized and must be provided when
        deserializing: ``dist_manager``, ``collate_fn``.

        Returns
        -------
        str
            A JSON string representation of the config.
        """
        # base dict representation skips Python objects
        dict_repr = {
            key: self.__dict__[key]
            for key in self.__dict__
            if key
            not in ["dist_manager", "collate_fn", "root_log_dir", "device", "dtype"]
        }
        # Note: checkpoint flags are included in dict_repr automatically
        dict_repr["default_dtype"] = str(torch.get_default_dtype())
        dict_repr["log_dir"] = str(self.root_log_dir)
        # Convert dtype to string for JSON serialization
        if self.dtype is not None:
            dict_repr["dtype"] = str(self.dtype)
        else:
            dict_repr["dtype"] = None
        if self.dist_manager is not None:
            dict_repr["world_size"] = self.dist_manager.world_size
            dict_repr["device"] = self.dist_manager.device.type
            dict_repr["dist_manager_init_method"] = (
                self.dist_manager._initialization_method
            )
        else:
            if dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1
            dict_repr["world_size"] = world_size
            if self.device is not None:
                dict_repr["device"] = (
                    str(self.device)
                    if hasattr(self.device, "type")
                    else str(self.device)
                )
            else:
                dict_repr["device"] = torch.get_default_device().type
            dict_repr["dist_manager_init_method"] = None
        if self.collate_fn is not None:
            dict_repr["collate_fn"] = self.collate_fn.__name__
        else:
            dict_repr["collate_fn"] = None
        return dumps(dict_repr, indent=2)

    @classmethod
    def from_json(cls, json_str: str, **kwargs: Any) -> DriverConfig:
        """
        Creates a DriverConfig instance from a JSON string.

        This method reconstructs a DriverConfig from JSON. Note that certain
        fields cannot be serialized and must be provided via kwargs:
        - ``dist_manager``: DistributedManager instance (optional)
        - ``collate_fn``: Custom collate function (optional)

        Parameters
        ----------
        json_str: str
            A JSON string that was previously serialized using ``to_json()``.
        **kwargs: Any
            Additional keyword arguments to override or provide non-serializable
            fields like ``dist_manager`` and ``collate_fn``.

        Returns
        -------
        DriverConfig
            A new ``DriverConfig`` instance.

        Raises
        ------
        ValueError
            If the JSON cannot be parsed or required fields are missing.

        Notes
        -----
        The device and dtype fields are reconstructed from their string
        representations. The ``log_dir`` field in JSON is mapped to
        ``root_log_dir`` in the config.
        """
        import json

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}") from e

        # Define fields that are not actual DriverConfig constructor parameters
        metadata_fields = [
            "default_dtype",
            "world_size",
            "dist_manager_init_method",
            "log_dir",  # handled separately as root_log_dir
        ]
        non_serializable_fields = [
            "dist_manager",
            "collate_fn",
            "root_log_dir",
            "device",
            "dtype",
        ]

        # Extract serializable fields that map directly
        config_fields = {
            key: value
            for key, value in data.items()
            if key not in metadata_fields + non_serializable_fields
        }

        # Handle root_log_dir (stored as "log_dir" in JSON)
        if "log_dir" in data:
            config_fields["root_log_dir"] = Path(data["log_dir"])

        # Handle device reconstruction from string
        if "device" in data and data["device"] is not None:
            device_str = data["device"]
            # Parse device strings like "cuda:0", "cpu", "cuda", etc.
            config_fields["device"] = torch.device(device_str)

        # Handle dtype reconstruction from string
        if "dtype" in data and data["dtype"] is not None:
            dtype_str = data["dtype"]
            # Map string representations to torch dtypes
            dtype_map = {
                "torch.float32": torch.float32,
                "torch.float64": torch.float64,
                "torch.float16": torch.float16,
                "torch.bfloat16": torch.bfloat16,
                "torch.int32": torch.int32,
                "torch.int64": torch.int64,
                "torch.int8": torch.int8,
                "torch.uint8": torch.uint8,
            }
            if dtype_str in dtype_map:
                config_fields["dtype"] = dtype_map[dtype_str]
            else:
                warn(
                    f"Unknown dtype string '{dtype_str}' in JSON. "
                    "Using default dtype instead."
                )

        # Merge with provided kwargs (allows overriding and adding non-serializable fields)
        config_fields.update(kwargs)

        # Create the config
        try:
            config = cls(**config_fields)
        except Exception as e:
            raise ValueError(
                "Failed to construct ``DriverConfig`` from JSON string."
            ) from e

        return config
