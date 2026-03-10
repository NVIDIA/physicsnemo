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

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from physicsnemo.active_learning import protocols as p
from physicsnemo.core import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.capture import StaticCaptureEvaluateNoGrad, StaticCaptureTraining
from physicsnemo.utils.logging import LaunchLogger

__all__ = ["DefaultTrainingLoop"]


def _recursive_data_device_cast(
    data: Any,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> Any:
    """
    Recursively moves/cast input data to a specified device and dtype.

    For iterable objects, we recurse through the elements depending on
    the type of iterable until we reach an object that either has a ``to``
    method that can be called, or just returns the data unchanged.

    Parameters
    ----------
    data: Any
        The data to move to the device.
    device: torch.device | str | None = None
        The device to move the data to.
    dtype: torch.dtype | None = None
        The dtype to move the data to.
    kwargs: Any
        Additional keyword arguments to pass to the `to` method.
        By default, `non_blocking` is set to `True` to allow
        asynchronous data transfers.

    Returns
    -------
    Any
        The data moved to the device.
    """
    kwargs.setdefault("non_blocking", True)
    if hasattr(data, "to"):
        # if there is a `to` method, then we can just call it
        return data.to(device=device, dtype=dtype, **kwargs)
    elif isinstance(data, dict):
        return {
            k: _recursive_data_device_cast(v, device, dtype) for k, v in data.items()
        }
    elif isinstance(data, list):
        return [_recursive_data_device_cast(v, device, dtype) for v in data]
    elif isinstance(data, tuple):
        return tuple(_recursive_data_device_cast(v, device, dtype) for v in data)
    else:
        return data


class DefaultTrainingLoop(p.TrainingLoop):
    """
    Default implementation of the :class:`~physicsnemo.active_learning.protocols.TrainingLoop` protocol.

    This provides a functional training loop with support for static capture,
    progress bars, checkpointing, and distributed training. It implements the
    standard epoch-based training pattern with optional validation.

    See Also
    --------
    TrainingLoop : Protocol specification for training loops
    Driver : Uses training loops in the training phase
    TrainingConfig : Configuration for training
    TrainingProtocol : Training step protocol
    ValidationProtocol : Validation step protocol
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> DefaultTrainingLoop:
        """
        Wrapper for instantiating DefaultTrainingLoop.

        This method captures arguments used to instantiate the loop
        and stores them in the ``_args`` attribute for serialization.
        This follows the same pattern as :meth:`~physicsnemo.active_learning.protocols.ActiveLearningProtocol.__new__`.

        Parameters
        ----------
        args: Any
            Arguments to pass to the loop's constructor.
        kwargs: Any
            Keyword arguments to pass to the loop's constructor.

        Returns
        -------
        :class:`DefaultTrainingLoop`
            A new instance with an ``_args`` attribute for serialization.
        """
        out = super().__new__(cls)

        # Get signature of __init__ function
        sig = inspect.signature(cls.__init__)

        # Bind args and kwargs to signature
        bound_args = sig.bind_partial(
            *([None] + list(args)), **kwargs
        )  # Add None to account for self
        bound_args.apply_defaults()

        # Get args and kwargs (excluding self and unroll kwargs)
        instantiate_args = {}
        for param, (k, v) in zip(sig.parameters.values(), bound_args.arguments.items()):
            # Skip self
            if k == "self":
                continue

            # Add args and kwargs to instantiate_args
            if param.kind == param.VAR_KEYWORD:
                instantiate_args.update(v)
            else:
                # Special handling for device: convert torch.device to string
                if k == "device" and isinstance(v, torch.device):
                    instantiate_args[k] = str(v)
                # Special handling for dtype: convert to string representation
                elif k == "dtype" and isinstance(v, torch.dtype):
                    instantiate_args[k] = str(v)
                else:
                    instantiate_args[k] = v

        # Store args needed for instantiation
        out._args = {
            "__name__": cls.__name__,
            "__module__": cls.__module__,
            "__args__": instantiate_args,
        }
        return out

    def __init__(
        self,
        train_step_fn: p.TrainingProtocol | None = None,
        validate_step_fn: p.ValidationProtocol | None = None,
        enable_static_capture: bool = True,
        use_progress_bars: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        checkpoint_frequency: int = 0,
        **capture_kwargs: Any,
    ) -> None:
        """
        Initializes the default training loop.

        The general usage of this loop is to

        TODO: add support for early stopping

        Parameters
        ----------
        train_step_fn: :class:`~physicsnemo.active_learning.protocols.TrainingProtocol` or None
            A callable that implements the logic for performing a single
            training step. See :class:`~physicsnemo.active_learning.protocols.TrainingProtocol` for the expected
            interface, but ultimately the function should return a scalar loss
            value that has a ``backward`` method.
        validate_step_fn: :class:`~physicsnemo.active_learning.protocols.ValidationProtocol` or None
            A callable that implements the logic for performing a single
            validation step. See :class:`~physicsnemo.active_learning.protocols.ValidationProtocol` for the expected
            interface, but in contrast to ``train_step_fn`` this function should
            not return anything.
        enable_static_capture: bool
            Whether to enable static capture for the training and validation steps. Defaults to True.
        use_progress_bars: bool
            Whether to show ``tqdm`` progress bars to display epoch and step progress. Defaults to True.
        device: str or `torch.device` or None
            The device used for performing the loop. If not provided, then the device
            will default to the model's device at runtime.
        dtype: `torch.dtype` or None
            The dtype used for performing the loop. If not provided, then the dtype
            will default to ``torch.get_default_dtype()``.
        checkpoint_frequency: int
            How often to save checkpoints during training (every N epochs).
            If 0, no checkpoints are saved during training. Set via :class:`~physicsnemo.active_learning.driver.Driver` before
            training execution. Defaults to 0.
        capture_kwargs: Any
            Additional keyword arguments to pass to the static capture decorators.
        """
        self.train_step_fn = train_step_fn
        self.validate_step_fn = validate_step_fn
        self.enable_static_capture = enable_static_capture
        if isinstance(device, str):
            device = torch.device(device)
        # check to see if we can rely on DistributedManager
        if device is None and DistributedManager.is_initialized():
            device = DistributedManager.device
        self.device = device
        if dtype is None:
            dtype = torch.get_default_dtype()
        self.dtype = dtype
        self.capture_kwargs = capture_kwargs
        self.use_progress_bars = use_progress_bars
        self.capture_functions = {}
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_base_dir: Path | None = None

    def save_training_checkpoint(
        self,
        checkpoint_dir: Path,
        model: Module | p.LearnerProtocol,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler | None = None,
        training_epoch: int | None = None,
    ) -> None:
        """
        Save training state to checkpoint directory.

        Model weights are saved separately. Optimizer, scheduler, and epoch
        metadata are combined into a single training_state.pt file.

        Parameters
        ----------
        checkpoint_dir: :class:`pathlib.Path`
            Directory to save checkpoint files.
        model: :class:`~physicsnemo.Module` or :class:`~physicsnemo.active_learning.protocols.LearnerProtocol`
            Model to save weights for.
        optimizer: `Optimizer`
            Optimizer to save state from.
        lr_scheduler: `_LRScheduler` or None
            Optional LR scheduler to save state from.
        training_epoch: int or None
            Current training epoch for metadata.
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights separately
        if isinstance(model, Module):
            model_path = checkpoint_dir / "model.mdlus"
            model.save(str(model_path))
        else:
            model_path = checkpoint_dir / "model_state.pt"
            torch.save(model.state_dict(), model_path)

        # Combine optimizer, scheduler, and epoch metadata into single file
        training_state = {
            "optimizer_state": optimizer.state_dict(),
            "lr_scheduler_state": lr_scheduler.state_dict() if lr_scheduler else None,
            "training_epoch": training_epoch,
        }
        training_state_path = checkpoint_dir / "training_state.pt"
        torch.save(training_state, training_state_path)

    @staticmethod
    def load_training_checkpoint(
        checkpoint_dir: Path,
        model: Module | p.LearnerProtocol,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler | None = None,
    ) -> int | None:
        """
        Load training state from checkpoint directory.

        Model weights are loaded separately. Optimizer, scheduler, and epoch
        metadata are loaded from the combined training_state.pt file.

        Parameters
        ----------
        checkpoint_dir: :class:`pathlib.Path`
            Directory containing checkpoint files.
        model: :class:`~physicsnemo.Module` or :class:`~physicsnemo.active_learning.protocols.LearnerProtocol`
            Model to load weights into.
        optimizer: `Optimizer`
            Optimizer to load state into.
        lr_scheduler: `_LRScheduler` or None
            Optional LR scheduler to load state into.

        Returns
        -------
        int or None
            Training epoch from metadata if available, else None.
        """
        # Load model weights separately
        if isinstance(model, Module):
            model_path = checkpoint_dir / "model.mdlus"
            if model_path.exists():
                model.load(str(model_path))
        else:
            model_state_path = checkpoint_dir / "model_state.pt"
            if model_state_path.exists():
                state_dict = torch.load(model_state_path, map_location="cpu")
                model.load_state_dict(state_dict)

        # Load combined training state (optimizer, scheduler, epoch)
        training_state_path = checkpoint_dir / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location="cpu")

            # Restore optimizer state
            if "optimizer_state" in training_state:
                optimizer.load_state_dict(training_state["optimizer_state"])

            # Restore scheduler state if present
            if lr_scheduler and training_state.get("lr_scheduler_state"):
                lr_scheduler.load_state_dict(training_state["lr_scheduler_state"])

            # Return epoch metadata
            return training_state.get("training_epoch", None)

        return None

    @property
    def amp_type(self) -> torch.dtype:
        if self.dtype in [torch.float16, torch.bfloat16]:
            return self.dtype
        else:
            return torch.float16

    def _create_capture_functions(
        self,
        model: Module | p.LearnerProtocol,
        optimizer: Optimizer,
        train_step_fn: p.TrainingProtocol | None = None,
        validate_step_fn: p.ValidationProtocol | None = None,
    ) -> tuple[p.TrainingProtocol | None, p.ValidationProtocol | None]:
        """
        Attempt to create static capture functions based off training and validation
        functions.

        This uses the Python object IDs to unique identify functions, and adds the
        decorated functions to an internal `capture_functions` dictionary. If the
        decorated functions already exist, then this function will be no-op.

        Parameters
        ----------
        model: Module | p.LearnerProtocol
            The model to train.
        optimizer: Optimizer
            The optimizer to use for training.
        train_step_fn: p.TrainingProtocol | None = None
            The training function to use for training.
        validate_step_fn: p.ValidationProtocol | None = None
            The validation function to use for validation.

        Returns
        -------
        tuple[p.TrainingProtocol | None, p.ValidationProtocol | None]
            The training and validation functions with static capture applied.
        """
        if not train_step_fn:
            train_step_fn = self.train_step_fn
        train_func_id = id(train_step_fn)
        if train_func_id not in self.capture_functions:
            try:
                train_step_fn = StaticCaptureTraining(
                    model=model,
                    optim=optimizer,
                    amp_type=self.amp_type,
                    **self.capture_kwargs,
                )(train_step_fn)
                self.capture_functions[train_func_id] = train_step_fn
            except Exception as e:
                raise RuntimeError(
                    "Failed to create static capture for `train_step_fn`. "
                ) from e
        else:
            train_step_fn = self.capture_functions[train_func_id]
        if not validate_step_fn:
            validate_step_fn = self.validate_step_fn
        if validate_step_fn:
            val_func_id = id(validate_step_fn)
            if val_func_id not in self.capture_functions:
                try:
                    validate_step_fn = StaticCaptureEvaluateNoGrad(
                        model=model, amp_type=self.amp_type, **self.capture_kwargs
                    )(validate_step_fn)
                    self.capture_functions[val_func_id] = validate_step_fn
                except Exception as e:
                    raise RuntimeError(
                        "Failed to create static capture for `validate_step_fn`. "
                    ) from e
            else:
                validate_step_fn = self.capture_functions[val_func_id]
        return train_step_fn, validate_step_fn

    def __call__(
        self,
        model: Module | p.LearnerProtocol,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        max_epochs: int,
        validation_dataloader: DataLoader | None = None,
        train_step_fn: p.TrainingProtocol | None = None,
        validate_step_fn: p.ValidationProtocol | None = None,
        lr_scheduler: _LRScheduler | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Performs ``max_epochs`` epochs of training and optionally validation.

        Some of the arguments, such as ``train_step_fn`` and ``validate_step_fn``,
        are optional only if the ``model`` implements the ``p.LearnerProtocol``.
        If they are passed, however, they will take precedence over the methods
        originally provided to the constructor method.

        The bare minimum required arguments for this loop to work are:
        1. A model to train
        2. An optimizer to step
        3. A training dataloader to iterate over
        4. The maximum number of epochs to train for

        If validation is required, then both ``validation_dataloader`` and
        ``validate_step_fn`` must be specified.

        Parameters
        ----------
        model: Module | p.LearnerProtocol
            The model to train.
        optimizer: torch.optim.Optimizer
            The optimizer to use for training.
        train_dataloader: DataLoader
            The dataloader to use for training.
        max_epochs: int
            The number of epochs to train for.
        validation_dataloader: DataLoader | None
            The dataloader to use for validation. If not provided, then validation
            will not be performed.
        train_step_fn: p.TrainingProtocol | None = None
            The training function to use for training. If passed, it will take
            precedence over the method provided to the constructor method.
        validate_step_fn: p.ValidationProtocol | None = None
            The validation function to use for validation.
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
            The learning rate scheduler to use for training.
        device: str | torch.device | None = None
            The device used for performing the loop. If provided, it will
            override the device specified in the constructor. If both values
            are not provided, then we default to PyTorch's default device.
        dtype: torch.dtype | None = None
            The dtype used for performing the loop. If provided, it will
            override the dtype specified in the constructor. If both values
            are not provided, then we default to PyTorch's default dtype.
        args: Any
            Additional arguments to pass the training and validation
            step functions.
        kwargs: Any
            Additional keyword arguments to pass the training and validation
            step functions.
        """
        if not train_step_fn and not self.train_step_fn:
            raise RuntimeError(
                """
                No training step function provided.
                Either provide a `train_step_fn` to this constructor, or
                provide a `train_step_fn` to the `__call__` method.
                """
            )
        if not device and not self.device:
            device = torch.get_default_device()
        if not dtype and not self.dtype:
            dtype = torch.get_default_dtype()
        # if a device is specified, move the model
        if device and device != model.device:
            # not 100% sure this will trigger issues with the optimizer
            # but allows a potentially different device to be used
            model = model.to(device)
        if self.enable_static_capture:
            # if static capture is enabled, we check for a cache hit based on
            # the incoming function IDs. If we miss, we then create new wrappers.
            train_step_fn, validate_step_fn = self._create_capture_functions(
                model, optimizer, train_step_fn, validate_step_fn
            )
        epoch_iter = range(1, max_epochs + 1)
        if self.use_progress_bars:
            epoch_iter = tqdm(epoch_iter, desc="Epoch", leave=False, position=0)
        ########### EPOCH LOOP ###########
        for epoch in epoch_iter:
            model.train()
            train_iter = iter(train_dataloader)
            if self.use_progress_bars:
                train_iter = tqdm(
                    train_iter, desc="Training step", leave=False, unit="batch"
                )
            ########### TRAINING STEP LOOP ###########
            with LaunchLogger(
                "train", epoch=epoch, num_mini_batch=len(train_dataloader)
            ) as log:
                for batch in train_iter:
                    batch = _recursive_data_device_cast(
                        batch, device=device, dtype=dtype
                    )
                    model.zero_grad(set_to_none=True)
                    loss = train_step_fn(model, batch, *args, **kwargs)
                    log.log_minibatch({"train_loss": loss.detach().item()})
                    # normally, static capture will call backward because of AMP
                    if not self.enable_static_capture:
                        loss.backward()
                    optimizer.step()
                    if lr_scheduler:
                        lr_scheduler.step()
            ########### VALIDATION STEP LOOP ###########
            if validate_step_fn and validation_dataloader:
                model.eval()
                val_iter = iter(validation_dataloader)
                if self.use_progress_bars:
                    val_iter = tqdm(
                        val_iter, desc="Validation step", leave=False, unit="batch"
                    )
                with LaunchLogger(
                    "validation", epoch=epoch, num_mini_batch=len(validation_dataloader)
                ) as log:
                    for batch in val_iter:
                        batch = _recursive_data_device_cast(
                            batch, device=device, dtype=dtype
                        )
                        validate_step_fn(model, batch, *args, **kwargs)

            ########### CHECKPOINT SAVE ###########
            # Save training state at specified frequency
            if self.checkpoint_base_dir and self.checkpoint_frequency > 0:
                if epoch % self.checkpoint_frequency == 0:
                    epoch_checkpoint_dir = self.checkpoint_base_dir / f"epoch_{epoch}"
                    self.save_training_checkpoint(
                        checkpoint_dir=epoch_checkpoint_dir,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        training_epoch=epoch,
                    )
