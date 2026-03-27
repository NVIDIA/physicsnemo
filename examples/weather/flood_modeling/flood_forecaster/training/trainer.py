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

r"""
PhysicsNeMo-style Trainer for neural operator training.

This module provides a Trainer class rewritten from neuralop's Trainer to follow
PhysicsNeMo patterns and conventions. It integrates with PhysicsNeMo's checkpointing,
logging, and distributed training infrastructure.

Key features:
- PhysicsNeMo checkpoint system (save_checkpoint/load_checkpoint)
- DistributedManager for distributed training
- PhysicsNeMo logging patterns
- Support for data processors, regularizers, and mixed precision
- Best model tracking and interval checkpointing
- Autoregressive evaluation support
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from timeit import default_timer
from typing import Any, Dict, Literal, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import physicsnemo
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.utils.checkpoint import load_checkpoint, save_checkpoint

import fsspec

# Optional wandb import
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _has_pytorch_submodules(model: nn.Module) -> bool:
    r"""
    Check if a PhysicsNeMo Module contains PyTorch submodules that would prevent saving.
    
    PhysicsNeMo's Module.save() doesn't support saving modules that contain
    PyTorch submodules (they must be converted using Module.from_torch).
    This helper detects such cases so we can save them as PyTorch models instead.
    
    Note: With Option 1 implementation, GINOWrapper now auto-converts PyTorch models
    at initialization, so this check is mainly for backward compatibility and
    other edge cases.
    
    Parameters
    ----------
    model : nn.Module
        Model to check.
        
    Returns
    -------
    bool
        True if model is a PhysicsNeMo Module containing PyTorch submodules.
    """
    if not isinstance(model, physicsnemo.models.Module):
        return False
    
    # Check if any direct submodules are PyTorch modules (not PhysicsNeMo modules)
    # Skip checking inner_model of converted wrappers (they're intentionally PyTorch)
    for name, child in model.named_children():
        # Skip inner_model - it's a PyTorch model wrapped by PhysicsNeMo, which is fine
        if name == 'inner_model':
            continue
        if isinstance(child, torch.nn.Module) and not isinstance(child, physicsnemo.models.Module):
            return True
    return False


def save_model_checkpoint(
    model: nn.Module,
    save_dir: Union[str, Path],
    epoch: int,
    metadata: Optional[Dict[str, Any]] = None,
    model_parallel_rank: int = 0,
) -> bool:
    r"""
    Save a model checkpoint, handling both PhysicsNeMo modules and wrappers with PyTorch submodules.
    
    This function intelligently saves models:
    - Pure PhysicsNeMo modules: Returns False (caller should use save_checkpoint normally)
    - Wrappers with PyTorch submodules: Saves state_dict as PyTorch model, returns True
    
    Parameters
    ----------
    model : nn.Module
        Model to save. Can be a PhysicsNeMo Module or a wrapper.
    save_dir : str or Path
        Directory to save checkpoint.
    epoch : int
        Epoch number for checkpoint filename.
    metadata : Dict[str, Any], optional
        Additional metadata (not used here, but kept for API consistency).
    model_parallel_rank : int, optional
        Model parallel rank for distributed training. Default is 0.
        
    Returns
    -------
    bool
        True if model was saved as PyTorch model (caller should skip model in save_checkpoint),
        False if model should be saved via save_checkpoint normally.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle DDP-wrapped models
    if isinstance(model, DDP):
        model = model.module
    
    # Check if we need to save as PyTorch model (due to PyTorch submodules)
    save_as_pytorch = _has_pytorch_submodules(model)
    
    if save_as_pytorch:
        # Save model state_dict manually as PyTorch model
        # This bypasses PhysicsNeMo's Module.save() which doesn't support PyTorch submodules
        model_name = model.__class__.__name__
        
        # Create filename matching PhysicsNeMo format: {model_name}.{rank}.{epoch}.pt
        model_filename = f"{model_name}.{model_parallel_rank}.{epoch}.pt"
        model_path = save_dir / model_filename
        
        # Save model state_dict
        protocol = fsspec.utils.get_protocol(str(save_dir))
        fs = fsspec.filesystem(protocol)
        with fs.open(str(model_path), "wb") as fp:
            torch.save(model.state_dict(), fp)
        return True  # Indicate model was saved separately
    
    return False  # Model should be saved via save_checkpoint


class NeuralOperatorTrainer:
    r"""
    A Trainer class for neural operators following PhysicsNeMo patterns.

    This trainer provides a comprehensive training loop for neural operator models
    with support for:
    - Multiple evaluation loaders with different metrics
    - Autoregressive evaluation modes
    - Best model checkpointing based on validation metrics
    - Interval-based checkpointing
    - Mixed precision training
    - Data preprocessing/postprocessing via data processors
    - Regularizers (e.g., L1/L2 regularization)
    - Distributed training via PhysicsNeMo's DistributedManager

    The trainer expects datasets to provide batches as key-value dictionaries,
    e.g., ``{'x': x, 'y': y}``, that are keyed to the arguments expected by
    models and losses.

    Parameters
    ----------
    model : nn.Module
        The neural operator model to train.
    n_epochs : int
        Total number of training epochs.
    device : str or torch.device, optional
        Device to train on. If None, uses DistributedManager.device if available,
        otherwise defaults to 'cpu'. Default is None.
    mixed_precision : bool, optional
        Whether to use mixed precision training with torch.autocast.
        Default is False.
    data_processor : nn.Module, optional
        Data processor module to transform data before/after model forward pass.
        If provided, data is preprocessed with ``data_processor.preprocess()``
        before model forward, and postprocessed with ``data_processor.postprocess()``
        after model forward. Default is None.
    eval_interval : int, optional
        Frequency (in epochs) to evaluate model on validation sets.
        Default is 1 (evaluate every epoch).
    log_output : bool, optional
        If True and wandb_log is True, log output images to wandb.
        Default is False.
    wandb_log : bool, optional
        Whether to log results to wandb. Only logs if wandb is installed
        and a wandb run is active. Default is False.
    verbose : bool, optional
        Whether to print training progress to stdout. Default is False.
    logger : PythonLogger or RankZeroLoggingWrapper, optional
        Optional logger instance. If None, creates a default logger.
        Default is None.
    scaler : GradScaler, optional
        Optional gradient scaler for mixed precision training. If None and
        mixed_precision is True, creates a new scaler. Default is None.

    Examples
    --------
    >>> from neuralop import get_model
    >>> from neuralop.training import AdamW
    >>> from neuralop.losses import LpLoss
    >>> 
    >>> model = get_model(config)
    >>> trainer = NeuralOperatorTrainer(
    ...     model=model,
    ...     n_epochs=100,
    ...     device="cuda",
    ...     wandb_log=True,
    ...     verbose=True
    ... )
    >>> 
    >>> trainer.train(
    ...     train_loader=train_loader,
    ...     test_loaders={"val": val_loader},
    ...     optimizer=optimizer,
    ...     scheduler=scheduler,
    ...     training_loss=LpLoss(d=2),
    ...     eval_losses={"l2": LpLoss(d=2)},
    ...     save_dir="./checkpoints",
    ...     save_best="val_l2"
    ... )
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        n_epochs: int,
        device: Optional[Union[str, torch.device]] = None,
        mixed_precision: bool = False,
        data_processor: Optional[nn.Module] = None,
        eval_interval: int = 1,
        log_output: bool = False,
        wandb_log: bool = False,
        verbose: bool = False,
        logger: Optional[Union[PythonLogger, RankZeroLoggingWrapper]] = None,
        scaler: Optional[GradScaler] = None,
    ) -> None:
        # Model and training configuration
        self.model = model
        self.n_epochs = n_epochs
        self.eval_interval = eval_interval
        self.log_output = log_output
        self.verbose = verbose
        self.data_processor = data_processor

        # Mixed precision configuration
        self.mixed_precision = mixed_precision
        if mixed_precision and scaler is None:
            self.scaler = GradScaler()
        else:
            self.scaler = scaler

        # Device configuration
        if device is None:
            if DistributedManager.is_initialized():
                dist_manager = DistributedManager()
                self.device = dist_manager.device if dist_manager.device else torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Determine autocast device type
        if isinstance(self.device, torch.device):
            self.autocast_device_type = self.device.type
        else:
            self.autocast_device_type = "cuda" if "cuda" in str(self.device) else "cpu"

        # Wandb logging (only if available and run is active)
        self.wandb_log = False
        if _WANDB_AVAILABLE and wandb_log and wandb.run is not None:
            self.wandb_log = True

        # Logging setup
        if logger is None:
            self.logger = PythonLogger(name="neural_operator_trainer")
            if DistributedManager.is_initialized():
                dist_manager = DistributedManager()
                self.logger = RankZeroLoggingWrapper(self.logger, dist_manager)
        else:
            self.logger = logger

        # Training state
        self.start_epoch = 0
        self.epoch = 0
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        self.regularizer: Optional[Any] = None

        # Checkpointing configuration
        self.save_every: Optional[int] = None
        self.save_best: Optional[str] = None
        self.best_metric_value: float = float("inf")

        # Metrics accumulation for wandb
        self.wandb_epoch_metrics: Optional[Dict[str, Any]] = None

    def train(
        self,
        train_loader: DataLoader,
        test_loaders: Dict[str, DataLoader],
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        regularizer: Optional[Any] = None,
        training_loss: Optional[Any] = None,
        eval_losses: Optional[Dict[str, Any]] = None,
        eval_modes: Optional[Dict[str, Literal["single_step", "autoregression"]]] = None,
        save_every: Optional[int] = None,
        save_best: Optional[str] = None,
        save_dir: Union[str, Path] = "./checkpoints",
        resume_from_dir: Optional[Union[str, Path]] = None,
        max_autoregressive_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        r"""
        Train the model on the given dataset.

        This method implements the main training loop with support for:
        - Training on a training dataloader
        - Evaluation on multiple test dataloaders with different metrics
        - Best model checkpointing based on validation metrics
        - Interval-based checkpointing
        - Resuming from checkpoints

        Parameters
        ----------
        train_loader : DataLoader
            Training dataloader providing batches for training.
        test_loaders : Dict[str, DataLoader]
            Dictionary of test/validation dataloaders keyed by name.
            Each loader will be evaluated with all metrics in eval_losses.
        optimizer : Optimizer
            Optimizer to use during training.
        scheduler : _LRScheduler
            Learning rate scheduler to use during training.
        regularizer : Any, optional
            Optional regularizer (e.g., L1/L2) to add to training loss.
            Must have a ``loss`` attribute and ``reset()`` method.
            Default is None.
        training_loss : Any, optional
            Loss function for training. Must be callable as ``loss(pred, **kwargs)``.
            If None, defaults to LpLoss(d=2). Default is None.
        eval_losses : Dict[str, Any], optional
            Dictionary of loss functions for evaluation, keyed by loss name.
            Each loss will be evaluated on all test_loaders, with metrics
            named as ``{loader_name}_{loss_name}``. If None, uses training_loss
            as "l2". Default is None.
        eval_modes : Dict[str, Literal["single_step", "autoregression"]], optional
            Optional mapping from loader name to evaluation mode.
            - "single_step": Predict one input-output pair and evaluate loss.
            - "autoregression": Autoregressively predict output using last step's
              output as input for multiple steps. Requires data processor with
              step-aware preprocess/postprocess methods.
            If not provided, defaults to "single_step" for all loaders.
            Default is None.
        save_every : int, optional
            Interval (in epochs) at which to save checkpoints.
            If None, no interval-based checkpointing is performed.
            Default is None.
        save_best : str, optional
            Metric name (format: ``{loader_name}_{loss_name}``) to monitor
            for best model saving. When this metric improves, a checkpoint is saved.
            Overrides save_every when set. Default is None.
        save_dir : str or Path, optional
            Directory to save training checkpoints. Default is "./checkpoints".
        resume_from_dir : str or Path, optional
            Directory containing checkpoint to resume from. If provided, loads
            model, optimizer, scheduler, and regularizer states and resumes
            training from the saved epoch. Default is None.
        max_autoregressive_steps : int, optional
            Maximum number of autoregressive steps to perform during evaluation.
            Only used when eval_mode is "autoregression". If None, runs full rollout.
            Default is None.

        Returns
        -------
        Dict[str, Any]
            Dictionary of metrics from the last validation epoch, keyed as
            ``{loader_name}_{loss_name}`` for each test loader and loss combination.

        Raises
        ------
        ValueError
            If save_best metric name is not found in available metrics.
        FileNotFoundError
            If resume_from_dir is provided but checkpoint files are not found.
        """
        # Store training components
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.regularizer = regularizer

        # Default training loss
        if training_loss is None:
            from neuralop.losses import LpLoss

            training_loss = LpLoss(d=2)

        # Warn if training loss reduces across batch dimension
        if hasattr(training_loss, "reduction") and training_loss.reduction == "mean":
            warnings.warn(
                f"Training loss has reduction='mean'. Trainer expects losses "
                f"to sum across batch dimension, not average.",
                UserWarning,
                stacklevel=2,
            )

        # Default evaluation losses
        if eval_losses is None:
            eval_losses = {"l2": training_loss}

        # Default evaluation modes
        if eval_modes is None:
            eval_modes = {}

        # Checkpointing configuration
        self.save_every = save_every
        self.save_best = save_best

        # Resume from checkpoint if provided
        if resume_from_dir is not None:
            self._resume_from_checkpoint(resume_from_dir)

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup distributed training if available
        if DistributedManager.is_initialized():
            dist_manager = DistributedManager()
            if dist_manager.distributed:
                self.model = DDP(
                    self.model,
                    device_ids=[dist_manager.local_rank],
                    output_device=dist_manager.local_rank,
                )
                if self.verbose and dist_manager.rank == 0:
                    self.logger.info(f"Using distributed training (rank {dist_manager.rank})")

        # Move data processor to device
        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)

        # Validate save_best metric exists
        if self.save_best is not None:
            available_metrics = []
            for loader_name in test_loaders.keys():
                for loss_name in eval_losses.keys():
                    available_metrics.append(f"{loader_name}_{loss_name}")
            if self.save_best not in available_metrics:
                raise ValueError(
                    f"save_best metric '{self.save_best}' not found in available metrics. "
                    f"Available metrics: {available_metrics}"
                )
            self.best_metric_value = float("inf")
            # Best model saving overrides interval saving
            self.save_every = None

        # Log training setup
        if self.verbose:
            self.logger.info(f"Training on {len(train_loader.dataset)} samples")
            self.logger.info(
                f"Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples "
                f"on loaders {list(test_loaders.keys())}"
            )

        # Initialize epoch_metrics in case loop doesn't execute
        epoch_metrics = {}

        # Main training loop
        # Only show progress bar on rank 0 in distributed training
        is_rank_zero = True
        if DistributedManager.is_initialized():
            dist_manager = DistributedManager()
            is_rank_zero = dist_manager.rank == 0
        epoch_range = range(self.start_epoch, self.n_epochs)
        if is_rank_zero and self.verbose:
            epoch_range = tqdm(epoch_range, desc="Training", unit="epoch")
        
        for epoch in epoch_range:
            self.epoch = epoch

            # Train for one epoch
            train_metrics = self._train_one_epoch(epoch, train_loader, training_loss)
            epoch_metrics = train_metrics.copy()

            # Evaluate if at eval interval
            if epoch % self.eval_interval == 0:
                eval_metrics = self._evaluate_all(
                    epoch=epoch,
                    eval_losses=eval_losses,
                    test_loaders=test_loaders,
                    eval_modes=eval_modes,
                    max_autoregressive_steps=max_autoregressive_steps,
                )
                epoch_metrics.update(eval_metrics)

                # Save best model if metric improved
                if save_best is not None and eval_metrics[save_best] < self.best_metric_value:
                    self.best_metric_value = eval_metrics[save_best]
                    self._save_checkpoint(save_dir, is_best=True)

            # Save checkpoint at interval
            if self.save_every is not None and epoch % self.save_every == 0:
                self._save_checkpoint(save_dir, is_best=False)

        return epoch_metrics

    def _train_one_epoch(
        self, epoch: int, train_loader: DataLoader, training_loss: Any
    ) -> Dict[str, Any]:
        r"""
        Train the model for one epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        train_loader : DataLoader
            Training dataloader.
        training_loss : Any
            Training loss function.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing training metrics:
            - train_err: Average training error per batch
            - avg_loss: Average loss per sample
            - avg_lasso_loss: Average regularizer loss (if regularizer exists)
            - epoch_train_time: Time taken for epoch
        """
        self.model.train()
        if self.data_processor is not None:
            self.data_processor.train()

        avg_loss = 0.0
        avg_lasso_loss = 0.0
        train_err = 0.0
        n_samples = 0

        t1 = default_timer()

        # Only show progress bar on rank 0 in distributed training
        is_rank_zero = True
        if DistributedManager.is_initialized():
            dist_manager = DistributedManager()
            is_rank_zero = dist_manager.rank == 0
        loader_iter = train_loader
        if is_rank_zero and self.verbose:
            loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.n_epochs}", unit="batch", leave=False)

        for idx, sample in enumerate(loader_iter):
            loss = self._train_one_batch(idx, sample, training_loss)
            
            # Track number of samples in batch
            if isinstance(sample.get("y"), torch.Tensor):
                n_samples += sample["y"].shape[0]
            else:
                n_samples += 1
            
            # Backward pass with optional mixed precision
            if self.mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            train_err += loss.item()
            with torch.no_grad():
                avg_loss += loss.item()
                if self.regularizer is not None:
                    avg_lasso_loss += self.regularizer.loss
            
            # Update progress bar with current loss
            if is_rank_zero and self.verbose and hasattr(loader_iter, 'set_postfix'):
                loader_iter.set_postfix({'loss': f'{loss.item():.6f}'})

        # Update learning rate scheduler
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        epoch_train_time = default_timer() - t1

        # Normalize metrics
        train_err /= len(train_loader)
        avg_loss /= n_samples if n_samples > 0 else 1
        if self.regularizer is not None:
            avg_lasso_loss /= n_samples if n_samples > 0 else 1
        else:
            avg_lasso_loss = None

        # Get current learning rate
        lr = None
        for param_group in self.optimizer.param_groups:
            lr = param_group["lr"]
            break

        # Log training metrics
        if self.verbose and epoch % self.eval_interval == 0:
            self._log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )

        return {
            "train_err": train_err,
            "avg_loss": avg_loss,
            "avg_lasso_loss": avg_lasso_loss,
            "epoch_train_time": epoch_train_time,
        }

    def _train_one_batch(self, idx: int, sample: Dict[str, Any], training_loss: Any) -> torch.Tensor:
        r"""
        Train on a single batch.

        Parameters
        ----------
        idx : int
            Batch index.
        sample : Dict[str, Any]
            Batch data dictionary.
        training_loss : Any
            Training loss function.

        Returns
        -------
        torch.Tensor
            Training loss tensor.
        """
        self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer is not None:
            self.regularizer.reset()

        # Preprocess data
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # Move tensors to device if no processor
            sample = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in sample.items()
            }

        # Forward pass with optional mixed precision
        if self.mixed_precision:
            # Use autocast with device_type for newer PyTorch, fallback for older versions
            if hasattr(torch.amp, 'autocast') and self.autocast_device_type == 'cuda':
                # PyTorch 2.0+ with torch.amp.autocast
                with torch.amp.autocast(device_type=self.autocast_device_type):
                    out = self.model(**sample)
            else:
                # Older PyTorch versions or CPU - use torch.cuda.amp.autocast (CPU will be no-op)
                with autocast():
                    out = self.model(**sample)
        else:
            out = self.model(**sample)

        # Log output shape on first batch of first epoch
        if self.epoch == 0 and idx == 0 and self.verbose and isinstance(out, torch.Tensor):
            self.logger.info(f"Model output shape: {out.shape}")

        # Postprocess output
        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        # Compute loss
        if self.mixed_precision:
            if hasattr(torch.amp, 'autocast') and self.autocast_device_type == 'cuda':
                # PyTorch 2.0+ with torch.amp.autocast
                with torch.amp.autocast(device_type=self.autocast_device_type):
                    loss = training_loss(out, **sample)
            else:
                # Older PyTorch versions or CPU
                with autocast():
                    loss = training_loss(out, **sample)
        else:
            loss = training_loss(out, **sample)

        # Add regularizer loss
        if self.regularizer is not None:
            loss = loss + self.regularizer.loss

        return loss

    def _evaluate_all(
        self,
        epoch: int,
        eval_losses: Dict[str, Any],
        test_loaders: Dict[str, DataLoader],
        eval_modes: Dict[str, Literal["single_step", "autoregression"]],
        max_autoregressive_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        r"""
        Evaluate model on all test loaders.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        eval_losses : Dict[str, Any]
            Dictionary of loss functions for evaluation.
        test_loaders : Dict[str, DataLoader]
            Dictionary of test dataloaders.
        eval_modes : Dict[str, Literal["single_step", "autoregression"]]
            Evaluation mode for each loader.
        max_autoregressive_steps : int, optional
            Maximum autoregressive steps.

        Returns
        -------
        Dict[str, Any]
            Dictionary of evaluation metrics keyed as ``{loader_name}_{loss_name}``.
        """
        all_metrics = {}
        for loader_name, loader in test_loaders.items():
            loader_eval_mode = eval_modes.get(loader_name, "single_step")
            loader_metrics = self._evaluate(
                eval_losses=eval_losses,
                data_loader=loader,
                log_prefix=loader_name,
                mode=loader_eval_mode,
                max_steps=max_autoregressive_steps,
            )
            all_metrics.update(loader_metrics)

        if self.verbose:
            self._log_eval(epoch=epoch, eval_metrics=all_metrics)

        return all_metrics

    def _evaluate(
        self,
        eval_losses: Dict[str, Any],
        data_loader: DataLoader,
        log_prefix: str = "",
        mode: Literal["single_step", "autoregression"] = "single_step",
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        r"""
        Evaluate model on a single dataloader.

        Parameters
        ----------
        eval_losses : Dict[str, Any]
            Dictionary of loss functions.
        data_loader : DataLoader
            Dataloader to evaluate on.
        log_prefix : str, optional
            Prefix for metric names. Default is "".
        mode : Literal["single_step", "autoregression"], optional
            Evaluation mode. Default is "single_step".
        max_steps : int, optional
            Maximum steps for autoregressive mode. Default is None.

        Returns
        -------
        Dict[str, Any]
            Dictionary of evaluation metrics.
        """
        self.model.eval()
        if self.data_processor is not None:
            self.data_processor.eval()

        # Initialize error tracking
        errors = {f"{log_prefix}_{loss_name}": 0.0 for loss_name in eval_losses.keys()}

        # Warn if eval losses reduce across batch
        for eval_loss in eval_losses.values():
            if hasattr(eval_loss, "reduction") and eval_loss.reduction == "mean":
                warnings.warn(
                    f"Eval loss has reduction='mean'. Trainer expects losses "
                    f"to sum across batch dimension.",
                    UserWarning,
                    stacklevel=2,
                )

        n_samples = 0
        # Only show progress bar on rank 0 in distributed training
        is_rank_zero = True
        if DistributedManager.is_initialized():
            dist_manager = DistributedManager()
            is_rank_zero = dist_manager.rank == 0
        loader_iter = data_loader
        if is_rank_zero and self.verbose:
            loader_iter = tqdm(data_loader, desc=f"Evaluating ({log_prefix})", unit="batch", leave=False)
        
        with torch.no_grad():
            for idx, sample in enumerate(loader_iter):
                return_output = idx == len(data_loader) - 1

                # Track samples before processing
                if "y" in sample:
                    if isinstance(sample["y"], torch.Tensor):
                        n_samples += sample["y"].shape[0]
                    else:
                        n_samples += 1

                if mode == "single_step":
                    eval_step_losses, outs = self._eval_one_batch(
                        sample, eval_losses, return_output=return_output
                    )
                elif mode == "autoregression":
                    eval_step_losses, outs = self._eval_one_batch_autoreg(
                        sample,
                        eval_losses,
                        return_output=return_output,
                        max_steps=max_steps,
                    )
                else:
                    raise ValueError(f"Unknown evaluation mode: {mode}")

                # Accumulate losses
                for loss_name, val_loss in eval_step_losses.items():
                    errors[f"{log_prefix}_{loss_name}"] += val_loss

        # Normalize by number of samples
        for key in errors.keys():
            errors[key] /= n_samples if n_samples > 0 else 1

        # Log outputs to wandb if requested
        if self.log_output and self.wandb_log and outs is not None:
            errors[f"{log_prefix}_outputs"] = wandb.Image(outs)

        return errors

    def _eval_one_batch(
        self, sample: Dict[str, Any], eval_losses: Dict[str, Any], return_output: bool = False
    ) -> tuple[Dict[str, float], Optional[torch.Tensor]]:
        r"""
        Evaluate on a single batch (single step mode).

        Parameters
        ----------
        sample : Dict[str, Any]
            Batch data dictionary.
        eval_losses : Dict[str, Any]
            Dictionary of loss functions.
        return_output : bool, optional
            Whether to return model outputs. Default is False.

        Returns
        -------
        tuple[Dict[str, float], Optional[torch.Tensor]]
            Dictionary of losses and optional model outputs.
        """
        # Preprocess data
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            sample = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in sample.items()
            }

        # Forward pass
        out = self.model(**sample)

        # Postprocess output
        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        # Compute losses
        eval_step_losses = {}
        for loss_name, loss_fn in eval_losses.items():
            val_loss = loss_fn(out, **sample)
            eval_step_losses[loss_name] = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss

        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None

    def _eval_one_batch_autoreg(
        self,
        sample: Dict[str, Any],
        eval_losses: Dict[str, Any],
        return_output: bool = False,
        max_steps: Optional[int] = None,
    ) -> tuple[Dict[str, float], Optional[torch.Tensor]]:
        r"""
        Evaluate on a single batch (autoregressive mode).

        Parameters
        ----------
        sample : Dict[str, Any]
            Batch data dictionary.
        eval_losses : Dict[str, Any]
            Dictionary of loss functions.
        return_output : bool, optional
            Whether to return model outputs. Default is False.
        max_steps : int, optional
            Maximum number of autoregressive steps. Default is None.

        Returns
        -------
        tuple[Dict[str, float], Optional[torch.Tensor]]
            Dictionary of losses and optional model outputs.
        """
        eval_step_losses = {loss_name: 0.0 for loss_name in eval_losses.keys()}
        t = 0
        max_steps = max_steps if max_steps is not None else float("inf")
        final_out = None

        while sample is not None and t < max_steps:
            # Preprocess data with step index
            if self.data_processor is not None:
                sample = self.data_processor.preprocess(sample, step=t)
            else:
                sample = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in sample.items()
                }

            if sample is None:
                break

            # Forward pass
            out = self.model(**sample)

            # Postprocess output with step index
            if self.data_processor is not None:
                out, sample = self.data_processor.postprocess(out, sample, step=t)

            # Accumulate losses
            for loss_name, loss_fn in eval_losses.items():
                step_loss = loss_fn(out, **sample)
                step_loss_val = step_loss.item() if isinstance(step_loss, torch.Tensor) else step_loss
                eval_step_losses[loss_name] += step_loss_val

            final_out = out
            t += 1

        # Average over steps
        if t > 0:
            for loss_name in eval_step_losses.keys():
                eval_step_losses[loss_name] /= t

        if return_output:
            return eval_step_losses, final_out
        else:
            return eval_step_losses, None

    def _log_training(
        self,
        epoch: int,
        time: float,
        avg_loss: float,
        train_err: float,
        avg_lasso_loss: Optional[float] = None,
        lr: Optional[float] = None,
    ) -> None:
        r"""
        Log training metrics.

        Parameters
        ----------
        epoch : int
            Current epoch.
        time : float
            Training time for epoch.
        avg_loss : float
            Average loss per sample.
        train_err : float
            Training error per batch.
        avg_lasso_loss : float, optional
            Average regularizer loss.
        lr : float, optional
            Current learning rate.
        """
        msg = f"[Epoch {epoch}] time={time:.2f}s, "
        msg += f"avg_loss={avg_loss:.4f}, "
        msg += f"train_err={train_err:.4f}"
        if avg_lasso_loss is not None:
            msg += f", avg_lasso={avg_lasso_loss:.4f}"
        if lr is not None:
            msg += f", lr={lr:.6f}"

        self.logger.info(msg)

        # Log to wandb
        if self.wandb_log:
            values_to_log = {
                "train_err": train_err,
                "time": time,
                "avg_loss": avg_loss,
                "lr": lr,
            }
            if avg_lasso_loss is not None:
                values_to_log["avg_lasso_loss"] = avg_lasso_loss
            wandb.log(data=values_to_log, step=epoch + 1, commit=False)

    def _log_eval(self, epoch: int, eval_metrics: Dict[str, Any]) -> None:
        r"""
        Log evaluation metrics.

        Parameters
        ----------
        epoch : int
            Current epoch.
        eval_metrics : Dict[str, Any]
            Dictionary of evaluation metrics.
        """
        msg = "Eval: "
        values_to_log = {}
        for metric, value in eval_metrics.items():
            if isinstance(value, (float, int)) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                val = float(value.item() if isinstance(value, torch.Tensor) else value)
                msg += f"{metric}={val:.4f}, "
                if self.wandb_log:
                    values_to_log[metric] = val

        msg = msg.rstrip(", ")
        self.logger.info(msg)

        # Log to wandb
        if self.wandb_log:
            wandb.log(data=values_to_log, step=epoch + 1, commit=True)

    def _save_checkpoint(self, save_dir: Union[str, Path], is_best: bool = False) -> None:
        r"""
        Save training checkpoint using PhysicsNeMo checkpoint system.

        This method handles both pure PhysicsNeMo modules and wrapper modules
        that contain PyTorch submodules (like GINOWrapper). For wrappers with
        PyTorch submodules, it saves the model's state_dict as a PyTorch model
        instead of using PhysicsNeMo's Module.save().

        Parameters
        ----------
        save_dir : str or Path
            Directory to save checkpoint.
        is_best : bool, optional
            Whether this is the best model checkpoint. Default is False.
        """
        # Only save on rank 0 in distributed training
        if DistributedManager.is_initialized():
            dist_manager = DistributedManager()
            if dist_manager.rank != 0:
                return

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Prepare metadata
        metadata = {
            "epoch": self.epoch,
            "is_best": is_best,
            "best_metric_value": self.best_metric_value if is_best else None,
        }

        # Determine model parallel rank (for distributed training compatibility)
        model_parallel_rank = 0
        if DistributedManager.is_initialized():
            dist_manager = DistributedManager()
            if "model_parallel" in dist_manager.group_names:
                model_parallel_rank = dist_manager.group_rank("model_parallel")
        
        # Use actual epoch number for checkpoint filename
        # Best model is tracked via metadata, not filename
        save_epoch = self.epoch
        
        # Handle model saving separately if it contains PyTorch submodules
        model_to_save = self.model
        if isinstance(self.model, DDP):
            model_to_save = self.model.module
        
        save_as_pytorch = _has_pytorch_submodules(model_to_save)
        
        if save_as_pytorch:
            # Save model state_dict manually as PyTorch model
            model_name = model_to_save.__class__.__name__
            model_filename = f"{model_name}.{model_parallel_rank}.{save_epoch}.pt"
            model_path = save_dir / model_filename
            
            protocol = fsspec.utils.get_protocol(str(save_dir))
            fs = fsspec.filesystem(protocol)
            with fs.open(str(model_path), "wb") as fp:
                torch.save(model_to_save.state_dict(), fp)
            
            if self.verbose:
                self.logger.info(f"Saved model state_dict as PyTorch model: {model_path}")
        
        # Save training state (optimizer, scheduler, etc.) using PhysicsNeMo
        # Include model only if it's not a PyTorch wrapper
        save_checkpoint(
            path=str(save_dir),
            models=None if save_as_pytorch else model_to_save,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=save_epoch,
            metadata=metadata,
        )

        if self.verbose:
            checkpoint_type = "best model" if is_best else "checkpoint"
            self.logger.info(f"Saved {checkpoint_type} to {save_dir} (epoch {self.epoch})")

    def _resume_from_checkpoint(self, resume_dir: Union[str, Path]) -> None:
        r"""
        Resume training from checkpoint.

        Parameters
        ----------
        resume_dir : str or Path
            Directory containing checkpoint.

        Raises
        ------
        FileNotFoundError
            If checkpoint directory or files are not found.
        """
        resume_dir = Path(resume_dir)
        if not resume_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {resume_dir}")

        # Load checkpoint using PhysicsNeMo system
        # Load latest checkpoint (epoch=None loads most recent)
        metadata_dict = {}
        resume_epoch = load_checkpoint(
            path=str(resume_dir),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=None,  # Load latest
            metadata_dict=metadata_dict,
            device=self.device,
        )

        # Update training state
        if resume_epoch is not None and resume_epoch > self.start_epoch:
            self.start_epoch = resume_epoch + 1  # Resume from next epoch
            if self.verbose:
                self.logger.info(f"Resuming training from epoch {resume_epoch}")

        # Extract best metric value if available
        if "best_metric_value" in metadata_dict:
            self.best_metric_value = metadata_dict["best_metric_value"]

