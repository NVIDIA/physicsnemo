# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Checkpointing, optimizer factory, and training-state setup.

Consolidates three concerns that all sit at the boundary between "model" and
"training loop":

* Optimizer construction (Adam + optional Muon hybrid).
* Best / best-QoI checkpoint management (save, prune, top-model symlink).
* Training-state assembly (`create_training_components`) and resume/pretrain
  loading (`resume_or_pretrain`).

DDP / seeding / batch-size logging helpers live in ``trainer.py``.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.checkpoint import load_checkpoint
from physicsnemo.utils.logging.launch import LaunchLogger

# Sibling import: scheduler factory lives in losses.py.
from losses import create_scheduler


# =========================================================================
# Optimizers
# =========================================================================

def create_optimizer(
    model: nn.Module,
    optimizer_type: Literal["adam", "muon"] = "adam",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    muon_momentum_beta: float = 0.95,
    muon_lr: Optional[float] = None,
    logger=None,
) -> torch.optim.Optimizer:
    """Create optimizer based on configuration.

    For ``optimizer_type='muon'`` returns a hybrid optimizer that uses Muon for
    2D weight matrices and Adam for 1D parameters (biases, layer norms, etc.).
    Muon only supports 2D weight matrices, hence the split.

    Args:
        model: The model to optimize.
        optimizer_type: ``'adam'`` or ``'muon'``.
        learning_rate: Learning rate for Adam (and for 1D params when using Muon).
        weight_decay: Weight decay coefficient.
        muon_momentum_beta: Momentum beta for the Muon optimizer.
        muon_lr: Learning rate for Muon (defaults to ``learning_rate`` if ``None``).
        logger: Optional logger for info messages.

    Returns:
        Configured optimizer.
    """
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        if logger:
            logger.info(
                f"Using Adam optimizer with lr={learning_rate}, weight_decay={weight_decay}"
            )
        return optimizer

    elif optimizer_type == "muon":
        return _create_muon_optimizer(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            muon_momentum_beta=muon_momentum_beta,
            muon_lr=muon_lr,
            logger=logger,
        )

    raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def _create_muon_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    muon_momentum_beta: float,
    muon_lr: Optional[float],
    logger=None,
) -> torch.optim.Optimizer:
    """Create a hybrid Muon + Adam optimizer.

    Muon is used for 2D weight matrices, Adam is used for 1D parameters
    (biases, layer norms, etc.) and embeddings.

    Uses ``torch.optim.Muon`` (PyTorch's built-in Newton-Schulz orthogonalized
    optimizer for 2-D hidden-layer weights). Available since PyTorch 2.6.
    """
    try:
        from torch.optim import Muon
    except ImportError as e:
        raise ImportError(
            "torch.optim.Muon was not found. Upgrade to PyTorch >= 2.6 "
            "(verified working with 2.9)."
        ) from e

    muon_lr = muon_lr if muon_lr is not None else learning_rate

    # Separate parameters by dimensionality.
    muon_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim == 2:
            muon_params.append(param)
        else:
            adam_params.append(param)

    if logger:
        logger.info(
            f"Muon optimizer: {len(muon_params)} 2D params, {len(adam_params)} other params"
        )

    optimizers: List[torch.optim.Optimizer] = []

    if muon_params:
        # torch.optim.Muon uses ``momentum=`` (the original emerging-optimizers
        # implementation called the same hyperparameter ``momentum_beta``); we
        # keep ``muon_momentum_beta`` as the config key for continuity and map
        # it through here.
        muon_optimizer = Muon(
            muon_params,
            lr=muon_lr,
            momentum=muon_momentum_beta,
            weight_decay=weight_decay,
        )
        optimizers.append(muon_optimizer)
        if logger:
            logger.info(f"Muon: lr={muon_lr}, momentum={muon_momentum_beta}")

    if adam_params:
        adam_optimizer = torch.optim.Adam(
            adam_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        optimizers.append(adam_optimizer)
        if logger:
            logger.info(f"Adam (for 1D params): lr={learning_rate}")

    if len(optimizers) == 1:
        return optimizers[0]

    return CombinedOptimizer(optimizers)


class CombinedOptimizer(torch.optim.Optimizer):
    """Wrapper to combine multiple optimizers into a single interface.

    This allows using Muon for 2D params and Adam for 1D params while
    maintaining a standard optimizer interface for the training loop.
    Inherits from ``torch.optim.Optimizer`` for compatibility with LR schedulers.
    """

    def __init__(self, optimizers: List[torch.optim.Optimizer]):
        self.optimizers = optimizers

        # Collect all params for the base Optimizer init.
        all_params = []
        for opt in optimizers:
            for group in opt.param_groups:
                all_params.extend(group["params"])

        # Initialize base Optimizer with dummy defaults; the actual param_groups
        # come from the wrapped optimizers below.
        super().__init__(all_params, defaults={})

        # Replace param_groups with the ones from the wrapped optimizers.
        self.param_groups = []
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients for all wrapped optimizers."""
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> None:
        """Step every wrapped optimizer."""
        for opt in self.optimizers:
            opt.step(closure=closure)

    def state_dict(self) -> dict:
        """Return combined state dict."""
        return {
            "optimizers": [opt.state_dict() for opt in self.optimizers],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load combined state dict."""
        for opt, opt_state in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(opt_state)


# =========================================================================
# Save / load checkpoints
# =========================================================================

# Maximum number of best checkpoints to keep (by validation loss).
MAX_BEST_CHECKPOINTS = 3

# Folder name for the single best model (lowest validation loss).
TOP_MODEL_DIR = "top_model"

# Folder name for the best model by QoI relative error.
BEST_QOI_MODEL_DIR = "best_qoi_model"


def save_best_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    val_loss: float,
    best_val_losses: List[Tuple[float, int]],
    save_checkpoint_fn,
    logger: logging.Logger = None,
    **checkpoint_kwargs,
) -> List[Tuple[float, int]]:
    """Save checkpoint if it's in top-N best models, and clean up old checkpoints.

    Also maintains a ``top_model`` folder containing the single best model by
    validation loss.

    Args:
        checkpoint_dir: Directory to save checkpoints.
        epoch: Current epoch number.
        val_loss: Current validation loss.
        best_val_losses: List of ``(loss, epoch)`` tuples for best models
            (will be modified in-place and returned).
        save_checkpoint_fn: Function to call to save the checkpoint
            (e.g. PhysicsNeMo's ``save_checkpoint``).
        logger: Optional logger for log messages.
        **checkpoint_kwargs: Additional arguments forwarded to ``save_checkpoint_fn``.

    Returns:
        Updated list of ``(loss, epoch)`` tuples.
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Handle legacy format: convert List[float] to List[Tuple[float, int]].
    if best_val_losses and isinstance(best_val_losses[0], (int, float)):
        # Legacy format detected, reset to empty (can't recover epoch info).
        best_val_losses = []

    # Check whether this is a top-N model.
    current_losses = [loss for loss, _ in best_val_losses]
    is_top_n = len(best_val_losses) < MAX_BEST_CHECKPOINTS or val_loss < max(
        current_losses
    )

    if not is_top_n:
        return best_val_losses

    # Save new best model to epoch-specific directory.
    best_model_dir = checkpoint_dir / f"best_model_epoch_{epoch}"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    save_checkpoint_fn(path=str(best_model_dir), epoch=epoch, **checkpoint_kwargs)

    # Update best losses list with the new (loss, epoch) tuple.
    best_val_losses.append((val_loss, epoch))
    best_val_losses.sort(key=lambda x: x[0])  # Sort by loss.

    # Cleanup if we have more than MAX_BEST_CHECKPOINTS.
    while len(best_val_losses) > MAX_BEST_CHECKPOINTS:
        worst_loss, worst_epoch = best_val_losses.pop()
        cleanup_checkpoint_by_epoch(checkpoint_dir, worst_epoch, logger)

    # Update top_model folder if this is the new best.
    _update_top_model(
        checkpoint_dir,
        val_loss,
        epoch,
        best_val_losses,
        save_checkpoint_fn,
        logger,
        **checkpoint_kwargs,
    )

    if logger:
        logger.info(
            f"  Saved top-{MAX_BEST_CHECKPOINTS} model! Val loss: {val_loss:.6f}"
        )
        loss_strs = [f"{loss:.6f}" for loss, _ in best_val_losses[:3]]
        logger.info(f"  Top 3 losses: {loss_strs}")

    return best_val_losses


def _update_top_model(
    checkpoint_dir: Path,
    val_loss: float,
    epoch: int,
    best_val_losses: List[Tuple[float, int]],
    save_checkpoint_fn,
    logger: logging.Logger = None,
    **checkpoint_kwargs,
) -> None:
    """Update the ``top_model`` folder if this epoch has the lowest val loss.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        val_loss: Current validation loss.
        epoch: Current epoch number.
        best_val_losses: List of ``(loss, epoch)`` tuples sorted by loss (best first).
        save_checkpoint_fn: Function to save the checkpoint.
        logger: Optional logger.
        **checkpoint_kwargs: Additional arguments forwarded to ``save_checkpoint_fn``.
    """
    if not best_val_losses:
        return

    # Check whether the current epoch is the best (first in sorted list).
    best_loss, best_epoch = best_val_losses[0]
    if epoch != best_epoch:
        return  # Not the best — nothing to update.

    checkpoint_dir = Path(checkpoint_dir)
    top_model_path = checkpoint_dir / TOP_MODEL_DIR

    # Remove any existing top_model folder.
    if top_model_path.exists():
        shutil.rmtree(top_model_path)

    # Save the best model directly to the top_model folder.
    top_model_path.mkdir(parents=True, exist_ok=True)
    save_checkpoint_fn(path=str(top_model_path), epoch=epoch, **checkpoint_kwargs)

    if logger:
        logger.info(f"  Updated top_model (epoch {epoch}, val_loss: {val_loss:.6f})")


def save_best_qoi_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    qoi_error: float,
    best_qoi_error: float,
    save_checkpoint_fn,
    logger: logging.Logger = None,
    **checkpoint_kwargs,
) -> float:
    """Save checkpoint if QoI loss improved.

    Maintains a single ``best_qoi_model`` folder with the model that achieved
    the lowest QoI loss during training.

    Args:
        checkpoint_dir: Directory to save checkpoints.
        epoch: Current epoch number.
        qoi_error: Current QoI loss value (lower is better).
        best_qoi_error: Previous best QoI loss value.
        save_checkpoint_fn: Function to save the checkpoint.
        logger: Optional logger.
        **checkpoint_kwargs: Additional arguments forwarded to ``save_checkpoint_fn``.

    Returns:
        Updated best QoI loss value.
    """
    if qoi_error >= best_qoi_error:
        return best_qoi_error

    checkpoint_dir = Path(checkpoint_dir)
    qoi_model_path = checkpoint_dir / BEST_QOI_MODEL_DIR

    if qoi_model_path.exists():
        shutil.rmtree(qoi_model_path)

    qoi_model_path.mkdir(parents=True, exist_ok=True)
    save_checkpoint_fn(path=str(qoi_model_path), epoch=epoch, **checkpoint_kwargs)

    if logger:
        logger.info(
            f"  New best QoI model! epoch={epoch}, "
            f"qoi_loss={qoi_error:.6e} (prev best: {best_qoi_error:.6e})"
        )

    return qoi_error


def cleanup_checkpoint_by_epoch(
    checkpoint_dir: Path, epoch: int, logger: logging.Logger = None
) -> None:
    """Remove the checkpoint directory for a specific epoch.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        epoch: Epoch number of the checkpoint to remove.
        logger: Optional logger for log messages.
    """
    checkpoint_dir = Path(checkpoint_dir)
    target_dir = checkpoint_dir / f"best_model_epoch_{epoch}"

    if target_dir.exists():
        shutil.rmtree(target_dir)
        if logger:
            logger.info(f"  Removed old checkpoint: {target_dir.name}")


# =========================================================================
# Training-state setup
# =========================================================================

def create_training_components(
    cfg: DictConfig,
    model: nn.Module,
    dist: DistributedManager,
    logger: Any,
    tensorboard: bool = True,
) -> Tuple[
    torch.optim.Optimizer,
    Any,
    GradScaler,
    Optional[SummaryWriter],
    str,
    List,
]:
    """Create optimizer, scheduler, scaler, TensorBoard writer, and checkpoint dir.

    Also initializes ``LaunchLogger`` and returns an empty ``best_val_losses`` list
    that the training loop can hand to :func:`save_best_checkpoint`.

    Args:
        cfg: Hydra configuration.
        model: The model (possibly DDP-wrapped).
        dist: ``DistributedManager`` instance.
        logger: Logger for rank-0 messages.
        tensorboard: Whether to create a TensorBoard writer (default ``True``).

    Returns:
        ``(optimizer, scheduler, scaler, writer, checkpoint_dir, best_val_losses)``.
    """
    optimizer_cfg = cfg.train.get("optimizer", {})
    optimizer_type = optimizer_cfg.get("type", "adam")
    weight_decay = optimizer_cfg.get(
        "weight_decay", cfg.train.get("weight_decay", 0.0)
    )
    muon_momentum_beta = optimizer_cfg.get("muon_momentum_beta", 0.95)
    muon_lr = optimizer_cfg.get("muon_lr", None)

    optimizer = create_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        learning_rate=cfg.train.learning_rate,
        weight_decay=weight_decay,
        muon_momentum_beta=muon_momentum_beta,
        muon_lr=muon_lr,
        logger=logger if dist.rank == 0 else None,
    )

    scheduler = create_scheduler(cfg, optimizer, logger if dist.rank == 0 else None)

    # GradScaler is only needed for FP16 AMP. For BF16 (and for amp=false),
    # we disable scaling to avoid overhead and potential instability.
    amp_enabled = bool(cfg.train.get("amp", False))
    amp_dtype = str(cfg.train.get("amp_dtype", "fp16")).lower()
    scaler_enabled = amp_enabled and amp_dtype in ("fp16", "float16")
    scaler = GradScaler(enabled=scaler_enabled)

    LaunchLogger.initialize(use_wandb=False, use_mlflow=False)

    writer = None
    if tensorboard and dist.rank == 0:
        writer = SummaryWriter(os.path.join(cfg.output, "tensorboard"))

    checkpoint_dir = os.path.join(cfg.output, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_losses: List = []

    return optimizer, scheduler, scaler, writer, checkpoint_dir, best_val_losses


def resume_or_pretrain(
    cfg: DictConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    dist: DistributedManager,
    logger: Any,
) -> Tuple[int, List, float]:
    """Handle checkpoint resume or pretrain weight loading.

    * If ``cfg.train.resume_checkpoint`` exists, loads full training state
      (model, optimizer, scheduler, scaler) and returns the next epoch.
    * Else if ``cfg.train.pretrain_checkpoint`` exists, loads model weights only
      (optimizer/scheduler stay fresh) for fine-tuning from epoch 0.
    * Otherwise returns epoch 0 with an empty ``best_val_losses`` list and
      ``best_qoi_loss = +inf``.

    Args:
        cfg: Hydra config.
        model: Model (possibly DDP-wrapped).
        optimizer: Optimizer.
        scheduler: LR scheduler.
        scaler: ``GradScaler``.
        dist: ``DistributedManager``.
        logger: Logger.

    Returns:
        ``(start_epoch, best_val_losses, best_qoi_loss)``.
    """
    start_epoch = 0
    best_val_losses: List = []
    best_qoi_loss = float("inf")

    resume_checkpoint = cfg.train.get("resume_checkpoint", None)
    pretrain_checkpoint = cfg.train.get("pretrain_checkpoint", None)

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        if dist.rank == 0:
            logger.info(f"\nResuming from checkpoint: {resume_checkpoint}")

        metadata: Dict[str, Any] = {}
        start_epoch = load_checkpoint(
            path=resume_checkpoint,
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metadata_dict=metadata,
            device=dist.device,
        )

        if "best_val_losses" in metadata:
            best_val_losses = metadata["best_val_losses"]
        if "best_qoi_loss" in metadata:
            best_qoi_loss = metadata["best_qoi_loss"]

        if dist.rank == 0:
            logger.info(f"  Resumed from epoch {start_epoch}")
            if best_val_losses:
                if isinstance(best_val_losses[0], (int, float)):
                    loss_strs = [f"{v:.6f}" for v in best_val_losses[:3]]
                else:
                    loss_strs = [f"{loss:.6f}" for loss, _ in best_val_losses[:3]]
                logger.info(f"  Top val losses: {loss_strs}")
            if best_qoi_loss < float("inf"):
                logger.info(f"  Best QoI loss: {best_qoi_loss:.6e}")

        start_epoch += 1

    elif pretrain_checkpoint and os.path.exists(pretrain_checkpoint):
        if dist.rank == 0:
            logger.info(
                f"\nLoading pretrained weights for fine-tuning: {pretrain_checkpoint}"
            )

        load_checkpoint(
            path=pretrain_checkpoint,
            models=model,
            device=dist.device,
        )

        if dist.rank == 0:
            logger.info("  Pretrained weights loaded successfully")
            logger.info("  Optimizer and scheduler reset for fine-tuning")
            logger.info("  Starting from epoch 0")

    return start_epoch, best_val_losses, best_qoi_loss
