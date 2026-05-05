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
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.distributed import DistributedManager
from physicsnemo.optim import CombinedOptimizer
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


# =========================================================================
# Save / load checkpoints
# =========================================================================

# Maximum number of best checkpoints to keep (by validation loss).
MAX_BEST_CHECKPOINTS = 3

# Folder name for the single best model (lowest validation loss).
TOP_MODEL_DIR = "top_model"

# Folder name for the best model by QoI relative error.
BEST_QOI_MODEL_DIR = "best_qoi_model"

# Folder name for the latest full training-state checkpoint.
LATEST_CHECKPOINT_DIR = "latest_checkpoint"


def _capture_rng_state() -> Dict[str, Any]:
    """Snapshot torch / numpy RNGs for exact-reproducible resume."""
    state: Dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _require_checkpoint_files(path: Path, *, require_training_state: bool) -> None:
    """Raise FileNotFoundError if expected checkpoint files are missing."""
    pt_files = list(path.glob("*.pt"))
    mdlus_files = list(path.glob("*.mdlus"))
    missing = []
    if require_training_state and not pt_files:
        missing.append("*.pt (training state)")
    if not mdlus_files:
        missing.append("*.mdlus (model state)")
    if missing:
        present = [f.name for f in path.iterdir() if f.is_file()]
        raise FileNotFoundError(
            f"Checkpoint at {path} is incomplete; missing {missing}. "
            f"Files present: {present}"
        )


def _checkpoint_kwargs_with_metadata(
    checkpoint_kwargs: Dict[str, Any], **metadata_updates
) -> Dict[str, Any]:
    """Return checkpoint kwargs with selected metadata keys refreshed.

    Always refreshes ``rng_state`` so every save automatically captures the
    current torch / numpy RNG state for reproducible resume.
    """
    updated_kwargs = dict(checkpoint_kwargs)
    metadata = dict(updated_kwargs.get("metadata") or {})
    metadata.update(metadata_updates)
    metadata["rng_state"] = _capture_rng_state()
    updated_kwargs["metadata"] = metadata
    return updated_kwargs


def save_latest_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    save_checkpoint_fn,
    logger: logging.Logger = None,
    **checkpoint_kwargs,
) -> Path:
    """Replace ``latest_checkpoint`` with the most recent training state.

    This checkpoint is meant for robust resume, not model selection. It is
    overwritten at the caller's cadence, usually every epoch.

    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories.
        epoch: Current epoch number.
        save_checkpoint_fn: Function to save the checkpoint.
        logger: Optional logger.
        **checkpoint_kwargs: Additional arguments forwarded to ``save_checkpoint_fn``.

    Returns:
        Path to the refreshed ``latest_checkpoint`` directory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    latest_path = checkpoint_dir / LATEST_CHECKPOINT_DIR
    tmp_path = checkpoint_dir / f".{LATEST_CHECKPOINT_DIR}.tmp"

    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    # Route through metadata helper so every latest checkpoint captures the
    # current RNG state for reproducible resume.
    checkpoint_kwargs = _checkpoint_kwargs_with_metadata(checkpoint_kwargs)

    save_checkpoint_fn(path=str(tmp_path), epoch=epoch, **checkpoint_kwargs)

    if latest_path.exists():
        shutil.rmtree(latest_path)
    tmp_path.rename(latest_path)

    if logger:
        logger.info(f"  Updated latest checkpoint (epoch {epoch})")

    return latest_path


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
            (returned with any updates applied).
        save_checkpoint_fn: Function to call to save the checkpoint
            (e.g. PhysicsNeMo's ``save_checkpoint``).
        logger: Optional logger for log messages.
        **checkpoint_kwargs: Additional arguments forwarded to ``save_checkpoint_fn``.

    Returns:
        Updated list of ``(loss, epoch)`` tuples.
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not math.isfinite(float(val_loss)):
        if logger:
            logger.warning(
                "  Skipping best-checkpoint save for epoch %s: non-finite val_loss=%s",
                epoch,
                val_loss,
            )
        return list(best_val_losses)

    best_val_losses = list(best_val_losses)

    # Check whether this is a top-N model.
    current_losses = [loss for loss, _ in best_val_losses]
    is_top_n = len(best_val_losses) < MAX_BEST_CHECKPOINTS or val_loss < max(
        current_losses
    )

    if not is_top_n:
        return best_val_losses

    updated_best_val_losses = best_val_losses + [(val_loss, epoch)]
    updated_best_val_losses.sort(key=lambda x: x[0])  # Sort by loss.

    pruned_epochs = []
    while len(updated_best_val_losses) > MAX_BEST_CHECKPOINTS:
        _worst_loss, worst_epoch = updated_best_val_losses.pop()
        pruned_epochs.append(worst_epoch)

    checkpoint_kwargs = _checkpoint_kwargs_with_metadata(
        checkpoint_kwargs,
        best_val_losses=updated_best_val_losses,
    )

    # Save new best model to epoch-specific directory.
    best_model_dir = checkpoint_dir / f"best_model_epoch_{epoch}"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    save_checkpoint_fn(path=str(best_model_dir), epoch=epoch, **checkpoint_kwargs)

    for worst_epoch in pruned_epochs:
        cleanup_checkpoint_by_epoch(checkpoint_dir, worst_epoch, logger)

    # Update top_model folder if this is the new best.
    _update_top_model(
        checkpoint_dir,
        val_loss,
        epoch,
        updated_best_val_losses,
        save_checkpoint_fn,
        logger,
        **checkpoint_kwargs,
    )

    if logger:
        logger.info(
            f"  Saved top-{MAX_BEST_CHECKPOINTS} model! Val loss: {val_loss:.6f}"
        )
        loss_strs = [f"{loss:.6f}" for loss, _ in updated_best_val_losses[:3]]
        logger.info(f"  Top 3 losses: {loss_strs}")

    return updated_best_val_losses


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
    if not math.isfinite(float(qoi_error)):
        if logger:
            logger.warning(
                "  Skipping best-QoI checkpoint save for epoch %s: non-finite "
                "qoi_error=%s",
                epoch,
                qoi_error,
            )
        return best_qoi_error

    if qoi_error >= best_qoi_error:
        return best_qoi_error

    checkpoint_dir = Path(checkpoint_dir)
    qoi_model_path = checkpoint_dir / BEST_QOI_MODEL_DIR

    if qoi_model_path.exists():
        shutil.rmtree(qoi_model_path)

    checkpoint_kwargs = _checkpoint_kwargs_with_metadata(
        checkpoint_kwargs,
        best_qoi_loss=qoi_error,
    )

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
    weight_decay = optimizer_cfg.get("weight_decay", cfg.train.get("weight_decay", 0.0))
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

    if resume_checkpoint:
        resume_path = Path(str(resume_checkpoint))
        if not resume_path.exists():
            raise FileNotFoundError(
                f"train.resume_checkpoint does not exist: {resume_path}"
            )
        if not resume_path.is_dir():
            raise NotADirectoryError(
                "train.resume_checkpoint must be a checkpoint directory, "
                f"not a file: {resume_path}"
            )
        _require_checkpoint_files(resume_path, require_training_state=True)

        if dist.rank == 0:
            logger.info(f"\nResuming from checkpoint: {resume_path}")

        metadata: Dict[str, Any] = {}
        start_epoch = load_checkpoint(
            path=str(resume_path),
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

        rng_state = metadata.get("rng_state")
        if rng_state:
            if "torch" in rng_state:
                torch.set_rng_state(rng_state["torch"].cpu().to(torch.uint8))
            if "numpy" in rng_state:
                np.random.set_state(rng_state["numpy"])
            if "torch_cuda_all" in rng_state and torch.cuda.is_available():
                cuda_states = [
                    s.cpu().to(torch.uint8) for s in rng_state["torch_cuda_all"]
                ]
                torch.cuda.set_rng_state_all(cuda_states)
            if dist.rank == 0:
                logger.info("  RNG state restored from checkpoint")

        if dist.rank == 0:
            logger.info(f"  Resumed from epoch {start_epoch}")
            if best_val_losses:
                loss_strs = [f"{loss:.6f}" for loss, _ in best_val_losses[:3]]
                logger.info(f"  Top val losses: {loss_strs}")
            if best_qoi_loss < float("inf"):
                logger.info(f"  Best QoI loss: {best_qoi_loss:.6e}")

        start_epoch += 1

    elif pretrain_checkpoint:
        pretrain_path = Path(str(pretrain_checkpoint))
        if not pretrain_path.exists():
            raise FileNotFoundError(
                f"train.pretrain_checkpoint does not exist: {pretrain_path}"
            )
        if not pretrain_path.is_dir():
            raise NotADirectoryError(
                "train.pretrain_checkpoint must be a checkpoint directory, "
                f"not a file: {pretrain_path}"
            )
        _require_checkpoint_files(pretrain_path, require_training_state=False)

        if dist.rank == 0:
            logger.info(
                f"\nLoading pretrained weights for fine-tuning: {pretrain_path}"
            )

        load_checkpoint(
            path=str(pretrain_path),
            models=model,
            device=dist.device,
        )

        if dist.rank == 0:
            logger.info("  Pretrained weights loaded successfully")
            logger.info("  Optimizer and scheduler reset for fine-tuning")
            logger.info("  Starting from epoch 0")

    return start_epoch, best_val_losses, best_qoi_loss
