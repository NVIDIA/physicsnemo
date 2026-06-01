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

import math
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from physicsnemo.distributed import DistributedManager
from physicsnemo.optim import CombinedOptimizer
from physicsnemo.utils.checkpoint import load_checkpoint


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adam",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    muon_momentum_beta: float = 0.95,
    logger=None,
) -> torch.optim.Optimizer:
    """Create optimizer based on configuration.

    For ``optimizer_type='muon'`` returns a hybrid: Muon for 2D weight
    matrices, AdamW for 1D params (biases, layer norms, embeddings). Muon
    only supports 2D weight matrices, hence the split. The shared
    ``learning_rate`` drives both halves because Muon is constructed with
    ``adjust_lr_fn='match_rms_adamw'``.
    """
    if optimizer_type not in ("adam", "muon"):
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    if optimizer_type == "muon":
        return _create_muon_optimizer(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            muon_momentum_beta=muon_momentum_beta,
            logger=logger,
        )

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


def _create_muon_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    muon_momentum_beta: float,
    logger=None,
) -> torch.optim.Optimizer:
    """Build a Muon + AdamW combined optimizer (Muon for 2D, AdamW for the rest).

    Requires PyTorch >= 2.9 for ``torch.optim.Muon`` with the
    ``adjust_lr_fn`` argument.
    """
    if not hasattr(torch.optim, "Muon"):
        raise ImportError(
            "Muon optimizer requires PyTorch >= 2.9. "
            "Install a newer PyTorch or use optimizer.type=adam."
        )
    base_model = model.module if hasattr(model, "module") else model
    muon_params = [p for p in base_model.parameters() if p.ndim == 2]
    other_params = [p for p in base_model.parameters() if p.ndim != 2]

    if logger:
        logger.info(
            f"Muon optimizer: {len(muon_params)} 2D params, "
            f"{len(other_params)} other params, lr={learning_rate}"
        )

    muon = (
        torch.optim.Muon(
            muon_params,
            lr=learning_rate,
            momentum=muon_momentum_beta,
            weight_decay=weight_decay,
            adjust_lr_fn="match_rms_adamw",
        )
        if muon_params
        else None
    )
    adamw = (
        torch.optim.AdamW(
            other_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        if other_params
        else None
    )

    if muon and adamw:
        return CombinedOptimizer([muon, adamw])
    return muon or adamw


def save_best_checkpoint(
    checkpoint_dir: Path,
    val_loss: float,
    best_val_loss: float,
    save_checkpoint_fn,
    logger=None,
    **checkpoint_kwargs,
) -> float:
    """Save a single ``best_model/`` checkpoint when ``val_loss`` improves.

    Returns the (possibly unchanged) ``best_val_loss``. Skips with a warning
    when ``val_loss`` is not finite, and is a no-op when the current loss does
    not beat the previous best.
    """
    if not math.isfinite(float(val_loss)):
        if logger:
            logger.warning(
                "  Skipping best-checkpoint save: non-finite val_loss=%s", val_loss
            )
        return best_val_loss

    if val_loss >= best_val_loss:
        return best_val_loss

    checkpoint_dir = Path(checkpoint_dir)
    best_model_dir = checkpoint_dir / "best_model"
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    epoch = checkpoint_kwargs.pop("epoch")
    save_checkpoint_fn(path=str(best_model_dir), epoch=epoch, **checkpoint_kwargs)

    if logger:
        logger.info(
            f"  New best model! val_loss={val_loss:.6f} (prev best: {best_val_loss:.6f})"
        )
    return float(val_loss)


def resume_if_available(
    cfg: DictConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    dist: DistributedManager,
    logger: Any,
) -> Tuple[int, float]:
    """Resume full training state or load pretrain weights, if configured.

    Returns ``(start_epoch, best_val_loss)``. PhysicsNeMo's ``load_checkpoint``
    raises on missing files, so no pre-validation is performed here.
    """
    resume_checkpoint = cfg.train.get("resume_checkpoint", None)
    pretrain_checkpoint = cfg.train.get("pretrain_checkpoint", None)

    if resume_checkpoint:
        resume_path = Path(str(resume_checkpoint))
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
        best_val_loss = float(metadata.get("best_val_loss", float("inf")))
        if dist.rank == 0:
            logger.info(f"  Resumed from epoch {start_epoch}")
            if best_val_loss < float("inf"):
                logger.info(f"  Best val_loss: {best_val_loss:.6f}")
        return start_epoch + 1, best_val_loss

    if pretrain_checkpoint:
        pretrain_path = Path(str(pretrain_checkpoint))
        if dist.rank == 0:
            logger.info(
                f"\nLoading pretrained weights for fine-tuning: {pretrain_path}"
            )
        load_checkpoint(path=str(pretrain_path), models=model, device=dist.device)
        if dist.rank == 0:
            logger.info("  Pretrained weights loaded; starting from epoch 0")
        return 0, float("inf")

    return 0, float("inf")


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    cfg: DictConfig,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build the Transolver model from cfg.model and load weights from checkpoint_path.

    The caller supplies the full Hydra cfg (so the model definition is
    fully controlled by the inference-time config, not pulled from a
    saved training-time snapshot). The checkpoint_path must contain
    matching ``checkpoint.0.*.pt`` + ``Transolver.0.*.mdlus`` shards.

    Returns (model in eval mode, metadata dict from the checkpoint).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    # Build model from cfg.model. Strip RTE-specific keys consumed elsewhere.
    cfg_model = OmegaConf.to_container(cfg.model, resolve=True)
    for k in ("num_spatial_points", "include_q_in_embedding"):
        cfg_model.pop(k, None)
    model = hydra.utils.instantiate(cfg_model).to(device)

    metadata: Dict[str, Any] = {}
    epoch = load_checkpoint(
        path=str(checkpoint_path),
        models=model,
        metadata_dict=metadata,
        device=device,
    )
    metadata.setdefault("epoch", epoch)

    model.eval()
    print(
        f"Loaded model from {checkpoint_path} "
        f"(epoch={metadata.get('epoch', '?')}, "
        f"params={sum(p.numel() for p in model.parameters()):,})"
    )
    return model, metadata
