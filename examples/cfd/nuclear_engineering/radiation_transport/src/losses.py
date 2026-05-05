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

"""Losses: MSE / region-weighted / physics-informed / QoI helpers + LR schedulers.

This module consolidates every "loss" concept the trainer touches:

* Learning-rate schedulers (warmup + cosine, plus a constant fallback).
* Regression losses on the (possibly padded) flux tensor: ``loss_fn``,
  ``masked_mse_loss``, ``region_weighted_loss_fn``.
* Physics-informed loss for the radiation-transport surrogate: per-case
  QoI loss (lattice / hohlraum) computed in physical flux space using the
  differentiable PyTorch QoI evaluators, plus a ``compute_physics_loss``
  dispatcher that ``train.py`` drives.
* Differentiable PyTorch QoI evaluators
  (``evaluate_lattice_qoi_torch`` / ``evaluate_hohlraum_qoi_torch``)
  used by the physics loss above. The numpy-side evaluators used by
  ``inference.py`` live in ``inference.py``.
* ``parse_loss_config`` — pulls the ``train.physics_loss`` and
  ``train.region_weights`` blocks out of the Hydra config into a flat dict
  the trainer consumes.

Module is pure compute: it has no dependency on sibling source files.
``denormalize_flux_from_stats`` delegates to ``transforms.denormalize_flux`` so the physics loss
can convert log-normalized model outputs back to physical flux space
without importing from ``transforms.py``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import torch
from omegaconf import DictConfig


# =========================================================================
# Schedulers
# =========================================================================


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.

    During warmup (epochs 0 to warmup_epochs-1):
        lr = min_lr + (max_lr - min_lr) * (epoch / warmup_epochs)

    After warmup (epochs warmup_epochs to total_epochs):
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
        where progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Return the per-group learning rates for the current ``last_epoch``."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


def create_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer, logger=None):
    """Create the LR scheduler.

    Supports:
    - cosine: Warmup + cosine annealing (recommended; default)
    - constant: No decay (useful for overfit tests)
    """
    scheduler_type = cfg.train.get("scheduler_type", "cosine")
    warmup_epochs = cfg.train.get("warmup_epochs", 5)
    min_lr = cfg.train.get("min_learning_rate", 1e-6)

    if logger:
        logger.info("\nLearning rate schedule:")
        logger.info(f"  Type: {scheduler_type}")
        logger.info(f"  Peak LR: {cfg.train.learning_rate}")
        logger.info(f"  Min LR: {min_lr}")
        if warmup_epochs > 0:
            logger.info(f"  Warmup epochs: {warmup_epochs}")

    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=cfg.train.epochs,
        )
    if scheduler_type == "cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=cfg.train.epochs,
            min_lr=min_lr,
        )
    raise ValueError(
        f"Unknown scheduler_type {scheduler_type!r}; expected 'cosine' or 'constant'."
    )


# =========================================================================
# Regression losses
# =========================================================================


def masked_mse_loss(
    output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Calculate MSE loss with optional masking for padded values.

    Used by Transolver training.

    Args:
        output: Predicted values (B, N, 1)
        target: Ground truth values (B, N, 1)
        mask: Boolean mask (B, N) - True for real points, False for padding

    Returns:
        Scalar loss value
    """
    squared_error = (output - target) ** 2

    if mask is not None:
        # expand mask to match output shape
        mask_expanded = mask.unsqueeze(-1)
        # only compute loss on non-padded points
        masked_error = squared_error * mask_expanded
        loss = masked_error.sum() / mask_expanded.sum()
    else:
        loss = torch.mean(squared_error)

    return loss


def region_weighted_loss_fn(
    output: torch.Tensor,
    target: torch.Tensor,
    material_labels: torch.Tensor,
    case_type: str,
    loss_type: str = "mse",
    padded_value: float = -10,
    void_weight: float = 3.0,
    material_weight: float = 1.0,
) -> torch.Tensor:
    """
    Calculate region-weighted loss based on material labels.

    Uses discrete material labels from the material mappers to identify regions:
    - Void (fill gas): radiation streams through, creates fine features
    - Material (walls, capsule, absorbers): solid regions

    Weights void regions more heavily than material regions to improve
    fine feature learning where radiation streaming occurs.

    Material label definitions:
        Hohlraum:
            0: Black (walls) - material
            1: Red (walls) - material
            2: Green (walls) - material
            3: Blue (capsule) - material
            4: White (fill gas) - void

        Lattice:
            0: Blue (absorber) - material
            1: Red (scattering source) - material
            2: White (background) - void

    Args:
        output: Predicted values (B, N, 1)
        target: Ground truth values (B, N, 1)
        material_labels: Material label per cell (B, N) or (B, N, 1), integer values
        case_type: "hohlraum" or "lattice"
        loss_type: Type of loss - "mse" or "weighted_rel_l2" (relative-L2 form, not true RMSE)
        padded_value: Value used for padding (will be masked out)
        void_weight: Weight for void (fill gas) regions
        material_weight: Weight for solid material regions

    Returns:
        Scalar loss value
    """
    # Create padding mask
    mask = (abs(target - padded_value) > 1e-3).float()

    # Squeeze material_labels if needed
    labels = (
        material_labels.squeeze(-1) if material_labels.dim() == 3 else material_labels
    )

    # Identify void regions based on case type
    # Void label: 4 for hohlraum (white/fill gas), 2 for lattice (white/background)
    if case_type.lower() == "hohlraum":
        is_void = (labels == 4).float()  # (B, N)
    elif case_type.lower() == "lattice":
        is_void = (labels == 2).float()  # (B, N)
    else:
        raise ValueError(
            f"Unknown case_type: {case_type}. Must be 'hohlraum' or 'lattice'"
        )

    # Compute per-point weights
    weights = is_void * void_weight + (1.0 - is_void) * material_weight  # (B, N)
    weights = weights.unsqueeze(-1)  # (B, N, 1) to match output shape

    # Apply padding mask to weights
    weights = weights * mask

    # Compute weighted squared error
    squared_error = (output - target) ** 2.0
    weighted_error = weights * squared_error

    if loss_type == "weighted_rel_l2":
        weighted_target_sq = weights * target**2.0
        loss = torch.sqrt(weighted_error.sum() / (weighted_target_sq.sum() + 1e-8))
    else:  # mse
        loss = weighted_error.sum() / (weights.sum() + 1e-8)

    return loss


def parse_loss_config(
    cfg: DictConfig,
    dist: Any,
    logger: Any,
) -> dict:
    """
    Parse the common loss configuration options shared across all models:
    physics loss, region-weighted loss.

    Args:
        cfg: Hydra config
        dist: DistributedManager (only ``dist.rank`` is read)
        logger: Logger

    Returns:
        Dict with keys: use_physics_loss, physics_loss_weight, physics_loss_mse_weight,
        qoi_region, use_region_weighted_loss, region_weight_cfg
    """
    use_physics_loss = cfg.train.get("use_physics_loss", False)
    if use_physics_loss:
        physics_loss_weight = cfg.train.physics_loss.weight
        physics_loss_mse_weight = cfg.train.physics_loss.mse_weight
        qoi_region = cfg.train.physics_loss.get("qoi_region", "center")
    else:
        physics_loss_weight = 0.0
        physics_loss_mse_weight = 1.0
        qoi_region = "center"

    use_region_weighted_loss = cfg.train.get("use_region_weighted_loss", False)
    region_weight_cfg = {
        "void_weight": cfg.train.get("region_weights", {}).get("void_weight", 3.0),
        "material_weight": cfg.train.get("region_weights", {}).get(
            "material_weight", 1.0
        ),
    }

    if dist.rank == 0:
        if use_physics_loss:
            logger.info("\nPhysics loss configuration:")
            logger.info(f"  Weight: {physics_loss_weight}")
            logger.info(f"  MSE weight: {physics_loss_mse_weight}")
            logger.info(f"  QoI region: {qoi_region}")
        if use_region_weighted_loss:
            logger.info("Region-weighted loss: enabled")
            logger.info(f"  Void weight: {region_weight_cfg['void_weight']}")
            logger.info(f"  Material weight: {region_weight_cfg['material_weight']}")

    return {
        "use_physics_loss": use_physics_loss,
        "physics_loss_weight": physics_loss_weight,
        "physics_loss_mse_weight": physics_loss_mse_weight,
        "qoi_region": qoi_region,
        "use_region_weighted_loss": use_region_weighted_loss,
        "region_weight_cfg": region_weight_cfg,
    }


# =========================================================================
# Physics loss
# =========================================================================
#
# Physics-based loss functions using QoI computations.
#
# Compares physics-based quantities of interest (QoIs) computed from model
# predictions against ground truth.
#
# - QoIs (absorption integrals) are defined in **physical flux space**.
#   If your model is trained on a normalized/log-transformed flux, you must
#   denormalize before computing QoIs, otherwise the "physics loss" is
#   optimizing a different (non-physical) quantity.
# - QoI computations can be numerically sensitive under AMP/FP16 due to large
#   reductions (sums over many cells). Prefer running these in FP32 (disable
#   autocast) in the training loop.


def denormalize_flux_from_stats(
    normalized_flux: torch.Tensor,
    flux_normalization_stats: Mapping[str, Any],
) -> torch.Tensor:
    """Invert the ``RTEFluxLogClip + Normalize`` chain for QoI evaluation.

    Thin wrapper over ``transforms.denormalize_flux`` that enforces the
    presence of the stats dict (callers in physics-loss code reach here
    only after validating shapes, so the stats must be available).
    """
    if flux_normalization_stats is None:
        raise ValueError("flux_normalization_stats is required for QoI denormalization")
    # Sibling import is safe: transforms.py is foundational and does not
    # import from losses.py (verified via static cross-module check).
    from transforms import denormalize_flux

    return denormalize_flux(normalized_flux, flux_normalization_stats)


def _relative_squared_error_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    epsilon: float = 1e-10,
) -> tuple[torch.Tensor, dict]:
    """Compute mean relative squared error between pred and target vectors."""
    relative_error = (pred - target) / (torch.abs(target) + epsilon)
    squared_error = relative_error**2

    is_valid = (
        torch.isfinite(squared_error) & torch.isfinite(pred) & torch.isfinite(target)
    )

    if not is_valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True), {}

    loss = squared_error[is_valid].mean()
    return loss, {}


def compute_lattice_qoi_loss(
    predicted_flux: torch.Tensor,
    target_flux: torch.Tensor,
    cell_centers: torch.Tensor,
    cell_areas: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    sim_time: torch.Tensor,
    flux_normalization_stats: Optional[Mapping[str, Any]] = None,
    epsilon: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute QoI-based physics loss for lattice problems.

    Computes instantaneous absorption QoI from predicted flux and target flux,
    then compares them using relative squared error to provide scale-invariant gradients.

    QoIs are computed in physical flux space. If `flux_normalization_stats` is provided,
    both `predicted_flux` and `target_flux` are denormalized before QoI evaluation.

    Uses PyTorch operations throughout to maintain gradient flow for backpropagation.

    Args:
        predicted_flux: Model predictions (normalized), shape (B, N, 1) or (B, N)
        target_flux: Ground truth flux (normalized), shape (B, N, 1) or (B, N)
        cell_centers: Cell center coordinates (unnormalized), shape (B, N, 3)
        cell_areas: Cell areas, shape (B, N)
        sigma_t: Total cross-section, shape (B, N)
        sigma_s: Scattering cross-section, shape (B, N)
        sim_time: Simulation time for each sample, shape (B,)

    Returns:
        Scalar tensor with relative squared error loss between predicted and target QoI
    """
    # ensure correct shape: (B, N)
    if predicted_flux.ndim == 3:
        predicted_flux = predicted_flux.squeeze(-1)  # (B, N, 1) -> (B, N)
    if target_flux.ndim == 3:
        target_flux = target_flux.squeeze(-1)  # (B, N, 1) -> (B, N)

    # compute QoIs in physical flux space
    if flux_normalization_stats is not None:
        predicted_flux = denormalize_flux_from_stats(
            predicted_flux, flux_normalization_stats
        )
        target_flux = denormalize_flux_from_stats(target_flux, flux_normalization_stats)

    # reshape for QoI computation: (B, 1, N) for single timestep
    predicted_flux_qoi = predicted_flux.unsqueeze(1)  # (B, 1, N)
    target_flux_qoi = target_flux.unsqueeze(1)  # (B, 1, N)

    # prepare sim_times: (B, 1)
    sim_times = sim_time.unsqueeze(-1) if sim_time.ndim == 1 else sim_time  # (B, 1)

    # compute QoI for predicted flux using differentiable PyTorch implementation
    qoi_pred = evaluate_lattice_qoi_torch(
        cell_centers=cell_centers,  # (B, N, 3)
        cell_areas=cell_areas,  # (B, N)
        sigma_t=sigma_t,  # (B, N)
        sigma_s=sigma_s,  # (B, N)
        scalar_flux=predicted_flux_qoi,  # (B, 1, N)
        sim_times=sim_times,  # (B, 1)
    )

    # compute QoI for target flux (no gradients needed for target)
    with torch.no_grad():
        qoi_target = evaluate_lattice_qoi_torch(
            cell_centers=cell_centers,
            cell_areas=cell_areas,
            sigma_t=sigma_t,
            sigma_s=sigma_s,
            scalar_flux=target_flux_qoi,
            sim_times=sim_times,
        )

    # extract instantaneous absorption: (B, 1) -> (B,)
    qoi_pred_value = qoi_pred["cur_absorption"][:, 0]
    qoi_target_value = qoi_target["cur_absorption"][:, 0]

    loss, loss_details = _relative_squared_error_loss(
        qoi_pred_value, qoi_target_value, predicted_flux.device, epsilon
    )
    loss_details["loss_qoi_absorption"] = loss.item()

    return loss, loss_details


def compute_hohlraum_qoi_loss(
    predicted_flux: torch.Tensor,
    target_flux: torch.Tensor,
    cell_centers: torch.Tensor,
    cell_areas: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    sim_time: torch.Tensor,
    geometry_params: dict,
    qoi_region: str = "all",
    flux_normalization_stats: Optional[Mapping[str, Any]] = None,
    epsilon: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute QoI-based physics loss for hohlraum problems.

    The loss used for backpropagation is determined by ``qoi_region``:
      - "all" (default): mean of the four region losses (center, vertical,
        horizontal, total) — every region contributes to the gradient
      - "center" | "vertical" | "horizontal": loss on that single region
      - "total":  loss on the integrated absorption over the whole domain
        (computed from the sum of the three spatial regions)

    All four region losses are *always* recorded in the details dict so they
    are visible in the training log regardless of which region drives the
    gradient.

    Returns:
        Tuple of (loss_tensor, details_dict).
    """
    # ensure correct shape: (B, N)
    if predicted_flux.ndim == 3:
        predicted_flux = predicted_flux.squeeze(-1)
    if target_flux.ndim == 3:
        target_flux = target_flux.squeeze(-1)

    if flux_normalization_stats is not None:
        predicted_flux = denormalize_flux_from_stats(
            predicted_flux, flux_normalization_stats
        )
        target_flux = denormalize_flux_from_stats(target_flux, flux_normalization_stats)

    predicted_flux_qoi = predicted_flux.unsqueeze(1)
    target_flux_qoi = target_flux.unsqueeze(1)
    sim_times = sim_time.unsqueeze(-1) if sim_time.ndim == 1 else sim_time

    qoi_pred = evaluate_hohlraum_qoi_torch(
        cell_centers=cell_centers,
        cell_areas=cell_areas,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        scalar_flux=predicted_flux_qoi,
        sim_times=sim_times,
        geometry_params=geometry_params,
    )

    with torch.no_grad():
        qoi_target = evaluate_hohlraum_qoi_torch(
            cell_centers=cell_centers,
            cell_areas=cell_areas,
            sigma_t=sigma_t,
            sigma_s=sigma_s,
            scalar_flux=target_flux_qoi,
            sim_times=sim_times,
            geometry_params=geometry_params,
        )

    region_keys = (
        "cur_absorption_center",
        "cur_absorption_vertical",
        "cur_absorption_horizontal",
    )
    details: dict[str, float] = {}

    total_pred = torch.zeros(predicted_flux.shape[0], device=predicted_flux.device)
    total_target = torch.zeros_like(total_pred)
    region_losses: dict[str, torch.Tensor] = {}

    for key in region_keys:
        p = qoi_pred[key][:, 0]
        t = qoi_target[key][:, 0]
        region_loss, _ = _relative_squared_error_loss(
            p, t, predicted_flux.device, epsilon
        )
        short = key.replace("cur_absorption_", "")
        region_losses[short] = region_loss
        total_pred = total_pred + p
        total_target = total_target + t

    total_loss, _ = _relative_squared_error_loss(
        total_pred, total_target, predicted_flux.device, epsilon
    )
    region_losses["total"] = total_loss

    # Always log every region's loss so all four are visible in train.log
    # regardless of which region(s) drive the gradient.
    for region_name, region_loss in region_losses.items():
        details[f"loss_qoi_{region_name}"] = region_loss.item()

    if qoi_region == "all":
        # Mean of the four region losses — every region contributes to the gradient.
        loss = torch.stack(list(region_losses.values())).mean()
        details["loss_qoi_all"] = loss.item()
    elif qoi_region in region_losses:
        loss = region_losses[qoi_region]
    else:
        raise ValueError(
            f"Unknown qoi_region: {qoi_region}. "
            f"Must be 'all' or one of: {list(region_losses.keys())}"
        )

    return loss, details


def extract_geometry_params(filename) -> dict:
    """Extract hohlraum geometry parameters from zarr filename."""
    # handle list (batched) or single string filename
    if isinstance(filename, (list, tuple)):
        filename = filename[0] if len(filename) > 0 else ""

    if not isinstance(filename, str):
        filename = str(filename)

    # remove .zarr extension if present
    filename = filename.replace(".zarr", "")

    parts = filename.split("_")

    geometry_params = {}
    for part in parts:
        if part.startswith("ulr"):
            geometry_params["ulr"] = float(part[3:])
        elif part.startswith("llr"):
            geometry_params["llr"] = float(part[3:])
        elif part.startswith("urr"):
            geometry_params["urr"] = float(part[3:])
        elif part.startswith("lrr"):
            geometry_params["lrr"] = float(part[3:])
        elif part.startswith("hlr"):
            geometry_params["hlr"] = float(part[3:])
        elif part.startswith("hrr"):
            geometry_params["hrr"] = float(part[3:])
        elif part.startswith("cx"):
            geometry_params["cx"] = float(part[2:])
        elif part.startswith("cy"):
            geometry_params["cy"] = float(part[2:])

    return geometry_params


def compute_physics_loss(
    case_type: str,
    predicted_flux: torch.Tensor,
    target_flux: torch.Tensor,
    cell_centers: torch.Tensor,
    cell_areas: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    sim_time: torch.Tensor,
    metadata: list = None,
    flux_normalization_stats: dict | None = None,
    qoi_epsilon: float = 1e-10,
    qoi_region: str = "all",
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute physics loss based on case type.

    For hohlraum, ``qoi_region`` selects which region(s) drive the gradient:
    ``"all"`` (default) averages the four region losses (center, vertical,
    horizontal, total) so every region contributes; ``"center"`` /
    ``"vertical"`` / ``"horizontal"`` / ``"total"`` use that single region.
    Either way, all four region losses are recorded in the details dict.

    Returns:
        Tuple of (loss_tensor, details_dict) with per-region QoI losses for logging.
    """
    if case_type == "lattice":
        return compute_lattice_qoi_loss(
            predicted_flux=predicted_flux,
            target_flux=target_flux,
            cell_centers=cell_centers,
            cell_areas=cell_areas,
            sigma_t=sigma_t,
            sigma_s=sigma_s,
            sim_time=sim_time,
            flux_normalization_stats=flux_normalization_stats,
            epsilon=qoi_epsilon,
        )
    elif case_type == "hohlraum":
        if metadata is None:
            raise ValueError("hohlraum physics loss requires metadata with filename")

        if isinstance(metadata, dict):
            filename = metadata.get("filename", "")
        elif isinstance(metadata, list) and len(metadata) > 0:
            filename = metadata[0].get("filename", "")
        else:
            raise ValueError(
                f"hohlraum physics loss requires metadata with filename, got: {type(metadata)}"
            )

        geometry_params = extract_geometry_params(filename)

        if not geometry_params:
            raise ValueError(
                f"could not extract geometry parameters from filename: {filename}"
            )

        return compute_hohlraum_qoi_loss(
            predicted_flux=predicted_flux,
            target_flux=target_flux,
            cell_centers=cell_centers,
            cell_areas=cell_areas,
            sigma_t=sigma_t,
            sigma_s=sigma_s,
            sim_time=sim_time,
            geometry_params=geometry_params,
            qoi_region=qoi_region,
            flux_normalization_stats=flux_normalization_stats,
            epsilon=qoi_epsilon,
        )
    else:
        raise ValueError(
            f"unknown case type: {case_type}. must be 'lattice' or 'hohlraum'"
        )


# =========================================================================
# QoI helpers (torch)
# =========================================================================
#
# Differentiable PyTorch QoI evaluators used by the physics loss above.
# These match KiT-RT SNSolverHPC::IterPostprocessing() exactly.
# The numpy-side equivalents (``evaluate_lattice_qoi``,
# ``evaluate_hohlraum_qoi``) live in ``inference.py``.


def evaluate_lattice_qoi_torch(
    cell_centers: torch.Tensor,
    cell_areas: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    scalar_flux: torch.Tensor,
    sim_times: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute lattice absorption QoI using PyTorch (differentiable).

    Matches KiT-RT SNSolverHPC::IterPostprocessing() exactly. Steady-state
    surrogate ⇒ T=1; ``sim_times`` is accepted for callsite uniformity but
    not used. ``batch_size=1`` is enforced repo-wide; if a leading batch dim
    is present we recurse on the squeezed slot and re-add it on the way out.

    Args:
        cell_centers: (N, 3) or (1, N, 3)
        cell_areas: (N,) or (1, N)
        sigma_t: (N,) or (1, N)
        sigma_s: (N,) or (1, N)
        scalar_flux: (T, N) or (1, T, N) — only T=1 is exercised
        sim_times: (T,) or (1, T) — unused, kept for callsite uniformity

    Returns:
        ``{"cur_absorption": (T,) or (1, T)}``
    """
    if cell_centers.ndim == 3:
        if cell_centers.shape[0] != 1:
            raise NotImplementedError(
                "evaluate_lattice_qoi_torch only supports batch_size=1; "
                f"got batch={cell_centers.shape[0]}."
            )
        result = evaluate_lattice_qoi_torch(
            cell_centers[0],
            cell_areas[0],
            sigma_t[0],
            sigma_s[0],
            scalar_flux[0],
            sim_times[0] if sim_times.ndim == 2 else sim_times,
        )
        return {k: v.unsqueeze(0) for k, v in result.items()}

    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    sigma_a = sigma_t - sigma_s

    xy_corrector = -3.5
    lbounds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]) + xy_corrector
    ubounds = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]) + xy_corrector

    in_absorption = torch.zeros_like(x, dtype=torch.bool)
    for k in range(5):
        for l in range(5):  # noqa: E741
            if (l + k) % 2 == 1:
                continue
            if (k == 2 and l == 2) or (k == 2 and l == 4):
                continue
            in_square = (
                (x >= lbounds[k])
                & (x <= ubounds[k])
                & (y >= lbounds[l])
                & (y <= ubounds[l])
            )
            in_absorption = in_absorption | in_square

    if scalar_flux.ndim != 2:
        raise ValueError(f"Expected scalar_flux shape (T, N), got {scalar_flux.shape}")

    absorption_density = scalar_flux * sigma_a.unsqueeze(0) * cell_areas.unsqueeze(0)
    cur_absorption = torch.sum(
        absorption_density * in_absorption.unsqueeze(0).float(), dim=1
    )

    return {"cur_absorption": cur_absorption}


def evaluate_hohlraum_qoi_torch(
    cell_centers: torch.Tensor,
    cell_areas: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    scalar_flux: torch.Tensor,
    sim_times: torch.Tensor,
    geometry_params: dict[str, float],
) -> dict[str, torch.Tensor]:
    """Compute hohlraum absorption QoI using PyTorch (differentiable).

    Matches KiT-RT SNSolverHPC hohlraum geometry exactly (including known KiT-RT
    quirk of using pos_red_left_bottom for both vertical wall sides). Steady-state
    surrogate ⇒ T=1; ``sim_times`` is accepted for callsite uniformity but
    not used. ``batch_size=1`` is enforced repo-wide; if a leading batch dim
    is present we recurse on the squeezed slot and re-add it on the way out.

    Args:
        cell_centers: (N, 3) or (1, N, 3)
        cell_areas: (N,) or (1, N)
        sigma_t: (N,) or (1, N)
        sigma_s: (N,) or (1, N)
        scalar_flux: (T, N) or (1, T, N) — only T=1 is exercised
        sim_times: (T,) or (1, T) — unused, kept for callsite uniformity
        geometry_params: dict with cx, cy, hlr, hrr, llr, ulr, lrr, urr

    Returns:
        Dict with ``cur_absorption_{center,vertical,horizontal}``.
    """
    if cell_centers.ndim == 3:
        if cell_centers.shape[0] != 1:
            raise NotImplementedError(
                "evaluate_hohlraum_qoi_torch only supports batch_size=1; "
                f"got batch={cell_centers.shape[0]}."
            )
        result = evaluate_hohlraum_qoi_torch(
            cell_centers[0],
            cell_areas[0],
            sigma_t[0],
            sigma_s[0],
            scalar_flux[0],
            sim_times[0] if sim_times.ndim == 2 else sim_times,
            geometry_params,
        )
        return {k: v.unsqueeze(0) for k, v in result.items()}

    x = cell_centers[:, 0]
    y = cell_centers[:, 1]

    cx = geometry_params["cx"]
    cy = geometry_params["cy"]
    pos_red_left_border = geometry_params["hlr"]
    pos_red_right_border = geometry_params["hrr"]
    pos_red_left_bottom = geometry_params["llr"]
    pos_red_left_top = geometry_params["ulr"]
    pos_red_right_top = geometry_params["urr"]

    sigma_a = sigma_t - sigma_s

    in_center = (x > -0.2 + cx) & (x < 0.2 + cx) & (y > -0.4 + cy) & (y < 0.4 + cy)
    # IMPORTANT: matches KiT-RT's behavior of using pos_red_left_bottom for both sides
    in_vertical = (
        (x < pos_red_left_border) & (y > pos_red_left_bottom) & (y < pos_red_left_top)
    ) | (
        (x > pos_red_right_border) & (y > pos_red_left_bottom) & (y < pos_red_right_top)
    )
    in_horizontal = (y > 0.6) | (y < -0.6)

    if scalar_flux.ndim != 2:
        raise ValueError(f"Expected scalar_flux shape (T, N), got {scalar_flux.shape}")

    absorption_density = scalar_flux * sigma_a.unsqueeze(0) * cell_areas.unsqueeze(0)

    return {
        "cur_absorption_center": torch.sum(
            absorption_density * in_center.unsqueeze(0).float(), dim=1
        ),
        "cur_absorption_vertical": torch.sum(
            absorption_density * in_vertical.unsqueeze(0).float(), dim=1
        ),
        "cur_absorption_horizontal": torch.sum(
            absorption_density * in_horizontal.unsqueeze(0).float(), dim=1
        ),
    }


__all__ = [
    # Schedulers
    "WarmupCosineScheduler",
    "create_scheduler",
    # Regression losses
    "masked_mse_loss",
    "region_weighted_loss_fn",
    "parse_loss_config",
    # Physics loss
    "compute_physics_loss",
    "compute_lattice_qoi_loss",
    "compute_hohlraum_qoi_loss",
    "denormalize_flux_from_stats",
    "extract_geometry_params",
    # QoI helpers (torch)
    "evaluate_lattice_qoi_torch",
    "evaluate_hohlraum_qoi_torch",
]
