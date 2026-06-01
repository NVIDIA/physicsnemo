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

from typing import Any, Mapping, Optional

import torch
from omegaconf import DictConfig

from qoi import (
    evaluate_hohlraum_qoi_torch,
    evaluate_lattice_qoi_torch,
    extract_geometry_params,
)
from transforms import denormalize_flux

__all__ = [
    # Schedulers
    "create_scheduler",
    # Regression losses
    "region_weighted_loss_fn",
    "parse_loss_config",
    "physics_loss_weight_for_epoch",
    # Physics loss
    "compute_physics_loss",
    "compute_lattice_qoi_loss",
    "compute_hohlraum_qoi_loss",
]


def create_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer, logger=None):
    """Build the LR scheduler: linear warmup chained into cosine annealing."""
    warmup_epochs = cfg.train.get("warmup_epochs", 5)
    peak_lr = cfg.train.learning_rate
    min_lr = cfg.train.get("min_learning_rate", 1e-6)
    total_epochs = cfg.train.epochs

    if logger:
        logger.info("\nLearning rate schedule (warmup + cosine):")
        logger.info(f"  Peak LR: {peak_lr}")
        logger.info(f"  Min LR: {min_lr}")
        logger.info(f"  Warmup epochs: {warmup_epochs}")

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=min_lr / peak_lr,
        end_factor=1.0,
        total_iters=max(warmup_epochs, 1),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_epochs, 1),
        eta_min=min_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


_VOID_LABELS = {"hohlraum": 4, "lattice": 2}


def region_weighted_loss_fn(
    output: torch.Tensor,
    target: torch.Tensor,
    material_labels: torch.Tensor,
    case_type: str,
    void_weight: float = 3.0,
    material_weight: float = 1.0,
) -> torch.Tensor:
    """Weighted MSE that penalizes void cells more than material cells.

    Material-label definitions (set by ``MaterialPropertyExtractor``):

    * Hohlraum: ``0`` black wall, ``1`` red wall, ``2`` green wall,
      ``3`` blue capsule (all material); ``4`` white fill gas (void).
    * Lattice: ``0`` blue absorber, ``1`` red scattering source
      (material); ``2`` white background (void).

    Void cells are where radiation streams through and the surrogate has
    to capture fine flux features, so we weight their squared error more
    heavily.

    Args:
        output, target: Predicted vs ground-truth flux, shape ``(B, N, 1)``.
        material_labels: Per-cell label, shape ``(B, N)`` or ``(B, N, 1)``.
        case_type: ``"hohlraum"`` or ``"lattice"``.
        void_weight, material_weight: Per-region weights.

    Returns:
        Scalar weighted-MSE loss.
    """
    if case_type not in _VOID_LABELS:
        raise ValueError(
            f"Unknown case_type: {case_type}. Must be 'hohlraum' or 'lattice'."
        )

    labels = (
        material_labels.squeeze(-1) if material_labels.dim() == 3 else material_labels
    )
    is_void = labels == _VOID_LABELS[case_type]  # (B, N) bool
    weights = (
        torch.where(is_void, float(void_weight), float(material_weight))
        .to(dtype=torch.float32)
        .unsqueeze(-1)
    )  # (B, N, 1)

    squared_error = (output - target) ** 2
    return (weights * squared_error).sum() / (weights.sum() + 1e-8)


def parse_loss_config(
    cfg: DictConfig,
    dist: Any,
    logger: Any,
) -> dict:
    """
    Parse the common loss configuration options shared across all models:
    physics loss (including warmup schedule), region-weighted loss.

    The returned ``physics_loss_weight`` is the **base** weight; per-epoch
    warmup ramping is applied by :func:`physics_loss_weight_for_epoch` inside
    the trainer loop.

    Args:
        cfg: Hydra config
        dist: DistributedManager (only ``dist.rank`` is read)
        logger: Logger

    Returns:
        Dict with keys: ``use_physics_loss``, ``physics_loss_weight``,
        ``physics_loss_mse_weight``, ``physics_loss_warmup_epochs``,
        ``physics_loss_warmup_start_fraction``,
        ``use_region_weighted_loss``, ``region_weight_cfg``.
    """
    use_physics_loss = cfg.train.get("use_physics_loss", False)
    if use_physics_loss:
        physics_loss_weight = cfg.train.physics_loss.weight
        physics_loss_mse_weight = cfg.train.physics_loss.mse_weight
        physics_loss_warmup_epochs = cfg.train.physics_loss.get("warmup_epochs", 0)
        physics_loss_warmup_start_fraction = cfg.train.physics_loss.get(
            "warmup_start_fraction", 0.0
        )
    else:
        physics_loss_weight = 0.0
        physics_loss_mse_weight = 1.0
        physics_loss_warmup_epochs = 0
        physics_loss_warmup_start_fraction = 0.0

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
            if physics_loss_warmup_epochs > 0:
                logger.info(f"  Warmup epochs: {physics_loss_warmup_epochs}")
                logger.info(
                    f"  Warmup start fraction: {physics_loss_warmup_start_fraction}"
                )
        if use_region_weighted_loss:
            logger.info("Region-weighted loss: enabled")
            logger.info(f"  Void weight: {region_weight_cfg['void_weight']}")
            logger.info(f"  Material weight: {region_weight_cfg['material_weight']}")

    return {
        "use_physics_loss": use_physics_loss,
        "physics_loss_weight": physics_loss_weight,
        "physics_loss_mse_weight": physics_loss_mse_weight,
        "physics_loss_warmup_epochs": physics_loss_warmup_epochs,
        "physics_loss_warmup_start_fraction": physics_loss_warmup_start_fraction,
        "use_region_weighted_loss": use_region_weighted_loss,
        "region_weight_cfg": region_weight_cfg,
    }


def physics_loss_weight_for_epoch(loss_cfg: dict, epoch: int) -> float:
    """Linear ramp of the physics-loss weight over the warmup window.

    Ramps from ``warmup_start_fraction * base`` at epoch 0 to ``base`` at
    ``warmup_epochs``, then stays at ``base``. With no warmup configured
    (``warmup_epochs <= 0``), returns ``base`` unchanged.
    """
    base = loss_cfg.get("physics_loss_weight", 0.0)
    warmup_epochs = loss_cfg.get("physics_loss_warmup_epochs", 0)
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return base
    start_frac = loss_cfg.get("physics_loss_warmup_start_fraction", 0.0)
    progress = epoch / max(1, warmup_epochs)
    return (start_frac + (1.0 - start_frac) * progress) * base


def _relative_squared_error_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """Mean of ``((pred - target) / |target|)^2`` over finite cells.

    Returns ``0.0`` (no graph) when every cell is non-finite — degenerate but
    keeps the trainer alive instead of propagating NaN.
    """
    squared = ((pred - target) / (torch.abs(target) + epsilon)) ** 2
    is_valid = torch.isfinite(squared) & torch.isfinite(pred) & torch.isfinite(target)
    if not is_valid.any():
        return torch.zeros((), device=pred.device)
    return squared[is_valid].mean()


def _prepare_for_qoi(
    pred: torch.Tensor,
    target: torch.Tensor,
    sim_time: torch.Tensor,
    stats: Optional[Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Squeeze ``(B, N, 1) -> (B, N)``, denormalize, then ``(B, 1, N)`` for QoI."""
    if pred.ndim == 3:
        pred = pred.squeeze(-1)
    if target.ndim == 3:
        target = target.squeeze(-1)
    if stats is not None:
        pred = denormalize_flux(pred, stats)
        target = denormalize_flux(target, stats)
    sim_times = sim_time.unsqueeze(-1) if sim_time.ndim == 1 else sim_time
    return pred.unsqueeze(1), target.unsqueeze(1), sim_times


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
    """Relative-squared-error loss on the lattice absorption QoI.

    QoIs are evaluated in physical flux space; if normalization stats are
    supplied, both flux tensors are denormalized first. Differentiable end
    to end so the loss backprops into the model.
    """
    pred_qoi, target_qoi, sim_times = _prepare_for_qoi(
        predicted_flux, target_flux, sim_time, flux_normalization_stats
    )
    qoi_pred = evaluate_lattice_qoi_torch(
        cell_centers,
        cell_areas,
        sigma_t,
        sigma_s,
        pred_qoi,
        sim_times,
    )
    with torch.no_grad():
        qoi_target = evaluate_lattice_qoi_torch(
            cell_centers,
            cell_areas,
            sigma_t,
            sigma_s,
            target_qoi,
            sim_times,
        )
    loss = _relative_squared_error_loss(
        qoi_pred["cur_absorption"][:, 0],
        qoi_target["cur_absorption"][:, 0],
        epsilon,
    )
    return loss, {"loss_qoi_absorption": loss.item()}


def compute_hohlraum_qoi_loss(
    predicted_flux: torch.Tensor,
    target_flux: torch.Tensor,
    cell_centers: torch.Tensor,
    cell_areas: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    sim_time: torch.Tensor,
    geometry_params: dict,
    flux_normalization_stats: Optional[Mapping[str, Any]] = None,
    epsilon: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean of the four hohlraum region relative-squared-error losses.

    Loss = mean of {center, vertical, horizontal, total} so every region
    contributes to the gradient. All four are recorded in the details dict.
    """
    pred_qoi, target_qoi, sim_times = _prepare_for_qoi(
        predicted_flux, target_flux, sim_time, flux_normalization_stats
    )
    qoi_pred = evaluate_hohlraum_qoi_torch(
        cell_centers,
        cell_areas,
        sigma_t,
        sigma_s,
        pred_qoi,
        sim_times,
        geometry_params,
    )
    with torch.no_grad():
        qoi_target = evaluate_hohlraum_qoi_torch(
            cell_centers,
            cell_areas,
            sigma_t,
            sigma_s,
            target_qoi,
            sim_times,
            geometry_params,
        )

    region_losses: dict[str, torch.Tensor] = {}
    pred_sum = target_sum = None
    for key in (
        "cur_absorption_center",
        "cur_absorption_vertical",
        "cur_absorption_horizontal",
    ):
        p, t = qoi_pred[key][:, 0], qoi_target[key][:, 0]
        region_losses[key.removeprefix("cur_absorption_")] = (
            _relative_squared_error_loss(p, t, epsilon)
        )
        pred_sum = p if pred_sum is None else pred_sum + p
        target_sum = t if target_sum is None else target_sum + t
    region_losses["total"] = _relative_squared_error_loss(pred_sum, target_sum, epsilon)

    loss = torch.stack(list(region_losses.values())).mean()
    details = {f"loss_qoi_{name}": val.item() for name, val in region_losses.items()}
    return loss, details


def compute_physics_loss(
    case_type: str,
    predicted_flux: torch.Tensor,
    target_flux: torch.Tensor,
    cell_centers: torch.Tensor,
    cell_areas: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    sim_time: torch.Tensor,
    sample=None,
    flux_normalization_stats: dict | None = None,
    qoi_epsilon: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dispatch the per-case QoI loss; returns ``(loss, per-region details)``."""
    common = dict(
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
    if case_type == "lattice":
        return compute_lattice_qoi_loss(**common)
    if case_type == "hohlraum":
        if sample is None:
            raise ValueError(
                "hohlraum physics loss requires the sample TensorDict to read "
                "geometry parameters (ulr, llr, urr, lrr, hlr, hrr, cx, cy)"
            )
        geometry_params = extract_geometry_params(sample)
        if not geometry_params:
            raise ValueError(
                "could not read hohlraum geometry parameters from the sample "
                "TensorDict; expected 8 0-D float32 tensors (ulr, llr, urr, "
                "lrr, hlr, hrr, cx, cy) on the TD top level (see "
                "MeshDataReader.load)"
            )
        return compute_hohlraum_qoi_loss(**common, geometry_params=geometry_params)
    raise ValueError(
        f"Unknown case type: {case_type}. Must be 'lattice' or 'hohlraum'."
    )
