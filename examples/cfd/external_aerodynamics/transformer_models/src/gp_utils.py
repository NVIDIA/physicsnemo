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

"""Utilities for the GeoTransolver + GP drag prediction pipeline.

This module provides:

* Aerodynamic force-coefficient computation from surface fields.
* Drag-target extraction from dataloader batches.
* An MLP drag-prediction head (``DragMLP``) used as a GP-free baseline.
* Embedding-reduction factory (``create_embedding_reduction``).
* GP warmup helpers, inducing-point re-initialisation, and gradient syncing.
* Spectral-norm utilities for SNGP-style distance preservation.
* Checkpoint loading helpers.

Designed to be imported by the training script (``train_gp_combined.py``)
and evaluation / plotting scripts.
"""

from __future__ import annotations

import types
from typing import Any, Literal

import fsspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import physicsnemo
from physicsnemo.nn.module.pooling import AttentionPooling, MeanPooling
from physicsnemo.utils.checkpoint import (
    _get_checkpoint_filename,
    _unique_model_names,
    checkpoint_logging,
)
from physicsnemo.models.domino.utils import unstandardize

from train import cast_precisions  # noqa: F401 (re-exported for convenience)

# ---------------------------------------------------------------------------
# Aerodynamic reference constants
# ---------------------------------------------------------------------------

FRONTAL_AREA = 1.85  # m²
REFERENCE_VELOCITY = 40.0  # m/s
REFERENCE_DENSITY = 1.225  # kg/m³
DRAG_COEFF_SCALE = 0.35  # GP target = Cd / DRAG_COEFF_SCALE


# ---------------------------------------------------------------------------
# Force-coefficient computation
# ---------------------------------------------------------------------------


def compute_force_coefficients_torch(
    normals: torch.Tensor,
    area: torch.Tensor,
    coeff: float,
    p: torch.Tensor,
    wss: torch.Tensor,
    force_direction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute force coefficients from surface pressure and wall shear stress.

    Parameters
    ----------
    normals : torch.Tensor
        Surface normals, shape ``(N, 3)``.
    area : torch.Tensor
        Cell areas, shape ``(N,)`` or ``(N, 1)``.
    coeff : float
        Reference coefficient ``2 / (A * rho * U²)``.
    p : torch.Tensor
        Surface pressure, shape ``(N,)``.
    wss : torch.Tensor
        Wall shear stress, shape ``(N, 3)``.
    force_direction : torch.Tensor | None
        Unit vector for force projection; defaults to ``[1, 0, 0]`` (drag).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(c_total, c_pressure, c_friction)`` — scalar tensors.
    """
    if force_direction is None:
        force_direction = torch.tensor(
            [1.0, 0.0, 0.0],
            device=normals.device,
            dtype=normals.dtype,
        )
    area = area.view(-1)
    n_dot_f = (normals * force_direction).sum(dim=-1)
    c_p = coeff * (n_dot_f * area * p).sum()
    wss_dot_f = (wss * force_direction).sum(dim=-1)
    c_f = -coeff * (wss_dot_f * area).sum()
    return c_p + c_f, c_p, c_f


def compute_drag_target_from_batch(
    batch: dict,
    surface_factors: dict,
    device: torch.device,
    drag_scale: float = DRAG_COEFF_SCALE,
) -> torch.Tensor:
    """Extract a GP-scaled drag target from a dataloader batch.

    Unnormalises predicted surface fields, integrates pressure and shear to
    obtain the drag coefficient Cd, then returns ``Cd / drag_scale`` as a
    ``(1,)`` tensor suitable for GP training.
    """
    if "fields_full" in batch:
        fields = batch["fields_full"]
    else:
        fields = batch["fields"]
    if isinstance(fields, list):
        fields = fields[0]

    fields_phys = unstandardize(fields, surface_factors["mean"], surface_factors["std"])
    fields_phys = fields_phys.squeeze(0)
    p = fields_phys[:, 0]
    wss = fields_phys[:, 1:4]

    normals = batch["surface_normals"].squeeze(0).to(device, dtype=fields_phys.dtype)
    area = batch["surface_areas"].squeeze(0).to(device, dtype=fields_phys.dtype)
    p, wss = p.to(device), wss.to(device)

    coeff = 2.0 / (FRONTAL_AREA * REFERENCE_DENSITY * REFERENCE_VELOCITY**2)
    c_total, _, _ = compute_force_coefficients_torch(normals, area, coeff, p, wss)
    return (c_total / drag_scale).unsqueeze(0)


def compute_drag_from_subsampled_outputs(
    outputs: torch.Tensor,
    batch: dict,
    surface_factors: dict,
    device: torch.device,
    drag_scale: float = DRAG_COEFF_SCALE,
) -> torch.Tensor:
    """Monte-Carlo drag estimate from subsampled GeoTransolver predictions.

    Preserves the computational graph through *outputs* so gradients can
    flow back into the GeoTransolver.  Returns ``(1,)`` in GP-scaled space.
    """
    fields_phys = unstandardize(
        outputs,
        surface_factors["mean"],
        surface_factors["std"],
    ).squeeze(0)
    p = fields_phys[:, 0]
    wss = fields_phys[:, 1:4]

    normals = (
        batch["surface_normals_sub"].squeeze(0).to(device, dtype=fields_phys.dtype)
    )
    areas = batch["surface_areas_sub"].squeeze(0).to(device, dtype=fields_phys.dtype)

    n_full = batch["surface_areas"].squeeze(0).shape[0]
    n_sub = p.shape[0]
    coeff = 2.0 / (FRONTAL_AREA * REFERENCE_DENSITY * REFERENCE_VELOCITY**2)
    scale = n_full / n_sub

    c_total, _, _ = compute_force_coefficients_torch(
        normals,
        areas,
        coeff * scale,
        p,
        wss,
    )
    return (c_total / drag_scale).unsqueeze(0)


# ---------------------------------------------------------------------------
# DragMLP — simple baseline head
# ---------------------------------------------------------------------------


class DragMLP(nn.Module):
    """Simple MLP head for drag prediction — drop-in replacement for GP head.

    Provides the same ``forward_and_loss`` / ``predict`` interface so training
    and evaluation scripts can swap between GP and MLP via config.  ``predict``
    returns zero variance so downstream plotting code works unchanged.

    Parameters
    ----------
    input_dim : int
        Dimension of the input embedding.
    hidden : list[int] | None
        Hidden layer sizes. Defaults to ``[256, 256]``.
    """

    def __init__(self, input_dim: int = 32, hidden: list[int] | None = None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding).squeeze(-1)

    def forward_and_loss(
        self,
        embedding: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.forward(embedding)
        return pred, F.mse_loss(pred, target)

    def loss(self, embedding: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between predicted and target drag."""
        _, mse = self.forward_and_loss(embedding, target)
        return mse

    @torch.no_grad()
    def predict(
        self,
        embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict drag with zero variance (MLP has no uncertainty estimate)."""
        self.eval()
        pred = self.forward(embedding)
        zeros = torch.zeros_like(pred)
        return pred, zeros, pred, pred


# ---------------------------------------------------------------------------
# Embedding-reduction factory
# ---------------------------------------------------------------------------


def create_embedding_reduction(
    pooling: Literal["attention", "mean"],
    feat_dim: int = 256,
    embed_dim: int = 32,
    spectral_norm: bool = False,
    normalize: bool = False,
    target_scale: float = 1.0,
    **kwargs: Any,
) -> nn.Module:
    """Create an embedding-reduction (pooling) module from config strings.

    Parameters
    ----------
    pooling : ``"attention"`` or ``"mean"``
        Pooling strategy.
    feat_dim, embed_dim, spectral_norm, normalize, target_scale
        Forwarded to the pooling constructor.

    Returns
    -------
    nn.Module
        An ``AttentionPooling`` or ``MeanPooling`` instance.
    """
    if pooling == "attention":
        return AttentionPooling(
            feat_dim=feat_dim,
            embed_dim=embed_dim,
            spectral_norm=spectral_norm,
            normalize=normalize,
            target_scale=target_scale,
            **kwargs,
        )
    if pooling == "mean":
        return MeanPooling(
            feat_dim=feat_dim,
            embed_dim=embed_dim,
            spectral_norm=spectral_norm,
            normalize=normalize,
            target_scale=target_scale,
        )
    raise ValueError(f"Unknown pooling: {pooling!r}. Use 'attention' or 'mean'.")


# ---------------------------------------------------------------------------
# GP warmup helpers
# ---------------------------------------------------------------------------


def gp_ramp_weight(epoch: int, warmup_start: int, warmup_end: int) -> float:
    """Linear ramp: 0 before *warmup_start*, 0→1 over [start, end), 1 after."""
    if epoch < warmup_start:
        return 0.0
    if epoch >= warmup_end:
        return 1.0
    return (epoch - warmup_start) / (warmup_end - warmup_start)


def sync_non_ddp_gradients(modules: list[nn.Module], world_size: int) -> None:
    """All-reduce gradients for modules that are not wrapped in DDP."""
    if world_size <= 1:
        return
    for module in modules:
        for p in module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


@torch.no_grad()
def reinitialize_inducing_points(
    model: nn.Module,
    embedding_reduction: nn.Module,
    head,
    dataloader,
    n_inducing: int,
    n_train: int,
    train_indices: list[int],
    precision: str,
    device: torch.device,
    logger,
) -> None:
    """Re-collect inducing-point embeddings from the current (trained) model.

    Temporarily uses the full dataset so every rank can collect at least
    *n_inducing* samples, then restores distributed indices.
    """
    dataloader.dataset.set_indices(list(range(n_train)))

    model.eval()
    embedding_reduction.eval()
    init_embeddings: list[torch.Tensor] = []
    for batch in dataloader:
        if len(init_embeddings) >= n_inducing:
            break
        features = cast_precisions(batch["fx"], precision)
        embeddings = cast_precisions(batch["embeddings"], precision)
        geometry = (
            cast_precisions(batch["geometry"], precision)
            if "geometry" in batch
            else None
        )
        local_positions = embeddings[:, :, :3]
        _, emb_states = model(
            global_embedding=features,
            local_embedding=embeddings,
            geometry=geometry,
            local_positions=local_positions,
            return_embedding_states=True,
        )
        reduced = embedding_reduction(emb_states.flatten(1, 2))
        init_embeddings.append(reduced.cpu())

    dataloader.dataset.set_indices(train_indices)

    init_embeddings_t = torch.cat(init_embeddings, dim=0)[:n_inducing].to(device)
    if head.feature_extractor is not None:
        init_embeddings_t = head.feature_extractor(init_embeddings_t)
    head.gp_layer.variational_strategy.inducing_points.data.copy_(init_embeddings_t)

    vd = head.gp_layer.variational_strategy._variational_distribution
    vd.variational_mean.data.zero_()
    vd.chol_variational_covar.data.copy_(torch.eye(n_inducing, device=device) * 0.01)
    logger.info(
        f"Re-initialised {n_inducing} inducing points from current embeddings "
        f"(norm range [{init_embeddings_t.norm(dim=1).min():.4f}, "
        f"{init_embeddings_t.norm(dim=1).max():.4f}])"
    )


# ---------------------------------------------------------------------------
# Spectral-norm utilities
# ---------------------------------------------------------------------------


def apply_spectral_norm_to_model(
    model: nn.Module,
    coeff: float = 1.0,
    skip_output_proj: bool = True,
) -> None:
    """Apply spectral normalization to all ``nn.Linear`` layers in *model*.

    When *coeff* > 1 the constraint is "soft" (DUE / SNGP convention).
    *skip_output_proj* excludes the final output projection (``ln_mlp_out``).
    """
    for name, module in list(model.named_modules()):
        if skip_output_proj and "ln_mlp_out" in name:
            continue
        if isinstance(module, nn.Linear):
            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            wrapped = torch.nn.utils.parametrizations.spectral_norm(module)
            if coeff != 1.0:
                wrapped._coeff = coeff
                orig_forward = wrapped.forward

                def _scaled_forward(self, x, _c=coeff, _fwd=orig_forward):
                    return _fwd(x) * _c

                wrapped.forward = types.MethodType(_scaled_forward, wrapped)
            setattr(parent, attr_name, wrapped)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_pretrained_model_only(
    model: torch.nn.Module,
    path: str,
    epoch: int | None = None,
) -> bool:
    """Load only the model state from a checkpoint (no optimizer/scheduler).

    Returns ``True`` if at least one model file was loaded.
    """
    fs = fsspec.filesystem(fsspec.utils.get_protocol(path))
    if not fs.exists(path):
        checkpoint_logging.warning(
            f"Pretrained checkpoint path does not exist: {path}, skipping load"
        )
        return False
    models_dict = _unique_model_names([model], loading=True)
    loaded_any = False
    for name, m in models_dict.items():
        if not isinstance(m, physicsnemo.core.Module):
            continue
        file_name = _get_checkpoint_filename(
            path,
            base_name=name,
            index=epoch,
            model_type="mdlus",
        )
        if fs.exists(file_name):
            m.load(file_name)
            checkpoint_logging.success(f"Loaded pretrained model state: {file_name}")
            loaded_any = True
        else:
            checkpoint_logging.warning(
                f"Could not find pretrained model file: {file_name}, skipping"
            )
    return loaded_any
