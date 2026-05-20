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

"""Generic GP / UQ recipe helpers for the active-learning example.

This is the *recipe* layer of the example: nothing in here knows about
external aerodynamics. CFD-specific constants and surface-field
integrals live in ``aero_physics.py``.

Contents:

* Embedding-reduction factory (``create_embedding_reduction``).
* GP warmup helpers, inducing-point re-initialisation, and gradient
  syncing for non-DDP modules.
* Spectral-norm utilities for SNGP-style distance preservation.
* Simple MLP head (``DragMLP``) — a drop-in baseline for the GP head.
* Checkpoint loading helpers.
"""

from __future__ import annotations

import logging
import types
from typing import Any, Literal

import fsspec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

import physicsnemo
from physicsnemo.experimental.uq import VariationalGPHead
from physicsnemo.nn.module.pooling import AttentionPooling, MeanPooling
from physicsnemo.utils.checkpoint import (
    _get_checkpoint_filename,
    _unique_model_names,
    checkpoint_logging,
)

from utils import cast_precisions

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
    elif pooling == "mean":
        return MeanPooling(
            feat_dim=feat_dim,
            embed_dim=embed_dim,
            spectral_norm=spectral_norm,
            normalize=normalize,
            target_scale=target_scale,
        )
    else:
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
    head: VariationalGPHead,
    dataloader: DataLoader,
    n_inducing: int,
    n_train: int,
    train_indices: list[int],
    precision: str,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    """Re-collect inducing-point embeddings from the current trained model.

    The inducing points seeded at GP construction time become stale once
    the backbone has moved through its initial warm-up. This helper re-
    seeds them from a forward pass over the current data so the GP
    posterior covers the current embedding distribution. The variational
    mean is zeroed and the variational covariance is reset to a small
    identity, restarting GP-side optimisation cleanly while leaving the
    encoder unchanged.

    Parameters
    ----------
    model : nn.Module
        Backbone encoder (DDP-wrapped or unwrapped).
    embedding_reduction : nn.Module
        Pooling module mapping per-token embeddings to a global vector.
    head : VariationalGPHead
        GP head whose inducing points / variational params are reset.
    dataloader : torch.utils.data.DataLoader
        Loader over the training pool. Its dataset is temporarily set
        to cover the full ``n_train`` range so every rank can collect
        at least ``n_inducing`` samples, then ``train_indices`` is
        restored on exit.
    n_inducing : int
        Number of inducing points to collect (must match the GP head).
    n_train : int
        Total number of training samples in the pool.
    train_indices : list[int]
        The training indices to restore after collection.
    precision : str
        Forward-pass precision ("float32" / "bfloat16" / "float16").
    device : torch.device
        Device on which the new inducing points and variational params
        are stored.
    logger : logging.Logger
        Logger for the post-collection summary line.
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
    """Apply spectral normalization to all ``nn.Linear`` layers in a model.

    Spectral normalization bounds each linear layer's largest singular
    value, which makes the encoder approximately distance-preserving —
    a prerequisite for SNGP / DUE-style uncertainty estimation. The
    final output projection is typically excluded so the regression
    head can still freely scale outputs.

    Parameters
    ----------
    model : nn.Module
        Model whose ``nn.Linear`` children should be wrapped in place.
    coeff : float, optional
        Spectral-norm coefficient. ``coeff == 1`` is the hard
        constraint provided by ``torch.nn.utils.parametrizations
        .spectral_norm``; ``coeff > 1`` is the "soft" DUE / SNGP
        convention where the constraint is relaxed by a constant
        factor. Default ``1.0``.
    skip_output_proj : bool, optional
        If ``True`` (default), skip any submodule whose qualified name
        contains ``"ln_mlp_out"`` — the GeoTransolver output head.
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
