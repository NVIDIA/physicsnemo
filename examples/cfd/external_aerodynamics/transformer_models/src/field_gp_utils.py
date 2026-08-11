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

"""Utilities for the GeoTransolver + field-GP per-point UQ pipeline.

This module supports a pointwise multitask GP head
(:class:`~physicsnemo.experimental.uq.FieldVariationalGPHead`) that
replaces the GeoTransolver readout: the GP posterior mean is the per-point
surface field prediction (pressure + 3 wall-shear-stress components) and the
posterior variance is the per-point uncertainty.

Provides:

* ``compute_field_targets_from_batch`` — per-point field-target extraction.
* ``probe_feature_dim`` — the backbone's per-point feature width, which the
  head needs as its ``input_dim``.
* ``collect_inducing_features`` — gather per-point backbone features to seed
  the GP inducing points.

The KL ramp and the non-DDP gradient sync are shared with the scalar-GP recipe
and live in :mod:`gp_utils` (``gp_ramp_weight``, ``sync_non_ddp_gradients``).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from jaxtyping import Float
from torch.utils.data import DataLoader

from gp_utils import cast_precisions

# Number of surface field channels predicted by the field GP:
#   index 0      -> pressure
#   indices 1:4  -> wall shear stress (x, y, z)
NUM_SURFACE_TASKS = 4


# ---------------------------------------------------------------------------
# Per-point targets
# ---------------------------------------------------------------------------


def compute_field_targets_from_batch(
    batch: dict,
) -> Float[torch.Tensor, "batch points tasks"]:
    """Return the per-point (normalized) surface field targets.

    Parameters
    ----------
    batch : dict
        A batch from the Transolver datapipe.  Only ``fields`` is read: the
        point-subsampled target stream, which is the same set of points the GP
        head sees, so targets and features stay aligned.  The datapipe emits it
        as a one-element list in ``combined`` mode and as a tensor otherwise.

    Returns
    -------
    Float[torch.Tensor, "batch points tasks"]
        Targets in the normalized space the GP is trained in, with ``tasks``
        ordered pressure first and then the three wall-shear-stress components.
    """
    fields = batch["fields"]
    if isinstance(fields, list):
        fields = fields[0]
    return fields


# ---------------------------------------------------------------------------
# Backbone feature width
# ---------------------------------------------------------------------------


@torch.no_grad()
def probe_feature_dim(
    model: nn.Module,
    batch: dict,
    precision: str,
    max_points: int | None = None,
    logger: logging.Logger | None = None,
) -> int:
    """Return the backbone's per-point feature width from one forward pass.

    The width is a property of the backbone's configuration, but reading it off
    the config would mean re-deriving GeoTransolver's internals here; one
    forward pass is authoritative instead.  Training and inference both need it
    to build a head whose ``input_dim`` matches the checkpoint.

    Parameters
    ----------
    model : nn.Module
        Backbone accepting ``return_point_features=True``.  Run in eval mode;
        the previous mode is restored before returning.
    batch : dict
        Any batch from the datapipe, used only for its shapes and dtypes.
    precision : str
        Precision string understood by ``cast_precisions``.
    max_points : int | None, optional
        Truncate the point dimension to this many points.  The result does not
        depend on the count, so a small value keeps the probe cheap on
        full-mesh batches.  ``None`` (default) uses every point.
    logger : logging.Logger | None, optional
        If given, the probed width is logged.

    Returns
    -------
    int
        Size of the last dimension of the backbone's per-point features.
    """
    was_training = model.training
    model.eval()
    try:
        embeddings = batch["embeddings"]
        if max_points is not None:
            embeddings = embeddings[:, : min(embeddings.shape[1], max_points)]
        embeddings = cast_precisions(embeddings, precision)
        features = cast_precisions(batch["fx"], precision)
        geometry = (
            cast_precisions(batch["geometry"], precision)
            if "geometry" in batch
            else None
        )
        _, point_features = model(
            global_embedding=features,
            local_embedding=embeddings,
            geometry=geometry,
            local_positions=embeddings[:, :, :3],
            return_point_features=True,
        )
    finally:
        if was_training:
            model.train()
    dim = int(point_features.shape[-1])
    if logger is not None:
        logger.info(f"Probed per-point backbone feature dim: {dim}")
    return dim


# ---------------------------------------------------------------------------
# Inducing-point seeding
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_inducing_features(
    model: nn.Module,
    dataloader: DataLoader,
    n_inducing: int,
    precision: str,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> Float[torch.Tensor, "n_inducing input_dim"]:
    """Collect ``n_inducing`` per-point backbone features to seed the GP.

    The head's default inducing points are random normal draws, which sit
    nowhere near the backbone's feature distribution; the first epochs would be
    spent dragging them into it.  Seeding from real features instead starts the
    variational approximation where the data is.

    Runs the backbone (eval mode) over batches, harvesting a random subset of
    per-point features from each until ``n_inducing`` have been gathered.  Which
    points are drawn follows torch's global RNG, so it is reproducible exactly
    when the run is seeded (``cfg.seed``); it does not need to agree across
    ranks, because the caller broadcasts one rank's result to the others.

    Parameters
    ----------
    model : nn.Module
        Backbone accepting ``return_point_features=True``.
    dataloader : DataLoader
        Source of batches; iterated only until enough points are collected.
    n_inducing : int
        Number of feature vectors to return.
    precision : str
        Precision string understood by ``cast_precisions``.
    device : torch.device
        Device for the returned tensor.
    logger : logging.Logger | None, optional
        If given, logs the collected count, width and feature-norm range.

    Returns
    -------
    Float[torch.Tensor, "n_inducing input_dim"]
        Features in the backbone's *raw* output space, not the kernel's:
        :meth:`~physicsnemo.experimental.uq.FieldVariationalGPHead.set_inducing_points`
        applies the DKL extractor and feature norm when it installs them.
    """
    model.eval()
    collected: list[torch.Tensor] = []
    n_have = 0
    for batch in dataloader:
        features = cast_precisions(batch["fx"], precision)
        embeddings = cast_precisions(batch["embeddings"], precision)
        geometry = (
            cast_precisions(batch["geometry"], precision)
            if "geometry" in batch
            else None
        )
        local_positions = embeddings[:, :, :3]
        _, point_features = model(
            global_embedding=features,
            local_embedding=embeddings,
            geometry=geometry,
            local_positions=local_positions,
            return_point_features=True,
        )
        # point_features: (B, N, D) -> (B*N, D)
        pf = point_features.reshape(-1, point_features.shape[-1])
        take = min(pf.shape[0], n_inducing - n_have)
        idx = torch.randperm(pf.shape[0], device=pf.device)[:take]
        collected.append(pf[idx].detach())
        n_have += take
        if n_have >= n_inducing:
            break

    inducing = torch.cat(collected, dim=0)[:n_inducing].to(device)
    if logger is not None:
        logger.info(
            f"Collected {inducing.shape[0]} inducing-point features "
            f"(dim {inducing.shape[1]}, norm range "
            f"[{inducing.norm(dim=1).min():.4f}, {inducing.norm(dim=1).max():.4f}])"
        )
    return inducing
