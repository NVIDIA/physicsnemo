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

"""FGN training loss functions (arXiv:2506.10772 §2.2).

fair_crps          — eq. (4): fair CRPS over ensemble members.
ensemble_mean_mse  — MSE of the ensemble mean (supplementary term).
build_channel_weights — per-variable weights (§2.2.3 / GraphCast scheme).
build_area_weights    — cos(lat) area weights (§2.2.3).
"""

from __future__ import annotations

import re

import numpy as np
import torch


def fair_crps(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fair CRPS (paper eq. 4), mean-reduced over batch and spatial dims.

    Parameters
    ----------
    ensemble : ``(B, M, C, H, W)``
        M ensemble members.
    target : ``(B, C, H, W)``
        Ground-truth state.
    weights : broadcastable to ``(B, C, H, W)``, optional
        Per-location/channel loss weights (eq. 5).

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if ensemble.ndim != 5:
        raise ValueError(
            f"ensemble must have shape [B, M, C, H, W], got {tuple(ensemble.shape)}"
        )
    if target.ndim != 4:
        raise ValueError(
            f"target must have shape [B, C, H, W], got {tuple(target.shape)}"
        )
    M = ensemble.shape[1]
    if M < 2:
        raise ValueError(f"fair_crps requires at least two ensemble members, got M={M}")

    # term1: E[|X - y|] per location — shape (B, C, H, W)
    term1 = (ensemble - target.unsqueeze(1)).abs().mean(dim=1)

    # term2: (1/2) E[|X - X'|] per location via exhaustive pairwise sum.
    # The diagonal is zero so including it is free; the factor 2M(M-1) in the
    # denominator matches the fair (unbiased) estimator in eq. (4).
    x_i = ensemble.unsqueeze(2)  # (B, M, 1, C, H, W)
    x_j = ensemble.unsqueeze(1)  # (B, 1, M, C, H, W)
    term2 = (x_i - x_j).abs().sum(dim=(1, 2)) / (2.0 * M * (M - 1))

    per_loc = term1 - term2  # (B, C, H, W)
    if weights is not None:
        per_loc = per_loc * weights
    return per_loc.mean()


def ensemble_mean_mse(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE of the ensemble mean, mean-reduced over batch and spatial dims.

    Parameters
    ----------
    ensemble : ``(B, M, C, H, W)``
    target : ``(B, C, H, W)``
    weights : broadcastable to ``(B, C, H, W)``, optional
    """
    sq_err = (ensemble.mean(dim=1) - target).pow(2)
    if weights is not None:
        sq_err = sq_err * weights
    return sq_err.mean()


def build_channel_weights(state_channels: list[str]) -> np.ndarray:
    """Per-channel loss weights following paper §2.2.3 / GraphCast scheme.

    Rules
    -----
    - Atmospheric channels (name matches ``<prefix><level>`` with a purely
      numeric suffix, e.g. ``t500``): weight = level / sum_of_levels_in_prefix.
      Geopotential (``z*``) weights are halved to tame overfitting.
    - Surface channels (anything else): weight = 0.1, except ``t2m`` = 1.0.

    Returns
    -------
    np.ndarray, shape ``(C,)``, float32.
    """
    # Identify atmospheric channels and their pressure levels.
    _atmos_re = re.compile(r"^([a-zA-Z]+)(\d+)$")
    prefix_levels: dict[str, list[int]] = {}
    for ch in state_channels:
        m = _atmos_re.match(ch)
        if m:
            prefix_levels.setdefault(m.group(1), []).append(int(m.group(2)))

    prefix_sum: dict[str, float] = {
        p: float(sum(lvls)) for p, lvls in prefix_levels.items()
    }

    weights = np.zeros(len(state_channels), dtype=np.float32)
    for i, ch in enumerate(state_channels):
        m = _atmos_re.match(ch)
        if m:
            prefix, level = m.group(1), int(m.group(2))
            w = level / prefix_sum[prefix]
            if prefix == "z":
                w *= 0.5
            weights[i] = w
        elif ch == "t2m":
            weights[i] = 1.0
        else:
            weights[i] = 0.1

    return weights


def build_area_weights(H: int) -> np.ndarray:
    """Cos(lat) area weights, normalised so the mean over rows equals 1.

    Follows ERA5 north-to-south ordering (lat 90° → −90°).

    Parameters
    ----------
    H : int
        Number of latitude rows (e.g. 721 for 0.25° ERA5).

    Returns
    -------
    np.ndarray, shape ``(H, 1)``, float32.
    """
    lats = np.linspace(90.0, -90.0, H, dtype=np.float64)
    w = np.cos(np.deg2rad(lats))
    w /= w.mean()
    return w.astype(np.float32).reshape(H, 1)
