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

r"""Hydra-friendly builders for the AeroJEPA training-loss stack.

Library-side loss modules live in
:mod:`physicsnemo.experimental.metrics.aerojepa`. This module is the
recipe-side glue that picks one out by name + kwargs from a Hydra
config block (the ``loss:`` section of ``conf/training/superwing.yaml``)
and instantiates it. It also exposes :func:`compute_latent_loss`, the
MSE + cosine blend the predictor minimises against the target encoder
in latent space.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from physicsnemo.experimental.metrics.aerojepa import (
    MSELoss,
    RelativeL2Loss,
    RelativeL2MSELoss,
    RelativeMSELoss,
    TokenLatentSIGReg,
)


def build_sigreg_from_config(
    sigreg_cfg: DictConfig | dict[str, Any],
) -> TokenLatentSIGReg:
    r"""Build a :class:`TokenLatentSIGReg` from a config block.

    Recognises the ``knots`` and ``num_proj`` fields. Other keys (e.g.
    ``weight`` / ``warmup_epochs``) are ignored here — they're scheduled
    by the training loop, not by the SIGReg module.

    Parameters
    ----------
    sigreg_cfg : DictConfig or dict
        ``loss.sigreg`` config block.

    Returns
    -------
    TokenLatentSIGReg
        The constructed regulariser module.
    """
    return TokenLatentSIGReg(
        knots=int(sigreg_cfg.get("knots", 17)),
        num_proj=int(sigreg_cfg.get("num_proj", 1024)),
    )


def build_recon_loss_from_config(
    recon_cfg: DictConfig | dict[str, Any],
) -> nn.Module:
    r"""Build a reconstruction-loss module from a config block.

    The ``kind`` field selects one of the library-side reconstruction
    losses; remaining keys are forwarded to that class. Recognised
    kinds:

    * ``"mse"``              — :class:`MSELoss`.
    * ``"relative_l2"``      — :class:`RelativeL2Loss`.
    * ``"relative_mse"``     — :class:`RelativeMSELoss`.
    * ``"relative_l2_mse"``  — :class:`RelativeL2MSELoss` (the hybrid
      default used in the SuperWing tutorial).

    Parameters
    ----------
    recon_cfg : DictConfig or dict
        ``loss.recon`` config block.

    Returns
    -------
    torch.nn.Module
        The constructed loss module.

    Raises
    ------
    ValueError
        If ``kind`` is missing or not recognised.
    """
    kind = str(recon_cfg.get("kind", "mse")).lower()
    channel_weights = recon_cfg.get("channel_weights", None)
    if channel_weights is not None:
        channel_weights = list(channel_weights)

    if kind == "mse":
        return MSELoss(channel_weights=channel_weights)
    if kind == "relative_l2":
        return RelativeL2Loss(
            eps=float(recon_cfg.get("eps", 1e-12)),
            channel_weights=channel_weights,
        )
    if kind == "relative_mse":
        return RelativeMSELoss(
            eps=float(recon_cfg.get("eps", 1e-6)),
            mode=str(recon_cfg.get("relative_mse_mode", "pointwise")),
            channel_weights=channel_weights,
        )
    if kind == "relative_l2_mse":
        return RelativeL2MSELoss(
            relative_l2_weight=float(recon_cfg.get("relative_l2_weight", 1.0)),
            mse_weight=float(recon_cfg.get("mse_weight", 0.1)),
            relative_l2_eps=float(recon_cfg.get("eps", 1e-12)),
            channel_weights=channel_weights,
        )
    raise ValueError(
        f"Unknown reconstruction loss kind: {kind!r} "
        "(expected one of: 'mse', 'relative_l2', 'relative_mse', "
        "'relative_l2_mse')."
    )


def compute_latent_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mse_weight: float = 0.5,
    cosine_weight: float = 0.5,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Compute the JEPA latent-space loss between predicted and target tokens.

    Combines token-wise MSE with token-wise cosine distance
    ``1 - cos(predicted, target)``:

    .. code-block:: text

        loss = mse_weight * mean_valid(|pred - tgt|^2)
             + cosine_weight * mean_valid(1 - cos(pred, tgt))

    The target tensor is treated as a stop-gradient — the JEPA target
    encoder is updated through EMA or its own optimiser branch, not by
    this loss.

    Parameters
    ----------
    predicted : torch.Tensor
        Predictor output of shape ``(B, T, D)`` or ``(T, D)``.
    target : torch.Tensor
        Target encoder output of the same shape as ``predicted``.
    mse_weight : float, optional
        Multiplier on the MSE component. Default ``0.5``.
    cosine_weight : float, optional
        Multiplier on the cosine-distance component. Default ``0.5``.
    mask : torch.Tensor, optional
        Boolean mask of shape ``(B, T)`` selecting valid tokens. Padded
        positions are excluded from the mean. ``None`` means every
        token is valid.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor. Returns a zero scalar when the mask leaves
        no valid token.
    """
    target = target.detach()
    if predicted.shape != target.shape:
        raise ValueError(
            "predicted and target must share shape; "
            f"got {tuple(predicted.shape)} vs {tuple(target.shape)}."
        )

    if mask is None:
        per_token_mse = (predicted - target).pow(2).mean(dim=-1)
        per_token_cosine = 1.0 - F.cosine_similarity(predicted, target, dim=-1)
        mse_term = per_token_mse.mean()
        cosine_term = per_token_cosine.mean()
    else:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.shape != predicted.shape[:-1]:
            raise ValueError(
                "mask must match predicted's leading dims; "
                f"got {tuple(mask.shape)} vs {tuple(predicted.shape[:-1])}."
            )
        valid = int(mask.sum().item())
        if valid == 0:
            return predicted.new_zeros(())
        per_token_mse = (predicted - target).pow(2).mean(dim=-1)
        per_token_cosine = 1.0 - F.cosine_similarity(predicted, target, dim=-1)
        mse_term = per_token_mse[mask].mean()
        cosine_term = per_token_cosine[mask].mean()

    return float(mse_weight) * mse_term + float(cosine_weight) * cosine_term
