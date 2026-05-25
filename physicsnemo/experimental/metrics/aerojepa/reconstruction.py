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

"""Reconstruction losses for AeroJEPA training.

Four loss families, each exposed as both a functional form and an
``nn.Module``:

* :func:`mse_loss` / :class:`MSELoss` — channel-weighted MSE with optional
  per-point weights and a validity mask.
* :func:`relative_l2_loss` / :class:`RelativeL2Loss` — per-channel relative
  L2 averaged across the channel axis.
* :func:`relative_mse_loss` / :class:`RelativeMSELoss` — relative MSE with
  selectable normalization (``pointwise`` or ``channel_max``).
* :func:`relative_l2_mse_loss` / :class:`RelativeL2MSELoss` — convex
  combination of the relative-L2 and MSE losses.

Each Module variant stores ``channel_weights`` as a persistent buffer when
supplied and as a non-persistent ``None`` buffer otherwise, so checkpoints
record the weights for reproducibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Plain MSE
# ---------------------------------------------------------------------------


def mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    point_weights: torch.Tensor | None = None,
    channel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Channel-weighted mean squared error with optional mask and point weights.

    The per-channel squared error is multiplied by ``channel_weights`` (if
    given), then by ``point_weights`` (broadcast over the last axis), then
    masked by ``mask``. The denominator counts valid contributions so the
    result is a true weighted mean.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Target values; must have the same shape as ``pred``.
    mask : torch.Tensor, optional
        Validity mask broadcast over the channel axis. ``True``/non-zero
        entries contribute to the mean.
    point_weights : torch.Tensor, optional
        Per-point weights broadcast over the channel axis.
    channel_weights : torch.Tensor, optional
        Per-channel weights of shape ``(C,)`` where ``C = pred.shape[-1]``.

    Returns
    -------
    torch.Tensor
        Scalar MSE.

    Raises
    ------
    ValueError
        If ``pred.shape != target.shape`` or if ``channel_weights`` has the
        wrong last dimension.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shapes must match, got {pred.shape} vs {target.shape}"
        )

    err = (pred - target).pow(2)
    weight_sum = float(pred.shape[-1])
    if channel_weights is not None:
        cw = channel_weights.to(device=err.device, dtype=err.dtype).reshape(1, -1)
        if cw.shape[-1] != err.shape[-1]:
            raise ValueError(
                f"channel_weights dim must match last dim ({err.shape[-1]}), "
                f"got {cw.shape[-1]}"
            )
        while cw.ndim < err.ndim:
            cw = cw.unsqueeze(0)
        err = err * cw
        weight_sum = float(channel_weights.to(dtype=err.dtype).sum().item())

    point_weight_t = None
    if point_weights is not None:
        point_weight_t = point_weights.to(device=err.device, dtype=err.dtype)
        while point_weight_t.ndim < err.ndim:
            point_weight_t = point_weight_t.unsqueeze(-1)
        err = err * point_weight_t

    if mask is not None:
        mask_t = mask.to(device=err.device, dtype=err.dtype)
        while mask_t.ndim < err.ndim:
            mask_t = mask_t.unsqueeze(-1)
        err = err * mask_t
        denom_weights = mask_t if point_weight_t is None else point_weight_t * mask_t
        denom = denom_weights.sum().clamp_min(1.0) * max(weight_sum, 1e-12)
        return err.sum() / denom

    if point_weight_t is not None:
        denom = point_weight_t.sum().clamp_min(1.0) * max(weight_sum, 1e-12)
        return err.sum() / denom

    denom = err.numel() / err.shape[-1] * max(weight_sum, 1e-12)
    return err.sum() / denom


class MSELoss(nn.Module):
    r"""Module wrapper around :func:`mse_loss` with cached channel weights.

    Parameters
    ----------
    channel_weights : list of float, optional
        Per-channel weights to apply on every forward call. Stored as a
        persistent ``float32`` buffer when given; otherwise the buffer is
        registered as ``None`` (non-persistent).

    Shape
    -----
    - ``pred`` / ``target``: any matching shapes; last axis is the channel.
    - Optional ``mask`` and ``point_weights`` broadcast over the channel axis.
    - Output: scalar tensor.
    """

    def __init__(self, channel_weights: list[float] | None = None):
        super().__init__()
        if channel_weights is None:
            self.register_buffer("channel_weights", None, persistent=False)
        else:
            w = torch.tensor(channel_weights, dtype=torch.float32)
            self.register_buffer("channel_weights", w, persistent=True)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        point_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return mse_loss(
            pred,
            target,
            mask=mask,
            point_weights=point_weights,
            channel_weights=self.channel_weights,
        )


# ---------------------------------------------------------------------------
# Per-channel relative L2
# ---------------------------------------------------------------------------


def relative_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-12,
    mask: torch.Tensor | None = None,
    point_weights: torch.Tensor | None = None,
    channel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Per-channel relative L2 error, averaged across the channel axis.

    For inputs shaped ``(..., C)`` this computes
    ``sqrt(sum((pred - target)^2) / sum(target^2))`` per channel, where the
    sums run over every non-channel axis. When ``pred.ndim >= 3`` the
    leading axis is treated as the batch and preserved; the resulting
    per-batch, per-channel scores are averaged after channel weighting.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values with ``ndim >= 2``.
    target : torch.Tensor
        Target values; same shape as ``pred``.
    eps : float, optional
        Floor applied to the denominator before dividing. Default ``1e-12``.
    mask : torch.Tensor, optional
        Validity mask broadcast over the channel axis.
    point_weights : torch.Tensor, optional
        Per-point weights broadcast over the channel axis.
    channel_weights : torch.Tensor, optional
        Per-channel weights.

    Returns
    -------
    torch.Tensor
        Scalar relative-L2 loss.

    Raises
    ------
    ValueError
        If shapes do not match, if ``pred.ndim < 2``, or if
        ``channel_weights`` has the wrong last dimension.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shapes must match, got {pred.shape} vs {target.shape}"
        )
    if pred.ndim < 2:
        raise ValueError(f"pred/target must have ndim >= 2, got {pred.ndim}")

    err_sq = (pred - target).pow(2)
    tgt_sq = target.pow(2)

    if point_weights is not None:
        point_weight_t = point_weights.to(device=err_sq.device, dtype=err_sq.dtype)
        while point_weight_t.ndim < err_sq.ndim:
            point_weight_t = point_weight_t.unsqueeze(-1)
        err_sq = err_sq * point_weight_t
        tgt_sq = tgt_sq * point_weight_t

    if mask is not None:
        mask_t = mask.to(device=err_sq.device, dtype=err_sq.dtype)
        while mask_t.ndim < err_sq.ndim:
            mask_t = mask_t.unsqueeze(-1)
        err_sq = err_sq * mask_t
        tgt_sq = tgt_sq * mask_t

    if pred.ndim == 2:
        reduce_dims = (0,)
    else:
        reduce_dims = tuple(range(1, pred.ndim - 1))

    num = torch.sum(err_sq, dim=reduce_dims)
    den = torch.clamp_min(torch.sum(tgt_sq, dim=reduce_dims), float(eps))
    rel_l2 = torch.sqrt(num / den)

    weight_sum = float(pred.shape[-1])
    if channel_weights is not None:
        if rel_l2.ndim == 1:
            cw = channel_weights.to(device=rel_l2.device, dtype=rel_l2.dtype).reshape(
                -1
            )
        else:
            cw = channel_weights.to(device=rel_l2.device, dtype=rel_l2.dtype).reshape(
                1, -1
            )
        if cw.shape[-1] != rel_l2.shape[-1]:
            raise ValueError(
                f"channel_weights dim must match last dim ({rel_l2.shape[-1]}), "
                f"got {cw.shape[-1]}"
            )
        rel_l2 = rel_l2 * cw
        weight_sum = float(channel_weights.to(dtype=rel_l2.dtype).sum().item())

    if rel_l2.ndim == 1:
        return rel_l2.sum() / max(weight_sum, 1e-12)

    batch_factor = max(rel_l2.shape[0], 1)
    return rel_l2.sum() / float(batch_factor * max(weight_sum, 1e-12))


class RelativeL2Loss(nn.Module):
    r"""Module wrapper around :func:`relative_l2_loss`.

    Parameters
    ----------
    eps : float, optional
        Denominator floor passed to :func:`relative_l2_loss`. Default
        ``1e-12``.
    channel_weights : list of float, optional
        Per-channel weights. Stored as a persistent ``float32`` buffer when
        given.

    Shape
    -----
    Same as :func:`relative_l2_loss`. Output is a scalar.
    """

    def __init__(
        self,
        eps: float = 1e-12,
        channel_weights: list[float] | None = None,
    ):
        super().__init__()
        self.eps = eps
        if channel_weights is None:
            self.register_buffer("channel_weights", None, persistent=False)
        else:
            w = torch.tensor(channel_weights, dtype=torch.float32)
            self.register_buffer("channel_weights", w, persistent=True)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        point_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return relative_l2_loss(
            pred,
            target,
            eps=self.eps,
            mask=mask,
            point_weights=point_weights,
            channel_weights=self.channel_weights,
        )


# ---------------------------------------------------------------------------
# Relative MSE (selectable normalization)
# ---------------------------------------------------------------------------


def relative_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
    mode: str = "pointwise",
    mask: torch.Tensor | None = None,
    point_weights: torch.Tensor | None = None,
    channel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Relative MSE with selectable normalization.

    With ``mode="pointwise"`` the per-element squared error is divided by
    ``target ** 2 + eps`` and averaged. With ``mode="channel_max"`` the
    denominator is the per-batch per-channel max-absolute-target squared,
    broadcast back across the non-channel axes.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Target values; same shape as ``pred``. Must have ``ndim >= 2`` for
        the ``channel_max`` mode.
    eps : float, optional
        Floor added to the denominator. Default ``1e-6``.
    mode : str, optional
        ``"pointwise"`` or ``"channel_max"``. Default ``"pointwise"``.
    mask : torch.Tensor, optional
        Validity mask broadcast over the channel axis.
    point_weights : torch.Tensor, optional
        Per-point weights broadcast over the channel axis.
    channel_weights : torch.Tensor, optional
        Per-channel weights.

    Returns
    -------
    torch.Tensor
        Scalar relative MSE.

    Raises
    ------
    ValueError
        If shapes do not match, if ``mode`` is not recognised, if
        ``target.ndim < 2`` for ``channel_max``, or if ``channel_weights``
        has the wrong last dimension.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shapes must match, got {pred.shape} vs {target.shape}"
        )

    mode = mode.lower()
    if mode == "pointwise":
        denom = target.pow(2) + eps
    elif mode == "channel_max":
        if target.ndim == 2:
            reduce_dims: tuple[int, ...] = (0,)
        elif target.ndim >= 3:
            reduce_dims = tuple(range(1, target.ndim - 1))
        else:
            raise ValueError(f"target must have ndim >= 2, got {target.ndim}")
        ch_max = target.abs().amax(dim=reduce_dims, keepdim=True)
        denom = ch_max.pow(2) + eps
    else:
        raise ValueError(
            f"Unsupported mode '{mode}'. Use 'pointwise' or 'channel_max'."
        )

    rel = (pred - target).pow(2) / denom
    weight_sum = float(pred.shape[-1])
    if channel_weights is not None:
        cw = channel_weights.to(device=rel.device, dtype=rel.dtype).reshape(1, -1)
        if cw.shape[-1] != rel.shape[-1]:
            raise ValueError(
                f"channel_weights dim must match last dim ({rel.shape[-1]}), "
                f"got {cw.shape[-1]}"
            )
        while cw.ndim < rel.ndim:
            cw = cw.unsqueeze(0)
        rel = rel * cw
        weight_sum = float(channel_weights.to(dtype=rel.dtype).sum().item())

    point_weight_t = None
    if point_weights is not None:
        point_weight_t = point_weights.to(device=rel.device, dtype=rel.dtype)
        while point_weight_t.ndim < rel.ndim:
            point_weight_t = point_weight_t.unsqueeze(-1)
        rel = rel * point_weight_t

    if mask is not None:
        mask_t = mask.to(device=rel.device, dtype=rel.dtype)
        while mask_t.ndim < rel.ndim:
            mask_t = mask_t.unsqueeze(-1)
        rel = rel * mask_t
        denom_weights = mask_t if point_weight_t is None else point_weight_t * mask_t
        out_denom = denom_weights.sum().clamp_min(1.0) * max(weight_sum, 1e-12)
        return rel.sum() / out_denom

    if point_weight_t is not None:
        out_denom = point_weight_t.sum().clamp_min(1.0) * max(weight_sum, 1e-12)
        return rel.sum() / out_denom

    out_denom = rel.numel() / rel.shape[-1] * max(weight_sum, 1e-12)
    return rel.sum() / out_denom


class RelativeMSELoss(nn.Module):
    r"""Module wrapper around :func:`relative_mse_loss`.

    Parameters
    ----------
    eps : float, optional
        Denominator floor passed to :func:`relative_mse_loss`. Default
        ``1e-6``.
    mode : str, optional
        ``"pointwise"`` or ``"channel_max"``. Default ``"pointwise"``.
    channel_weights : list of float, optional
        Per-channel weights. Stored as a persistent ``float32`` buffer when
        given.

    Shape
    -----
    Same as :func:`relative_mse_loss`. Output is a scalar.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        mode: str = "pointwise",
        channel_weights: list[float] | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.mode = mode
        if channel_weights is None:
            self.register_buffer("channel_weights", None, persistent=False)
        else:
            w = torch.tensor(channel_weights, dtype=torch.float32)
            self.register_buffer("channel_weights", w, persistent=True)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        point_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return relative_mse_loss(
            pred,
            target,
            eps=self.eps,
            mode=self.mode,
            mask=mask,
            point_weights=point_weights,
            channel_weights=self.channel_weights,
        )


# ---------------------------------------------------------------------------
# Hybrid: relative L2 + MSE
# ---------------------------------------------------------------------------


def relative_l2_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    relative_l2_weight: float = 1.0,
    mse_weight: float = 0.1,
    relative_l2_eps: float = 1e-12,
    mask: torch.Tensor | None = None,
    point_weights: torch.Tensor | None = None,
    channel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Weighted combination of relative-L2 and plain MSE.

    Computes ``relative_l2_weight * relative_l2_loss(...) + mse_weight *
    mse_loss(...)``. Useful as a stable reconstruction objective: the
    relative-L2 term controls scale-invariance while the MSE tail keeps
    near-zero targets honest.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Target values.
    relative_l2_weight : float, optional
        Multiplier on the relative-L2 term. Default ``1.0``.
    mse_weight : float, optional
        Multiplier on the MSE term. Default ``0.1``.
    relative_l2_eps : float, optional
        ``eps`` forwarded to :func:`relative_l2_loss`. Default ``1e-12``.
    mask : torch.Tensor, optional
        Forwarded to both sub-losses.
    point_weights : torch.Tensor, optional
        Forwarded to both sub-losses.
    channel_weights : torch.Tensor, optional
        Forwarded to both sub-losses.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    rel_term = relative_l2_loss(
        pred,
        target,
        eps=float(relative_l2_eps),
        mask=mask,
        point_weights=point_weights,
        channel_weights=channel_weights,
    )
    mse_term = mse_loss(
        pred,
        target,
        mask=mask,
        point_weights=point_weights,
        channel_weights=channel_weights,
    )
    return float(relative_l2_weight) * rel_term + float(mse_weight) * mse_term


class RelativeL2MSELoss(nn.Module):
    r"""Module wrapper around :func:`relative_l2_mse_loss`.

    Parameters
    ----------
    relative_l2_weight : float, optional
        Multiplier on the relative-L2 term. Default ``1.0``.
    mse_weight : float, optional
        Multiplier on the MSE term. Default ``0.1``.
    relative_l2_eps : float, optional
        ``eps`` forwarded to :func:`relative_l2_loss`. Default ``1e-12``.
    channel_weights : list of float, optional
        Per-channel weights. Stored as a persistent ``float32`` buffer when
        given.

    Shape
    -----
    Same as :func:`relative_l2_mse_loss`. Output is a scalar.
    """

    def __init__(
        self,
        *,
        relative_l2_weight: float = 1.0,
        mse_weight: float = 0.1,
        relative_l2_eps: float = 1e-12,
        channel_weights: list[float] | None = None,
    ):
        super().__init__()
        self.relative_l2_weight = float(relative_l2_weight)
        self.mse_weight = float(mse_weight)
        self.relative_l2_eps = float(relative_l2_eps)
        if channel_weights is None:
            self.register_buffer("channel_weights", None, persistent=False)
        else:
            w = torch.tensor(channel_weights, dtype=torch.float32)
            self.register_buffer("channel_weights", w, persistent=True)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        point_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return relative_l2_mse_loss(
            pred,
            target,
            relative_l2_weight=self.relative_l2_weight,
            mse_weight=self.mse_weight,
            relative_l2_eps=self.relative_l2_eps,
            mask=mask,
            point_weights=point_weights,
            channel_weights=self.channel_weights,
        )
