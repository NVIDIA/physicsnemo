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

"""Relative (target-normalized) regression errors.

The relative counterparts of :mod:`physicsnemo.metrics.general.mse`, mirroring
its ``mse`` / ``rmse`` pair as a squared form and its square root:

* :func:`relative_mse` — relative mean squared error
  ``sum((pred - target)^2) / sum(target^2)`` (the primary, squared form).
* :func:`relative_l2` — relative L2 error ``||pred - target|| / ||target||``,
  i.e. ``sqrt(relative_mse)``.

Both reduce over ``dim`` (all elements when ``None``), like
:func:`physicsnemo.metrics.general.mse.mse`, and accept optional element
``weights`` (which subsumes a validity mask and per-point weights).
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def _relative_reduce(
    err_sq: Tensor,
    tgt_sq: Tensor,
    *,
    dim: int | tuple[int, ...] | None,
    weights: Tensor | None,
    eps: float,
) -> Tensor:
    """Weighted ratio of summed squared error to summed squared target.

    Parameters
    ----------
    err_sq : Tensor
        Squared error ``(pred - target) ** 2``.
    tgt_sq : Tensor
        Squared target ``target ** 2``.
    dim : int or tuple of int or None
        Dimension(s) over which the sums are taken. ``None`` reduces over all
        elements.
    weights : Tensor or None
        Optional element weights, broadcastable to ``err_sq``. Applied to both
        the numerator and denominator sums.
    eps : float
        Floor applied to the denominator before dividing.

    Returns
    -------
    Tensor
        ``sum(weights * err_sq) / max(sum(weights * tgt_sq), eps)`` reduced over
        ``dim``.
    """
    if weights is not None:
        # Broadcast explicitly (as in ``mse.py``) so a non-broadcastable weight
        # raises a clear error naming the target shape.
        w = torch.broadcast_to(
            weights.to(device=err_sq.device, dtype=err_sq.dtype), err_sq.shape
        )
        err_sq = err_sq * w
        tgt_sq = tgt_sq * w
    if dim is None:
        num = err_sq.sum()
        den = tgt_sq.sum()
    else:
        num = err_sq.sum(dim=dim)
        den = tgt_sq.sum(dim=dim)
    return num / den.clamp_min(eps)


def _check_shapes(pred: Tensor, target: Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            "pred and target must have the same shape, got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}"
        )


def relative_mse(
    pred: Tensor,
    target: Tensor,
    dim: int | tuple[int, ...] | None = None,
    weights: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    r"""Relative mean squared error, target-normalized.

    Computes :math:`\sum (pred - target)^2 / \sum target^2`, with the sums
    taken over ``dim`` (all elements when ``None``). This is the squared,
    primary form; :func:`relative_l2` is its square root.

    Parameters
    ----------
    pred : Tensor
        Predicted values.
    target : Tensor
        Target values; must have the same shape as ``pred``.
    dim : int or tuple of int or None, optional
        Dimension(s) over which the error is normalized and reduced. ``None``
        reduces over all elements, returning a scalar. Default ``None``.
    weights : Tensor, optional
        Element weights broadcastable to ``pred`` (e.g. a 0/1 validity mask or
        per-point weights). Applied inside both the numerator and denominator
        sums. Default ``None``.
    eps : float, optional
        Floor applied to the (summed) denominator to guard against all-zero
        targets. Default ``1e-8``.

    Returns
    -------
    Tensor
        Relative MSE reduced over ``dim``.

    Raises
    ------
    ValueError
        If ``pred`` and ``target`` do not have the same shape.

    Notes
    -----
    Because the result is a ratio of weighted sums, ``weights`` that are
    *constant* along a kept axis cancel between numerator and denominator and
    have no effect. In particular, to weight per-channel contributions, keep the
    channel axis out of ``dim`` (so the result is per-channel) and apply the
    channel weighting to that output yourself; a per-channel weight passed via
    ``weights`` would cancel.
    """
    _check_shapes(pred, target)
    return _relative_reduce(
        (pred - target).square(),
        target.square(),
        dim=dim,
        weights=weights,
        eps=eps,
    )


def relative_l2(
    pred: Tensor,
    target: Tensor,
    dim: int | tuple[int, ...] | None = None,
    weights: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    r"""Relative L2 error (target-normalized L2 norm).

    Computes :math:`\sqrt{\sum (pred - target)^2 / \sum target^2}`, the square
    root of :func:`relative_mse`, with the sums taken over ``dim`` (all elements
    when ``None``).

    Parameters
    ----------
    pred : Tensor
        Predicted values.
    target : Tensor
        Target values; must have the same shape as ``pred``.
    dim : int or tuple of int or None, optional
        Dimension(s) over which the norm is taken and reduced. ``None`` reduces
        over all elements, returning a scalar. Default ``None``.
    weights : Tensor, optional
        Element weights broadcastable to ``pred`` (e.g. a 0/1 validity mask or
        per-point weights). Applied inside both sums. Default ``None``. See the
        ``Notes`` on :func:`relative_mse` regarding per-channel weighting.
    eps : float, optional
        Floor applied to the (summed) denominator to guard against all-zero
        targets. Default ``1e-8``.

    Returns
    -------
    Tensor
        Relative L2 error reduced over ``dim``.

    Raises
    ------
    ValueError
        If ``pred`` and ``target`` do not have the same shape.
    """
    # Enforce the documented contract directly rather than relying on the
    # delegated relative_mse call.
    _check_shapes(pred, target)
    return relative_mse(pred, target, dim=dim, weights=weights, eps=eps).sqrt()
