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

# TODO(Dallas) Introduce Ensemble RMSE and MSE routines.

from typing import Union

import torch

Tensor = torch.Tensor


def mse(
    pred: Tensor,
    target: Tensor,
    dim: int = None,
    weights: Tensor = None,
    eps: float = 1e-8,
) -> Union[Tensor, float]:
    """Calculates Mean Squared error between two tensors

    Parameters
    ----------
    pred : Tensor
        Input prediction tensor
    target : Tensor
        Target tensor
    dim : int, optional
        Reduction dimension. When None the losses are averaged or summed over all
        observations, by default None
    weights : Tensor, optional
        Element weights broadcastable to ``pred`` (e.g. a 0/1 validity mask or
        per-point weights). When given, a *weighted* mean
        ``sum(weights * err) / sum(weights)`` is taken over ``dim`` instead of a
        plain mean. When None (default), the result is identical to a plain
        ``torch.mean``.
    eps : float, optional
        Floor applied to the summed weights to guard against an all-zero
        (fully masked) reduction. Only used when ``weights`` is given, by
        default 1e-8

    Returns
    -------
    Union[Tensor, float]
        Mean squared error value(s)

    Notes
    -----
    With ``weights=None`` and ``dim=None`` this returns the same value as
    ``torch.nn.functional.mse_loss(pred, target)`` (i.e. ``reduction="mean"``).
    It differs from ``mse_loss`` in two ways: ``dim`` selects the specific
    axis/axes to reduce over (``mse_loss`` only reduces over all elements via
    ``reduction="mean"``/``"sum"``, or not at all), and ``weights`` turns the
    reduction into a *weighted* mean (subsuming a validity mask or per-element
    weighting), with ``eps`` guarding a fully-masked reduction. ``mse_loss`` has
    no weighting argument.
    """
    squared_error = (pred - target) ** 2
    if weights is None:
        return torch.mean(squared_error, dim=dim)
    w = torch.broadcast_to(
        weights.to(device=squared_error.device, dtype=squared_error.dtype),
        squared_error.shape,
    )
    if dim is None:
        return (w * squared_error).sum() / w.sum().clamp_min(eps)
    return (w * squared_error).sum(dim=dim) / w.sum(dim=dim).clamp_min(eps)


def rmse(
    pred: Tensor,
    target: Tensor,
    dim: int = None,
    weights: Tensor = None,
    eps: float = 1e-8,
) -> Union[Tensor, float]:
    """Calculates Root mean Squared error between two tensors

    Parameters
    ----------
    pred : Tensor
        Input prediction tensor
    target : Tensor
        Target tensor
    dim : int, optional
        Reduction dimension. When None the losses are averaged or summed over all
        observations, by default None
    weights : Tensor, optional
        Element weights broadcastable to ``pred``; see :func:`mse`. When None
        (default), the result is identical to the unweighted RMSE.
    eps : float, optional
        Floor applied to the summed weights; see :func:`mse`. Only used when
        ``weights`` is given, by default 1e-8

    Returns
    -------
    Union[Tensor, float]
        Root mean squared error value(s)
    """
    return torch.sqrt(mse(pred, target, dim=dim, weights=weights, eps=eps))
