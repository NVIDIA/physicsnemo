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

from typing import Optional, Union

import torch

Tensor = torch.Tensor


def relative_lp(
    pred: Tensor,
    target: Tensor,
    p: float = 2.0,
    dim: Optional[Union[int, tuple]] = None,
    eps: float = 1e-8,
) -> Tensor:
    r"""Calculates the relative :math:`L_p` error between two tensors.

    The relative :math:`L_p` error is scale-invariant and is defined as

    .. math::

        \frac{\lVert \mathrm{pred} - \mathrm{target} \rVert_p}
             {\lVert \mathrm{target} \rVert_p + \epsilon}.

    This is a common error measure in neural-operator learning where the
    output magnitude varies across samples (sometimes referred to as the
    ``LpLoss``).

    Parameters
    ----------
    pred : Tensor
        Input prediction tensor.
    target : Tensor
        Target tensor.
    p : float, optional
        Order of the norm, by default ``2.0``.
    dim : int or tuple of int, optional
        Dimension(s) over which to compute the norms. When ``None`` the norms
        are computed over the flattened tensors, by default ``None``.
    eps : float, optional
        Small value added to the denominator for numerical stability,
        by default ``1e-8``.

    Returns
    -------
    Tensor
        Relative :math:`L_p` error value(s).
    """
    # ``torch.linalg.vector_norm`` treats ``dim=None`` as "flatten then reduce",
    # so a single call handles both the global and per-dimension cases.
    diff_norm = torch.linalg.vector_norm(pred - target, ord=p, dim=dim)
    target_norm = torch.linalg.vector_norm(target, ord=p, dim=dim)
    return diff_norm / (target_norm + eps)


def relative_l2(
    pred: Tensor,
    target: Tensor,
    dim: Optional[Union[int, tuple]] = None,
    eps: float = 1e-8,
) -> Tensor:
    r"""Calculates the relative :math:`L_2` error between two tensors.

    Convenience wrapper around :func:`relative_lp` with ``p=2``:

    .. math::

        \frac{\lVert \mathrm{pred} - \mathrm{target} \rVert_2}
             {\lVert \mathrm{target} \rVert_2 + \epsilon}.

    Parameters
    ----------
    pred : Tensor
        Input prediction tensor.
    target : Tensor
        Target tensor.
    dim : int or tuple of int, optional
        Dimension(s) over which to compute the norms. When ``None`` the norms
        are computed over the flattened tensors, by default ``None``.
    eps : float, optional
        Small value added to the denominator for numerical stability,
        by default ``1e-8``.

    Returns
    -------
    Tensor
        Relative :math:`L_2` error value(s).
    """
    return relative_lp(pred, target, p=2.0, dim=dim, eps=eps)
