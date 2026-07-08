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
import torch.nn.functional as F

Tensor = torch.Tensor


def huber(
    pred: Tensor,
    target: Tensor,
    delta: float = 1.0,
    dim: Optional[Union[int, tuple]] = None,
) -> Union[Tensor, float]:
    r"""Calculates the Huber error between two tensors.

    The Huber error behaves quadratically for element-wise errors smaller than
    ``delta`` and linearly for larger errors, combining the sensitivity of the
    mean squared error near zero with the robustness of the mean absolute error
    for outliers:

    .. math::

        \mathrm{Huber}(x) = \begin{cases}
            \tfrac{1}{2} x^2 & |x| \le \delta \\
            \delta (|x| - \tfrac{1}{2}\delta) & |x| > \delta,
        \end{cases}

    where :math:`x = \mathrm{pred} - \mathrm{target}`.

    Parameters
    ----------
    pred : Tensor
        Input prediction tensor.
    target : Tensor
        Target tensor.
    delta : float, optional
        Threshold at which the error transitions from quadratic to linear,
        by default ``1.0``.
    dim : int or tuple of int, optional
        Reduction dimension(s). When ``None`` the error is averaged over all
        observations, by default ``None``.

    Returns
    -------
    Union[Tensor, float]
        Huber error value(s).
    """
    unreduced = F.huber_loss(pred, target, reduction="none", delta=delta)
    return torch.mean(unreduced, dim=dim)
