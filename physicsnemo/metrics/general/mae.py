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


def mae(
    pred: Tensor, target: Tensor, dim: Optional[Union[int, tuple]] = None
) -> Union[Tensor, float]:
    r"""Calculates the Mean Absolute Error (MAE) between two tensors.

    The mean absolute error, also known as the :math:`L_1` error, is

    .. math::

        \mathrm{MAE} = \mathrm{mean}(|\mathrm{pred} - \mathrm{target}|).

    Parameters
    ----------
    pred : Tensor
        Input prediction tensor.
    target : Tensor
        Target tensor.
    dim : int or tuple of int, optional
        Reduction dimension(s). When ``None`` the error is averaged over all
        observations, by default ``None``.

    Returns
    -------
    Union[Tensor, float]
        Mean absolute error value(s).
    """
    return torch.mean(torch.abs(pred - target), dim=dim)


# ``l1`` is a common alias for the mean absolute error.
l1 = mae
