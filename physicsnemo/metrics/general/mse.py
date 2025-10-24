# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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


def mse(pred: Tensor, target: Tensor, dim: int = None) -> Union[Tensor, float]:
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

    Returns
    -------
    Union[Tensor, float]
        Mean squared error value(s)
    """
    return torch.mean((pred - target) ** 2, dim=dim)


def rmse(pred: Tensor, target: Tensor, dim: int = None) -> Union[Tensor, float]:
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

    Returns
    -------
    Union[Tensor, float]
        Root mean squared error value(s)
    """
    return torch.sqrt(mse(pred, target, dim=dim))
