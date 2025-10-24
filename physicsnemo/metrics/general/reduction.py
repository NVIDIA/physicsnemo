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

from abc import ABC

import torch
from torch import Tensor


class WeightedStatistic(ABC):
    """A convenience class for computing weighted statistics of some input

    Parameters
    ----------
    weights : Tensor
        Weight tensor
    """

    def __init__(self, weights: Tensor):
        if not torch.all(weights > 0.0).item():
            raise ValueError("Expected all weights to be positive.")
        self.weights = self._normalize(weights)

    def __call__(self, x: Tensor, dim: int):
        """
        Convenience method to make sure weights have appropriate shapes.
        """
        w = self.weights
        if w.ndim == 1:
            if x.shape[dim] != len(w):
                raise ValueError(
                    "Expected inputs and weights to have the same size along the reduction dimension but have dimensions"
                    + str(len(x[dim]))
                    + " and "
                    + str(len(w))
                    + "."
                )
            if dim < 0:
                dim = x.ndim + dim
            for i in range(x.ndim):
                if i < dim:
                    w = w.unsqueeze(0)
                elif i > dim:
                    w = w.unsqueeze(-1)
        else:
            if not ((x.ndim == w.ndim) and (x.shape[dim] == w.shape[dim])):
                raise ValueError(
                    "Expected inputs and weights to have compatible shapes."
                )
        return w

    def _normalize(self, weights: Tensor) -> Tensor:
        """Normalize unnormalized weights, for convenience

        Parameters
        ----------
        weights : Tensor
            Unnormalized weights

        Returns
        -------
        Tensor
            Normalized weights
        """
        return weights / torch.sum(weights)


class WeightedMean(WeightedStatistic):
    """
    Compute weighted mean of some input.

    Parameters
    ----------
    weights : Tensor
        Weight tensor
    """

    def __init__(self, weights: Tensor):
        super().__init__(weights)

    def __call__(self, x: Tensor, dim: int, keepdims: bool = False) -> Tensor:
        """Compute weighted mean

        Parameters
        ----------
        x : Tensor
            Input data
        dim : int
            Dimension to take aggregate
        keepdims : bool, optional
            Keep aggregated dimension, by default False

        Returns
        -------
        Tensor
            Weighted mean
        """
        w = super().__call__(x, dim)
        return torch.sum(x * w, dim=dim, keepdims=keepdims)


class WeightedVariance(WeightedStatistic):
    """
    Compute weighted variance of some input.

    Parameters
    ----------
    weights : Tensor
        Weight tensor
    """

    def __init__(self, weights: Tensor):
        super().__init__(weights)
        self.wm = WeightedMean(self.weights)

    def __call__(self, x: Tensor, dim: int, keepdims: bool = False):
        """Compute weighted variance

        Parameters
        ----------
        x : Tensor
            Input data
        dim : int
            Dimension to take aggregate
        keepdims : bool, optional
            Keep aggregated dimension, by default False

        Returns
        -------
        Tensor
            Weighted variance
        """
        w = super().__call__(x, dim)

        # Compute weighted mean
        wm = self.wm(x, dim, keepdims=True)

        # Computing scaling for standard deviation
        number_of_non_zero_weights = torch.sum(w > 0.0)
        scale = (number_of_non_zero_weights - 1.0) / number_of_non_zero_weights
        return torch.sum(w * (x - wm) ** 2, dim=dim, keepdims=keepdims) / scale
