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

"""EDM noise scheduler with log-uniform time sampling."""

import math

import torch
from jaxtyping import Float
from torch import Tensor

from .edm import EDMNoiseScheduler


class EDMLogUniformNoiseScheduler(EDMNoiseScheduler):
    r"""
    EDM noise scheduler with log-uniform sigma sampling for training.

    Inherits time-step generation, noise addition, and loss weighting from
    :class:`EDMNoiseScheduler`.  The only difference is the training-time
    sampling strategy: instead of drawing :math:`\ln(\sigma)` from a normal
    distribution, this scheduler draws :math:`\sigma` *uniformly in
    log-space* between ``sigma_min`` and ``sigma_max``:

    .. math::
        \ln(\sigma) \sim \mathcal{U}\!\bigl[\ln(\sigma_{\min}),\;
        \ln(\sigma_{\max})\bigr]

    This can be preferable when the useful noise range is well characterised
    and you want equal probability density across the full range in log-space.

    Parameters
    ----------
    sigma_min : float, optional
        Minimum noise level, by default 0.002.
    sigma_max : float, optional
        Maximum noise level, by default 80.
    rho : float, optional
        Exponent controlling time-step spacing. By default 7.
    sigma_data : float or Tensor, optional
        Expected standard deviation of the training data, by default 0.5.
        Accepts per-channel values; see :class:`EDMNoiseScheduler`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import (
    ...     EDMLogUniformNoiseScheduler,
    ... )
    >>>
    >>> scheduler = EDMLogUniformNoiseScheduler(sigma_min=0.002, sigma_max=80.0)
    >>> t = scheduler.sample_time(8)
    >>> t.shape
    torch.Size([8])
    >>> ((t >= 0.002).all() and (t <= 80.0).all()).item()
    True

    Per-channel ``sigma_data`` works the same as :class:`EDMNoiseScheduler`:

    >>> scheduler_ch = EDMLogUniformNoiseScheduler(
    ...     sigma_data=torch.tensor([0.3, 0.5, 0.7])
    ... )
    >>> w = scheduler_ch.loss_weight(t)
    >>> w.shape
    torch.Size([8, 3])
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float | Float[Tensor, " C"] = 0.5,
    ) -> None:
        super().__init__(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            sigma_data=sigma_data,
        )

    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""
        Sample N diffusion times uniformly in log-space:
        :math:`\ln(\sigma) \sim \mathcal{U}[\ln(\sigma_{\min}),
        \ln(\sigma_{\max})]`.

        Parameters
        ----------
        N : int
            Number of time values to sample.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Sampled diffusion times of shape :math:`(N,)`.
        """
        u = torch.rand(N, device=device, dtype=dtype)
        log_min = math.log(self.sigma_min)
        log_max = math.log(self.sigma_max)
        return (log_min + u * (log_max - log_min)).exp()
