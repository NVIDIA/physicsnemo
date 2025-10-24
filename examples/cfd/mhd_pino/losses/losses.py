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

import numpy as np
import torch
import torch.nn.functional as F
import math
from torch import Tensor
from typing import List


class LpLoss(object):
    """
    loss function with rel/abs Lp loss
    """

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def fourier_derivatives_lap(x: Tensor, ell: List[float]) -> Tensor:
    """
    Fourier derivative laplacian function
    """

    # check that input shape maches domain length
    if len(x.shape) - 2 != len(ell):
        raise ValueError("input shape doesn't match domain dims")

    # set pi from numpy
    pi = float(np.pi)

    # get needed dims
    n = x.shape[2:]
    dim = len(ell)

    # get device
    device = x.device

    # compute fourier transform
    x_h = torch.fft.fftn(x, dim=list(range(2, dim + 2)))

    # make wavenumbers
    k_x = []
    for i, nx in enumerate(n):
        k_x.append(
            (2 * pi / ell[i])
            * torch.cat(
                (
                    torch.arange(start=0, end=nx // 2, step=1, device=device),
                    torch.arange(start=-nx // 2, end=0, step=1, device=device),
                ),
                0,
            ).reshape((i + 2) * [1] + [nx] + (dim - i - 1) * [1])
        )
    lap = torch.zeros_like(k_x[0])
    for i in k_x:
        lap = lap - i**2

    # compute laplacian in fourier space
    wx_h = lap * x_h

    # inverse fourier transform out
    wx = torch.fft.ifftn(wx_h, dim=list(range(2, dim + 2))).real
    return wx


def fourier_derivatives_ptot(
    p: Tensor,
    div_vel_grad_vel: Tensor,
    div_B_grad_B: Tensor,
    B2_h: Tensor,
    rho0: float,
    ell: List[float],
) -> List[Tensor]:
    """
    Fourier derivative function to calculate ptot in MHD equations
    """

    # check that input shape maches domain length
    if len(div_vel_grad_vel.shape) - 2 != len(ell):
        raise ValueError("input shape doesn't match domain dims")

    # set pi from numpy
    pi = float(np.pi)

    # get needed dims
    n = div_vel_grad_vel.shape[2:]
    dim = len(ell)

    # get device
    device = div_vel_grad_vel.device

    # make wavenumbers
    k_x = []
    for i, nx in enumerate(n):
        k_x.append(
            torch.cat(
                (
                    torch.arange(start=0, end=nx // 2, step=1, device=device),
                    torch.arange(start=-nx // 2, end=0, step=1, device=device),
                ),
                0,
            ).reshape((i + 2) * [1] + [nx] + (dim - i - 1) * [1])
        )
    # note that for pressure, the zero mode (the mean) cannot be zero for invertability so it is set to 1
    lap = torch.zeros_like(k_x[0])
    for i, k_x_i in enumerate(k_x):
        lap = lap - ((2 * pi / ell[i]) * k_x_i) ** 2
    lap[..., 0, 0] = -1.0

    if p is None:
        # compute fourier transform
        div_vel_grad_vel_h = torch.fft.fftn(
            div_vel_grad_vel, dim=list(range(2, dim + 2))
        )
        div_B_grad_B_h = torch.fft.fftn(div_B_grad_B, dim=list(range(2, dim + 2)))
        ptot_h = (div_B_grad_B_h - rho0 * div_vel_grad_vel_h) / lap
        ptot_h[..., 0, 0] = B2_h[..., 0, 0] / 2.0
    else:
        p_h = torch.fft.fftn(p, dim=list(range(2, dim + 2)))
        ptot_h = p_h + B2_h / 2.0

    # compute laplacian in fourier space
    j = torch.complex(
        torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)
    )  # Cuda graphs does not work here
    wx_h = [j * k_x_i * ptot_h * (2 * pi / ell[i]) for i, k_x_i in enumerate(k_x)]

    # inverse fourier transform out
    wx = torch.cat(
        [torch.fft.ifftn(wx_h_i, dim=list(range(2, dim + 2))).real for wx_h_i in wx_h],
        dim=1,
    )
    return wx


def fourier_derivatives_vec_pot(x: Tensor, ell: List[float]) -> List[Tensor]:
    """
    Fourier derivative function for vector potential
    """

    # check that input shape maches domain length
    if len(x.shape) - 2 != len(ell):
        raise ValueError("input shape doesn't match domain dims")

    # set pi from numpy
    pi = float(np.pi)

    # get needed dims
    n = x.shape[2:]
    dim = len(ell)

    # get device
    device = x.device

    # compute fourier transform
    x_h = torch.fft.fftn(x, dim=list(range(2, dim + 2)))

    # make wavenumbers
    k_x = []
    for i, nx in enumerate(n):
        k_x.append(
            torch.cat(
                (
                    torch.arange(start=0, end=nx // 2, step=1, device=device),
                    torch.arange(start=-nx // 2, end=0, step=1, device=device),
                ),
                0,
            ).reshape((i + 2) * [1] + [nx] + (dim - i - 1) * [1])
        )

    # compute laplacian in fourier space
    j = torch.complex(
        torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)
    )  # Cuda graphs does not work here
    Ax_h = j * k_x[0] * x_h * (2 * pi / ell[0])
    Ay_h = j * k_x[1] * x_h * (2 * pi / ell[1])

    B2_h = (Ay_h) ** 2 + (-Ax_h) ** 2

    Bx_h = [j * k_x_i * Ay_h * (2 * pi / ell[i]) for i, k_x_i in enumerate(k_x)]
    By_h = [j * k_x_i * -Ax_h * (2 * pi / ell[i]) for i, k_x_i in enumerate(k_x)]

    # inverse fourier transform out
    wA = torch.cat(
        [
            torch.fft.ifftn(w_h_i, dim=list(range(2, dim + 2))).real
            for w_h_i in [Ax_h, Ay_h]
        ],
        dim=1,
    )
    wB = torch.cat(
        [
            torch.fft.ifftn(w_h_i, dim=list(range(2, dim + 2))).real
            for w_h_i in [Ay_h, -Ax_h]
        ],
        dim=1,
    )
    wx = torch.cat(
        [torch.fft.ifftn(wx_h_i, dim=list(range(2, dim + 2))).real for wx_h_i in Bx_h],
        dim=1,
    )
    wy = torch.cat(
        [torch.fft.ifftn(wx_h_i, dim=list(range(2, dim + 2))).real for wx_h_i in By_h],
        dim=1,
    )

    return wx, wy, wA, wB, B2_h
