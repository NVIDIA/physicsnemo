# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex, Float
from torch import Tensor


class SpectralConv1d(nn.Module):
    """1D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply, at most floor(N/2) + 1
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.empty(in_channels, out_channels, self.modes1, 2)
        )
        self.reset_parameters()

    def compl_mul1d(
        self,
        input: Complex[Tensor, "batch in_channels modes"],
        weights: Float[Tensor, "in_channels out_channels modes 2"],
    ) -> Complex[Tensor, "batch out_channels modes"]:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channels, modes), (in_channels, out_channels, modes) -> (batch, out_channels, modes)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bix,iox->box", input, cweights)

    def forward(
        self, x: Float[Tensor, "batch in_channels x"]
    ) -> Float[Tensor, "batch out_channels x"]:
        bsize = x.shape[0]
        x_ft = torch.fft.rfft(x)  # (batch, in_channels, x//2+1) complex

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            bsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )  # (batch, out_channels, x//2+1) complex
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1],
            self.weights1,
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))  # (batch, out_channels, x) real
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)


class SpectralConv2d(nn.Module):
    """2D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.empty(in_channels, out_channels, self.modes1, self.modes2, 2)
        )
        self.weights2 = nn.Parameter(
            torch.empty(in_channels, out_channels, self.modes1, self.modes2, 2)
        )
        self.reset_parameters()

    def compl_mul2d(
        self,
        input: Complex[Tensor, "batch in_channels modes1 modes2"],
        weights: Float[Tensor, "in_channels out_channels modes1 modes2 2"],
    ) -> Complex[Tensor, "batch out_channels modes1 modes2"]:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channels, modes1, modes2), (in_channels, out_channels, modes1, modes2)
        #   -> (batch, out_channels, modes1, modes2)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bixy,ioxy->boxy", input, cweights)

    def forward(
        self, x: Float[Tensor, "batch in_channels h w"]
    ) -> Float[Tensor, "batch out_channels h w"]:
        x_ft = torch.fft.rfft2(x)  # (batch, in_channels, h, w//2+1) complex
        h, w = x_ft.size(-2), x_ft.size(-1)  # h=h, w=w//2+1

        # Initialize output in frequency space
        out_ft = torch.zeros(
            x.size(0), self.out_channels, h, w, dtype=torch.cfloat, device=x.device
        )  # (batch, out_channels, h, w) complex

        # Accumulate Fourier modes. Use .contiguous() on sliced complex tensors and
        # padding (not slice assignment) for torch.compile compatibility.
        # Slice assignment causes gradient stride issues in the Inductor backward pass.
        # Pad format: (left, right, top, bottom) for last 2 dims
        out_ft = out_ft + F.pad(
            self.compl_mul2d(
                x_ft[:, :, : self.modes1, : self.modes2].contiguous(), self.weights1
            ),
            (0, w - self.modes2, 0, h - self.modes1),
        )
        out_ft = out_ft + F.pad(
            self.compl_mul2d(
                x_ft[:, :, -self.modes1 :, : self.modes2].contiguous(), self.weights2
            ),
            (0, w - self.modes2, h - self.modes1, 0),
        )

        # Return to physical space
        return torch.fft.irfft2(
            out_ft, s=(x.size(-2), x.size(-1))
        )  # (batch, out_channels, h, w) real

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * torch.rand(self.weights2.data.shape)


class SpectralConv3d(nn.Module):
    """3D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    modes3 : int
        Number of Fourier modes to multiply in third dimension, at most floor(N/2) + 1
    """

    def __init__(
        self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.weights2 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.weights3 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.weights4 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.reset_parameters()

    def compl_mul3d(
        self,
        input: Complex[Tensor, "batch in_channels modes1 modes2 modes3"],
        weights: Float[Tensor, "in_channels out_channels modes1 modes2 modes3 2"],
    ) -> Complex[Tensor, "batch out_channels modes1 modes2 modes3"]:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channels, modes1, modes2, modes3),
        #   (in_channels, out_channels, modes1, modes2, modes3)
        #   -> (batch, out_channels, modes1, modes2, modes3)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bixyz,ioxyz->boxyz", input, cweights)

    def forward(
        self, x: Float[Tensor, "batch in_channels d1 d2 d3"]
    ) -> Float[Tensor, "batch out_channels d1 d2 d3"]:
        x_ft = torch.fft.rfftn(
            x, dim=[-3, -2, -1]
        )  # (batch, in_channels, d1, d2, d3//2+1) complex
        d1, d2, d3 = x_ft.size(-3), x_ft.size(-2), x_ft.size(-1)  # d3 = d3//2+1

        # Initialize output in frequency space
        out_ft = torch.zeros(
            x.size(0),
            self.out_channels,
            d1,
            d2,
            d3,
            dtype=torch.cfloat,
            device=x.device,
        )  # (batch, out_channels, d1, d2, d3) complex

        # Accumulate Fourier modes. Use .contiguous() on sliced complex tensors and
        # padding (not slice assignment) for torch.compile compatibility.
        # Slice assignment causes gradient stride issues in the Inductor backward pass.
        # Pad format for 3D: (d3_left, d3_right, d2_top, d2_bottom, d1_front, d1_back)
        pad_d3 = d3 - self.modes3
        pad_d2 = d2 - self.modes2
        pad_d1 = d1 - self.modes1
        out_ft = out_ft + F.pad(
            self.compl_mul3d(
                x_ft[:, :, : self.modes1, : self.modes2, : self.modes3].contiguous(),
                self.weights1,
            ),
            (0, pad_d3, 0, pad_d2, 0, pad_d1),
        )
        out_ft = out_ft + F.pad(
            self.compl_mul3d(
                x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3].contiguous(),
                self.weights2,
            ),
            (0, pad_d3, 0, pad_d2, pad_d1, 0),
        )
        out_ft = out_ft + F.pad(
            self.compl_mul3d(
                x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3].contiguous(),
                self.weights3,
            ),
            (0, pad_d3, pad_d2, 0, 0, pad_d1),
        )
        out_ft = out_ft + F.pad(
            self.compl_mul3d(
                x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3].contiguous(),
                self.weights4,
            ),
            (0, pad_d3, pad_d2, 0, pad_d1, 0),
        )

        # Return to physical space
        return torch.fft.irfftn(
            out_ft, s=(x.size(-3), x.size(-2), x.size(-1))
        )  # (batch, out_channels, d1, d2, d3) real

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * torch.rand(self.weights2.data.shape)
        self.weights3.data = self.scale * torch.rand(self.weights3.data.shape)
        self.weights4.data = self.scale * torch.rand(self.weights4.data.shape)


class SpectralConv4d(nn.Module):
    """4D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    modes3 : int
        Number of Fourier modes to multiply in third dimension, at most floor(N/2) + 1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        modes4: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.weights2 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.weights3 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.weights4 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.weights5 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.weights6 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.weights7 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.weights8 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.reset_parameters()

    def compl_mul4d(
        self,
        input: Complex[Tensor, "batch in_channels modes1 modes2 modes3 modes4"],
        weights: Float[
            Tensor, "in_channels out_channels modes1 modes2 modes3 modes4 2"
        ],
    ) -> Complex[Tensor, "batch out_channels modes1 modes2 modes3 modes4"]:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channels, modes1, modes2, modes3, modes4),
        #   (in_channels, out_channels, modes1, modes2, modes3, modes4)
        #   -> (batch, out_channels, modes1, modes2, modes3, modes4)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bixyzt,ioxyzt->boxyzt", input, cweights)

    def forward(
        self, x: Float[Tensor, "batch in_channels d1 d2 d3 d4"]
    ) -> Float[Tensor, "batch out_channels d1 d2 d3 d4"]:
        x_ft = torch.fft.rfftn(
            x, dim=[-4, -3, -2, -1]
        )  # (batch, in_channels, d1, d2, d3, d4//2+1) complex
        d1, d2, d3, d4 = (
            x_ft.size(-4),
            x_ft.size(-3),
            x_ft.size(-2),
            x_ft.size(-1),
        )  # d4 = d4//2+1

        # Initialize output in frequency space
        out_ft = torch.zeros(
            x.size(0),
            self.out_channels,
            d1,
            d2,
            d3,
            d4,
            dtype=torch.cfloat,
            device=x.device,
        )  # (batch, out_channels, d1, d2, d3, d4) complex

        # Accumulate Fourier modes. Use .contiguous() on sliced complex tensors and
        # padding (not slice assignment) for torch.compile compatibility.
        # Slice assignment causes gradient stride issues in the Inductor backward pass.
        # Pad format for 4D: (d4_left, d4_right, d3_front, d3_back, d2_top, d2_bottom, d1_near, d1_far)
        pad_d4 = d4 - self.modes4
        pad_d3 = d3 - self.modes3
        pad_d2 = d2 - self.modes2
        pad_d1 = d1 - self.modes1
        # [:modes1, :modes2, :modes3, :modes4]
        out_ft = out_ft + F.pad(
            self.compl_mul4d(
                x_ft[
                    :, :, : self.modes1, : self.modes2, : self.modes3, : self.modes4
                ].contiguous(),
                self.weights1,
            ),
            (0, pad_d4, 0, pad_d3, 0, pad_d2, 0, pad_d1),
        )
        # [-modes1:, :modes2, :modes3, :modes4]
        out_ft = out_ft + F.pad(
            self.compl_mul4d(
                x_ft[
                    :, :, -self.modes1 :, : self.modes2, : self.modes3, : self.modes4
                ].contiguous(),
                self.weights2,
            ),
            (0, pad_d4, 0, pad_d3, 0, pad_d2, pad_d1, 0),
        )
        # [:modes1, -modes2:, :modes3, :modes4]
        out_ft = out_ft + F.pad(
            self.compl_mul4d(
                x_ft[
                    :, :, : self.modes1, -self.modes2 :, : self.modes3, : self.modes4
                ].contiguous(),
                self.weights3,
            ),
            (0, pad_d4, 0, pad_d3, pad_d2, 0, 0, pad_d1),
        )
        # [:modes1, :modes2, -modes3:, :modes4]
        out_ft = out_ft + F.pad(
            self.compl_mul4d(
                x_ft[
                    :, :, : self.modes1, : self.modes2, -self.modes3 :, : self.modes4
                ].contiguous(),
                self.weights4,
            ),
            (0, pad_d4, pad_d3, 0, 0, pad_d2, 0, pad_d1),
        )
        # [-modes1:, -modes2:, :modes3, :modes4]
        out_ft = out_ft + F.pad(
            self.compl_mul4d(
                x_ft[
                    :, :, -self.modes1 :, -self.modes2 :, : self.modes3, : self.modes4
                ].contiguous(),
                self.weights5,
            ),
            (0, pad_d4, 0, pad_d3, pad_d2, 0, pad_d1, 0),
        )
        # [-modes1:, :modes2, -modes3:, :modes4]
        out_ft = out_ft + F.pad(
            self.compl_mul4d(
                x_ft[
                    :, :, -self.modes1 :, : self.modes2, -self.modes3 :, : self.modes4
                ].contiguous(),
                self.weights6,
            ),
            (0, pad_d4, pad_d3, 0, 0, pad_d2, pad_d1, 0),
        )
        # [:modes1, -modes2:, -modes3:, :modes4]
        out_ft = out_ft + F.pad(
            self.compl_mul4d(
                x_ft[
                    :, :, : self.modes1, -self.modes2 :, -self.modes3 :, : self.modes4
                ].contiguous(),
                self.weights7,
            ),
            (0, pad_d4, pad_d3, 0, pad_d2, 0, 0, pad_d1),
        )
        # [-modes1:, -modes2:, -modes3:, :modes4]
        out_ft = out_ft + F.pad(
            self.compl_mul4d(
                x_ft[
                    :, :, -self.modes1 :, -self.modes2 :, -self.modes3 :, : self.modes4
                ].contiguous(),
                self.weights8,
            ),
            (0, pad_d4, pad_d3, 0, pad_d2, 0, pad_d1, 0),
        )

        # Return to physical space
        return torch.fft.irfftn(
            out_ft, s=(x.size(-4), x.size(-3), x.size(-2), x.size(-1))
        )  # (batch, out_channels, d1, d2, d3, d4) real

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * torch.rand(self.weights2.data.shape)
        self.weights3.data = self.scale * torch.rand(self.weights3.data.shape)
        self.weights4.data = self.scale * torch.rand(self.weights4.data.shape)
        self.weights5.data = self.scale * torch.rand(self.weights5.data.shape)
        self.weights6.data = self.scale * torch.rand(self.weights6.data.shape)
        self.weights7.data = self.scale * torch.rand(self.weights7.data.shape)
        self.weights8.data = self.scale * torch.rand(self.weights8.data.shape)


# ==========================================
# Utils for PINO exact gradients
# ==========================================


def fourier_derivatives(x: Tensor, ell: List[float]) -> Tuple[Tensor, Tensor]:
    """
    Fourier derivative function for PINO
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
    wx_h = [j * k_x_i * x_h * (2 * pi / ell[i]) for i, k_x_i in enumerate(k_x)]
    wxx_h = [
        j * k_x_i * wx_h_i * (2 * pi / ell[i])
        for i, (wx_h_i, k_x_i) in enumerate(zip(wx_h, k_x))
    ]

    # inverse fourier transform out
    wx = torch.cat(
        [torch.fft.ifftn(wx_h_i, dim=list(range(2, dim + 2))).real for wx_h_i in wx_h],
        dim=1,
    )
    wxx = torch.cat(
        [
            torch.fft.ifftn(wxx_h_i, dim=list(range(2, dim + 2))).real
            for wxx_h_i in wxx_h
        ],
        dim=1,
    )
    return (wx, wxx)


def calc_latent_derivatives(
    x: Tensor, domain_length: List[int] = 2
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Compute first and second order derivatives of latent variables
    """

    dim = len(x.shape) - 2
    # Compute derivatives of latent variables via fourier methods
    # Padd domain by factor of 2 for non-periodic domains
    padd = [(i - 1) // 2 for i in list(x.shape[2:])]
    # Scale domain length by padding amount
    domain_length = [
        domain_length[i] * (2 * padd[i] + x.shape[i + 2]) / x.shape[i + 2]
        for i in range(dim)
    ]
    padding = padd + padd
    x_p = F.pad(x, padding, mode="replicate")
    dx, ddx = fourier_derivatives(x_p, domain_length)
    # Trim padded domain
    if len(x.shape) == 3:
        dx = dx[..., padd[0] : -padd[0]]
        ddx = ddx[..., padd[0] : -padd[0]]
        dx_list = torch.split(dx, x.shape[1], dim=1)
        ddx_list = torch.split(ddx, x.shape[1], dim=1)
    elif len(x.shape) == 4:
        dx = dx[..., padd[0] : -padd[0], padd[1] : -padd[1]]
        ddx = ddx[..., padd[0] : -padd[0], padd[1] : -padd[1]]
        dx_list = torch.split(dx, x.shape[1], dim=1)
        ddx_list = torch.split(ddx, x.shape[1], dim=1)
    else:
        dx = dx[..., padd[0] : -padd[0], padd[1] : -padd[1], padd[2] : -padd[2]]
        ddx = ddx[..., padd[0] : -padd[0], padd[1] : -padd[1], padd[2] : -padd[2]]
        dx_list = torch.split(dx, x.shape[1], dim=1)
        ddx_list = torch.split(ddx, x.shape[1], dim=1)

    return dx_list, ddx_list


def first_order_pino_grads(
    u: Tensor,
    ux: List[Tensor],
    weights_1: Tensor,
    weights_2: Tensor,
    bias_1: Tensor,
) -> Tuple[Tensor]:  # pragma: no cover
    """
    Compute first order derivatives of output variables
    """

    # dim for einsum
    dim = len(u.shape) - 2
    dim_str = "xyz"[:dim]

    # compute first order derivatives of input
    # compute first layer
    if dim == 1:
        u_hidden = F.conv1d(u, weights_1, bias_1)
    elif dim == 2:
        weights_1 = weights_1.unsqueeze(-1)
        weights_2 = weights_2.unsqueeze(-1)
        u_hidden = F.conv2d(u, weights_1, bias_1)
    elif dim == 3:
        weights_1 = weights_1.unsqueeze(-1).unsqueeze(-1)
        weights_2 = weights_2.unsqueeze(-1).unsqueeze(-1)
        u_hidden = F.conv3d(u, weights_1, bias_1)

    # compute derivative hidden layer
    diff_tanh = 1 / torch.cosh(u_hidden) ** 2

    # compute diff(f(g))
    diff_fg = torch.einsum(
        "mi" + dim_str + ",bm" + dim_str + ",km" + dim_str + "->bi" + dim_str,
        weights_1,
        diff_tanh,
        weights_2,
    )

    # compute diff(f(g)) * diff(g)
    vx = [
        torch.einsum("bi" + dim_str + ",bi" + dim_str + "->b" + dim_str, diff_fg, w)
        for w in ux
    ]
    vx = [torch.unsqueeze(w, dim=1) for w in vx]

    return vx


def second_order_pino_grads(
    u: Tensor,
    ux: Tensor,
    uxx: Tensor,
    weights_1: Tensor,
    weights_2: Tensor,
    bias_1: Tensor,
) -> Tuple[Tensor]:  # pragma: no cover
    """
    Compute second order derivatives of output variables
    """

    # dim for einsum
    dim = len(u.shape) - 2
    dim_str = "xyz"[:dim]

    # compute first order derivatives of input
    # compute first layer
    if dim == 1:
        u_hidden = F.conv1d(u, weights_1, bias_1)
    elif dim == 2:
        weights_1 = weights_1.unsqueeze(-1)
        weights_2 = weights_2.unsqueeze(-1)
        u_hidden = F.conv2d(u, weights_1, bias_1)
    elif dim == 3:
        weights_1 = weights_1.unsqueeze(-1).unsqueeze(-1)
        weights_2 = weights_2.unsqueeze(-1).unsqueeze(-1)
        u_hidden = F.conv3d(u, weights_1, bias_1)

    # compute derivative hidden layer
    diff_tanh = 1 / torch.cosh(u_hidden) ** 2

    # compute diff(f(g))
    diff_fg = torch.einsum(
        "mi" + dim_str + ",bm" + dim_str + ",km" + dim_str + "->bi" + dim_str,
        weights_1,
        diff_tanh,
        weights_2,
    )

    # compute diagonal of hessian
    # double derivative of hidden layer
    diff_diff_tanh = -2 * diff_tanh * torch.tanh(u_hidden)

    # compute diff(g) * hessian(f) * diff(g)
    vxx1 = [
        torch.einsum(
            "bi"
            + dim_str
            + ",mi"
            + dim_str
            + ",bm"
            + dim_str
            + ",mj"
            + dim_str
            + ",bj"
            + dim_str
            + "->b"
            + dim_str,
            w,
            weights_1,
            weights_2 * diff_diff_tanh,
            weights_1,
            w,
        )
        for w in ux
    ]  # (b,x,y,t)

    # compute diff(f) * hessian(g)
    vxx2 = [
        torch.einsum("bi" + dim_str + ",bi" + dim_str + "->b" + dim_str, diff_fg, w)
        for w in uxx
    ]
    vxx = [torch.unsqueeze(a + b, dim=1) for a, b in zip(vxx1, vxx2)]

    return vxx
