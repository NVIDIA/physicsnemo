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

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import tltorch


class FactorizedSpectralConv1d(nn.Module):
    """1D Factorized Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply, at most floor(N/2) + 1
    rank : float
        Rank of the decomposition
    factorization : {'CP', 'TT', 'Tucker'}
        Tensor factorization to use to decompose the tensor
    fixed_rank_modes : List[int]
        A list of modes for which the initial value is not modified
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict
        Additional arguments to initialization of factorized tensors
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        rank: float,
        factorization: str,
        fixed_rank_modes: bool,
        decomposition_kwargs: dict,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = tltorch.FactorizedTensor.new(
            (in_channels, out_channels, self.modes1, 2),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.reset_parameters()

    def compl_mul1d(
        self,
        input: Tensor,
        weights: Tensor,
    ) -> Tensor:
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
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        cweights = torch.view_as_complex(weights.to_tensor().contiguous())
        return torch.einsum("bix,iox->box", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        bsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            bsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1],
            self.weights1,
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*N(0,1)"""
        self.weights1.normal_(0, self.scale)


class FactorizedSpectralConv2d(nn.Module):
    """2D Factorized Fourier layer. It does FFT, linear transform, and Inverse FFT.

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
    rank : float
        Rank of the decomposition
    factorization : {'CP', 'TT', 'Tucker'}
        Tensor factorization to use to decompose the tensor
    fixed_rank_modes : List[int]
        A list of modes for which the initial value is not modified
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict
        Additional arguments to initialization of factorized tensors
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        rank: float,
        factorization: str,
        fixed_rank_modes: bool,
        decomposition_kwargs: dict,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = tltorch.FactorizedTensor.new(
            (in_channels, out_channels, self.modes1, self.modes2, 2),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights2 = tltorch.FactorizedTensor.new(
            (in_channels, out_channels, self.modes1, self.modes2, 2),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.reset_parameters()

    def compl_mul2d(self, input: Tensor, weights: Tensor) -> Tensor:
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
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        cweights = torch.view_as_complex(weights.to_tensor().contiguous())
        return torch.einsum("bixy,ioxy->boxy", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*N(0,1)"""
        self.weights1.normal_(0, self.scale)
        self.weights2.normal_(0, self.scale)


class FactorizedSpectralConv3d(nn.Module):
    """3D Factorized Fourier layer. It does FFT, linear transform, and Inverse FFT.

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
    rank : float
        Rank of the decomposition
    factorization : {'CP', 'TT', 'Tucker'}
        Tensor factorization to use to decompose the tensor
    fixed_rank_modes : List[int]
        A list of modes for which the initial value is not modified
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict
        Additional arguments to initialization of factorized tensors
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        rank: float,
        factorization: str,
        fixed_rank_modes: bool,
        decomposition_kwargs: dict,
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
        self.weights1 = tltorch.FactorizedTensor.new(
            (in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights2 = tltorch.FactorizedTensor.new(
            (in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights3 = tltorch.FactorizedTensor.new(
            (in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights4 = tltorch.FactorizedTensor.new(
            (in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.reset_parameters()

    def compl_mul3d(self, input: Tensor, weights: Tensor) -> Tensor:
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
        # (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
        cweights = torch.view_as_complex(weights.to_tensor().contiguous())
        return torch.einsum("bixyz,ioxyz->boxyz", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.normal_(0, self.scale)
        self.weights2.normal_(0, self.scale)
        self.weights3.normal_(0, self.scale)
        self.weights4.normal_(0, self.scale)


class FactorizedSpectralConv4d(nn.Module):
    """4D Factorized Fourier layer. It does FFT, linear transform, and Inverse FFT.

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
    rank : float
        Rank of the decomposition
    factorization : {'CP', 'TT', 'Tucker'}
        Tensor factorization to use to decompose the tensor
    fixed_rank_modes : List[int]
        A list of modes for which the initial value is not modified
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict
        Additional arguments to initialization of factorized tensors
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        modes4: int,
        rank: float,
        factorization: str,
        fixed_rank_modes: bool,
        decomposition_kwargs: dict,
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
        self.weights1 = tltorch.FactorizedTensor.new(
            (
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights2 = tltorch.FactorizedTensor.new(
            (
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights3 = tltorch.FactorizedTensor.new(
            (
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights4 = tltorch.FactorizedTensor.new(
            (
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights5 = tltorch.FactorizedTensor.new(
            (
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights6 = tltorch.FactorizedTensor.new(
            (
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights7 = tltorch.FactorizedTensor.new(
            (
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights8 = tltorch.FactorizedTensor.new(
            (
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.reset_parameters()

    def compl_mul4d(
        self,
        input: Tensor,
        weights: Tensor,
    ) -> Tensor:
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
        # (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
        cweights = torch.view_as_complex(weights.to_tensor().contiguous())
        return torch.einsum("bixyzt,ioxyzt->boxyzt", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-4, -3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-4),
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # print(f'mod: size x: {x_ft.size()}, out: {out_ft.size()}')
        # print(f'mod: x_ft[weight4]: {x_ft[:, :, self.modes1 :, self.modes2 :, : -self.modes3, :self.modes4].size()} weight4: {self.weights4.size()}')

        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3, : self.modes4] = (
            self.compl_mul4d(
                x_ft[:, :, : self.modes1, : self.modes2, : self.modes3, : self.modes4],
                self.weights1,
            )
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3, : self.modes4] = (
            self.compl_mul4d(
                x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3, : self.modes4],
                self.weights2,
            )
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3, : self.modes4] = (
            self.compl_mul4d(
                x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3, : self.modes4],
                self.weights3,
            )
        )
        out_ft[:, :, : self.modes1, : self.modes2, -self.modes3 :, : self.modes4] = (
            self.compl_mul4d(
                x_ft[:, :, : self.modes1, : self.modes2, -self.modes3 :, : self.modes4],
                self.weights4,
            )
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3, : self.modes4] = (
            self.compl_mul4d(
                x_ft[
                    :, :, -self.modes1 :, -self.modes2 :, : self.modes3, : self.modes4
                ],
                self.weights5,
            )
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, -self.modes3 :, : self.modes4] = (
            self.compl_mul4d(
                x_ft[
                    :, :, -self.modes1 :, : self.modes2, -self.modes3 :, : self.modes4
                ],
                self.weights6,
            )
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, -self.modes3 :, : self.modes4] = (
            self.compl_mul4d(
                x_ft[
                    :, :, : self.modes1, -self.modes2 :, -self.modes3 :, : self.modes4
                ],
                self.weights7,
            )
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, -self.modes3 :, : self.modes4] = (
            self.compl_mul4d(
                x_ft[
                    :, :, -self.modes1 :, -self.modes2 :, -self.modes3 :, : self.modes4
                ],
                self.weights8,
            )
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-4), x.size(-3), x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*N(0,1)"""
        self.weights1.normal_(0, self.scale)
        self.weights2.normal_(0, self.scale)
        self.weights3.normal_(0, self.scale)
        self.weights4.normal_(0, self.scale)
        self.weights5.normal_(0, self.scale)
        self.weights6.normal_(0, self.scale)
        self.weights7.normal_(0, self.scale)
        self.weights8.normal_(0, self.scale)
