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

import random

import pytest
import torch
import torch.nn as nn

import physicsnemo.nn as layers


class SpectralConv4d(nn.Module):
    """Spectral 4D layer from https://github.com/gegewen/nested-fno/blob/main/FNO4D.py"""

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        super(SpectralConv4d, self).__init__()

        """
        4D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights5 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights6 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights7 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights8 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )

    # Complex multiplication
    def compl_mul4d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyzt,ioxyzt->boxyzt", input, weights)

    def forward(self, x):
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


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_conv_nd(device, dimension):
    """compare output of ConvNdKernel1Layer with that of layer for specfic n_dim"""

    bsize = 2
    in_channels = 4
    out_channels = 2
    tens_size = 128

    conv_nd = layers.ConvNdKernel1Layer(in_channels, out_channels).to(device)

    ini_w, ini_b = random.uniform(0, 1), random.uniform(0, 1)
    if dimension == 1:
        invar = torch.randn(bsize, in_channels, tens_size).to(device)
        comp_nn = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True).to(
            device
        )
    elif dimension == 2:
        invar = torch.randn(bsize, in_channels, tens_size, tens_size).to(device)
        comp_nn = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True).to(
            device
        )
    elif dimension == 3:
        invar = torch.randn(bsize, in_channels, tens_size, tens_size, tens_size).to(
            device
        )
        comp_nn = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True).to(
            device
        )

    nn.init.constant_(conv_nd.conv.bias, ini_b)
    nn.init.constant_(conv_nd.conv.weight, ini_w)
    nn.init.constant_(comp_nn.bias, ini_b)
    nn.init.constant_(comp_nn.weight, ini_w)
    with torch.no_grad():
        assert torch.allclose(conv_nd(invar), comp_nn(invar), rtol=1e-05, atol=1e-03), (
            f"ConvNdKernel1Layer output not identical to that of layer specific for {dimension}d fields :("
        )


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_conv_ndfc(device, dimension):
    """compare output of ConvNdFCLayer with that of layer for specfic n_dim"""
    bsize = 2
    in_channels = 4
    out_channels = 2
    tens_size = 128

    conv_nd = layers.ConvNdFCLayer(in_channels, out_channels).to(device)

    if dimension == 1:
        invar = torch.randn(bsize, in_channels, tens_size).to(device)
        comp_nn = layers.Conv1dFCLayer(in_channels, out_channels).to(device)
    elif dimension == 2:
        invar = torch.randn(bsize, in_channels, tens_size, tens_size).to(device)
        comp_nn = layers.Conv2dFCLayer(in_channels, out_channels).to(device)
    elif dimension == 3:
        invar = torch.randn(bsize, in_channels, tens_size, tens_size, tens_size).to(
            device
        )
        comp_nn = layers.Conv3dFCLayer(in_channels, out_channels).to(device)

    # initialise weights, biases
    torch.manual_seed(0)
    conv_nd.reset_parameters()
    torch.manual_seed(0)
    comp_nn.reset_parameters()
    with torch.no_grad():
        assert torch.allclose(conv_nd(invar), comp_nn(invar), rtol=1e-05, atol=1e-03), (
            f"ConvNdFCLayer output not identical to that of layer specific for {dimension}d fields :("
        )


def test_spec_conv_4d(device):
    """compare output of SpectralConv4d with that of layer used in literature."""
    bsize = 8
    in_channels = 8
    out_channels = 4
    tens_size = 16
    fno_modes = 6

    torch.manual_seed(0)
    spec_conv_orig = SpectralConv4d(
        in_channels, out_channels, fno_modes, fno_modes, fno_modes, fno_modes
    ).to(device)
    torch.manual_seed(0)
    spec_conv_physicsnemo = layers.SpectralConv4d(
        in_channels, out_channels, fno_modes, fno_modes, fno_modes, fno_modes
    ).to(device)

    invar = torch.randn(
        bsize, in_channels, tens_size, tens_size, tens_size, tens_size
    ).to(device)
    with torch.no_grad():
        assert torch.allclose(
            spec_conv_orig(invar), spec_conv_physicsnemo(invar), rtol=1e-06, atol=1e-06
        ), "SpectralConv4d output not identical to that of refrence layer"
