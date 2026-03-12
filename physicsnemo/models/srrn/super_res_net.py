# ignore_header_test
# ruff: noqa: E402

""""""

"""
SRResNet model. This code was modified from,
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution

The following license is provided from their source,

MIT License

Copyright (c) 2020 Sagar Vinodababu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import nn

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn import get_activation

Tensor = torch.Tensor


@dataclass
class MetaData(ModelMetaData):
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False  # TODO: Investigate this
    amp_cpu: bool = False
    amp_gpu: bool = False
    # Inference
    onnx: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = True
    auto_grad: bool = True


class SRResNet(Module):
    r"""3D convolutional super-resolution network.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    large_kernel_size : int, optional, default=7
        Convolutional kernel size for first and last convolution.
    small_kernel_size : int, optional, default=3
        Convolutional kernel size for internal convolutions.
    conv_layer_size : int, optional, default=32
        Latent channel size.
    n_resid_blocks : int, optional, default=8
        Number of residual blocks.
    scaling_factor : int, optional, default=8
        Scaling factor to increase the output feature size
        compared to the input. Must be ``2``, ``4``, or ``8``.
    activation_fn : str, optional, default="prelu"
        Activation function.

    Forward
    -------
    in_vars : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, D, H, W)` where :math:`B` is
        batch size, :math:`C_{in}` is the number of input channels, and
        :math:`D, H, W` are the spatial dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, D \times s, H \times s, W \times s)`
        where :math:`s` is the ``scaling_factor``.

    Examples
    --------
    >>> import torch
    >>> model = physicsnemo.models.srrn.SRResNet(
    ...     in_channels=1,
    ...     out_channels=2,
    ...     conv_layer_size=4,
    ...     scaling_factor=2,
    ... )
    >>> input = torch.randn(4, 1, 8, 8, 8)  # (B, C, D, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([4, 2, 16, 16, 16])

    Notes
    -----
    Based on the implementation:
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        large_kernel_size: int = 7,
        small_kernel_size: int = 3,
        conv_layer_size: int = 32,
        n_resid_blocks: int = 8,
        scaling_factor: int = 8,
        activation_fn: str = "prelu",
    ):
        super().__init__(meta=MetaData())

        # Store constructor arguments for introspection
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scaling_factor = int(scaling_factor)
        self.var_dim = 1

        # Activation function
        if isinstance(activation_fn, str):
            activation_fn = get_activation(activation_fn)

        # Scaling factor must be 2, 4, or 8
        if self.scaling_factor not in {2, 4, 8}:
            raise ValueError(
                f"The scaling factor must be 2, 4, or 8, got {self.scaling_factor}"
            )

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock3d(
            in_channels=in_channels,
            out_channels=conv_layer_size,
            kernel_size=large_kernel_size,
            batch_norm=False,
            activation_fn=activation_fn,
        )

        # A sequence of n_resid_blocks residual blocks,
        # each containing a skip-connection across the block
        self.residual_blocks = nn.Sequential(
            *[
                ResidualConvBlock3d(
                    n_layers=2,
                    kernel_size=small_kernel_size,
                    conv_layer_size=conv_layer_size,
                    activation_fn=activation_fn,
                )
                for i in range(n_resid_blocks)
            ]
        )

        # Another convolutional block
        self.conv_block2 = ConvolutionalBlock3d(
            in_channels=conv_layer_size,
            out_channels=conv_layer_size,
            kernel_size=small_kernel_size,
            batch_norm=True,
        )

        # Upscaling is done by sub-pixel convolution,
        # with each such block upscaling by a factor of 2
        n_subpixel_convolution_blocks = int(math.log2(self.scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[
                SubPixel_ConvolutionalBlock3d(
                    kernel_size=small_kernel_size,
                    conv_layer_size=conv_layer_size,
                    scaling_factor=2,
                )
                for i in range(n_subpixel_convolution_blocks)
            ]
        )

        # The last convolutional block
        self.conv_block3 = ConvolutionalBlock3d(
            in_channels=conv_layer_size,
            out_channels=out_channels,
            kernel_size=large_kernel_size,
            batch_norm=False,
        )

    def forward(
        self, in_vars: Float[Tensor, "batch in_channels depth height width"]
    ) -> Float[Tensor, "batch out_channels depth_out height_out width_out"]:
        r"""Forward pass of SRResNet."""
        # Input validation
        if not torch.compiler.is_compiling():
            if in_vars.ndim != 5:
                raise ValueError(
                    f"Expected 5D input tensor (B, C, D, H, W), "
                    f"got {in_vars.ndim}D tensor with shape {tuple(in_vars.shape)}"
                )
            if in_vars.shape[1] != self.in_channels:
                raise ValueError(
                    f"Expected {self.in_channels} input channels, "
                    f"got {in_vars.shape[1]} channels"
                )

        output = self.conv_block1(in_vars)  # (B, conv_layer_size, D, H, W)
        residual = output  # (B, conv_layer_size, D, H, W)
        output = self.residual_blocks(output)  # (B, conv_layer_size, D, H, W)
        output = self.conv_block2(output)  # (B, conv_layer_size, D, H, W)
        output = output + residual  # (B, conv_layer_size, D, H, W)
        output = self.subpixel_convolutional_blocks(
            output
        )  # (B, conv_layer_size, D*s, H*s, W*s)
        output = self.conv_block3(output)  # (B, out_channels, D*s, H*s, W*s)

        return output


class ConvolutionalBlock3d(nn.Module):
    r"""3D convolutional block with optional batch normalization and activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolutional kernel size.
    stride : int, optional, default=1
        Convolutional stride.
    batch_norm : bool, optional, default=False
        Whether to use batch normalization.
    activation_fn : nn.Module, optional, default=nn.Identity()
        Activation function.

    Forward
    -------
    input : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, D, H, W)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, D', H', W')`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        batch_norm: bool = False,  # TODO set the train/eval model context
        activation_fn: nn.Module = nn.Identity(),
    ):
        super().__init__()

        # A container that will hold the layers in this convolutional block
        layers = list()

        # A convolutional layer
        layers.append(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        )

        # A batch normalization (BN) layer, if wanted
        if batch_norm is True:
            layers.append(nn.BatchNorm3d(num_features=out_channels))

        self.activation_fn = activation_fn

        # Put together the convolutional block as a sequence of the layers
        self.conv_block = nn.Sequential(*layers)

    def forward(
        self, input: Float[Tensor, "batch in_channels depth height width"]
    ) -> Float[Tensor, "batch out_channels depth_out height_out width_out"]:
        r"""Forward pass of the convolutional block."""
        output = self.activation_fn(self.conv_block(input))
        return output  # (B, out_channels, D', H', W')


class PixelShuffle3d(nn.Module):
    r"""3D pixel-shuffle operation for sub-pixel upscaling.

    Rearranges elements in a tensor of shape :math:`(B, C \times r^3, D, H, W)`
    to a tensor of shape :math:`(B, C, D \times r, H \times r, W \times r)`
    where :math:`r` is the upscale factor.

    Parameters
    ----------
    scale : int
        Upscale factor. Channel dimension is reduced by ``scale^3``.

    Forward
    -------
    input : torch.Tensor
        Input tensor of shape :math:`(B, C \times r^3, D, H, W)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C, D \times r, H \times r, W \times r)`.

    Notes
    -----
    Reference: http://www.multisilicon.com/blog/a25332339.html
    """

    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale

    def forward(
        self, input: Float[Tensor, "batch channels_in depth height width"]
    ) -> Float[Tensor, "batch channels_out depth_out height_out width_out"]:
        r"""Forward pass of pixel shuffle."""
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = int(channels // self.scale**3)

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(
            batch_size,
            nOut,
            self.scale,
            self.scale,
            self.scale,
            in_depth,
            in_height,
            in_width,
        )

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class SubPixel_ConvolutionalBlock3d(nn.Module):
    r"""Convolutional block with pixel shuffle for sub-pixel upscaling.

    Parameters
    ----------
    kernel_size : int, optional, default=3
        Convolutional kernel size.
    conv_layer_size : int, optional, default=64
        Latent channel size.
    scaling_factor : int, optional, default=2
        Pixel shuffle scaling factor.

    Forward
    -------
    input : torch.Tensor
        Input tensor of shape :math:`(B, C, D, H, W)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C, D \times s, H \times s, W \times s)`
        where :math:`s` is the ``scaling_factor``.
    """

    def __init__(
        self, kernel_size: int = 3, conv_layer_size: int = 64, scaling_factor: int = 2
    ):
        super().__init__()

        # A convolutional layer that increases the number of channels
        # by scaling factor^3, followed by pixel shuffle and PReLU
        self.conv = nn.Conv3d(
            in_channels=conv_layer_size,
            out_channels=conv_layer_size * (scaling_factor**3),
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        # These additional channels are shuffled to form additional pixels,
        #  upscaling each dimension by the scaling factor
        self.pixel_shuffle = PixelShuffle3d(scaling_factor)
        self.prelu = nn.PReLU()

    def forward(
        self, input: Float[Tensor, "batch channels depth height width"]
    ) -> Float[Tensor, "batch channels depth_out height_out width_out"]:
        r"""Forward pass of sub-pixel convolutional block."""
        output = self.conv(input)  # (B, C * s^3, D, H, W)
        output = self.pixel_shuffle(output)  # (B, C, D*s, H*s, W*s)
        output = self.prelu(output)  # (B, C, D*s, H*s, W*s)

        return output


class ResidualConvBlock3d(nn.Module):
    r"""3D residual convolutional block.

    Parameters
    ----------
    n_layers : int, optional, default=1
        Number of convolutional layers.
    kernel_size : int, optional, default=3
        Convolutional kernel size.
    conv_layer_size : int, optional, default=64
        Latent channel size.
    activation_fn : nn.Module, optional, default=nn.Identity()
        Activation function.

    Forward
    -------
    input : torch.Tensor
        Input tensor of shape :math:`(B, C, D, H, W)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C, D, H, W)` (same as input).
    """

    def __init__(
        self,
        n_layers: int = 1,
        kernel_size: int = 3,
        conv_layer_size: int = 64,
        activation_fn: nn.Module = nn.Identity(),
    ):
        super().__init__()

        layers = [
            ConvolutionalBlock3d(
                in_channels=conv_layer_size,
                out_channels=conv_layer_size,
                kernel_size=kernel_size,
                batch_norm=True,
                activation_fn=activation_fn,
            )
            for _ in range(n_layers - 1)
        ]
        # The final convolutional block with no activation
        layers.append(
            ConvolutionalBlock3d(
                in_channels=conv_layer_size,
                out_channels=conv_layer_size,
                kernel_size=kernel_size,
                batch_norm=True,
            )
        )

        self.conv_layers = nn.Sequential(*layers)

    def forward(
        self, input: Float[Tensor, "batch channels depth height width"]
    ) -> Float[Tensor, "batch channels depth height width"]:
        r"""Forward pass of residual block."""
        residual = input  # (B, C, D, H, W)
        output = self.conv_layers(input)  # (B, C, D, H, W)
        output = output + residual  # (B, C, D, H, W)

        return output
