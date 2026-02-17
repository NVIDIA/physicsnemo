# ignore_header_test
# ruff: noqa: E402
""""""

"""
Pix2Pix model. This code was modified from, https://github.com/NVIDIA/pix2pixHD

The following license is provided from their source,

Copyright (C) 2019 NVIDIA Corporation. Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu.
BSD License. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


--------------------------- LICENSE FOR pytorch-CycleGAN-and-pix2pix ----------------
Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import warnings
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
from jaxtyping import Float

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn import get_activation

Tensor = torch.Tensor


@dataclass
class MetaData(ModelMetaData):
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = False  # Reflect padding not supported in bfloat16
    amp_gpu: bool = True
    # Inference
    onnx: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = True
    auto_grad: bool = True


class Pix2Pix(Module):
    r"""Convolutional encoder-decoder based on pix2pix generator models.

    .. deprecated::
        ``Pix2Pix`` is deprecated and will be removed in a future release.
        Consider using a custom encoder-decoder architecture built with
        :class:`~physicsnemo.nn.ConvResidualBlock` or other building blocks
        from :mod:`physicsnemo.nn`.

    Note
    ----
    The pix2pix architecture supports options for 1D, 2D and 3D fields which can
    be controlled using the ``dimension`` parameter.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    conv_layer_size : int, optional, default=64
        Latent channel size after first convolution.
    n_downsampling : int, optional, default=3
        Number of downsampling blocks.
    n_upsampling : int, optional, default=3
        Number of upsampling blocks.
    n_blocks : int, optional, default=3
        Number of residual blocks in middle of model.
    activation_fn : str, optional, default="relu"
        Activation function name.
    batch_norm : bool, optional, default=False
        Whether to use batch normalization.
    padding_type : str, optional, default="reflect"
        Padding type (``'reflect'``, ``'replicate'`` or ``'zero'``).

    Forward
    -------
    input : torch.Tensor
        Input tensor of shape :math:`(N, C_{in}, *)` where :math:`*` denotes
        spatial dimensions (1D, 2D, or 3D depending on ``dimension``).

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(N, C_{out}, *)` with the same spatial
        dimensions as input (adjusted by upsampling/downsampling ratio).

    Examples
    --------
    >>> import torch
    >>> import physicsnemo
    >>> model = physicsnemo.models.pix2pix.Pix2Pix(
    ...     in_channels=1,
    ...     out_channels=2,
    ...     dimension=2,
    ...     conv_layer_size=4)
    >>> input = torch.randn(4, 1, 32, 32)  # (N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([4, 2, 32, 32])

    Note
    ----
    Reference: Isola, Phillip, et al. "Image-To-Image translation with conditional
    adversarial networks" Conference on Computer Vision and Pattern Recognition, 2017.
    https://arxiv.org/abs/1611.07004

    Reference: Wang, Ting-Chun, et al. "High-Resolution image synthesis and semantic
    manipulation with conditional GANs" Conference on Computer Vision and Pattern
    Recognition, 2018. https://arxiv.org/abs/1711.11585

    Note
    ----
    Based on the implementation: https://github.com/NVIDIA/pix2pixHD
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,
        conv_layer_size: int = 64,
        n_downsampling: int = 3,
        n_upsampling: int = 3,
        n_blocks: int = 3,
        activation_fn: Union[str, nn.Module] = "relu",
        batch_norm: bool = False,
        padding_type: str = "reflect",
    ):
        warnings.warn(
            "Pix2Pix is deprecated and will be removed in a future release. "
            "Consider using a custom encoder-decoder architecture built with "
            "physicsnemo.nn.ConvResidualBlock or other building blocks from "
            "physicsnemo.nn.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not (n_blocks >= 0 and n_downsampling >= 0 and n_upsampling >= 0):
            raise ValueError("Invalid arch params")
        if padding_type not in ["reflect", "zero", "replicate"]:
            raise ValueError("Invalid padding type")
        super().__init__(meta=MetaData())

        # Store constructor parameters as attributes for introspection
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimension = dimension
        self.conv_layer_size = conv_layer_size
        self.n_downsampling = n_downsampling
        self.n_upsampling = n_upsampling
        self.n_blocks = n_blocks
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.padding_type = padding_type

        # activation function
        if isinstance(activation_fn, str):
            activation = get_activation(activation_fn)
        else:
            activation = activation_fn

        # set padding and convolutions
        if dimension == 1:
            padding = nn.ReflectionPad1d(3)
            conv = nn.Conv1d
            trans_conv = nn.ConvTranspose1d
            norm = nn.BatchNorm1d
        elif dimension == 2:
            padding = nn.ReflectionPad2d(3)
            conv = nn.Conv2d
            trans_conv = nn.ConvTranspose2d
            norm = nn.BatchNorm2d
        elif dimension == 3:
            padding = nn.ReflectionPad3d(3)
            conv = nn.Conv3d
            trans_conv = nn.ConvTranspose3d
            norm = nn.BatchNorm3d
        else:
            raise NotImplementedError(
                f"Pix2Pix only supported dimensions 1, 2, 3. Got {dimension}"
            )

        model = [
            padding,
            conv(in_channels, conv_layer_size, kernel_size=7, padding=0),
        ]
        if batch_norm:
            model.append(norm(conv_layer_size))
        model.append(activation)

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model.append(
                conv(
                    conv_layer_size * mult,
                    conv_layer_size * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            if batch_norm:
                model.append(norm(conv_layer_size * mult * 2))
            model.append(activation)

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    dimension,
                    conv_layer_size * mult,
                    padding_type=padding_type,
                    activation=activation,
                    use_batch_norm=batch_norm,
                )
            ]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model.append(
                trans_conv(
                    int(conv_layer_size * mult),
                    int(conv_layer_size * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            if batch_norm:
                model.append(norm(int(conv_layer_size * mult / 2)))
            model.append(activation)

        # super-resolution layers
        for i in range(max([0, n_upsampling - n_downsampling])):
            model.append(
                trans_conv(
                    int(conv_layer_size),
                    int(conv_layer_size),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            if batch_norm:
                model.append(norm(conv_layer_size))
            model.append(activation)

        model += [
            padding,
            conv(conv_layer_size, out_channels, kernel_size=7, padding=0),
        ]
        self.model = nn.Sequential(*model)

    def forward(
        self, input: Float[Tensor, "batch in_channels *spatial"]
    ) -> Float[Tensor, "batch out_channels *spatial"]:
        r"""Forward pass through the encoder-decoder network.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape :math:`(N, C_{in}, *)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(N, C_{out}, *)`.
        """
        # Input validation
        if not torch.compiler.is_compiling():
            if input.ndim < 2:
                raise ValueError(
                    f"Expected input tensor with at least 2 dimensions (N, C, ...), "
                    f"got {input.ndim}D tensor with shape {tuple(input.shape)}"
                )

        y = self.model(input)
        return y


class ResnetBlock(Module):
    r"""A simple ResNet block with skip connection.

    .. deprecated::
        ``ResnetBlock`` is deprecated and will be removed in a future release.
        Use :class:`~physicsnemo.nn.ConvResidualBlock` instead, which provides
        similar functionality with additional features.

    Parameters
    ----------
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    channels : int
        Number of feature channels.
    padding_type : str, optional, default="reflect"
        Padding type (``'reflect'``, ``'replicate'`` or ``'zero'``).
    activation : nn.Module, optional, default=nn.ReLU()
        Activation function module.
    use_batch_norm : bool, optional, default=False
        Whether to use batch normalization.
    use_dropout : bool, optional, default=False
        Whether to use dropout (currently unused).

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(N, C, *)` where :math:`*` denotes
        spatial dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(N, C, *)`, same as input.
    """

    def __init__(
        self,
        dimension: int,
        channels: int,
        padding_type: str = "reflect",
        activation: nn.Module = nn.ReLU(),
        use_batch_norm: bool = False,
        use_dropout: bool = False,
    ):
        warnings.warn(
            "ResnetBlock is deprecated and will be removed in a future release. "
            "Use physicsnemo.nn.ConvResidualBlock instead, which provides similar "
            "functionality with additional features.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(meta=None)
        if padding_type not in [
            "reflect",
            "zero",
            "replicate",
        ]:
            raise ValueError(f"Invalid padding type {padding_type}")

        if dimension == 1:
            conv = nn.Conv1d
            if padding_type == "reflect":
                padding = nn.ReflectionPad1d(1)
            elif padding_type == "replicate":
                padding = nn.ReplicationPad1d(1)
            else:
                padding = None
            norm = nn.BatchNorm1d
        elif dimension == 2:
            conv = nn.Conv2d
            if padding_type == "reflect":
                padding = nn.ReflectionPad2d(1)
            elif padding_type == "replicate":
                padding = nn.ReplicationPad2d(1)
            else:
                padding = None
            norm = nn.BatchNorm2d
        elif dimension == 3:
            conv = nn.Conv3d
            if padding_type == "reflect":
                padding = nn.ReflectionPad3d(1)
            elif padding_type == "replicate":
                padding = nn.ReplicationPad3d(1)
            else:
                padding = None
            norm = nn.BatchNorm3d
        else:
            raise NotImplementedError(
                f"Pix2Pix ResnetBlock only supported dimensions 1, 2, 3. Got {dimension}"
            )

        conv_block = []
        if padding_type != "zero":
            conv_block += [padding]
            p = 0
        else:
            p = 1  # Use built in conv padding

        conv_block.append(conv(channels, channels, kernel_size=3, padding=p))
        if use_batch_norm:
            conv_block.append(norm(channels))
        conv_block.append(activation)

        if padding_type != "zero":
            conv_block += [padding]
        conv_block += [
            conv(channels, channels, kernel_size=3, padding=p),
        ]
        if use_batch_norm:
            conv_block.append(norm(channels))

        self.conv_block = nn.Sequential(*conv_block)

    def forward(
        self, x: Float[Tensor, "batch channels *spatial"]
    ) -> Float[Tensor, "batch channels *spatial"]:
        r"""Forward pass with residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(N, C, *)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(N, C, *)`.
        """
        # Apply convolutions and add residual connection
        out = x + self.conv_block(x)
        return out
