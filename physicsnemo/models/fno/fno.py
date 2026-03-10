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

from dataclasses import dataclass
from typing import List, Union

import torch
from jaxtyping import Float
from torch import Tensor

import physicsnemo.nn as layers
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.models.mlp import FullyConnected

# Import FNO encoder layers from physicsnemo.nn
from physicsnemo.nn.module.fno_layers import (
    FNO1DEncoder,
    FNO2DEncoder,
    FNO3DEncoder,
    FNO4DEncoder,
)

# ===================================================================
# ===================================================================
# General FNO Model
# ===================================================================
# ===================================================================


@dataclass
class MetaData(ModelMetaData):
    # Optimization
    jit: bool = False
    cuda_graphs: bool = True
    amp: bool = False
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class FNO(Module):
    r"""Fourier neural operator (FNO) model.

    The FNO architecture supports options for 1D, 2D, 3D and 4D fields which can
    be controlled using the ``dimension`` parameter.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    decoder_layers : int, optional, default=1
        Number of decoder layers.
    decoder_layer_size : int, optional, default=32
        Number of neurons in decoder layers.
    decoder_activation_fn : str, optional, default="silu"
        Activation function for decoder.
    dimension : int, optional, default=2
        Model dimensionality (supports 1, 2, 3, 4).
    latent_channels : int, optional, default=32
        Latent features size in spectral convolutions.
    num_fno_layers : int, optional, default=4
        Number of spectral convolutional layers.
    num_fno_modes : Union[int, List[int]], optional, default=16
        Number of Fourier modes kept in spectral convolutions.
    padding : int, optional, default=8
        Domain padding for spectral convolutions.
    padding_type : str, optional, default="constant"
        Type of padding for spectral convolutions.
    activation_fn : str, optional, default="gelu"
        Activation function.
    coord_features : bool, optional, default=True
        Use coordinate grid as additional feature map.

    Forward
    -------
    x : torch.Tensor
        Input tensor. Shape depends on ``dimension``:

        - 1D: :math:`(B, C_{in}, L)` where :math:`L` is sequence length
        - 2D: :math:`(B, C_{in}, H, W)`
        - 3D: :math:`(B, C_{in}, D, H, W)`
        - 4D: :math:`(B, C_{in}, X, Y, Z, T)`

    Outputs
    -------
    torch.Tensor
        Output tensor with same spatial dimensions as input:

        - 1D: :math:`(B, C_{out}, L)`
        - 2D: :math:`(B, C_{out}, H, W)`
        - 3D: :math:`(B, C_{out}, D, H, W)`
        - 4D: :math:`(B, C_{out}, X, Y, Z, T)`

    Examples
    --------
    >>> import torch
    >>> import physicsnemo
    >>> # Define a 2D FNO model
    >>> model = physicsnemo.models.fno.FNO(
    ...     in_channels=4,
    ...     out_channels=3,
    ...     decoder_layers=2,
    ...     decoder_layer_size=32,
    ...     dimension=2,
    ...     latent_channels=32,
    ...     num_fno_layers=2,
    ...     padding=0,
    ... )
    >>> input = torch.randn(32, 4, 32, 32)  # (N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([32, 3, 32, 32])

    See Also
    --------
    `Fourier Neural Operator (FNO) <https://arxiv.org/abs/2010.08895>`_ :
        Original FNO paper.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        decoder_layers: int = 1,
        decoder_layer_size: int = 32,
        decoder_activation_fn: str = "silu",
        dimension: int = 2,
        latent_channels: int = 32,
        num_fno_layers: int = 4,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: int = 8,
        padding_type: str = "constant",
        activation_fn: str = "gelu",
        coord_features: bool = True,
    ) -> None:
        # Disable torch.compile for 4D FNO due to PyTorch FFT stride bug
        # (https://github.com/pytorch/pytorch/issues/106623)
        # The _fft_c2c meta kernel has incorrect strides for 4+ dimensions
        jit_enabled = dimension != 4
        super().__init__(meta=MetaData(jit=jit_enabled))

        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.num_fno_modes = num_fno_modes
        self.padding = padding
        self.padding_type = padding_type
        self.activation_fn = layers.get_activation(activation_fn)
        self.coord_features = coord_features
        self.dimension = dimension

        # decoder net
        self.decoder_net = FullyConnected(
            in_features=latent_channels,
            layer_size=decoder_layer_size,
            out_features=out_channels,
            num_layers=decoder_layers,
            activation_fn=decoder_activation_fn,
        )

        FNOModel = self._getFNOEncoder()

        self.spec_encoder = FNOModel(
            in_channels,
            num_fno_layers=self.num_fno_layers,
            fno_layer_size=latent_channels,
            num_fno_modes=self.num_fno_modes,
            padding=self.padding,
            padding_type=self.padding_type,
            activation_fn=self.activation_fn,
            coord_features=self.coord_features,
        )

    def _getFNOEncoder(self) -> type:
        r"""Return the correct FNO encoder class based on the dimension.

        Returns
        -------
        type
            The appropriate encoder class for the configured dimension.

        Raises
        ------
        NotImplementedError
            If dimension is not 1, 2, 3, or 4.
        """
        if self.dimension == 1:
            return FNO1DEncoder
        elif self.dimension == 2:
            return FNO2DEncoder
        elif self.dimension == 3:
            return FNO3DEncoder
        elif self.dimension == 4:
            return FNO4DEncoder
        else:
            raise NotImplementedError(
                f"Invalid dimensionality {self.dimension}. "
                "Only 1D, 2D, 3D and 4D FNO implemented"
            )

    def forward(self, x: Float[Tensor, "B C *dims"]) -> Float[Tensor, "B C_out *dims"]:
        r"""Forward pass of the FNO model."""
        # Input validation: single check for ndim and channels
        if not torch.compiler.is_compiling():
            expected_ndim = self.dimension + 2  # batch + channels + spatial dims
            if x.ndim != expected_ndim or x.shape[1] != self.in_channels:
                raise ValueError(
                    f"Expected {expected_ndim}D input (B, {self.in_channels}, ...) for "
                    f"{self.dimension}D FNO, got {x.ndim}D tensor with shape {tuple(x.shape)}"
                )

        # Encode in Fourier space
        y_latent = self.spec_encoder(x)

        # Reshape to pointwise inputs for decoder
        y_shape = y_latent.shape
        y_latent, y_shape = self.spec_encoder.grid_to_points(y_latent)

        # Decode to output channels
        y = self.decoder_net(y_latent)

        # Convert back into grid representation
        y = self.spec_encoder.points_to_grid(y, y_shape)

        return y
