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

# Make physicsnemo.nn.Module an available import like torch.nn.Module
from physicsnemo.core import Module

from .healpix import (
    HEALPixAvgPool,
    HEALPixFoldFaces,
    HEALPixLayer,
    HEALPixMaxPool,
    HEALPixPadding,
    HEALPixPaddingv2,
    HEALPixUnfoldFaces,
)
from .module.activations import (
    CappedGELU,
    CappedLeakyReLU,
    Identity,
    SquarePlus,
    Stan,
    get_activation,
)
from .module.attention_layers import (
    AttentionOp,
    EarthAttention2D,
    EarthAttention3D,
    UNetAttention,
)
from .module.ball_query import BQWarp
from .module.conv_layers import (
    Conv2d,
    ConvBlock,
    ConvGRULayer,
    ConvLayer,
    ConvResidualBlock,
    CubeEmbedding,
    TransposeConvLayer,
)
from .module.dgm_layers import DGMLayer
from .module.embedding_layers import FourierEmbedding, PositionalEmbedding
from .module.fourier_layers import (
    FourierFilter,
    FourierLayer,
    FourierMLP,
    GaborFilter,
    fourier_encode,
)
from .module.fully_connected_layers import (
    Conv1dFCLayer,
    Conv2dFCLayer,
    Conv3dFCLayer,
    ConvNdFCLayer,
    ConvNdKernel1Layer,
    FCLayer,
    Linear,
)
from .module.group_norm import GroupNorm, get_group_norm
from .module.kan_layers import KolmogorovArnoldNetwork
from .module.mlp_layers import Mlp
from .module.resample_layers import (
    DownSample2D,
    DownSample3D,
    UpSample2D,
    UpSample3D,
)
from .module.siren_layers import SirenLayer, SirenLayerType
from .module.spectral_layers import (
    SpectralConv1d,
    SpectralConv2d,
    SpectralConv3d,
    SpectralConv4d,
)
from .module.transformer_layers import (
    DecoderLayer,
    EncoderLayer,
    FuserLayer,
    SwinTransformer,
)
from .module.unet_layers import UNetBlock
from .module.weight_fact import WeightFactLinear
from .module.weight_norm import WeightNormLinear
