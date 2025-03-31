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

from .activations import (
    CappedGELU,
    CappedLeakyReLU,
    Identity,
    SquarePlus,
    Stan,
    get_activation,
)
from .conv_layers import ConvBlock, CubeEmbedding
from .dgm_layers import DGMLayer
from .fourier_layers import FourierFilter, FourierLayer, GaborFilter
from .fully_connected_layers import (
    Conv1dFCLayer,
    Conv2dFCLayer,
    Conv3dFCLayer,
    ConvNdFCLayer,
    ConvNdKernel1Layer,
    FCLayer,
)
from .mlp_layers import Mlp
from .resample_layers import (
    DownSample2D,
    DownSample3D,
    UpSample2D,
    UpSample3D,
)
from .siren_layers import SirenLayer, SirenLayerType
from .spectral_layers import (
    SpectralConv1d,
    SpectralConv2d,
    SpectralConv3d,
    SpectralConv4d,
)
from .transformer_layers import (
    DecoderLayer,
    EncoderLayer,
    FuserLayer,
    SwinTransformer,
)
from .weight_fact import WeightFactLinear
from .weight_norm import WeightNormLinear
