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

"""Experimental neural network components for PhysicsNemo.

This subpackage contains experimental neural network layers and utilities
that are under active development. These components may have breaking API
changes between releases.
"""

from .attention_blocks import (
    LocalPointTransformerBlock,
    LocalTokenCrossAttentionBlock,
    ResidualMLP,
)
from .diffusion_unet_3d_blocks import UNetBlock3D, Conv3D, GroupNorm3D, UNetAttention3D
from .flare_attention import FLARE
from .point_tokenizer import PointCloudTokenizer
from .point_utils import (
    chunked_knn_indices,
    compute_batch_offset_step,
    counts_to_mask,
    flatten_batched_coords,
    flatten_padded_batch,
    gather_rows,
    masked_mean,
    unflatten_to_padded,
)
from .positional_encoding import FourierPositionalEncoding

__all__ = [
    "Conv3D",
    "FLARE",
    "FourierPositionalEncoding",
    "GroupNorm3D",
    "LocalPointTransformerBlock",
    "LocalTokenCrossAttentionBlock",
    "PointCloudTokenizer",
    "ResidualMLP",
    "UNetAttention3D",
    "UNetBlock3D",
    "chunked_knn_indices",
    "compute_batch_offset_step",
    "counts_to_mask",
    "flatten_batched_coords",
    "flatten_padded_batch",
    "gather_rows",
    "masked_mean",
    "unflatten_to_padded",
]
