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

from .flare_attention import FLARE
from .diffusion_unet_3d_blocks import UNetBlock3D, Conv3D, GroupNorm3D, UNetAttention3D
from .rope import (
    build_axial_rope_cos_sin_2d_continuous,
    build_rope_cos_sin_1d_continuous,
    spherical_centroid,
    stereographic_projection,
)
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

__all__ = [
    "FLARE",
    "UNetBlock3D",
    "Conv3D",
    "GroupNorm3D",
    "UNetAttention3D",
    "build_axial_rope_cos_sin_2d_continuous",
    "build_rope_cos_sin_1d_continuous",
    "spherical_centroid",
    "stereographic_projection",
    "PointCloudTokenizer",
    "chunked_knn_indices",
    "compute_batch_offset_step",
    "counts_to_mask",
    "flatten_batched_coords",
    "flatten_padded_batch",
    "gather_rows",
    "masked_mean",
    "unflatten_to_padded",
]
