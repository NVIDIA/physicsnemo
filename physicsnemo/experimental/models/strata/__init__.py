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

r"""StrataTransformer3D and Strata: 3D transformer regression models for fields on a sphere.

This package provides the StrataTransformer3D backbone and the two-stage Strata model, the
3D analogs of :class:`physicsnemo.models.dit.DiT`. They combine 3D neighborhood
attention (:func:`physicsnemo.nn.functional.na3d`), 3D patch embedding, and an
optional stereographic rotary position embedding
(:func:`physicsnemo.experimental.nn.build_axial_rope_cos_sin_2d_continuous`) for
spherical geometry.

.. important::

    These models reuse the Diffusion-Transformer (DiT) *architecture* but are
    **deterministic regression** models (e.g. weather emulation), **not**
    generative diffusion models. There is no diffusion / denoising process and
    no noise, timestep, class-label, or text conditioning: the diffusion-specific
    conditioning of the original DiT has been removed. The "DiT" in the names
    refers to the architecture lineage only.

Classes
-------
StrataTransformer3D
    3D Diffusion Transformer backbone (field-to-field, no diffusion conditioning).
StrataTransformer3DMetaData
    Metadata for :class:`StrataTransformer3D`.
Strata
    Two-stage regression model: a StrataTransformer3D backbone stage conditions a
    pixel-resolution stage via pixel-wise adaptive layer norm (conditioned on
    the backbone features, not on a diffusion timestep).
StrataMetaData
    Metadata for :class:`Strata`.
Natten3DSelfAttention, StrataTransformer3DBlock, PatchEmbed3D, FinalLayer3D
    Building-block layers used by the models.
StrataPixel3DBlock
    Building-block layer used by :class:`Strata`.

Examples
--------
>>> import torch
>>> from physicsnemo.experimental.models.strata import StrataTransformer3D
>>> model = StrataTransformer3D(
...     in_channels=4,
...     input_shape=(4, 8, 8),
...     patch_size=(1, 2, 2),
...     embed_dim=32,
...     num_heads=4,
...     num_layers=2,
...     attn_kernel=-1,
... )
>>> x = torch.randn(2, 4, 4, 8, 8)
>>> model(x).shape
torch.Size([2, 4, 4, 8, 8])
"""

from .transformer import StrataTransformer3D, StrataTransformer3DMetaData
from .layers import (
    StrataTransformer3DBlock,
    FinalLayer3D,
    Natten3DSelfAttention,
    PatchEmbed3D,
)
from .strata import Strata, StrataPixel3DBlock, StrataMetaData

__all__ = [
    "StrataTransformer3D",
    "StrataTransformer3DMetaData",
    "Strata",
    "StrataMetaData",
    "Natten3DSelfAttention",
    "StrataTransformer3DBlock",
    "PatchEmbed3D",
    "FinalLayer3D",
    "StrataPixel3DBlock",
]
