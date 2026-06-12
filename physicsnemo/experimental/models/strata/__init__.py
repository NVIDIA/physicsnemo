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

r"""DiT3D and PixelDiT: 3D transformer regression models for fields on a sphere.

This package provides the DiT3D backbone and the two-stage PixelDiT model, the
3D analogs of :class:`physicsnemo.models.dit.DiT`. They combine 3D neighborhood
attention (:func:`physicsnemo.nn.functional.na3d`), 3D patch embedding, and an
optional stereographic rotary position embedding
(:class:`physicsnemo.nn.StereographicRotaryPositionEmbedding2D`) for spherical geometry.

.. important::

    These models reuse the Diffusion-Transformer (DiT) *architecture* but are
    **deterministic regression** models (e.g. weather emulation), **not**
    generative diffusion models. There is no diffusion / denoising process and
    no noise, timestep, class-label, or text conditioning: the diffusion-specific
    conditioning of the original DiT has been removed. The "DiT" in the names
    refers to the architecture lineage only.

Classes
-------
DiT3D
    3D Diffusion Transformer backbone (field-to-field, no diffusion conditioning).
DiT3DMetaData
    Metadata for :class:`DiT3D`.
PixelDiT
    Two-stage regression model: a DiT3D semantic stage conditions a
    pixel-resolution stage via pixel-wise adaptive layer norm (conditioned on
    the semantic features, not on a diffusion timestep).
PixelDiTMetaData
    Metadata for :class:`PixelDiT`.
Natten3DSelfAttention, DiT3DBlock, PatchEmbed3D, FinalLayer3D
    Building-block layers used by the models.
PixelDiTBlock, PixelDiTLastLayer
    Building-block layers used by :class:`PixelDiT`.

Examples
--------
>>> import torch
>>> from physicsnemo.experimental.models.strata import DiT3D
>>> model = DiT3D(
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

from .dit3d import DiT3D, DiT3DMetaData
from .layers import (
    DiT3DBlock,
    FinalLayer3D,
    Natten3DSelfAttention,
    PatchEmbed3D,
)
from .pixel import PixelDiT, PixelDiTBlock, PixelDiTLastLayer, PixelDiTMetaData

__all__ = [
    "DiT3D",
    "DiT3DMetaData",
    "PixelDiT",
    "PixelDiTMetaData",
    "Natten3DSelfAttention",
    "DiT3DBlock",
    "PatchEmbed3D",
    "FinalLayer3D",
    "PixelDiTBlock",
    "PixelDiTLastLayer",
]
