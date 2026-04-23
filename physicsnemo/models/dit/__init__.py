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

"""Diffusion Transformer (DiT) model.

The DiT layer components (``DiTBlock``, ``TokenizerModuleBase``, etc.) and conditioning
embedders (``ConditioningEmbedder``, ``DiTConditionEmbedder``, etc.) have been moved to
:mod:`physicsnemo.nn`. Import them from there instead:

.. code-block:: python

    from physicsnemo.nn import DiTBlock, ConditioningEmbedder

Importing these names from ``physicsnemo.models.dit`` is deprecated and will be removed
in a future release.
"""

import warnings as warnings

from physicsnemo.core.warnings import LegacyFeatureWarning

from .dit import DiT

__DEPRECATED_NAMES = {
    # From physicsnemo.nn.module.dit_layers
    "AttentionModuleBase",
    "DetokenizerModuleBase",
    "DiTBlock",
    "Natten2DSelfAttention",
    "PatchEmbed2DTokenizer",
    "PerSampleDropout",
    "ProjLayer",
    "ProjReshape2DDetokenizer",
    "TESelfAttention",
    "TimmSelfAttention",
    "TokenizerModuleBase",
    "get_attention",
    "get_detokenizer",
    "get_layer_norm",
    "get_tokenizer",
    # From physicsnemo.nn.module.conditioning_embedders
    "ConditioningEmbedder",
    "ConditioningEmbedderType",
    "DiTConditionEmbedder",
    "EDMConditionEmbedder",
    "ZeroConditioningEmbedder",
    "get_conditioning_embedder",
}


def __getattr__(name):
    if name in __DEPRECATED_NAMES:
        import physicsnemo.nn as _nn

        warnings.warn(
            f"Importing '{name}' from 'physicsnemo.models.dit' is deprecated. "
            f"Use 'from physicsnemo.nn import {name}' instead. "
            "This backward-compatibility shim will be removed in a future release.",
            LegacyFeatureWarning,
            stacklevel=2,
        )
        return getattr(_nn, name)
    raise AttributeError(f"module 'physicsnemo.models.dit' has no attribute {name!r}")
