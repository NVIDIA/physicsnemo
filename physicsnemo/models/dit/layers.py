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

"""Backward-compatible shim for DiT layers.

.. deprecated::
    ``physicsnemo.models.dit.layers`` has been moved to
    ``physicsnemo.nn.module.dit_layers``. Import from :mod:`physicsnemo.nn` instead.
    This shim will be removed in a future release.
"""

import warnings

from physicsnemo.core.warnings import LegacyFeatureWarning

warnings.warn(
    "physicsnemo.models.dit.layers is deprecated. "
    "Import DiT layer components from physicsnemo.nn instead "
    "(e.g. `from physicsnemo.nn import DiTBlock`). "
    "This backward-compatibility shim will be removed in a future release.",
    LegacyFeatureWarning,
    stacklevel=2,
)

from physicsnemo.nn.module.dit_layers import (  # noqa: E402, F401
    AttentionModuleBase,
    DetokenizerModuleBase,
    DiTBlock,
    Natten2DSelfAttention,
    PatchEmbed2DTokenizer,
    PerSampleDropout,
    ProjLayer,
    ProjReshape2DDetokenizer,
    TESelfAttention,
    TimmSelfAttention,
    TokenizerModuleBase,
    get_attention,
    get_detokenizer,
    get_layer_norm,
    get_tokenizer,
)

__all__ = [
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
]
