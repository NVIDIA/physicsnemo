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

"""AeroJEPA neural network building blocks (experimental).

Reusable layers for the AeroJEPA framework: token dataclasses
(:class:`TokenSet`, :class:`EncoderOutput`), a deterministic Fourier
positional encoding, residual MLP and local point/cross attention blocks,
the point-cloud tokenizer, batching/mask/k-NN helpers, and the
prototype-anchor build/load utilities. These are exposed for composition
by users who want to build JEPA-style architectures for steady 3D
aerodynamic surrogate modeling independently of the full ``AeroJEPA``
model class (which lives in ``physicsnemo.experimental.models.aerojepa``).

API stability: experimental. Names and signatures may change between releases
until the design graduates out of ``physicsnemo.experimental``.

References
----------
Giral et al., "AeroJEPA: Learning Semantic Latent Representations for
Scalable 3D Aerodynamic Field Modeling", preprint arXiv:2605.05586 (2026).
"""

from .attention_blocks import (
    LocalPointTransformerBlock,
    LocalTokenCrossAttentionBlock,
    ResidualMLP,
)
from .point_tokenizer import PointCloudTokenizer
from .positional_encoding import FourierPositionalEncoding
from .prototype_anchors import (
    build_context_prototype_anchors,
    build_target_prototype_anchors,
    ensure_context_prototype_anchors,
    ensure_target_prototype_anchors,
    load_context_prototype_anchors,
    load_target_prototype_anchors,
)
from .token_utils import (
    chunked_knn_indices,
    compute_batch_offset_step,
    counts_to_mask,
    flatten_batched_coords,
    flatten_padded_batch,
    gather_rows,
    masked_mean,
    pad_token_sets,
    trim_batched_tokens,
    unflatten_to_padded,
)
from .types import EncoderOutput, TokenSet

__all__ = [
    # Core dataclasses
    "EncoderOutput",
    "TokenSet",
    # Positional encoding
    "FourierPositionalEncoding",
    # Attention blocks
    "LocalPointTransformerBlock",
    "LocalTokenCrossAttentionBlock",
    "ResidualMLP",
    # Tokenizer
    "PointCloudTokenizer",
    # Token batching / mask / k-NN helpers
    "chunked_knn_indices",
    "compute_batch_offset_step",
    "counts_to_mask",
    "flatten_batched_coords",
    "flatten_padded_batch",
    "gather_rows",
    "masked_mean",
    "pad_token_sets",
    "trim_batched_tokens",
    "unflatten_to_padded",
    # Prototype anchors
    "build_context_prototype_anchors",
    "build_target_prototype_anchors",
    "ensure_context_prototype_anchors",
    "ensure_target_prototype_anchors",
    "load_context_prototype_anchors",
    "load_target_prototype_anchors",
]
