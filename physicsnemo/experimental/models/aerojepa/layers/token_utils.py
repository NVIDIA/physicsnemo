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

"""TokenSet-coupled helpers for AeroJEPA.

Two functions that pack and unpack lists of :class:`TokenSet` instances.
The generic batching, masking and k-NN utilities they build on live at
:mod:`physicsnemo.experimental.nn.point_utils`.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from physicsnemo.experimental.nn.point_utils import masked_mean

from .types import TokenSet


def trim_batched_tokens(tokens: TokenSet, index: int, count: int) -> TokenSet:
    r"""Extract one batch element of a batched ``TokenSet`` and trim its length.

    Parameters
    ----------
    tokens : TokenSet
        A batched ``TokenSet`` with rank-3 ``features`` and ``coords``.
    index : int
        Batch index to extract.
    count : int
        Number of leading tokens to keep from that batch element.

    Returns
    -------
    TokenSet
        Unbatched ``TokenSet`` (rank-2 ``features`` and ``coords``) sliced
        to length ``count``. The ``global_token`` and ``mask`` are sliced
        accordingly; ``aux`` is shallow-copied.

    Raises
    ------
    ValueError
        If ``tokens`` is not batched.
    """
    if not tokens.is_batched:
        raise ValueError("trim_batched_tokens expects a batched TokenSet.")
    mask = None
    if tokens.mask is not None:
        mask = tokens.mask[index, :count]
    global_token = None
    if tokens.global_token is not None:
        global_token = tokens.global_token[index : index + 1]
    return TokenSet(
        features=tokens.features[index, :count],
        coords=tokens.coords[index, :count],
        mask=mask,
        global_token=global_token,
        aux=dict(tokens.aux),
    )


def pad_token_sets(token_sets: Iterable[TokenSet]) -> TokenSet:
    r"""Pack a list of unbatched ``TokenSet`` instances into a batched one.

    Each item's tokens are placed in the leading positions of the batched
    output and a boolean ``mask`` records validity. Missing ``global_token``
    entries are synthesised by taking the masked mean of features so the
    output always carries a well-defined per-batch global token.

    Parameters
    ----------
    token_sets : Iterable[TokenSet]
        Unbatched token sets to pack. Must be non-empty and share feature
        / coordinate dimensions, device, and dtype.

    Returns
    -------
    TokenSet
        Batched ``TokenSet`` of length ``max_i features[i].shape[0]``.

    Raises
    ------
    ValueError
        If ``token_sets`` is empty.
    """
    token_sets = list(token_sets)
    if not token_sets:
        raise ValueError("pad_token_sets expects at least one TokenSet.")
    max_tokens = max(int(ts.features.shape[0]) for ts in token_sets)
    feat_dim = int(token_sets[0].features.shape[-1])
    coord_dim = int(token_sets[0].coords.shape[-1])
    batch_size = len(token_sets)
    device = token_sets[0].features.device
    feat_dtype = token_sets[0].features.dtype
    coord_dtype = token_sets[0].coords.dtype
    padded_features = torch.zeros(
        (batch_size, max_tokens, feat_dim), device=device, dtype=feat_dtype
    )
    padded_coords = torch.zeros(
        (batch_size, max_tokens, coord_dim), device=device, dtype=coord_dtype
    )
    mask = torch.zeros((batch_size, max_tokens), device=device, dtype=torch.bool)
    global_tokens = []
    for i, token_set in enumerate(token_sets):
        count = int(token_set.features.shape[0])
        padded_features[i, :count] = token_set.features
        padded_coords[i, :count] = token_set.coords
        mask[i, :count] = True
        if token_set.global_token is not None:
            global_tokens.append(token_set.global_token.reshape(1, -1))
        else:
            global_tokens.append(masked_mean(token_set.features, None).reshape(1, -1))
    return TokenSet(
        features=padded_features,
        coords=padded_coords,
        mask=mask,
        global_token=torch.cat(global_tokens, dim=0),
    )
