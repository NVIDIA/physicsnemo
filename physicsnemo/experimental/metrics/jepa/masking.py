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

"""Padding-aware masking helpers for JEPA-style token statistics.

These two utilities sit between the layers that produce per-token features
and the regularizers / losses that compute statistics over them.
:func:`flatten_valid_token_features` collapses a padded batched feature
tensor into a flat ``(M, D)`` view of just the real tokens, and
:func:`reshape_token_features_for_sigreg` adds the leading "projection
group" axis that :class:`physicsnemo.experimental.metrics.jepa.SIGReg`
expects.
"""

from __future__ import annotations

import torch


def flatten_valid_token_features(
    features: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Flatten token features and drop padded rows when a mask is present.

    Rank-2 inputs of shape ``(N, D)`` are returned unchanged. Rank-3 inputs
    of shape ``(B, N, D)`` are reshaped to ``(B * N, D)`` when ``mask`` is
    ``None`` and indexed by the mask otherwise.

    Parameters
    ----------
    features : torch.Tensor
        Token features of shape ``(N, D)`` or ``(B, N, D)``.
    mask : torch.Tensor, optional
        Boolean mask of shape ``(B, N)``; ``True`` selects valid positions.
        Required to be ``None`` for rank-2 inputs.

    Returns
    -------
    torch.Tensor
        Flat tensor of shape ``(M, D)`` where ``M`` is the number of valid
        rows after masking (or ``B * N`` when no mask is provided).

    Raises
    ------
    ValueError
        If ``features`` is not rank 2 or 3, or if ``mask.shape`` does not
        match ``features.shape[:2]``.
    """
    if features.ndim == 2:
        return features
    if features.ndim != 3:
        raise ValueError(
            f"Expected rank-2 or rank-3 features, got {tuple(features.shape)}"
        )
    if mask is None:
        return features.reshape(-1, int(features.shape[-1]))
    if mask.shape != features.shape[:2]:
        raise ValueError(
            "mask must match features.shape[:2], "
            f"got {tuple(mask.shape)} vs {tuple(features.shape[:2])}"
        )
    return features[mask]


def reshape_token_features_for_sigreg(
    features: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Flatten token features into the ``(T, B, D)`` shape SIGReg expects.

    The leading ``T`` axis groups multiple sets of projections; this
    function emits ``T=1``. When the flattened result has zero rows the
    return is a zero-element ``(1, 0, D)`` placeholder so downstream code
    can keep working with a well-shaped tensor.

    Parameters
    ----------
    features : torch.Tensor
        Token features of shape ``(N, D)`` or ``(B, N, D)``.
    mask : torch.Tensor, optional
        Forwarded to :func:`flatten_valid_token_features`.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, M, D)`` ready to feed into ``SIGReg.forward``.
    """
    flat = flatten_valid_token_features(features, mask)
    if int(flat.shape[0]) == 0:
        return flat.new_zeros((1, 0, int(flat.shape[-1]) if flat.ndim == 2 else 0))
    return flat.unsqueeze(0)
