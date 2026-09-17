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

"""Core data types for the AeroJEPA framework.

Defines the two lightweight dataclasses that flow through the AeroJEPA
pipeline: :class:`TokenSet` bundles token features with their geometric
coordinates and optional mask/global token, and :class:`EncoderOutput` is
what context and target encoders return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TokenSet:
    r"""Bundle of token features and their geometric coordinates.

    A ``TokenSet`` is the canonical token representation passed between
    AeroJEPA components. Tensors may be either batched (rank 3) or unbatched
    (rank 2); the :attr:`is_batched` property reflects which.

    Parameters
    ----------
    features : torch.Tensor
        Token features of shape :math:`[N, D]` (unbatched) or
        :math:`[B, N, D]` (batched), where :math:`B` is the batch size,
        :math:`N` is the number of tokens and :math:`D` is the feature
        dimension.
    coords : torch.Tensor
        Token coordinates of shape :math:`[N, 3]` or :math:`[B, N, 3]`,
        matching the batching of ``features``.
    mask : torch.Tensor, optional
        Boolean validity mask of shape :math:`[N]` or :math:`[B, N]`. Entries
        equal to ``True`` indicate real tokens; ``False`` entries are padding
        that downstream components should ignore.
    global_token : torch.Tensor, optional
        Optional per-set global token of shape :math:`[D]` or :math:`[B, D]`.
    aux : dict, optional
        Arbitrary side data carried alongside the tokens (e.g. intermediate
        attention statistics). Defaults to an empty dict.

    Examples
    --------
    >>> import torch
    >>> ts = TokenSet(
    ...     features=torch.zeros(2, 10, 64),
    ...     coords=torch.zeros(2, 10, 3),
    ... )
    >>> ts.is_batched
    True
    >>> ts.token_dim
    64
    """

    features: torch.Tensor
    coords: torch.Tensor
    mask: torch.Tensor | None = None
    global_token: torch.Tensor | None = None
    aux: dict[str, Any] = field(default_factory=dict)

    @property
    def is_batched(self) -> bool:
        r"""Whether ``features`` is batched (rank 3) rather than unbatched (rank 2)."""
        return self.features.ndim == 3

    @property
    def token_dim(self) -> int:
        r"""Feature dimension :math:`D` (the last axis of ``features``)."""
        return int(self.features.shape[-1])

    def with_updates(
        self,
        *,
        features: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        global_token: torch.Tensor | None = None,
        aux: dict[str, Any] | None = None,
    ) -> TokenSet:
        r"""Return a new ``TokenSet`` with the supplied fields replaced.

        Fields left as ``None`` are kept from ``self``. When ``aux`` is left
        as ``None`` a shallow copy of the existing dict is taken so that the
        returned instance does not share later mutation with ``self``.

        Parameters
        ----------
        features : torch.Tensor, optional
            Replacement features.
        coords : torch.Tensor, optional
            Replacement coordinates.
        mask : torch.Tensor, optional
            Replacement mask.
        global_token : torch.Tensor, optional
            Replacement global token.
        aux : dict, optional
            Replacement auxiliary data. ``None`` shallow-copies the existing
            dict.

        Returns
        -------
        TokenSet
            A new instance with the requested updates applied.

        Examples
        --------
        >>> import torch
        >>> ts = TokenSet(
        ...     features=torch.zeros(1, 4, 8),
        ...     coords=torch.zeros(1, 4, 3),
        ... )
        >>> ts2 = ts.with_updates(mask=torch.ones(1, 4, dtype=torch.bool))
        >>> ts2.mask.shape
        torch.Size([1, 4])
        >>> ts2.features is ts.features
        True
        """
        return TokenSet(
            features=self.features if features is None else features,
            coords=self.coords if coords is None else coords,
            mask=self.mask if mask is None else mask,
            global_token=self.global_token if global_token is None else global_token,
            aux=dict(self.aux) if aux is None else aux,
        )


@dataclass
class EncoderOutput:
    r"""Output of an AeroJEPA encoder.

    A thin wrapper around :class:`TokenSet` that lets an encoder surface a
    separate ``global_token`` and arbitrary side data alongside the per-token
    features without conflating them with the token set itself.

    Parameters
    ----------
    tokens : TokenSet
        The encoded token set.
    global_token : torch.Tensor, optional
        Optional global summary token of shape :math:`[D]` or :math:`[B, D]`.
    aux : dict, optional
        Arbitrary side data the encoder wants to surface (e.g. attention
        weights, intermediate representations). Defaults to an empty dict.

    Examples
    --------
    >>> import torch
    >>> ts = TokenSet(
    ...     features=torch.zeros(2, 10, 64),
    ...     coords=torch.zeros(2, 10, 3),
    ... )
    >>> out = EncoderOutput(tokens=ts)
    >>> out.tokens.token_dim
    64
    >>> out.global_token is None
    True
    """

    tokens: TokenSet
    global_token: torch.Tensor | None = None
    aux: dict[str, Any] = field(default_factory=dict)
