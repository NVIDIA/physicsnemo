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

r"""Utility functions for working with TensorDict objects.

This module provides helper functions for manipulating TensorDict objects,
including concatenation of leaf tensors, computing total lengths, and
splitting by tensor rank.

For field-to-rank *schema metadata* (e.g. ``{"pressure": 0, "velocity": 1}``),
see :mod:`~physicsnemo.experimental.models.globe.utilities.rank_spec` and its
:class:`RankSpecDict` type.
"""

from math import prod

import torch
from tensordict import TensorDict


def concatenated_length(td: TensorDict[str, torch.Tensor]) -> int:
    r"""Computes the total flattened length of all leaf tensors in a TensorDict.

    This function calculates the sum of the number of elements in each leaf tensor,
    excluding the batch dimensions. This is useful for determining the total feature
    dimension when all tensors will be concatenated along their last dimensions.

    Parameters
    ----------
    td : TensorDict[str, torch.Tensor]
        TensorDict containing tensors with shared batch dimensions. The batch
        dimensions are determined by ``td.batch_size``.

    Returns
    -------
    int
        The total number of elements across all leaf tensors when their
        non-batch dimensions are flattened. Returns 0 for empty TensorDicts.

    Examples
    --------
    >>> td = TensorDict({
    ...     "scalars": torch.randn(10, 3),      # 3 elements per batch
    ...     "vectors": torch.randn(10, 5, 2),   # 5*2=10 elements per batch
    ... }, batch_size=torch.Size([10]))
    >>> concatenated_length(td)
    13
    """
    return sum(
        prod(t.shape[td.batch_dims :])
        for t in td.values(include_nested=True, leaves_only=True)
    )


def concatenate_leaves(td: TensorDict[str, torch.Tensor]) -> torch.Tensor:
    r"""Concatenates all leaf tensors in a TensorDict along the last dimension.

    This function flattens all non-batch dimensions of each leaf tensor and
    concatenates them along a new last dimension. The resulting tensor has shape
    :math:`(*, F_{\text{total}})` where :math:`F_{\text{total}}` is the sum of
    flattened features across all leaf tensors.

    Parameters
    ----------
    td : TensorDict[str, torch.Tensor]
        TensorDict containing tensors to concatenate. All leaf tensors must
        share the same batch dimensions as specified by ``td.batch_size``.

    Returns
    -------
    torch.Tensor
        A tensor with shape :math:`(*, F_{\text{total}})`.
        For empty TensorDicts, returns an empty tensor with shape :math:`(*, 0)`
        on the same device as the TensorDict.

    Examples
    --------
    >>> td = TensorDict({
    ...     "a": torch.ones(2, 3, 4),      # Will contribute 3*4=12 features
    ...     "b": torch.ones(2, 5),         # Will contribute 5 features
    ... }, batch_size=torch.Size([2]))
    >>> result = concatenate_leaves(td)
    >>> result.shape
    torch.Size([2, 17])
    """
    tensors = tuple(td.values(include_nested=True, leaves_only=True))
    if len(tensors) == 0:
        return torch.empty(td.batch_size + torch.Size([0]), device=td.device)
    else:
        return torch.cat(
            [t.reshape(td.batch_size + torch.Size([-1])) for t in tensors],
            dim=-1,
        )


class TensorsByRank(dict):
    r"""Dictionary that auto-creates TensorDict values based on integer rank keys.

    This specialized dictionary behaves like ``collections.defaultdict``, but the default
    value depends on the key (rank). When accessing a missing rank key, it automatically
    creates an empty TensorDict with a ``batch_size`` appropriate for that rank.

    This is used internally by ``split_by_leaf_rank`` to group tensors by their rank
    (number of non-batch dimensions). The auto-initialization ensures that accessing
    any rank returns a valid TensorDict, even if no tensors of that rank exist.

    Parameters
    ----------
    batch_size : torch.Size
        Base batch size for rank-0 tensors (scalars).
    new_batch_dim : int or None, optional
        Optional dimension to append per rank level. If provided,
        rank-k tensors get ``batch_size + (new_batch_dim,) * k``. If ``None``, all
        ranks share the same ``batch_size``.
    device : torch.device or None, optional
        Device for auto-created TensorDicts.

    Attributes
    ----------
    batch_size : torch.Size
        Stored base batch size.
    new_batch_dim : int or None
        Stored per-rank dimension.
    device : torch.device or None
        Stored device.

    Examples
    --------
    >>> rd = TensorsByRank(batch_size=torch.Size([10]), device="cpu")
    >>> # Accessing rank 0 (scalars) auto-creates empty TensorDict
    >>> td0 = rd[0]  # TensorDict with batch_size=(10,)
    >>> # Accessing rank 1 (vectors) also auto-creates
    >>> td1 = rd[1]  # TensorDict with batch_size=(10,)
    >>> td0["scalar"] = torch.randn(10)
    >>> td1["vector"] = torch.randn(10, 3)

    Note
    ----
    Negative keys raise ``ValueError`` since negative-rank tensors are nonsensical.
    For external use, this behaves like a regular dict after initialization.
    """

    def __init__(
        self,
        batch_size: torch.Size,
        new_batch_dim: int | None = None,
        device: torch.device | None = None,
    ):
        self.batch_size = batch_size
        self.new_batch_dim = new_batch_dim
        self.device = device

    def __missing__(self, key: int) -> TensorDict[str, torch.Tensor]:
        r"""Auto-creates an empty TensorDict when accessing a missing rank key.

        Parameters
        ----------
        key : int
            Tensor rank (number of non-batch dimensions). Must be non-negative.

        Returns
        -------
        TensorDict[str, torch.Tensor]
            Empty TensorDict with ``batch_size`` appropriate for this rank.

        Raises
        ------
        ValueError
            If key is negative.
        """
        if key < 0:
            raise ValueError(f"No such thing as a tensor with rank {key}!")
        if self.new_batch_dim is None:
            new_batch_size = self.batch_size
        else:
            new_batch_size = self.batch_size + torch.Size([self.new_batch_dim] * key)
        self[key] = TensorDict({}, batch_size=new_batch_size, device=self.device)
        return self[key]


def split_by_leaf_rank(
    td: TensorDict[str, torch.Tensor], new_batch_dim: int | None = None
) -> TensorsByRank[int, TensorDict[str, torch.Tensor]]:
    r"""Splits a TensorDict into multiple TensorDicts grouped by tensor rank.

    This function groups leaf tensors by their rank (number of dimensions excluding
    batch dimensions) and returns a dictionary mapping each rank to a TensorDict
    containing all tensors of that rank. This is useful for processing tensors
    with different semantic meanings separately (e.g., scalars vs vectors vs matrices).

    Parameters
    ----------
    td : TensorDict[str, torch.Tensor]
        TensorDict to split. The batch dimensions are determined by ``td.batch_size``.
    new_batch_dim : int or None, optional
        Optional dimension to append per rank level in the output TensorDicts.

    Returns
    -------
    TensorsByRank[int, TensorDict[str, torch.Tensor]]
        Dictionary mapping rank (int) to TensorDict. Each
        TensorDict has the same ``batch_size`` and device as the input, and contains
        only the tensors with the corresponding rank. Empty TensorDicts are created
        for ranks that have tensors.

    Examples
    --------
    >>> td = TensorDict({
    ...     "scalar1": torch.randn(10),         # rank 0 (only batch dim)
    ...     "scalar2": torch.randn(10),         # rank 0
    ...     "vector": torch.randn(10, 3),       # rank 1
    ...     "matrix": torch.randn(10, 4, 4),    # rank 2
    ... }, batch_size=torch.Size([10]))
    >>> split_td = split_by_leaf_rank(td)
    >>> list(split_td.keys())
    [0, 1, 2]
    >>> list(split_td[0].keys())
    ['scalar1', 'scalar2']
    >>> list(split_td[1].keys())
    ['vector']
    >>> list(split_td[2].keys())
    ['matrix']
    """
    result = TensorsByRank(
        batch_size=td.batch_size, new_batch_dim=new_batch_dim, device=td.device
    )

    for k, v in td.items(include_nested=True, leaves_only=True):
        rank = v.ndim - td.batch_dims
        result[rank][k] = v

    return result


def combine_tensordicts(
    *tds: TensorDict[str, torch.Tensor],
) -> TensorDict[str, torch.Tensor]:
    r"""Combines multiple TensorDicts into a single TensorDict by merging their keys.

    This function creates a new TensorDict containing all key-value pairs from the
    input TensorDicts. Keys are merged via sequential update, so if multiple input
    TensorDicts contain the same key, the last one wins.

    Parameters
    ----------
    *tds : TensorDict[str, torch.Tensor]
        Variable number of TensorDict objects to combine. Must all have the
        same ``batch_size`` and device.

    Returns
    -------
    TensorDict[str, torch.Tensor]
        New TensorDict with ``batch_size`` and device matching the inputs,
        containing the union of all keys. For duplicate keys, the value from the
        last TensorDict containing that key is used.

    Raises
    ------
    ValueError
        If TensorDicts have different batch sizes or devices.

    Examples
    --------
    >>> td1 = TensorDict({"a": torch.tensor([1, 2])}, batch_size=[2])
    >>> td2 = TensorDict({"b": torch.tensor([3, 4])}, batch_size=[2])
    >>> combined = combine_tensordicts(td1, td2)
    >>> list(combined.keys())
    ['a', 'b']

    Note
    ----
    This is essentially a batch-aware dictionary merge operation. The validation
    is skipped during ``torch.compile`` for performance.
    """
    if not torch.compiler.is_compiling():
        if not all(td.batch_size == tds[0].batch_size for td in tds):
            raise ValueError("All TensorDicts must have the same batch size")
        if not all(td.device == tds[0].device for td in tds):
            raise ValueError("All TensorDicts must have the same device")

    batch_size = tds[0].batch_size
    device = tds[0].device
    result = TensorDict({}, batch_size=batch_size, device=device)

    for td in tds:
        result.update(td)
    return result
