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

r"""Rank specification types and utilities for GLOBE field kernels.

A rank spec describes the tensor rank (0 = scalar, 1 = vector) of each
field in a kernel's input or output. It is a plain Python dict - not a
TensorDict - so that ``torch.compile`` can specialize on it as a
compile-time constant without graph breaks.

Rank specs may be flat::

    {"pressure": 0, "velocity": 1}

or nested to mirror a hierarchical TensorDict structure::

    {"group": {"field_a": 0, "field_b": 1}}

Use :func:`flatten_rank_spec` to reduce a nested spec to dot-separated
flat keys, and :func:`ranks_from_tensordict` to derive a rank spec from
an existing data TensorDict.

For *runtime* grouping of actual tensor data by observed rank, see
:func:`~physicsnemo.experimental.models.globe.utilities.tensordict_utils.split_by_leaf_rank`.
"""

from collections import Counter

from tensordict import TensorDict

### Type definition
type RankSpecDict = dict[str, int | RankSpecDict]


def flatten_rank_spec(
    rank_spec: RankSpecDict, sep: str = ".",
) -> dict[str, int]:
    r"""Flatten a possibly-nested rank spec to dot-separated string keys.

    Parameters
    ----------
    rank_spec : RankSpecDict
        Rank spec mapping field names to integer ranks or nested sub-specs.
    sep : str
        Separator for joining nested key segments. Defaults to ``"."``.

    Returns
    -------
    dict[str, int]
        Flat mapping from dot-separated field name to integer rank.

    Examples
    --------
    >>> flatten_rank_spec({"pressure": 0, "velocity": 1})
    {'pressure': 0, 'velocity': 1}

    >>> flatten_rank_spec({"group": {"a": 0, "b": 1}})
    {'group.a': 0, 'group.b': 1}
    """
    result: dict[str, int] = {}
    for k, v in rank_spec.items():
        if isinstance(v, dict):
            for sub_k, sub_v in flatten_rank_spec(v, sep).items():
                result[f"{k}{sep}{sub_k}"] = sub_v
        else:
            result[k] = v
    return result


def rank_counts(rank_spec: RankSpecDict) -> Counter[int]:
    r"""Count leaves by rank value in a rank spec.

    Parameters
    ----------
    rank_spec : RankSpecDict
        Rank spec mapping field names to integer ranks.

    Returns
    -------
    Counter[int]
        Mapping from rank value to the number of fields with that rank.
        Missing ranks default to 0.

    Examples
    --------
    >>> rank_counts({"a": 0, "b": 0, "c": 1})
    Counter({0: 2, 1: 1})
    """
    return Counter(flatten_rank_spec(rank_spec).values())


def ranks_from_tensordict(td: TensorDict) -> RankSpecDict:
    r"""Derive a :class:`RankSpecDict` from a data TensorDict's leaf shapes.

    Each leaf tensor is replaced by its rank (number of non-batch dimensions),
    producing a plain dict whose structure mirrors the TensorDict nesting.

    Parameters
    ----------
    td : TensorDict
        Data TensorDict whose leaf ranks should be extracted.

    Returns
    -------
    RankSpecDict
        Rank spec with integer leaves.

    Examples
    --------
    >>> import torch
    >>> from tensordict import TensorDict
    >>> td = TensorDict({
    ...     "pressure": torch.randn(10),
    ...     "velocity": torch.randn(10, 3),
    ... }, batch_size=[10])
    >>> ranks_from_tensordict(td)
    {'pressure': 0, 'velocity': 1}
    """
    result: RankSpecDict = {}
    for k in td.keys():
        v = td[k]
        if isinstance(v, TensorDict):
            result[k] = ranks_from_tensordict(v)  # ty: ignore[invalid-assignment]
        else:
            result[k] = v.ndim - td.batch_dims  # ty: ignore[invalid-assignment]
    return result
