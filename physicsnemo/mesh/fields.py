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

r"""Field-rank schema metadata shared by mesh-based models.

A rank specification maps field names to their tensor ranks. It is a plain
Python dictionary rather than a TensorDict so model construction can treat it
as static metadata. Specifications may be flat or nested to mirror a
hierarchical TensorDict structure.

This module contains the representation-neutral schema helpers originally
implemented by GLOBE. Runtime grouping and packing of tensor data belongs to
the consuming model rather than this metadata module.
"""

from __future__ import annotations

from collections import Counter
from typing import TypeAlias, Union

from tensordict import TensorDict

# TODO: replace with ``type RankSpecDict = ...`` after Python 3.11 support is
# dropped (PEP 695).
RankSpecDict: TypeAlias = dict[str, Union[int, "RankSpecDict"]]


def flatten_rank_spec(rank_spec: RankSpecDict, sep: str = ".") -> dict[str, int]:
    r"""Flatten a nested rank specification to separator-joined names.

    The insertion order of ``rank_spec`` is retained for compatibility.

    Parameters
    ----------
    rank_spec : RankSpecDict
        Mapping from field names to integer ranks or nested mappings.
    sep : str, default="."
        Separator used to join nested path components.

    Returns
    -------
    dict[str, int]
        Flat field-name to semantic-rank mapping.

    Examples
    --------
    >>> flatten_rank_spec({"pressure": 0, "velocity": 1})
    {'pressure': 0, 'velocity': 1}
    >>> flatten_rank_spec({"fluid": {"pressure": 0, "velocity": 1}})
    {'fluid.pressure': 0, 'fluid.velocity': 1}
    """
    result: dict[str, int] = {}
    for key, value in rank_spec.items():
        if isinstance(value, dict):
            for sub_key, rank in flatten_rank_spec(value, sep=sep).items():
                result[f"{key}{sep}{sub_key}"] = rank
        else:
            result[key] = value
    return result


def rank_counts(rank_spec: RankSpecDict) -> Counter[int]:
    r"""Count leaves of each semantic rank in ``rank_spec``."""
    return Counter(flatten_rank_spec(rank_spec).values())


def ranks_from_tensordict(td: TensorDict) -> RankSpecDict:
    r"""Derive semantic-rank-shaped metadata from TensorDict leaf shapes.

    A leaf rank is its number of non-batch dimensions. For a point-field
    TensorDict with batch size ``(N,)``, ``(N,)`` is therefore rank 0 and
    ``(N, D)`` is rank 1.
    """
    result: RankSpecDict = {}
    for key in td.keys():
        value = td[key]
        if isinstance(value, TensorDict):
            result[key] = ranks_from_tensordict(value)  # ty: ignore[invalid-assignment]
        else:
            result[key] = value.ndim - td.batch_dims  # ty: ignore[invalid-assignment]
    return result


def validate_data_contains_ranks(
    *,
    data: TensorDict,
    declared_ranks: RankSpecDict,
    source_label: str,
) -> None:
    r"""Validate that ``data`` contains every declared leaf at its stated rank.

    Additional leaves in ``data`` are allowed. Missing leaves and rank
    mismatches are reported together to make schema errors easier to diagnose.

    Parameters
    ----------
    data : TensorDict
        Data TensorDict to validate.
    declared_ranks : RankSpecDict
        Rank specification that ``data`` must contain as a subset.
    source_label : str
        Human-readable description used in the error message.

    Raises
    ------
    ValueError
        If a declared field is missing or has a different rank.
    """
    declared = flatten_rank_spec(declared_ranks)
    actual = flatten_rank_spec(ranks_from_tensordict(data))

    lines = [
        f"  - missing leaf {key!r} (declared rank {declared[key]})"
        for key in sorted(declared.keys() - actual.keys())
    ]
    lines.extend(
        [
            f"  - rank mismatch for {key!r}: declared {declared[key]}, "
            f"got {actual[key]}"
            for key in sorted(declared.keys() & actual.keys())
            if declared[key] != actual[key]
        ]
    )
    if lines:
        raise ValueError(
            f"{source_label} does not contain its declared rank spec:\n"
            + "\n".join(lines)
        )


__all__ = [
    "RankSpecDict",
    "flatten_rank_spec",
    "rank_counts",
    "ranks_from_tensordict",
    "validate_data_contains_ranks",
]
