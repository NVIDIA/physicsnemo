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

"""Fast operations for small bounded integer index tuples."""

import math

import torch


def _packed_tuple_capacity_fits(index_bound: int, n_columns: int) -> bool:
    """Return whether ``index_bound**n_columns`` safely fits in signed int64."""
    if index_bound <= 1:
        return True
    return math.log2(index_bound) * n_columns < 63


def unique_index_tuples(
    rows: torch.Tensor,
    index_bound: int,
    *,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Deduplicate rows of bounded non-negative integer tuples.

    This is a faster specialization of :func:`torch.unique` with ``dim=0`` for
    mesh topology tensors. Each row is packed into one int64 key using
    ``index_bound`` as the radix, then one-dimensional ``torch.unique`` is used
    and the unique keys are unpacked back to rows.

    Parameters
    ----------
    rows : torch.Tensor
        Integer tensor with shape ``(n_rows, n_columns)``.
    index_bound : int
        Strict upper bound for every value in ``rows``. For mesh vertex-index
        tuples this is usually ``mesh.n_points``; for cell-pair tuples it is
        usually ``mesh.n_cells``.
    return_inverse : bool, optional
        Whether to return inverse indices, matching ``torch.unique``.
    return_counts : bool, optional
        Whether to return counts, matching ``torch.unique``.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, ...]
        Same return structure as ``torch.unique(rows, dim=0, ...)``.
    """
    if rows.ndim != 2:
        raise ValueError(f"rows must be 2D, got {rows.ndim}D with {rows.shape=}.")
    if torch.is_floating_point(rows):
        raise TypeError(f"rows must contain integer indices, got {rows.dtype=}.")
    if rows.shape[0] == 0:
        return torch.unique(
            rows,
            dim=0,
            return_inverse=return_inverse,
            return_counts=return_counts,
        )
    if index_bound < 1:
        raise ValueError(f"index_bound must be >= 1, got {index_bound=!r}.")

    n_columns = rows.shape[1]
    if n_columns == 0 or not _packed_tuple_capacity_fits(index_bound, n_columns):
        return torch.unique(
            rows,
            dim=0,
            return_inverse=return_inverse,
            return_counts=return_counts,
        )

    rows_i64 = rows.to(dtype=torch.int64)
    packed = rows_i64[:, 0]
    for column_idx in range(1, n_columns):
        packed = packed * index_bound + rows_i64[:, column_idx]

    unique_result = torch.unique(
        packed,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    if isinstance(unique_result, tuple):
        unique_packed = unique_result[0]
        extra_outputs = unique_result[1:]
    else:
        unique_packed = unique_result
        extra_outputs = ()

    unique_rows = torch.empty(
        (unique_packed.numel(), n_columns),
        dtype=rows.dtype,
        device=rows.device,
    )
    remainder = unique_packed
    for column_idx in range(n_columns - 1, -1, -1):
        unique_rows[:, column_idx] = (remainder % index_bound).to(rows.dtype)
        remainder = remainder // index_bound

    if extra_outputs:
        return (unique_rows, *extra_outputs)
    return unique_rows
