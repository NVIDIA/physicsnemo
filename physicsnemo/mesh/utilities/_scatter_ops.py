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

"""Scatter operation utilities for aggregating data across mesh elements.

This module provides unified scatter-based aggregation operations that are
commonly used throughout physicsnemo.mesh for transferring data between different
mesh entities (points, cells, facets).
"""

import torch
from jaxtyping import Float, Int, Shaped

from physicsnemo.mesh.utilities._tolerances import safe_eps


def scatter_aggregate(
    src_data: Shaped[torch.Tensor, "n_src ..."],
    src_to_dst_mapping: Int[torch.Tensor, " n_src"],
    n_dst: int,
    weights: Float[torch.Tensor, " n_src"] | None = None,
    aggregation: str = "mean",
) -> Shaped[torch.Tensor, "n_dst ..."]:
    """Aggregate source data to destination using scatter operations.

    This is the core scatter-based aggregation pattern used throughout physicsnemo.mesh
    for operations like:

    - Aggregating cell data to points
    - Aggregating parent cell data to facets
    - Merging duplicate point data

    The pattern is:
    1. Initialize destination tensor with zeros
    2. Scatter-add weighted source data to destinations
    3. Scatter-add weights to compute normalization
    4. Divide aggregated data by total weights

    Parameters
    ----------
    src_data : torch.Tensor
        Source data to aggregate, shape (n_src, *data_shape).
    src_to_dst_mapping : torch.Tensor
        Mapping from each source to its destination index,
        shape (n_src,). Each value should be in [0, n_dst).
    n_dst : int
        Number of destination elements.
    weights : torch.Tensor or None
        Optional weights for each source element, shape (n_src,).
        If None, uses uniform weights of 1.0.
    aggregation : str
        Aggregation mode:

        - "mean": Weighted mean (uses weights if provided, uniform otherwise)
        - "sum": Weighted sum (no normalization)

    Returns
    -------
    torch.Tensor
        Aggregated data at destinations, shape (n_dst, *data_shape).
        For "mean" mode, values are weighted averages.
        For "sum" mode, values are weighted sums.

    Notes
    -----
    The output dtype follows ``src_data``, with one exception: a ``"mean"`` of
    an integer or boolean ``src_data`` is promoted to ``torch.float64``. A mean
    of integers is generally non-integral, so computing it in the source integer
    dtype would truncate (e.g. ``(1 + 2) // 2 == 1``); promoting to a floating
    dtype avoids this. A ``"sum"`` always preserves the source dtype.

    Examples
    --------
    >>> # Aggregate cell data to points
    >>> src_data = torch.tensor([[1.0], [2.0], [3.0]])  # 3 cells
    >>> src_to_dst = torch.tensor([0, 0, 1])  # map to 2 points
    >>> result = scatter_aggregate(src_data, src_to_dst, n_dst=2)
    >>> # result = [[1.5], [3.0]]  # point 0 gets mean of cells 0,1
    """
    device = src_data.device
    dtype = src_data.dtype

    ### Get data shape beyond the first dimension
    data_shape = src_data.shape[1:]

    if aggregation not in ("mean", "sum"):
        raise ValueError(f"Invalid {aggregation=}. Must be 'mean' or 'sum'.")

    ### Choose the compute dtype. A "mean" of integer/bool data must be computed in a
    ### floating dtype: integer division truncates (e.g. (1 + 2) // 2 == 1), and the
    ### division guard ``safe_eps()`` -> ``torch.finfo`` raises on integer dtypes. A
    ### "sum" preserves the native (possibly integer) dtype.
    if aggregation == "mean" and not torch.is_floating_point(src_data):
        compute_dtype = torch.float64
    else:
        compute_dtype = dtype

    ### Fast path: unweighted sum is a single scatter_add_ with no extra work
    if weights is None and aggregation == "sum":
        aggregated_data = torch.zeros((n_dst, *data_shape), dtype=dtype, device=device)
        expanded_indices = src_to_dst_mapping.view(
            -1, *([1] * len(data_shape))
        ).expand_as(src_data)
        aggregated_data.scatter_add_(dim=0, index=expanded_indices, src=src_data)
        return aggregated_data

    ### Initialize weights if not provided
    if weights is None:
        weights = torch.ones(
            len(src_to_dst_mapping), dtype=compute_dtype, device=device
        )

    ### Ensure weights share the compute dtype (avoid dtype mismatch in multiplication)
    if weights.dtype != compute_dtype:
        weights = weights.to(compute_dtype)

    ### Weight the source data
    # Broadcast weights to match data shape: (n_src, *data_shape)
    weight_shape = [len(weights)] + [1] * len(data_shape)
    weighted_data = src_data.to(compute_dtype) * weights.view(weight_shape)

    ### Scatter-add weighted data to destinations
    aggregated_data = torch.zeros(
        (n_dst, *data_shape),
        dtype=compute_dtype,
        device=device,
    )

    # Expand src_to_dst_mapping to match data dimensions
    expanded_indices = src_to_dst_mapping.view(-1, *([1] * len(data_shape))).expand_as(
        weighted_data
    )

    aggregated_data.scatter_add_(
        dim=0,
        index=expanded_indices,
        src=weighted_data,
    )

    ### Normalize weighted sum to weighted mean
    if aggregation == "mean":
        ### Compute sum of weights at each destination
        weight_sums = torch.zeros(n_dst, dtype=compute_dtype, device=device)
        weight_sums.scatter_add_(
            dim=0,
            index=src_to_dst_mapping,
            src=weights,
        )

        ### Normalize by total weight (avoid division by zero)
        weight_sums = weight_sums.clamp(min=safe_eps(weight_sums.dtype))
        aggregated_data = aggregated_data / weight_sums.view(
            -1, *([1] * len(data_shape))
        )

    return aggregated_data
