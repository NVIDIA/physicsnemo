# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

"""
Unary operation helpers and dispatch wrappers for `ShardTensor`.

This module provides:
- A dispatch-level wrapper for `torch.unsqueeze` that preserves and adjusts
  sharding metadata for `ShardTensor`.
- Small utility helpers for normalizing dimensions and constructing shapes.
"""

from typing import Dict, List, Sequence

import torch

from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")

from torch.distributed.tensor.placement_types import (  # noqa: E402
    Shard,
)

from physicsnemo.distributed import ShardTensor  # noqa: E402

aten = torch.ops.aten


def unsqueeze_shape(shape: torch.Size | Sequence[int], dim: int) -> torch.Size:
    """
    Return a new `torch.Size` with a singleton dimension inserted at `dim`.

    - If `dim` is within the current rank, the new dimension is inserted at that index.
    - This mirrors the behavior of `torch.unsqueeze` at the shape level.

    Args:
        shape: The original shape as a `torch.Size` or sequence of integers.
        dim: The dimension index at which to insert a singleton dimension.

    Returns:
        A new `torch.Size` with the inserted dimension.
    """
    o_shape = list(shape)
    o_shape.insert(dim, 1)
    return torch.Size(tuple(o_shape))


def normalize_dim(dim: int, tensor_rank: int) -> int:
    """
    Normalize a possibly negative `dim` to a non-negative index for a given rank.

    Follows PyTorch semantics for unsqueeze:
    when `dim < 0`, the effective index is `tensor_rank + dim + 1`.

    Args:
        dim: The (possibly negative) dimension index.
        tensor_rank: The rank (number of dimensions) of the tensor.

    Returns:
        The normalized non-negative dimension index.
    """
    return dim if dim >= 0 else (dim % (tensor_rank + 1))


def unsqueeze_wrapper(input: ShardTensor, dim: int) -> ShardTensor:
    """
    Dispatch-level wrapper for `aten.unsqueeze` on `ShardTensor`.

    Ensures the output `ShardTensor` has correct placements and sharding shapes
    after inserting a singleton dimension. Replicated placements stay replicated.
    Sharded placements remain sharded, but their shard dimension is shifted by
    one if the unsqueezed dimension is before or equal to the shard dimension.

    Args:
        input: The input `ShardTensor`.
        dim: The dimension index at which to insert a singleton dimension. May be negative.

    Returns:
        A new `ShardTensor` with the local tensor unsqueezed and sharding metadata adjusted.
    """
    # This is a _dispatch_level_ wrapper, so we're intercepting aten.unsqueeze

    # The reason we have this intercept is to ensure we get the output
    # sharding shapes correct on irregular data.

    # Unsqueeze the underlying tensor:
    local_input = input.to_local()
    local_output = torch.unsqueeze(local_input, dim)

    tensor_rank = len(input.shape)

    # Normalize the dim against negative numbers:
    dim = normalize_dim(dim, tensor_rank)

    # Now, deal with tensor spec:

    in_placements = input._spec.placements

    output_placements = []

    for p in in_placements:
        # Replicated placements stay replicated

        # Sharded placements stay sharded, but if the unsqueeze
        # dim is before the sharded dim, the sharded dim is shifted by
        if p.is_shard() and p.dim >= dim:
            output_placements.append(Shard(p.dim + 1))
        else:
            output_placements.append(p)

    in_sharding_shapes = input._spec.sharding_shapes()
    out_sharding_shapes: Dict[int, List[torch.Size]] = {
        mesh_dim: [unsqueeze_shape(s, dim) for s in in_sharding_shapes[mesh_dim]]
        for mesh_dim in in_sharding_shapes.keys()
    }

    # If the unsqueeze dim is > the sharding dim, adjust it

    output = ShardTensor.from_local(
        local_output,
        input._spec.mesh,
        output_placements,
        out_sharding_shapes,
    )

    return output


ShardTensor.register_dispatch_handler(
    torch.ops.aten.unsqueeze.default, unsqueeze_wrapper
)
