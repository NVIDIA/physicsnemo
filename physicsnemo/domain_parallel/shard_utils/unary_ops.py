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


r"""Unary operation helpers and functional intercept wrappers for ShardTensor.

This module provides:

- A functional-level wrapper for ``torch.unsqueeze`` that preserves and adjusts
  sharding metadata for ``ShardTensor``.
- Handlers for ``aten.unsqueeze.default`` at both ``__torch_function__`` and
  ``__torch_dispatch__`` so that direct ATen calls use the same sharding logic.
- Small utility helpers for normalizing dimensions and constructing shapes.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
from torch.distributed.tensor.placement_types import (
    Shard,
)

from physicsnemo.domain_parallel import ShardTensor

aten = torch.ops.aten


def unsqueeze_shape(shape: torch.Size | Sequence[int], dim: int) -> torch.Size:
    r"""Return a new torch.Size with a singleton dimension inserted at ``dim``.

    If ``dim`` is within the current rank, the new dimension is inserted at
    that index. This mirrors the behavior of ``torch.unsqueeze`` at the shape level.

    Parameters
    ----------
    shape : torch.Size | Sequence[int]
        The original shape as a ``torch.Size`` or sequence of integers.
    dim : int
        The dimension index at which to insert a singleton dimension.

    Returns
    -------
    torch.Size
        A new ``torch.Size`` with the inserted dimension.
    """
    o_shape = list(shape)
    o_shape.insert(dim, 1)
    return torch.Size(tuple(o_shape))


def normalize_dim(dim: int, tensor_rank: int) -> int:
    r"""Normalize a possibly negative ``dim`` to a non-negative index for a given rank.

    Follows PyTorch semantics for unsqueeze: when ``dim < 0``, the effective
    index is ``tensor_rank + dim + 1``.

    Parameters
    ----------
    dim : int
        The (possibly negative) dimension index.
    tensor_rank : int
        The rank (number of dimensions) of the tensor.

    Returns
    -------
    int
        The normalized non-negative dimension index.
    """
    return dim if dim >= 0 else (dim % (tensor_rank + 1))


def unsqueeze_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:
    r"""Functional-level wrapper for ``torch.unsqueeze`` on ShardTensor.

    Ensures the output ShardTensor has correct placements and sharding shapes
    after inserting a singleton dimension. Replicated placements stay replicated.
    Sharded placements remain sharded, but their shard dimension is shifted by
    one if the unsqueezed dimension is before or equal to the shard dimension.

    Parameters
    ----------
    func : Callable
        The original function being wrapped (``torch.unsqueeze`` or
        ``torch.Tensor.unsqueeze``).
    types : tuple[Any, ...]
        Types of the input arguments (unused).
    args : tuple[Any, ...]
        Positional arguments. Expected to contain ``(input, dim)`` for
        ``torch.unsqueeze`` and ``(self, dim)`` for ``Tensor.unsqueeze``.
    kwargs : dict[str, Any]
        Keyword arguments (unused).

    Returns
    -------
    ShardTensor
        A new ShardTensor with the local tensor unsqueezed and sharding
        metadata adjusted.
    """
    # This is a _functional_level_ wrapper, so we're intercepting
    # torch.unsqueeze / Tensor.unsqueeze before they reach aten dispatch.

    # The reason we have this intercept is to ensure we get the output
    # sharding shapes correct on irregular data.

    # Unpack args from the __torch_function__ signature:
    input: ShardTensor = args[0]
    dim: int = args[1] if len(args) > 1 else kwargs.get("dim", 0)

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
        # dim is before the sharded dim, the sharded dim is shifted by one
        if p.is_shard() and p.dim >= dim:
            output_placements.append(Shard(p.dim + 1))
        else:
            output_placements.append(p)

    in_sharding_shapes = input._spec.sharding_shapes()
    out_sharding_shapes: dict[int, list[torch.Size]] = {
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


def _unsqueeze_dispatch(tensor: ShardTensor, dim: int) -> ShardTensor:
    r"""Dispatch handler for ``aten.unsqueeze.default`` on :class:`ShardTensor`.

    Called at the ``__torch_dispatch__`` level so that direct ATen calls
    (e.g. from internal PyTorch or DTensor code) use the same sharding logic
    as the Python-level ``torch.unsqueeze`` / ``Tensor.unsqueeze``.

    Parameters
    ----------
    tensor : ShardTensor
        Input sharded tensor.
    dim : int
        Dimension at which to insert the singleton dimension.

    Returns
    -------
    ShardTensor
        Unsqueezed ShardTensor with correct placements and sharding shapes.
    """
    return unsqueeze_wrapper(aten.unsqueeze.default, (type(tensor),), (tensor, dim), {})


# Python-level function handlers (__torch_function__).
ShardTensor.register_function_handler(torch.unsqueeze, unsqueeze_wrapper)
ShardTensor.register_function_handler(torch.Tensor.unsqueeze, unsqueeze_wrapper)

# ATen op: can be invoked via __torch_function__ (e.g. PyTorch 2.6+ internal
# or DTensor codepaths) or via __torch_dispatch__.
ShardTensor.register_function_handler(aten.unsqueeze.default, unsqueeze_wrapper)
ShardTensor.register_dispatch_handler(aten.unsqueeze.default, _unsqueeze_dispatch)
