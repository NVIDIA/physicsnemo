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

r"""Shard-aware ``F.linear`` for inputs sharded on non-feature dims.

A linear layer acts only on the last (feature) dimension, so for an input
sharded on any other dimension the computation is embarrassingly parallel:
every rank can apply the weight to its local rows independently.

Why the DTensor fallback is not enough: its linear decomposition flattens
all leading dims into one before the matmul, and DTensor can only shard a
flattened dim when the per-rank chunks stay expressible in its even-chunk
model. Concretely, with ``N = 2345`` unevenly sharded over 2 ranks as
``(1173, 1172)``:

- 3-D input ``(B, N, C)``, ``Shard(1)``: the flatten groups ``(B, N)`` with
  the sharded dim **last**, so each rank's flattened chunk is just its rows
  (``B*1173`` / ``B*1172``) -- DTensor allows this, which is why every
  transolver / DoMINO linear (all 3-D) works through the fallback.
- 4-D input ``(B, N, H, D)``, ``Shard(1)``: the flatten groups ``(N, H)``
  with a trailing dim **after** the sharded one. Per-rank flattened sizes
  become ``1173*H`` / ``1172*H`` -- unevenly sharded on the *new* dim, which
  DTensor's strict view rejects ("Cannot flatten unevenly sharded tensor").
  This shape appears in GeoTransolver's slice projection,
  ``Linear(dim_head, slice_num)`` applied to ``(B, N_geo, heads, dim_head)``.

This handler skips the flatten entirely: localize weight/bias to plain
tensors, compute the linear locally, and wrap the output with the input's
placements (feature dim replaced by ``out_features`` in the shard shapes --
uneven shapes are fine at the ShardTensor level). Inputs outside that
contract (``Partial`` placements, feature-dim sharding) fall back to the
DTensor path unchanged.

Weight/bias grads are per-rank partial sums over the input's sharded mesh
dims. Distributed parameters declare that via ``grad_placements``, so the
cotangent leaves the wrapper as a genuine ``Partial`` and is reduced lazily
downstream; plain tensors have no distributed boundary to carry that
through, so :class:`GradReducer` reduces their grads eagerly.
"""

from typing import Any, Callable

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate

from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.domain_parallel.shard_tensor import (
    _torch_function_fallback_via_dtensor,
)
from physicsnemo.domain_parallel.shard_utils.grad_ops import GradReducer


def linear_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:
    r"""Wrapper for ``torch.nn.functional.linear`` on ShardTensor inputs.

    Parameters
    ----------
    func : Callable
        Will be ``torch.nn.functional.linear``.
    types : Any
        The object types of the inputs (unused).
    args : tuple
        Positional arguments: ``(input, weight)`` or ``(input, weight, bias)``.
    kwargs : dict
        Keyword arguments (may contain ``bias``).

    Returns
    -------
    ShardTensor
        Output carrying the input's placements, with the feature dimension
        sized ``out_features``.
    """
    input = args[0]
    weight = args[1] if len(args) > 1 else kwargs.get("weight")
    bias = args[2] if len(args) > 2 else kwargs.get("bias")

    feature_dim = input.ndim - 1
    local_path = isinstance(input, ShardTensor) and all(
        p.is_replicate() or (p.is_shard() and p.dim != feature_dim)
        for p in input._spec.placements
    )
    if not local_path:
        return _torch_function_fallback_via_dtensor(func, args, kwargs)

    input_spec = input._spec

    # Param grads are partial sums on mesh dims where the input is sharded.
    param_grad_placements = tuple(
        Partial() if placement.is_shard() else Replicate()
        for placement in input_spec.placements
    )

    def localize_param(param: torch.Tensor | None) -> torch.Tensor | None:
        if param is None or not isinstance(param, (ShardTensor, DTensor)):
            # Plain tensor: grads must be reduced eagerly.
            return param if param is None else GradReducer.apply(param, input_spec)
        if all(p.is_replicate() for p in param._spec.placements):
            # No-comm unwrap; the Partial cotangent resolves lazily downstream.
            return param.to_local(grad_placements=param_grad_placements)
        # Sharded param: a real gather, whose backward reduce-scatters the
        # Partial cotangent.
        return param.full_tensor(grad_placements=param_grad_placements)

    local_weight = localize_param(weight)
    local_bias = localize_param(bias)

    local_output = torch.nn.functional.linear(
        input.to_local(), local_weight, local_bias
    )

    out_features = weight.shape[0]
    if any(p.is_shard() for p in input_spec.placements):
        output_shard_shapes = {
            mesh_dim: [tuple(s[:-1]) + (out_features,) for s in shapes]
            for mesh_dim, shapes in input_spec.sharding_shapes().items()
        }
        return ShardTensor.from_local(
            local_output,
            input_spec.mesh,
            input_spec.placements,
            sharding_shapes=output_shard_shapes,
        )
    return ShardTensor.from_local(local_output, input_spec.mesh, input_spec.placements)


ShardTensor.register_function_handler(torch.nn.functional.linear, linear_wrapper)
