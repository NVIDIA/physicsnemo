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

r"""Custom tensor operations for ShardTensor dispatch.

This module provides dispatch and function handlers for tensor operations
that need special handling when applied to ``ShardTensor`` objects. Handlers
are registered with both ``__torch_dispatch__`` (ATen level) and
``__torch_function__`` (Python level) on :class:`ShardTensor`.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor.placement_types import (
    Replicate,
    Shard,
)

from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.domain_parallel._shard_tensor_spec import (
    ShardTensorSpec,
    _stride_from_contiguous_shape_C_style,
)

aten = torch.ops.aten


def _unbind_output_metadata(
    input_spec: ShardTensorSpec, dim: int
) -> tuple[int, list, dict[int, list[torch.Size]]]:
    r"""Compute the normalized dim, output placements, and sharding shapes for unbind.

    Validates that the unbind dimension is not sharded and does not use
    ``Partial`` placement, then returns the metadata needed to construct
    the output ``ShardTensor`` objects.

    Parameters
    ----------
    input_spec : ShardTensorSpec
        Specification of the input sharded tensor.
    dim : int
        Dimension along which to unbind (may be negative).

    Returns
    -------
    tuple[int, list, dict[int, list[torch.Size]]]
        - Normalized (non-negative) ``dim``.
        - Output placements (shard dims above ``dim`` shifted down by 1).
        - Output sharding shapes with the unbind dimension removed.

    Raises
    ------
    RuntimeError
        If attempting to unbind along a sharded dimension (not yet implemented).
        If attempting to unbind with ``Partial`` placement (not yet supported).
    """
    ndim = len(input_spec.shape)
    if dim < 0:
        dim = dim % ndim

    # if the unbind dimension is along a dimension that is sharded, we have to handle that.
    # If it's along an unsharded dimension, there is nearly nothing to do.
    input_placements = input_spec.placements
    shards = [s for s in input_placements if isinstance(s, Shard)]

    if dim in [i.dim for i in shards]:
        raise RuntimeError("No implementation for unbinding along sharding axis yet.")

    new_placements: list = []
    for p in input_placements:
        if p.is_replicate():
            new_placements.append(p)
        elif p.is_shard():
            if p.dim > dim:
                new_placements.append(Shard(p.dim - 1))
            else:
                new_placements.append(p)
        elif p.is_partial():
            raise RuntimeError("Partial placement not supported yet for unbind")

    # Plain int tuples (never torch.Size) -- see ShardTensorSpec._sharding_shapes
    # field docs for the dynamo / fakeification rationale.
    out_sharding_shapes: dict[int, list[tuple[int, ...]]] = {
        mesh_dim: [tuple(list(cs[:dim]) + list(cs[dim + 1 :])) for cs in shard_shapes]
        for mesh_dim, shard_shapes in input_spec.sharding_shapes().items()
    }

    return dim, new_placements, out_sharding_shapes


def _unbind_dispatch(tensor: ShardTensor, dim: int = 0) -> tuple[ShardTensor, ...]:
    r"""Dispatch handler for ``aten.unbind.int`` on :class:`ShardTensor`.

    Called at the ``__torch_dispatch__`` level (below autograd).  Operates
    directly on the local tensor and constructs output ``ShardTensor``
    objects with the correct metadata; the autograd engine above handles
    gradient tracking.

    Parameters
    ----------
    tensor : ShardTensor
        Input sharded tensor.
    dim : int, default=0
        Dimension along which to unbind.

    Returns
    -------
    tuple[ShardTensor, ...]
        Tuple of ShardTensors, one per slice along ``dim``.

    Note
    ----
    This handler is needed for operations like attention in Stormcast and other
    models that unbind tensors along non-sharded dimensions.
    """
    input_spec = tensor._spec
    dim, new_placements, out_sharding_shapes = _unbind_output_metadata(input_spec, dim)

    # We are reducing tensor rank and returning one tensor per slice
    original_shape = list(input_spec.shape)
    original_shape.pop(dim)

    output_spec = ShardTensorSpec(
        mesh=input_spec.mesh,
        placements=tuple(new_placements),
        tensor_meta=TensorMeta(
            torch.Size(tuple(original_shape)),
            stride=_stride_from_contiguous_shape_C_style(original_shape),
            dtype=input_spec.tensor_meta.dtype,
        ),
        _sharding_shapes={k: tuple(v) for k, v in out_sharding_shapes.items()},
    )

    local_results = aten.unbind.int(tensor._local_tensor, dim)

    return tuple(
        ShardTensor(
            local_result,
            output_spec,
            requires_grad=False,  # Adjusted after the dispatcher
        )
        for local_result in local_results
    )


def unbind_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[ShardTensor, ...]:
    r"""Functional-level wrapper for ``torch.unbind`` on ShardTensor.

    This is a ``__torch_function__``-level intercept (above autograd).  It
    uses ``to_local()`` / ``from_local()`` so that the autograd graph is
    preserved through the unbind operation.

    Parameters
    ----------
    func : Callable
        The original function being wrapped (``torch.unbind`` or
        ``torch.Tensor.unbind``).
    types : tuple[Any, ...]
        Types of the input arguments (unused).
    args : tuple[Any, ...]
        Positional arguments. Expected ``(input,)`` or ``(input, dim)``.
    kwargs : dict[str, Any]
        Keyword arguments (may contain ``dim``).

    Returns
    -------
    tuple[ShardTensor, ...]
        Tuple of ShardTensors, one per slice along the unbind dimension.
    """
    input_tensor: ShardTensor = args[0]
    dim: int = args[1] if len(args) > 1 else kwargs.get("dim", 0)

    input_spec = input_tensor._spec
    dim, new_placements, out_sharding_shapes = _unbind_output_metadata(input_spec, dim)

    # to_local() / from_local() preserve the autograd graph
    local_input = input_tensor.to_local()
    local_results = torch.unbind(local_input, dim)

    return tuple(
        ShardTensor.from_local(
            local_result,
            input_spec.mesh,
            new_placements,
            out_sharding_shapes,
        )
        for local_result in local_results
    )


def _resolve_partial_placements(
    tensor: ShardTensor | DTensor,
) -> ShardTensor | DTensor:
    r"""Redistribute ``Partial`` placements to ``Replicate``, keeping ``Shard``.

    Cross products are bilinear, so a local cross of unreduced partial sums
    is not the partial sum of the cross -- pending reductions must resolve
    before the local math (same treatment as the SDPA wrapper).
    """
    if any(p.is_partial() for p in tensor._spec.placements):
        tensor = tensor.redistribute(
            placements=tuple(
                Replicate() if p.is_partial() else p for p in tensor._spec.placements
            )
        )
    return tensor


def _normalize_cross_dim(
    out_shape: tuple[int, ...], dim: int | None, op_name: str
) -> int:
    r"""Normalize the cross dim to a negative (trailing) offset.

    A negative offset stays valid on every operand and on the broadcast
    output regardless of prepended broadcast dims. ``None`` (``torch.cross``
    semantics) selects the first dimension of size 3 in the global broadcast
    shape.

    Parameters
    ----------
    out_shape : tuple[int, ...]
        Global broadcast shape of the two operands.
    dim : int or None
        Requested dimension, possibly negative or ``None``.
    op_name : str
        Operation name for error messages.

    Returns
    -------
    int
        Negative dimension offset (``-ndim <= offset <= -1``).
    """
    ndim = len(out_shape)
    if dim is None:
        for i, size in enumerate(out_shape):
            if size == 3:
                return i - ndim
        raise RuntimeError(
            f"{op_name} with dim=None requires an input dimension of size 3"
        )
    if not isinstance(dim, int):
        raise TypeError(
            f"{op_name}(): argument 'dim' must be int, not {type(dim).__name__}"
        )
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )
    return dim - ndim if dim >= 0 else dim


def _cross_output_ref(
    input_tensor: Any,
    other_tensor: Any,
    dim_offset: int,
    op_name: str,
) -> Any:
    r"""Validate placements and pick the operand the output layout follows.

    Rules, all against GLOBAL shapes:

    - Neither operand may be sharded on the cross dimension.
    - Placements must be identical, or one operand fully replicated.
    - The reference operand's global shape must equal the broadcast output
      shape, so its placements and shard shapes describe the result exactly.

    Parameters
    ----------
    input_tensor, other_tensor : ShardTensor or DTensor
        The distributed operands.
    dim_offset : int
        Negative (trailing) cross-dimension offset.
    op_name : str
        Operation name for error messages.

    Returns
    -------
    ShardTensor or DTensor
        The operand whose placements/shard shapes describe the output.
    """
    in_spec = input_tensor._spec
    other_spec = other_tensor._spec
    if in_spec.mesh != other_spec.mesh:
        raise RuntimeError(f"{op_name} requires both inputs on the same device mesh")

    out_shape = torch.broadcast_shapes(input_tensor.shape, other_tensor.shape)

    for t in (input_tensor, other_tensor):
        if any(
            p.is_shard() and p.dim == t.ndim + dim_offset for p in t._spec.placements
        ):
            raise RuntimeError(
                f"{op_name} along a sharded dimension is not supported; "
                "gather or reshard first"
            )

    placements_match = in_spec.placements == other_spec.placements
    if not placements_match and not (
        all(p.is_replicate() for p in in_spec.placements)
        or all(p.is_replicate() for p in other_spec.placements)
    ):
        raise RuntimeError(
            f"{op_name} requires identical placements or one fully replicated "
            f"input; got {in_spec.placements} and {other_spec.placements}"
        )

    # Prefer a sharded operand as the reference; it must span the full
    # broadcast output for its shard shapes to describe the result.
    candidates = sorted(
        (input_tensor, other_tensor),
        key=lambda t: not any(p.is_shard() for p in t._spec.placements),
    )
    for t in candidates:
        if tuple(t.shape) == tuple(out_shape):
            return t
    raise RuntimeError(
        f"{op_name}: unsupported broadcast pattern for sharded inputs -- no "
        f"operand spans the broadcast shape {tuple(out_shape)}"
    )


def _cross_wrapper_impl(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    default_dim: int | None,
    op_name: str,
) -> ShardTensor:
    r"""Shared ``__torch_function__`` implementation for the cross variants.

    Computes the cross product locally per shard: with the cross dimension
    unsharded, the product is elementwise over the (possibly sharded) batch
    dimensions. ``to_local`` / ``from_local`` preserve the autograd graph.
    """
    input_tensor = args[0] if len(args) > 0 else kwargs.get("input")
    other_tensor = args[1] if len(args) > 1 else kwargs.get("other")
    dim = args[2] if len(args) > 2 else kwargs.get("dim", default_dim)
    if kwargs.get("out") is not None:
        raise RuntimeError(f"{op_name}(out=...) is not supported for ShardTensor")

    if not isinstance(input_tensor, (ShardTensor, DTensor)) or not isinstance(
        other_tensor, (ShardTensor, DTensor)
    ):
        # Plain tensors are normally promoted before handlers run; reaching
        # here means promotion is disabled and mixed inputs are ambiguous.
        raise RuntimeError(
            f"{op_name} on ShardTensor requires both inputs to be distributed "
            "tensors (enable tensor promotion for plain-tensor operands)"
        )

    input_tensor = _resolve_partial_placements(input_tensor)
    other_tensor = _resolve_partial_placements(other_tensor)

    out_shape = torch.broadcast_shapes(input_tensor.shape, other_tensor.shape)
    dim_offset = _normalize_cross_dim(tuple(out_shape), dim, op_name)
    ref = _cross_output_ref(input_tensor, other_tensor, dim_offset, op_name)

    local_result = torch.linalg.cross(
        input_tensor.to_local(),
        other_tensor.to_local(),
        dim=dim_offset,
    )

    if not any(p.is_shard() for p in ref._spec.placements):
        return ShardTensor.from_local(
            local_result, ref._spec.mesh, ref._spec.placements
        )
    return ShardTensor.from_local(
        local_result,
        ref._spec.mesh,
        ref._spec.placements,
        sharding_shapes=ref._spec.sharding_shapes(),
    )


def linalg_cross_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:
    r"""``__torch_function__`` handler for ``torch.linalg.cross``.

    Parameters
    ----------
    func : Callable
        The intercepted function (unused).
    types : tuple[Any, ...]
        Types of the input arguments (unused).
    args : tuple[Any, ...]
        Positional arguments: ``(input, other)`` and optionally ``dim``.
    kwargs : dict[str, Any]
        Keyword arguments (may contain ``dim`` and ``out``).

    Returns
    -------
    ShardTensor
        Cross product carrying the sharded input's placements.
    """
    return _cross_wrapper_impl(
        args, kwargs or {}, default_dim=-1, op_name="linalg.cross"
    )


def cross_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:
    r"""``__torch_function__`` handler for ``torch.cross`` / ``Tensor.cross``.

    ``torch.cross`` defaults ``dim`` to the first dimension of size 3
    (evaluated on the global shape).

    Parameters
    ----------
    func : Callable
        The intercepted function (unused).
    types : tuple[Any, ...]
        Types of the input arguments (unused).
    args : tuple[Any, ...]
        Positional arguments: ``(input, other)`` and optionally ``dim``.
    kwargs : dict[str, Any]
        Keyword arguments (may contain ``dim`` and ``out``).

    Returns
    -------
    ShardTensor
        Cross product carrying the sharded input's placements.
    """
    return _cross_wrapper_impl(args, kwargs or {}, default_dim=None, op_name="cross")


def _cross_dispatch_impl(
    input_tensor: Any, other_tensor: Any, dim: int | None, op_name: str
) -> ShardTensor:
    r"""Shared ``__torch_dispatch__`` implementation for the aten cross ops.

    Below autograd: operates on raw local tensors and constructs the output
    ShardTensor directly (no ``to_local`` / ``from_local`` autograd bridges;
    the engine above tracks gradients). Partial placements are rejected
    rather than resolved -- collectives are the function-level handler's job.
    """
    for t in (input_tensor, other_tensor):
        if isinstance(t, (ShardTensor, DTensor)) and any(
            p.is_partial() for p in t._spec.placements
        ):
            raise RuntimeError(
                f"{op_name} on a Partial-placement tensor at the dispatch "
                "level is not supported; resolve the pending reduction first"
            )

    if not isinstance(input_tensor, (ShardTensor, DTensor)) or not isinstance(
        other_tensor, (ShardTensor, DTensor)
    ):
        raise RuntimeError(
            f"{op_name} at the dispatch level requires both inputs to be "
            "distributed tensors"
        )

    out_shape = torch.broadcast_shapes(input_tensor.shape, other_tensor.shape)
    dim_offset = _normalize_cross_dim(tuple(out_shape), dim, op_name)
    ref = _cross_output_ref(input_tensor, other_tensor, dim_offset, op_name)

    local_result = aten.linalg_cross.default(
        input_tensor._local_tensor, other_tensor._local_tensor, dim=dim_offset
    )

    ref_spec = ref._spec
    output_spec = ShardTensorSpec(
        mesh=ref_spec.mesh,
        placements=ref_spec.placements,
        tensor_meta=TensorMeta(
            ref_spec.tensor_meta.shape,
            stride=_stride_from_contiguous_shape_C_style(ref_spec.tensor_meta.shape),
            dtype=ref_spec.tensor_meta.dtype,
        ),
        _sharding_shapes=dict(ref_spec.sharding_shapes()),
    )
    return ShardTensor(
        local_result,
        output_spec,
        requires_grad=False,  # Adjusted after the dispatcher
    )


def _linalg_cross_dispatch(
    input_tensor: Any, other_tensor: Any, *, dim: int = -1
) -> ShardTensor:
    r"""Dispatch handler for ``aten.linalg_cross.default``."""
    return _cross_dispatch_impl(input_tensor, other_tensor, dim, "linalg.cross")


def _cross_dispatch(
    input_tensor: Any, other_tensor: Any, dim: int | None = None
) -> ShardTensor:
    r"""Dispatch handler for ``aten.cross.default``."""
    return _cross_dispatch_impl(input_tensor, other_tensor, dim, "cross")


# Python-level function handlers (__torch_function__).
ShardTensor.register_function_handler(torch.unbind, unbind_wrapper)
ShardTensor.register_function_handler(torch.Tensor.unbind, unbind_wrapper)
ShardTensor.register_function_handler(torch.linalg.cross, linalg_cross_wrapper)
ShardTensor.register_function_handler(torch.cross, cross_wrapper)
ShardTensor.register_function_handler(torch.Tensor.cross, cross_wrapper)

# ATen-level dispatch handler (__torch_dispatch__).
ShardTensor.register_dispatch_handler(aten.unbind.int, _unbind_dispatch)
ShardTensor.register_dispatch_handler(aten.linalg_cross.default, _linalg_cross_dispatch)
ShardTensor.register_dispatch_handler(aten.cross.default, _cross_dispatch)
