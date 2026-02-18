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

r"""Sharded view and reshape operations for :class:`ShardTensor`.

This module provides a proper implementation of ``view`` / ``reshape`` for
:class:`ShardTensor`, replacing the inline contiguity hack that previously
lived in :meth:`ShardTensor.__torch_dispatch__`.

Supported overloads:

- **view(*shape)** / **reshape(*shape)** — change shape; shard dimensions
  flow through dimension-group matching.
- **view(dtype)** — reinterpret storage as another dtype (e.g. ``.view(torch.float32)``).
  Same shape when dtypes have the same ``itemsize``, otherwise 1D; matches PyTorch.

The core challenge for shape-changing view/reshape is that they change the
tensor's dimensionality, so we must track how sharded dimensions flow through
the dimension-group matching to compute:

1. The **local target shape** (global target shape with shard dims adjusted).
2. The **new placements** (shard dim index may change after merge/split).
3. The **new sharding shapes** (per-rank local shapes in the new layout).

All of the above are computed locally — no collective communication is required.

Handlers are registered at both the ``__torch_function__`` level (for
``torch.Tensor.view``, ``torch.Tensor.reshape``, ``torch.reshape``, and
``aten.view.default``) and the ``__torch_dispatch__`` level (for
``aten.view.default``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Callable

import torch
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard

from physicsnemo.domain_parallel._shard_tensor_spec import (
    ShardTensorSpec,
    _stride_from_contiguous_shape_C_style,
)
from physicsnemo.domain_parallel.shard_tensor import ShardTensor

aten = torch.ops.aten


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_target_shape(target_shape: Sequence[int], numel: int) -> list[int]:
    r"""Resolve ``-1`` in a target view/reshape shape.

    At most one ``-1`` is allowed (matches PyTorch behavior). The inferred
    dimension is chosen so that the total number of elements is preserved.

    Parameters
    ----------
    target_shape : Sequence[int]
        Target shape, possibly containing a single ``-1``.
    numel : int
        Total number of elements in the tensor.

    Returns
    -------
    list[int]
        Fully-resolved shape with ``-1`` replaced by the inferred size.

    Raises
    ------
    ValueError
        If more than one ``-1`` is present, if any non-inferred dimension
        is 0 (would cause division by zero), or if ``numel`` is not divisible
        by the product of known dimensions.
    """
    result = list(target_shape)
    neg_indices = [i for i, s in enumerate(result) if s == -1]
    if len(neg_indices) > 1:
        raise ValueError(
            f"Only one dimension can be inferred (-1). Got {len(neg_indices)}."
        )
    if -1 in result:
        neg_idx = result.index(-1)
        known = math.prod(s for i, s in enumerate(result) if i != neg_idx)
        if known == 0:
            raise ValueError(
                "Cannot infer dimension when other dimensions have product 0."
            )
        if numel % known != 0:
            raise ValueError(
                f"Shape {tuple(target_shape)} is invalid for tensor with "
                f"{numel} elements (not divisible by product of known dims)."
            )
        result[neg_idx] = numel // known
    return result


def _match_view_dim_groups(
    old_shape: Sequence[int], new_shape: Sequence[int]
) -> list[tuple[list[int], list[int]]]:
    r"""Match contiguous dimension groups between old and new view shapes.

    For a valid ``view``, consecutive dimensions from the old shape can be
    merged into a single dimension in the new shape, or vice versa.  This
    function finds those matching groups by walking both shapes and
    accumulating products until they are equal.

    Parameters
    ----------
    old_shape : Sequence[int]
        Original tensor shape (all positive, no ``-1``). Zero-size dimensions
        are allowed and match when products align (e.g. ``(2, 0)`` and
        ``(1, 0)``).
    new_shape : Sequence[int]
        Target tensor shape (all positive, no ``-1``). Zero-size dimensions
        are allowed.

    Returns
    -------
    list[tuple[list[int], list[int]]]
        List of ``(old_indices, new_indices)`` pairs where the product of
        dimensions in each pair is equal.

    Raises
    ------
    ValueError
        If the shapes are not compatible for a ``view``.
    """
    groups: list[tuple[list[int], list[int]]] = []
    old_i = 0
    new_i = 0

    while old_i < len(old_shape) and new_i < len(new_shape):
        old_start = old_i
        new_start = new_i
        old_prod = old_shape[old_i]
        new_prod = new_shape[new_i]

        while old_prod != new_prod:
            # When one side has product 0 and the other does not, extend the
            # non-zero side until it becomes 0 so both groups can match
            # (zero-size dims are view-compatible, e.g. (2, 0) <-> (1, 0)).
            if new_prod == 0 and old_prod != 0:
                old_i += 1
                if old_i >= len(old_shape):
                    raise ValueError(
                        f"View shapes {tuple(old_shape)} and {tuple(new_shape)} "
                        f"are not compatible"
                    )
                old_prod *= old_shape[old_i]
            elif old_prod == 0 and new_prod != 0:
                new_i += 1
                if new_i >= len(new_shape):
                    raise ValueError(
                        f"View shapes {tuple(old_shape)} and {tuple(new_shape)} "
                        f"are not compatible"
                    )
                new_prod *= new_shape[new_i]
            elif old_prod < new_prod:
                old_i += 1
                if old_i >= len(old_shape):
                    raise ValueError(
                        f"View shapes {tuple(old_shape)} and {tuple(new_shape)} "
                        f"are not compatible"
                    )
                old_prod *= old_shape[old_i]
            else:
                new_i += 1
                if new_i >= len(new_shape):
                    raise ValueError(
                        f"View shapes {tuple(old_shape)} and {tuple(new_shape)} "
                        f"are not compatible"
                    )
                new_prod *= new_shape[new_i]

        groups.append(
            (list(range(old_start, old_i + 1)), list(range(new_start, new_i + 1)))
        )
        old_i += 1
        new_i += 1

    return groups


def _find_shard_in_new_dims(
    old_dims: list[int],
    new_dims: list[int],
    global_old: Sequence[int],
    local_old: Sequence[int],
    global_new: Sequence[int],
) -> tuple[int, int]:
    r"""Find which new dimension inherits the shard after a view.

    In C-contiguous layout, sharding on an old dimension gives each rank a
    contiguous chunk of ``chunk_size`` elements within the group.  After the
    view, we walk the new dims from **left to right** (outermost first) and
    find the dimension where the chunk boundary falls: i.e. the leftmost new
    dim ``nd`` such that ``chunk_size`` is divisible by the product of all
    new dims to the right of ``nd``, and the quotient divides ``global_new[nd]``.

    Parameters
    ----------
    old_dims : list[int]
        Indices of old dimensions in this group (into ``global_old``).
    new_dims : list[int]
        Indices of new dimensions in this group (into ``global_new``).
    global_old : Sequence[int]
        Global shape before view.
    local_old : Sequence[int]
        Local shape before view (on this rank).
    global_new : Sequence[int]
        Global shape after view.

    Returns
    -------
    tuple[int, int]
        ``(new_dim_index, local_size)`` where ``new_dim_index`` is the
        absolute index into ``global_new`` of the new shard dimension, and
        ``local_size`` is its local extent on this rank.

    Raises
    ------
    ValueError
        If no valid new shard dimension can be found.
    """
    chunk_size = math.prod(local_old[d] for d in old_dims)

    # Zero-size group: shard maps to first new dim with local size 0.
    if chunk_size == 0:
        return new_dims[0], 0

    # Walk new dims left-to-right, tracking the suffix product.
    suffix_prods: list[int] = [0] * len(new_dims)
    sp = 1
    for i in range(len(new_dims) - 1, -1, -1):
        suffix_prods[i] = sp
        sp *= global_new[new_dims[i]]

    for i, nd in enumerate(new_dims):
        right_prod = suffix_prods[i]  # product of new dims to the RIGHT of nd
        if right_prod == 0:
            continue
        if chunk_size % right_prod != 0:
            continue
        local_part = chunk_size // right_prod
        if 0 < local_part <= global_new[nd]:
            return nd, local_part

    raise ValueError(
        f"Cannot view sharded tensor: unable to find a valid new shard "
        f"dimension.  chunk_size={chunk_size}, "
        f"Global: {tuple(global_old)} -> {tuple(global_new)}, "
        f"Local: {tuple(local_old)}"
    )


def _compute_local_view_shape(
    global_old: Sequence[int],
    local_old: Sequence[int],
    global_new: Sequence[int],
    placements: tuple[Placement, ...],
) -> list[int]:
    r"""Compute the local target shape for a view/reshape on a ShardTensor.

    Maps a global target shape to the corresponding local shape by tracking
    how sharded dimensions flow through the dimension-group matching.

    For each group that contains a sharded old dimension, uses
    :func:`_find_shard_in_new_dims` to locate the new dimension that absorbs
    the shard and compute its local extent.

    Parameters
    ----------
    global_old : Sequence[int]
        Global shape of the input tensor.
    local_old : Sequence[int]
        Local shape of the input tensor on this rank.
    global_new : Sequence[int]
        Global target shape (fully resolved, no ``-1``).
    placements : tuple[Placement, ...]
        Current placement specifications.

    Returns
    -------
    list[int]
        Local target shape.

    Raises
    ------
    ValueError
        If the sharded group cannot be mapped to the new shape.
    """
    shard_dims = {p.dim for p in placements if isinstance(p, Shard)}

    if not shard_dims:
        # No sharding — local == global.
        return list(global_new)

    groups = _match_view_dim_groups(list(global_old), list(global_new))
    local_target = list(global_new)

    for old_dims, new_dims in groups:
        if not any(d in shard_dims for d in old_dims):
            continue

        new_shard_dim, local_size = _find_shard_in_new_dims(
            old_dims, new_dims, global_old, local_old, global_new
        )
        local_target[new_shard_dim] = local_size

    return local_target


def _compute_view_placements(
    global_old: Sequence[int],
    local_old: Sequence[int],
    global_new: Sequence[int],
    placements: tuple[Placement, ...],
) -> tuple[Placement, ...]:
    r"""Determine new placements after a view operation.

    Tracks where each sharded tensor dimension ends up in the new shape
    using :func:`_find_shard_in_new_dims` to correctly identify the new
    shard dimension (which may not be the outermost dim in the group).

    Parameters
    ----------
    global_old : Sequence[int]
        Global shape before view.
    local_old : Sequence[int]
        Local shape on this rank before view.
    global_new : Sequence[int]
        Global shape after view (fully resolved).
    placements : tuple[Placement, ...]
        Input placements.

    Returns
    -------
    tuple[Placement, ...]
        Updated placements reflecting the new dimension layout.
    """
    shard_to_mesh: dict[int, int] = {}
    for mesh_dim, p in enumerate(placements):
        if isinstance(p, Shard):
            shard_to_mesh[p.dim] = mesh_dim

    if not shard_to_mesh:
        return placements

    groups = _match_view_dim_groups(list(global_old), list(global_new))
    new_placements = list(placements)

    for old_dims, new_dims in groups:
        sharded_in_group = [d for d in old_dims if d in shard_to_mesh]
        if not sharded_in_group:
            continue
        new_shard_dim, _ = _find_shard_in_new_dims(
            old_dims, new_dims, global_old, local_old, global_new
        )
        for d in sharded_in_group:
            new_placements[shard_to_mesh[d]] = Shard(new_shard_dim)

    return tuple(new_placements)


def _compute_view_sharding_shapes(
    old_sharding_shapes: dict[int, tuple[torch.Size, ...]] | None,
    global_old: Sequence[int],
    global_new: Sequence[int],
    placements: tuple[Placement, ...],
) -> dict[int, tuple[torch.Size, ...]] | None:
    r"""Compute new per-rank sharding shapes after a view.

    For each rank, maps its old local shape to the new local shape using the
    same dimension-group logic as :func:`_compute_local_view_shape`.  No
    collective communication is required.

    Parameters
    ----------
    old_sharding_shapes : dict[int, tuple[torch.Size, ...]] or None
        Old sharding shapes from the input spec.
    global_old : Sequence[int]
        Global shape before view.
    global_new : Sequence[int]
        Global shape after view (fully resolved).
    placements : tuple[Placement, ...]
        Input placements.

    Returns
    -------
    dict[int, tuple[torch.Size, ...]] or None
        New sharding shapes, or ``None`` if input was ``None``.
    """
    if old_sharding_shapes is None:
        return None

    new_sharding: dict[int, tuple[torch.Size, ...]] = {}
    for mesh_dim, rank_shapes in old_sharding_shapes.items():
        new_rank_shapes: list[torch.Size] = []
        for rank_shape in rank_shapes:
            new_shape = _compute_local_view_shape(
                global_old, tuple(rank_shape), global_new, placements
            )
            new_rank_shapes.append(torch.Size(new_shape))
        new_sharding[mesh_dim] = tuple(new_rank_shapes)

    return new_sharding


# ---------------------------------------------------------------------------
# Core forward implementation (shared by autograd function + dispatch handler)
# ---------------------------------------------------------------------------


def _sharded_view_forward(
    tensor: ShardTensor, target_shape: Sequence[int]
) -> ShardTensor:
    r"""Core view/reshape implementation for :class:`ShardTensor`.

    Makes the local tensor contiguous, computes the local target shape, and
    constructs a new :class:`ShardTensor` with updated metadata.  All shape
    information is derived locally — no collective communication is needed.

    Parameters
    ----------
    tensor : ShardTensor
        Input sharded tensor.
    target_shape : Sequence[int]
        Global target shape (may contain ``-1``).

    Returns
    -------
    ShardTensor
        Viewed/reshaped ShardTensor.
    """
    spec = tensor._spec
    local_tensor = tensor._local_tensor

    global_old = tuple(tensor.shape)
    local_old = tuple(local_tensor.shape)

    # Resolve -1 using global element count.
    global_new = _resolve_target_shape(target_shape, math.prod(global_old))

    # Compute local target shape.
    local_new = _compute_local_view_shape(
        global_old, local_old, global_new, spec.placements
    )

    # Make contiguous and reshape.  Ensure result is contiguous so downstream
    # ops (e.g. F.linear) that call .view() on the local tensor do not fail.
    if not local_tensor.is_contiguous():
        local_tensor = local_tensor.contiguous()
    local_result = local_tensor.reshape(local_new).contiguous()

    # Compute new placements and sharding shapes.
    new_placements = _compute_view_placements(
        global_old, local_old, global_new, spec.placements
    )
    new_sharding = _compute_view_sharding_shapes(
        spec._sharding_shapes, global_old, global_new, spec.placements
    )

    # Build new spec — no communication required.
    new_stride = _stride_from_contiguous_shape_C_style(tuple(global_new))
    new_meta = TensorMeta(
        shape=torch.Size(global_new),
        stride=new_stride,
        dtype=spec.tensor_meta.dtype,
    )
    output_spec = ShardTensorSpec(
        mesh=spec.mesh,
        placements=new_placements,
        tensor_meta=new_meta,
        _local_shape=local_result.shape,
        _sharding_shapes=new_sharding,
    )

    return ShardTensor(local_result, output_spec, requires_grad=False)


def _sharded_view_dtype(tensor: ShardTensor, dtype: torch.dtype) -> ShardTensor:
    r"""Reinterpret sharded tensor storage as a different dtype (view(dtype)).

    Applies ``view(dtype)`` locally on each shard. When the old and new dtypes
    have the same ``itemsize``, the shape and placements are preserved (matches
    PyTorch). When they differ, the result is 1D with size equal to
    ``total_bytes // dtype.itemsize``.

    Parameters
    ----------
    tensor : ShardTensor
        Input sharded tensor.
    dtype : torch.dtype
        Target dtype to reinterpret the storage as.

    Returns
    -------
    ShardTensor
        ShardTensor with the given dtype (view of the same storage); same
        shape as input when itemsizes match, otherwise 1D.

    Raises
    ------
    RuntimeError
        If the tensor's byte size is not divisible by ``dtype.itemsize``
        (same condition as PyTorch's view(dtype)).
    """
    spec = tensor._spec
    local_tensor = tensor._local_tensor
    old_dtype = spec.tensor_meta.dtype
    old_global_shape = spec.tensor_meta.shape
    old_global_numel = math.prod(old_global_shape)
    total_bytes = old_global_numel * old_dtype.itemsize
    if total_bytes % dtype.itemsize != 0:
        raise RuntimeError(
            f"view(dtype) requires tensor byte size ({total_bytes}) to be "
            f"divisible by {dtype.itemsize} (dtype {dtype})"
        )
    new_global_numel = total_bytes // dtype.itemsize

    if not local_tensor.is_contiguous():
        local_tensor = local_tensor.contiguous()
    local_result = local_tensor.view(dtype)

    if old_dtype.itemsize == dtype.itemsize:
        # Same itemsize: preserve shape and placements (PyTorch behavior).
        new_global_shape = old_global_shape
        new_stride = _stride_from_contiguous_shape_C_style(tuple(old_global_shape))
        new_placements = spec.placements
        new_sharding = spec._sharding_shapes
    else:
        # Different itemsize: result is 1D.
        new_global_shape = (new_global_numel,)
        new_stride = (1,)
        new_placements = tuple(
            Shard(0) if p.is_shard() else Replicate() for p in spec.placements
        )
        new_sharding = None

    new_meta = TensorMeta(
        shape=torch.Size(new_global_shape),
        stride=new_stride,
        dtype=dtype,
    )
    output_spec = ShardTensorSpec(
        mesh=spec.mesh,
        placements=new_placements,
        tensor_meta=new_meta,
        _local_shape=local_result.shape,
        _sharding_shapes=new_sharding,
    )
    return ShardTensor(local_result, output_spec, requires_grad=False)


# ---------------------------------------------------------------------------
# Autograd function (for __torch_function__ path)
# ---------------------------------------------------------------------------


class ShardedView(torch.autograd.Function):
    r"""Autograd function for differentiable view/reshape on :class:`ShardTensor`.

    Forward maps the global target shape to a local target shape, views the
    local tensor, and constructs a new :class:`ShardTensor`.  Backward is
    simply a view back to the original global shape.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: ShardTensor,
        target_shape: tuple[int, ...],
    ) -> ShardTensor:
        r"""View a ShardTensor to a new global shape.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context for saving state for backward.
        tensor : ShardTensor
            Input sharded tensor.
        target_shape : tuple[int, ...]
            Global target shape (may contain ``-1``).

        Returns
        -------
        ShardTensor
            Viewed ShardTensor.
        """
        ctx.input_global_shape = tuple(tensor.shape)
        return _sharded_view_forward(tensor, target_shape)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: ShardTensor,
    ) -> tuple[ShardTensor, None]:
        r"""View gradient back to the original global shape.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved state from forward.
        grad_output : ShardTensor
            Gradient with respect to the viewed output.

        Returns
        -------
        tuple[ShardTensor, None]
            Gradient for the input tensor, and ``None`` for ``target_shape``.
        """
        return (
            _sharded_view_forward(grad_output, ctx.input_global_shape),
            None,
        )


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


def sharded_view(tensor: ShardTensor, target_shape: Sequence[int]) -> ShardTensor:
    r"""Differentiable view/reshape for :class:`ShardTensor`.

    Implements a view when possible (no copy); the output is made contiguous
    for downstream ops. At most one ``-1`` is allowed in ``target_shape``
    (inferred so that the total number of elements is preserved).

    Parameters
    ----------
    tensor : ShardTensor
        Input tensor.
    target_shape : Sequence[int]
        Target global shape (may contain a single ``-1`` for inference).

    Returns
    -------
    ShardTensor
        Viewed/reshaped ShardTensor.

    Examples
    --------
    1D sharded on dim 0, view to 2D: ``st.view(2, -1)`` or
    ``sharded_view(st, (2, -1))``; the shard flows to the first new dimension.
    """
    return ShardedView.apply(tensor, tuple(target_shape))


# ---------------------------------------------------------------------------
# __torch_function__ handlers: argument repackaging
# ---------------------------------------------------------------------------


def _reshape_args(*shape_args: Any) -> tuple[int, ...]:
    r"""Normalize shape arguments to a single tuple of ints.

    Handles both a single sequence (e.g. ``(2, 3, 4)``) and variadic ints
    (e.g. ``2, 3, 4``) as used by ``Tensor.view`` and ``Tensor.reshape``.
    """
    if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list, torch.Size)):
        return tuple(shape_args[0])
    return tuple(shape_args)


def extract_view_and_reshape_arguments(
    *args: Any, **kwargs: Any
) -> tuple[
    ShardTensor,
    tuple[int, ...] | None,
    torch.dtype | None,
]:
    r"""Extract (tensor, shape, dtype) from view/reshape __torch_function__ args.

    Used by Tensor.view, Tensor.reshape, torch.reshape, and aten.view.default.
    For view(dtype), returns (tensor, None, dtype). Otherwise returns
    (tensor, shape, None) with shape normalized to tuple[int, ...].
    """
    tensor = args[0]
    # If there is a dtype, catch and exit early:
    if len(args) == 2 and isinstance(args[1], torch.dtype):
        # Honestly this execution path makes no sense to me ...
        return (tensor, None, args[1])
    # If it's in kwargs, use that:
    shape = kwargs.get("shape", None)
    if shape is not None:
        return (tensor, shape, None)
    # Otherwise, all remaning args get massaged into a tuple:
    shape = _reshape_args(*args[1:])
    return (tensor, shape, None)


# ---------------------------------------------------------------------------
# __torch_function__ handlers
# ---------------------------------------------------------------------------


def view_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:
    r"""``__torch_function__`` handler for ``torch.Tensor.view``."""
    tensor, shape, dtype = extract_view_and_reshape_arguments(*args, **kwargs)
    if dtype is not None:
        return _sharded_view_dtype(tensor, dtype)
    if shape is None:
        raise ValueError(
            "ShardTensor.view_wrapper: Shape is required for view operation"
        )
    return sharded_view(tensor, shape)


def reshape_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:
    r"""``__torch_function__`` handler for ``torch.Tensor.reshape``."""
    tensor, shape, dtype = extract_view_and_reshape_arguments(*args, **kwargs)
    if dtype is not None:
        raise ValueError(
            "ShardTensor.reshape_wrapper: Dtype is not supported for reshape operation"
        )
    if shape is None:
        raise ValueError(
            "ShardTensor.reshape_wrapper: Shape is required for reshape operation"
        )
    return sharded_view(tensor, shape)


def torch_reshape_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:
    r"""``__torch_function__`` handler for ``torch.reshape``."""
    tensor, shape, _ = extract_view_and_reshape_arguments(*args, **kwargs)
    if shape is None:
        raise ValueError(
            "ShardTensor.torch_reshape_wrapper: Shape is required for reshape operation"
        )
    return sharded_view(tensor, shape)


# ---------------------------------------------------------------------------
# __torch_dispatch__ handler
# ---------------------------------------------------------------------------


def _sharded_view_dispatch(
    tensor: ShardTensor, target_shape: Sequence[int]
) -> ShardTensor:
    r"""Dispatch handler for ``aten.view.default`` on :class:`ShardTensor`.

    Called at the ``__torch_dispatch__`` level.  Autograd wraps above this
    level, so no autograd function is needed here.

    Parameters
    ----------
    tensor : ShardTensor
        Input sharded tensor.
    target_shape : Sequence[int]
        Global target shape.

    Returns
    -------
    ShardTensor
        Viewed ShardTensor.
    """
    return _sharded_view_forward(tensor, target_shape)


def aten_view_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:
    r"""``__torch_function__`` handler for ``aten.view.default``.

    In PyTorch 2.6+, some internal codepaths (including DTensor's processing
    of higher-level ops like ``F.linear``) call ``aten.view.default`` directly
    on tensor subclasses, which triggers ``__torch_function__`` with the ATen
    op as ``func``.  This handler catches those calls before DTensor's
    sharding propagator rejects them.

    Parameters
    ----------
    func : Callable
        The ``aten.view.default`` op.
    types : tuple[Any, ...]
        Tensor subclass types involved.
    args : tuple[Any, ...]
        Positional args ``(tensor, shape)``.
    kwargs : dict[str, Any]
        Keyword args (unused).

    Returns
    -------
    ShardTensor
        Viewed ShardTensor.
    """
    tensor, shape, _ = extract_view_and_reshape_arguments(*args, **kwargs)
    if shape is None:
        raise ValueError(
            "ShardTensor.aten_view_wrapper: Shape is required for view operation"
        )
    return sharded_view(tensor, shape)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

# Python-level function handlers (__torch_function__).
ShardTensor.register_function_handler(torch.Tensor.view, view_wrapper)
ShardTensor.register_function_handler(torch.Tensor.reshape, reshape_wrapper)
ShardTensor.register_function_handler(torch.reshape, torch_reshape_wrapper)

# ATen ops can also arrive at __torch_function__ (not just __torch_dispatch__)
# when internal PyTorch code calls them directly on a tensor subclass.
ShardTensor.register_function_handler(aten.view.default, aten_view_wrapper)
ShardTensor.register_function_handler(aten.reshape.default, aten_view_wrapper)

# ATen-level dispatch handler (__torch_dispatch__).
ShardTensor.register_dispatch_handler(aten.view.default, _sharded_view_dispatch)
ShardTensor.register_dispatch_handler(aten.reshape.default, _sharded_view_dispatch)
