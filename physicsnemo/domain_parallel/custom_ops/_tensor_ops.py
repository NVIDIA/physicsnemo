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
import torch.distributed._functional_collectives as funcol
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
from physicsnemo.domain_parallel.custom_ops._reductions import (
    create_sharded_grad_input,
    resolve_partial_cotangent,
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
    out_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
    dim: int | None,
    op_name: str,
) -> int:
    r"""Normalize the cross dim to a negative (trailing) offset.

    A negative offset stays valid on every operand and on the broadcast
    output regardless of prepended broadcast dims. ``None`` (``torch.cross``
    semantics) selects the first dimension of size 3 in the FIRST INPUT's
    global shape -- native PyTorch scans the input, not the broadcast
    output, and the two disagree when broadcasting prepends a size-3 dim.

    Parameters
    ----------
    out_shape : tuple[int, ...]
        Global broadcast shape of the two operands.
    input_shape : tuple[int, ...]
        Global shape of the first operand; scanned for the default dim.
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
        for i, size in enumerate(input_shape):
            if size == 3:
                return i - len(input_shape)
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


# ---------------------------------------------------------------------------
# Cross products (torch.cross / Tensor.cross / torch.linalg.cross), in the
# _reductions.py idiom: metadata prep + local aten op + direct ShardTensor
# construction, with a hand-written backward. Plain-tensor operands stay
# plain (wrapper construction inside a handler does not survive dynamo fake
# propagation); their layout is implicitly Replicate. ShardTensor subclasses
# torch.Tensor, NOT DTensor, so distributed checks must name both types.
# ---------------------------------------------------------------------------


class _CrossPrep:
    r"""Per-call cross-product metadata; recomputed deterministically in
    ``forward`` and ``setup_context`` (which cannot share intermediates)."""

    __slots__ = (
        "mesh",
        "out_shape",
        "dim_offset",
        "specs",
        "replicated",
        "ref_spec",
        "ref_placements",
        "ref_sharded",
        "locals_",
        "full_local_shapes",
        "slices",
    )


def _cross_pick_ref(
    shapes: tuple,
    placements: tuple,
    out_shape: tuple,
    dim_offset: int,
    op_name: str,
) -> int:
    r"""Validate placements and pick the index of the reference operand.

    Against global shapes: no sharding on the cross dim; placements
    identical or one operand fully replicated; the reference (sharded
    preferred) must span the broadcast output shape.

    Parameters
    ----------
    shapes : tuple
        Global shape of each operand (a plain tensor's shape is global).
    placements : tuple
        Placements of each operand (plain tensors: all ``Replicate``).
    out_shape : tuple
        Broadcast output shape.
    dim_offset : int
        Negative (trailing) cross-dimension offset.
    op_name : str
        Operation name for error messages.

    Returns
    -------
    int
        Index (0 or 1) of the reference operand.

    Raises
    ------
    RuntimeError
        If the layout combination is unsupported.
    """
    for shape, plc in zip(shapes, placements):
        if any(p.is_shard() and p.dim == len(shape) + dim_offset for p in plc):
            raise RuntimeError(
                f"{op_name} along a sharded dimension is not supported; "
                "gather or reshard first"
            )

    if placements[0] != placements[1] and not (
        all(p.is_replicate() for p in placements[0])
        or all(p.is_replicate() for p in placements[1])
    ):
        raise RuntimeError(
            f"{op_name} requires identical placements or one fully replicated "
            f"input; got {placements[0]} and {placements[1]}"
        )

    order = sorted((0, 1), key=lambda i: not any(p.is_shard() for p in placements[i]))
    for i in order:
        if shapes[i] == out_shape:
            return i
    raise RuntimeError(
        f"{op_name}: unsupported broadcast pattern for sharded inputs -- no "
        f"operand spans the broadcast shape {out_shape}"
    )


def _replicated_shard_slices(
    local_shape: tuple, ref_spec: ShardTensorSpec
) -> list[tuple[int, int, int]]:
    r"""Compute slices localizing a replicated local tensor to the ref shard.

    Broadcast dims get no slice; offsets come from the reference's shard
    shapes (uneven-safe). No communication.

    Parameters
    ----------
    local_shape : tuple
        Shape of the replicated operand's local tensor.
    ref_spec : ShardTensorSpec
        Spec of the reference operand whose shard defines the slices.

    Returns
    -------
    list[tuple[int, int, int]]
        ``(dim, offset, length)`` per sharded dim the operand spans.
    """
    slices: list[tuple[int, int, int]] = []
    coords = ref_spec.mesh.get_coordinate()
    if coords is None:
        return slices
    ref_shape = tuple(ref_spec.tensor_meta.shape)
    ndim_gap = len(ref_shape) - len(local_shape)
    for mesh_dim, placement in enumerate(ref_spec.placements):
        if not placement.is_shard():
            continue
        local_dim = placement.dim - ndim_gap
        if local_dim < 0 or local_shape[local_dim] != ref_shape[placement.dim]:
            continue
        sizes = [s[placement.dim] for s in ref_spec.sharding_shapes()[mesh_dim]]
        slices.append(
            (local_dim, sum(sizes[: coords[mesh_dim]]), sizes[coords[mesh_dim]])
        )
    return slices


def _cross_prepare(
    input_tensor: Any, other_tensor: Any, dim: int | None, op_name: str
) -> _CrossPrep:
    r"""Resolve metadata, the cross dim, and localized locals for a cross.

    Reads only ``_spec`` / ``_local_tensor`` from distributed operands and
    calls tensor methods only on raw locals, so it never re-enters dispatch.

    Parameters
    ----------
    input_tensor, other_tensor : ShardTensor, DTensor, or torch.Tensor
        The operands; plain tensors are implicitly replicated.
    dim : int or None
        Raw cross dim argument, normalized against global shapes.
    op_name : str
        Operation name for error messages.

    Returns
    -------
    _CrossPrep
        Metadata plus localized local tensors.
    """
    tensors = (input_tensor, other_tensor)
    specs = []
    mesh = None
    for t in tensors:
        if isinstance(t, (ShardTensor, DTensor)):
            spec = t._spec
            if mesh is None:
                mesh = spec.mesh
            elif mesh != spec.mesh:
                raise RuntimeError(
                    f"{op_name} requires both inputs on the same device mesh"
                )
            specs.append(spec)
        else:
            specs.append(None)
    if mesh is None:
        raise RuntimeError(f"{op_name} requires at least one distributed input")

    shapes = tuple(
        tuple(spec.tensor_meta.shape) if spec is not None else tuple(t.shape)
        for spec, t in zip(specs, tensors)
    )
    placements = tuple(
        tuple(spec.placements) if spec is not None else (Replicate(),) * mesh.ndim
        for spec in specs
    )
    out_shape = tuple(torch.broadcast_shapes(*shapes))
    dim_offset = _normalize_cross_dim(out_shape, shapes[0], dim, op_name)

    ref_index = _cross_pick_ref(shapes, placements, out_shape, dim_offset, op_name)

    prep = _CrossPrep()
    prep.mesh = mesh
    prep.out_shape = out_shape
    prep.dim_offset = dim_offset
    prep.specs = specs
    prep.replicated = [all(p.is_replicate() for p in plc) for plc in placements]
    prep.ref_spec = specs[ref_index]
    prep.ref_placements = placements[ref_index]
    prep.ref_sharded = any(p.is_shard() for p in prep.ref_placements)
    prep.locals_ = [
        t._local_tensor if spec is not None else t for spec, t in zip(specs, tensors)
    ]
    prep.full_local_shapes = [tuple(local.shape) for local in prep.locals_]
    prep.slices = [[], []]
    if prep.ref_sharded:
        # Localize replicated operands to this rank's shard of the reference.
        for i in range(2):
            if prep.replicated[i]:
                prep.slices[i] = _replicated_shard_slices(
                    prep.full_local_shapes[i], prep.ref_spec
                )
                for dim, offset, length in prep.slices[i]:
                    prep.locals_[i] = prep.locals_[i].narrow(dim, offset, length)
    return prep


def _build_cross_output(
    local_result: torch.Tensor, prep: _CrossPrep, requires_grad: bool
) -> ShardTensor:
    r"""Construct the output ShardTensor directly from reference metadata.

    Explicit spec + ``__new__``; no ``from_local`` autograd side effects.

    Parameters
    ----------
    local_result : torch.Tensor
        Locally computed cross product.
    prep : _CrossPrep
        Metadata from :func:`_cross_prepare`.
    requires_grad : bool
        Forwarded to the ShardTensor constructor.

    Returns
    -------
    ShardTensor
        Result carrying the reference operand's layout.
    """
    if prep.ref_sharded:
        placements = tuple(prep.ref_placements)
        sharding_shapes = {
            k: tuple(tuple(s) for s in v)
            for k, v in prep.ref_spec.sharding_shapes().items()
        }
    else:
        placements = (Replicate(),) * prep.mesh.ndim
        sharding_shapes = {}

    spec = ShardTensorSpec(
        mesh=prep.mesh,
        placements=placements,
        tensor_meta=TensorMeta(
            shape=tuple(prep.out_shape),
            stride=_stride_from_contiguous_shape_C_style(prep.out_shape),
            dtype=local_result.dtype,
        ),
        _local_shape=local_result.shape,
        _sharding_shapes=sharding_shapes,
    )
    return ShardTensor.__new__(
        ShardTensor,
        local_tensor=local_result,
        spec=spec,
        requires_grad=requires_grad,
    )


def _sum_grad_to_shape(grad: torch.Tensor, shape: tuple) -> torch.Tensor:
    r"""Standard broadcast gradient reduction: sum ``grad`` down to ``shape``."""
    while grad.ndim > len(shape):
        grad = grad.sum(dim=0)
    for i, size in enumerate(shape):
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(dim=i, keepdim=True)
    return grad


def _assemble_cross_grad(
    grad_local: torch.Tensor, i: int, ctx: Any
) -> torch.Tensor | ShardTensor:
    r"""Assemble one operand's gradient from the local cross-product gradient.

    Broadcast-reduce, zero-pad any forward slices, all-reduce a replicated
    operand's partial sums, then wrap.

    Parameters
    ----------
    grad_local : torch.Tensor
        Local gradient from the backward cross product.
    i : int
        Operand index (0 or 1).
    ctx : Any
        Autograd context saved by ``setup_context``.

    Returns
    -------
    torch.Tensor or ShardTensor
        Plain gradient for plain inputs; ShardTensor otherwise.
    """
    grad_local = _sum_grad_to_shape(grad_local, tuple(ctx.saved_tensors[i].shape))

    for dim, offset, length in reversed(ctx.slices[i]):
        padded_shape = list(grad_local.shape)
        padded_shape[dim] = ctx.full_local_shapes[i][dim]
        padded = grad_local.new_zeros(padded_shape)
        padded.narrow(dim, offset, length).copy_(grad_local)
        grad_local = padded

    if ctx.replicated[i] and ctx.ref_sharded:
        # Rank-local partial sum; funcol keeps the AOT-captured backward
        # graph deepcopy-safe (see shard_utils/grad_ops.py).
        for mesh_dim, placement in enumerate(ctx.ref_placements):
            if placement.is_shard():
                grad_local = funcol.all_reduce(grad_local, "sum", (ctx.mesh, mesh_dim))
        if isinstance(grad_local, funcol.AsyncCollectiveTensor):
            grad_local = grad_local.wait()

    if ctx.specs[i] is None:
        return grad_local
    return create_sharded_grad_input(grad_local, ctx.specs[i])


class ShardedCross(torch.autograd.Function):
    r"""Custom autograd function for cross products on ShardTensor.

    With the cross dim unsharded, forward is elementwise over the batch
    dims and runs locally per shard. Backward uses bilinearity:
    ``<g, da x b> = da . (b x g)`` and ``<g, a x db> = db . (g x a)``.
    """

    @staticmethod
    def forward(
        input_tensor: Any, other_tensor: Any, dim: int | None, op_name: str
    ) -> ShardTensor:
        r"""Local cross product plus direct output construction.

        Shielded like ``ShardedSum.forward`` so metadata accesses on
        ShardTensor inputs cannot re-enter ``__torch_function__``.

        Parameters
        ----------
        input_tensor, other_tensor : ShardTensor, DTensor, or torch.Tensor
            The operands; plain tensors are implicitly replicated.
        dim : int or None
            Raw cross dim argument.
        op_name : str
            Operation name for error messages.

        Returns
        -------
        ShardTensor
            Cross product carrying the reference operand's layout.
        """
        with torch._C.DisableTorchFunctionSubclass():
            prep = _cross_prepare(input_tensor, other_tensor, dim, op_name)
            local_result = aten.linalg_cross.default(
                prep.locals_[0], prep.locals_[1], dim=prep.dim_offset
            )
            requires_grad = bool(
                input_tensor.requires_grad or other_tensor.requires_grad
            )
            return _build_cross_output(local_result, prep, requires_grad)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        r"""Recompute the (deterministic) prep and save it for backward."""
        input_tensor, other_tensor, dim, op_name = inputs
        with torch._C.DisableTorchFunctionSubclass():
            prep = _cross_prepare(input_tensor, other_tensor, dim, op_name)
            ctx.save_for_backward(prep.locals_[0], prep.locals_[1])
            ctx.dim_offset = prep.dim_offset
            ctx.specs = prep.specs
            ctx.replicated = prep.replicated
            ctx.slices = prep.slices
            ctx.full_local_shapes = prep.full_local_shapes
            ctx.ref_placements = prep.ref_placements
            ctx.ref_sharded = prep.ref_sharded
            ctx.mesh = prep.mesh

    @staticmethod
    def backward(ctx, grad_output):
        r"""Hand-written backward: local cross gradients + explicit reduction.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context saved by ``setup_context``.
        grad_output : ShardTensor, DTensor, or torch.Tensor
            Cotangent of the output.

        Returns
        -------
        tuple
            Gradients for ``(input, other)`` and ``None`` for dim/op_name.
        """
        dim_offset = ctx.dim_offset
        local_input, local_other = ctx.saved_tensors

        # Resolve a Partial cotangent before the local math (cross is bilinear).
        if isinstance(grad_output, ShardTensor):
            grad_output = resolve_partial_cotangent(grad_output)
            local_grad = grad_output._local_tensor
        elif isinstance(grad_output, DTensor):
            local_grad = grad_output._local_tensor
        else:
            local_grad = grad_output

        grads = [None, None]
        cross_args = (
            (local_other, local_grad),  # <g, da x b> = da . (b x g)
            (local_grad, local_input),  # <g, a x db> = db . (g x a)
        )
        for i in range(2):
            if not ctx.needs_input_grad[i]:
                continue
            grad_local = aten.linalg_cross.default(*cross_args[i], dim=dim_offset)
            grads[i] = _assemble_cross_grad(grad_local, i, ctx)
        return grads[0], grads[1], None, None


def _cross_wrapper_impl(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    default_dim: int | None,
    op_name: str,
) -> ShardTensor:
    r"""Shared ``__torch_function__`` implementation for the cross variants.

    Unpack args, normalize DTensors to ShardTensors, resolve Partials, and
    delegate to ``ShardedCross.apply``. Plain tensors pass through untouched.

    Parameters
    ----------
    args : tuple
        Positional arguments: ``(input, other)`` and optionally ``dim``.
    kwargs : dict
        Keyword arguments (may contain ``dim`` and ``out``).
    default_dim : int or None
        Default dim of the intercepted op (``-1`` for linalg.cross).
    op_name : str
        Operation name for error messages.

    Returns
    -------
    ShardTensor
        Cross product carrying the reference operand's layout.
    """
    input_tensor = args[0] if len(args) > 0 else kwargs.get("input")
    other_tensor = args[1] if len(args) > 1 else kwargs.get("other")
    dim = args[2] if len(args) > 2 else kwargs.get("dim", default_dim)
    if kwargs.get("out") is not None:
        raise RuntimeError(f"{op_name}(out=...) is not supported for ShardTensor")
    if not isinstance(input_tensor, torch.Tensor) or not isinstance(
        other_tensor, torch.Tensor
    ):
        raise RuntimeError(f"{op_name} on ShardTensor requires tensor inputs")

    def normalize(t):
        if not isinstance(t, (ShardTensor, DTensor)):
            return t  # plain tensor: implicitly replicated, handled natively
        if (
            not isinstance(t, ShardTensor)
            and t.requires_grad
            and t.grad_fn is None
            and torch.is_grad_enabled()
        ):
            # Graph-connect a leaf DTensor so from_dtensor takes its
            # differentiable bridge; the plain path would deposit the
            # gradient on the throwaway conversion wrapper.
            t = t.view_as(t)
        return _resolve_partial_placements(ShardTensor.from_dtensor(t))

    input_tensor = normalize(input_tensor)
    other_tensor = normalize(other_tensor)

    return ShardedCross.apply(input_tensor, other_tensor, dim, op_name)


def linalg_cross_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:
    r"""``__torch_function__`` handler for ``torch.linalg.cross``."""
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

    ``torch.cross`` defaults ``dim`` to the first dimension of size 3,
    evaluated on the first input's global shape (native semantics).
    """
    return _cross_wrapper_impl(args, kwargs or {}, default_dim=None, op_name="cross")


def _cross_dispatch_impl(
    input_tensor: Any, other_tensor: Any, dim: int | None, op_name: str
) -> ShardTensor:
    r"""Shared ``__torch_dispatch__`` implementation for the aten cross ops.

    The forward path of ``ShardedCross``, below autograd. Partials are
    rejected, not resolved -- collectives are the function-level handler's job.

    Parameters
    ----------
    input_tensor, other_tensor : ShardTensor or DTensor
        The distributed operands.
    dim : int or None
        Cross dim from the aten schema.
    op_name : str
        Operation name for error messages.

    Returns
    -------
    ShardTensor
        Cross product; ``requires_grad`` adjusted after the dispatcher.
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

    # DTensors convert for uniform spec metadata; below autograd they carry
    # no grad_fn, so from_dtensor is metadata-only.
    input_tensor = ShardTensor.from_dtensor(input_tensor)
    other_tensor = ShardTensor.from_dtensor(other_tensor)

    prep = _cross_prepare(input_tensor, other_tensor, dim, op_name)
    local_result = aten.linalg_cross.default(
        prep.locals_[0], prep.locals_[1], dim=prep.dim_offset
    )
    return _build_cross_output(
        local_result,
        prep,
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
