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

r"""Custom reduction operations for ShardTensor.

This module provides custom autograd functions for reduction operations
(sum, mean) on ``ShardTensor`` objects. The key challenges addressed are:

1. **Uneven sharding**: Requires careful accumulation of partial results.
   This is particularly important for ``mean`` where the weight of each
   local contribution depends on its size relative to the global tensor.

2. **Gradient distribution**: Backward gradient distribution ensures that
   the shape of local gradients matches the local tensor shape on each rank.

The module provides:

- ``ShardedSum``: Custom autograd function for sum reduction
- ``ShardedMean``: Custom autograd function for mean reduction with proper weighting
- ``sum_wrapper``: Function handler for ``torch.sum`` on ShardTensor
- ``mean_wrapper``: Function handler for ``torch.mean`` on ShardTensor
"""

from __future__ import annotations

import dataclasses
from typing import (
    Any,
    Callable,
    Iterable,
    TypeVar,
)

import torch
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)

# noqa: E402
from physicsnemo.domain_parallel._shard_tensor_spec import (
    ShardTensorSpec,
    _stride_from_contiguous_shape_C_style,
)
from physicsnemo.domain_parallel.shard_tensor import ShardTensor

aten = torch.ops.aten

# Type variable for dimension parameter
DimT = TypeVar("DimT", None, int, Iterable[int])


def normalize_dim(
    dim: DimT, tensor_ndim: int, as_set: bool = False, handle_negatives: bool = True
) -> tuple[int, ...] | set[int] | None:
    r"""Normalize dimension argument to a consistent form.

    Parameters
    ----------
    dim : DimT
        The dimension(s) to normalize. Can be ``None``, ``int``, or iterable of ints.
    tensor_ndim : int
        Number of dimensions in the tensor.
    as_set : bool, default=False
        If ``True``, return a set of dimensions instead of a tuple.
    handle_negatives : bool, default=True
        If ``True``, convert negative dimensions to positive ones.

    Returns
    -------
    Union[Optional[Tuple[int, ...]], Set[int]]
        - ``None`` if ``dim`` is ``None`` and ``as_set`` is ``False``
        - A set of all dimensions if ``dim`` is ``None`` and ``as_set`` is ``True``
        - A tuple of dimensions (or set if ``as_set`` is ``True``)
    """
    if dim is None:
        if as_set:
            return set(range(tensor_ndim))
        return None

    # Convert to tuple if iterable
    if isinstance(dim, Iterable) and not isinstance(dim, torch.Tensor):
        dims = tuple(dim)
    else:
        dims = (dim,)

    # Handle negative dimensions
    if handle_negatives:
        dims = tuple(d % tensor_ndim for d in dims)

    # Return as set or tuple based on as_set flag
    if as_set:
        return set(dims)
    return dims


def is_full_reduction(dim: DimT, tensor_ndim: int) -> bool:
    r"""Determine if this is a full reduction.

    Parameters
    ----------
    dim : DimT
        The dimension(s) to check. Can be ``None``, ``int``, or iterable of ints.
    tensor_ndim : int
        Number of dimensions in the tensor.

    Returns
    -------
    bool
        ``True`` if all dimensions are being reduced, ``False`` otherwise.
    """
    if dim is None:
        return True
    if isinstance(dim, Iterable) and len(dim) == tensor_ndim:
        return True
    return False


def compute_result_placements(
    tensor: ShardTensor, dim: DimT, reduction_name: str, keepdim: bool = False
) -> list[Partial | Shard]:
    r"""Compute placement info for reduction result.

    Parameters
    ----------
    tensor : ShardTensor
        The input ShardTensor being reduced.
    dim : DimT
        The dimension(s) to reduce. Can be ``None``, ``int``, or iterable of ints.
    reduction_name : str
        Type of reduction operation (``"sum"``, ``"avg"``, etc.).
    keepdim : bool, default=False
        Whether to preserve reduced dimensions with size 1.

    Returns
    -------
    List[Union[Partial, Shard]]
        Placement specifications for the result tensor.
    """
    if is_full_reduction(dim, tensor.ndim):
        return [
            p
            if p.is_replicate()
            else Partial("sum" if reduction_name != "avg" else "avg")
            for p in tensor._spec.placements
        ]

    # Use enhanced normalize_dim to get dimensions as a set
    dims = normalize_dim(dim, tensor.ndim, as_set=True)

    placements = []
    for p in tensor._spec.placements:
        if isinstance(p, Shard):
            shard_dim = p.dim
            # Count how many reduction dims are less than this shard dim
            num_lower = sum(1 for d in dims if d < shard_dim)
            # If this sharded dim is being reduced, it becomes Partial
            if shard_dim in dims:
                placements.append(Partial(reduction_name))
            else:
                # If keepdim is False, dims to the left are removed, so shift left
                new_dim = shard_dim - num_lower if not keepdim else shard_dim
                placements.append(Shard(new_dim))
        else:
            placements.append(p)
    return placements


def reduction_shape(
    S: tuple[int, ...], dim: DimT = None, keepdim: bool = False
) -> tuple[int, ...]:
    r"""Calculate the resulting shape after a reduction operation.

    Parameters
    ----------
    S : tuple[int, ...]
        Original shape of the tensor (may be a ``torch.Size`` or plain tuple).
    dim : DimT, optional
        The dimension(s) to reduce. Can be ``None``, ``int``, or iterable of ints.
    keepdim : bool, default=False
        Whether to preserve reduced dimensions with size 1.

    Returns
    -------
    tuple[int, ...]
        The shape after reduction, returned as a plain int tuple (not
        ``torch.Size``) so the result can be safely embedded in a
        ``ShardTensorSpec._sharding_shapes`` dict without triggering dynamo's
        symbolic-shape special-casing for ``torch.Size``.
    """
    shape = list(S)
    if dim is None:
        return tuple([1] * len(shape)) if keepdim else tuple()

    # Use enhanced normalize_dim to handle iterable and negative dims
    dim = normalize_dim(dim, len(shape), handle_negatives=True)

    if keepdim:
        for d in dim:
            shape[d] = 1
    else:
        for d in sorted(dim, reverse=True):
            del shape[d]
    return tuple(shape)


def compute_result_sharding_shapes(
    tensor: ShardTensor, dim: DimT, keepdim: bool
) -> dict[int, list[tuple[int, ...]]]:
    r"""Compute sharding sizes for the result of a reduction operation.

    Parameters
    ----------
    tensor : ShardTensor
        The input ShardTensor being reduced.
    dim : DimT
        The dimension(s) to reduce. Can be ``None``, ``int``, or iterable of ints.
    keepdim : bool
        Whether to preserve reduced dimensions with size 1.

    Returns
    -------
    Dict[int, List[Tuple[int, ...]]]
        Mapping of mesh dimensions to plain int tuple sharding shapes.
    """
    if is_full_reduction(dim, tensor.ndim):
        return {}
    else:
        # Create a dictionary to store sharding sizes for dimensions that remain in the output
        result_sharding_shapes = {}

        # Get the original sharding sizes
        original_sharding_shapes = tensor._spec.sharding_shapes()
        # Use normalize_dim directly
        normalized_dim = normalize_dim(dim, tensor.ndim)

        for mesh_dim, sharding_shapes in original_sharding_shapes.items():
            result_sharding_shapes[mesh_dim] = [
                reduction_shape(shape, normalized_dim, keepdim)
                for shape in sharding_shapes
            ]

        return result_sharding_shapes


def build_reduction_result(
    local_result: torch.Tensor,
    input_tensor: ShardTensor,
    placements: list[Partial | Shard],
    sharding_shapes: dict[int, list[tuple[int, ...]]],
) -> ShardTensor:
    r"""Construct a ShardTensor result from a local reduction output.

    Builds the ``ShardTensorSpec`` directly from the already-computed placements
    and sharding shapes, avoiding the overhead and autograd side-effects of
    ``ShardTensor.from_local``.

    Parameters
    ----------
    local_result : torch.Tensor
        The locally-computed reduction result.
    input_tensor : ShardTensor
        The original input ShardTensor (used for device mesh).
    placements : List[Union[Partial, Shard]]
        Result placements from :func:`compute_result_placements`.
    sharding_shapes : Dict[int, List[Tuple[int, ...]]]
        Result sharding shapes from :func:`compute_result_sharding_shapes`.

    Returns
    -------
    ShardTensor
        Wrapped result with correct sharding metadata.
    """
    global_shape = list(local_result.shape)
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            tensor_dim = placement.dim
            global_shape[tensor_dim] = sum(
                s[tensor_dim] for s in sharding_shapes[mesh_dim]
            )

    stride = _stride_from_contiguous_shape_C_style(global_shape)
    spec = ShardTensorSpec(
        mesh=input_tensor.device_mesh,
        placements=tuple(placements),
        tensor_meta=TensorMeta(
            shape=tuple(global_shape),
            stride=stride,
            dtype=local_result.dtype,
        ),
        _local_shape=local_result.shape,
        # Normalize to plain int tuples (never torch.Size) for the
        # _sharding_shapes field; see ShardTensorSpec docstring.
        _sharding_shapes={
            dim: tuple(tuple(inner) for inner in s)
            for dim, s in sharding_shapes.items()
        },
    )
    return ShardTensor.__new__(
        ShardTensor,
        local_tensor=local_result,
        spec=spec,
        requires_grad=input_tensor.requires_grad,
    )


def create_sharded_grad_input(
    local_grad_input: torch.Tensor, original_spec: Any
) -> ShardTensor:
    r"""Create a ShardTensor from local gradient input.

    Parameters
    ----------
    local_grad_input : torch.Tensor
        The local gradient tensor.
    original_spec : ShardTensorSpec
        The original ShardTensor's spec to use for placement.

    Returns
    -------
    ShardTensor
        A distributed tensor matching the original sharding, with any input
        ``Partial`` placements normalized to ``Replicate``.
    """
    # A reduction's cotangent is fully materialized on every rank before it is
    # expanded to the input's local shape. If the primal input was Partial,
    # reusing that label would falsely claim this complete gradient still
    # needed reduction.
    grad_placements = tuple(
        Replicate() if placement.is_partial() else placement
        for placement in original_spec.placements
    )
    grad_spec = (
        original_spec
        if grad_placements == original_spec.placements
        else dataclasses.replace(original_spec, placements=grad_placements)
    )

    # In custom autograd backward, return the input gradient directly as a
    # ShardTensor value. Avoid ``from_local`` here (which routes through a
    # separate autograd Function) so the gradient is attached unambiguously to
    # the original ShardTensor input.
    return ShardTensor.__new__(
        ShardTensor,
        local_tensor=local_grad_input,
        spec=grad_spec,
        requires_grad=False,
    )


def resolve_partial_cotangent(grad_output: ShardTensor) -> ShardTensor:
    r"""Resolve pending reductions before consuming a reduction cotangent.

    Parameters
    ----------
    grad_output : ShardTensor
        Incoming cotangent for a sharded reduction.

    Returns
    -------
    ShardTensor
        The cotangent with every ``Partial`` placement reduced to ``Replicate``.
    """
    if not any(placement.is_partial() for placement in grad_output.placements):
        return grad_output
    placements = tuple(
        Replicate() if placement.is_partial() else placement
        for placement in grad_output.placements
    )
    return grad_output.redistribute(
        device_mesh=grad_output.device_mesh,
        placements=placements,
    )


class ShardedReductionBase(torch.autograd.Function):
    r"""Base class for implementing custom autograd functions for sharded tensor reductions.

    This class provides common setup functionality for reduction operations,
    saving necessary context for the backward pass including the original spec,
    dimensions being reduced, and local tensor shape.
    """

    @staticmethod
    def setup_ctx(
        ctx: Any, tensor: ShardTensor, dim: DimT, keepdim: bool
    ) -> tuple[tuple[int, ...] | None, bool]:
        r"""Save common context information for backward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            The autograd context object.
        tensor : ShardTensor
            The input ShardTensor being reduced.
        dim : DimT
            The dimension(s) to reduce.
        keepdim : bool
            Whether to preserve reduced dimensions with size 1.

        Returns
        -------
        Tuple[Optional[Tuple[int, ...]], bool]
            Tuple containing normalized dimension and keepdim flag.
        """
        ctx.original_spec = tensor._spec
        ctx.output_requires_grad = tensor.requires_grad

        # Normalize dim to tuple form
        dim = normalize_dim(dim, tensor.ndim)

        # Ensure keepdim is a boolean
        keepdim = bool(keepdim)

        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.is_full_reduction = is_full_reduction(dim, tensor.ndim)

        # Save the shape of the local tensor
        ctx.local_grad_shape = tensor._local_tensor.shape

        return dim, keepdim


class ShardedSum(ShardedReductionBase):
    r"""Custom autograd function for sum reduction of sharded tensors.

    Handles both forward and backward passes with proper gradient computation.
    The forward pass computes local sums and creates appropriate partial
    placements. The backward pass broadcasts gradients back to match the
    original tensor shape.
    """

    @staticmethod
    def forward(
        tensor: ShardTensor,
        dim: DimT = None,
        keepdim: bool = False,
        dtype: torch.dtype | None = None,
    ) -> ShardTensor:
        r"""Forward pass for sum reduction on ShardTensor.

        Parameters
        ----------
        tensor : ShardTensor
            The input ShardTensor to be reduced.
        dim : DimT, optional
            The dimension(s) to reduce.
        keepdim : bool, default=False
            Whether to preserve reduced dimensions with size 1.
        dtype : Optional[torch.dtype], optional
            Output data type.

        Returns
        -------
        ShardTensor
            The result of sum reduction.

        Notes
        -----
        The body runs under ``torch._C.DisableTorchFunctionSubclass``.
        Reason: new-style autograd.Function (per-PyTorch design) executes
        ``forward`` with grad-mode ON, and any property access on the
        ShardTensor input (e.g. ``tensor.ndim`` -- a C-level getset
        descriptor) re-enters ``__torch_function__`` -> the DTensor
        fallback -> ``_ShardTensorToDTensor.apply``. The resulting
        ``BackwardCFunction`` has a ``next_functions`` accessor that
        raises a "legacy access pattern" error on newer PyTorch,
        blocking AOTAutograd from walking the autograd graph. Shielding
        these metadata-only accesses fully avoids that bridge for the
        sum path.
        """
        with torch._C.DisableTorchFunctionSubclass():
            dim_n = normalize_dim(dim, tensor.ndim)
            keepdim_n = bool(keepdim)

            local_result = aten.sum(
                tensor._local_tensor, dim=dim_n, keepdim=keepdim_n, dtype=dtype
            )

            placements = compute_result_placements(tensor, dim_n, "sum")
            sharding_shapes = compute_result_sharding_shapes(tensor, dim_n, keepdim_n)

            return build_reduction_result(
                local_result, tensor, placements, sharding_shapes
            )

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        r"""Save the input ShardTensorSpec + normalized dim/keepdim for backward.

        Same ``DisableTorchFunctionSubclass`` shielding as ``forward`` so the
        property accesses inside ``ShardedReductionBase.setup_ctx`` (e.g.
        ``tensor.ndim``, ``tensor.requires_grad``) don't bridge through the
        AOT-hostile autograd Function fallback.
        """
        tensor, dim, keepdim, _dtype = inputs
        with torch._C.DisableTorchFunctionSubclass():
            ShardedReductionBase.setup_ctx(ctx, tensor, dim, keepdim)

    @staticmethod
    def backward(
        ctx: Any, grad_output: ShardTensor
    ) -> tuple[ShardTensor, None, None, None]:
        r"""Backward pass for sum reduction.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            The autograd context object.
        grad_output : ShardTensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        Tuple[ShardTensor, None, None, None]
            Tuple containing gradient for input tensor and ``None`` for
            dim, keepdim, and dtype (not differentiable).
        """
        original_spec = ctx.original_spec
        dim = ctx.dim
        is_full_reduction = ctx.is_full_reduction
        keepdim = ctx.keepdim
        local_grad_shape = ctx.local_grad_shape

        # A genuine Partial cotangent contains only this rank's contribution.
        # Resolve it before broadcasting across the original local input.
        grad_output = resolve_partial_cotangent(grad_output)
        local_grad_output = grad_output._local_tensor

        if is_full_reduction:
            # For full reduction, broadcast to original size
            grad_input = local_grad_output.expand(local_grad_shape)
        else:
            # For dimension-specific reduction
            if keepdim:
                # Just expand along reduced dimensions
                expand_shape = list(local_grad_shape)
                grad_input = local_grad_output.expand(expand_shape)
            else:
                # Need to unsqueeze first
                grad_shape = list(local_grad_output.shape)
                for d in sorted(dim):
                    if d < 0:
                        d += original_spec.tensor_meta.ndim
                    grad_shape.insert(d, 1)

                grad_expanded = local_grad_output.reshape(grad_shape)
                expand_shape = list(local_grad_shape)
                grad_input = grad_expanded.expand(expand_shape)

        # Create ShardTensor from local grad
        grad_input = create_sharded_grad_input(grad_input, original_spec)
        # Return gradients for all inputs
        return grad_input, None, None, None


class ShardedMean(ShardedReductionBase):
    r"""Custom autograd function for mean reduction of sharded tensors.

    Handles both forward and backward passes with proper gradient computation
    and scaling. The key challenge is that with uneven sharding, each rank's
    local mean must be weighted by its local size relative to the global size
    to produce the correct global mean.
    """

    @staticmethod
    def forward(
        tensor: ShardTensor,
        dim: DimT = None,
        keepdim: bool = False,
        dtype: torch.dtype | None = None,
    ) -> ShardTensor:
        r"""Forward pass for mean reduction on ShardTensor.

        Parameters
        ----------
        tensor : ShardTensor
            The input ShardTensor to be reduced.
        dim : DimT, optional
            The dimension(s) to reduce.
        keepdim : bool, default=False
            Whether to preserve reduced dimensions with size 1.
        dtype : Optional[torch.dtype], optional
            Output data type.

        Returns
        -------
        ShardTensor
            The result of mean reduction.

        Notes
        -----
        The body runs under ``torch._C.DisableTorchFunctionSubclass``.
        Reason: new-style autograd.Function (per-PyTorch design) executes
        ``forward`` with grad-mode ON, and any property access on the
        ShardTensor input (e.g. ``tensor.ndim`` -- a C-level getset
        descriptor) re-enters ``__torch_function__`` -> the DTensor
        fallback -> ``_ShardTensorToDTensor.apply``. The resulting
        ``BackwardCFunction`` has a ``next_functions`` accessor that
        raises a "legacy access pattern" error on newer PyTorch,
        blocking AOTAutograd from walking the autograd graph. Shielding
        these metadata-only accesses fully avoids that bridge for the
        mean path.
        """
        with torch._C.DisableTorchFunctionSubclass():
            dim_n = normalize_dim(dim, tensor.ndim)
            keepdim_n = bool(keepdim)

            # Get local tensor
            local_tensor = tensor._local_tensor

            # Compute proper weighting for mean
            weight = 1.0

            # Normalize dimensions for consistent handling
            if is_full_reduction(dim_n, tensor.ndim):
                # For full reduction, use all dimensions
                reduction_dims = set(range(tensor.ndim))
            else:
                # Only use the normalized dimensions for partial reduction
                reduction_dims = dim_n

            # Calculate weight based on local vs global shape ratio for reduction dimensions
            local_shape = local_tensor.shape
            global_shape = tensor.shape

            for d in reduction_dims:
                weight *= local_shape[d] / global_shape[d]

            # Perform local mean and apply weighting for uneven shards
            local_result = aten.mean(
                local_tensor, dim=dim_n, keepdim=keepdim_n, dtype=dtype
            )
            local_result = local_result * weight

            placements = compute_result_placements(tensor, dim_n, "sum")
            sharding_shapes = compute_result_sharding_shapes(tensor, dim_n, keepdim_n)

            return build_reduction_result(
                local_result, tensor, placements, sharding_shapes
            )

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        r"""Save the input ShardTensorSpec + normalized dim/keepdim for backward.

        Same ``DisableTorchFunctionSubclass`` shielding as ``forward`` so the
        property accesses inside ``ShardedReductionBase.setup_ctx`` (e.g.
        ``tensor.ndim``, ``tensor.requires_grad``) don't bridge through the
        AOT-hostile autograd Function fallback.
        """
        tensor, dim, keepdim, _dtype = inputs
        with torch._C.DisableTorchFunctionSubclass():
            ShardedReductionBase.setup_ctx(ctx, tensor, dim, keepdim)

    @staticmethod
    def backward(
        ctx: Any, grad_output: ShardTensor
    ) -> tuple[ShardTensor, None, None, None]:
        r"""Backward pass for mean reduction.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            The autograd context object.
        grad_output : ShardTensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        Tuple[ShardTensor, None, None, None]
            Tuple containing gradient for input tensor and ``None`` for
            dim, keepdim, and dtype (not differentiable).
        """
        original_spec = ctx.original_spec
        dim = ctx.dim
        is_full_reduction = ctx.is_full_reduction
        keepdim = ctx.keepdim
        local_grad_shape = ctx.local_grad_shape
        global_shape = original_spec.tensor_meta.shape

        # A genuine Partial cotangent contains only this rank's contribution.
        # Resolve it before broadcasting across the original local input.
        grad_output = resolve_partial_cotangent(grad_output)
        local_grad_output = grad_output._local_tensor

        if is_full_reduction:
            # For full reduction, broadcast to original size with scaling
            factor = 1.0 / torch.prod(torch.tensor(global_shape))
            grad_input = local_grad_output.expand(local_grad_shape) * factor
        else:
            # For dimension-specific reduction
            if keepdim:
                # Just expand along reduced dimensions
                expand_shape = list(local_grad_shape)
                grad_input = local_grad_output.expand(expand_shape)
            else:
                # Need to unsqueeze first
                grad_shape = list(local_grad_output.shape)
                for d in sorted(dim):
                    if d < 0:
                        d += original_spec.tensor_meta.ndim
                    grad_shape.insert(d, 1)

                grad_expanded = local_grad_output.reshape(grad_shape)
                expand_shape = list(local_grad_shape)
                grad_input = grad_expanded.expand(expand_shape)

            # Apply scaling factor for mean
            factor = 1.0
            for d in dim:
                if d < 0:
                    d += original_spec.tensor_meta.ndim
                factor /= global_shape[d]
            grad_input = grad_input * factor

        # Create ShardTensor from local grad
        grad_input = create_sharded_grad_input(grad_input, original_spec)

        # Return gradients for all inputs
        return grad_input, None, None, None


def sum_wrapper(
    func: Callable, types: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> ShardTensor:
    r"""Wrapper function for ShardTensor sum reduction.

    This function is registered as a handler for ``torch.sum`` on ShardTensor
    inputs. It unpacks the arguments and delegates to ``ShardedSum.apply``.

    Parameters
    ----------
    func : Callable
        The original function being wrapped (``torch.sum``).
    types : Any
        Types of the arguments (unused).
    args : Tuple[Any, ...]
        Positional arguments containing tensor, dim, keepdim, etc.
    kwargs : Dict[str, Any]
        Keyword arguments.

    Returns
    -------
    ShardTensor
        Result of sum reduction.
    """
    tensor, dim, keepdim, extra_args, extra_kwargs = unpack_args(*args, **kwargs)

    return ShardedSum.apply(tensor, dim, keepdim, *extra_args, **extra_kwargs)


def mean_wrapper(
    func: Callable, types: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> ShardTensor:
    r"""Wrapper function for ShardTensor mean reduction.

    This function is registered as a handler for ``torch.mean`` on ShardTensor
    inputs. It unpacks the arguments and delegates to ``ShardedMean.apply``.

    Parameters
    ----------
    func : Callable
        The original function being wrapped (``torch.mean``).
    types : Any
        Types of the arguments (unused).
    args : Tuple[Any, ...]
        Positional arguments containing tensor, dim, keepdim, etc.
    kwargs : Dict[str, Any]
        Keyword arguments.

    Returns
    -------
    ShardTensor
        Result of mean reduction.
    """
    tensor, dim, keepdim, extra_args, extra_kwargs = unpack_args(*args, **kwargs)

    return ShardedMean.apply(tensor, dim, keepdim, *extra_args, **extra_kwargs)


def unpack_args(
    tensor: ShardTensor,
    dim: DimT = None,
    keepdim: bool = False,
    *args: Any,
    **kwargs: Any,
) -> tuple[ShardTensor, DimT, bool, tuple[Any, ...], dict[str, Any]]:
    r"""Unpack arguments for reduction functions.

    Maps default arguments from torch reduction functions to a consistent format.

    Parameters
    ----------
    tensor : ShardTensor
        Input ShardTensor to reduce.
    dim : DimT, optional
        The dimension(s) to reduce.
    keepdim : bool, default=False
        Whether to preserve reduced dimensions with size 1.
    *args : Any
        Additional positional arguments.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Tuple[ShardTensor, DimT, bool, Tuple[Any, ...], Dict[str, Any]]
        Tuple containing tensor, dim, keepdim, extra args, and extra kwargs.
    """
    return tensor, dim, keepdim, args, kwargs


# Map the reduction ops to their handlers
reduction_mapping: dict[str, Callable] = {
    "sum": sum_wrapper,
    "avg": mean_wrapper,
}


# Register handlers for standalone functions and methods
ShardTensor.register_function_handler(torch.mean, mean_wrapper)
ShardTensor.register_function_handler(torch.Tensor.mean, mean_wrapper)
ShardTensor.register_function_handler(torch.sum, sum_wrapper)
ShardTensor.register_function_handler(torch.Tensor.sum, sum_wrapper)
