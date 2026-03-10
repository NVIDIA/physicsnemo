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

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.domain_parallel.shard_utils.patch_core import (
    MissingShardPatch,
    UndeterminedShardingError,
)

aten = torch.ops.aten


def compute_output_shape(input_shape, pool_kwargs):
    r"""Compute the output shape of a pooling operation.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input tensor.
    pool_kwargs : dict
        Keyword arguments for the pooling operation.

    Returns
    -------
    tuple
        Output shape after pooling operation.
    """
    # Extract pooling parameters
    kernel_size = pool_kwargs.get("kernel_size")
    stride = pool_kwargs.get("stride", kernel_size)
    padding = pool_kwargs.get("padding", 0)

    # Handle scalar parameters
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * (len(input_shape) - 2)
    if isinstance(stride, int):
        stride = (stride,) * (len(input_shape) - 2)
    if isinstance(padding, int):
        padding = (padding,) * (len(input_shape) - 2)

    # Batch and channel dimensions remain unchanged
    output_shape = list(input_shape[:2])

    # Compute spatial dimensions
    for i, (size, k, s, p) in enumerate(
        zip(input_shape[2:], kernel_size, stride, padding)
    ):
        output_size = ((size + 2 * p - k) // s) + 1
        output_shape.append(output_size)

    return tuple(output_shape)


def repackage_pool_args(
    input: torch.Tensor | ShardTensor,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] = None,
    padding: int | tuple[int, ...] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
    *args,
    **kwargs,
) -> tuple[torch.Tensor | ShardTensor, dict[str, Any]]:
    r"""Repackage pooling arguments into standard format.

    Takes the full set of arguments that could be passed to an avg_pool operation
    and separates them into the input tensor and configuration parameters
    packaged as a kwargs dict.

    Parameters
    ----------
    input : Union[torch.Tensor, ShardTensor]
        Input tensor to pool.
    kernel_size : Union[int, Tuple[int, ...]]
        Size of the pooling window.
    stride : Union[int, Tuple[int, ...]], optional
        Stride of the pooling window, defaults to ``kernel_size``.
    padding : Union[int, Tuple[int, ...]], default=0
        Padding added to both sides of the input.
    ceil_mode : bool, default=False
        When ``True``, will use ceil instead of floor to compute the output shape.
    count_include_pad : bool, default=True
        When ``True``, will include the zero-padding in the averaging calculation.
    divisor_override : Optional[int], optional
        If specified, will be used as divisor, otherwise ``kernel_size`` is used.
    *args : Any
        Additional positional arguments (unused).
    **kwargs : Any
        Additional keyword arguments (unused).

    Returns
    -------
    Tuple[Union[torch.Tensor, ShardTensor], Dict[str, Any]]
        Tuple containing (input tensor, dict of pooling configuration parameters).
    """
    # Handle stride=None case (defaults to kernel_size)
    if stride is None:
        stride = kernel_size

    # Package all non-tensor parameters into a kwargs dictionary
    return_kwargs = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
    }

    # Only add divisor_override if it's not None
    if divisor_override is not None:
        return_kwargs["divisor_override"] = divisor_override

    return input, return_kwargs


def generic_avg_pool_nd_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:
    r"""Generic wrapper for torch N-dimensional average pooling operations.

    For ShardTensor inputs, handles applying distributed pooling.

    Parameters
    ----------
    func : Callable
        Original torch pooling function being wrapped.
    types : Tuple[Any, ...]
        Types of the arguments.
    args : Tuple[Any, ...]
        Positional arguments for pooling.
    kwargs : Dict[str, Any]
        Keyword arguments for pooling.

    Returns
    -------
    ShardTensor
        Pooling result as ShardTensor.

    Raises
    ------
    MissingShardPatch
        If stride does not equal kernel_size.
    UndeterminedShardingError
        If input tensor types are invalid or sharded dimensions are not
        divisible by stride.
    """

    # Extract the input tensor and package the remaining arguments
    input, pool_kwargs = repackage_pool_args(*args, **kwargs)

    # For pooling, the main challenge is to predict the output shape

    # Get the local tensor:
    local_input = input.to_local()

    local_pooled_output = func(local_input, **pool_kwargs)

    # Reject cases where stride != kernel_size
    if pool_kwargs.get("stride") != pool_kwargs.get("kernel_size"):
        raise MissingShardPatch(
            "Sharded pooling is not implemented for kernels without matching stride. "
            "If you need this functionality, please open an issue at https://github.com/NVIDIA/PhysicsNemo/issues"
        )

    # Check divisibility by stride only for sharded dimensions
    stride = pool_kwargs.get("stride")
    if isinstance(stride, int):
        # Assuming channels first ...
        stride = (stride,) * (len(local_input.shape) - 2)

    for mesh_dim, placement in enumerate(input._spec.placements):
        if isinstance(placement, Shard):
            # This dimension is sharded on this mesh dimension
            shard_dim = placement.dim
            # Skip batch and channel dimensions (first two dims)
            if shard_dim >= 2:
                spatial_dim = shard_dim - 2  # Convert to spatial dimension index
                # Get the sizes for this mesh dimension
                shard_shapes = input._spec.sharding_shapes()[mesh_dim]
                for shard_shape in shard_shapes:
                    if (
                        spatial_dim < len(shard_shape) - 2
                    ):  # Check if dimension is valid
                        spatial_size = shard_shape[shard_dim]
                        stride_for_dim = stride[spatial_dim]
                        if spatial_size % stride_for_dim != 0:
                            raise UndeterminedShardingError(
                                f"Sharded dimension {shard_dim} with local size {spatial_size} "
                                f"must be divisible by stride {stride_for_dim}"
                            )

        # Compute the sharding shapes:
        updated_placements = {}
        for mesh_dim, shard_shapes in input._spec.sharding_shapes().items():
            updated_shard_shapes = [
                compute_output_shape(shard_shape, pool_kwargs)
                for shard_shape in shard_shapes
            ]
            updated_placements[mesh_dim] = updated_shard_shapes

        output = ShardTensor.from_local(
            local_pooled_output,
            input._spec.mesh,
            input._spec.placements,
            sharding_shapes=updated_placements,
        )
        return output
        # Use the convolution args to compute the sharded halo

    else:
        msg = (
            "input must be a valid type "
            "(torch.Tensor or ShardTensor), but got "
            f"{type(input)}"
        )
        raise UndeterminedShardingError(msg)


ShardTensor.register_function_handler(
    torch.nn.functional.avg_pool1d, generic_avg_pool_nd_wrapper
)
ShardTensor.register_function_handler(
    torch.nn.functional.avg_pool2d, generic_avg_pool_nd_wrapper
)
ShardTensor.register_function_handler(
    torch.nn.functional.avg_pool3d, generic_avg_pool_nd_wrapper
)


def repackage_max_pool_args(
    input: torch.Tensor | ShardTensor,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] = None,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
    *args,
    **kwargs,
) -> tuple[torch.Tensor | ShardTensor, dict[str, Any]]:
    r"""Repackage max pooling arguments into standard format.

    Takes the full set of arguments that could be passed to a max_pool operation
    and separates them into the input tensor and configuration parameters
    packaged as a kwargs dict.

    Parameters
    ----------
    input : Union[torch.Tensor, ShardTensor]
        Input tensor to pool.
    kernel_size : Union[int, Tuple[int, ...]]
        Size of the pooling window.
    stride : Union[int, Tuple[int, ...]], optional
        Stride of the pooling window, defaults to ``kernel_size``.
    padding : Union[int, Tuple[int, ...]], default=0
        Padding added to both sides of the input.
    dilation : Union[int, Tuple[int, ...]], default=1
        Controls the spacing between kernel elements.
    ceil_mode : bool, default=False
        When ``True``, will use ceil instead of floor to compute the output shape.
    return_indices : bool, default=False
        When ``True``, returns indices of max locations along with outputs.
    *args : Any
        Additional positional arguments (unused).
    **kwargs : Any
        Additional keyword arguments (unused).

    Returns
    -------
    Tuple[Union[torch.Tensor, ShardTensor], Dict[str, Any]]
        Tuple containing (input tensor, dict of pooling configuration parameters).
    """
    # Handle stride=None case (defaults to kernel_size)
    if stride is None:
        stride = kernel_size

    # Package all non-tensor parameters into a kwargs dictionary
    return_kwargs = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices,
    }

    return input, return_kwargs


def generic_max_pool_nd_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:
    r"""Generic wrapper for torch N-dimensional max pooling operations.

    Handles distributed ShardTensor inputs.

    Parameters
    ----------
    func : Callable
        Original torch pooling function being wrapped.
    types : Tuple[Any, ...]
        Types of the arguments.
    args : Tuple[Any, ...]
        Positional arguments for pooling.
    kwargs : Dict[str, Any]
        Keyword arguments for pooling.

    Returns
    -------
    ShardTensor or Tuple[ShardTensor, ShardTensor]
        Pooling result as ShardTensor, or tuple of (output, indices) if
        ``return_indices=True``.

    Raises
    ------
    MissingShardPatch
        If stride does not equal kernel_size.
    UndeterminedShardingError
        If input tensor types are invalid or sharded dimensions are not
        divisible by stride.
    """

    # Extract the input tensor and package the remaining arguments
    input, pool_kwargs = repackage_max_pool_args(*args, **kwargs)

    # Get the local tensor:
    local_input = input.to_local()

    # Call the local pooling operation
    local_pooled_output = func(local_input, **pool_kwargs)

    # Handle return_indices case
    return_indices = pool_kwargs.get("return_indices", False)
    if return_indices:
        local_pooled_output, indices = local_pooled_output

    # Everything below here is computing output meta data

    # Reject cases where stride != kernel_size
    if pool_kwargs.get("stride") != pool_kwargs.get("kernel_size"):
        raise MissingShardPatch("Stride must equal kernel_size for pooling operations")

    # Check divisibility by stride only for sharded dimensions
    stride = pool_kwargs.get("stride")
    if isinstance(stride, int):
        # Assuming channels first ...
        stride = (stride,) * (len(local_input.shape) - 2)

    for mesh_dim, placement in enumerate(input._spec.placements):
        if isinstance(placement, Shard):
            # This dimension is sharded on this mesh dimension
            shard_dim = placement.dim
            # Skip batch and channel dimensions (first two dims)
            if shard_dim >= 2:
                spatial_dim = shard_dim - 2  # Convert to spatial dimension index
                # Get the sizes for this mesh dimension
                shard_shapes = input._spec.sharding_shapes()[mesh_dim]
                for shard_shape in shard_shapes:
                    if (
                        spatial_dim < len(shard_shape) - 2
                    ):  # Check if dimension is valid
                        spatial_size = shard_shape[shard_dim]
                        stride_for_dim = stride[spatial_dim]
                        if spatial_size % stride_for_dim != 0:
                            raise UndeterminedShardingError(
                                f"Sharded dimension {shard_dim} with local size {spatial_size} "
                                f"must be divisible by stride {stride_for_dim}"
                            )

        # Compute the sharding shapes:
        updated_placements = {}
        for mesh_dim, shard_shapes in input._spec.sharding_shapes().items():
            updated_shard_shapes = [
                compute_output_shape(shard_shape, pool_kwargs)
                for shard_shape in shard_shapes
            ]
            updated_placements[mesh_dim] = updated_shard_shapes

        output = ShardTensor.from_local(
            local_pooled_output,
            input._spec.mesh,
            input._spec.placements,
            sharding_shapes=updated_placements,
        )

        if return_indices:
            # Also create a ShardTensor for indices with the same sharding
            indices_output = ShardTensor.from_local(
                indices,
                input._spec.mesh,
                input._spec.placements,
                sharding_shapes=updated_placements,
            )
            return output, indices_output
        else:
            return output

    else:
        msg = (
            "input must be a valid type "
            "(torch.Tensor or ShardTensor), but got "
            f"{type(input)}"
        )
        raise UndeterminedShardingError(msg)


ShardTensor.register_function_handler(
    torch.nn.functional.max_pool3d, generic_max_pool_nd_wrapper
)
ShardTensor.register_function_handler(
    torch.nn.functional.max_pool2d, generic_max_pool_nd_wrapper
)
ShardTensor.register_function_handler(
    torch.nn.functional.max_pool1d, generic_max_pool_nd_wrapper
)
