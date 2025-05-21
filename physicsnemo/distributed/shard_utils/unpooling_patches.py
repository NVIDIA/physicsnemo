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

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.autograd.profiler import record_function
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import ShardTensor
from physicsnemo.distributed.shard_utils.halo import (
    HaloConfig,
    halo_padding,
    unhalo_padding,
)


def repackage_interpolate_args(
    input: Union[torch.Tensor, ShardTensor],
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Tuple[Union[torch.Tensor, ShardTensor], Dict[str, Any]]:
    """Repackages interpolation arguments into standard format.

    For allowed modes, and details on other arguments, see upstream pytorch documentation:
    https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html

    Args:
        input: Input tensor to interpolate
        size: Output spatial size
        scale_factor: Multiplier for spatial size
        mode: Algorithm used for upsampling
        align_corners: Geometrically, whether corners are aligned
        recompute_scale_factor: Whether to recompute scale_factor
        antialias: Whether to use anti-aliasing
        *args: Additional positional args (unused)
        **kwargs: Additional keyword args (unused)

    Returns:
        Tuple containing:
        - Input tensor
        - Dict of interpolation configuration parameters
    """
    # Package all non-tensor parameters into a kwargs dictionary
    return_kwargs = {
        "size": size,
        "scale_factor": scale_factor,
        "mode": mode,
        "align_corners": align_corners,
        "recompute_scale_factor": recompute_scale_factor,
        "antialias": antialias,
    }

    return input, return_kwargs


def compute_interpolate_output_shape(
    input_shape: Tuple[int, ...], interp_kwargs: Dict[str, Any]
) -> Tuple[int, ...]:
    """Compute the output shape of an interpolation operation.

    Args:
        input_shape: Shape of the input tensor
        interp_kwargs: Keyword arguments for the interpolation operation

    Returns:
        tuple: Output shape after interpolation operation
    """
    size = interp_kwargs.get("size")
    scale_factor = interp_kwargs.get("scale_factor")

    # Batch and channel dimensions remain unchanged
    output_shape = list(input_shape[:2])

    if size is not None:
        # If size is provided, use it directly
        if isinstance(size, int):
            # If size is a single integer, use it for the last dimension only
            output_shape.extend(list(input_shape[2:-1]))
            output_shape.append(size)
        else:
            # If size is a sequence, it specifies output size for all spatial dimensions
            output_shape.extend(list(size))
    elif scale_factor is not None:
        # If scale_factor is provided, compute output sizes
        if isinstance(scale_factor, (int, float)):
            # Single scale factor for all spatial dimensions
            spatial_dims = [int(dim * scale_factor) for dim in input_shape[2:]]
        else:
            # Separate scale factor for each spatial dimension
            spatial_dims = [
                int(dim * scale_factor[i]) for i, dim in enumerate(input_shape[2:])
            ]
        output_shape.extend(spatial_dims)
    else:
        # If neither is provided, output shape is the same as input
        output_shape.extend(list(input_shape[2:]))

    return tuple(output_shape)


def compute_halo_sizes(
    input_shape: Tuple[int, ...],
    placements: Sequence[Any],
    interp_kwargs: Dict[str, Any],
) -> Dict[int, Tuple[int, int]]:
    """Compute the necessary halo sizes for different interpolation modes.

    Args:
        input_shape: Shape of the input tensor
        placements: Placements from the ShardTensor spec
        interp_kwargs: Keyword arguments for the interpolation operation

    Returns:
        dict: Halo sizes for each sharded dimension
    """
    mode = interp_kwargs.get("mode", "nearest")

    # Default halo sizes based on interpolation mode
    # For most modes, we need at least 1 element of overlap
    default_halo = {
        "nearest": 1,
        "linear": 2,
        "bilinear": 2,
        "bicubic": 4,
        "trilinear": 2,
        "area": 1,
    }

    halo_size = default_halo.get(mode, 1)
    halo_sizes = {}

    # Determine which dimensions are sharded
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_dim = placement.dim
            # Only add halos for spatial dimensions (skip batch and channel)
            if shard_dim >= 2:
                # The halo might need to be asymmetric based on scale_factor
                # This is a simplified implementation; might need refinement
                halo_sizes[shard_dim] = (halo_size, halo_size)

    return halo_sizes


def compute_halo_configs_from_interpolate_args(
    input: ShardTensor,
    interp_kwargs: Dict[str, Any],
) -> List[HaloConfig]:
    """Compute halo configurations for a sharded tensor based on interpolation arguments.

    Args:
        input: The sharded tensor that will be used in interpolation
        interp_kwargs: Dictionary of interpolation arguments including mode, size,
                      scale_factor, etc.

    Returns:
        List of HaloConfig objects for each sharded dimension
    """
    # Get the placements from the input tensor's spec
    placements = input._spec.placements

    # Compute halo sizes using the existing function
    halo_sizes = compute_halo_sizes(input.shape, placements, interp_kwargs)

    # Create halo configs from the computed sizes
    halo_configs = []

    for tensor_dim, (left_halo, right_halo) in halo_sizes.items():
        # Find which mesh dimension this tensor dimension is sharded on
        for mesh_dim, p in enumerate(placements):
            if isinstance(p, Shard) and p.dim == tensor_dim:
                # Create a halo config for this dimension
                halo_configs.append(
                    HaloConfig(
                        mesh_dim=mesh_dim,
                        tensor_dim=tensor_dim,
                        halo_size=max(
                            left_halo, right_halo
                        ),  # Using max as a simplification
                        edge_padding_size=0,  # No explicit padding in interpolation
                        communication_method="a2a",
                    )
                )
                break

    return halo_configs


def partial_interpolate_nd(
    input: ShardTensor,
    interp_kwargs: Dict[str, Any],
) -> ShardTensor:
    """Perform a convolution on a sharded tensor with halo exchange.

    This high-level, differentiable function computes a convolution on a sharded tensor
    by performing these steps:
    1. Calculate the size of halos needed
    2. Apply halo padding (differentiable)
    3. Perform convolution on the padded tensor with padding=0 on sharded dimensions
    4. Return the result as a ShardTensor

    Args:
        input: The sharded input tensor
        weight: Convolution filter weights
        bias: Optional bias parameter
        conv_kwargs: Dictionary of convolution parameters (stride, padding, etc.)

    Returns:
        Resulting ShardTensor after convolution operation
    """

    # This will produce one config per sharded dim
    # It also *updates* conv_kwargs in place to set padding to 0 on the sharded dims
    halo_configs = compute_halo_configs_from_interpolate_args(input, interp_kwargs)

    local_input = input.to_local()

    # Apply the halo padding to the input tensor
    for halo_config in halo_configs:
        local_input = halo_padding(local_input, input._spec.mesh, halo_config)

    unhalo_configs = []
    for h in halo_configs:
        unhalo_configs.append(  # noqa PERF401
            HaloConfig(
                mesh_dim=h.mesh_dim,
                tensor_dim=h.tensor_dim,
                halo_size=int(interp_kwargs["scale_factor"] * h.halo_size),
                edge_padding_size=h.edge_padding_size,
                communication_method=h.communication_method,
            )
        )

    # Perform the convolution on the padded tensor
    output = torch.nn.functional.interpolate(local_input, **interp_kwargs)

    # Remove halos and convert back to ShardTensor
    # x = UnSliceHaloND.apply(x, halo, q._spec)
    for halo_config in unhalo_configs:
        output = unhalo_padding(output, input._spec.mesh, halo_config)

    result_shapes = {}
    for mesh_dim, sharding_shape in input._spec.sharding_shapes().items():
        updated_shapes = tuple(
            torch.Size(compute_interpolate_output_shape(s, interp_kwargs))
            for s in sharding_shape
        )
        result_shapes[mesh_dim] = updated_shapes

    with record_function("upsampling.from_local"):
        # Convert back to ShardTensor
        output = ShardTensor.from_local(
            output,
            input._spec.mesh,
            input._spec.placements,
            sharding_shapes=result_shapes,
        )

    return output


def generic_interpolate_wrapper(
    func: Callable,
    types: Tuple[Any, ...],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> ShardTensor:
    """Wrapper for torch.nn.functional.interpolate.

    Handles both distributed ShardTensor inputs.
    For ShardTensor inputs, handles distributed interpolation with halo exchanges.

    Args:
        wrapped: Original interpolation function being wrapped
        instance: Instance the wrapped function is bound to
        args: Positional arguments for interpolation
        kwargs: Keyword arguments for interpolation

    Returns:
        Interpolation result as or ShardTensor

    Raises:
        UndeterminedShardingError: If input tensor types are invalid
    """
    # Extract the input tensor and package the remaining arguments
    input, interp_kwargs = repackage_interpolate_args(*args, **kwargs)

    # Handle distributed ShardTensor inputs
    return partial_interpolate_nd(input, interp_kwargs)


ShardTensor.register_function_handler(
    torch.nn.functional.interpolate, generic_interpolate_wrapper
)
