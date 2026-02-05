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
from torch.distributed.tensor.placement_types import (
    Shard,
)

from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.domain_parallel.shard_utils.patch_core import (
    MissingShardPatch,
)
from physicsnemo.utils.profiling import profile


def compute_local_padding_and_output_shape(
    input_tensor_shape: tuple[int, ...],
    pad: tuple[int, ...],
    mesh_coords: tuple[int, ...],
    mesh_sizes: tuple[int, ...],
    tensor_sharding_map: dict[int, int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    r"""Compute the local padding and output shape for a given input tensor shape.

    Parameters
    ----------
    input_tensor_shape : tuple[int, ...]
        The shape of the input tensor.
    pad : tuple[int, ...]
        The padding size(s).
    mesh_coords : tuple[int, ...]
        The coordinates of the tensor in the mesh.
    mesh_sizes : tuple[int, ...]
        The sizes of the mesh.
    tensor_sharding_map : dict[int, int]
        A map from tensor dimension to mesh dimension.

    Returns
    -------
    tuple[tuple[int, ...], tuple[int, ...]]
        Tuple of (local_output_shape, local_padding).
    """

    tensor_rank = len(input_tensor_shape)

    pad_dims = len(pad) // 2

    output_padding = []

    local_output_shape = list(input_tensor_shape)

    # We have to loop over this backwards:
    for dim_from_last in range(pad_dims):
        tensor_dim = tensor_rank - 1 - dim_from_last

        left = pad[2 * dim_from_last]
        right = pad[2 * dim_from_last + 1]

        # If this axis of the tensor is not sharded, we keep these as they are:
        if tensor_dim not in tensor_sharding_map.keys():
            output_padding.append(left)
            output_padding.append(right)
            local_output_shape[tensor_dim] += left + right
        else:
            # The tensor is sharded on this dim.
            # So, determine if this is an edge or not.

            # four cases here.
            # - is left and not is right
            # - not is left and is right
            # - is left AND is right:
            # - not is left and not is right

            mesh_dim = tensor_sharding_map[tensor_dim]
            is_left = mesh_coords[mesh_dim] == 0
            is_right = mesh_coords[mesh_dim] == mesh_sizes[mesh_dim] - 1

            if is_left and not is_right:
                output_padding.append(left)
                output_padding.append(0)
                local_output_shape[tensor_dim] += left
            elif not is_left and is_right:
                output_padding.append(0)
                output_padding.append(right)
                local_output_shape[tensor_dim] += right
            elif is_left and is_right:
                output_padding.append(left)
                output_padding.append(right)
                local_output_shape[tensor_dim] += left + right
            else:
                output_padding.append(0)
                output_padding.append(0)

    return tuple(local_output_shape), tuple(output_padding)


def generic_pad_nd_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:
    r"""Wrapper function for N-dimensional padding operations supporting ShardTensors.

    Parameters
    ----------
    func : callable
        The padding function to be wrapped.
    types : tuple
        Tuple of input types (unused).
    args : tuple
        Positional arguments to the padding function.
    kwargs : dict
        Keyword arguments to the padding function.

    Returns
    -------
    ShardTensor
        The result of the padding operation.

    Raises
    ------
    MissingShardPatch
        If circular padding is requested (not yet implemented).
    ValueError
        If padding specification is invalid.
    """

    # Padding is a no-communication operation unless it's circular padding
    # Circular padding is not implemented yet, and probably won't get implemented
    # until it's requested.

    inputs, pad, mode, value = repackage_pad_args(*args, **kwargs)

    if mode == "circular":
        raise MissingShardPatch(
            "Circular padding is not implemented yet.  Please open an issue at https://github.com/NVIDIA/PhysicsNemo/issues if you need this functionality."
        )

    # Now, get the local tensor:
    local_input = inputs.to_local()

    # We have to update the padding values based on where this tensor is, and if
    # it is on the edge or not.
    #
    # The only way to do that is to loop over the paddings to determine
    # the tensor axes, and then loop over the mesh / spec to see if that axis is
    # sharded.
    #
    # Further, because we don't want to communicate across GPUs unless it's needed,
    # We need to compute this for all tensors in the shard spec.

    # Sanity checks
    if len(pad) % 2 != 0:
        raise ValueError("Sharded Padding requires len(pad) to be divisible by 2.")

    pad_dims = len(pad) // 2

    if pad_dims > len(inputs.shape):
        raise ValueError(
            f"Sharded Padding specified for {pad_dims} but tensor has only {len(inputs.shape)} dimensions."
        )

    # By default, all output tensors are unsharded
    # This maps tensor dim to mesh dim but ONLY if it's sharded
    tensor_sharding_map = {}
    mesh_sizes = []
    spec = inputs._spec

    # Loop over the mesh spec and extract sharding vs tensor dim:
    for mesh_dim, placement in enumerate(spec.placements):
        if isinstance(placement, Shard):
            tensor_sharding_map[placement.dim] = mesh_dim
        mesh_sizes.append(spec.mesh.size(mesh_dim))

    # If the tensor_shard_map is all False, still (so no sharding)
    # We can just use a local computation and be done.
    if len(tensor_sharding_map) == 0:
        local_output = func(local_input, pad, mode, value)
        return ShardTensor.from_local(local_output, spec.mesh, spec.placements)

    # at this point, at least one dimension is sharded.  Maybe more.
    # So, loop over the mesh sharding shapes and compute the local output
    # shape and padding for that chunk:
    output_shapes = {}
    self_mesh_coords = [spec.mesh.get_local_rank(m) for m in range(spec.mesh.ndim)]
    self_padding = None
    for mesh_dim, sharding_shapes in spec.sharding_shapes().items():
        output_shapes[mesh_dim] = []
        for i, local_shape in enumerate(sharding_shapes):
            # Update the mesh sharding coords:
            mesh_coords = list(self_mesh_coords)
            mesh_coords[mesh_dim] = i
            output_shape, local_padding = compute_local_padding_and_output_shape(
                local_input.shape, pad, mesh_coords, mesh_sizes, tensor_sharding_map
            )

            # Catch and cache the one that applies to this rank:
            if mesh_coords == self_mesh_coords:
                self_padding = local_padding

            output_shapes[mesh_dim].append(output_shape)

    # From here, apply the local padding to this tensor:
    local_output = func(local_input, self_padding, mode, value)

    # Now, convert back to shard tensor.
    # We already have all the output shapes
    return ShardTensor.from_local(
        local_output, spec.mesh, spec.placements, sharding_shapes=output_shapes
    )


@profile
def repackage_pad_args(
    inputs: ShardTensor,
    pad: int | tuple[int, ...] = 0,
    mode: str = "constant",
    value: float | None = None,
    *args,
    **kwargs,
) -> tuple[
    ShardTensor,
    tuple[int, ...],
    str,
    float,
]:
    r"""Repackage pad arguments into standard format.

    Takes the full set of arguments that could be passed to a pad operation
    and separates them into core tensor inputs (inputs, pad, mode, value).

    Parameters
    ----------
    inputs : ShardTensor
        Input tensor to pad.
    pad : int | tuple[int, ...], default=0
        Padding size(s).
    mode : str, default="constant"
        Padding mode (``"constant"``, ``"reflect"``, ``"replicate"``, or ``"circular"``).
    value : float | None, optional
        Padding value for constant padding mode.
    *args : Any
        Additional positional arguments (unused).
    **kwargs : Any
        Additional keyword arguments (unused).

    Returns
    -------
    tuple[ShardTensor, tuple[int, ...], str, float]
        Tuple containing (input tensor, padding size(s), padding mode, padding value).
    """

    return inputs, pad, mode, value


ShardTensor.register_function_handler(torch.nn.functional.pad, generic_pad_nd_wrapper)
