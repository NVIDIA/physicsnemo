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

from typing import Any, Tuple

import torch

from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")

from torch.distributed.tensor.placement_types import (  # noqa: E402
    Replicate,
    Shard,
)

from physicsnemo.distributed import ShardTensor  # noqa: E402
from physicsnemo.distributed._shard_tensor_spec import (  # noqa: E402
    ShardTensorSpec,
    TensorMeta,
    _stride_from_contiguous_shape_C_style,
)
from physicsnemo.distributed.shard_utils.patch_core import (  # noqa: E402
    MissingShardPatch,
)

aten = torch.ops.aten

__all__ = [
    "index_select_wrapper",
]


class ShardedIndexSelect(torch.autograd.Function):
    """
    Autograd function implementing a differentiable index_select operation for ShardTensors.

    This class provides both forward and backward pass implementations to enable
    gradient computation through the index_select operation when working with
    distributed sharded tensors.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: ShardTensor,
        dim: int,
        index: ShardTensor,
    ) -> ShardTensor:
        """
        Implementation of a differentiable index select operation on ShardTensors.

        This requires collectives and temporarily utilizing the full shape.
        It could be optimized, for large tensors, to use a ring and smarter indexing.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context object to store information for backward pass
        tensor : ShardTensor
            Input tensor to select from
        dim : int
            Dimension along which to index
        index : ShardTensor
            Indices to select

        Returns
        -------
        ShardTensor
            Output tensor containing the selected elements

        Raises
        ------
        MissingShardPatch
            If the index sharding strategy is not implemented
        """
        # This is the simplest implementation, to enable functionality.
        # It could be optimized for very large tensors to ensure performace.

        # We save the local version of the index and the input tensor spec for the backwards pass

        ctx.spec = tensor._spec
        ctx.grad_shape = tensor._local_tensor.shape
        ctx.dim = dim

        # First - Make sure we have the full input tensor
        # Triggers an all_gather(_v) for (uneven) tensors.
        local_tensor = tensor.full_tensor()

        # Perform the index select using the local values of the index:
        local_index = index.to_local()
        ctx.save_for_backward(index)

        # Get everything requested from the local index:
        local_values = aten.index_select(local_tensor, dim, local_index)

        # Now, we do gymnastics to make sure the output is correctly sharded.
        # Because index is one dimensional, by requirement of the underlying function,
        # it's not as annoying as it could be.
        index_placement = index._spec.placements[0]

        if index_placement.is_shard():
            # Then, we return a tensor sharded along dim aka Shard(dim).
            # Size per rank is easy to compute, no communication needed.
            output_size = list(tensor.shape)
            output_shard_sizes = {}
            for mesh_dim, index_shard_sizes in index._spec.sharding_shapes().items():
                output_shard_sizes[mesh_dim] = []
                for local_chunk_size in index_shard_sizes:
                    this_shard_size = output_size
                    this_shard_size[dim] = local_chunk_size[0]
                    # Make sure it's a tuple:
                    output_shard_sizes[mesh_dim].append(
                        torch.Size(tuple(this_shard_size))
                    )
                # Make sure it's a tuple:
                output_shard_sizes[mesh_dim] = tuple(output_shard_sizes[mesh_dim])

            ctx.output_shard_sizes = output_shard_sizes

            return_tensor = ShardTensor.from_local(
                local_values,
                device_mesh=tensor._spec.mesh,
                placements=[
                    Shard(dim),
                ],
                sharding_shapes=output_shard_sizes,
            )
            return return_tensor
        elif index_placement.is_replicate():
            # The output sharding should match the sharding of the original tensor.
            output_size = list(tensor.shape)

            # Replace the output size along the indexing dim with the right size:
            output_size[dim] = local_values.shape[dim]
            # Cast to shard tensor (as replicated, right now):
            output = ShardTensor.from_local(
                local_values,
                device_mesh=tensor._spec.mesh,
                placements=[
                    Replicate(),
                ],
            )

            # Redistribute to the original sharding of the input tensor:
            output = output.redistribute(tensor._spec.mesh, tensor._spec.placements)

            return output

        else:
            raise MissingShardPatch(
                f"Index select is not implemented for {index_placement} sharding."
            )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: ShardTensor
    ) -> Tuple[ShardTensor, None, None]:
        """
        Backward pass for the index_select operation on ShardTensors.

        The backward pass sends gradients appropriately to the input tensor.
        Therefore, its sharding should match the input tensor's sharding.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context object containing saved tensors and attributes from forward pass
        grad_output : ShardTensor
            Gradient of the loss with respect to the output of forward pass

        Returns
        -------
        Tuple[ShardTensor, None, None]
            Tuple containing:
            - Gradient with respect to input tensor
            - None for dim parameter (not differentiable)
            - None for index parameter (not differentiable)
        """
        (index,) = ctx.saved_tensors
        spec = ctx.spec
        dim = ctx.dim

        local_index = index.full_tensor()

        grad_inputs = torch.zeros(
            spec.tensor_meta.shape, device=grad_output._local_tensor.device
        )
        # local_grad_output = grad_output.to_local()
        local_grad_output = grad_output.full_tensor()

        grad_inputs = aten.index_add(grad_inputs, dim, local_index, local_grad_output)

        # Now, grad_inputs is replicated on all devices.
        # Shard it along the original sharding of the input tensor.
        grad_inputs = ShardTensor.from_local(
            grad_inputs,
            device_mesh=spec.mesh,
            placements=[
                Replicate(),
            ],
        )
        grad_inputs = grad_inputs.redistribute(spec.mesh, spec.placements)

        return grad_inputs, None, None


def sharded_index_select(
    tensor: ShardTensor,
    dim: int,
    index: ShardTensor,
) -> ShardTensor:
    """
    Performs an index_select operation on ShardTensors with autograd support.

    This is a thin wrapper around the ShardedIndexSelect autograd function
    to make the operation differentiable.

    Parameters
    ----------
    tensor : ShardTensor
        Input tensor to select from
    dim : int
        Dimension along which to index
    index : ShardTensor
        Indices to select

    Returns
    -------
    ShardTensor
        Output tensor containing the selected elements
    """
    return ShardedIndexSelect.apply(tensor, dim, index)


def index_select_wrapper(
    func: Any, instance: Any, args: tuple, kwargs: dict
) -> ShardTensor:
    """
    Wrapper for index_select operation that handles ShardTensors

    Returns
    -------
    ShardTensor
        Output tensor containing the selected elements

    """

    # Extract the tensor and index from the arguments
    tensor, dim, index = args

    return sharded_index_select(tensor, dim, index)


ShardTensor.register_function_handler(torch.index_select, index_select_wrapper)


def sharded_select_helper(tensor: ShardTensor, dim: int, index: int) -> ShardTensor:
    """
    This function contains the logic for performing a select operation on a ShardTensor.
    """

    # if the chunking dimension is along a dimension that is sharded, we have to handle that.
    # If it's along an unsharded dimension, there is nearly nothing to do.

    input_spec = tensor._spec

    input_placements = input_spec.placements

    shards = [s for s in input_placements if isinstance(s, Shard)]

    # We are reducing tensor rank and returning one sharding per tensor:
    original_shape = list(input_spec.shape)

    if dim in [i.dim for i in shards]:
        raise MissingShardPatch(
            "No implementation for aten.select.int along sharding axis yet."
        )

    else:
        # We are reducing tensor rank:
        original_shape.pop(dim)
        output_stride = _stride_from_contiguous_shape_C_style(original_shape)

        # Need to create a new global meta:
        new_meta = TensorMeta(
            torch.Size(tuple(original_shape)),
            stride=output_stride,
            dtype=input_spec.tensor_meta.dtype,
        )
        # The placements get adjusted too
        new_placements = []
        for p in input_spec.placements:
            if p.is_replicate():
                new_placements.append(p)
            elif p.is_shard():
                if p.dim > dim:
                    new_placements.append(Shard(p.dim - 1))
                else:
                    new_placements.append(p)
            elif p.is_partial():
                raise MissingShardPatch(
                    "Partial placement not supported yet for select"
                )

        # We can directly compute the sizes from the input spec sharding sizes:
        # Since the constraint above prevents selecting along a sharded dimension,
        # we can be sure that none of these adjusted shapes will be sharded.
        output_shard_sizes = {}
        for mesh_dim, index_shard_sizes in input_spec.sharding_shapes().items():
            output_shard_sizes[mesh_dim] = []
            for local_chunk_size in index_shard_sizes:
                local_chunk_size_list = list(local_chunk_size)
                local_chunk_size_list.pop(dim)
                output_shard_sizes[mesh_dim].append(
                    torch.Size(tuple(local_chunk_size_list))
                )
            output_shard_sizes[mesh_dim] = tuple(output_shard_sizes[mesh_dim])

        output_spec = ShardTensorSpec(
            mesh=input_spec.mesh,
            placements=tuple(new_placements),
            tensor_meta=new_meta,
            _sharding_shapes=output_shard_sizes,
        )
        # Finally, actually perform the select:
        local_result = aten.select.int(tensor._local_tensor, dim, index)

        return ShardTensor(
            local_result,
            output_spec,
            requires_grad=False,  # This will get adjusted after the dispatcher
        )


def sharded_select_backward_helper(
    grad_output: ShardTensor, input_sizes: torch.Size, dim: int, index: int
) -> ShardTensor:
    """
    This function contains the logic for performing a gradient of a select operation on a ShardTensor.

    We shard the gradients analogously to the output gradients.

    """

    # if the chunking dimension is along a dimension that is sharded, we have to handle that.
    # If it's along an unsharded dimension, there is nearly nothing to do.

    input_placements = grad_output._spec.placements

    output_stride = _stride_from_contiguous_shape_C_style(input_sizes)

    # Need to create a new global meta:
    new_meta = TensorMeta(
        torch.Size(tuple(input_sizes)),
        stride=output_stride,
        dtype=grad_output._spec.tensor_meta.dtype,
    )

    new_placements = input_placements
    # The placements get adjusted too
    new_placements = []
    for p in grad_output._spec.placements:
        if p.is_replicate():
            new_placements.append(p)
        elif p.is_shard():
            if p.dim >= dim:
                new_placements.append(Shard(p.dim + 1))
            else:
                new_placements.append(p)
        elif p.is_partial():
            raise Exception("Partial placement not supported yet for select_backward")

    # Next, calculate the sharding sizes for the output tensor:
    output_shard_sizes = {}
    for mesh_dim, index_shard_sizes in grad_output._spec.sharding_shapes().items():
        output_shard_sizes[mesh_dim] = []
        for local_chunk_size in index_shard_sizes:
            # We need to insert input_sizes[dim] at index:
            local_chunk_size_list = list(local_chunk_size)
            local_chunk_size_list.insert(dim, input_sizes[dim])
            output_shard_sizes[mesh_dim].append(
                torch.Size(tuple(local_chunk_size_list))
            )
        output_shard_sizes[mesh_dim] = tuple(output_shard_sizes[mesh_dim])

    output_spec = ShardTensorSpec(
        mesh=grad_output._spec.mesh,
        placements=tuple(new_placements),
        tensor_meta=new_meta,
        _sharding_shapes=output_shard_sizes,
    )

    # Finally, make sure we use the correct local size:
    mesh_rank = grad_output._spec.mesh.get_local_rank()
    if len(output_shard_sizes.keys()) > 0:
        local_output_size = output_shard_sizes[0][mesh_rank]
    else:
        # Fall back to the global shape if nothing is sharded:
        local_output_size = output_spec.tensor_meta.shape

    # Now, compute the local result:
    local_result = aten.select_backward(
        grad_output._local_tensor, local_output_size, dim, index
    )

    return ShardTensor(
        local_result,
        output_spec,
        requires_grad=False,  # This will get adjusted after the dispatcher
    )


ShardTensor.register_dispatch_handler(torch.ops.aten.select.int, sharded_select_helper)
ShardTensor.register_dispatch_handler(
    torch.ops.aten.select_backward.default, sharded_select_backward_helper
)
