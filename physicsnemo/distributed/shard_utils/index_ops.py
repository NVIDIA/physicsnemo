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

from typing import Any, Tuple, Union

import torch
import wrapt

from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")

from torch.distributed.tensor.placement_types import (  # noqa: E402
    Replicate,
    Shard,
)

from physicsnemo.distributed import ShardTensor  # noqa: E402
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
            for mesh_dim, index_shard_sizes in index._spec.sharding_sizes().items():
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


@wrapt.patch_function_wrapper(
    "torch",
    "index_select",
    enabled=ShardTensor.patches_enabled,
)
def index_select_wrapper(
    wrapped: Any, instance: Any, args: tuple, kwargs: dict
) -> Union[ShardTensor, torch.Tensor]:
    """
    Wrapper for index_select operation that handles both ShardTensors and regular Tensors.

    This function dispatches to the appropriate implementation based on the input types.
    For ShardTensors, it uses sharded_index_select, otherwise falls back to torch's index_select.


    Returns
    -------
    Union[ShardTensor, torch.Tensor]
        Output tensor containing the selected elements

    Raises
    ------
    TypeError
        If the input combination is not supported
    """

    # Extract the tensor and index from the arguments
    tensor, dim, index = args

    if isinstance(tensor, ShardTensor) and isinstance(index, ShardTensor):
        return sharded_index_select(tensor, dim, index)
    elif isinstance(tensor, torch.Tensor) and isinstance(index, torch.Tensor):
        return torch.index_select(tensor, dim, index)
    else:
        raise TypeError(
            f"Unsupported input types: tensor {type(tensor)}, index {type(index)}"
        )


# def index_add_wrapper(
#     tensor: Union[ShardTensor, torch.Tensor],
#     dim: int,
#     index: Union[ShardTensor, torch.Tensor],
#     source: Union[ShardTensor, torch.Tensor],
#     alpha: float = 1.0,
# ) -> Union[ShardTensor, torch.Tensor]:
#     """
#     Wrapper for index_add operation that handles both ShardTensors and regular Tensors.

#     This function adds values from the source tensor at positions specified by index
#     to the input tensor along the specified dimension.

#     Parameters
#     ----------
#     tensor : Union[ShardTensor, torch.Tensor]
#         Input tensor to add to
#     dim : int
#         Dimension along which to index
#     index : Union[ShardTensor, torch.Tensor]
#         Indices to add at
#     source : Union[ShardTensor, torch.Tensor]
#         Tensor containing values to add
#     alpha : float, optional
#         Scaling factor for source values, by default 1.0

#     Returns
#     -------
#     Union[ShardTensor, torch.Tensor]
#         Result tensor with added values

#     Raises
#     ------
#     TypeError
#         If the input combination is not supported
#     """
#     if isinstance(tensor, torch.Tensor) and isinstance(index, torch.Tensor) and isinstance(source, torch.Tensor):
#         return tensor.index_add(dim, index, source, alpha=alpha)
#     else:
#         # This implementation is pending - current function is included to maintain API compatibility
#         raise NotImplementedError("index_add for ShardTensor is not yet implemented")
