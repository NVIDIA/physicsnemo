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

from itertools import accumulate
from typing import List, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh

from physicsnemo.distributed.autograd import (
    all_gather_v,
)
from physicsnemo.utils.version_check import check_module_requirements

# This is to make sure the torch minimum version is installed.
check_module_requirements("physicsnemo.distributed.shard_tensor")

from torch.distributed.tensor._dtensor_spec import (  # noqa: E402
    TensorMeta,
)
from torch.distributed.tensor._redistribute import (  # noqa: E402
    _gen_transform_infos,
)
from torch.distributed.tensor.placement_types import (  # noqa: E402
    Partial,
    Placement,
    Replicate,
    Shard,
)

import physicsnemo.distributed.shard_tensor as shard_tensor  # noqa: E402
from physicsnemo.distributed._shard_tensor_spec import ShardTensorSpec  # noqa: E402

# TODO:
# DTensor makes assumptions about sharding sizes.
# I need to figure out the target spec  manually, based on input/output placements.
# I'm already intercepting the collectives and using the right input sizes.
# But the output placements are containing the wrong sharding sizes.
# It should all "just work" once that's fixed.


# Worker functions for the collectives specific to uneven shaped tensors:
def _to_replicate_tensor(
    local_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    mesh_dim: int,
    tensor_dim: int,
    current_spec: ShardTensorSpec,
) -> torch.Tensor:
    """
    Converts a sharded tensor to a replicated tensor by gathering all shards.

    Args:
        local_tensor (torch.Tensor): The local shard of the tensor to replicate
        device_mesh (DeviceMesh): The device mesh containing process groups
        mesh_dim (int): The mesh dimension along which to gather
        tensor_dim (int): The tensor dimension along which data is sharded
        current_spec (ShardTensorSpec): Specification of current sharding scheme

    Returns:
        torch.Tensor: The fully replicated tensor on this rank

    Note:
        This function handles uneven sharding by using all_gather_v instead of regular all_gather
    """

    # Get the mesh for the group:
    mesh = current_spec.mesh
    group = mesh.get_group(mesh_dim)

    # Get all sizes:
    sizes = current_spec.sharding_sizes()
    this_sizes = tuple(s[tensor_dim] for s in sizes[mesh_dim])
    # # Ensure contiguous data for the reduction:
    local_tensor = local_tensor.contiguous()
    # We can implement this with a straightforward allgather_v
    local_tensor = all_gather_v(
        local_tensor, sizes=this_sizes, dim=tensor_dim, group=group
    )

    return local_tensor


def _select_slice_from_replicate(
    local_tensor: torch.Tensor,
    target_spec: ShardTensorSpec,
    mesh_dim: int,
    mesh_coord: int,
    sizes: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Selects the appropriate slice from a replicated tensor to create a shard.

    Args:
        local_tensor (torch.Tensor): The replicated tensor to slice from
        target_spec (ShardTensorSpec): Specification of target sharding scheme
        mesh_dim (int): The mesh dimension along which to shard
        mesh_coord (int): The coordinate of this rank in the mesh dimension

    Returns:
        torch.Tensor: The selected slice that will become this rank's shard

    Note:
        This function handles uneven sharding by using the sharding sizes from the target spec
        to split the tensor into potentially uneven chunks
    """

    # TODO - This needs a rework to enable caching of shapes for a grad pass.

    # We really only need the sizes from this dimension:
    tensor_dim = target_spec.placements[mesh_dim].dim
    mesh_size = target_spec.mesh.size(mesh_dim=mesh_dim)

    # Can we use the size hint here?
    if sizes is not None and len(sizes) != mesh_size:
        sizes = None

    # Split the tensor:
    if sizes is None:
        chunks = torch.tensor_split(local_tensor, mesh_size, dim=tensor_dim)
    else:
        chunks = torch.tensor_split(local_tensor, sizes[:-1], dim=tensor_dim)

    return chunks[mesh_coord], sizes


def _to_new_shard_dim(
    local_tensor: torch.Tensor,
    target_spec: ShardTensorSpec,
    mesh_dim: int,  # the device mesh dimensionwe're transposing on.
    size_hint: Optional[
        Tuple[int, ...]
    ],  # If provided, use this to chunk the tensor - both send and recv
    current_dim: int,  # currently sharded on this tensor dimension
    target_dim: int,  # Want to be sharded on this tensor dimension
) -> torch.Tensor:

    # We're essentially transposing the tensor here.
    # We could implement this as an all_gather_v / scatter_v, but
    # it's more efficient to do an all_to_all.

    device_mesh = target_spec.mesh
    mesh_size = device_mesh.size(mesh_dim=mesh_dim)
    group = device_mesh.get_group(mesh_dim=mesh_dim)

    # To use the size hint, and preserve the original sharding, we need to insist that
    # the mesh_size and the length of size hint is equal
    if size_hint is not None and mesh_size != len(size_hint):
        # Setting to None will prevent it being used further
        size_hint = None

    # First, we need to split the tensor along the target dimension:
    if size_hint is None:
        chunks = torch.tensor_split(local_tensor, mesh_size, dim=target_dim)
    else:
        chunk_starts = list(accumulate(size_hint))
        chunks = torch.tensor_split(local_tensor, chunk_starts[:-1], dim=target_dim)

    # MUST be contiguous for all_to_all:
    # Also, cast to list for all_to_all:
    chunks = [c.contiguous() for c in chunks]

    # TODO - remove this all_to_all by enabling recv shape from known information.

    send_shapes = [
        torch.tensor(c.shape, device=local_tensor.device, dtype=torch.int32)
        for c in chunks
    ]
    recv_shapes = [torch.empty_like(s) for s in send_shapes]

    # Gather the send shape from every rank:
    # For all to all, we _have_ to send and receive from every rank.
    # But we can optimize the null-communication
    dist.all_to_all(recv_shapes, send_shapes, group=group)

    # Turn the recv_shapes back into torch shapes:
    recv_shapes = [list(torch.Size(r)) for r in recv_shapes]

    # Create the buffers for recv:
    recv_buffers = [
        torch.empty(shape, device=local_tensor.device, dtype=local_tensor.dtype)
        for shape in recv_shapes
    ]

    # chunks is the send buffer.
    dist.all_to_all(recv_buffers, chunks, group=group)

    # Take the received tensors and stack them along the target dimension:
    stacked_tensor = torch.cat(recv_buffers, dim=current_dim).contiguous()

    # Return the size hint in case we discarded it
    return stacked_tensor, size_hint


def redistribute_local_shard_tensor(
    local_tensor: torch.Tensor,
    current_spec: ShardTensorSpec,
    target_spec: ShardTensorSpec,
    *,
    async_op: bool = False,
    is_backward: bool = False,
    target_sharding_sizes: Optional[dict[int, Tuple[torch.Size, ...]]] = {},
) -> torch.Tensor:
    """
    This redistribute the local tensor (torch.Tensor) from the current ShardTensorSpec to
    the target ShardTensorSpec, which involves the necessary collective calls to transform
    the local shard of the ShardTensor from its current spec to the target spec.

    The collective operations are implemented in the Placement classes, which we avoid
    modifying.  To get around that, we mimic the logic from pytorch's original redistribute.
    But, in cases where a tensor is Sharded and the shards are uneven (spec.is_uneven)
    we intercept and replace the collectives:

    ``Shard(dim)`` -> ``Replicate()``: ``all_gather_v`` instead of ``all_gather``
    ``Shard(src_dim)`` -> ``Shard(dst_dim)``: remains all_to_all but reimplemented to handle sizes correctly
    ``Replicate()`` -> ``Shard(dim)``: local chunking is **unchanged** but return value is ShardTensorSpec instead.
    ``Partial()`` -> ``Replicate()``: ``all_reduce``needs to become a weighted ``all_reduce``, depending on operation.
    ``Partial()`` -> ``Shard(dim)``: ``reduce_scatter`` needs to become a weighted ``reduce_scatter``, depending on operation
    """

    if current_spec.mesh != target_spec.mesh:
        # TODO: alltoall/permute reshuffling to change device_mesh if they are not the same
        raise NotImplementedError("Cross device mesh comm not supported yet!")

    new_local_tensor = None
    device_mesh = current_spec.mesh

    my_coordinate = device_mesh.get_coordinate()

    if my_coordinate is None:
        # if rank is not part of mesh, we skip redistribute and simply return local_tensor,
        # which should be an empty tensor
        return local_tensor

    # This is an internal-focused step.  If the target_spec has the same placements and mesh
    # as the current, but is missing sharding sizes, we can use the current spec's sharding sizes.
    # if target_spec._sharding_sizes is None:
    #     if target_spec.placements == current_spec.placements and target_spec.mesh == current_spec.mesh:
    #         target_spec._sharding_sizes = current_spec.sharding_sizes()

    # For sharded tensors, we use the same order of transformation as DTensor.
    # However, often we need to ignore the provided logical shape and substitute
    # a sharded shape instead.
    # This is done by providing a target_sharding_sizes dict above.

    transform_infos = _gen_transform_infos(current_spec, target_spec)

    if len(transform_infos) == 0:
        return local_tensor

    for transform_info in transform_infos:
        dist.barrier()
        i = transform_info.mesh_dim
        current, target = transform_info.src_dst_placements
        device_mesh.size(mesh_dim=i)

        if current == target:
            # short cut, just use the original local tensor
            new_local_tensor = local_tensor
            continue

        # logger.debug("redistribute from %s to %s on mesh dim %s", current, target, i)
        if target.is_replicate():
            # Case 1: target is Replicate
            if current.is_partial():
                partial_spec = cast(Partial, current)
                new_local_tensor = partial_spec._reduce_value(
                    local_tensor, device_mesh, i
                )
            elif current.is_shard():
                current_placement = cast(Shard, current)
                new_local_tensor = _to_replicate_tensor(
                    local_tensor,
                    device_mesh,
                    mesh_dim=i,
                    tensor_dim=current_placement.dim,
                    current_spec=current_spec,
                )
            else:
                raise RuntimeError(
                    f"redistribute from {current} to {target} not supported yet"
                )
        elif target.is_shard():
            # Case 2: target is Shard
            target_placement = cast(Shard, target)
            if current.is_partial():
                partial_spec = cast(Partial, current)
                new_local_tensor = partial_spec._reduce_shard_value(
                    local_tensor, device_mesh, i, target_placement
                )
            elif current.is_replicate():
                # split the tensor and return the corresponding cloned local shard
                # Are there suggested placements for the shards?
                if target_placement.dim in target_sharding_sizes:
                    size_hint = target_sharding_sizes[target_placement.dim]
                else:
                    size_hint = None
                new_local_tensor, size_hint = _select_slice_from_replicate(
                    local_tensor,
                    target_spec,
                    i,
                    my_coordinate[i],
                    size_hint,
                )
                if (
                    size_hint is not None
                    and target_placement.dim in target_sharding_sizes
                ):
                    target_sharding_sizes[target_placement.dim] = size_hint

            else:
                if not current.is_shard():
                    raise RuntimeError(
                        f"Current placement should be shard but found {current}"
                    )
                shard_spec = cast(Shard, current)
                if shard_spec.dim != target_placement.dim:
                    # Here we need to essentially transpose the tensor along two dimensions.
                    # We cached shardings that appear in both the input and output shards, along tensor dimensions.
                    # So, if the target tensor dimension is in there,
                    # That is how we're going to shard the local tensor on the tensor_dim,
                    # and it also defines how we'll receive the tensor .
                    if target_placement.dim in target_sharding_sizes:
                        size_hint = target_sharding_sizes[target_placement.dim]
                    else:
                        size_hint = None

                    new_local_tensor, size_hint = _to_new_shard_dim(
                        local_tensor,
                        target_spec,  # Send the whole spec so we can infer full recv sizes.
                        i,  # The mesh dim we're transposing sharding on.
                        size_hint,
                        current.dim,  # Current tensor dimension.
                        target_placement.dim,  # Target tensor dimension.
                    )
                    if (
                        size_hint is None
                        and target_placement.dim in target_sharding_sizes
                    ):
                        target_sharding_sizes.pop(target_placement.dim)
                    if size_hint is not None and current.dim in target_sharding_sizes:
                        target_sharding_sizes.pop(current.dim)

        elif target.is_partial():
            if current.is_replicate():
                partial_spec = cast(Partial, target)
                # skip the replicate to partial transformation when we are in backward pass
                # In this case we keep the grad as replicate, this is because we don't
                # want to convert the replicated gradients back to partial, although
                # that's logically conform with the same layout, converting the gradients
                # back to partial is actually useless as you would have to do reduce later
                # which would be more expensive than keeping it replicate! For this reason,
                # we keep the replicate grad here.
                new_local_tensor = (
                    partial_spec._partition_value(local_tensor, device_mesh, i)
                    if not is_backward
                    else local_tensor
                )
            elif current.is_shard():
                if not is_backward:
                    raise RuntimeError(
                        f"redistribute from {current} to {target} not supported yet"
                    )
                # for backward shard -> partial, we just need to convert the shard to replicate
                current_placement = cast(Shard, current)
                # TODO - resolve sharding to partials?
                new_local_tensor = current_placement._to_replicate_tensor(
                    local_tensor, device_mesh, i, transform_info.logical_shape
                )
            else:
                # partial -> partial no op, should never hit
                new_local_tensor = local_tensor

        if new_local_tensor is None:
            raise RuntimeError(
                "Failed to create new local tensor during redistribution"
            )
        local_tensor = new_local_tensor

    if new_local_tensor is None:
        raise RuntimeError("redistribute failed!")

    if not async_op and isinstance(new_local_tensor, funcol.AsyncCollectiveTensor):
        new_local_tensor = new_local_tensor.wait()

    return new_local_tensor


def get_tensor_sharding_shapes_by_dim(
    current_spec: ShardTensorSpec,
    target_placements: Tuple[Placement, ...],
) -> ShardTensorSpec:
    """
    Generate a target spec from the current spec and target_placements.
    """

    target_sharding_sizes = {}
    # Look through the target placements for shardings:
    for target_mesh_dim, target_placement in enumerate(target_placements):
        if isinstance(target_placement, Shard):
            # If the target tensor dim is in the current target_placements,
            # Maintain that sharding.
            target_tensor_dim = target_placement.dim
            # Find if this tensor dim is in the current spec's placements:
            for current_mesh_dim, current_placement in enumerate(
                current_spec.placements
            ):
                if (
                    isinstance(current_placement, Shard)
                    and target_tensor_dim == current_placement.dim
                ):
                    # The tensor dim is the same in both current and target,
                    # But the rest of the tensors dimensions may change.
                    # Therefore only save the dimension on this axis.
                    current_shardings = current_spec.sharding_sizes()[current_mesh_dim]
                    target_sharding_sizes[target_tensor_dim] = [
                        c[target_tensor_dim] for c in current_shardings
                    ]

    return target_sharding_sizes


class ShardRedistribute(torch.autograd.Function):
    """
    This is a ShardTensor enhanced version of redistribute. It extends
    the functionality in DTensor to allow redistribution of sharded tensors.

    This autograd function handles both forward and backward passes for redistributing
    sharded tensors between different sharding schemes.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: "shard_tensor.ShardTensor",
        device_mesh: DeviceMesh,
        placements: Tuple[Placement, ...],
        async_op: bool = False,
    ) -> "shard_tensor.ShardTensor":
        """
        Forward pass for redistributing a sharded tensor.

        Args:
            ctx: Autograd context for saving tensors/variables for backward
            input: Input sharded tensor to redistribute
            device_mesh: Target device mesh for redistribution
            placements: Target placement scheme for redistribution
            async_op: Whether to perform redistribution asynchronously

        Returns:
            Redistributed sharded tensor with new placement scheme
        """
        current_spec = input._spec

        ctx.current_spec = current_spec
        ctx.async_op = async_op

        if current_spec.placements != placements:

            # We have to assume, here, that the current spec has correct sharding_sizes.
            # Therefore, we can use the target placement + current sharding_sizes
            # to get the target sharding sizes correctly.

            # target_spec = generate_target_spec_from_current_and_placements(
            #     current_spec,
            #     placements,
            # )

            target_spec = ShardTensorSpec(
                device_mesh,
                placements,
                tensor_meta=input._spec.tensor_meta,
            )

            # The target sharding sizes are potentially incomplete.
            # They're only provided for shardings that are the same in input/output.
            target_sharding_sizes = get_tensor_sharding_shapes_by_dim(
                current_spec, placements
            )
            # ctx.target_sharding_sizes = target_sharding_sizes
            local_tensor = input._local_tensor
            output = redistribute_local_shard_tensor(
                local_tensor,
                current_spec,
                target_spec,
                async_op=async_op,
                target_sharding_sizes=target_sharding_sizes,
            )
            # Set the local shape:
            target_spec._local_shape = output.shape
        else:
            # use the same local tensor if placements are the same.
            output = input._local_tensor
            target_spec = current_spec

        return shard_tensor.ShardTensor(
            output,
            target_spec,
            requires_grad=input.requires_grad,
        )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: "shard_tensor.ShardTensor",
    ) -> Tuple["shard_tensor.ShardTensor", None, None, None]:
        """
        Backward pass for redistributing a sharded tensor.

        Args:
            ctx: Autograd context containing saved tensors/variables from forward
            grad_output: Gradient output tensor to redistribute back

        Returns:
            Tuple containing:
            - Redistributed gradient tensor
            - None for device_mesh gradient (not needed)
            - None for placements gradient (not needed)
            - None for async_op gradient (not needed)
        """
        previous_spec = ctx.current_spec
        current_spec = grad_output._spec

        async_op = ctx.async_op

        local_tensor = grad_output._local_tensor
        target_sharding_sizes = get_tensor_sharding_shapes_by_dim(
            previous_spec, previous_spec.placements
        )

        output = redistribute_local_shard_tensor(
            local_tensor,
            current_spec,
            previous_spec,
            async_op=async_op,
            is_backward=True,
            target_sharding_sizes=target_sharding_sizes,
        )

        # normalize the target placement to replicate if it is partial
        normalized_placements: List[Placement] = []
        for previous_placement in previous_spec.placements:
            if previous_placement.is_partial():
                # keep target placement to replicate instead of partial in this case
                normalized_placements.append(Replicate())
            else:
                normalized_placements.append(previous_placement)

        spec = ShardTensorSpec(
            previous_spec.device_mesh,
            tuple(normalized_placements),
            tensor_meta=TensorMeta(
                shape=grad_output.shape,
                stride=grad_output.stride(),
                dtype=grad_output.dtype,
            ),
            _local_shape=output.shape,
        )
        output_shard_tensor = shard_tensor.ShardTensor(
            output,
            spec,
            requires_grad=grad_output.requires_grad,
        )

        return (
            output_shard_tensor,
            None,
            None,
            None,
        )
