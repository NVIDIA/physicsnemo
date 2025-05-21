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
import torch.distributed as dist
import warp as wp

from physicsnemo.models.layers.ball_query import (
    _ball_query_backward_primitive_,
    _ball_query_forward_primitive_,
    ball_query_layer,
)
from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")

from torch.distributed.tensor.placement_types import (  # noqa: E402
    Replicate,
    Shard,
)

from physicsnemo.distributed import ShardTensor, ShardTensorSpec  # noqa: E402
from physicsnemo.distributed.shard_utils.patch_core import (  # noqa: E402
    MissingShardPatch,
)
from physicsnemo.distributed.shard_utils.ring import (  # noqa: E402
    RingPassingConfig,
    perform_ring_iteration,
)

wp.config.quiet = True

__all__ = ["ball_query_layer_wrapper"]


def ring_ball_query(
    points1: ShardTensor,
    points2: ShardTensor,
    bq_kwargs: dict,
) -> Tuple[ShardTensor, ShardTensor, ShardTensor]:
    """
    Performs ball query operation on points distributed across ranks in a ring configuration.

    Args:
        points1: First set of points as a ShardTensor
        points2: Second set of points as a ShardTensor
        bq_kwargs: Keyword arguments for the ball query operation

    Returns:
        Tuple of (mapping, num_neighbors, outputs) as ShardTensors
    """
    mesh = points1._spec.mesh
    # We can be confident of this because 1D meshes are enforced
    mesh_dim = 0

    local_group = mesh.get_group(mesh_dim)
    local_size = dist.get_world_size(group=local_group)

    # Create a config object to simplify function args for message passing:
    ring_config = RingPassingConfig(
        mesh_dim=mesh_dim,
        mesh_size=local_size,
        communication_method="p2p",
        ring_direction="forward",
    )

    # Now, get the inputs locally:
    local_points1 = points1.to_local()
    local_points2 = points2.to_local()

    # Get the shard sizes for the point cloud going around the ring.
    # We've already checked that the mesh is 1D so call the '0' index.
    p2_shard_sizes = points2._spec.sharding_shapes()[0]

    # Call the differentiable version of the ring-ball-query:
    mapping_shard, num_neighbors_shard, outputs_shard = RingBallQuery.apply(
        local_points1,
        local_points2,
        mesh,
        ring_config,
        p2_shard_sizes,
        bq_kwargs,
    )

    # TODO
    # the output shapes can be computed directly from the input sharding of points1
    # Requires a little work to fish out parameters but that's it.
    # For now, using blocking inference to get the output shapes.

    # For the output shapes, we can compute the output sharding if needed.  If the placement
    # is Replicate, just infer since there aren't shardings.
    if type(points1._spec.placements[0]) == Replicate:
        map_shard_shapes = "infer"
        neighbors_shard_shapes = "infer"
        outputs_shard_shapes = "infer"
    elif type(points1._spec.placements[0]) == Shard:

        p1_shard_sizes = points1._spec.sharding_shapes()[0]

        # This conversion to shard tensor can be done explicitly computing the output shapes.

        b = mapping_shard.shape[0]
        mp = mapping_shard.shape[-1]
        d = points1.shape[-1]
        mapping_shard_output_sharding = {
            0: tuple(torch.Size([b, s[1], mp]) for s in p1_shard_sizes),
        }
        num_neighbors_shard_output_sharding = {
            0: tuple(torch.Size([b, s[1]]) for s in p1_shard_sizes),
        }
        outputs_shard_output_sharding = {
            0: tuple(torch.Size([b, s[1], mp, d]) for s in p1_shard_sizes),
        }

        map_shard_shapes = mapping_shard_output_sharding
        # map_shard_shapes = "infer"
        neighbors_shard_shapes = num_neighbors_shard_output_sharding
        outputs_shard_shapes = outputs_shard_output_sharding

    # Convert back to ShardTensor
    mapping_shard = ShardTensor.from_local(
        mapping_shard, points1._spec.mesh, points1._spec.placements, map_shard_shapes
    )
    num_neighbors_shard = ShardTensor.from_local(
        num_neighbors_shard,
        points1._spec.mesh,
        points1._spec.placements,
        neighbors_shard_shapes,
    )
    outputs_shard = ShardTensor.from_local(
        outputs_shard,
        points1._spec.mesh,
        points1._spec.placements,
        outputs_shard_shapes,
    )
    return mapping_shard, num_neighbors_shard, outputs_shard


def ringless_ball_query(
    points1: ShardTensor,
    points2: ShardTensor,
    bq_kwargs: dict,
) -> Tuple[ShardTensor, ShardTensor, ShardTensor]:
    """
    Performs ball query operation on points distributed across ranks, without a ring.
    Used when points2 is replicated (not sharded).

    points1 may or may not be sharded.

    Args:
        points1: First set of points as a ShardTensor
        points2: Second set of points as a ShardTensor
        bq_kwargs: Keyword arguments for the ball query operation

    Returns:
        Tuple of (mapping, num_neighbors, outputs) as ShardTensors
    """

    local_p1 = points1.to_local()
    local_p2 = points2.to_local()

    # if points1 is sharded, then it will compute a partial gradient of points2
    # in the backwards pass.  So, this operation will do the reduction going backward
    # by summing:
    p1_placement = points1._spec.placements[0]
    if p1_placement.is_shard():
        local_p2 = GradReducer.apply(local_p2, points2._spec)

    mapping, num_neighbors, outputs = ball_query_layer(
        local_p1,
        local_p2,
        **bq_kwargs,
    )

    k = bq_kwargs["k"]
    b = points1.shape[0]

    mapping_placement = {}
    num_neighbors_placement = {}
    outputs_placement = {}

    for k, s in points1._spec.sharding_shapes().items():
        n_points = [int(_s[1]) for _s in s]
        mapping_placement[k] = tuple(torch.Size([b, np, k]) for np in n_points)
        num_neighbors_placement[k] = tuple(torch.Size([b, np]) for np in n_points)
        outputs_placement[k] = tuple(torch.Size([b, np, k, 3]) for np in n_points)

    mapping = ShardTensor.from_local(
        mapping,
        points1._spec.mesh,
        points1._spec.placements,
        sharding_shapes=mapping_placement,
    )
    num_neighbors = ShardTensor.from_local(
        num_neighbors,
        points1._spec.mesh,
        points1._spec.placements,
        sharding_shapes=num_neighbors_placement,
    )
    outputs = ShardTensor.from_local(
        outputs,
        points1._spec.mesh,
        points1._spec.placements,
        sharding_shapes=outputs_placement,
    )

    return mapping, num_neighbors, outputs


def merge_outputs(
    current_mapping: Union[torch.Tensor, None],
    current_num_neighbors: Union[torch.Tensor, None],
    current_outputs: Union[torch.Tensor, None],
    incoming_mapping: torch.Tensor,
    incoming_num_neighbors: torch.Tensor,
    incoming_outputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform a gather/scatter operation on the mapping and outputs tensors.
    This is an _inplace_ operation on the current tensors, assuming they are not None

    Args:
        current_mapping: Current mapping tensor or None
        current_num_neighbors: Current number of neighbors tensor or None
        current_outputs: Current outputs tensor or None
        incoming_mapping: Incoming mapping tensor to merge
        incoming_num_neighbors: Incoming number of neighbors tensor to merge
        incoming_outputs: Incoming outputs tensor to merge

    Returns:
        Tuple of merged (mapping, num_neighbors, outputs) tensors
    """

    @wp.kernel
    def merge_mapping_and_outputs(
        current_m: wp.array3d(dtype=wp.int32),
        current_nn: wp.array2d(dtype=wp.int32),
        current_o: wp.array4d(dtype=wp.float32),
        incoming_m: wp.array3d(dtype=wp.int32),
        incoming_nn: wp.array2d(dtype=wp.int32),
        incoming_o: wp.array4d(dtype=wp.float32),
        max_neighbors: int,
    ):
        # This is a kernel that is essentially doing a gather/scatter operation.

        # Which points are we looking at?
        tid = wp.tid()

        # How many neighbors do we have?
        num_neighbors = current_nn[0, tid]
        available_space = max_neighbors - num_neighbors

        # How many neighbors do we have in the incoming tensor?
        incoming_num_neighbors = incoming_nn[0, tid]

        # Can't add more neighbors than we have space for:
        neighbors_to_add = min(incoming_num_neighbors, available_space)

        # Now, copy the incoming neighbors to offset locations in the current tensor:
        for i in range(neighbors_to_add):

            # incoming has no offset
            # current has offset of num_neighbors
            current_m[0, tid, num_neighbors + i] = incoming_m[0, tid, i]
            current_o[0, tid, num_neighbors + i, 0] = incoming_o[0, tid, i, 0]
            current_o[0, tid, num_neighbors + i, 1] = incoming_o[0, tid, i, 1]
            current_o[0, tid, num_neighbors + i, 2] = incoming_o[0, tid, i, 2]

        # Finally, update the number of neighbors:
        current_nn[0, tid] = num_neighbors + incoming_num_neighbors
        return

    if (
        current_mapping is None
        and current_num_neighbors is None
        and current_outputs is None
    ):
        return incoming_mapping, incoming_num_neighbors, incoming_outputs

    _, n_points, max_neighbors = current_mapping.shape

    # This is a gather/scatter operation:
    # We need to merge the incoming values into the current arrays.  The arrays
    # are essentially a ragged tensor that has been padded to a consistent shape.
    # What happens here is:
    # - Compare the available space in current tensors to the number of incoming values.
    #   - If there are more values coming in than there is space, they are truncated.
    # - Using the available space, determine the section in the incoming tensor to gather.
    # - Using the (trucated) size of incoming values, determine the region of the current tensor for scatter.
    # - gather / scatter from incoming to current.
    # - Update the current num neighbors correctly

    wp.launch(
        merge_mapping_and_outputs,
        dim=n_points,
        inputs=[
            wp.from_torch(current_mapping, return_ctype=True),
            wp.from_torch(current_num_neighbors, return_ctype=True),
            wp.from_torch(current_outputs, return_ctype=True),
            wp.from_torch(incoming_mapping, return_ctype=True),
            wp.from_torch(incoming_num_neighbors, return_ctype=True),
            wp.from_torch(incoming_outputs, return_ctype=True),
            max_neighbors,
        ],
    )

    return current_mapping, current_num_neighbors, current_outputs


class RingBallQuery(torch.autograd.Function):
    """
    Custom autograd function for performing ball query operations in a distributed ring configuration.

    Handles the forward pass of ball queries across multiple ranks, enabling distributed computation
    of nearest neighbors for point clouds.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        points1: torch.Tensor,
        points2: torch.Tensor,
        mesh: Any,
        ring_config: RingPassingConfig,
        shard_sizes: list,
        bq_kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for distributed ball query computation.

        Args:
            ctx: Context for saving variables for backward pass
            points1: First set of points
            points2: Second set of points
            lengths1: Lengths of each batch in points1
            lengths2: Lengths of each batch in points2
            mesh: Distribution mesh specification
            ring_config: Configuration for ring passing
            shard_sizes: Sizes of each shard across ranks
            wrapped: The original ball query function
            *args: Additional positional arguments for the wrapped function
            **kwargs: Additional keyword arguments for the wrapped function

        Returns:
            Tuple of (mapping, num_neighbors, outputs) tensors
        """
        ctx.mesh = mesh
        ctx.ring_config = ring_config

        # Create buffers to store outputs
        current_mapping = None
        current_num_neighbors = None
        current_outputs = None

        # For the first iteration, use local tensors
        current_p1, current_p2 = (points1, points2)

        mesh_rank = mesh.get_local_rank()

        # Get all the ranks in the mesh:
        world_size = ring_config.mesh_size

        # Store results from each rank to merge in the correct order
        rank_results = [None] * world_size
        # For uneven point clouds, the global stide is important:
        strides = [s[1] for s in shard_sizes]

        ctx.k = bq_kwargs["k"]
        ctx.radius = bq_kwargs["radius"]
        ctx.hash_grid = bq_kwargs["hash_grid"]

        for i in range(world_size):

            source_rank = (mesh_rank - i) % world_size

            (
                local_mapping,
                local_num_neighbors,
                local_outputs,
            ) = _ball_query_forward_primitive_(
                current_p1[0],
                current_p2[0],
                ctx.k,
                ctx.radius,
                ctx.hash_grid,
            )
            # Store the result with its source rank
            rank_results[source_rank] = (
                local_mapping,
                local_num_neighbors,
                local_outputs,
            )

            # For point clouds, we need to pass the size of the incoming shard.
            next_source_rank = (source_rank - 1) % world_size

            # TODO - this operation should be done async and checked for completion at the start of the next loop.
            if i != world_size - 1:
                # Don't do a ring on the last iteration.
                current_p2 = perform_ring_iteration(
                    current_p2,
                    ctx.mesh,
                    ctx.ring_config,
                    recv_shape=shard_sizes[next_source_rank],
                )

        # Now merge the results in rank order (0, 1, 2, ...)
        stride = 0
        for r in range(world_size):
            if rank_results[r] is not None:
                local_mapping, local_num_neighbors, local_outputs = rank_results[r]

                current_mapping, current_num_neighbors, current_outputs = merge_outputs(
                    current_mapping,
                    current_num_neighbors,
                    current_outputs,
                    local_mapping + stride,
                    local_num_neighbors,
                    local_outputs,
                )

                stride += strides[r]
        ctx.save_for_backward(
            points1, points2, current_mapping, current_num_neighbors, current_outputs
        )

        return current_mapping, current_num_neighbors, current_outputs

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        mapping_grad: torch.Tensor,
        num_neighbors_grad: torch.Tensor,
        outputs_grad: torch.Tensor,
    ) -> Tuple[None, ...]:
        """
        Backward pass for distributed ring ball query computation.

        Args:
            ctx: Context containing saved variables from forward pass
            grad_output: Gradients from subsequent layers

        Returns:
            Gradients for inputs (currently not implemented)
        """

        raise MissingShardPatch("Backward pass for ring ball query not implemented.")

        (
            points1,
            points2,
            current_mapping,
            current_num_neighbors,
            current_outputs,
        ) = ctx.saved_tensors

        # We need to do a ring again in the backward direction.
        # The backward pass is computed locally, and then the gradients
        # and p2 are moved along the ring together.
        # for i in range(world_size):
        # Calculate which source rank this data is from

        local_p2_grad = _ball_query_backward_primitive_(
            points1[0],
            points2[0],
            current_mapping,
            current_num_neighbors,
            current_outputs,
            mapping_grad,
            num_neighbors_grad,
            outputs_grad,
        )
        local_p1_grad = torch.zeros_like(points1)

        return (
            local_p1_grad,
            local_p2_grad.unsqueeze(0),
            None,
            None,
            None,
            None,
        )


class GradReducer(torch.autograd.Function):
    """
    A custom autograd function that performs an allreduce on the gradients if they are sharded
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        spec: ShardTensorSpec,
    ) -> torch.Tensor:
        ctx.spec = spec
        return input

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:

        spec = ctx.spec
        placement = spec.placements[0]
        # Perform an allreduce on the gradient
        if placement.is_replicate():
            dist.all_reduce(
                grad_output, op=dist.ReduceOp.SUM, group=spec.mesh.get_group(0)
            )
        return grad_output, None


def ball_query_layer_wrapper(
    func: Any, type: Any, args: tuple, kwargs: dict
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[ShardTensor, ShardTensor, ShardTensor],
]:
    """
    Wrapper for BallQueryLayer.forward to support sharded tensors.

    Handles 4 situations, based on the sharding of points 1 and points 2:
    - Points 2 is sharded: a ring computation is performed.
        - Points 1 is sharded: each rank contains a partial output,
          which is returned sharded like Points 1.
        - Points 1 is replicated: each rank returns the full output,
          even though the input points 2 is sharded.
    - Points 1 is replicated: No ring is needed.
        - Points 1 is sharded: each rank contains a partial output,
          which is returned sharded like Points 1.
        - Points 1 is replicated: each rank returns the full output,
          even though the input points 2 is sharded.

    All input sharding has to be over a 1D mesh.  2D Point cloud sharding
    is not supported at this time.

    Regardless of the input sharding, the output will always be sharded like
    points 1, and the output points will always have queried every input point
    like in the non-sharded case.

    Args:
        func: Original forward method
        type: Types of the inputs
        args: Positional arguments (points1, points2)
        kwargs: Keyword arguments

    Returns:
        Tuple of (mapping, num_neighbors, outputs) as torch.Tensor or ShardTensor
    """

    points1, points2, bq_kwargs = repackage_ball_query_args(*args, **kwargs)

    # Make sure all meshes are the same
    if points1._spec.mesh != points2._spec.mesh:
        raise MissingShardPatch(
            "point_cloud_ops.ball_query_layer_wrapper: All point inputs must be on the same mesh"
        )

    # make sure all meshes are 1D
    if points1._spec.mesh.ndim != 1:
        raise MissingShardPatch(
            "point_cloud_ops.ball_query_layer_wrapper: All point inputs must be on 1D meshes"
        )

    # Do we need a ring?
    points2_placement = points2._spec.placements[0]
    if points2_placement.is_shard():
        # We need a ring
        mapping, num_neighbors, outputs = ring_ball_query(points1, points2, bq_kwargs)
    else:
        # No ring is needed
        mapping, num_neighbors, outputs = ringless_ball_query(
            points1, points2, bq_kwargs
        )

    return mapping, num_neighbors, outputs


def repackage_ball_query_args(
    points1: Union[torch.Tensor, ShardTensor],
    points2: Union[torch.Tensor, ShardTensor],
    k: int,
    radius: float,
    hash_grid: wp.HashGrid,
    *args: Any,
    **kwargs: Any,
) -> Tuple[
    Union[torch.Tensor, ShardTensor],
    Union[torch.Tensor, ShardTensor],
    Union[torch.Tensor, ShardTensor],
    Union[torch.Tensor, ShardTensor],
    dict,
]:
    """Repackages ball query arguments into a standard format.

    Takes the arguments that could be passed to a ball query operation
    and separates them into core tensor inputs (points1, points2, lengths1, lengths2)
    and configuration parameters packaged as a kwargs dict.

    Args:
        points1: First set of points
        points2: Second set of points
        lengths1: Lengths of each batch in points1
        lengths2: Lengths of each batch in points2
        *args: Additional positional args
        **kwargs: Additional keyword args

    Returns:
        Tuple containing:
        - points1 tensor
        - points2 tensor
        - Dict of ball query configuration parameters
    """
    # Extract any additional parameters that might be in kwargs
    # or use defaults if not provided
    return_kwargs = {
        "k": k,
        "radius": radius,
        "hash_grid": hash_grid,
    }

    # Add any explicitly passed parameters
    if kwargs:
        return_kwargs.update(kwargs)

    return points1, points2, return_kwargs


ShardTensor.register_function_handler(ball_query_layer, ball_query_layer_wrapper)
