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

from physicsnemo.utils.neighbors.radius_search._warp_impl import radius_search_impl
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


def ring_ball_query(
    points: ShardTensor,
    queries: ShardTensor,
    bq_kwargs: dict,
) -> Tuple[ShardTensor, ShardTensor, ShardTensor]:
    """
    Performs ball query operation on points distributed across ranks in a ring configuration.

    Args:
        points: First set of points as a ShardTensor
        queries: Second set of points as a ShardTensor
        bq_kwargs: Keyword arguments for the ball query operation

    Returns:
        Tuple of (mapping, num_neighbors, outputs) as ShardTensors
    """
    mesh = points._spec.mesh
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
    local_points = points.to_local()
    local_queries = queries.to_local()

    # Get the shard sizes for the point cloud going around the ring.
    # We've already checked that the mesh is 1D so call the '0' index.

    points_shard_sizes = points._spec.sharding_shapes()[0]

    # Call the differentiable version of the ring-ball-query:
    indices_shard, outputs_shard, _, num_neighbors_shard = RingBallQuery.apply(
        local_points,
        local_queries,
        mesh,
        ring_config,
        points_shard_sizes,
        bq_kwargs,
    )

    # TODO
    # the output shapes can be computed directly from the input sharding of queries
    # Requires a little work to fish out parameters but that's it.
    # For now, using blocking inference to get the output shapes.

    # For the output shapes, we can compute the output sharding if needed.  If the placement
    # is Replicate, just infer since there aren't shardings.
    if isinstance(queries._spec.placements[0], Replicate):
        indices_shard_shapes = "infer"
        neighbors_shard_shapes = "infer"
        outputs_shard_shapes = "infer"
    elif isinstance(queries._spec.placements[0], Shard):
        queries_shard_sizes = queries._spec.sharding_shapes()[0]

        # This conversion to shard tensor can be done explicitly computing the output shapes.

        mp = indices_shard.shape[-1]
        d = queries.shape[-1]
        indices_shard_output_sharding = {
            0: tuple(torch.Size([s[0], mp]) for s in queries_shard_sizes),
        }
        num_neighbors_shard_output_sharding = {
            0: tuple(torch.Size([s[0]]) for s in queries_shard_sizes),
        }
        outputs_shard_output_sharding = {
            0: tuple(torch.Size([s[0], mp, d]) for s in queries_shard_sizes),
        }

        indices_shard_shapes = indices_shard_output_sharding
        neighbors_shard_shapes = num_neighbors_shard_output_sharding
        outputs_shard_shapes = outputs_shard_output_sharding

    # Convert back to ShardTensor
    indices_shard = ShardTensor.from_local(
        indices_shard,
        queries._spec.mesh,
        queries._spec.placements,
        indices_shard_shapes,
    )
    num_neighbors_shard = ShardTensor.from_local(
        num_neighbors_shard,
        queries._spec.mesh,
        queries._spec.placements,
        neighbors_shard_shapes,
    )
    outputs_shard = ShardTensor.from_local(
        outputs_shard,
        queries._spec.mesh,
        queries._spec.placements,
        outputs_shard_shapes,
    )
    return indices_shard, outputs_shard, None, num_neighbors_shard


def ringless_ball_query(
    points: ShardTensor,
    queries: ShardTensor,
    bq_kwargs: dict,
) -> Tuple[ShardTensor, ShardTensor, ShardTensor]:
    """
    Performs ball query operation on queries distributed across ranks, without a ring.
    Used when points is replicated (not sharded).

    queries may or may not be sharded.  Outputs will match queries for sharding.

    Args:
        points: First set of points as a ShardTensor
        queries: Second set of points as a ShardTensor
        bq_kwargs: Keyword arguments for the ball query operation

    Returns:
        Tuple of (mapping, num_neighbors, outputs) as ShardTensors
    """

    local_points = points.to_local()
    local_queries = queries.to_local()

    # if queries is sharded, then it will compute a partial gradient of queries
    # in the backwards pass.  So, this operation will do the reduction going backward
    # by summing:
    queries_placement = queries._spec.placements[0]
    if queries_placement.is_shard():
        local_points = GradReducer.apply(local_points, queries._spec)

    local_indices, local_points, _, local_num_neighbors = radius_search_impl(
        local_points,
        local_queries,
        **bq_kwargs,
    )

    max_points = bq_kwargs["max_points"]

    indices_placement = {}
    num_neighbors_placement = {}
    output_points_placement = {}

    # Output sharding should match the query shapes:
    for i_dim, s in queries._spec.sharding_shapes().items():
        n_points = [int(_s[0]) for _s in s]
        indices_placement[i_dim] = tuple(
            torch.Size([np, max_points]) for np in n_points
        )
        num_neighbors_placement[i_dim] = tuple(torch.Size([np]) for np in n_points)
        output_points_placement[i_dim] = tuple(
            torch.Size([np, max_points, 3]) for np in n_points
        )

    indices = ShardTensor.from_local(
        local_indices,
        queries._spec.mesh,
        queries._spec.placements,
        sharding_shapes=indices_placement,
    )
    num_neighbors = ShardTensor.from_local(
        local_num_neighbors,
        queries._spec.mesh,
        queries._spec.placements,
        sharding_shapes=num_neighbors_placement,
    )
    output_points = ShardTensor.from_local(
        local_points,
        queries._spec.mesh,
        queries._spec.placements,
        sharding_shapes=output_points_placement,
    )

    return indices, num_neighbors, output_points


def merge_outputs(
    current_indices: Union[torch.Tensor, None],
    current_num_neighbors: Union[torch.Tensor, None],
    current_points: Union[torch.Tensor, None],
    incoming_indices: torch.Tensor,
    incoming_num_neighbors: torch.Tensor,
    incoming_points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform a gather/scatter operation on the mapping and outputs tensors.
    This is an _inplace_ operation on the current tensors, assuming they are not None

    Args:
        current_indices: Current mapping tensor or None
        current_num_neighbors: Current number of neighbors tensor or None
        current_points: Current outputs tensor or None
        incoming_indices: Incoming mapping tensor to merge
        incoming_num_neighbors: Incoming number of neighbors tensor to merge
        incoming_points: Incoming outputs tensor to merge

    Returns:
        Tuple of merged (mapping, num_neighbors, outputs) tensors
    """

    @wp.kernel
    def merge_indices_and_points(
        current_m: wp.array2d(dtype=wp.int32),
        current_nn: wp.array(dtype=wp.int32),
        current_o: wp.array3d(dtype=wp.float32),
        incoming_m: wp.array2d(dtype=wp.int32),
        incoming_nn: wp.array(dtype=wp.int32),
        incoming_o: wp.array3d(dtype=wp.float32),
        max_neighbors: int,
    ):
        # This is a kernel that is essentially doing a ragged concat + truncate

        # Which points are we looking at?
        tid = wp.tid()

        # How many neighbors do we have?
        num_neighbors = current_nn[tid]
        available_space = max_neighbors - num_neighbors

        # How many neighbors do we have in the incoming tensor?
        incoming_num_neighbors = incoming_nn[tid]

        # Can't add more neighbors than we have space for:
        neighbors_to_add = min(incoming_num_neighbors, available_space)
        # Now, copy the incoming neighbors to offset locations in the current tensor:
        for i in range(neighbors_to_add):
            # incoming has no offset
            # current has offset of num_neighbors
            current_m[tid, num_neighbors + i] = incoming_m[tid, i]
            current_o[tid, num_neighbors + i, 0] = incoming_o[tid, i, 0]
            current_o[tid, num_neighbors + i, 1] = incoming_o[tid, i, 1]
            current_o[tid, num_neighbors + i, 2] = incoming_o[tid, i, 2]

        # Finally, update the number of neighbors:
        current_nn[tid] = num_neighbors + neighbors_to_add
        return

    if (
        current_indices is None
        and current_num_neighbors is None
        and current_points is None
    ):
        return incoming_indices, incoming_num_neighbors, incoming_points

    n_points, max_neighbors = current_indices.shape

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

    stream = wp.stream_from_torch(current_indices.device)
    wp.launch(
        merge_indices_and_points,
        dim=n_points,
        inputs=[
            wp.from_torch(current_indices, return_ctype=True),
            wp.from_torch(current_num_neighbors, return_ctype=True),
            wp.from_torch(current_points, return_ctype=True),
            wp.from_torch(incoming_indices, return_ctype=True),
            wp.from_torch(incoming_num_neighbors, return_ctype=True),
            wp.from_torch(incoming_points, return_ctype=True),
            max_neighbors,
        ],
        stream=stream,
    )

    return current_indices, current_num_neighbors, current_points


class RingBallQuery(torch.autograd.Function):
    """
    Custom autograd function for performing ball query operations in a distributed ring configuration.

    Handles the forward pass of ball queries across multiple ranks, enabling distributed computation
    of nearest neighbors for point clouds.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        points: torch.Tensor,
        queries: torch.Tensor,
        mesh: Any,
        ring_config: RingPassingConfig,
        shard_sizes: list,
        bq_kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for distributed ball query computation.

        Args:
            ctx: Context for saving variables for backward pass
            points: First set of points
            queries: Second set of points
            lengths1: Lengths of each batch in points
            lengths2: Lengths of each batch in queries
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
        current_indices = None
        current_num_neighbors = None
        current_out_points = None

        # For the first iteration, use local tensors
        current_points, current_queries = (points, queries)

        mesh_rank = mesh.get_local_rank()

        # Get all the ranks in the mesh:
        world_size = ring_config.mesh_size

        # Store results from each rank to merge in the correct order
        rank_results = [None] * world_size
        # For uneven point clouds, the global stide is important:
        strides = [s[0] for s in shard_sizes]

        ctx.max_points = bq_kwargs["max_points"]
        ctx.radius = bq_kwargs["radius"]
        ctx.return_dists = bq_kwargs["return_dists"]
        ctx.return_points = bq_kwargs["return_points"]

        for i in range(world_size):
            source_rank = (mesh_rank - i) % world_size

            (
                local_indices,
                local_points,
                _,
                local_num_neighbors,
            ) = radius_search_impl(
                current_points,
                current_queries,
                ctx.radius,
                ctx.max_points,
                ctx.return_dists,
                ctx.return_points,
            )
            # Store the result with its source rank
            rank_results[source_rank] = (
                local_indices,
                local_points,
                local_num_neighbors,
            )

            # For point clouds, we need to pass the size of the incoming shard.
            next_source_rank = (source_rank - 1) % world_size

            # TODO - this operation should be done async and checked for completion at the start of the next loop.
            if i != world_size - 1:
                # Don't do a ring on the last iteration.
                current_points = perform_ring_iteration(
                    current_points,
                    ctx.mesh,
                    ctx.ring_config,
                    recv_shape=shard_sizes[next_source_rank],
                )

        # Now merge the results in rank order (0, 1, 2, ...)
        stride = 0
        for r in range(world_size):
            if rank_results[r] is not None:
                local_indices, local_points, local_num_neighbors = rank_results[r]

                current_indices, current_num_neighbors, current_out_points = (
                    merge_outputs(
                        current_indices,
                        current_num_neighbors,
                        current_out_points,
                        local_indices + stride,
                        local_num_neighbors,
                        local_points,
                    )
                )

                stride += strides[r]
        ctx.save_for_backward(
            points, queries, current_indices, current_num_neighbors, current_out_points
        )

        return current_indices, current_out_points, None, current_num_neighbors

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

        # (
        #     points,
        #     queries,
        #     current_indices,
        #     current_num_neighbors,
        #     current_outputs,
        # ) = ctx.saved_tensors

        # # We need to do a ring again in the backward direction.
        # # The backward pass is computed locally, and then the gradients
        # # and p2 are moved along the ring together.
        # # for i in range(world_size):
        # # Calculate which source rank this data is from

        # local_p2_grad = _ball_query_backward_primitive_(
        #     points[0],
        #     queries[0],
        #     current_indices,
        #     current_num_neighbors,
        #     current_outputs,
        #     mapping_grad,
        #     num_neighbors_grad,
        #     outputs_grad,
        # )
        # local_points_grad = torch.zeros_like(points)

        # return (
        #     local_points_grad,
        #     local_p2_grad,
        #     None,
        #     None,
        #     None,
        #     None,
        # )


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


def radius_search_wrapper(
    func: Any, type: Any, args: tuple, kwargs: dict
) -> tuple[ShardTensor, ShardTensor, ShardTensor]:
    """
    Wrapper for radius_search to support sharded tensors.

    Handles 4 situations, based on the sharding of points and queries:
    - Points is sharded: a ring computation is performed.
        - queries is sharded: each rank contains a partial output,
          which is returned sharded like queries.
        - queries is replicated: each rank returns the full output,
          even though the input points is sharded.
    - Points is replicated: No ring is needed.
        - queries is sharded: each rank contains a partial output,
          which is returned sharded like queries.
        - queries is replicated: each rank returns the full output,
          even though the input points 2 is sharded.

    All input sharding has to be over a 1D mesh.  2D Point cloud sharding
    is not supported at this time.

    Regardless of the input sharding, the output will always be sharded like
    queries, and the output points will always have queried every input point
    like in the non-sharded case.

    Args:
        func: Original forward method
        type: Types of the inputs
        args: Positional arguments (points, queries)
        kwargs: Keyword arguments

    Returns:
        Tuple of (mapping, num_neighbors, outputs) as ShardTensor
    """

    points, queries, bq_kwargs = repackage_radius_search_wrapper_args(*args, **kwargs)

    if bq_kwargs["max_points"] is None:
        raise MissingShardPatch(
            "sharded radius_search_wrapper for radius_search does not currently support max_points=None"
        )

    if bq_kwargs["return_dists"] is True:
        raise MissingShardPatch(
            "sharded radius_search_wrapper for radius_search does not currently support return_dists=True"
        )

    # Make sure all meshes are the same
    if points._spec.mesh != queries._spec.mesh:
        raise MissingShardPatch(
            "point_cloud_ops.radius_search_wrapper: All point inputs must be on the same mesh"
        )

    # make sure all meshes are 1D
    if points._spec.mesh.ndim != 1:
        raise MissingShardPatch(
            "point_cloud_ops.radius_search_wrapper: All point inputs must be on 1D meshes"
        )

    # Do we need a ring?
    points_placement = points._spec.placements[0]

    if points_placement.is_shard():
        # We need a ring
        mapping, outputs, _, num_neighbors = ring_ball_query(points, queries, bq_kwargs)
    else:
        # No ring is needed
        mapping, num_neighbors, outputs = ringless_ball_query(
            points, queries, bq_kwargs
        )

    return mapping, outputs, None, num_neighbors


def repackage_radius_search_wrapper_args(
    points: ShardTensor,
    queries: ShardTensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
    *args,
    **kwargs,
) -> tuple[ShardTensor, ShardTensor, dict]:
    """Repackages ball query arguments into a standard format.

    Takes the arguments that could be passed to a ball query operation
    and separates them into core tensor inputs (points, queries, lengths1, lengths2)
    and configuration parameters packaged as a kwargs dict.

    Args:
        points: First set of points
        queries: Second set of points
        lengths1: Lengths of each batch in points
        lengths2: Lengths of each batch in queries
        *args: Additional positional args
        **kwargs: Additional keyword args

    Returns:
        Tuple containing:
        - points tensor
        - queries tensor
        - Dict of ball query configuration parameters
    """
    # Extract any additional parameters that might be in kwargs
    # or use defaults if not provided
    return_kwargs = {
        "radius": radius,
        "max_points": max_points,
        "return_dists": return_dists,
        "return_points": return_points,
    }

    # Add any explicitly passed parameters
    if kwargs:
        return_kwargs.update(kwargs)

    return points, queries, return_kwargs


ShardTensor.register_named_function_handler(
    "physicsnemo.radius_search_warp.default", radius_search_wrapper
)
