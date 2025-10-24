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

from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist

from physicsnemo.utils.neighbors.knn._cuml_impl import knn_impl
from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")

from physicsnemo.distributed import ShardTensor  # noqa: E402
from physicsnemo.distributed.shard_utils.patch_core import (  # noqa: E402
    MissingShardPatch,
)
from physicsnemo.distributed.shard_utils.ring import (  # noqa: E402
    RingPassingConfig,
    perform_ring_iteration,
)


def ring_knn(
    points: ShardTensor, queries: ShardTensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ring based kNN implementation where points travel around a ring and queries stay local.

    This function performs k-nearest neighbor search using a distributed ring-based
    algorithm. The points are passed around different devices in a ring topology while
    the queries remain local to each device.

    Parameters
    ----------
    points : ShardTensor
        The point cloud data tensor that will be distributed around the ring.
        Must be sharded on the same mesh as queries.
    queries : ShardTensor
        The query points tensor that stays local on each device.
        Must be sharded on the same mesh as points.
    k : int
        Number of nearest neighbors to find for each query point.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - shard_idx : torch.Tensor
            Indices of the k nearest neighbors for each query point
        - shard_distances : torch.Tensor
            Distances to the k nearest neighbors for each query point

    Raises
    ------
    NotImplementedError
        If points and queries tensors are not sharded on the same mesh.
    """
    # Each tensor has a _spec attribute, which contains information about the tensor's placement
    # and the devices it lives on:
    points_spec = points._spec
    queries_spec = queries._spec

    # ** In general ** you want to do some checking on the placements, since each
    # point cloud might be sharded differently.  By construction, I know they're both
    # sharded along the points axis here (and not, say, replicated).

    if not points_spec.mesh == queries_spec.mesh:
        raise NotImplementedError("Tensors must be sharded on the same mesh")

    mesh = points_spec.mesh
    local_group = mesh.get_group(0)
    local_size = dist.get_world_size(group=local_group)
    mesh_rank = mesh.get_local_rank()

    # points and queries are both sharded - and since we're returning the nearest
    # neighbors to points, let's make sure the output keeps that sharding too.

    # One memory-efficient way to do this is with with a ring computation.
    # We'll compute the knn on the local tensors, get the distances and outputs,
    # then shuffle the queries shards along the mesh.

    # we'll need to sort the results and make sure we have just the top-k,
    # which is a little extra computation.

    # Physics nemo has a ring passing utility we can use.
    ring_config = RingPassingConfig(
        mesh_dim=0,
        mesh_size=local_size,
        ring_direction="forward",
        communication_method="p2p",
    )

    local_points, local_queries = points.to_local(), queries.to_local()
    current_dists = None
    current_topk_idx = None

    points_spec = points._spec

    points_sharding_shapes = points_spec.sharding_shapes()[0]

    sharding_dim = points_spec.placements[0].dim

    # This is to help specify the offset from local to global tensor.
    points_strides_along_ring = [s[sharding_dim] for s in points_sharding_shapes]
    points_strides_along_ring = np.cumsum(points_strides_along_ring)
    points_strides_along_ring = [
        0,
    ] + list(points_strides_along_ring[0:-1])

    for i in range(local_size):
        source_rank = (mesh_rank - i) % local_size

        # For point clouds, we need to pass the size of the incoming shard.
        next_source_rank = (source_rank - 1) % local_size
        recv_shape = points_sharding_shapes[next_source_rank]
        if i != local_size - 1:
            # Don't do a ring on the last iteration.
            next_local_points = perform_ring_iteration(
                local_points,
                mesh,
                ring_config,
                recv_shape=recv_shape,
            )

        # Compute the knn on the local tensors:
        local_idx, local_distances = knn_impl(local_points, local_queries, k)

        # The local_idx indexes into the _local_ tensor, but for
        # Correctness we need it to index into the _global_ tensor.
        # Make sure to index using the rank the points came from!
        offset = points_strides_along_ring[source_rank]
        local_idx = local_idx + offset

        if current_dists is None:
            current_dists = local_distances
            current_topk_idx = local_idx
        else:
            # Combine with the topk so far:
            current_dists = torch.cat([current_dists, local_distances], dim=1)
            current_topk_idx = torch.cat([current_topk_idx, local_idx], dim=1)
            # And take the topk again:
            current_dists, running_indexes = torch.topk(
                current_dists, k=k, dim=1, sorted=True, largest=False
            )

            # This creates proper indexing to select specific elements along dim 1

            current_topk_idx = torch.gather(current_topk_idx, 1, running_indexes)

        if i != local_size - 1:
            # Don't do a ring on the last iteration.
            local_points = next_local_points

    return current_topk_idx, current_dists


def extract_knn_args(
    points: torch.Tensor, queries: torch.Tensor, k: int, *args, **kwargs
):
    """
    Minimal function to use python's argument unpacking to extract the points, queries, and k values.
    """
    return points, queries, k


def knn_sharded_wrapper(
    func: Callable, types: Any, args: tuple, kwargs: dict
) -> tuple[ShardTensor, ShardTensor]:
    """
    Dispatch the proper kNN tools based on the input sharding.

    `args` and `kwargs` are passed to `extract_knn_args` to extract
    the points, queries, and k values needed for the kNN operation.

    Parameters
    ----------
    func : Callable
        The function to dispatch.
    types : Any
        The types of the inputs.
    args : tuple
        The positional arguments.
    kwargs : dict
        The keyword arguments.

    Returns
    -------
    tuple[ShardTensor, ShardTensor]
        A tuple containing the shard_idx and shard_distances.

    Raises
    ------
    MissingShardPatch
        If the points and queries tensors are not sharded on the same mesh.
    """

    points, queries, k = extract_knn_args(*args, **kwargs)

    # kNN will only work with 1D sharding
    if points._spec.mesh != queries._spec.mesh:
        raise MissingShardPatch(
            "sharded knn: All point inputs must be on the same mesh"
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
        idx, distances = ring_knn(points, queries, k)
    else:
        # No ring is needed.  Get the local tensors and compute directly:
        local_points = points.to_local()  # This is replicated, getting all of it
        local_queries = queries.to_local()  # This sharding doesn't matter!
        idx, distances = knn_impl(local_points, local_queries, k)

    # The outputs only depend on the local queries shape
    input_queries_spec = queries._spec
    # The global output tensor will be (N_q, k)

    output_queries_shard_shapes = {}
    for mesh_dim in input_queries_spec.sharding_shapes().keys():
        shard_shapes = tuple(
            torch.Size((s[0], k))
            for s in input_queries_spec.sharding_shapes()[mesh_dim]
        )
        output_queries_shard_shapes[mesh_dim] = shard_shapes

    # Convert the selected points and indexes to shards:
    shard_idx = ShardTensor.from_local(
        idx,
        queries._spec.mesh,
        queries._spec.placements,
        sharding_shapes=output_queries_shard_shapes,
    )
    shard_distances = ShardTensor.from_local(
        distances,
        queries._spec.mesh,
        queries._spec.placements,
        sharding_shapes=output_queries_shard_shapes,
    )

    return shard_idx, shard_distances


ShardTensor.register_named_function_handler(
    "physicsnemo.knn_cuml.default", knn_sharded_wrapper
)
