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

import torch
import torch.distributed as dist
from torch.overrides import handle_torch_function, has_torch_function
import time

from physicsnemo.distributed import DistributedManager, scatter_tensor, ShardTensor
from torch.distributed.tensor.placement_types import Shard, Replicate

from physicsnemo.distributed.shard_utils.ring import (
    perform_ring_iteration,
    RingPassingConfig,
)

# This time, let's make two moderately large tensors since we'll have to, at least briefly,
# construct a tensor of their point-by-point difference.
N_points_to_search = 234_568
N_target_points = 12_496
num_neighbors = 17

DistributedManager.initialize()
dm = DistributedManager()

device = dm.device


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We'll make these 3D tensors to represent 3D points
a = torch.randn(N_points_to_search, 3, device=device)
b = torch.randn(N_target_points, 3, device=device)


def knn(x, y, n):
    # This is to enable torch to track this knn function and route it correctly in ShardTensor:
    if has_torch_function((x, y)):
        return handle_torch_function(knn, (x, y), x, y, n)

    # Return the n nearest neighbors in x for each point in y.

    # First, compute the pairwise difference between all points in x and y.
    displacement_vec = x[None, :, :] - y[:, None, :]

    # Use the norm to compute the distance:
    distance = torch.norm(displacement_vec, dim=2)

    distances, indices = torch.topk(distance, k=n, dim=1, largest=False)

    x_results = x[indices]

    return x_results, distances


# Get the baseline result
y_neighbors_to_x, neighbor_disances = knn(a, b, num_neighbors)

if dm.rank == 0:
    print(y_neighbors_to_x.shape)  # should be (N_target_points, num_neighbors, 3)
    print(neighbor_disances.shape)  # should be (N_target_points, num_neighbors)

# DeviceMesh is a pytorch object - you can initialize it directly, or for added
# flexibility physicsnemo can infer up to one mesh dimension for you
# (as a -1, like in a tensor.reshape() call...)
mesh = dm.initialize_mesh(
    mesh_shape=[
        -1,
    ],
    mesh_dim_names=["domain"],
)
# Shard(i) indicates we want the final tensor to be sharded along the tensor dimension i
# But the placements is a tuple or list, indicating the desired placement along the mesh.
placements = (Shard(0),)
# This function will distribute the tensor from global_src to the specified mesh,
# using the input placements.
# Note that in multi-level parallelism, the source is the _global_ rank not the mesh group rank.
a_sharded = scatter_tensor(tensor=a, global_src=0, mesh=mesh, placements=placements)
b_sharded = scatter_tensor(tensor=b, global_src=0, mesh=mesh, placements=placements)


def knn_ring(func, types, args, kwargs):
    # Wrapper to intercept knn and compute it in a ring.
    # Never fully realizes the distance product.

    def extract_args(x, y, n, *args, **kwargs):
        return x, y, n

    x, y, n = extract_args(*args, **kwargs)

    # Each tensor has a _spec attribute, which contains information about the tensor's placement
    # and the devices it lives on:
    x_spec = x._spec
    y_spec = y._spec

    # ** In general ** you want to do some checking on the placements, since each
    # point cloud might be sharded differently.  By construction, I know they're both
    # sharded along the points axis here (and not, say, replicated).

    if not x_spec.mesh == y_spec.mesh:
        raise NotImplementedError("Tensors must be sharded on the same mesh")

    mesh = x_spec.mesh
    local_group = mesh.get_group(0)
    local_size = dist.get_world_size(group=local_group)
    mesh_rank = mesh.get_local_rank()

    # x and y are both sharded - and since we're returning the nearest
    # neighbors to x, let's make sure the output keeps that sharding too.

    # One memory-efficient way to do this is with with a ring computation.
    # We'll compute the knn on the local tensors, get the distances and outputs,
    # then shuffle the y shards along the mesh.

    # we'll need to sort the results and make sure we have just the top-k,
    # which is a little extra computation.

    # Physics nemo has a ring passing utility we can use.
    ring_config = RingPassingConfig(
        mesh_dim=0,
        mesh_size=local_size,
        ring_direction="forward",
        communication_method="p2p",
    )

    local_x, local_y = x.to_local(), y.to_local()
    current_dists = None
    current_topk_y = None

    x_sharding_shapes = x._spec.sharding_shapes()[0]

    for i in range(local_size):
        source_rank = (mesh_rank - i) % local_size

        # For point clouds, we need to pass the size of the incoming shard.
        next_source_rank = (source_rank - 1) % local_size
        recv_shape = x_sharding_shapes[next_source_rank]
        if i != local_size - 1:
            # Don't do a ring on the last iteration.
            next_local_x = perform_ring_iteration(
                local_x,
                mesh,
                ring_config,
                recv_shape=recv_shape,
            )

        # Compute the knn on the local tensors:
        local_x_results, local_distances = func(local_x, local_y, n)

        if current_dists is None:
            current_dists = local_distances
            current_topk_y = local_x_results
        else:
            # Combine with the topk so far:
            current_dists = torch.cat([current_dists, local_distances], dim=1)
            current_topk_y = torch.cat([current_topk_y, local_x_results], dim=1)
            # And take the topk again:
            current_dists, running_indexes = torch.topk(
                current_dists, k=n, dim=1, largest=False
            )

            # This creates proper indexing to select specific elements along dim 1
            current_topk_y = torch.gather(
                current_topk_y, 1, running_indexes.unsqueeze(-1).expand(-1, -1, 3)
            )

        if i != local_size - 1:
            # Don't do a ring on the last iteration.
            local_x = next_local_x

    # Finally, return the outputs as ShardTensors.
    topk_y = ShardTensor.from_local(
        current_topk_y,
        device_mesh=mesh,
        placements=y._spec.placements,
        sharding_shapes=y._spec.sharding_shapes(),
    )

    distances = ShardTensor.from_local(
        current_dists,
        device_mesh=mesh,
        placements=y._spec.placements,
        sharding_shapes=y._spec.sharding_shapes(),
    )

    return topk_y, distances


ShardTensor.register_function_handler(knn, knn_ring)

# Get the sharded result
y_neighbors_to_x_sharded, neighbor_disances_sharded = knn(
    a_sharded, b_sharded, num_neighbors
)

# Check for agreement:
y_neighbors_to_x_sharded = y_neighbors_to_x_sharded.full_tensor()
neighbor_disances_sharded = neighbor_disances_sharded.full_tensor()

if dm.rank == 0:
    print(
        f"Neighbors agreement? {torch.allclose(y_neighbors_to_x, y_neighbors_to_x_sharded)}"
    )
    print(
        f"Distances agreement? {torch.allclose(neighbor_disances, neighbor_disances_sharded)}"
    )


# run a couple times to warmup:
for i in range(5):
    _ = knn(a_sharded, b_sharded, num_neighbors)
# Optional: Benchmark it if you like:

# Measure execution time
torch.cuda.synchronize()
start_time = time.time()
for i in range(10):
    _ = knn(a_sharded, b_sharded, num_neighbors)
torch.cuda.synchronize()
end_time = time.time()
elapsed_time = end_time - start_time

if dm.rank == 0:
    print(f"Execution time for 10 runs: {elapsed_time:.4f} seconds")
