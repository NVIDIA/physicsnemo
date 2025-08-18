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
import time

from physicsnemo.distributed import DistributedManager, scatter_tensor, ShardTensor
from torch.distributed.tensor.placement_types import Shard, Replicate

from physicsnemo.distributed.shard_utils.ring import perform_ring_iteration, RingPassingConfig

# This time, let's make two moderately large tensors since we'll have to, at least briefly,
# construct a tensor of their point-by-point difference.
N_points_to_search = 234_568
N_target_points = 12_496
num_neighbors = 17

DistributedManager.initialize()
dm = DistributedManager()

# We'll make these 3D tensors to represent 3D points
a = torch.randn(N_points_to_search, 3, device=dm.device)
b = torch.randn(N_target_points, 3, device=dm.device)

def knn(x, y, n):
    # Return the n nearest neighbors in x for each point in y.
    
    # First, compute the pairwise difference between all points in x and y.
    displacement_vec = x[None, :, :] - y[:, None, :]
    
    # Use the norm to compute the distance:
    distance = torch.norm(displacement_vec, dim=2)
    
    distances, indices = torch.topk(distance, k=n, dim=1, largest=False)
    
    x_results = x[indices]
    
    return x_results, distances

# Get the baseline result
y_neighbors_to_x, neighbor_distances = knn(a, b, num_neighbors)

if dm.rank == 0:
    print(y_neighbors_to_x.shape)    # should be (N_target_points, num_neighbors, 3)
    print(neighbor_distances.shape)  # should be (N_target_points, num_neighbors)

# DeviceMesh is a pytorch object - you can initialize it directly, or for added
# flexibility physicsnemo can infer up to one mesh dimension for you 
# (as a -1, like in a tensor.reshape() call...)
mesh = dm.initialize_mesh(mesh_shape = [-1,], mesh_dim_names = ["domain"])
# Shard(i) indicates we want the final tensor to be sharded along the tensor dimension i
# But the placements is a tuple or list, indicating the desired placement along the mesh.
placements = (Shard(0),)
# This function will distribute the tensor from global_src to the specified mesh,
# using the input placements.
# Note that in multi-level parallelism, the source is the _global_ rank not the mesh group rank.
a_sharded = scatter_tensor(tensor = a, global_src = 0, mesh = mesh, placements = placements)
b_sharded = scatter_tensor(tensor = b, global_src = 0, mesh = mesh, placements = placements)

# Get the sharded result
y_neighbors_to_x_sharded, neighbor_distances_sharded = knn(a_sharded, b_sharded, num_neighbors)

# Check for agreement:
y_neighbors_to_x_sharded = y_neighbors_to_x_sharded.full_tensor()
neighbor_distances_sharded = neighbor_distances_sharded.full_tensor()

if dm.rank == 0:
    # Note - do the ``full_tensor`` call outside this if-block or it will hang!
    print(f"Neighbors agreement? {torch.allclose(y_neighbors_to_x, y_neighbors_to_x_sharded)}")
    print(f"Distances agreement? {torch.allclose(neighbor_distances, neighbor_distances_sharded)}")

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

