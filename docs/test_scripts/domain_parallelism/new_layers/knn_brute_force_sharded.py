import torch
import torch.distributed as dist
import time

from physicsnemo.distributed import DistributedManager, scatter_tensor, ShardTensor
from torch.distributed.tensor.placement_types import Shard, Replicate

from physicsnemo.distributed.shard_utils.ring import perform_ring_iteration, RingPassingConfig

# This time, let's make two moderately large tensors since we'll have to, at least briefly,
# construct a tensor of their point-by-point difference.
N_points_to_search = 234_567
N_target_points = 12_345
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

mesh = dm.initialize_mesh([-1,], ["domain"])
placements = (Shard(0),)
a_sharded = scatter_tensor(a, 0, mesh, placements)
b_sharded = scatter_tensor(b, 0, mesh, placements)

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

