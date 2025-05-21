import torch
import time

# This time, let's make two moderately large tensors since we'll have to, at least briefly,
# construct a tensor of their point-by-point difference.
N_points_to_search = 234_567
N_target_points = 12_345
num_neighbors = 17


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We'll make these 3D tensors to represent 3D points
a = torch.randn(N_points_to_search, 3, device=device)
b = torch.randn(N_target_points, 3, device=device)

def knn(x, y, n):
    # Return the n nearest neighbors in x for each point in y.
    # Returns the 
    
    # First, compute the pairwise difference between all points in x and y.
    displacement_vec = x[None, :, :] - y[:, None, :]
    
    # Use the norm to compute the distance:
    distance = torch.norm(displacement_vec, dim=2)
    
    distances, indices = torch.topk(distance, k=n, dim=1, largest=False)
    
    x_results = x[indices]
    # distance = distances[indices]
    
    return x_results, distances

y_neighbors_to_x, neighbor_disances = knn(a,b, num_neighbors)
print(y_neighbors_to_x.shape) # should be (N_target_points, num_neighbors, 3)
print(neighbor_disances.shape) # should be (N_target_points, num_neighbors)

# run a couple times to warmup:
for i in range(5):
    _ = knn(a,b, num_neighbors)

# Optional: Benchmark it if you like:

# Measure execution time
torch.cuda.synchronize()
start_time = time.time()
for i in range(10):
    _ = knn(a,b, num_neighbors)
torch.cuda.synchronize()
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Execution time for 10 runs: {elapsed_time:.4f} seconds")