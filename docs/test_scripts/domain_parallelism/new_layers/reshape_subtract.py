import torch
import torch.distributed as dist
import time

from physicsnemo.distributed import DistributedManager, scatter_tensor, ShardTensor
from torch.distributed.tensor.placement_types import Shard, Replicate

from physicsnemo.distributed.shard_utils.ring import perform_ring_iteration, RingPassingConfig

# This time, let's make two moderately large tensors since we'll have to, at least briefly,
# construct a tensor of their point-by-point difference.
N1 = 234_567
N2 = 12_345
num_neighbors = 17

DistributedManager.initialize()
dm = DistributedManager()

# We'll make these 3D tensors to represent 3D points
a = torch.randn(N1, 3, device=dm.device)
b = torch.randn(N2, 3, device=dm.device)

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

if dm.rank == 0:
    print(f"a_sharded shape and placement: {a_sharded.shape}, {a_sharded.placements}")
    print(f"b_sharded shape and placement: {b_sharded.shape}, {b_sharded.placements}")
a_sharded = a_sharded[None, :, :]
b_sharded = b_sharded[:, None, :]

if dm.rank == 0:
    print(f"a_sharded shape and placement: {a_sharded.shape}, {a_sharded.placements}")
    print(f"b_sharded shape and placement: {b_sharded.shape}, {b_sharded.placements}")

distance_vec = a_sharded - b_sharded
if dm.rank == 0:
    print(f"distance_vec shape and placement: {distance_vec.shape}, {distance_vec.placements}")

