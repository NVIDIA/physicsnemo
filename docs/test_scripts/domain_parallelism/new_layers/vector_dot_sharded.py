import torch
import torch.distributed as dist
import time

from physicsnemo.distributed import DistributedManager, scatter_tensor, ShardTensor
from torch.distributed.tensor.placement_types import Shard, Replicate

def sharded_dot_product(func, types, args, kwargs):
    # NOTE: all functions overloaded and used by __torch_function__ will have 
    # the same input signature.  You can use python argument unpacking to 
    # extract what you need:
    def extract_args(x, y, *args, **kwargs):
        return x, y
    x, y = extract_args(*args, **kwargs)
    
    # Each tensor has a _spec attribute, which contains information about the tensor's placement
    # and the devices it lives on:
    x_spec = x._spec
    y_spec = y._spec
    
    # IT'S usually good to ensure the tensor placements work:
    if not x_spec.placements == y_spec.placements:
        raise NotImplementedError("Tensors must be sharded on the same device")
    
    if not x_spec.mesh == y_spec.mesh:
        raise NotImplementedError("Tensors must be sharded on the same mesh")
    
    # And, you might want to check placements are valid in more complex cases
    
    # Extract the mesh - we'll want it for the all reduce:
    mesh = x_spec.mesh
    
    # This is a straightforward implementation, for clarity
    # Get the local values of each tensor:
    local_x = x.to_local()
    local_y = y.to_local()
    
    # This is a purely single-gpu operation:
    local_dot_product = torch.dot(local_x, local_y)
    # If you wanted to write a generic sharding handler for this type of operation, 
    # you could do:
    # local_dot_product = func(local_x, local_y)
    # But it's over kill here...
    
    # SUM_Reduce the local result across all ranks:
    dist.all_reduce(local_dot_product, op=dist.ReduceOp.SUM, group=mesh.get_group())

    # We do want to return the result as a ShardTensor, for consistency.
    # We can easily create one on the same mesh as a "Replicated" tensor:

    output = ShardTensor.from_local(
        local_tensor = local_dot_product, 
        device_mesh =  mesh, 
        placements = (Replicate(),)
    )

    return output

# Register the implementation with ShardTensor's function dispatch:
ShardTensor.register_function_handler(torch.dot, sharded_dot_product)


# Another really big tensor:
N = 1_000_000_000

DistributedManager.initialize()
dm = DistributedManager()

device = dm.device

a = torch.randn(N, device=device)
b = torch.randn(N, device=device)

def f(x, y):
    return torch.dot(x , y)

# Get the baseline result
c_baseline = f(a,b)

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


c_sharded = f(a_sharded,b_sharded)

# Comparison requires that we coalesce the results:
c_sharded = c_sharded.full_tensor()


# Now, performance measurement:

# Warm up:
for i in range(5):
    c = f(a_sharded,b_sharded)

# Measure execution time
torch.cuda.synchronize()
start_time = time.time()
for i in range(10):
    c = f(a_sharded,b_sharded)
torch.cuda.synchronize()
end_time = time.time()
elapsed_time = end_time - start_time

if dm.rank == 0:
    print(f"Rank {dm.rank}, Tensor agreement? {torch.allclose(c_baseline, c_sharded)}")
    print(f"Execution time for 10 runs: {elapsed_time:.4f} seconds")