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

def sharded_dot_product(func: callable, types: tuple, args: tuple, kwargs: dict):
    """
    Overload for torch.dot to support sharded tensors.
    
    This function enables mutli-gpu dot product operations on ShardTensors,
    by computing the dot product locally on each rank and then summin across 
    all GPUs.  Requires the placements and mesh to agree across the two tensors.
    
    This is tutorial code: it does not handle all cases and you should 
    not use it in production.
    
    Note the function signature: we are using this function in the 
    __torch_function__ protocol and it has to follow the specific signature
    requirements.
    
    Args:
        func (callable): The function to overload (e.g., torch.dot).
        types (Tuple): Tuple of types passed by __torch_function__ protocol.
        args (Tuple): Positional arguments passed to the function.
        kwargs (Dict): Keyword arguments passed to the function.
        
    In general, torch will use the values in `types` to determine which
    path of execution to take.  In this function, we don't have to worry
    about that as much because it's already selected for execution.
    """
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


    # The output placements are now Replicated, not sharded.  We have used all_reduce
    # to sum the local results across all ranks, and each rank has the full data - 
    # exactly what the Replicate() placement expects.
    # (Even though it's a scalar output, we still have to specify a placement)
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