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
import time

from physicsnemo.distributed import DistributedManager, scatter_tensor
from torch.distributed.tensor.placement_types import Shard

# Another really big tensor:
N = 1_000_000_000

DistributedManager.initialize()
dm = DistributedManager()

device = dm.device

a = torch.randn(N, device=device)
b = torch.randn(N, device=device)

def f(x, y):
    return x + y

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
