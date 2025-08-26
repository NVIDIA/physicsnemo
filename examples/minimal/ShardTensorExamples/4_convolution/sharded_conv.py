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

from torch.distributed.tensor import (
    Shard,
    distribute_module,
)


from physicsnemo.distributed import (
    DistributedManager,
    ShardTensor,
    scatter_tensor,
)

DistributedManager.initialize()
dm = DistributedManager()

###########################
# Single GPU - Create input
###########################
original_tensor = torch.randn(1, 8, 1024, 1024, device=dm.device, requires_grad=True)

###########################################
# Single GPU - Create a single-layer model:
###########################################
conv = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1).to(dm.device)

########################################
# Single GPU - forward + loss + backward
########################################
single_gpu_output = conv(original_tensor)

# This isn't really a loss, just a pretend one that's scalar!
single_gpu_output.mean().backward()
# Copy the gradients produced here - so we don't overwrite them later.
original_tensor_grad = original_tensor.grad.data.clone()

####################
# Single GPU - DONE!
####################


#################
# Sharded - Setup
#################


# DeviceMesh is a pytorch object - you can initialize it directly, or for added
# flexibility physicsnemo can infer up to one mesh dimension for you
# (as a -1, like in a tensor.reshape() call...)
mesh = dm.initialize_mesh(mesh_shape=(-1,), mesh_dim_names=("domain_parallel",))

# A mesh, by the way, refers to devices and not data: it's a mesh of connected
# GPUs in this case, and the python DeviceMesh can be reused as many times as needed.
# That said, it can be decomposed similar to a tensor - multiple mesh axes, and
# you can axis sub-meshes.  Each mesh also has ways to access process groups
# for targeted collectives.


###########################
# Sharded - Distribute Data
###########################

# This is now a tensor across all GPUs, spread on the "height" dimension == 2
# In general, to create a ShardTensor (or DTensor) you need to specify placements.
# Placements must be a list or tuple of `Shard()` or `Replicate()` objects
# from torch.distributed.tensor.
#
# Each index in the tuple represents the placement over the corresponding mesh dimension
# (so, mesh.ndim == len(placements)! )
# `Shard()` takes an argument representing the **tensor** index that is sharded.
# So below, the tensor is sharded over the tensor dimension 2 on the mesh dimension 0.
sharded_tensor = scatter_tensor(
    original_tensor, 0, mesh, (Shard(2),), requires_grad=True
)


################################
# Sharded - distribute the model
################################

# We tell pytorch that the convolution will work on distributed tensors:
# And, over the same mesh!
distributed_conv = distribute_module(conv, mesh)


#####################################
# Sharded - forward + loss + backward
#####################################

# Now, we can do the distributed convolution:
sharded_output = distributed_conv(sharded_tensor)
sharded_output.mean().backward()


############################################
# Sharded - gather up outputs to all devices
############################################

# This triggers a collective allgather.
full_output = sharded_output.full_tensor()
full_grad = sharded_tensor.grad.full_tensor()


#################
# Accuracy Checks
#################

if dm.rank == 0:
    # Only check on rank 0 because we used it's data and weights for the sharded tensor.
    # Check that the output is the same as the single-device output:
    assert torch.allclose(full_output, single_gpu_output, atol=1e-3, rtol=1e-3)
    print(f"Global operation matches local! ")

    # Check that the gradient is correct:
    assert torch.allclose(original_tensor_grad, full_grad)
    print(f"Gradient check passed!")


print(
    f"Distributed grad sharding and local shape: {sharded_tensor.grad._spec.placements}, {sharded_tensor.grad.to_local().shape}"
)
