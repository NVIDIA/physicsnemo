# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""
Pooling operations are testing on average and max pooling for 1, 2 and 3
dimensions as well as 1d and 2d meshes.  Testing over image like data,
and the channels dimension is largely irrelevant.

Sharding is only over spatial dimensions (Shard(2),) (or 3, or 4)
"""

import pytest
import torch
from torch.distributed.tensor import distribute_module
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, scatter_tensor

from .utils import generate_image_like_data, numerical_shard_tensor_check


@pytest.mark.multigpu_static
@pytest.mark.parametrize("padding", [2, [1, 2, 3, 4]])
@pytest.mark.parametrize("backward", [False, True])
def test_constant_pad_2d_1dmesh(distributed_mesh, padding, backward):
    H = 128
    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H)).to(dm.device)

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.ConstantPad2d(padding=padding, value=1.234)

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("padding", [2, [1, 2, 3, 4]])
@pytest.mark.parametrize("backward", [False, True])
def test_constant_pad_2d_2dmesh(distributed_mesh_2d, padding, backward):
    H = 128
    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H)).to(dm.device)

    placements = (Shard(2), Shard(3))

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh_2d, placements, requires_grad=backward
    )

    module = torch.nn.ConstantPad2d(padding=padding, value=1.234)

    numerical_shard_tensor_check(
        distributed_mesh_2d, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("padding", [2, [1, 2, 3, 4]])
@pytest.mark.parametrize("backward", [False, True])
def test_reflection_pad_2d_1dmesh(distributed_mesh, padding, backward):
    H = 128
    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H)).to(dm.device)

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.ReflectionPad2d(padding=padding)

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("padding", [2, [1, 2, 3, 4]])
@pytest.mark.parametrize("backward", [False, True])
def test_replication_pad_2d_1dmesh(distributed_mesh, padding, backward):
    H = 128
    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H)).to(dm.device)

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.ReplicationPad2d(padding=padding)

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


# This tests convolution with padding_mode != zeros.  This is actually the main
# motivation for supporting torch.nn.functional.pad


@pytest.mark.multigpu_static
@pytest.mark.parametrize("padding_mode", ["reflect", "replicate"])
@pytest.mark.parametrize("backward", [False, True])
def test_padded_convolution_2d_1dmesh(distributed_mesh, padding_mode, backward):
    H = 256
    C_in = 8

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H)).to(dm.device)

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.Conv2d(
        in_channels=C_in,
        out_channels=C_in,
        kernel_size=3,
        padding=2,
        padding_mode=padding_mode,
    )
    module = module.to(dm.device)

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


class test_uneven_pad_functionals(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, 0, 2, 0), mode="replicate")
        x = torch.nn.functional.pad(x, (0, 0, 0, 2), mode="constant", value=1.0)
        return x


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_uneven_pad_2d_1dmesh(distributed_mesh, backward):
    H = 256
    C_in = 8

    dm = DistributedManager()
    n_gpus = dm.world_size

    image = generate_image_like_data(2, C_in, (H, H)).to(dm.device)

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )
    # Local reference tensors that should be the same shape as the shards
    if dm.rank in [0, n_gpus - 1]:
        local_ref = torch.ones(2, C_in, H // n_gpus + 2, H).to(dm.device)
    else:
        local_ref = torch.ones(2, C_in, H // n_gpus, H).to(dm.device)

    local_ref_sharded = ShardTensor.from_local(
        local_ref, distributed_mesh, (Shard(dim=2),), sharding_shapes="infer"
    )

    dist_test_pad = distribute_module(
        test_uneven_pad_functionals().to(dm.device), device_mesh=distributed_mesh
    )

    sharded_image_pad = dist_test_pad(sharded_image)

    _ = local_ref_sharded + sharded_image_pad

    numerical_shard_tensor_check(
        distributed_mesh,
        test_uneven_pad_functionals(),
        [sharded_image],
        {},
        check_grads=backward,
    )

    full_image_pad = torch.nn.functional.pad(image, (0, 0, 2, 0), mode="replicate")
    full_image_pad = torch.nn.functional.pad(
        full_image_pad, (0, 0, 0, 2), mode="constant", value=1.0
    )

    assert torch.allclose(sharded_image_pad.full_tensor(), full_image_pad)
