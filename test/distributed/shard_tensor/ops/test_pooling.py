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

"""
Pooling operations are testing on average and max pooling for 1, 2 and 3
dimensions as well as 1d and 2d meshes.  Testing over image like data,
and the channels dimension is largely irrelevant.

Sharding is only over spatial dimensions (Shard(2),) (or 3, or 4)
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager, scatter_tensor

from .utils import generate_image_like_data, numerical_shard_tensor_check


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "K",
    [
        2,
        4,
    ],
)
@pytest.mark.parametrize("stride", [2, 4])
@pytest.mark.parametrize("backward", [False, True])
def test_avg_pool_1d_1dmesh(distributed_mesh, H, K, stride, backward):
    if K != stride:
        pytest.xfail("Pooling requires stride == K")

    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H,)).to(dm.device)

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.AvgPool1d(kernel_size=K, stride=stride)

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "K",
    [
        2,
        4,
    ],
)
@pytest.mark.parametrize("stride", [2, 4])
@pytest.mark.parametrize("backward", [False, True])
def test_avg_pool_2d_1dmesh(distributed_mesh, H, K, stride, backward):
    if K != stride:
        pytest.xfail("Pooling requires stride == K")

    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H)).to(dm.device)

    placements = (Shard(3),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.AvgPool2d(kernel_size=[K, K], stride=[stride, stride])

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "K",
    [
        2,
        4,
    ],
)
@pytest.mark.parametrize("stride", [2, 4])
@pytest.mark.parametrize("backward", [False, True])
def test_avg_pool_2d_2dmesh(distributed_mesh_2d, H, K, stride, backward):
    if K != stride:
        pytest.xfail("Pooling requires stride == K")

    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H)).to(dm.device)

    placements = (
        Shard(2),
        Shard(3),
    )

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh_2d, placements, requires_grad=backward
    )

    module = torch.nn.AvgPool2d(kernel_size=[K, K], stride=[stride, stride])

    numerical_shard_tensor_check(
        distributed_mesh_2d, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 128])
@pytest.mark.parametrize(
    "K",
    [
        2,
        4,
    ],
)
@pytest.mark.parametrize("stride", [2, 4])
@pytest.mark.parametrize("backward", [False, True])
def test_avg_pool_3d_1dmesh(distributed_mesh, H, K, stride, backward):
    if K != stride:
        pytest.xfail("Pooling requires stride == K")

    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H, H)).to(dm.device)

    placements = (Shard(4),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.AvgPool3d(kernel_size=[K, K, K], stride=[stride, stride, stride])

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 128])
@pytest.mark.parametrize(
    "K",
    [
        2,
        4,
    ],
)
@pytest.mark.parametrize("stride", [2, 4])
@pytest.mark.parametrize("backward", [False, True])
def test_avg_pool_3d_2dmesh(distributed_mesh_2d, H, K, stride, backward):
    if K != stride:
        pytest.xfail("Pooling requires stride == K")

    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H, H)).to(dm.device)

    placements = (
        Shard(3),
        Shard(4),
    )

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh_2d, placements, requires_grad=backward
    )

    module = torch.nn.AvgPool3d(kernel_size=[K, K, K], stride=[stride, stride, stride])

    numerical_shard_tensor_check(
        distributed_mesh_2d, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "K",
    [
        2,
        4,
    ],
)
@pytest.mark.parametrize("stride", [2, 4])
@pytest.mark.parametrize("backward", [False, True])
def test_max_pool_1d_1dmesh(distributed_mesh, H, K, stride, backward):
    if K != stride:
        pytest.xfail("Pooling requires stride == K")

    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H,)).to(dm.device)

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.MaxPool1d(kernel_size=K, stride=stride)

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "K",
    [
        2,
        4,
    ],
)
@pytest.mark.parametrize("stride", [2, 4])
@pytest.mark.parametrize("backward", [False, True])
def test_max_pool_2d_1dmesh(distributed_mesh, H, K, stride, backward):
    if K != stride:
        pytest.xfail("Pooling requires stride == K")

    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H), device=dm.device)

    placements = (Shard(3),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.MaxPool2d(kernel_size=[K, K], stride=[stride, stride])

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "K",
    [
        2,
        4,
    ],
)
@pytest.mark.parametrize("stride", [2, 4])
@pytest.mark.parametrize("backward", [False, True])
def test_max_pool_2d_2dmesh(distributed_mesh_2d, H, K, stride, backward):
    if K != stride:
        pytest.xfail("Pooling requires stride == K")

    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H), device=dm.device)

    placements = (
        Shard(2),
        Shard(3),
    )

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh_2d, placements, requires_grad=backward
    )

    module = torch.nn.MaxPool2d(kernel_size=[K, K], stride=[stride, stride])

    numerical_shard_tensor_check(
        distributed_mesh_2d, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 128])
@pytest.mark.parametrize(
    "K",
    [
        2,
        4,
    ],
)
@pytest.mark.parametrize("stride", [2, 4])
@pytest.mark.parametrize("backward", [False, True])
def test_max_pool_3d_1dmesh(distributed_mesh, H, K, stride, backward):
    if K != stride:
        pytest.xfail("Pooling requires stride == K")

    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H, H), device=dm.device)

    placements = (Shard(4),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.MaxPool3d(kernel_size=[K, K, K], stride=[stride, stride, stride])

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 128])
@pytest.mark.parametrize(
    "K",
    [
        2,
        4,
    ],
)
@pytest.mark.parametrize("stride", [2, 4])
@pytest.mark.parametrize("backward", [False, True])
def test_max_pool_3d_2dmesh(distributed_mesh_2d, H, K, stride, backward):
    if K != stride:
        pytest.xfail("Pooling requires stride == K")

    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H, H), device=dm.device)

    placements = (
        Shard(3),
        Shard(4),
    )

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh_2d, placements, requires_grad=backward
    )

    module = torch.nn.MaxPool3d(kernel_size=[K, K, K], stride=[stride, stride, stride])

    numerical_shard_tensor_check(
        distributed_mesh_2d, module, [sharded_image], {}, check_grads=backward
    )
