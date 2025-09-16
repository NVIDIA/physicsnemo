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
Interpolation is the operation used in upsampling, which is why we're
testing this with upsampling.

Interpolation generally needs a spatial halo + local interpolation,
and the sharding is always over spatial dims.  The channel
count never really matters except > 0.

Here, we're testing 1d, 2d, 3d interpolation with Upsample on 1d
and 2d meshes.
"""

import pytest
from torch.distributed.tensor.placement_types import Shard
from torch.nn import Upsample

from physicsnemo.distributed import DistributedManager, scatter_tensor

from .utils import generate_image_like_data, numerical_shard_tensor_check


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "W",
    [
        32,
    ],
)
@pytest.mark.parametrize(
    "scale_factor",
    [
        2,
        3,
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "nearest",
    ],
)
@pytest.mark.parametrize("backward", [False, True])
def test_upsample_2d_1dmesh(distributed_mesh, H, W, scale_factor, mode, backward):
    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, W), device=dm.device)

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = Upsample(scale_factor=scale_factor, mode=mode)

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "W",
    [
        32,
    ],
)
@pytest.mark.parametrize(
    "scale_factor",
    [
        2,
        3,
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "nearest",
    ],
)
@pytest.mark.parametrize("backward", [False, True])
def test_upsample_2d_2dmesh(distributed_mesh_2d, H, W, scale_factor, mode, backward):
    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, W), device=dm.device)

    placements = (Shard(2), Shard(3))

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh_2d, placements, requires_grad=backward
    )

    module = Upsample(scale_factor=scale_factor, mode=mode)

    numerical_shard_tensor_check(
        distributed_mesh_2d, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "W",
    [
        32,
    ],
)
@pytest.mark.parametrize(
    "D",
    [
        32,
    ],
)
@pytest.mark.parametrize(
    "scale_factor",
    [
        2,
        3,
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "nearest",
    ],
)
@pytest.mark.parametrize("backward", [False, True])
def test_upsample_3d_1dmesh(distributed_mesh, H, W, D, scale_factor, mode, backward):
    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, W, D), device=dm.device)

    placements = (Shard(4),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = Upsample(scale_factor=scale_factor, mode=mode)

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "W",
    [
        32,
    ],
)
@pytest.mark.parametrize(
    "D",
    [
        32,
    ],
)
@pytest.mark.parametrize(
    "scale_factor",
    [
        2,
        3,
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "nearest",
    ],
)
@pytest.mark.parametrize("backward", [False, True])
def test_upsample_3d_2dmesh(distributed_mesh_2d, H, W, D, scale_factor, mode, backward):
    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, W, D), device=dm.device)

    placements = (Shard(3), Shard(4))

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh_2d, placements, requires_grad=backward
    )

    module = Upsample(scale_factor=scale_factor, mode=mode)

    numerical_shard_tensor_check(
        distributed_mesh_2d, module, [sharded_image], {}, check_grads=backward
    )
