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
Here, we're testing normalization layers.  Tests are currently only implemented
for 1D data.  LayerNorm is actually supported upstream in pytorch by DTensor
itself, this test just cross checks it works on ShardTensor too.

GroupNorm is supported in physicsnemo.  Notably, GroupNorm can suffer from
some numerical instabilities for smaller datasets. So the tolerance is turned
up slightly.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.domain_parallel.shard_utils.patch_core import MissingShardPatch

from .utils import generate_image_like_data, numerical_shard_tensor_check


@pytest.mark.multigpu_static
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("backward", [False])
@pytest.mark.parametrize("amp", [False, True])
def test_layer_norm_1d(distributed_mesh, affine, backward, amp):
    if affine:
        pytest.xfail("LayerNorm with affine=True is currently failing tests")

    H = 128
    C_in = 16

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H,), device=dm.device)

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.LayerNorm(normalized_shape=H, elementwise_affine=affine)

    atol, rtol = (1e-2, 1e-2) if amp else (1e-5, 1e-5)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_image],
        {},
        check_grads=backward,
        atol=atol,
        rtol=rtol,
        amp=amp,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize(
    "num_groups",
    [
        1,
        4,
    ],
)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("backward", [True, False])
@pytest.mark.parametrize("amp", [True, False])
def test_group_norm_1d(distributed_mesh, num_groups, affine, backward, amp):
    H = 257
    C_in = 256

    dm = DistributedManager()

    image = generate_image_like_data(
        2,
        C_in,
        (
            H,
            H,
        ),
        device=dm.device,
    )

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.GroupNorm(num_groups=num_groups, num_channels=C_in, affine=affine)

    atol, rtol = (1e-3, 1e-3) if amp else (1e-5, 1e-5)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_image],
        {},
        check_grads=backward,
        atol=atol,
        rtol=rtol,
        amp=amp,
    )


@pytest.mark.multigpu_static
def test_group_norm_batch_sharded_raises(distributed_mesh):
    """ShardTensor shards spatial dimensions, so a sharded batch is rejected."""
    dm = DistributedManager()

    image = generate_image_like_data(2 * dm.world_size, 8, (16, 16), device=dm.device)
    sharded_image = scatter_tensor(image, 0, distributed_mesh, (Shard(0),))

    module = torch.nn.GroupNorm(num_groups=4, num_channels=8).to(dm.device)

    with pytest.raises(MissingShardPatch, match="batch dimension"):
        module(sharded_image)
