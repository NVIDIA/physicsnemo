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

"""Domain-parallel sanity tests for DiffusionUNet3D.

These tests verify that the 3D U-Net backbone integrates with
``ShardTensor`` and ``distribute_module``: spatial inputs sharded along a
spatial axis flow through the encoder/decoder, produce an output of the
expected dense shape, and stay sharded on the same mesh. They mirror the
patterns used in ``test/domain_parallel/models/test_transolver.py``.
"""

import pytest
import torch
from tensordict import TensorDict
from torch.distributed.tensor import distribute_module
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.experimental.models.diffusion_unets import DiffusionUNet3D


def _build_model(x_channels, vol_cond_channels=0, vec_cond_dim=0, device="cpu"):
    """Construct a small DiffusionUNet3D suitable for distributed sanity checks."""
    return DiffusionUNet3D(
        x_channels=x_channels,
        vol_cond_channels=vol_cond_channels,
        vec_cond_dim=vec_cond_dim,
        num_levels=2,
        model_channels=16,
        channel_mult=[1, 2],
        num_blocks=1,
        attention_levels=[1],
        dropout=0.0,
    ).to(device)


@pytest.mark.multigpu_static
def test_diffusion_unet_3d_distributed(distributed_mesh):
    """Unconditional forward with `x` sharded along the H axis."""
    dm = DistributedManager()
    B, C, D, H, W = 2, 2, 8, 32, 32

    model = _build_model(x_channels=C, device=dm.device)
    model = distribute_module(model, device_mesh=distributed_mesh)

    x = torch.randn(B, C, D, H, W, device=dm.device)
    t = torch.rand(B, device=dm.device)

    # Shard along H (dim 3): every spatial dim of a 5D volume is fair game,
    # H is the conventional choice in the 2D SongUNet tests.
    placements = (Shard(3),)
    x_sharded = scatter_tensor(x, 0, distributed_mesh, placements, requires_grad=False)

    out = model(x_sharded, t)

    assert out.shape == (B, C, D, H, W)
    assert out._spec.placements == x_sharded._spec.placements


@pytest.mark.multigpu_static
def test_diffusion_unet_3d_conditional_distributed(distributed_mesh):
    """Forward with vector + volume conditioning, sharded along H."""
    dm = DistributedManager()
    B, C, D, H, W = 2, 2, 8, 32, 32
    C_vol, D_vec = 2, 8

    model = _build_model(
        x_channels=C,
        vol_cond_channels=C_vol,
        vec_cond_dim=D_vec,
        device=dm.device,
    )
    model = distribute_module(model, device_mesh=distributed_mesh)

    x = torch.randn(B, C, D, H, W, device=dm.device)
    t = torch.rand(B, device=dm.device)
    volume = torch.randn(B, C_vol, D, H, W, device=dm.device)
    vector = torch.randn(B, D_vec, device=dm.device)

    placements = (Shard(3),)
    x_sharded = scatter_tensor(x, 0, distributed_mesh, placements, requires_grad=False)
    volume_sharded = scatter_tensor(
        volume, 0, distributed_mesh, placements, requires_grad=False
    )

    condition = TensorDict(
        {"vector": vector, "volume": volume_sharded},
        batch_size=[B],
    )

    out = model(x_sharded, t, condition=condition)

    assert out.shape == (B, C, D, H, W)
    assert out._spec.placements == x_sharded._spec.placements
