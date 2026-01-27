# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Sharding tests for model-parallel operations.

These tests require multi-GPU and should be run with torchrun:
    torchrun --nproc_per_node=2 -m pytest test/models/healda/test_sharding.py
"""

import os

import pytest
import torch
import torch.distributed as dist

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.healda import dit
from physicsnemo.models.healda.sharding import shard_t, shard_x


def _init_distributed():
    """Initialize distributed using PNM's DistributedManager."""
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()


def _is_multi_rank_launch():
    """Check if launched with torchrun/mpirun with multiple ranks."""
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


requires_multi_gpu = pytest.mark.skipif(
    not _is_multi_rank_launch(),
    reason="Requires torchrun with >=2 ranks (torchrun --nproc_per_node=2)",
)


@requires_multi_gpu
def test_sharding_routines():
    if not torch.distributed.is_initialized():
        _init_distributed()

    group_size = 2
    world_size = dist.get_world_size()
    mesh = dist.init_device_mesh("cuda", [world_size // group_size, group_size])
    group = mesh.get_group(1)

    b, c, t, x = 1, 3, 2, 8

    tensor = torch.arange(b * c * t * x).view(b, t, x, c).cuda()
    out = shard_t(tensor, group)

    assert out.shape == (b, t // 2, x * 2, c)

    # back
    roundtrip = shard_x(out, group)
    assert torch.all(roundtrip == tensor)


@requires_multi_gpu
def test_sharding_dit():
    if not torch.distributed.is_initialized():
        _init_distributed()

    group_size = 2

    world_size = dist.get_world_size()
    mesh = dist.init_device_mesh("cuda", [world_size // group_size, group_size])
    group = mesh.get_group(1)
    level_model = 4  # HPX16

    b, c, t = 1, 3, 2

    model = dit.DiT(
        num_attention_heads=1,
        in_channels=c,
        out_channels=c,
        num_layers=7,
        temporal_attention=True,
        time_length=2 * group_size,
    )
    x = 12 * 4**model._level_in

    # register hooks to ensure that temporal attn dims have the expected shape
    def ensure_x_sharded(mod, inputs):
        (z,) = inputs  # b t x c
        x = 12 * 4**level_model
        assert z.shape[:3] == (1 * b, t * group_size, x // group_size)

    for module in model.modules():
        if isinstance(module, dit.TemporalAttention):
            module.register_forward_pre_hook(ensure_x_sharded)

    model.cuda()
    model.set_parallel_group(group)

    tensor = torch.arange(b * c * t * x).view(b, c, t, x).cuda().float()
    noise_labels = torch.ones(b).cuda()
    class_labels = torch.empty([b, 0], device=tensor.device)

    out = model(
        tensor,
        noise_labels=noise_labels,
        class_labels=class_labels,
        second_of_day=torch.zeros([b, t]).cuda(),
        day_of_year=torch.zeros([b, t]).cuda(),
        timestamp=torch.zeros(1).cuda(),
    )

    assert out.out.shape == tensor.shape
