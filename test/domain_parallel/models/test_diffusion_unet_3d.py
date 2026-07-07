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

"""Domain-parallel tests for DiffusionUNet3D.

The 3D U-Net backbone is distributed with DDP (its parameters are all
replicated) while the volumetric input is a ``ShardTensor`` sharded along the
outermost spatial axis (D, dim 2); ``ShardTensor`` auto-promotion lets the plain
(replicated) weights meet the sharded activations. The shared harness in
``test/domain_parallel/models/harness.py`` runs the distributed forward/backward
and checks the output and gradients against a single-GPU reference.
"""

import pytest
import torch
from tensordict import TensorDict
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.experimental.models.diffusion_unets import DiffusionUNet3D
from test.domain_parallel.models.harness import (
    DomainParallelModelCase,
    run_domain_parallel_model_check,
)

# Small volume; D (dim 2) is the sharded spatial axis. D is the outermost
# spatial dim, so the attention block's (D, H, W) -> D*H*W flatten (and its
# inverse) keeps the shard contiguous and unambiguous. D is sized so the
# deepest U-Net level (D / 2 for num_levels=2) stays divisible by common
# GPU counts (2/4/8).
B, C, D, H, W = 2, 2, 32, 32, 32
C_VOL, D_VEC = 2, 8


def _build_model(vol_cond_channels=0, vec_cond_dim=0):
    """Construct a small DiffusionUNet3D suitable for distributed sanity checks."""

    def build(device):
        return DiffusionUNet3D(
            x_channels=C,
            vol_cond_channels=vol_cond_channels,
            vec_cond_dim=vec_cond_dim,
            num_levels=2,
            model_channels=16,
            channel_mult=[1, 2],
            num_blocks=1,
            attention_levels=[1],
            dropout=0.0,
        ).to(device)

    return build


def _check_output(d_out):
    """Output stays dense-shaped and sharded along D, like the input."""
    assert d_out.shape == (B, C, D, H, W)
    assert d_out._spec.placements == (Shard(2),)


def _unconditional_inputs(device):
    x = torch.randn(B, C, D, H, W, device=device)
    t = torch.rand(B, device=device)
    return (x, t), {}


def _shard_unconditional(args, kwargs, mesh):
    x, t = args
    # Shard x along D (dim 2); t is a per-sample scalar and stays plain so
    # ShardTensor auto-promotes it against the (replicated) freqs buffer.
    x_sharded = scatter_tensor(x, 0, mesh, (Shard(2),), requires_grad=False)
    return (x_sharded, t), kwargs


def _conditional_inputs(device):
    x = torch.randn(B, C, D, H, W, device=device)
    t = torch.rand(B, device=device)
    volume = torch.randn(B, C_VOL, D, H, W, device=device)
    vector = torch.randn(B, D_VEC, device=device)
    condition = TensorDict({"vector": vector, "volume": volume}, batch_size=[B])
    return (x, t), {"condition": condition}


def _shard_conditional(args, kwargs, mesh):
    x, t = args
    condition = kwargs["condition"]
    x_sharded = scatter_tensor(x, 0, mesh, (Shard(2),), requires_grad=False)
    volume_sharded = scatter_tensor(
        condition["volume"], 0, mesh, (Shard(2),), requires_grad=False
    )
    # vector is a per-sample condition (no spatial axis): keep it plain.
    sharded_condition = TensorDict(
        {"vector": condition["vector"], "volume": volume_sharded},
        batch_size=[B],
    )
    return (x_sharded, t), {"condition": sharded_condition}


_CASES = [
    DomainParallelModelCase(
        name="unconditional",
        build_model=_build_model(),
        build_inputs=_unconditional_inputs,
        shard_inputs=_shard_unconditional,
        output_check_fn=_check_output,
    ),
    DomainParallelModelCase(
        name="conditional",
        build_model=_build_model(vol_cond_channels=C_VOL, vec_cond_dim=D_VEC),
        build_inputs=_conditional_inputs,
        shard_inputs=_shard_conditional,
        output_check_fn=_check_output,
    ),
]


@pytest.mark.multigpu_static
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_diffusion_unet_3d_distributed(distributed_mesh, case):
    """Distributed forward/backward matches a single-GPU reference."""
    run_domain_parallel_model_check(case, mesh=distributed_mesh)
