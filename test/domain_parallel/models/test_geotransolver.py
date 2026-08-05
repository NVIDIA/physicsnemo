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

"""Domain-parallel activation-checkpointing tests for GeoTransolver.

The local and geometry token axes are distributed as ``ShardTensor`` inputs,
while the global context is represented by a replicated ``ShardTensor``.  The
shared harness compares distributed forward/backward execution against a
single-GPU reference.  Running in training mode with activation checkpointing
enabled exercises recomputation through GALE's differentiable cross-rank slice
reductions.
"""

import math

import pytest
import torch
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.models.geotransolver import GeoTransolver
from test.domain_parallel.models.harness import (
    DomainParallelModelCase,
    run_domain_parallel_model_check,
)

_FULL_CHECKPOINT_SCOPE = ("context", "preprocess", "blocks", "output")


def _local_mean_loss(output):
    """Scale each rank-local mean so DDP reduction matches the global mean."""
    if hasattr(output, "to_local"):
        return output.to_local().square().mean() / output.device_mesh.size()
    return output.square().mean()


def _geotransolver_case(
    checkpoint_scope, structured_shape=None, *, activation_checkpointing=True
):
    """Build a GALE case with sharded geometry and replicated global context."""
    batch_size = 1
    token_count = math.prod(structured_shape) if structured_shape else 256
    global_token_count = 4
    token_shape = structured_shape or (token_count,)

    def build_model(device):
        return GeoTransolver(
            functional_dim=8,
            out_dim=3,
            geometry_dim=3,
            global_dim=5,
            n_layers=2,
            n_hidden=32,
            dropout=0.0,
            n_head=4,
            mlp_ratio=2,
            slice_num=8,
            use_te=False,
            include_local_features=False,
            structured_shape=structured_shape,
            attention_type="GALE",
            activation_checkpointing=activation_checkpointing,
            activation_checkpointing_components=checkpoint_scope,
        ).to(device)

    def build_inputs(device):
        local_embedding = torch.randn(batch_size, *token_shape, 8, device=device)
        geometry = torch.randn(batch_size, *token_shape, 3, device=device)
        global_embedding = torch.randn(batch_size, global_token_count, 5, device=device)
        # Keep differentiable tensors positional because the shared harness
        # verifies input gradients for positional arguments.
        return (local_embedding, None, global_embedding, geometry), {}

    def shard_inputs(args, kwargs, mesh):
        local_embedding, local_positions, global_embedding, geometry = args
        return (
            scatter_tensor(
                local_embedding,
                0,
                mesh,
                (Shard(1),),
                requires_grad=True,
            ),
            local_positions,
            scatter_tensor(
                global_embedding,
                0,
                mesh,
                (Replicate(),),
                requires_grad=True,
            ),
            scatter_tensor(
                geometry,
                0,
                mesh,
                (Shard(1),),
                requires_grad=True,
            ),
        ), kwargs

    def check_output(output):
        assert output.shape == (batch_size, *token_shape, 3)
        assert output._spec.placements == (Shard(1),)

    mesh_name = (
        "irregular" if structured_shape is None else f"structured-{len(token_shape)}d"
    )
    if not activation_checkpointing:
        scope_name = "default"
    else:
        scope_name = "full" if checkpoint_scope == _FULL_CHECKPOINT_SCOPE else "blocks"
    tolerance = 5e-2 if structured_shape is None else 2e-2
    return DomainParallelModelCase(
        name=f"{mesh_name}-{scope_name}",
        build_model=build_model,
        build_inputs=build_inputs,
        shard_inputs=shard_inputs,
        output_check_fn=check_output,
        loss_fn=_local_mean_loss,
        train_mode=True,
        # Slice aggregation is reduced in a different floating-point order
        # across ranks than in the one-GPU reference. Cross-attention adds a
        # second reduction-sensitive projection. The irregular attention path
        # has a wider established fp32 envelope than structured mesh kernels;
        # the uncheckpointed control case verifies this is not recomputation
        # drift.
        atol=tolerance,
        rtol=tolerance,
    )


_CASES = [
    _geotransolver_case(("blocks",), activation_checkpointing=False),
    _geotransolver_case(("blocks",)),
    _geotransolver_case(_FULL_CHECKPOINT_SCOPE),
    _geotransolver_case(_FULL_CHECKPOINT_SCOPE, structured_shape=(16, 16)),
    _geotransolver_case(_FULL_CHECKPOINT_SCOPE, structured_shape=(8, 8, 8)),
]


@pytest.mark.multigpu_static
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_geotransolver_activation_checkpointing_distributed(distributed_mesh, case):
    """ShardTensor forward/backward matches one-GPU execution across policies."""
    run_domain_parallel_model_check(case, mesh=distributed_mesh)
