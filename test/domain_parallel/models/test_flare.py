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

r"""Domain-parallel tests for FLARE (DDP strategy).

FLARE attention makes two SDPA passes with mixed placements (op-level
coverage in ``test/domain_parallel/ops/test_sdpa.py``):

- pass 1: ``sdpa(G_replicated, k_sharded, v_sharded)`` — the learned global
  queries attend to the sharded point cloud.
- pass 2: ``sdpa(q_sharded, K_replicated, V_replicated)`` — embarrassingly
  parallel over the query shards.

The per-head query/key norms (RMSNorm or LayerNorm over the head dim) and the
identity-initialized K/V projections are token-wise, so they keep the token
sharding. FLARE++ does not support token-sharded inputs yet and must raise.
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.models.flare import FLARE, FLAREPlusPlus
from test.domain_parallel.models.harness import (
    DomainParallelModelCase,
    run_domain_parallel_model_check,
)


def _build_flare(**kwargs):
    def build(device):
        model = FLARE(
            functional_dim=3,
            out_dim=2,
            embedding_dim=5,
            n_layers=2,
            n_hidden=32,
            dropout=0.0,
            n_head=4,
            mlp_ratio=1,
            slice_num=8,
            **kwargs,
        )
        return model.to(device)

    return build


def _check_output(n_points):
    def check(output):
        assert output.shape == (1, n_points, 2)
        assert output._spec.placements == (Shard(1),)

    return check


def _shard_irregular(args, kwargs, mesh):
    (fx,) = args
    src = dist.get_global_rank(mesh.get_group(), 0)
    sharded_fx = scatter_tensor(fx, src, mesh, (Shard(1),), requires_grad=False)
    sharded_embedding = scatter_tensor(
        kwargs["embedding"], src, mesh, (Shard(1),), requires_grad=False
    )
    return (sharded_fx,), {"embedding": sharded_embedding}


def _irregular_case(name="flare-irregular", **model_kwargs):
    n_points = 16384

    def build_inputs(device):
        fx = torch.randn(1, n_points, 3, device=device)
        embedding = torch.randn(1, n_points, 5, device=device)
        return (fx,), {"embedding": embedding}

    return DomainParallelModelCase(
        name=name,
        build_model=_build_flare(**model_kwargs),
        build_inputs=build_inputs,
        shard_inputs=_shard_irregular,
        strategy="ddp",
        output_check_fn=_check_output(n_points),
        atol=1e-4,
        rtol=1e-4,
    )


_CASES = [
    _irregular_case(),
    _irregular_case(
        "flare-irregular-qk-norm-layernorm", qk_norm=True, use_rmsnorm=False
    ),
]


@pytest.mark.multigpu_static
@pytest.mark.timeout(600)
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_flare_distributed(distributed_mesh, case):
    run_domain_parallel_model_check(case, mesh=distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(600)
def test_flare_plus_plus_rejects_token_sharded_inputs(distributed_mesh):
    r"""FLARE++ raises on token-sharded inputs instead of computing wrong results."""
    from physicsnemo.distributed import DistributedManager

    device = DistributedManager().device
    model = FLAREPlusPlus(
        functional_dim=3,
        out_dim=2,
        embedding_dim=5,
        n_layers=1,
        n_hidden=32,
        n_head=4,
        slice_num=8,
    ).to(device)
    fx = torch.randn(1, 64, 3, device=device)
    embedding = torch.randn(1, 64, 5, device=device)
    (sharded_fx,), kwargs = _shard_irregular(
        (fx,), {"embedding": embedding}, distributed_mesh
    )
    with pytest.raises(NotImplementedError, match="token-sharded"):
        model(sharded_fx, **kwargs)
