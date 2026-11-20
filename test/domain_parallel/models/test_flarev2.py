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

r"""Domain-parallel tests for FLAREv2 and FLARE++ (DDP strategy).

FLAREv2 attention makes two SDPA passes with mixed placements (op-level
coverage in ``test/domain_parallel/ops/test_sdpa.py``):

- pass 1: ``sdpa(G_replicated, k_sharded, v_sharded)`` — the learned global
  queries attend to the sharded point cloud.
- pass 2: ``sdpa(q_sharded, K_replicated, V_replicated)`` — embarrassingly
  parallel over the query shards.

The per-head query/key norms (RMSNorm or LayerNorm over the head dim) and the
identity-initialized K/V projections are token-wise, so they keep the token
sharding.

FLARE++ adds a query-synthesis pass ``sdpa(Q0_replicated, k0_sharded,
v0_sharded)`` (the same placement pattern as FLAREv2's pass 1); its output and
the fixed queries are replicated, so the fp32 gated mix stays replicated and
the routing passes are FLAREv2's.
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.experimental.models.flarev2 import FLAREPlusPlusv2, FLAREv2
from test.domain_parallel.models.harness import (
    DomainParallelModelCase,
    run_domain_parallel_model_check,
)


def _build_flare(cls=FLAREv2, **kwargs):
    def build(device):
        model = cls(
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


def _check_output(n_points, batch=1):
    def check(output):
        assert output.shape == (batch, n_points, 2)
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


def _irregular_case(
    name="flarev2-irregular", cls=FLAREv2, batch=1, n_points=16384, **model_kwargs
):
    def build_inputs(device):
        fx = torch.randn(batch, n_points, 3, device=device)
        embedding = torch.randn(batch, n_points, 5, device=device)
        return (fx,), {"embedding": embedding}

    return DomainParallelModelCase(
        name=name,
        build_model=_build_flare(cls, **model_kwargs),
        build_inputs=build_inputs,
        shard_inputs=_shard_irregular,
        strategy="ddp",
        output_check_fn=_check_output(n_points, batch),
        atol=1e-4,
        rtol=1e-4,
    )


_CASES = [
    _irregular_case(),
    _irregular_case(
        "flarev2-irregular-qk-norm-layernorm", qk_norm=True, use_rmsnorm=False
    ),
    _irregular_case("flarev2-irregular-no-qk-norm", qk_norm=False),
    _irregular_case("flareppv2-irregular", FLAREPlusPlusv2),
    _irregular_case(
        "flareppv2-irregular-unshared-layernorm",
        FLAREPlusPlusv2,
        share_k0_v0=False,
        use_rmsnorm=False,
    ),
    _irregular_case(
        "flareppv2-irregular-unshared-rmsnorm", FLAREPlusPlusv2, share_k0_v0=False
    ),
    _irregular_case(
        "flareppv2-irregular-shared-layernorm",
        FLAREPlusPlusv2,
        use_rmsnorm=False,
        k_norm=False,
        q_fixed_norm=False,
    ),
    # Batch > 1 and a token count that does not split evenly across ranks.
    _irregular_case("flarev2-irregular-uneven", batch=2, n_points=1001),
    _irregular_case(
        "flareppv2-irregular-uneven-unshared-layernorm",
        FLAREPlusPlusv2,
        batch=3,
        n_points=777,
        share_k0_v0=False,
        use_rmsnorm=False,
    ),
]


@pytest.mark.multigpu_static
@pytest.mark.timeout(600)
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_flarev2_distributed(distributed_mesh, case):
    run_domain_parallel_model_check(case, mesh=distributed_mesh)
