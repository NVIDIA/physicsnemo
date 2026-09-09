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

r"""Domain-parallel tests for FLARE (DDP strategy, use_te=False).

FLARE attention makes two SDPA passes with mixed placements (op-level
coverage in ``test/domain_parallel/ops/test_sdpa.py``):

- pass 1: ``sdpa(G_replicated, k_sharded, v_sharded)`` — the learned global
  queries attend to the sharded point cloud.
- pass 2: ``sdpa(q_sharded, K_replicated, V_replicated)`` — embarrassingly
  parallel over the query shards.

The ``unified_pos=True`` case shards the model's ``(1, N, ref^2)``
position-grid buffer over the domain mesh (fsdp_spatial pattern,
``test_dit.py`` template).
"""

import math

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.models.flare import FLARE
from test.domain_parallel.models.harness import (
    DomainParallelModelCase,
    run_domain_parallel_model_check,
)

_SIDE_BY_NDIMS = {2: 128, 3: 24}


def _build_flare(structured_shape=None, unified_pos=False, ref=8):
    def build(device):
        model = FLARE(
            functional_dim=3,
            out_dim=2,
            # unified_pos derives embedding_dim = ref^2 from the grid buffer.
            embedding_dim=None if unified_pos else 5,
            n_layers=2,
            n_hidden=32,
            dropout=0.0,
            n_head=4,
            act="gelu",
            mlp_ratio=1,
            slice_num=8,
            unified_pos=unified_pos,
            ref=ref,
            structured_shape=structured_shape,
            time_input=False,
            use_te=False,
        )
        return model.to(device)

    return build


def _check_output(n_points):
    def check(output):
        assert output.shape == (1, n_points, 2)
        assert output._spec.placements == (Shard(1),)

    return check


def _irregular_case():
    n_points = 16384

    def build_inputs(device):
        fx = torch.randn(1, n_points, 3, device=device)
        embedding = torch.randn(1, n_points, 5, device=device)
        return (fx,), {"embedding": embedding}

    def shard_inputs(args, kwargs, mesh):
        (fx,) = args
        src = dist.get_global_rank(mesh.get_group(), 0)
        sharded_fx = scatter_tensor(fx, src, mesh, (Shard(1),), requires_grad=False)
        sharded_embedding = scatter_tensor(
            kwargs["embedding"], src, mesh, (Shard(1),), requires_grad=False
        )
        return (sharded_fx,), {"embedding": sharded_embedding}

    return DomainParallelModelCase(
        name="flare-irregular",
        build_model=_build_flare(None),
        build_inputs=build_inputs,
        shard_inputs=shard_inputs,
        strategy="ddp",
        output_check_fn=_check_output(n_points),
        atol=1e-4,
        rtol=1e-4,
    )


def _structured_2d_case():
    dims = (_SIDE_BY_NDIMS[2],) * 2
    n_points = math.prod(dims)

    def build_inputs(device):
        fx = torch.randn(1, *dims, 3, device=device)
        embedding = torch.randn(1, *dims, 5, device=device)
        return (fx,), {"embedding": embedding}

    def shard_inputs(args, kwargs, mesh):
        (fx,) = args
        src = dist.get_global_rank(mesh.get_group(), 0)
        sharded_fx = scatter_tensor(
            fx, src, mesh, (Shard(1),), requires_grad=False
        ).reshape(1, -1, 3)
        sharded_embedding = scatter_tensor(
            kwargs["embedding"], src, mesh, (Shard(1),), requires_grad=False
        ).reshape(1, -1, 5)
        return (sharded_fx,), {"embedding": sharded_embedding}

    return DomainParallelModelCase(
        name="flare-structured2d",
        build_model=_build_flare(dims),
        build_inputs=build_inputs,
        shard_inputs=shard_inputs,
        strategy="ddp",
        output_check_fn=_check_output(n_points),
        atol=1e-3,
        rtol=1e-3,
    )


_CASES = [_irregular_case(), _structured_2d_case()]


@pytest.mark.multigpu_static
@pytest.mark.timeout(600)
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_flare_distributed(distributed_mesh, case):
    run_domain_parallel_model_check(case, mesh=distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(600)
def test_flare_unified_pos_buffer_sharded(distributed_mesh_2d):
    r"""``unified_pos=True``: the ``(1, N, ref^2)`` grid buffer is point-dim
    sized and must be sharded over the domain mesh (fsdp_spatial pattern),
    or the internal ``cat(embedding, fx)`` fights the input sharding.

    Mirrors ``test_dit.py``: 2D mesh (ddp x domain), exactly 4 ranks.
    The buffer is named ``embedding`` — not matched by
    ``default_spatial_param_selector`` — so the case supplies its own
    selector.
    """
    if dist.get_world_size() != 4:
        pytest.skip("unified_pos fsdp_spatial case is written for exactly 4 ranks")

    dims = (_SIDE_BY_NDIMS[2],) * 2
    n_points = math.prod(dims)

    def build_inputs(device):
        fx = torch.randn(1, *dims, 3, device=device)
        return (fx,), {}

    def shard_inputs(args, kwargs, mesh):
        (fx,) = args
        # The domain submesh does not contain global rank 0 for every DDP
        # replica: scatter from each domain group's own first rank
        # (test_dit.py pattern), or the other replicas hang waiting on a
        # source outside their group.
        domain_src = dist.get_global_rank(mesh.get_group(), 0)
        sharded_fx = scatter_tensor(
            fx, domain_src, mesh, (Shard(1),), requires_grad=False
        ).reshape(1, -1, 3)
        return (sharded_fx,), {}

    def spatial_selector(name):
        # FLARE's unified-pos grid buffer: (1, N, ref^2) -> shard dim 1.
        if name.endswith("embedding"):
            return 1
        return None

    case = DomainParallelModelCase(
        name="flare-unified-pos",
        build_model=_build_flare(dims, unified_pos=True, ref=8),
        build_inputs=build_inputs,
        shard_inputs=shard_inputs,
        strategy="fsdp_spatial",
        spatial_param_selector=spatial_selector,
        output_check_fn=_check_output(n_points),
        atol=1e-3,
        rtol=1e-3,
    )

    ddp_mesh = distributed_mesh_2d["axis1"]
    domain_mesh = distributed_mesh_2d["axis2"]
    run_domain_parallel_model_check(
        case, mesh=distributed_mesh_2d, ddp_mesh=ddp_mesh, domain_mesh=domain_mesh
    )
