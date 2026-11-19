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


@pytest.mark.multigpu_static
@pytest.mark.timeout(600)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=str)
def test_flare_compile_matches_eager(distributed_mesh, dtype):
    r"""``torch.compile(FLARE)`` on token-sharded inputs matches eager.

    FLARE's learned global queries are a plain parameter that meets the
    sharded keys/values in SDPA; under tracing it reaches the ShardTensor
    handler as a ``FakeTensor`` and must still be promoted to ``Replicate``.
    Compares the compiled forward output and parameter gradients to eager on
    the same sharded input. The fp16 case runs the replicated-query SDPA on
    the flash kernel where PyTorch supports it; there compiled and eager are
    each compared to an fp32 eager reference with the same weights, and the
    compiled error must stay within twice eager's (inductor's fp16 rounding
    differs from eager's, so a direct fp16-vs-fp16 bound would be arbitrary).
    """
    n_points = 4096
    device = torch.device("cuda", torch.cuda.current_device())
    model = _build_flare(None)(device).to(dtype)
    model.eval()

    fx = torch.randn(1, n_points, 3, device=device, dtype=dtype)
    embedding = torch.randn(1, n_points, 5, device=device, dtype=dtype)
    src = dist.get_global_rank(distributed_mesh.get_group(), 0)

    def shard(t):
        return scatter_tensor(
            t, src, distributed_mesh, (Shard(1),), requires_grad=False
        )

    def run(fn, mod, x, emb):
        mod.zero_grad(set_to_none=True)
        out = fn(x, embedding=emb)
        out.float().pow(2).mean().backward()
        return out, [p.grad.clone() for p in mod.parameters() if p.grad is not None]

    sharded_fx, sharded_embedding = shard(fx), shard(embedding)
    eager_out, eager_grads = run(model, model, sharded_fx, sharded_embedding)
    torch._dynamo.reset()
    compiled_out, compiled_grads = run(
        torch.compile(model), model, sharded_fx, sharded_embedding
    )

    _check_output(n_points)(compiled_out)
    assert compiled_out.dtype == dtype
    assert len(compiled_grads) == len(eager_grads) > 0
    assert all(type(g) is torch.Tensor for g in compiled_grads)
    if dtype == torch.float32:
        torch.testing.assert_close(
            compiled_out.to_local(), eager_out.to_local(), atol=1e-4, rtol=1e-4
        )
        for got, want in zip(compiled_grads, eager_grads):
            torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
        return

    model32 = _build_flare(None)(device)
    model32.load_state_dict({k: v.float() for k, v in model.state_dict().items()})
    model32.eval()
    ref_out, ref_grads = run(
        model32, model32, shard(fx.float()), shard(embedding.float())
    )

    def rel_err(a, b):
        a, b = a.float(), b.float()
        return ((a - b).norm() / b.norm().clamp_min(1e-6)).item()

    pairs = [(compiled_out.to_local(), eager_out.to_local(), ref_out.to_local())]
    pairs += list(zip(compiled_grads, eager_grads, ref_grads))
    for i, (got, eager, ref) in enumerate(pairs):
        e_c, e_e = rel_err(got, ref), rel_err(eager, ref)
        name = "output" if i == 0 else f"grad[{i - 1}]"
        assert e_c <= 2 * e_e + 1e-3, (
            f"{name}: compiled fp16 rel err {e_c:.3e} vs eager fp16 {e_e:.3e} "
            "(both against fp32 eager)"
        )
