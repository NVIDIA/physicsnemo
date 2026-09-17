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

r"""Domain-parallel tests for GeoTransolver (DDP strategy, use_te=False).

Placement contract:

- ``local_embedding`` (B, N, C) and ``geometry`` (B, N_geo, C) are both
  ``Shard(1)`` on the same 1D domain mesh with *different* shard shapes
  (N and N_geo are independent point clouds).
- ``global_embedding`` stays ``Replicate`` (a handful of global tokens; the
  domino ``_NON_SHARDED_KEYS`` pattern).
- Output is (B, N, out_dim) with ``Shard(1)``, numerically matching the
  gathered single-GPU reference.

The GALE cases exercise sharded slice attention with cross-attention
context; GALE_FA additionally exercises the mixed-placement SDPA paths
(op-level coverage in ``test/domain_parallel/ops/test_sdpa.py``).
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, scatter_tensor
from physicsnemo.models.geotransolver import GeoTransolver
from test.domain_parallel.models.harness import (
    DomainParallelModelCase,
    run_domain_parallel_model_check,
)
from test.domain_parallel.ops.utils import numerical_shard_tensor_check

# Full single-GPU reference forward+backward runs on the gathered tensors,
# so keep the point budgets modest. Sizes divisible by 2/4/8 for N; N_geo
# deliberately NOT divisible so the geometry cloud is unevenly sharded.
_N_POINTS = 16384
_N_GEO = 2345
_N_GLOBAL_TOKENS = 4
_GLOBAL_DIM = 8


def _build_geotransolver(
    attention_type, structured_shape=None, plus=False, include_local_features=False
):
    def build(device):
        local_kwargs = {}
        if include_local_features:
            # Small multi-scale ball-query config: the BQWarp + neighbor-MLP
            # path of the context projector (the recipe's volume setup).
            local_kwargs = dict(
                radii=[0.5, 2.0],
                neighbors_in_radius=[4, 8],
                n_hidden_local=8,
            )
        model = GeoTransolver(
            functional_dim=3,
            out_dim=2,
            geometry_dim=3,
            global_dim=_GLOBAL_DIM,
            n_layers=2,
            n_hidden=32,
            dropout=0.0,
            n_head=4,
            act="gelu",
            mlp_ratio=1,
            slice_num=8,
            use_te=False,
            time_input=False,
            plus=plus,
            include_local_features=include_local_features,
            structured_shape=structured_shape,
            attention_type=attention_type,
            **local_kwargs,
        )
        return model.to(device)

    return build


def _build_inputs(n_points, geometry_points=_N_GEO):
    def build_inputs(device):
        local_embedding = torch.randn(1, n_points, 3, device=device)
        geometry = torch.randn(1, geometry_points, 3, device=device)
        global_embedding = torch.randn(1, _N_GLOBAL_TOKENS, _GLOBAL_DIM, device=device)
        return (local_embedding,), {
            "geometry": geometry,
            "global_embedding": global_embedding,
        }

    return build_inputs


def _shard_inputs(args, kwargs, mesh):
    (local_embedding,) = args
    # Scatter from this domain group's own first rank: on a submesh of a
    # larger mesh, global rank 0 is not in every group and scattering from
    # it hangs the other replicas.
    src = dist.get_global_rank(mesh.get_group(), 0)
    sharded_args = (
        scatter_tensor(local_embedding, src, mesh, (Shard(1),), requires_grad=False),
    )
    sharded_kwargs = {
        # Independent point cloud, unevenly sharded on the same mesh.
        "geometry": scatter_tensor(
            kwargs["geometry"], src, mesh, (Shard(1),), requires_grad=False
        ),
        # Global tokens stay replicated (domino _NON_SHARDED_KEYS pattern).
        "global_embedding": scatter_tensor(
            kwargs["global_embedding"], src, mesh, (Replicate(),), requires_grad=False
        ),
    }
    return sharded_args, sharded_kwargs


def _check_output(n_points):
    def check(output):
        assert output.shape == (1, n_points, 2)
        assert output._spec.placements == (Shard(1),)

    return check


def _irregular_case(attention_type):
    return DomainParallelModelCase(
        name=f"geotransolver-irregular-{attention_type.lower()}",
        build_model=_build_geotransolver(attention_type),
        build_inputs=_build_inputs(_N_POINTS),
        shard_inputs=_shard_inputs,
        strategy="ddp",
        output_check_fn=_check_output(_N_POINTS),
        atol=1e-4,
        rtol=1e-4,
    )


def _structured_2d_case():
    dims = (128, 128)
    n_points = dims[0] * dims[1]

    def build_inputs(device):
        # Structured GALE flattens 4-D inputs internally; the sharded path
        # feeds the flattened layout directly (mirrors test_transolver.py's
        # reshape-after-scatter). geometry must have N_geo == prod(dims) in
        # the structured case and is passed pre-flattened (3-D passes
        # through the flattener untouched).
        local_embedding = torch.randn(1, *dims, 3, device=device)
        geometry = torch.randn(1, n_points, 3, device=device)
        global_embedding = torch.randn(1, _N_GLOBAL_TOKENS, _GLOBAL_DIM, device=device)
        return (local_embedding,), {
            "geometry": geometry,
            "global_embedding": global_embedding,
        }

    def shard_inputs(args, kwargs, mesh):
        (local_embedding,) = args
        src = dist.get_global_rank(mesh.get_group(), 0)
        sharded = scatter_tensor(
            local_embedding, src, mesh, (Shard(1),), requires_grad=False
        ).reshape(1, -1, 3)
        sharded_kwargs = {
            "geometry": scatter_tensor(
                kwargs["geometry"], src, mesh, (Shard(1),), requires_grad=False
            ),
            "global_embedding": scatter_tensor(
                kwargs["global_embedding"],
                src,
                mesh,
                (Replicate(),),
                requires_grad=False,
            ),
        }
        return (sharded,), sharded_kwargs

    return DomainParallelModelCase(
        name="geotransolver-structured2d-gale",
        build_model=_build_geotransolver("GALE", structured_shape=dims),
        build_inputs=build_inputs,
        shard_inputs=shard_inputs,
        strategy="ddp",
        output_check_fn=_check_output(n_points),
        # Conv/halo paths compare against a different cuDNN algorithm choice
        # on the gathered reference; match test_transolver.py's tolerance.
        atol=1e-3,
        rtol=1e-3,
    )


_CASES = [
    _irregular_case("GALE"),
    # GALE_FA has no structured variant; it additionally exercises the
    # mixed-placement SDPA wrapper.
    _irregular_case("GALE_FA"),
    _structured_2d_case(),
]


@pytest.mark.multigpu_static
@pytest.mark.timeout(600)
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_geotransolver_distributed(distributed_mesh, case):
    run_domain_parallel_model_check(case, mesh=distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(600)
def test_geotransolver_local_features_sharded_smoke(distributed_mesh):
    r"""``include_local_features=True`` sharded forward + backward is
    well-formed and every parameter gradient is a plain tensor.

    No reference comparison: ``radius_search`` guarantees neither which
    neighbors are returned under truncation nor their order, and the
    neighbor-feature MLP consumes an order-sensitive concatenation -- so
    the sharded output is a valid but different encoding than a gathered
    single-device run (the op-level tests in ``test_radius_search.py``
    sort and over-request to compare; a model cannot).

    The gradient-type assertion is the load-bearing check: this is the
    configuration (BQWarp + neighbor MLPs + the scalar ``state_mixing``
    gate) where grad-tracking scalars once leaked DTensor gradients onto
    plain parameters, breaking the optimizer's foreach kernels.
    """
    dm = DistributedManager()
    n_points = _N_POINTS // 4
    model = _build_geotransolver("GALE", include_local_features=True)(dm.device)

    src = dist.get_global_rank(distributed_mesh.get_group(), 0)
    torch.manual_seed(11)
    local_embedding = torch.randn(1, n_points, 3, device=dm.device)
    geometry = torch.randn(1, _N_GEO, 3, device=dm.device)
    global_embedding = torch.randn(1, _N_GLOBAL_TOKENS, _GLOBAL_DIM, device=dm.device)

    output = model(
        scatter_tensor(
            local_embedding, src, distributed_mesh, (Shard(1),), requires_grad=False
        ),
        geometry=scatter_tensor(
            geometry, src, distributed_mesh, (Shard(1),), requires_grad=False
        ),
        global_embedding=scatter_tensor(
            global_embedding, src, distributed_mesh, (Replicate(),)
        ),
        local_positions=scatter_tensor(
            local_embedding.clone(),
            src,
            distributed_mesh,
            (Shard(1),),
            requires_grad=False,
        ),
    )

    assert isinstance(output, ShardTensor)
    assert output.shape == (1, n_points, 2)
    assert output._spec.placements == (Shard(1),)
    assert torch.isfinite(output.to_local()).all()

    output.full_tensor().square().mean().backward()

    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        assert type(grad) is torch.Tensor, (
            f"parameter {name!r} received a {type(grad).__name__} gradient"
        )
        assert torch.isfinite(grad).all(), f"non-finite gradient on {name!r}"


@pytest.mark.multigpu_static
@pytest.mark.timeout(600)
def test_geotransolver_plus_sharded_smoke(distributed_mesh):
    r"""``plus=True`` (gumbel noise) sharded forward produces a well-formed
    sharded output.

    No reference comparison: the gumbel noise is drawn from the global RNG
    (``gumbel_softmax.py``), so a sharded draw cannot bitwise-match a
    gathered single-GPU run. This checks that the plus path's sharded
    slice-token handling yields a finite output with the right shape and
    placement.
    """
    dm = DistributedManager()
    model = _build_geotransolver("GALE", plus=True)(dm.device)
    model.eval()

    args, kwargs = _build_inputs(_N_POINTS)(dm.device)
    sharded_args, sharded_kwargs = _shard_inputs(args, kwargs, distributed_mesh)

    output = model(*sharded_args, **sharded_kwargs)

    assert isinstance(output, ShardTensor)
    assert output.shape == (1, _N_POINTS, 2)
    assert output._spec.placements == (Shard(1),)
    assert torch.isfinite(output.to_local()).all()


# ---------------------------------------------------------------------------
# Layer-level tests: isolate the two components with distributed-specific
# logic so a failure names its layer rather than surfacing as an end-to-end
# tolerance error.
# ---------------------------------------------------------------------------


class _ContextBuilderWrapper(nn.Module):
    r"""``GlobalContextBuilder.build_context`` on one embedding stream."""

    def __init__(self, builder: nn.Module):
        super().__init__()
        self.builder = builder

    def forward(self, local_embedding, geometry, global_embedding):
        context, _, _ = self.builder.build_context(
            (local_embedding,), None, geometry, global_embedding
        )
        return context


class _BlockWrapper(nn.Module):
    r"""``GALEBlock`` on one hidden-state stream with an explicit context."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x, context):
        return self.block((x,), context)[0]


def _assert_replicate(output):
    assert output._spec.placements == (Replicate(),), (
        f"build_context must resolve Partial context to Replicate, "
        f"got {output._spec.placements}"
    )


def _assert_sharded_like_hidden(output):
    # The block's output must keep the hidden states' sharding: a block that
    # quietly gathered to Replicate would still pass the value comparison.
    assert output.shape == (1, _N_POINTS, 32)
    assert output._spec.placements == (Shard(1),), (
        f"block output must keep the input's Shard(1) placement, "
        f"got {output._spec.placements}"
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
def test_context_builder_distributed(distributed_mesh):
    r"""Sharded context build matches the unsharded one and is Replicate.

    The tokenizers reduce over the sharded point axis, so the assembled
    context's Partial contributions must be resolved (one all-reduce) before
    it is handed to the blocks -- the output placement assertion pins that
    contract.
    """
    dm = DistributedManager()
    model = _build_geotransolver("GALE")(dm.device)
    model.eval()

    src = dist.get_global_rank(distributed_mesh.get_group(), 0)
    torch.manual_seed(3)
    local_embedding = torch.randn(1, _N_POINTS, 3, device=dm.device)
    geometry = torch.randn(1, _N_GEO, 3, device=dm.device)
    global_embedding = torch.randn(1, _N_GLOBAL_TOKENS, _GLOBAL_DIM, device=dm.device)

    numerical_shard_tensor_check(
        distributed_mesh,
        _ContextBuilderWrapper(model.context_builder),
        [
            scatter_tensor(local_embedding, src, distributed_mesh, (Shard(1),)),
            scatter_tensor(geometry, src, distributed_mesh, (Shard(1),)),
            scatter_tensor(global_embedding, src, distributed_mesh, (Replicate(),)),
        ],
        {},
        check_grads=True,
        atol=1e-4,
        rtol=1e-4,
        output_check_fn=_assert_replicate,
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
@pytest.mark.parametrize("attention_type", ["GALE", "GALE_FA"])
def test_gale_block_distributed(distributed_mesh, attention_type):
    r"""One GALEBlock: sharded hidden states + replicated context vs reference.

    Covers both attention backends; GALE_FA routes through the
    mixed-placement SDPA paths.
    """
    dm = DistributedManager()
    model = _build_geotransolver(attention_type)(dm.device)
    model.eval()
    block = model.blocks[0]

    src = dist.get_global_rank(distributed_mesh.get_group(), 0)
    torch.manual_seed(5)
    hidden = torch.randn(1, _N_POINTS, 32, device=dm.device)
    # Build a correctly-shaped context from the model's own builder (plain
    # tensors in -> plain context out), then replicate it.
    local_embedding = torch.randn(1, _N_POINTS, 3, device=dm.device)
    geometry = torch.randn(1, _N_GEO, 3, device=dm.device)
    global_embedding = torch.randn(1, _N_GLOBAL_TOKENS, _GLOBAL_DIM, device=dm.device)
    with torch.no_grad():
        context, _, _ = model.context_builder.build_context(
            (local_embedding,), None, geometry, global_embedding
        )

    numerical_shard_tensor_check(
        distributed_mesh,
        _BlockWrapper(block),
        [
            scatter_tensor(
                hidden, src, distributed_mesh, (Shard(1),), requires_grad=True
            ),
            scatter_tensor(context, src, distributed_mesh, (Replicate(),)),
        ],
        {},
        check_grads=True,
        atol=1e-4,
        rtol=1e-4,
        output_check_fn=_assert_sharded_like_hidden,
    )
