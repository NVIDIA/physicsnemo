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

r"""Compile-traceability tests for the refactored ShardTensor autograd ops.

These tests guarantee that the custom ``torch.autograd.Function`` subclasses
in ``physicsnemo/domain_parallel/`` can be traced through
``torch.compile(backend="aot_eager", fullgraph=True)``. The migration from
the old-style ``forward(ctx, ...)`` API to the new-style
``forward(...)`` + ``setup_context(ctx, inputs, output)`` API is the
prerequisite for AOTAutograd to traverse the backward graph; this file is
the regression net.

The assertions here are intentionally lightweight: each test compiles a
small module that exercises one refactored op and verifies that forward
(and, where backward is supported, ``loss.backward()``) does not raise.
End-to-end numerical correctness is already covered by the sibling
non-compile tests in this directory.
"""

from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.domain_parallel.shard_utils.conv_patches import ConvGradReducer
from physicsnemo.domain_parallel.shard_utils.halo import (
    HaloConfig,
    halo_padding,
    unhalo_padding,
)
from physicsnemo.domain_parallel.shard_utils.point_cloud_ops import GradReducer


def _scalar_loss(out: Any) -> torch.Tensor:
    """Reduce arbitrary tensor-like outputs to a scalar for ``.backward()``."""
    if isinstance(out, tuple):
        out = out[0]
    return out.float().sum()


def _run_compile_fwd_bwd(
    module: torch.nn.Module,
    inputs: list,
    *,
    backward: bool = True,
    fullgraph: bool = True,
) -> Any:
    r"""Compile ``module`` with ``aot_eager`` and run forward (+ optional bwd).

    Smoke check that the autograd Function dispatched inside ``module`` is
    AOT-traceable. Resets dynamo first so each test is independent.
    """
    torch._dynamo.reset()
    compiled = torch.compile(module, backend="aot_eager", fullgraph=fullgraph)
    output = compiled(*inputs)

    if backward:
        loss = _scalar_loss(output)
        loss.backward()
    return output


# ---------------------------------------------------------------------------
# Module wrappers
# ---------------------------------------------------------------------------


class MeanWrapper(torch.nn.Module):
    r"""``tensor.mean(dim)`` on a ShardTensor (exercises ``ShardedMean``)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.mean(dim=self.dim)


class ViewWrapper(torch.nn.Module):
    r"""``tensor.view(target_shape)`` on a ShardTensor (exercises ``ShardedView``)."""

    def __init__(self, target_shape: tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(self.target_shape)


class RedistributeWrapper(torch.nn.Module):
    r"""``tensor.redistribute(...)`` (exercises ``ShardRedistribute``)."""

    def __init__(self, mesh, placements):
        super().__init__()
        self.mesh = mesh
        self.placements = placements

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.redistribute(self.mesh, self.placements)


class IndexSelectWrapper(torch.nn.Module):
    r"""``torch.index_select(...)`` on a ShardTensor (exercises ``ShardedIndexSelect``)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return torch.index_select(tensor, self.dim, index.flatten())


class UnhaloPaddingWrapper(torch.nn.Module):
    r"""``unhalo_padding(...)`` (exercises ``UnHaloPadding``)."""

    def __init__(self, mesh, halo_config: HaloConfig):
        super().__init__()
        self.mesh = mesh
        self.halo_config = halo_config

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return unhalo_padding(tensor, self.mesh, self.halo_config)


class HaloPaddingWrapper(torch.nn.Module):
    r"""``halo_padding(...)`` (exercises ``HaloPadding``).

    Unlike ``UnHaloPadding``, this drives ``perform_halo_collective`` — the
    funcol ``all_to_all_single_autograd`` exchange — in both forward and
    backward.
    """

    def __init__(self, mesh, halo_config: HaloConfig):
        super().__init__()
        self.mesh = mesh
        self.halo_config = halo_config

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return halo_padding(tensor, self.mesh, self.halo_config)


class ConvWrapper(torch.nn.Module):
    r"""``nn.Conv1d`` on a ShardTensor input.

    End-to-end sharded convolution: halo exchange (funcol a2a) on the input
    plus ``ConvGradReducer`` on the weight gradients, all inside one compiled
    region.
    """

    def __init__(self, conv: torch.nn.Module):
        super().__init__()
        self.conv = conv

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.conv(tensor)


class GroupNormWrapper(torch.nn.Module):
    r"""``F.group_norm`` on a ShardTensor (exercises ``PartialGroupNorm``)."""

    def __init__(self, num_groups: int, num_channels: int):
        super().__init__()
        self.gn = torch.nn.GroupNorm(num_groups, num_channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.gn(tensor)


class SDPAWrapper(torch.nn.Module):
    r"""``F.scaled_dot_product_attention`` on sharded Q/K/V (exercises RingSDPA)."""

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


class GradReducerWrapper(torch.nn.Module):
    r"""``GradReducer.apply(tensor, spec)`` (exercises ``GradReducer``).

    The ``spec`` is captured as a non-tensor module attribute so it is a
    constant from dynamo's perspective. ``GradReducer`` is the trivial
    identity in forward; the work happens in backward (all-reduce on
    replicated mesh dims).
    """

    def __init__(self, spec):
        super().__init__()
        self.spec = spec

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return GradReducer.apply(tensor, self.spec)


class ConvGradReducerWrapper(torch.nn.Module):
    r"""``ConvGradReducer.apply(tensor, spec)`` (exercises ``ConvGradReducer``).

    The ``spec`` is captured as a non-tensor module attribute so it is a
    constant from dynamo's perspective. ``ConvGradReducer`` is the trivial
    identity in forward; the work happens in backward (all-reduce on sharded
    mesh dims).
    """

    def __init__(self, spec):
        super().__init__()
        self.spec = spec

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return ConvGradReducer.apply(tensor, self.spec)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_sharded_mean_1d(distributed_mesh):
    r"""Compile + backward through ``ShardedMean`` on a sharded dim."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 64, 16)
    original = torch.rand(shape, device=dm.device, requires_grad=True)
    sharded = scatter_tensor(
        original,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(1),),
        requires_grad=True,
    )

    _run_compile_fwd_bwd(MeanWrapper(dim=1), [sharded])


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_sharded_view_1d(distributed_mesh):
    r"""Compile + backward through ``ShardedView`` (merge last two dims)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 64, 8, 4)
    target_shape = (4, 64, 32)
    original = torch.rand(shape, device=dm.device, requires_grad=True)
    sharded = scatter_tensor(
        original,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(1),),
        requires_grad=True,
    )

    _run_compile_fwd_bwd(ViewWrapper(target_shape=target_shape), [sharded])


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_shard_redistribute_1d(distributed_mesh):
    r"""Compile + backward through ``ShardRedistribute`` (Shard -> Replicate)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 64, 16)
    original = torch.rand(shape, device=dm.device, requires_grad=True)
    sharded = scatter_tensor(
        original,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(1),),
        requires_grad=True,
    )

    _run_compile_fwd_bwd(
        RedistributeWrapper(mesh=distributed_mesh, placements=(Replicate(),)),
        [sharded],
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_shard_redistribute_2d(distributed_mesh_2d):
    r"""Compile + backward through ``ShardRedistribute`` on a 2D mesh."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 64, 32)
    original = torch.rand(shape, device=dm.device, requires_grad=True)
    sharded = scatter_tensor(
        original,
        global_src=0,
        mesh=distributed_mesh_2d,
        placements=(Shard(1), Shard(2)),
        requires_grad=True,
    )

    _run_compile_fwd_bwd(
        RedistributeWrapper(
            mesh=distributed_mesh_2d, placements=(Replicate(), Replicate())
        ),
        [sharded],
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_sharded_index_select_replicated_index_1d(distributed_mesh):
    r"""Compile + backward through ``ShardedIndexSelect`` with a replicated index.

    A replicated ``index`` keeps the output sharding aligned with the input,
    which is the cheaper / less collective-heavy code path inside the op.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (32, 32, 32)
    dim = 1
    n_idx = 8

    original = torch.rand(shape, device=dm.device, requires_grad=True)
    index = torch.randint(low=0, high=shape[dim] - 1, size=(n_idx,), device=dm.device)

    sharded = scatter_tensor(
        original,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(2),),
        requires_grad=True,
    )
    sharded_index = scatter_tensor(
        index,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Replicate(),),
        requires_grad=False,
    )

    _run_compile_fwd_bwd(IndexSelectWrapper(dim=dim), [sharded, sharded_index])


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_unhalo_padding_1d(distributed_mesh):
    r"""Compile + backward through ``UnHaloPadding``.

    Constructs a synthetic tensor that includes halo regions, then drops them
    via the public ``unhalo_padding`` wrapper. Numerical correctness is
    covered by ``test_padding.py``; this test only checks compile.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    local_group = distributed_mesh.get_group(0)
    local_size = dist.get_world_size(group=local_group)
    if local_size < 2:
        pytest.skip("UnHaloPadding requires at least 2 ranks on the mesh dim")

    halo_size = 2
    H = 32
    # Build a per-rank tensor that already has halos baked in: each rank gets
    # a slab of size H + 2*halo_size along dim 2 (except endpoints).
    tensor = torch.rand(2, 4, H + 2 * halo_size, device=dm.device, requires_grad=True)

    halo_config = HaloConfig(
        mesh_dim=0,
        tensor_dim=2,
        halo_size=halo_size,
        edge_padding_size=halo_size,
        communication_method="a2a",
    )

    _run_compile_fwd_bwd(
        UnhaloPaddingWrapper(mesh=distributed_mesh, halo_config=halo_config),
        [tensor],
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_halo_padding_1d(distributed_mesh):
    r"""Compile + backward through ``HaloPadding``'s funcol a2a exchange.

    This is the only compile test that executes ``perform_halo_collective``
    (``UnHaloPadding`` is collective-free in both directions), so it checks
    values against eager, not just traceability: a misrouted exchange would
    still produce well-shaped outputs.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    local_group = distributed_mesh.get_group(0)
    local_size = dist.get_world_size(group=local_group)
    if local_size < 2:
        pytest.skip("HaloPadding requires at least 2 ranks on the mesh dim")

    halo_size = 2
    H = 32
    torch.manual_seed(42 + dm.rank)
    base = torch.rand(2, 4, H, device=dm.device)

    halo_config = HaloConfig(
        mesh_dim=0,
        tensor_dim=2,
        halo_size=halo_size,
        communication_method="a2a",
    )

    # Eager reference: output + input grad.
    eager_input = base.clone().requires_grad_(True)
    eager_out = halo_padding(eager_input, distributed_mesh, halo_config)
    _scalar_loss(eager_out).backward()
    assert eager_input.grad is not None

    compiled_input = base.clone().requires_grad_(True)
    compiled_out = _run_compile_fwd_bwd(
        HaloPaddingWrapper(mesh=distributed_mesh, halo_config=halo_config),
        [compiled_input],
    )

    torch.testing.assert_close(compiled_out, eager_out)
    torch.testing.assert_close(compiled_input.grad, eager_input.grad)


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_sharded_conv1d_1d(distributed_mesh):
    r"""Compile + backward through a sharded Conv1d, value-checked.

    Drives the full sharded-conv chain under one compiled region: halo
    exchange (funcol a2a) on the sharded input and ``ConvGradReducer``'s
    all-reduce on the weight grads. Output, input grad, and weight/bias
    grads are compared against a single-device reference.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    local_group = distributed_mesh.get_group(0)
    local_size = dist.get_world_size(group=local_group)
    if local_size < 2:
        pytest.skip("halo exchange requires at least 2 ranks on the mesh dim")

    # Same weights and same global image on every rank (seeded identically).
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(4, 8, kernel_size=3, padding=1).to(dm.device)
    image = torch.rand(2, 4, 32 * local_size, device=dm.device)

    sharded = scatter_tensor(
        image,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(2),),
        requires_grad=True,
    )

    # Single-device reference with cloned weights.
    ref_conv = torch.nn.Conv1d(4, 8, kernel_size=3, padding=1).to(dm.device)
    ref_conv.load_state_dict(conv.state_dict())
    ref_input = image.clone().requires_grad_(True)
    ref_out = ref_conv(ref_input)
    ref_out.sum().backward()

    out = _run_compile_fwd_bwd(ConvWrapper(conv), [sharded])

    torch.testing.assert_close(out.full_tensor(), ref_out)
    assert sharded.grad is not None
    torch.testing.assert_close(sharded.grad.full_tensor(), ref_input.grad)
    torch.testing.assert_close(conv.weight.grad, ref_conv.weight.grad)
    torch.testing.assert_close(conv.bias.grad, ref_conv.bias.grad)


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_halo_padding_p2p_raises(distributed_mesh):
    r"""``torch.compile`` over a p2p halo exchange must fail with a clear error.

    The p2p branch of ``perform_halo_collective`` (``P2POp`` /
    ``batch_isend_irecv`` / ``set_device``) has no FX representation, so it
    guards itself with a trace-time ``RuntimeError`` pointing users at
    ``'a2a'``. Eager p2p is unaffected by the guard.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    local_group = distributed_mesh.get_group(0)
    local_size = dist.get_world_size(group=local_group)
    if local_size < 2:
        pytest.skip("HaloPadding requires at least 2 ranks on the mesh dim")

    tensor = torch.rand(2, 4, 32, device=dm.device, requires_grad=True)
    halo_config = HaloConfig(
        mesh_dim=0,
        tensor_dim=2,
        halo_size=2,
        communication_method="p2p",
    )

    torch._dynamo.reset()
    compiled = torch.compile(
        HaloPaddingWrapper(mesh=distributed_mesh, halo_config=halo_config),
        backend="aot_eager",
        fullgraph=True,
    )
    with pytest.raises(Exception, match="p2p"):
        compiled(tensor)


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_partial_group_norm_1d(distributed_mesh):
    r"""Compile + backward through ``PartialGroupNorm`` on a ShardTensor.

    Smoke-checks AOTAutograd traceability through the now-3-output
    autograd Function. End-to-end numerical agreement against single-GPU
    eager is covered by ``test_normalization.py``; here we only ensure
    the compile + fwd + bwd path doesn't raise.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    N, C, H, W = 2, 8, 32, 32
    num_groups = 4

    original = torch.rand(N, C, H, W, device=dm.device, requires_grad=True)
    sharded = scatter_tensor(
        original,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(2),),
        requires_grad=True,
    )

    module = GroupNormWrapper(num_groups=num_groups, num_channels=C).to(dm.device)

    _run_compile_fwd_bwd(module, [sharded])


# Note: ``test_compile_ring_sdpa_1d`` (smoke-test that compile around sharded
# SDPA succeeds) was removed because the overlap variant's ``record_stream``
# call cannot survive AOTAutograd functionalization, and the
# ``@torch.compiler.disable`` on ``ring_sdpa`` only suppresses dynamo's
# tracing of the body, not AOT's later re-execution of the captured FX
# graph (which re-enters via ``ShardTensor.__torch_function__`` on the
# captured SDPA node and trips ``record_stream``). Re-enabling compile of
# sharded SDPA requires a separate refactor (drop ``record_stream``, switch
# ``perform_ring_iteration`` to functional p2p collectives, etc.). The
# limitation is documented in ``ring_sdpa``'s docstring in
# ``shard_utils/attention_patches.py``. The eager path is fully covered by
# ``test_sdpa.py`` and ``test_ring_sdpa_overlap.py``.


@pytest.mark.multigpu_static
@pytest.mark.timeout(60)
def test_compile_ring_sdpa_fullgraph_errors(distributed_mesh):
    r"""Regression guard: ``torch.compile`` around sharded ring SDPA must error.

    Today compile around sharded SDPA fails at AOT functionalization of
    ``aten::record_stream`` (an alias-annotated op called inside the
    overlap variant). The exact exception type varies between PyTorch
    versions, but it is *some* ``Exception``. We assert that, so if a
    future refactor accidentally makes compile silently "succeed" without
    actually wiring up a functional-collective ring (the only way it can
    be correct under AOT), this test starts failing and forces us to
    re-evaluate.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    batch_size, num_heads, seq_len, head_dim = 1, 4, 128, 32
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=dm.device, requires_grad=True
    )
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=dm.device, requires_grad=True
    )
    v = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=dm.device, requires_grad=True
    )

    q_s = scatter_tensor(q, 0, distributed_mesh, (Shard(2),), requires_grad=True)
    k_s = scatter_tensor(k, 0, distributed_mesh, (Shard(2),), requires_grad=True)
    v_s = scatter_tensor(v, 0, distributed_mesh, (Shard(2),), requires_grad=True)

    with pytest.raises(Exception, match=r"record_stream|functionaliz"):
        _run_compile_fwd_bwd(SDPAWrapper(), [q_s, k_s, v_s], fullgraph=True)


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_grad_reducer_1d(distributed_mesh):
    r"""Compile + backward through ``GradReducer``.

    Forward is identity; backward all-reduces over each replicated mesh
    dim. We feed a plain tensor + the spec from a Replicate-placed
    ShardTensor so the backward path actually exercises the all-reduce.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()

    base = torch.rand(4, 16, device=dm.device)
    replicated_shard = scatter_tensor(
        base,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Replicate(),),
        requires_grad=False,
    )
    spec = replicated_shard._spec

    tensor = torch.rand(4, 16, device=dm.device, requires_grad=True)

    _run_compile_fwd_bwd(GradReducerWrapper(spec=spec), [tensor])


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_conv_grad_reducer_1d(distributed_mesh):
    r"""Compile + backward through ``ConvGradReducer``.

    Forward is identity; backward all-reduces over each sharded mesh dim.
    We feed a plain tensor + the spec from a Shard-placed ShardTensor so the
    backward path actually exercises the functional all-reduce.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()

    base = torch.rand(4, 16, device=dm.device)
    sharded = scatter_tensor(
        base,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(0),),
        requires_grad=False,
    )
    spec = sharded._spec

    tensor = torch.rand(4, 16, device=dm.device, requires_grad=True)

    _run_compile_fwd_bwd(ConvGradReducerWrapper(spec=spec), [tensor])
