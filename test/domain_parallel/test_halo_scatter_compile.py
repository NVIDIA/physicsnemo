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

r"""``torch.compile`` tests checking ``halo_scatter_correct`` (fwd+bwd) against an independent funcol reference."""

import os

import pytest
import torch
import torch.distributed as dist

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel.shard_utils.halo_scatter import (
    halo_forward_exchange,
    halo_reverse_exchange,
    halo_scatter_correct,
    pack_halo_routing,
)


def _ring_routing(rank: int, world_size: int, n_owned: int, lend: int):
    r"""Directed-ring halo: each rank lends its first ``lend`` rows to the next rank."""
    send_indices = [[] for _ in range(world_size)]
    send_indices[(rank + 1) % world_size] = list(range(lend))
    send_sizes = [[0] * world_size for _ in range(world_size)]
    for i in range(world_size):
        send_sizes[i][(i + 1) % world_size] = lend
    n_ghost = sum(send_sizes[i][rank] for i in range(world_size))
    return n_owned, n_owned + n_ghost, send_indices, send_sizes


def run_halo_scatter_correct(mesh, backend, n_owned=6, lend=2, feat=4):
    device = DistributedManager().device
    group = mesh.get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    n_owned, n_padded, send_indices, send_sizes = _ring_routing(
        rank, world_size, n_owned, lend
    )
    send_idx_t = [
        torch.tensor(s, dtype=torch.int64, device=device) for s in send_indices
    ]
    routing = pack_halo_routing(
        send_indices, send_sizes, n_owned, rank, world_size, device=device
    )

    torch.manual_seed(100 + rank)
    padded0 = torch.randn(n_padded, feat, dtype=torch.float64, device=device)

    def fn(p, r):
        return halo_scatter_correct(p, r, group=mesh)

    # Independent oracle: self-adjoint forward(reverse(.)), so grad == op(ones).
    def _plain_correct(p):
        return halo_forward_exchange(
            halo_reverse_exchange(
                p, n_owned, send_idx_t, send_sizes, rank, world_size, mesh
            ),
            send_idx_t,
            send_sizes,
            rank,
            world_size,
            mesh,
        )

    ref_fwd = _plain_correct(padded0)
    ref_grad = _plain_correct(torch.ones_like(padded0))
    assert not torch.allclose(ref_fwd, padded0), "halo correction is a no-op here"

    # Closed-form check: correct(ones) on owned row i == 1 + number of borrowers.
    lent_count = torch.zeros(n_owned, dtype=torch.float64, device=device)
    for j in range(world_size):
        for i in send_indices[j]:
            lent_count[i] += 1.0
    expected_owned = (1.0 + lent_count).unsqueeze(-1).expand(-1, feat)
    torch.testing.assert_close(
        ref_grad[:n_owned], expected_owned, rtol=1e-12, atol=1e-12
    )

    # Eager.
    pe = padded0.clone().requires_grad_(True)
    out_e = fn(pe, routing)
    (grad_e,) = torch.autograd.grad(out_e.sum(), pe)
    torch.testing.assert_close(out_e, ref_fwd, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad_e, ref_grad, rtol=1e-12, atol=1e-12)

    # Compiled.
    torch._dynamo.reset()
    pc = padded0.clone().requires_grad_(True)
    cf = torch.compile(fn, backend=backend, fullgraph=True)
    out_c = cf(pc, routing)
    (grad_c,) = torch.autograd.grad(out_c.sum(), pc)
    torch.testing.assert_close(out_c, ref_fwd, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad_c, ref_grad, rtol=1e-12, atol=1e-12)


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_halo_scatter_correct_1d(distributed_mesh, backend):
    if distributed_mesh.size() < 2:
        pytest.skip("halo correction needs >= 2 ranks")
    run_halo_scatter_correct(distributed_mesh, backend)


def _make_halo_shard_tensor(padded, mesh, routing):
    r"""A halo ShardTensor (local-honest ``Replicate``) carrying packed routing as an extra inner tensor."""
    from torch.distributed.tensor._dtensor_spec import TensorMeta
    from torch.distributed.tensor.placement_types import Replicate

    from physicsnemo.domain_parallel import ShardTensor
    from physicsnemo.domain_parallel.shard_tensor import ShardTensorSpec

    spec = ShardTensorSpec(
        mesh=mesh,
        placements=(Replicate(),) * mesh.ndim,
        tensor_meta=TensorMeta(
            shape=padded.shape, stride=padded.stride(), dtype=padded.dtype
        ),
        _local_shape=padded.shape,
        _sharding_shapes=None,
    )

    class _HaloShardTensor(ShardTensor):
        _extra_inner_tensors = ("_halo_meta_packed",)
        _halo_meta_packed_v = None
        _halo_meta_packed_c = None

        @property
        def _halo_meta_packed(self):
            v = self._halo_meta_packed_v
            return v if v is not None else self._stable_inner_sentinel("_halo_c")

        @_halo_meta_packed.setter
        def _halo_meta_packed(self, value):
            self._halo_meta_packed_v = value
            self._halo_meta_packed_c = None

    st = _HaloShardTensor.__new__(
        _HaloShardTensor,
        local_tensor=padded,
        spec=spec,
        requires_grad=padded.requires_grad,
    )
    st._halo_meta_packed = routing
    return st


def run_halo_shard_tensor_scatter_add(mesh, backend, n_owned=6, lend=2, feat=4):
    from physicsnemo.domain_parallel.shard_utils.halo_scatter import (
        register_halo_scatter_handlers,
    )

    register_halo_scatter_handlers()
    device = DistributedManager().device
    group = mesh.get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    n_owned, n_padded, send_indices, send_sizes = _ring_routing(
        rank, world_size, n_owned, lend
    )
    send_idx_t = [
        torch.tensor(s, dtype=torch.int64, device=device) for s in send_indices
    ]
    routing = pack_halo_routing(
        send_indices, send_sizes, n_owned, rank, world_size, device=device
    )
    idx = torch.arange(n_padded, device=device).unsqueeze(-1).expand(-1, feat)

    torch.manual_seed(100 + rank)
    src0 = torch.randn(n_padded, feat, dtype=torch.float64, device=device)

    # Reference: identity scatter into zeros, then the plain correction.
    def _plain_correct(p):
        return halo_forward_exchange(
            halo_reverse_exchange(
                p, n_owned, send_idx_t, send_sizes, rank, world_size, mesh
            ),
            send_idx_t,
            send_sizes,
            rank,
            world_size,
            mesh,
        )

    ref_fwd = _plain_correct(src0)
    ref_grad = _plain_correct(torch.ones_like(src0))
    assert not torch.allclose(ref_fwd, src0), "correction is a no-op here"

    def fn(agg, index, source):
        return agg.scatter_add(0, index, source)

    # Eager: the scatter_add handler applies the correction.
    src_e = src0.clone().requires_grad_(True)
    agg_e = _make_halo_shard_tensor(
        torch.zeros(n_padded, feat, dtype=torch.float64, device=device), mesh, routing
    )
    out_e = fn(agg_e, idx, src_e)
    (grad_e,) = torch.autograd.grad(out_e.to_local().sum(), src_e)
    torch.testing.assert_close(out_e._local_tensor, ref_fwd, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(grad_e, ref_grad, rtol=1e-9, atol=1e-9)

    # Compiled: correction survives as a differentiable node; source gradient comes back plain.
    torch._dynamo.reset()
    src_c = src0.clone().requires_grad_(True)
    agg_c = _make_halo_shard_tensor(
        torch.zeros(n_padded, feat, dtype=torch.float64, device=device), mesh, routing
    )
    cf = torch.compile(fn, backend=backend, fullgraph=True)
    out_c = cf(agg_c, idx, src_c)
    (grad_c,) = torch.autograd.grad(out_c.to_local().sum(), src_c)
    assert type(grad_c) is torch.Tensor, f"compiled grad is {type(grad_c)}"
    torch.testing.assert_close(out_c.to_local().detach(), ref_fwd, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(grad_c, ref_grad, rtol=1e-6, atol=1e-6)


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_halo_shard_tensor_scatter_add_1d(distributed_mesh, backend):
    if distributed_mesh.size() < 2:
        pytest.skip("halo correction needs >= 2 ranks")
    run_halo_shard_tensor_scatter_add(distributed_mesh, backend)


def _force_halo_backend(name):
    if name is None:
        os.environ.pop("PHYSICSNEMO_HALO_BACKEND", None)
    else:
        os.environ["PHYSICSNEMO_HALO_BACKEND"] = name


def _symm_mem_capable(mesh):
    r"""True when the symm-mem backend can serve this mesh (CUDA, >=2 ranks, rendezvous OK). Collective."""
    if not torch.cuda.is_available() or mesh.size() < 2:
        return False
    try:
        import torch.distributed._symmetric_memory as sm

        with torch.cuda.device(DistributedManager().device):
            sm.get_symm_mem_workspace(mesh.get_group().group_name, 1024)
        return True
    except Exception:
        return False


def run_symm_mem_equivalence(mesh, backend, n_owned=6, lend=2, feat=4):
    r"""The symm-mem transport must match the funcol oracle (fwd+bwd), eager and compiled."""
    device = DistributedManager().device
    group = mesh.get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    n_owned, n_padded, send_indices, send_sizes = _ring_routing(
        rank, world_size, n_owned, lend
    )
    routing = pack_halo_routing(
        send_indices, send_sizes, n_owned, rank, world_size, device=device
    )
    torch.manual_seed(100 + rank)
    padded0 = torch.randn(n_padded, feat, dtype=torch.float64, device=device)

    def fn(p, r):
        return halo_scatter_correct(p, r, group=mesh)

    # funcol oracle.
    _force_halo_backend("funcol")
    pf = padded0.clone().requires_grad_(True)
    ref_fwd = fn(pf, routing)
    (ref_grad,) = torch.autograd.grad(ref_fwd.sum(), pf)

    # symm-mem, eager: exact equality; repeated to surface any fence race.
    _force_halo_backend("symm_mem")
    try:
        for _ in range(50):
            ps = padded0.clone().requires_grad_(True)
            sm_fwd = fn(ps, routing)
            (sm_grad,) = torch.autograd.grad(sm_fwd.sum(), ps)
            torch.testing.assert_close(sm_fwd, ref_fwd, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(sm_grad, ref_grad, rtol=1e-12, atol=1e-12)
    finally:
        _force_halo_backend(None)

    # Auto-selection also picks symm-mem here.
    from physicsnemo.domain_parallel.shard_utils.halo_scatter import select_halo_backend

    assert select_halo_backend(mesh).name == "symm_mem"

    # symm-mem, compiled.
    _force_halo_backend("symm_mem")
    try:
        torch._dynamo.reset()
        pc = padded0.clone().requires_grad_(True)
        cf = torch.compile(fn, backend=backend, fullgraph=True)
        out_c = cf(pc, routing)
        (grad_c,) = torch.autograd.grad(out_c.sum(), pc)
    finally:
        _force_halo_backend(None)
    torch.testing.assert_close(out_c, ref_fwd, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad_c, ref_grad, rtol=1e-12, atol=1e-12)


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_halo_scatter_symm_mem_equivalence_1d(distributed_mesh, backend):
    if not _symm_mem_capable(distributed_mesh):
        pytest.skip("symmetric memory (>=2 P2P/NVSHMEM GPUs) not available")
    run_symm_mem_equivalence(distributed_mesh, backend)
