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

r"""Cross-product handlers on ShardTensor (``custom_ops/_tensor_ops.py``).

Locks in the ShardTensor-level cross support that older torch releases
cannot provide through DTensor sharding propagation (no strategy registered
for ``aten.linalg_cross``): replicated and Shard(0) inputs, the promoted
plain-tensor broadcast case, Partial resolution, sharded-dim rejection, and
gradient flow -- all value-checked against single-device references.
"""

import pytest
import torch
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, scatter_tensor

pytestmark = [pytest.mark.multigpu_static, pytest.mark.timeout(120)]

# Uneven on 2/4/8 ranks.
_N = 19


def _seeded_pair(device, shape=(_N, 3), seed=7):
    torch.manual_seed(seed)
    return torch.randn(*shape, device=device), torch.randn(*shape, device=device)


def test_cross_replicated(distributed_mesh):
    r"""Replicated x replicated (the exact CI failure configuration)."""
    dm = DistributedManager()
    a, b = _seeded_pair(dm.device, shape=(9, 3))
    a_s = ShardTensor.from_local(a, distributed_mesh, (Replicate(),))
    b_s = ShardTensor.from_local(b, distributed_mesh, (Replicate(),))

    result = torch.linalg.cross(a_s, b_s)

    assert isinstance(result, ShardTensor)
    torch.testing.assert_close(result.full_tensor(), torch.linalg.cross(a, b))


def test_cross_sharded(distributed_mesh):
    r"""Shard(0) x Shard(0) with an uneven split; output keeps the layout."""
    dm = DistributedManager()
    a, b = _seeded_pair(dm.device)
    a_s = scatter_tensor(a, 0, distributed_mesh, (Shard(0),))
    b_s = scatter_tensor(b, 0, distributed_mesh, (Shard(0),))

    result = torch.linalg.cross(a_s, b_s)

    assert isinstance(result, ShardTensor)
    assert result._spec.placements == (Shard(0),)
    assert result._spec.sharding_shapes()[0] == a_s._spec.sharding_shapes()[0]
    torch.testing.assert_close(result.full_tensor(), torch.linalg.cross(a, b))


def test_cross_sharded_with_plain_constant(distributed_mesh):
    r"""Shard(0) x plain (1, 3) constant: promotion + broadcast (the closed
    PR's crash case -- promoted plain tensors are DTensors, not ShardTensors).
    linalg.cross only broadcasts equal-rank inputs, so the constant is rank 2."""
    dm = DistributedManager()
    a, _ = _seeded_pair(dm.device)
    axis = torch.tensor([[0.0, 0.0, 1.0]], device=dm.device)
    a_s = scatter_tensor(a, 0, distributed_mesh, (Shard(0),))

    result = torch.linalg.cross(a_s, axis)

    assert isinstance(result, ShardTensor)
    assert result._spec.placements == (Shard(0),)
    torch.testing.assert_close(result.full_tensor(), torch.linalg.cross(a, axis))


def test_cross_rank_mismatch_matches_eager(distributed_mesh):
    r"""A rank-mismatched operand is rejected with eager torch's error:
    linalg.cross requires equal-rank inputs even on plain tensors."""
    dm = DistributedManager()
    a, _ = _seeded_pair(dm.device)
    axis = torch.tensor([0.0, 0.0, 1.0], device=dm.device)
    a_s = scatter_tensor(a, 0, distributed_mesh, (Shard(0),))

    with pytest.raises(RuntimeError, match="same number of dimensions"):
        torch.linalg.cross(a_s, axis)


def test_cross_resolves_partial(distributed_mesh):
    r"""A Partial operand is reduced before the local math: cross is
    bilinear, so a local cross of unreduced contributions would be wrong."""
    dm = DistributedManager()
    world_size = distributed_mesh.size(0)
    torch.manual_seed(11)
    contribution = torch.randn(9, 3, device=dm.device)
    # Every rank contributes the same tensor: the resolved value is
    # world_size * contribution.
    a_partial = ShardTensor.from_dtensor(
        DTensor.from_local(contribution, distributed_mesh, [Partial()], run_check=False)
    )
    b = torch.randn(9, 3, device=dm.device)
    b_s = ShardTensor.from_local(b, distributed_mesh, (Replicate(),))

    result = torch.linalg.cross(a_partial, b_s)

    expected = torch.linalg.cross(world_size * contribution, b)
    torch.testing.assert_close(result.full_tensor(), expected)


def test_cross_rejects_sharded_dim(distributed_mesh):
    r"""Cross along the sharded dimension is rejected. Built with
    ``from_local`` + explicit shard shapes so the raise is collective-free
    and identical on every rank (a half-posted collective wedges the suite).
    """
    dm = DistributedManager()
    ws = distributed_mesh.size(0)
    torch.manual_seed(13)
    m = 4 * ws
    # Chunk-style split of 3 rows over ws ranks, padded with empty shards.
    row_chunks = [c.numel() for c in torch.arange(3).chunk(ws)]
    row_chunks += [0] * (ws - len(row_chunks))
    shard_shapes = {0: [(rows, m) for rows in row_chunks]}

    local_a = torch.randn(row_chunks[dm.rank], m, device=dm.device)
    local_b = torch.randn(row_chunks[dm.rank], m, device=dm.device)
    a_s = ShardTensor.from_local(
        local_a, distributed_mesh, (Shard(0),), shard_shapes, global_shape=(3, m)
    )
    b_s = ShardTensor.from_local(
        local_b, distributed_mesh, (Shard(0),), shard_shapes, global_shape=(3, m)
    )

    with pytest.raises(RuntimeError, match="sharded dimension"):
        torch.linalg.cross(a_s, b_s, dim=0)


def test_cross_gradients(distributed_mesh):
    r"""Gradients through the sharded cross match the single-device run."""
    dm = DistributedManager()
    a, b = _seeded_pair(dm.device, seed=17)

    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    torch.linalg.cross(a_ref, b_ref).sum().backward()

    a_local = a.detach().clone().requires_grad_(True)
    b_local = b.detach().clone().requires_grad_(True)
    a_s = ShardTensor.from_local(a_local, distributed_mesh, (Replicate(),))
    b_s = ShardTensor.from_local(b_local, distributed_mesh, (Replicate(),))
    torch.linalg.cross(a_s, b_s).full_tensor().sum().backward()

    torch.testing.assert_close(a_local.grad, a_ref.grad)
    torch.testing.assert_close(b_local.grad, b_ref.grad)


def test_torch_cross_dim_none(distributed_mesh):
    r"""torch.cross with dim=None finds the size-3 dim in the GLOBAL shape."""
    dm = DistributedManager()
    n = 4 * distributed_mesh.size(0) + 1
    torch.manual_seed(19)
    a = torch.randn(n, 3, device=dm.device)
    b = torch.randn(n, 3, device=dm.device)
    a_s = scatter_tensor(a, 0, distributed_mesh, (Shard(0),))
    b_s = scatter_tensor(b, 0, distributed_mesh, (Shard(0),))

    result = torch.cross(a_s, b_s)

    assert isinstance(result, ShardTensor)
    torch.testing.assert_close(result.full_tensor(), torch.cross(a, b, dim=1))


def test_tensor_cross_method(distributed_mesh):
    r"""The Tensor.cross method routes through the same handler."""
    dm = DistributedManager()
    a, b = _seeded_pair(dm.device, seed=23)
    a_s = scatter_tensor(a, 0, distributed_mesh, (Shard(0),))
    b_s = scatter_tensor(b, 0, distributed_mesh, (Shard(0),))

    result = a_s.cross(b_s, dim=-1)

    assert isinstance(result, ShardTensor)
    torch.testing.assert_close(result.full_tensor(), a.cross(b, dim=-1))


# ---------------------------------------------------------------------------
# Regression tests: supported cross calls match native PyTorch in forward
# and backward.
# ---------------------------------------------------------------------------


def test_cross_replicated_operand_grad_is_reduced(distributed_mesh):
    r"""Primary reproducer: the replicated operand's gradient must be the
    sum of every rank's contribution."""
    dm = DistributedManager()
    ws = distributed_mesh.size(0)
    coord = dm.rank

    torch.manual_seed(29)
    full_a = torch.randn(2 * ws, 3, device=dm.device)

    local_a = full_a.chunk(ws, dim=0)[coord].clone().requires_grad_(True)
    sharded_a = ShardTensor.from_local(
        local_a,
        distributed_mesh,
        (Shard(0),),
        sharding_shapes="chunk",
        global_shape=tuple(full_a.shape),
    )
    axis = torch.tensor([[0.25, -0.5, 2.0]], device=dm.device, requires_grad=True)

    torch.linalg.cross(sharded_a, axis).full_tensor().sum().backward()

    ref_a = full_a.detach().clone().requires_grad_(True)
    ref_axis = axis.detach().clone().requires_grad_(True)
    torch.linalg.cross(ref_a, ref_axis).sum().backward()

    # Control: the sharded operand's local gradient is correct.
    torch.testing.assert_close(local_a.grad, ref_a.grad.chunk(ws, dim=0)[coord])
    # Bug: the replicated operand only sees its rank-local contribution.
    torch.testing.assert_close(axis.grad, ref_axis.grad)


def test_cross_sharded_with_full_size_replicate(distributed_mesh):
    r"""Shard(0) x full-size Replicate(): the replicated operand is
    localized to the reference shard."""
    dm = DistributedManager()
    n = 2 * distributed_mesh.size(0)
    torch.manual_seed(31)
    a = torch.randn(n, 3, device=dm.device)
    b = torch.randn(n, 3, device=dm.device)

    a_s = scatter_tensor(a, 0, distributed_mesh, (Shard(0),))
    b_s = ShardTensor.from_local(b, distributed_mesh, (Replicate(),))

    result = torch.linalg.cross(a_s, b_s)

    assert isinstance(result, ShardTensor)
    torch.testing.assert_close(result.full_tensor(), torch.linalg.cross(a, b))


def test_cross_replicated_shardtensor_with_sharded_dtensor(distributed_mesh):
    r"""Replicated ShardTensor x sharded DTensor works as a mixed pair."""
    dm = DistributedManager()
    n = 2 * distributed_mesh.size(0)
    torch.manual_seed(37)
    a = torch.randn(n, 3, device=dm.device)
    b = torch.randn(n, 3, device=dm.device)

    a_s = ShardTensor.from_local(a, distributed_mesh, (Replicate(),))
    b_d = distribute_tensor(b, distributed_mesh, [Shard(0)])

    result = torch.linalg.cross(a_s, b_d)

    torch.testing.assert_close(result.full_tensor(), torch.linalg.cross(a, b))


def test_cross_leaf_dtensor_gradient_not_dropped(distributed_mesh):
    r"""A leaf DTensor operand receives its gradient through backward.

    Converting a leaf (``grad_fn is None``) DTensor severs it from the
    graph: the gradient lands on the throwaway conversion wrapper and the
    user's ``.grad`` silently stays ``None``.
    """
    dm = DistributedManager()
    n = 2 * distributed_mesh.size(0)
    torch.manual_seed(47)
    a = torch.randn(n, 3, device=dm.device)
    w_full = torch.randn(n, 3, device=dm.device)

    a_s = ShardTensor.from_local(a, distributed_mesh, (Replicate(),))
    w = distribute_tensor(w_full, distributed_mesh, [Shard(0)]).requires_grad_(True)

    torch.linalg.cross(a_s, w).full_tensor().sum().backward()

    ref_w = w_full.detach().clone().requires_grad_(True)
    torch.linalg.cross(a, ref_w).sum().backward()

    assert w.grad is not None, "leaf DTensor gradient was dropped"
    torch.testing.assert_close(w.grad.full_tensor(), ref_w.grad)


def test_torch_cross_dim_none_uses_first_input_shape(distributed_mesh):
    r"""torch.cross(dim=None) resolves the dim from the first input's
    shape, matching native PyTorch (not the broadcast output shape)."""
    dm = DistributedManager()
    torch.manual_seed(41)
    a = torch.randn(1, 4, 3, device=dm.device)
    b = torch.randn(3, 4, 3, device=dm.device)

    a_s = ShardTensor.from_local(a, distributed_mesh, (Replicate(),))
    b_s = ShardTensor.from_local(b, distributed_mesh, (Replicate(),))

    result = torch.cross(a_s, b_s)

    torch.testing.assert_close(result.full_tensor(), torch.cross(a, b))


def test_cross_compile_with_plain_tensor(distributed_mesh):
    r"""The mixed plain-tensor/ShardTensor call survives torch.compile."""
    dm = DistributedManager()
    ws = distributed_mesh.size(0)
    n = 2 * ws
    torch.manual_seed(43)
    a = torch.randn(n, 3, device=dm.device)
    axis = torch.tensor([[0.25, -0.5, 2.0]], device=dm.device)

    # from_local "chunk" construction: communication-free, dynamo-friendly.
    local_a = a.chunk(ws, dim=0)[dm.rank].clone()
    a_s = ShardTensor.from_local(
        local_a,
        distributed_mesh,
        (Shard(0),),
        sharding_shapes="chunk",
        global_shape=(n, 3),
    )

    torch._dynamo.reset()

    def fn(x, y):
        return torch.linalg.cross(x, y)

    compiled = torch.compile(fn, fullgraph=True, backend="aot_eager")
    result = compiled(a_s, axis)

    torch.testing.assert_close(result.full_tensor(), torch.linalg.cross(a, axis))
