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

r"""``unbind`` handlers on ShardTensor (``custom_ops/_tensor_ops.py``).

Locks in the Partial-resolution semantics: unbinding an unreduced partial
sum must first resolve the pending reduction (differentiable allreduce to
``Replicate``), matching the SDPA/cross treatment. Slice attention in
physics_attention unbinds a Partial qkv projection under domain parallelism.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Partial, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, scatter_tensor

pytestmark = [pytest.mark.multigpu_static, pytest.mark.timeout(120)]


def test_unbind_partial_resolves_reduction(distributed_mesh):
    r"""``torch.unbind`` on a Partial(sum) tensor equals unbind of the summed
    global, with Replicate outputs."""
    dm = DistributedManager()
    world = distributed_mesh.size()
    torch.manual_seed(5)
    # Rank-distinct local contributions; the global value is their sum.
    contributions = [torch.randn(4, 3, 6) for _ in range(world)]
    local = contributions[distributed_mesh.get_local_rank()].to(dm.device)
    expected = torch.stack(contributions).sum(0).to(dm.device)

    st = ShardTensor.from_local(
        local,
        distributed_mesh,
        (Partial(),),
        sharding_shapes="chunk",
        global_shape=local.shape,
    )

    pieces = torch.unbind(st, 1)

    assert len(pieces) == 3
    for i, piece in enumerate(pieces):
        assert isinstance(piece, ShardTensor)
        assert all(not p.is_partial() for p in piece._spec.placements)
        torch.testing.assert_close(
            piece.full_tensor(), expected.select(1, i), atol=1e-5, rtol=1e-5
        )


def test_unbind_sharded_gradients(distributed_mesh):
    r"""Unbind along a non-sharded dim of a Shard(0) tensor: layout preserved
    and gradients flow back through the from_local/to_local bridge."""
    dm = DistributedManager()
    torch.manual_seed(6)
    full = torch.randn(19, 3, 4, device=dm.device)  # uneven on 2/4/8 ranks
    st = scatter_tensor(full, 0, distributed_mesh, (Shard(0),), requires_grad=True)

    pieces = torch.unbind(st, 1)

    assert len(pieces) == 3
    for piece in pieces:
        assert piece._spec.placements == (Shard(0),)
    torch.testing.assert_close(pieces[1].full_tensor(), full.select(1, 1))

    pieces[1].full_tensor().square().mean().backward()
    ref = full.clone().requires_grad_(True)
    ref.select(1, 1).square().mean().backward()
    assert st.grad is not None
    assert st.grad._spec.placements == (Shard(0),)
    torch.testing.assert_close(st.grad.full_tensor(), ref.grad)
