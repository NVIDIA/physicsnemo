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

r"""Tests for :func:`physicsnemo.domain_parallel.sync_module_over_mesh`.

The sync must: make every plain parameter AND buffer bitwise identical
across the mesh group (sourced from the mesh's rank 0), leave distributed
(DTensor) parameters untouched, reject multi-dim meshes, and pass its own
``verify`` check.  The 2-D mesh test exercises a submesh whose group rank 0
is NOT global rank 0, which would catch group/global rank confusion in the
broadcast source.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import (
    ShardTensor,
    sync_module_over_mesh,
)


class SmallModel(nn.Module):
    """Linear weights plus BatchNorm so both params and buffers are covered."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.bn(self.fc1(x)))


def _desync(model: nn.Module, rank: int) -> None:
    """Give every rank distinct weights, buffers, and BN step counts."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(float(rank + 1))
        for b in model.buffers():
            b.add_(rank + 1)


def _assert_synced_to_mesh_src(model: nn.Module, mesh) -> None:
    """Every plain tensor must equal the copy held by the mesh's rank 0."""
    group = mesh.get_group()
    src_global = dist.get_global_rank(group, 0)
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        reference = t.detach().clone()
        dist.broadcast(reference, src=src_global, group=group)
        assert torch.equal(t.detach(), reference), (
            f"{name} differs from mesh src after sync"
        )


@pytest.mark.multigpu_static
def test_sync_plain_params_and_buffers(distributed_mesh):
    dm = DistributedManager()
    model = SmallModel().to(dm.device)
    _desync(model, dm.rank)

    sync_module_over_mesh(model, distributed_mesh)

    _assert_synced_to_mesh_src(model, distributed_mesh)


@pytest.mark.multigpu_static
def test_sync_from_nonzero_mesh_rank(distributed_mesh):
    dm = DistributedManager()
    model = SmallModel().to(dm.device)
    _desync(model, dm.rank)
    src_mesh_rank = distributed_mesh.size(0) - 1

    sync_module_over_mesh(model, distributed_mesh, src_mesh_rank=src_mesh_rank)

    group = distributed_mesh.get_group()
    src_global = dist.get_global_rank(group, src_mesh_rank)
    for tensor in list(model.parameters()) + list(model.buffers()):
        reference = tensor.detach().clone()
        dist.broadcast(reference, src=src_global, group=group)
        assert torch.equal(tensor.detach(), reference)


@pytest.mark.multigpu_static
def test_sync_rejects_invalid_source_rank(distributed_mesh):
    dm = DistributedManager()
    model = SmallModel().to(dm.device)

    with pytest.raises(ValueError, match="src_mesh_rank"):
        sync_module_over_mesh(model, distributed_mesh, src_mesh_rank=-1)
    with pytest.raises(ValueError, match="src_mesh_rank"):
        sync_module_over_mesh(
            model, distributed_mesh, src_mesh_rank=distributed_mesh.size(0)
        )


@pytest.mark.multigpu_static
def test_sync_skips_dtensor_params(distributed_mesh):
    dm = DistributedManager()
    model = SmallModel().to(dm.device)
    _desync(model, dm.rank)

    # Convert one parameter to a DTensor sharded over the mesh; its local
    # shard must be untouched by the sync.
    model.fc1.weight = nn.Parameter(
        distribute_tensor(model.fc1.weight.data, distributed_mesh, [Shard(0)])
    )
    local_before = model.fc1.weight.to_local().clone()

    sync_module_over_mesh(model, distributed_mesh)

    assert torch.equal(model.fc1.weight.to_local(), local_before), (
        "DTensor parameter's local shard was modified by sync"
    )
    # The plain remainder must still be synced.
    reference = model.fc2.weight.detach().clone()
    src_global = dist.get_global_rank(distributed_mesh.get_group(), 0)
    dist.broadcast(reference, src=src_global, group=distributed_mesh.get_group())
    assert torch.equal(model.fc2.weight.detach(), reference)


@pytest.mark.multigpu_static
def test_sync_rejects_all_distributed_state(distributed_mesh):
    dm = DistributedManager()
    model = nn.Linear(4, 4, bias=False).to(dm.device)
    model.weight = nn.Parameter(
        distribute_tensor(model.weight.detach(), distributed_mesh, [Shard(0)])
    )

    with pytest.raises(RuntimeError, match="none of it is plain"):
        sync_module_over_mesh(model, distributed_mesh)


@pytest.mark.multigpu_static
def test_sync_skips_shard_tensor_params(distributed_mesh):
    dm = DistributedManager()
    model = SmallModel().to(dm.device)
    _desync(model, dm.rank)

    local = (
        model.fc1.weight.detach()
        .chunk(distributed_mesh.size(0), dim=0)[distributed_mesh.get_local_rank()]
        .contiguous()
    )
    sharded = ShardTensor.from_local(
        local,
        distributed_mesh,
        [Shard(0)],
        sharding_shapes="chunk",
        global_shape=tuple(model.fc1.weight.shape),
    )
    model.fc1.weight = nn.Parameter(sharded)
    local_before = model.fc1.weight.to_local().clone()

    sync_module_over_mesh(model, distributed_mesh)

    assert torch.equal(model.fc1.weight.to_local(), local_before)
    reference = model.fc2.weight.detach().clone()
    src_global = dist.get_global_rank(distributed_mesh.get_group(), 0)
    dist.broadcast(reference, src=src_global, group=distributed_mesh.get_group())
    assert torch.equal(model.fc2.weight.detach(), reference)


@pytest.mark.multigpu_static
def test_sync_verify_passes(distributed_mesh):
    dm = DistributedManager()
    model = SmallModel().to(dm.device)
    _desync(model, dm.rank)

    # verify=True must not raise on a correct sync.
    sync_module_over_mesh(model, distributed_mesh, verify=True)
    _assert_synced_to_mesh_src(model, distributed_mesh)


@pytest.mark.multigpu_static
def test_sync_verify_rejects_metadata_mismatch(distributed_mesh):
    dm = DistributedManager()
    model = SmallModel().to(dm.device)
    if distributed_mesh.get_local_rank() == 0:
        model.extra = nn.Parameter(torch.ones(1, device=dm.device))

    with pytest.raises(RuntimeError, match="metadata differs"):
        sync_module_over_mesh(model, distributed_mesh, verify=True)


@pytest.mark.multigpu_static
def test_sync_public_broadcast_fallback(distributed_mesh, monkeypatch):
    dm = DistributedManager()
    model = SmallModel().to(dm.device)
    _desync(model, dm.rank)
    monkeypatch.delattr(dist, "_broadcast_coalesced", raising=False)

    sync_module_over_mesh(model, distributed_mesh)

    _assert_synced_to_mesh_src(model, distributed_mesh)


@pytest.mark.multigpu_static
def test_sync_empty_module(distributed_mesh):
    sync_module_over_mesh(nn.Identity(), distributed_mesh, verify=True)


@pytest.mark.multigpu_static
def test_sync_rejects_multidim_mesh(distributed_mesh_2d):
    dm = DistributedManager()
    model = SmallModel().to(dm.device)

    with pytest.raises(ValueError, match="1-D mesh"):
        sync_module_over_mesh(model, distributed_mesh_2d)


@pytest.mark.multigpu_static
def test_sync_over_submesh(distributed_mesh_2d):
    """Sync over one axis of a 2-D mesh.

    The second-axis group containing the highest global ranks has a group
    rank 0 that is NOT global rank 0 -- this catches group-vs-global rank
    confusion in the broadcast source.  Ranks in different groups must NOT
    be synced with each other.
    """
    dm = DistributedManager()
    model = SmallModel().to(dm.device)
    _desync(model, dm.rank)

    submesh = distributed_mesh_2d["axis2"]
    sync_module_over_mesh(model, submesh, verify=True)

    _assert_synced_to_mesh_src(model, submesh)

    # Cross-group isolation: the fc2 bias was seeded with rank-dependent
    # values, so different groups' sources differ; compare across the OTHER
    # mesh axis and require a mismatch.
    other_axis_group = distributed_mesh_2d["axis1"].get_group()
    other_src = dist.get_global_rank(other_axis_group, 0)
    cross = model.fc2.bias.detach().clone()
    dist.broadcast(cross, src=other_src, group=other_axis_group)
    if dist.get_rank(other_axis_group) != 0 and other_src != dist.get_global_rank(
        submesh.get_group(), 0
    ):
        assert not torch.equal(model.fc2.bias.detach(), cross), (
            "ranks in different domain groups should not have been synced"
        )
