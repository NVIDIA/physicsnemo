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

"""Simulations x time x space acceptance of the public statistics classes."""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.metrics.general.ensemble_metrics import Mean, Variance


@pytest.mark.multigpu_static
@pytest.mark.parametrize("simulation_sizes", [(2, 2), (3, 2), (1, 0)])
def test_simulations_time_space(distributed_mesh, simulation_sizes):
    dm = DistributedManager()
    if dm.world_size != 4:
        pytest.skip("Acceptance topology requires four GPUs")
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("simulation", "space"))
    si, di = mesh.get_coordinate()
    space_sizes = (3, 7)
    kl, dl = simulation_sizes[si], space_sizes[di]
    ks, ds = sum(simulation_sizes[:si]), sum(space_sizes[:di])
    k, d = sum(simulation_sizes), sum(space_sizes)
    shapes = {
        0: [(n, dl, 2) for n in simulation_sizes],
        1: [(kl, n, 2) for n in space_sizes],
    }
    between = Variance(mode="tensor", correction=0)
    within = Mean(mode="tensor")
    first = Variance(mode="tensor", correction=0)
    second = Variance(mode="tensor", correction=0)
    observable = Variance(mode="tensor", correction=0)
    temporal = Variance(mode="tensor", correction=0)
    references = []
    # Logical data are replicated here only to construct independent references.
    for t in range(5):
        values = torch.arange(k * d * 2, device=dm.device, dtype=torch.float64).reshape(
            k, d, 2
        )
        values = 1e6 + values / 11 + t * torch.sin(values)
        references.append(values)
        x = ShardTensor.from_local(
            values[ks : ks + kl, ds : ds + dl].clone(),
            mesh,
            [Shard(0), Shard(1)],
            sharding_shapes=shapes,
        )
        between.update(x)
        temporal.update(x, dim=None)
        (first if t < 2 else second).update(x)
        within.update(x * 0 + 2.0)
        # RMS over simulations and space, one result for each time/channel.
        rms = (
            x.square().sum((0, 1)).redistribute(placements=[Replicate(), Replicate()])
            / (k * d)
        ).sqrt()
        torch.testing.assert_close(rms.to_local(), values.square().mean((0, 1)).sqrt())
        # Pending spatial sums must complete before a nonlinear moment update.
        observable.update(x.sum(1) / d)
    reference = torch.stack(references)
    expected = reference.var((0, 1), correction=0)[ds : ds + dl]
    torch.testing.assert_close(between.finalize().to_local(), expected)
    torch.testing.assert_close(
        between.mean.to_local(), reference.mean((0, 1))[ds : ds + dl]
    )
    torch.testing.assert_close(
        (within.finalize() + between.finalize()).to_local(), expected + 2.0
    )
    pooled_temporal_means = Variance(mode="tensor", correction=0)
    averaged_temporal_variances = Mean(mode="tensor")
    pooled_temporal_means.update(temporal.mean)
    averaged_temporal_variances.update(temporal.finalize())
    total_temporal = (
        pooled_temporal_means.finalize() + averaged_temporal_variances.finalize()
    )
    torch.testing.assert_close(total_temporal.to_local(), expected)
    assert between.count == 5 * k
    assert between.finalize().placements == (Replicate(), Shard(0))
    first.combine(second)
    torch.testing.assert_close(first.finalize().to_local(), expected)
    torch.testing.assert_close(
        observable.finalize().full_tensor(), reference.mean(2).var((0, 1), correction=0)
    )
    # Ordinary tensors stay local even when DistributedManager is initialized.
    local = Variance(mode="tensor", correction=0)
    local.update(torch.tensor([[dm.rank], [dm.rank + 2.0]], device=dm.device))
    torch.testing.assert_close(
        local.mean, torch.tensor([dm.rank + 1.0], device=dm.device)
    )
    torch.testing.assert_close(local.finalize(), torch.ones(1, device=dm.device))


@pytest.mark.multigpu_static
def test_legacy_world_reductions_remain_opt_out(distributed_mesh):
    """Existing constructors still combine ordinary tensors across WORLD."""
    dm = DistributedManager()
    value = torch.tensor([[dm.rank], [dm.rank + 2.0]], device=dm.device)
    reference = torch.tensor(
        [[rank + offset] for rank in range(dm.world_size) for offset in (0.0, 2.0)],
        device=dm.device,
    )
    mean = Mean((1,), device=dm.device)
    variance = Variance((1,), device=dm.device)
    torch.testing.assert_close(mean(value), reference.mean(0))
    torch.testing.assert_close(variance(value), reference.var(0))
    mean.update(value)
    variance.update(value)
    torch.testing.assert_close(mean.finalize(), reference.mean(0))
    torch.testing.assert_close(variance.finalize(), reference.repeat(2, 1).var(0))


@pytest.mark.multigpu_static
def test_field_gp_mixture_uses_both_moments(distributed_mesh):
    """Actual field GP outputs compose over member and uneven spatial partitions."""
    pytest.importorskip("gpytorch")
    from physicsnemo.experimental.uq.gp import FieldVariationalGPHead

    dm = DistributedManager()
    if dm.world_size != 4:
        pytest.skip("Acceptance topology requires four GPUs")
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("member", "space"))
    mi, si = mesh.get_coordinate()
    member_sizes, point_sizes = (2, 1), (3, 7)
    member_start, point_start = sum(member_sizes[:mi]), sum(point_sizes[:si])
    nm, nx = member_sizes[mi], point_sizes[si]
    features = (
        torch.arange(40, dtype=torch.float64, device=dm.device).reshape(10, 4) / 40
    )
    full_means, full_variances, local_means, local_variances = [], [], [], []
    with torch.no_grad():
        for index in range(3):
            torch.manual_seed(71 + index)
            head = (
                FieldVariationalGPHead(
                    input_dim=4,
                    num_tasks=2,
                    n_train=100,
                    n_inducing=4,
                    noise_mlp_hidden=[4],
                )
                .to(dm.device)
                .eval()
            )
            # Initialize each head consistently, then evaluate a spatial subset.
            full = head.predict(features)
            full_means.append(full.mean)
            full_variances.append(full.variance)
            if member_start <= index < member_start + nm:
                local = head.predict(features[point_start : point_start + nx])
                torch.testing.assert_close(
                    local.mean, full.mean[point_start : point_start + nx]
                )
                torch.testing.assert_close(
                    local.variance, full.variance[point_start : point_start + nx]
                )
                local_means.append(local.mean)
                local_variances.append(local.variance)
    shapes = {
        0: [(n, nx, 2) for n in member_sizes],
        1: [(nm, n, 2) for n in point_sizes],
    }
    between, within = Variance(mode="tensor", correction=0), Mean(mode="tensor")
    between.update(
        ShardTensor.from_local(
            torch.stack(local_means),
            mesh,
            [Shard(0), Shard(1)],
            sharding_shapes=shapes,
        )
    )
    within.update(
        ShardTensor.from_local(
            torch.stack(local_variances),
            mesh,
            [Shard(0), Shard(1)],
            sharding_shapes=shapes,
        )
    )
    means, variances = torch.stack(full_means), torch.stack(full_variances)
    expected = means.var(0, correction=0) + variances.mean(0)
    result = between.finalize() + within.finalize()
    torch.testing.assert_close(
        result.to_local(), expected[point_start : point_start + nx]
    )
    torch.testing.assert_close(
        between.mean.to_local(), means.mean(0)[point_start : point_start + nx]
    )
    assert result.placements == (Replicate(), Shard(0))
    assert between.count == within.count == 3


@pytest.mark.multigpu_static
@pytest.mark.parametrize("point_sizes", [(0, 10), (3, 7)])
def test_empty_batches_and_partition_alignment(distributed_mesh, point_sizes):
    """Global emptiness and zero-size spatial partitions preserve count/layout semantics."""
    dm = DistributedManager()
    if dm.world_size != 4:
        pytest.skip("Acceptance topology requires four GPUs")
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("member", "space"))
    mi, si = mesh.get_coordinate()
    nx, start = point_sizes[si], sum(point_sizes[:si])

    def field(values, members, points):
        """Construct the logical batch with explicit nonuniform shard boundaries."""
        return ShardTensor.from_local(
            values,
            mesh,
            [Shard(0), Shard(1)],
            sharding_shapes={
                0: [(n, points[si]) for n in members],
                1: [(members[mi], n) for n in points],
            },
        )

    stats = Variance(mode="tensor", correction=0)
    empty = field(torch.empty(0, nx, device=dm.device), (0, 0), point_sizes)
    assert stats.update(empty) is None
    assert stats.count == 0
    with pytest.raises(ValueError, match="empty"):
        _ = stats.mean
    with pytest.raises(ValueError, match="Insufficient"):
        stats.finalize()
    complete = torch.arange(20, device=dm.device, dtype=torch.float32).reshape(2, 10)
    stats.update(field(complete[mi : mi + 1, start : start + nx], (1, 1), point_sizes))
    torch.testing.assert_close(
        stats.mean.to_local(), complete.mean(0)[start : start + nx]
    )
    torch.testing.assert_close(
        stats.finalize().to_local(), complete.var(0, correction=0)[start : start + nx]
    )
    assert stats.count == 2
    # Identical global shapes/placements do not establish aligned output points.
    other_points = (4, 6)
    other_start = sum(other_points[:si])
    other = Variance(mode="tensor", correction=0)
    other.update(
        field(
            complete[mi : mi + 1, other_start : other_start + other_points[si]],
            (1, 1),
            other_points,
        )
    )
    with pytest.raises(ValueError, match="shard boundaries"):
        stats.combine(other)
    assert stats.count == 2


@pytest.mark.multigpu_static
def test_streamed_unequal_member_groups(distributed_mesh):
    """One member per update, including lockstep padding for an exhausted group."""
    dm = DistributedManager()
    if dm.world_size != 4:
        pytest.skip("Acceptance topology requires four GPUs")
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("member", "space"))
    mi, si = mesh.get_coordinate()
    member_counts, space_sizes = (3, 2), (3, 7)
    start, width = sum(space_sizes[:si]), space_sizes[si]
    offset = sum(member_counts[:mi])
    reference = (
        torch.arange(100, device=dm.device, dtype=torch.float64).reshape(5, 10, 2).sin()
    )
    mean, variance = Mean(mode="tensor"), Variance(mode="tensor", correction=0)
    for step in range(max(member_counts)):
        counts = [int(step < n) for n in member_counts]
        local = (
            reference[offset + step : offset + step + 1, start : start + width]
            if counts[mi]
            else reference.new_empty((0, width, 2))
        )
        field = ShardTensor.from_local(
            local.clone(),
            mesh,
            [Shard(0), Shard(1)],
            sharding_shapes={
                0: [(n, width, 2) for n in counts],
                1: [(counts[mi], n, 2) for n in space_sizes],
            },
        )
        assert mean.update(field) is None
        assert variance.update(field) is None
    torch.testing.assert_close(
        mean.finalize().to_local(), reference.mean(0)[start : start + width]
    )
    torch.testing.assert_close(
        variance.finalize().to_local(),
        reference.var(0, correction=0)[start : start + width],
    )
    assert mean.count == variance.count == 5
    assert (
        mean.finalize().placements
        == variance.finalize().placements
        == (Replicate(), Shard(0))
    )
