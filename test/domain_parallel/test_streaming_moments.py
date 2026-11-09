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
from torch.distributed.tensor import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.metrics.general.ensemble_metrics import Mean, Variance


@pytest.fixture(params=[0, 1], ids=["members-axis0", "members-axis1"])
def moment_mesh(distributed_mesh_2d, request):
    """Reuse the session mesh, exercising each statistical axis on two GPUs.

    The shared mesh is 2 x 1 in CI and 2 x 2 on four GPUs. Swapping the
    member/space roles tests both communication paths without new process groups.
    """
    return distributed_mesh_2d, request.param, 1 - request.param


def _partition_sizes(sizes, count):
    """Coalesce a singleton axis; keep uneven/empty partitions on distributed axes."""
    return (sum(sizes),) if count == 1 else sizes + (0,) * (count - 2)


def _chunk_sizes(total, count):
    """Match DTensor's regular chunk boundaries, including trailing empty ranks."""
    width = (total + count - 1) // count
    return tuple(min(width, max(0, total - rank * width)) for rank in range(count))


def _placements(member_axis, member, space):
    """Map logical member/space placements onto the shared mesh axes."""
    return (member, space) if member_axis == 0 else (space, member)


@pytest.mark.multigpu_static
@pytest.mark.parametrize("simulation_sizes", [(2, 2), (3, 2), (1, 0)])
def test_simulations_time_space(moment_mesh, simulation_sizes):
    dm = DistributedManager()
    mesh, member_axis, space_axis = moment_mesh
    si, di = (mesh.get_coordinate()[axis] for axis in (member_axis, space_axis))
    simulation_sizes = _partition_sizes(simulation_sizes, mesh.size(member_axis))
    space_sizes = _partition_sizes((3, 7), mesh.size(space_axis))
    kl, dl = simulation_sizes[si], space_sizes[di]
    ks, ds = sum(simulation_sizes[:si]), sum(space_sizes[:di])
    k, d = sum(simulation_sizes), sum(space_sizes)
    shapes = {
        member_axis: [(n, dl, 2) for n in simulation_sizes],
        space_axis: [(kl, n, 2) for n in space_sizes],
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
            _placements(member_axis, Shard(0), Shard(1)),
            sharding_shapes=shapes,
        )
        between.update(x)
        temporal.update(x, dim=None)
        (first if t < 2 else second).update(x)
        within.update(x * 0 + 2.0)
        # RMS over simulations and space, one result for each time/channel.
        rms = (x.square().sum((0, 1)) / (k * d)).sqrt()
        # sqrt must resolve pending sums before applying its nonlinear operation.
        assert all(placement.is_replicate() for placement in rms.placements)
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
    assert between.finalize().placements == _placements(
        member_axis, Replicate(), Shard(0)
    )
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
def test_field_gp_mixture_uses_both_moments(moment_mesh):
    """Actual field GP outputs compose over member and uneven spatial partitions."""
    pytest.importorskip("gpytorch")
    from physicsnemo.experimental.uq.gp import FieldVariationalGPHead

    dm = DistributedManager()
    mesh, member_axis, space_axis = moment_mesh
    mi, si = (mesh.get_coordinate()[axis] for axis in (member_axis, space_axis))
    member_sizes = _partition_sizes((2, 1), mesh.size(member_axis))
    point_sizes = _partition_sizes((3, 7), mesh.size(space_axis))
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
        member_axis: [(n, nx, 2) for n in member_sizes],
        space_axis: [(nm, n, 2) for n in point_sizes],
    }
    between, within = Variance(mode="tensor", correction=0), Mean(mode="tensor")
    between.update(
        ShardTensor.from_local(
            torch.stack(local_means)
            if local_means
            else full.mean.new_empty((0, nx, 2)),
            mesh,
            _placements(member_axis, Shard(0), Shard(1)),
            sharding_shapes=shapes,
        )
    )
    within.update(
        ShardTensor.from_local(
            torch.stack(local_variances)
            if local_variances
            else full.variance.new_empty((0, nx, 2)),
            mesh,
            _placements(member_axis, Shard(0), Shard(1)),
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
    assert result.placements == _placements(member_axis, Replicate(), Shard(0))
    assert between.count == within.count == 3


@pytest.mark.multigpu_static
@pytest.mark.parametrize("point_sizes", [(0, 10), (3, 7)])
def test_empty_batches_and_partition_alignment(moment_mesh, point_sizes):
    """Global emptiness and zero-size spatial partitions preserve count/layout semantics."""
    dm = DistributedManager()
    mesh, member_axis, space_axis = moment_mesh
    mi, si = (mesh.get_coordinate()[axis] for axis in (member_axis, space_axis))
    point_sizes = _partition_sizes(point_sizes, mesh.size(space_axis))
    member_sizes = _partition_sizes((1, 1), mesh.size(member_axis))
    nm, member_start = member_sizes[mi], sum(member_sizes[:mi])
    nx, start = point_sizes[si], sum(point_sizes[:si])

    def field(values, members, points):
        """Construct the logical batch with explicit nonuniform shard boundaries."""
        return ShardTensor.from_local(
            values,
            mesh,
            _placements(member_axis, Shard(0), Shard(1)),
            sharding_shapes={
                member_axis: [(n, points[si]) for n in members],
                space_axis: [(members[mi], n) for n in points],
            },
        )

    stats = Variance(mode="tensor", correction=0)
    empty = field(
        torch.empty(0, nx, device=dm.device), (0,) * mesh.size(member_axis), point_sizes
    )
    assert stats.update(empty) is None
    assert stats.count == 0
    with pytest.raises(ValueError, match="empty"):
        _ = stats.mean
    with pytest.raises(ValueError, match="Insufficient"):
        stats.finalize()
    complete = torch.arange(20, device=dm.device, dtype=torch.float32).reshape(2, 10)
    stats.update(
        field(
            complete[member_start : member_start + nm, start : start + nx],
            member_sizes,
            point_sizes,
        )
    )
    torch.testing.assert_close(
        stats.mean.to_local(), complete.mean(0)[start : start + nx]
    )
    torch.testing.assert_close(
        stats.finalize().to_local(), complete.var(0, correction=0)[start : start + nx]
    )
    assert stats.count == 2
    # Identical global shapes/placements do not establish aligned output points.
    other_points = _partition_sizes((4, 6), mesh.size(space_axis))
    other_start = sum(other_points[:si])
    other = Variance(mode="tensor", correction=0)
    other.update(
        field(
            complete[
                member_start : member_start + nm,
                other_start : other_start + other_points[si],
            ],
            member_sizes,
            other_points,
        )
    )
    if mesh.size(space_axis) > 1:
        with pytest.raises(ValueError, match="shard boundaries"):
            stats.combine(other)
        assert stats.count == 2
    else:
        # Coalesced spatial partitions have identical boundaries and can combine.
        stats.combine(other)
        assert stats.count == 4
        torch.testing.assert_close(
            stats.finalize().full_tensor(), complete.var(0, correction=0)
        )


@pytest.mark.multigpu_static
def test_streamed_unequal_member_groups(moment_mesh):
    """One member per update, including lockstep padding for an exhausted group."""
    dm = DistributedManager()
    mesh, member_axis, space_axis = moment_mesh
    mi, si = (mesh.get_coordinate()[axis] for axis in (member_axis, space_axis))
    member_counts = _partition_sizes((3, 2), mesh.size(member_axis))
    space_sizes = _partition_sizes((3, 7), mesh.size(space_axis))
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
            _placements(member_axis, Shard(0), Shard(1)),
            sharding_shapes={
                member_axis: [(n, width, 2) for n in counts],
                space_axis: [(counts[mi], n, 2) for n in space_sizes],
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
        == _placements(member_axis, Replicate(), Shard(0))
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize(
    "simulation_sizes,space_sizes",
    [((3, 2), (3, 7)), ((1, 0), (3, 7)), ((3, 2), (0, 10))],
)
def test_weighted_rule_uses_shared_accumulator(
    moment_mesh, simulation_sizes, space_sizes
):
    """An external rule preserves uneven spatial shards and handles empty ranks."""
    from physicsnemo.metrics.general.tensor_accumulator import TensorAccumulator
    from test._weighted_mean_rule import WeightedMeanRule

    dm = DistributedManager()
    mesh, member_axis, space_axis = moment_mesh
    si, di = (mesh.get_coordinate()[axis] for axis in (member_axis, space_axis))
    simulation_sizes = _partition_sizes(simulation_sizes, mesh.size(member_axis))
    space_sizes = _partition_sizes(space_sizes, mesh.size(space_axis))
    kl, dl = simulation_sizes[si], space_sizes[di]
    ks, ds = sum(simulation_sizes[:si]), sum(space_sizes[:di])
    k, d = sum(simulation_sizes), sum(space_sizes)
    shapes = {
        member_axis: [(n, dl, 2) for n in simulation_sizes],
        space_axis: [(kl, n, 2) for n in space_sizes],
    }

    def shard(values):
        return ShardTensor.from_local(
            values[ks : ks + kl, ds : ds + dl].clone(),
            mesh,
            _placements(member_axis, Shard(0), Shard(1)),
            sharding_shapes=shapes,
        )

    state = TensorAccumulator(WeightedMeanRule())
    tail = TensorAccumulator(WeightedMeanRule())
    weighted_sums, weight_sums = [], []
    for time in range(3):
        values = (
            torch.arange(k * d * 2, device=dm.device, dtype=torch.float64).reshape(
                k, d, 2
            )
            / 7
            + time
        )
        weights = (values.remainder(5) + time + 1) / 3
        (state if time < 2 else tail).update(shard(values), weights=shard(weights))
        weighted_sums.append((values * weights).sum(0))
        weight_sums.append(weights.sum(0))
    state.combine(tail)
    result = state.finalize()
    expected = torch.stack(weighted_sums).sum(0) / torch.stack(weight_sums).sum(0)
    torch.testing.assert_close(result.to_local(), expected[ds : ds + dl])
    assert result.placements == _placements(member_axis, Replicate(), Shard(0))
    assert result._spec.sharding_shapes() == tail.finalize()._spec.sharding_shapes()
    assert state.count == 3 * k
    # A negative weight present on just one spatial/member partition is rejected
    # consistently by every participant, including ranks with no local elements.
    invalid_weights = weights.clone()
    invalid_weights[0, 0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        state.update(shard(values), weights=shard(invalid_weights))
    assert state.count == 3 * k
    torch.testing.assert_close(state.finalize().to_local(), expected[ds : ds + dl])
    # A zero-weight output on one spatial shard is also rejected collectively.
    zero_weight = weights.clone()
    zero_weight[:, 0, :] = 0
    bad_result = TensorAccumulator(WeightedMeanRule())
    bad_result.update(shard(values), weights=shard(zero_weight))
    with pytest.raises(ValueError, match="positive finite"):
        bad_result.finalize()


@pytest.mark.multigpu_static
@pytest.mark.parametrize("tensor_kind", ["shard", "dtensor"])
@pytest.mark.parametrize("empty_space", [False, True])
def test_accumulator_result_ownership(
    moment_mesh, monkeypatch, tensor_kind, empty_space
):
    """Read-time ownership works on distributed state, including empty shards."""
    from torch.distributed.tensor import DTensor

    from physicsnemo.metrics.general.moment_rules import VarianceRule
    from physicsnemo.metrics.general.tensor_accumulator import TensorAccumulator

    dm = DistributedManager()
    mesh, member_axis, space_axis = moment_mesh
    si, di = (mesh.get_coordinate()[axis] for axis in (member_axis, space_axis))
    simulation_sizes = _partition_sizes((3, 2), mesh.size(member_axis))
    if tensor_kind == "shard":
        space_sizes = _partition_sizes(
            (0, 10) if empty_space else (3, 7), mesh.size(space_axis)
        )
    else:
        simulation_sizes = _chunk_sizes(5, mesh.size(member_axis))
        space_sizes = _chunk_sizes(1 if empty_space else 10, mesh.size(space_axis))
    kl, dl = simulation_sizes[si], space_sizes[di]
    ks, ds = sum(simulation_sizes[:si]), sum(space_sizes[:di])
    k, d = sum(simulation_sizes), sum(space_sizes)
    values = torch.arange(k * d * 2, device=dm.device, dtype=torch.float64).reshape(
        k, d, 2
    )
    local = values[ks : ks + kl, ds : ds + dl].clone().contiguous()
    if tensor_kind == "shard":
        field = ShardTensor.from_local(
            local,
            mesh,
            _placements(member_axis, Shard(0), Shard(1)),
            sharding_shapes={
                member_axis: [(n, dl, 2) for n in simulation_sizes],
                space_axis: [(kl, n, 2) for n in space_sizes],
            },
        )
    else:
        field = DTensor.from_local(
            local,
            mesh,
            _placements(member_axis, Shard(0), Shard(1)),
            shape=values.shape,
            stride=values.stride(),
        )
    original = VarianceRule.result
    returned = []

    def capture(self, state, **options):
        result = original(self, state, **options)
        returned.append(result)
        return result

    monkeypatch.setattr(VarianceRule, "result", capture)
    state = TensorAccumulator(VarianceRule())
    state.update(field)
    for statistic, expected in [("mean", values.mean(0)), ("variance", values.var(0))]:
        result = state.finalize(statistic=statistic)
        assert (result is returned[-1]) == (statistic == "variance")
        assert result.placements == _placements(member_axis, Replicate(), Shard(0))
        torch.testing.assert_close(result.to_local(), expected[ds : ds + dl])
        result.to_local().fill_(-999)
        torch.testing.assert_close(
            state.finalize(statistic=statistic).to_local(), expected[ds : ds + dl]
        )
