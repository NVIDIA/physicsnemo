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

r"""Tensor-mode ``Mean`` / ``Variance`` on distributed inputs.

Inputs are ``(members, points, channels)``. Statistics reduce the member
dimension, so the tests shard the members, the points, or both, and check
two things against plain torch on the full tensor:

- the values are right, and
- dimensions that were not reduced stay sharded (nothing is gathered).

Each ``run_*`` helper takes a mesh and placements; the ``_1d`` tests use the
shared 1D mesh and the ``_2d`` tests the shared 2D mesh. Every rank builds
the same reference data, so every assertion passes or fails on all ranks.
"""

import pytest
import torch
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, scatter_tensor
from physicsnemo.metrics.general.ensemble_metrics import Mean, Variance
from physicsnemo.metrics.general.tensor_accumulator import TensorAccumulator

pytestmark = pytest.mark.multigpu_static

MEMBERS, POINTS, CHANNELS = 8, 12, 2
BOTH = (Shard(0), Shard(1))  # members on mesh axis 0, points on mesh axis 1


def _data(*shape):
    """Deterministic float64 data, identical on every rank."""
    n = torch.tensor(shape).prod().item()
    values = torch.arange(n, dtype=torch.float64, device=DistributedManager().device)
    return (1e3 + values.sin() + values / n).reshape(shape)


def _after_member_reduction(placements):
    """Expected placements once the member dimension (0) is reduced away."""
    return tuple(Replicate() if p.dim == 0 else Shard(p.dim - 1) for p in placements)


def _uneven(mesh, full, dim, sizes):
    """Shard ``full`` along ``dim`` of a 1D mesh with explicit per-rank sizes."""
    rank = mesh.get_local_rank()
    shape = list(full.shape)
    return ShardTensor.from_local(
        full.narrow(dim, sum(sizes[:rank]), sizes[rank]).clone(),
        mesh,
        (Shard(dim),),
        sharding_shapes={
            0: [tuple(shape[:dim] + [n] + shape[dim + 1 :]) for n in sizes]
        },
    )


# ---------------------------------------------------------------------------
# Batches of members: update(x) reduces dim 0
# ---------------------------------------------------------------------------


def run_stream_and_combine(mesh, placements):
    data = _data(4, MEMBERS, POINTS, CHANNELS)
    mean = Mean(mode="tensor")
    variance = Variance(mode="tensor", correction=0)
    tail = Variance(mode="tensor", correction=0)
    for step, batch in enumerate(data):
        x = scatter_tensor(batch, 0, mesh, placements)
        assert mean.update(x) is None
        (variance if step < 2 else tail).update(x)
    variance.combine(tail)

    members = data.flatten(0, 1)
    assert mean.count == variance.count == members.shape[0]
    for result, expected in [
        (mean.finalize(), members.mean(0)),
        (variance.mean, members.mean(0)),
        (variance.finalize(), members.var(0, correction=0)),
    ]:
        assert result.placements == _after_member_reduction(placements)
        torch.testing.assert_close(result.full_tensor(), expected)

    # A result is the caller's to modify; the accumulated state is not touched.
    mean.finalize().to_local().fill_(0.0)
    torch.testing.assert_close(mean.finalize().full_tensor(), members.mean(0))


@pytest.mark.parametrize("shard_dim", [0, 1], ids=["members", "points"])
def test_stream_and_combine_1d(distributed_mesh, shard_dim):
    run_stream_and_combine(distributed_mesh, (Shard(shard_dim),))


def test_stream_and_combine_2d(distributed_mesh_2d):
    run_stream_and_combine(distributed_mesh_2d, BOTH)


# ---------------------------------------------------------------------------
# One whole field per update: update(x, dim=None) reduces nothing
# ---------------------------------------------------------------------------


def run_whole_field_observations(mesh, placements):
    data = _data(5, MEMBERS, POINTS, CHANNELS)
    mean = Mean(mode="tensor")
    variance = Variance(mode="tensor", correction=0)
    for field in data:
        x = scatter_tensor(field, 0, mesh, placements)
        mean.update(x, dim=None)
        variance.update(x, dim=None)

    assert mean.count == variance.count == data.shape[0]
    for result, expected in [
        (mean.finalize(), data.mean(0)),
        (variance.finalize(), data.var(0, correction=0)),
    ]:
        # The result keeps the input's layout exactly.
        assert result.placements == x.placements
        assert result._spec.sharding_shapes() == x._spec.sharding_shapes()
        torch.testing.assert_close(result.full_tensor(), expected)


@pytest.mark.parametrize("shard_dim", [0, 1], ids=["members", "points"])
def test_whole_field_observations_1d(distributed_mesh, shard_dim):
    run_whole_field_observations(distributed_mesh, (Shard(shard_dim),))


def test_whole_field_observations_2d(distributed_mesh_2d):
    run_whole_field_observations(distributed_mesh_2d, BOTH)


# ---------------------------------------------------------------------------
# Inputs that arrive with a pending (Partial) sum
# ---------------------------------------------------------------------------


def run_pending_sum_input(mesh, placements):
    """A spatial sum is still pending when it reaches ``update``.

    The variance is nonlinear, so the accumulator must finish that sum first.
    """
    data = _data(3, MEMBERS, POINTS, CHANNELS)
    variance = Variance(mode="tensor", correction=0)
    for batch in data:
        x = scatter_tensor(batch, 0, mesh, placements)
        variance.update(x.sum(1) / POINTS)

    result = variance.finalize()
    assert not any(p.is_partial() for p in result.placements)
    expected = data.flatten(0, 1).mean(1).var(0, correction=0)
    torch.testing.assert_close(result.full_tensor(), expected)


@pytest.mark.parametrize("shard_dim", [0, 1], ids=["members", "points"])
def test_pending_sum_input_1d(distributed_mesh, shard_dim):
    run_pending_sum_input(distributed_mesh, (Shard(shard_dim),))


def test_pending_sum_input_2d(distributed_mesh_2d):
    run_pending_sum_input(distributed_mesh_2d, BOTH)


# ---------------------------------------------------------------------------
# A custom rule with an auxiliary input
# ---------------------------------------------------------------------------


def run_weighted_rule(mesh, placements):
    from test._weighted_mean_rule import WeightedMeanRule

    values = _data(3, MEMBERS, POINTS, CHANNELS)
    weights = values.remainder(5) + 1

    def scatter(full):
        return scatter_tensor(full, 0, mesh, placements)

    state = TensorAccumulator(WeightedMeanRule())
    tail = TensorAccumulator(WeightedMeanRule())
    for step in range(3):
        (state if step < 2 else tail).update(
            scatter(values[step]), weights=scatter(weights[step])
        )
    state.combine(tail)

    expected = (values * weights).sum((0, 1)) / weights.sum((0, 1))
    result = state.finalize()
    assert result.placements == _after_member_reduction(placements)
    torch.testing.assert_close(result.full_tensor(), expected)
    assert state.count == 3 * MEMBERS

    # One bad weight lives on one rank; every rank must reject the update and
    # keep its state, or the ranks would disagree from here on.
    bad = weights[0].clone()
    bad[0, 0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        state.update(scatter(values[0]), weights=scatter(bad))
    assert state.count == 3 * MEMBERS
    torch.testing.assert_close(state.finalize().full_tensor(), expected)

    # Same for a result that is undefined at a single point.
    zero = weights[0].clone()
    zero[:, 0, :] = 0
    undefined = TensorAccumulator(WeightedMeanRule())
    undefined.update(scatter(values[0]), weights=scatter(zero))
    with pytest.raises(ValueError, match="positive finite"):
        undefined.finalize()


@pytest.mark.parametrize("shard_dim", [0, 1], ids=["members", "points"])
def test_weighted_rule_1d(distributed_mesh, shard_dim):
    run_weighted_rule(distributed_mesh, (Shard(shard_dim),))


def test_weighted_rule_2d(distributed_mesh_2d):
    run_weighted_rule(distributed_mesh_2d, BOTH)


# ---------------------------------------------------------------------------
# Uneven and empty shards (hand-built: scatter_tensor only makes even chunks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shard_dim", [0, 1], ids=["members", "points"])
@pytest.mark.parametrize("layout", ["uneven", "empty_rank"])
def test_uneven_and_empty_shards(distributed_mesh, shard_dim, layout):
    mesh = distributed_mesh
    ranks = mesh.size(0)
    sizes = [3] + [1] * (ranks - 1) if layout == "uneven" else [0] + [2] * (ranks - 1)
    shape = [2, MEMBERS, POINTS, CHANNELS]
    shape[shard_dim + 1] = sum(sizes)
    data = _data(*shape)

    mean = Mean(mode="tensor")
    variance = Variance(mode="tensor", correction=0)
    for batch in data:
        x = _uneven(mesh, batch, shard_dim, sizes)
        mean.update(x)
        variance.update(x)

    members = data.flatten(0, 1)
    assert mean.count == variance.count == members.shape[0]
    for result, expected in [
        (mean.finalize(), members.mean(0)),
        (variance.finalize(), members.var(0, correction=0)),
    ]:
        assert result.placements == _after_member_reduction((Shard(shard_dim),))
        if shard_dim == 1:
            # The uneven point boundaries survive the member reduction.
            shapes = result._spec.sharding_shapes()[0]
            assert [tuple(s) for s in shapes] == [(n, CHANNELS) for n in sizes]
        torch.testing.assert_close(result.full_tensor(), expected)


def test_empty_batch_adds_no_observations(distributed_mesh):
    """A batch with zero members sets the layout but is not an observation."""
    sizes = [3] + [1] * (distributed_mesh.size(0) - 1)
    variance = Variance(mode="tensor", correction=0)
    empty = _data(0, sum(sizes), CHANNELS)
    assert variance.update(_uneven(distributed_mesh, empty, 1, sizes)) is None
    assert variance.count == 0
    with pytest.raises(ValueError, match="empty"):
        _ = variance.mean
    with pytest.raises(ValueError, match="Insufficient"):
        variance.finalize()

    batch = _data(MEMBERS, sum(sizes), CHANNELS)
    variance.update(_uneven(distributed_mesh, batch, 1, sizes))
    assert variance.count == MEMBERS
    torch.testing.assert_close(
        variance.finalize().full_tensor(), batch.var(0, correction=0)
    )


def test_misaligned_shards_are_rejected(distributed_mesh):
    """Same global shape and placements, different rows per rank: must raise."""
    ranks = distributed_mesh.size(0)
    batch = _data(MEMBERS, ranks + 1, CHANNELS)
    first_heavy = [2] + [1] * (ranks - 1)
    last_heavy = [1] * (ranks - 1) + [2]

    variance = Variance(mode="tensor", correction=0)
    variance.update(_uneven(distributed_mesh, batch, 1, first_heavy))
    other = Variance(mode="tensor", correction=0)
    other.update(_uneven(distributed_mesh, batch, 1, last_heavy))

    with pytest.raises(ValueError, match="shard boundaries"):
        variance.update(_uneven(distributed_mesh, batch, 1, last_heavy))
    with pytest.raises(ValueError, match="shard boundaries"):
        variance.combine(other)
    assert variance.count == MEMBERS
    torch.testing.assert_close(
        variance.finalize().full_tensor(), batch.var(0, correction=0)
    )


# ---------------------------------------------------------------------------
# Other input types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shard_dim", [0, 1], ids=["members", "points"])
def test_dtensor_inputs(distributed_mesh, shard_dim):
    """Plain DTensors (even chunks only) follow the same rules."""
    data = _data(2, MEMBERS, POINTS, CHANNELS)
    variance = Variance(mode="tensor", correction=0)
    for batch in data:
        variance.update(distribute_tensor(batch, distributed_mesh, [Shard(shard_dim)]))

    members = data.flatten(0, 1)
    for result, expected in [
        (variance.mean, members.mean(0)),
        (variance.finalize(), members.var(0, correction=0)),
    ]:
        assert result.placements == _after_member_reduction((Shard(shard_dim),))
        torch.testing.assert_close(result.full_tensor(), expected)


def test_ordinary_tensors_stay_local():
    """Tensor mode never communicates for ordinary tensors: ranks differ."""
    dm = DistributedManager()
    variance = Variance(mode="tensor", correction=0)
    variance.update(torch.tensor([[dm.rank], [dm.rank + 2.0]], device=dm.device))
    torch.testing.assert_close(
        variance.mean, torch.tensor([dm.rank + 1.0], device=dm.device)
    )
    torch.testing.assert_close(variance.finalize(), torch.ones(1, device=dm.device))


def test_legacy_mode_still_reduces_over_world():
    """The default constructors keep combining ordinary tensors across ranks."""
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
