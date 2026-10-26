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

"""Simulations x time x space with direct and compound streaming statistics.

Run: torchrun --standalone --nproc-per-node=4 examples/minimal/uq/distributed_moments.py
The simulated field is analytic so results can be checked independently.
"""

import torch
from torch.distributed.tensor import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.metrics.general.ensemble_metrics import Mean, Variance


def main():
    """Preserve spatial partitions while reducing distinct statistical axes."""
    DistributedManager.initialize()
    dm = DistributedManager()
    if dm.world_size != 4:
        raise ValueError("Run with four GPUs for a 2 x 2 mesh")
    mesh = dm.initialize_mesh([2, 2], ["simulation", "space"])
    simulation_rank, space_rank = mesh.get_coordinate()
    simulation_sizes, space_sizes = [3, 2], [3, 7]
    ns, nx = simulation_sizes[simulation_rank], space_sizes[space_rank]
    simulation_start = sum(simulation_sizes[:simulation_rank])
    space_start = sum(space_sizes[:space_rank])
    simulation_ids = torch.arange(
        simulation_start, simulation_start + ns, device=dm.device
    )
    space_ids = torch.arange(space_start, space_start + nx, device=dm.device)
    shapes = {0: [(n, nx) for n in simulation_sizes], 1: [(ns, n) for n in space_sizes]}
    direct = Variance(mode="tensor", correction=0)
    per_simulation = Variance(mode="tensor", correction=0)
    observable = Variance(mode="tensor", correction=0)
    steps = 8
    for time in range(steps):
        local_field = simulation_ids[:, None] + torch.sin(
            space_ids[None, :] * 0.3 + time
        )
        field = ShardTensor.from_local(
            local_field, mesh, [Shard(0), Shard(1)], sharding_shapes=shapes
        )
        # A batch of simulations at this time: communicate over simulations.
        direct.update(field, dim=0)
        # One time observation per simulation/point: no axes are reduced here.
        per_simulation.update(field, dim=None)
        # Reduce space BEFORE computing the variance of this observable.
        spatial_mean = field.sum(1) / sum(space_sizes)
        observable.update(spatial_mean, dim=0)  # Completes pending spatial sum.
        # RMS across simulations and space at each time.
        rms = (
            field.square().sum().redistribute(placements=[Replicate(), Replicate()])
            / (sum(simulation_sizes) * sum(space_sizes))
        ).sqrt()
        if dm.rank == 0:
            print("time", time, "RMS", rms.to_local().item())

    # Compound statistics: equal time counts permit the law of total variance.
    # Only this stage reduces across the simulation mesh axis.
    means = Variance(mode="tensor", correction=0)
    within = Mean(mode="tensor")
    means.update(per_simulation.mean, dim=0)
    within.update(per_simulation.finalize(), dim=0)
    compound_variance = means.finalize() + within.finalize()
    torch.testing.assert_close(
        compound_variance.to_local(), direct.finalize().to_local()
    )
    torch.testing.assert_close(means.mean.to_local(), direct.mean.to_local())
    if dm.rank == 0:
        print("field variance layout:", direct.finalize().placements)
        print("variance of spatial mean:", observable.finalize().to_local())
        print(
            "Direct and compound results agree; neither uses local() or synchronize()."
        )
    # Stream one member at a time; every rank must participate in every update.
    # An exhausted member group supplies zero rows rather than skipping a call.
    streamed = Variance(mode="tensor", correction=0)
    for step in range(max(simulation_sizes)):
        counts = [int(step < n) for n in simulation_sizes]
        local = (
            (simulation_start + step + torch.sin(space_ids * 0.3)).unsqueeze(0)
            if counts[simulation_rank]
            else torch.empty(0, nx, device=dm.device)
        )
        field = ShardTensor.from_local(
            local,
            mesh,
            [Shard(0), Shard(1)],
            sharding_shapes={
                0: [(n, nx) for n in counts],
                1: [(counts[simulation_rank], n) for n in space_sizes],
            },
        )
        streamed.update(field)
    assert streamed.count == sum(simulation_sizes)
    # Each spatial value differs across members only by the integer member ID.
    expected = torch.arange(
        sum(simulation_sizes), device=dm.device, dtype=torch.float32
    ).var(correction=0)
    torch.testing.assert_close(streamed.finalize().to_local(), expected.expand(nx))
    if dm.rank == 0:
        print("Streamed unequal groups agree with the five-member reference.")
    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
