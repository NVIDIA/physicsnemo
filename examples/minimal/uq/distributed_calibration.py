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

"""Shared member statistics -> complete cases -> unchanged conformal calibration.

Run after Carmelo's conformal PR stack is available:
    torchrun --standalone --nproc-per-node=4 examples/minimal/uq/distributed_calibration.py

Analytic predictions check integration only. They do not demonstrate calibrated
coverage on real data. Each unique case is submitted once by the application.
"""

import tempfile
from pathlib import Path

import torch
from torch.distributed.tensor import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.experimental.uq.conformal import (
    ConformalPredictor,
    FunctionalBandCalibrator,
    NormalizedErrorScore,
    PointwiseCalibrator,
    RiskControlCalibrator,
)
from physicsnemo.metrics.general.ensemble_metrics import Mean, Variance


def complete(value):
    """Collect final summaries, not all member predictions, on the tensor mesh."""
    return value.redistribute(placements=[Replicate(), Replicate()]).to_local()


@torch.no_grad()
def main():
    """Compare distributed prediction inputs against a direct calibration reference."""
    DistributedManager.initialize()
    dm = DistributedManager()
    if dm.world_size != 4:
        raise ValueError("This example requires four GPUs")
    mesh = dm.initialize_mesh([2, 2], ["member", "space"])
    mi, si = mesh.get_coordinate()
    member_sizes, point_sizes = (3, 2), (3, 7)
    nm, nx = member_sizes[mi], point_sizes[si]
    member_start, point_start = sum(member_sizes[:mi]), sum(point_sizes[:si])
    shapes = {
        0: [(n, nx) for n in member_sizes],
        1: [(nm, n) for n in point_sizes],
    }
    calibrator_types = (
        PointwiseCalibrator,
        FunctionalBandCalibrator,
        RiskControlCalibrator,
    )
    if dm.rank == 0:
        distributed = [
            cls(NormalizedErrorScore(), alpha=0.1) for cls in calibrator_types
        ]
        reference = [cls(NormalizedErrorScore(), alpha=0.1) for cls in calibrator_types]

    # The complete arrays below are only independent references for this test.
    points = torch.arange(10, device=dm.device, dtype=torch.float64) / 10
    members = torch.arange(5, device=dm.device, dtype=torch.float64)[:, None]
    for case_id in range(20):
        values = 100 + torch.sin(points + case_id) + (members - 2) * (points + 1)
        variances = (0.2 + members * 0.1 + points * 0.05).expand_as(values)
        target = 100 + torch.sin(points + case_id) + torch.cos(points * 2 + case_id)
        between, within = Variance(mode="tensor", correction=0), Mean(mode="tensor")
        for source, accumulator in ((values, between), (variances, within)):
            field = ShardTensor.from_local(
                source[
                    member_start : member_start + nm, point_start : point_start + nx
                ].clone(),
                mesh,
                [Shard(0), Shard(1)],
                sharding_shapes=shapes,
            )
            accumulator.update(field, dim=0)
        mean = complete(between.mean)
        variance = complete(between.finalize() + within.finalize())
        # In a real workflow, gather the aligned targets as well. Only rank zero
        # submits a case: member replicas and domain shards are not extra cases.
        if dm.rank == 0:
            expected_mean = values.mean(0)
            expected_variance = values.var(0, correction=0) + variances.mean(0)
            for actual, expected in zip(distributed, reference):
                actual.update(mean, target, aux={"sigma": variance.sqrt()})
                expected.update(
                    expected_mean, target, aux={"sigma": expected_variance.sqrt()}
                )

    if dm.rank == 0:
        for actual, expected in zip(distributed, reference):
            fitted, baseline = actual.finalize(), expected.finalize()
            assert fitted.n_cal == 20
            torch.testing.assert_close(fitted.thresholds, baseline.thresholds)
            # Use the incoming predictor's existing persistence API, unchanged.
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "predictor.pt"
                fitted.save(path, provenance={"example": "distributed UQ", "cases": 20})
                restored = ConformalPredictor.load(path).to(dm.device)
                lower, upper = restored.predict_interval(
                    mean, aux={"sigma": variance.sqrt()}
                )
                expected_lower, expected_upper = baseline.predict_interval(
                    expected_mean,
                    aux={"sigma": expected_variance.sqrt()},
                )
                torch.testing.assert_close(lower, expected_lower)
                torch.testing.assert_close(upper, expected_upper)
        print(
            "All three unchanged conformal calibrators match direct inputs; exactly 20 unique cases"
        )
    torch.distributed.barrier()
    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
