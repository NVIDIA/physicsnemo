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

"""Gradient equivalence of the recipe's training step under domain parallelism.

``train.py`` syncs the model over the domain mesh and then wraps it in DDP over
the flat world, relying on every parameter gradient arriving at DDP as the
complete domain-group value (ShardTensor resolves the sharded-axis reductions
before ``.grad`` accumulates). This test runs the recipe's ``forward_pass`` +
``_reduce_and_average`` on one sample sharded over the whole world and checks
that the loss and every parameter gradient equal the unsharded single-process
result. Skipped unless launched with more than one process::

    torchrun --nproc-per-node 2 -m pytest tests/test_domain_parallel_gradients.py
"""

from __future__ import annotations

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

pytest.importorskip("tensorboard")

from collate import build_collate_fn  # noqa: E402
from loss import LossCalculator  # noqa: E402
from metrics import MetricCalculator  # noqa: E402
from train import _reduce_and_average, forward_pass  # noqa: E402

from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.domain_parallel import (  # noqa: E402
    scatter_tensor,
    sync_module_over_mesh,
)
from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402

pytestmark = [pytest.mark.timeout(180)]


@pytest.fixture(scope="module")
def distributed_mesh():
    """1-D mesh over every launched process; skips outside a multi-process launch."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    DistributedManager.initialize()
    dm = DistributedManager()
    if dm.world_size < 2:
        pytest.skip("launch with torchrun --nproc-per-node >= 2")
    yield dm.initialize_mesh([-1], ["domain"])


_N_POINTS = 1001  # uneven on 2/4/8 ranks
_TARGETS = {"pressure": "scalar", "wss": "vector"}


class _PointMLP(torch.nn.Module):
    """Point-wise model: coordinates -> 4 outputs (pressure + wss)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 32), torch.nn.GELU(), torch.nn.Linear(32, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sample(device) -> DomainMesh:
    torch.manual_seed(11)
    points = torch.randn(_N_POINTS, 3, device=device)
    interior = Mesh(
        points=points,
        point_data={
            "pressure": torch.randn(_N_POINTS, device=device),
            "wss": torch.randn(_N_POINTS, 3, device=device),
        },
    )
    return DomainMesh(interior=interior, boundaries={}, global_data={})


def _shard_sample(domain: DomainMesh, device_mesh) -> DomainMesh:
    """The same sample with interior points and point_data as Shard(0)."""
    interior = domain.interior
    shard = lambda t: scatter_tensor(t, 0, device_mesh, (Shard(0),))  # noqa: E731
    sharded = Mesh(
        points=shard(interior.points),
        point_data={k: shard(v) for k, v in interior.point_data.items()},
    )
    return DomainMesh(interior=sharded, boundaries={}, global_data={})


def _step(model, domain, dist_manager):
    """One recipe training step on *domain*: reduced loss and parameter grads."""
    collate = build_collate_fn("tensors", {"x": "interior.points"}, _TARGETS)
    batch = collate([(domain, {})])
    loss_calc = LossCalculator(target_config=_TARGETS, loss_type="mse")
    metric_calc = MetricCalculator(target_config=_TARGETS, metrics=["l2"])
    model.zero_grad(set_to_none=True)
    loss, losses, metrics = forward_pass(
        batch,
        model,
        "float32",
        loss_calc,
        metric_calc,
        output_type="tensors",
        target_config=_TARGETS,
    )
    loss.backward()
    avg_loss, _, _ = _reduce_and_average(
        loss.detach(), losses, metrics, 1, device=dist_manager.device
    )
    grads = [p.grad.detach().clone() for p in model.parameters()]
    return avg_loss, grads


def test_domain_parallel_step_matches_single_process(distributed_mesh):
    """Sharding the sample over the whole world (domain_size == world_size)
    reproduces the unsharded loss and parameter gradients through the recipe's
    sync -> DDP -> forward_pass -> reduce path."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    device = dm.device
    domain = _sample(device)

    # Reference: plain single-process step on the full sample.
    torch.manual_seed(3)
    reference = _PointMLP().to(device)
    ref_loss, ref_grads = _step(reference, domain, dm)

    # Domain-parallel: same weights on every rank, DDP over the flat world
    # (the recipe's exact wiring), sample sharded over the domain mesh.
    torch.manual_seed(3)
    model = _PointMLP().to(device)
    sync_module_over_mesh(model, distributed_mesh)
    ddp = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[dm.local_rank], output_device=device
    )
    dp_loss, dp_grads = _step(ddp, _shard_sample(domain, distributed_mesh), dm)

    assert abs(dp_loss - ref_loss) < 1e-5 * max(1.0, abs(ref_loss))
    for ref, got in zip(ref_grads, dp_grads):
        assert type(got) is torch.Tensor, "DDP parameter received a distributed grad"
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)
