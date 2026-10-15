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

"""
Tests for ring communication primitives (``perform_ring_iteration`` and
``perform_ring_iteration_funcol``).

These are direct unit tests for the ring.py module, verifying that data
arrives at the correct rank with the correct values for both blocking
and async variants.

Run with:
    torchrun --nproc-per-node 2 -m pytest --multigpu-static \
        test/domain_parallel/ops/test_ring.py
"""

import pytest
import torch
import torch.distributed as dist

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel.shard_utils.ring import (
    RingPassingConfig,
    finish_ring_iteration,
    perform_ring_iteration,
    perform_ring_iteration_funcol,
)

from .utils import collective_assert, collective_assert_close

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rank_tensor(shape, dtype, device, rank):
    """Create a tensor filled with (rank + 1) so every rank's data is distinct."""
    return torch.full(shape, float(rank + 1), dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Blocking single-step primitives: raw dist.* and functional collectives
# ---------------------------------------------------------------------------

# ``perform_ring_iteration`` honors ``communication_method``; the funcol variant
# is always an all-to-all, so it is exercised once per direction/dtype.
_RING_FNS = [
    pytest.param(perform_ring_iteration, "p2p", id="dist-p2p"),
    pytest.param(perform_ring_iteration, "a2a", id="dist-a2a"),
    pytest.param(perform_ring_iteration_funcol, "a2a", id="funcol"),
]


def _config(local_size, direction="forward", comm_method="a2a"):
    return RingPassingConfig(
        mesh_dim=0,
        mesh_size=local_size,
        ring_direction=direction,
        communication_method=comm_method,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("ring_fn,comm_method", _RING_FNS)
@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ring_iteration_single_step(
    distributed_mesh, ring_fn, comm_method, direction, dtype
):
    """One ring step: every rank sends its tensor and receives from its neighbor."""
    dm = DistributedManager()
    mesh = distributed_mesh
    local_rank = mesh.get_local_rank(0)
    local_size = dist.get_world_size(group=mesh.get_group(0))

    shape = (4, 8)
    tensor = _make_rank_tensor(shape, dtype, dm.device, local_rank)

    received = ring_fn(tensor, mesh, _config(local_size, direction, comm_method))

    # In "forward" mode, rank r receives from rank r-1 (wrapping).
    # In "backward" mode, rank r receives from rank r+1 (wrapping).
    if direction == "forward":
        expected_source = (local_rank - 1) % local_size
    else:
        expected_source = (local_rank + 1) % local_size

    expected = _make_rank_tensor(shape, dtype, dm.device, expected_source)
    collective_assert_close(
        received,
        expected,
        atol=0,
        rtol=0,
        msg=f"ring_iteration single step ({ring_fn.__name__}, {comm_method}, {direction})",
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("ring_fn,comm_method", _RING_FNS)
def test_ring_full_rotation(distributed_mesh, ring_fn, comm_method):
    """N ring steps should return the original tensor back to each rank."""
    dm = DistributedManager()
    mesh = distributed_mesh
    local_rank = mesh.get_local_rank(0)
    local_size = dist.get_world_size(group=mesh.get_group(0))

    shape = (3, 5)
    original = _make_rank_tensor(shape, torch.float32, dm.device, local_rank)
    current = original.clone()
    config = _config(local_size, comm_method=comm_method)

    for _ in range(local_size):
        current = ring_fn(current, mesh, config)

    collective_assert_close(
        current,
        original,
        atol=0,
        rtol=0,
        msg=f"ring full rotation ({ring_fn.__name__}, {comm_method})",
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("ring_fn,comm_method", _RING_FNS)
def test_ring_iteration_uneven_shapes(distributed_mesh, ring_fn, comm_method):
    """recv_shape != send shape (uneven shards): shape and values are the neighbor's."""
    dm = DistributedManager()
    mesh = distributed_mesh
    local_rank = mesh.get_local_rank(0)
    local_size = dist.get_world_size(group=mesh.get_group(0))

    # Each rank has a different number of rows; fill with the rank id so the
    # received values identify their source.
    n_cols = 4
    tensor = _make_rank_tensor(
        (10 + local_rank * 3, n_cols), torch.float32, dm.device, local_rank
    )

    # Compute the shape we expect to receive (from rank r-1)
    source_rank = (local_rank - 1) % local_size
    recv_shape = torch.Size([10 + source_rank * 3, n_cols])

    received = ring_fn(
        tensor,
        mesh,
        _config(local_size, comm_method=comm_method),
        recv_shape=recv_shape,
    )

    collective_assert(
        received.shape == recv_shape,
        msg=f"uneven shape mismatch: got {received.shape}, expected {recv_shape}",
    )
    expected = _make_rank_tensor(recv_shape, torch.float32, dm.device, source_rank)
    collective_assert_close(
        received, expected, atol=0, rtol=0, msg="uneven-shape values"
    )


# ---------------------------------------------------------------------------
# Deferred wait: the overlap form used by ring attention
# ---------------------------------------------------------------------------


@pytest.mark.multigpu_static
def test_ring_iteration_funcol_deferred_wait(distributed_mesh):
    """wait=False + finish_ring_iteration: the overlap form used by ring SDPA."""
    dm = DistributedManager()
    mesh = distributed_mesh
    local_rank = mesh.get_local_rank(0)
    local_size = dist.get_world_size(group=mesh.get_group(0))

    shape = (6, 7)
    tensor = _make_rank_tensor(shape, torch.float32, dm.device, local_rank)

    in_flight = perform_ring_iteration_funcol(
        tensor, mesh, _config(local_size), wait=False
    )
    # Unrelated compute in the overlap window.
    _ = tensor.square().sum()
    received = finish_ring_iteration(in_flight, tensor.shape)

    expected_source = (local_rank - 1) % local_size
    expected = _make_rank_tensor(shape, torch.float32, dm.device, expected_source)
    collective_assert_close(
        received,
        expected,
        atol=0,
        rtol=0,
        msg="funcol deferred-wait ring step",
    )
