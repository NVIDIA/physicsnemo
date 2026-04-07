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
Tests for ring communication primitives (perform_ring_iteration and
perform_ring_iteration_async).

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
    perform_ring_iteration,
    perform_ring_iteration_async,
)

from .utils import collective_assert, collective_assert_close

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rank_tensor(shape, dtype, device, rank):
    """Create a tensor filled with (rank + 1) so every rank's data is distinct."""
    return torch.full(shape, float(rank + 1), dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Tests for blocking perform_ring_iteration
# ---------------------------------------------------------------------------


@pytest.mark.multigpu_static
@pytest.mark.parametrize("comm_method", ["p2p", "a2a"])
@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ring_iteration_single_step(distributed_mesh, comm_method, direction, dtype):
    """One ring step: every rank sends its tensor and receives from its neighbor."""
    dm = DistributedManager()
    mesh = distributed_mesh
    local_group = mesh.get_group(0)
    local_rank = mesh.get_local_rank(0)
    local_size = dist.get_world_size(group=local_group)

    shape = (4, 8)
    tensor = _make_rank_tensor(shape, dtype, dm.device, local_rank)

    config = RingPassingConfig(
        mesh_dim=0,
        mesh_size=local_size,
        ring_direction=direction,
        communication_method=comm_method,
    )

    received = perform_ring_iteration(tensor, mesh, config)

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
        msg=f"ring_iteration single step ({comm_method}, {direction})",
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("comm_method", ["p2p", "a2a"])
def test_ring_full_rotation(distributed_mesh, comm_method):
    """N ring steps should return the original tensor back to each rank."""
    dm = DistributedManager()
    mesh = distributed_mesh
    local_group = mesh.get_group(0)
    local_rank = mesh.get_local_rank(0)
    local_size = dist.get_world_size(group=local_group)

    shape = (3, 5)
    original = _make_rank_tensor(shape, torch.float32, dm.device, local_rank)
    current = original.clone()

    config = RingPassingConfig(
        mesh_dim=0,
        mesh_size=local_size,
        ring_direction="forward",
        communication_method=comm_method,
    )

    for _ in range(local_size):
        current = perform_ring_iteration(current, mesh, config)

    collective_assert_close(
        current,
        original,
        atol=0,
        rtol=0,
        msg=f"ring full rotation ({comm_method})",
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("comm_method", ["p2p", "a2a"])
def test_ring_iteration_uneven_shapes(distributed_mesh, comm_method):
    """Test with recv_shape != send shape (uneven shards across ranks)."""
    dm = DistributedManager()
    mesh = distributed_mesh
    local_group = mesh.get_group(0)
    local_rank = mesh.get_local_rank(0)
    local_size = dist.get_world_size(group=local_group)

    # Each rank has a different number of rows
    n_rows = 10 + local_rank * 3
    n_cols = 4
    tensor = torch.randn(n_rows, n_cols, dtype=torch.float32, device=dm.device)

    config = RingPassingConfig(
        mesh_dim=0,
        mesh_size=local_size,
        ring_direction="forward",
        communication_method=comm_method,
    )

    # Compute the shape we expect to receive (from rank r-1)
    source_rank = (local_rank - 1) % local_size
    recv_n_rows = 10 + source_rank * 3
    recv_shape = torch.Size([recv_n_rows, n_cols])

    received = perform_ring_iteration(tensor, mesh, config, recv_shape=recv_shape)

    collective_assert(
        received.shape == recv_shape,
        msg=f"uneven shape mismatch: got {received.shape}, expected {recv_shape}",
    )


# ---------------------------------------------------------------------------
# Tests for async perform_ring_iteration_async
# ---------------------------------------------------------------------------


@pytest.mark.multigpu_static
@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ring_iteration_async_single_step(distributed_mesh, direction, dtype):
    """Async variant: one ring step with explicit wait."""
    dm = DistributedManager()
    mesh = distributed_mesh
    local_group = mesh.get_group(0)
    local_rank = mesh.get_local_rank(0)
    local_size = dist.get_world_size(group=local_group)

    shape = (4, 8)
    tensor = _make_rank_tensor(shape, dtype, dm.device, local_rank)

    config = RingPassingConfig(
        mesh_dim=0,
        mesh_size=local_size,
        ring_direction=direction,
        communication_method="p2p",
    )

    recv_tensor, work_handles = perform_ring_iteration_async(
        tensor,
        mesh,
        config,
    )

    # Wait for completion
    for w in work_handles:
        w.wait()

    if direction == "forward":
        expected_source = (local_rank - 1) % local_size
    else:
        expected_source = (local_rank + 1) % local_size

    expected = _make_rank_tensor(shape, dtype, dm.device, expected_source)
    collective_assert_close(
        recv_tensor,
        expected,
        atol=0,
        rtol=0,
        msg=f"async ring_iteration single step ({direction})",
    )


@pytest.mark.multigpu_static
def test_ring_iteration_async_preallocated_buffer(distributed_mesh):
    """Async variant with a caller-supplied recv buffer."""
    dm = DistributedManager()
    mesh = distributed_mesh
    local_group = mesh.get_group(0)
    local_rank = mesh.get_local_rank(0)
    local_size = dist.get_world_size(group=local_group)

    shape = (6, 4)
    tensor = _make_rank_tensor(shape, torch.float32, dm.device, local_rank)

    # Pre-allocate buffer
    recv_buf = torch.empty(shape, dtype=torch.float32, device=dm.device)

    config = RingPassingConfig(
        mesh_dim=0,
        mesh_size=local_size,
        ring_direction="forward",
        communication_method="p2p",
    )

    returned_recv, work_handles = perform_ring_iteration_async(
        tensor,
        mesh,
        config,
        recv_tensor=recv_buf,
    )

    for w in work_handles:
        w.wait()

    # The returned tensor should be the same object as our pre-allocated buffer
    collective_assert(
        returned_recv.data_ptr() == recv_buf.data_ptr(),
        msg="async recv should reuse the pre-allocated buffer",
    )

    expected_source = (local_rank - 1) % local_size
    expected = _make_rank_tensor(shape, torch.float32, dm.device, expected_source)
    collective_assert_close(
        recv_buf,
        expected,
        atol=0,
        rtol=0,
        msg="async preallocated buffer contents",
    )


@pytest.mark.multigpu_static
def test_ring_iteration_async_full_rotation(distributed_mesh):
    """N async ring steps should return the original tensor."""
    dm = DistributedManager()
    mesh = distributed_mesh
    local_group = mesh.get_group(0)
    local_rank = mesh.get_local_rank(0)
    local_size = dist.get_world_size(group=local_group)

    shape = (3, 5)
    original = _make_rank_tensor(shape, torch.float32, dm.device, local_rank)
    current = original.clone()

    config = RingPassingConfig(
        mesh_dim=0,
        mesh_size=local_size,
        ring_direction="forward",
        communication_method="p2p",
    )

    for _ in range(local_size):
        recv, handles = perform_ring_iteration_async(current, mesh, config)
        for w in handles:
            w.wait()
        current = recv

    collective_assert_close(
        current,
        original,
        atol=0,
        rtol=0,
        msg="async ring full rotation",
    )


@pytest.mark.multigpu_static
def test_ring_iteration_async_rejects_a2a(distributed_mesh):
    """Async variant should raise ValueError for a2a communication."""
    dm = DistributedManager()
    mesh = distributed_mesh
    local_group = mesh.get_group(0)
    local_size = dist.get_world_size(group=local_group)

    tensor = torch.randn(4, 4, device=dm.device)

    config = RingPassingConfig(
        mesh_dim=0,
        mesh_size=local_size,
        communication_method="a2a",
    )

    # All ranks must hit this to avoid hangs
    raised = False
    try:
        perform_ring_iteration_async(tensor, mesh, config)
    except ValueError:
        raised = True

    collective_assert(raised, msg="async should reject a2a communication")
