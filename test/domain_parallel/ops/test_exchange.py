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
Tests for the row-exchange primitives in ``shard_utils.exchange``
(``funcol_all_to_all_v_rows`` and ``resolve_group_name``), the transport
shared by the halo scatter correction and the routed gather.
"""

import pytest
import torch
import torch.distributed as dist

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel.shard_utils import exchange
from physicsnemo.domain_parallel.shard_utils.exchange import (
    funcol_all_to_all_v_rows,
    resolve_group_name,
)

from .utils import collective_assert, collective_assert_close


def _send_counts(rank: int, world_size: int) -> list[int]:
    """Rank ``r`` sends ``r + k`` rows to rank ``k``: every pair differs, some are 0."""
    return [rank + k for k in range(world_size)]


@pytest.mark.multigpu_static
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("trailing", [(), (3,), (2, 2)])
def test_all_to_all_v_rows_routes_by_rank(distributed_mesh, dtype, trailing):
    """Rows land on the destination rank, in source-rank order, with the right shape."""
    dm = DistributedManager()
    mesh = distributed_mesh
    rank = mesh.get_local_rank(0)
    world_size = dist.get_world_size(group=mesh.get_group(0))

    send_counts = _send_counts(rank, world_size)
    # Row values encode (source, destination) so the receiver can verify both.
    blocks = []
    for dst, n in enumerate(send_counts):
        block = torch.full(
            (n, *trailing), rank * 100 + dst, dtype=dtype, device=dm.device
        )
        blocks.append(block)
    send_rows = torch.cat(blocks, dim=0)
    # Rank r receives ``src + r`` rows from every source rank ``src``.
    recv_counts = [src + rank for src in range(world_size)]

    received = funcol_all_to_all_v_rows(send_rows, send_counts, recv_counts, mesh)

    collective_assert(
        tuple(received.shape) == (sum(recv_counts), *trailing),
        msg=f"received shape {tuple(received.shape)}",
    )
    expected = torch.cat(
        [
            torch.full((n, *trailing), src * 100 + rank, dtype=dtype, device=dm.device)
            for src, n in enumerate(recv_counts)
        ],
        dim=0,
    )
    collective_assert_close(received, expected, atol=0, rtol=0, msg="a2a-v rows")


@pytest.mark.multigpu_static
def test_all_to_all_v_rows_empty_on_one_rank(distributed_mesh):
    """A rank that sends and receives nothing still completes the exchange."""
    dm = DistributedManager()
    mesh = distributed_mesh
    rank = mesh.get_local_rank(0)
    world_size = dist.get_world_size(group=mesh.get_group(0))

    # Only rank 0 sends: 2 rows to every other rank, nothing to itself.
    send_counts = [0] + [2] * (world_size - 1) if rank == 0 else [0] * world_size
    recv_counts = [0] * world_size if rank == 0 else [2] + [0] * (world_size - 1)
    send_rows = torch.full((sum(send_counts), 4), float(rank), device=dm.device)

    received = funcol_all_to_all_v_rows(send_rows, send_counts, recv_counts, mesh)

    expected = torch.zeros((sum(recv_counts), 4), device=dm.device)
    collective_assert(tuple(received.shape) == tuple(expected.shape), msg="shape")
    collective_assert_close(received, expected, atol=0, rtol=0, msg="from rank 0")


@pytest.mark.multigpu_static
def test_resolve_group_name(distributed_mesh):
    """Mesh, process group, string and ``None`` all resolve to a c10d group-name token."""
    mesh = distributed_mesh
    group = mesh.get_group(0)

    name = resolve_group_name(mesh)
    collective_assert(isinstance(name, str) and name != "", msg="mesh -> name")
    collective_assert(resolve_group_name(group) == name, msg="group matches mesh")
    collective_assert(resolve_group_name(name) == name, msg="string passthrough")
    collective_assert(resolve_group_name(None) == "", msg="None -> default group")


def test_fp32_scatter_accumulator_setting():
    """``float32`` scatter folds accumulate in the configured dtype (default float64)."""
    assert exchange.get_fp32_scatter_accumulator() is torch.float64
    assert exchange._accumulator_dtype(torch.float32) is torch.float64
    assert exchange._accumulator_dtype(torch.bfloat16) is torch.float32
    assert exchange._accumulator_dtype(torch.int64) is torch.int64
    try:
        exchange.set_fp32_scatter_accumulator(torch.float32)
        assert exchange._accumulator_dtype(torch.float32) is torch.float32
        with pytest.raises(ValueError):
            exchange.set_fp32_scatter_accumulator(torch.float16)
    finally:
        exchange.set_fp32_scatter_accumulator(torch.float64)
