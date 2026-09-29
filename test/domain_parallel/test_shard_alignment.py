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

r"""Misaligned uneven shard boundaries are rejected, not silently computed.

Regression tests for issue #1943: two ShardTensors can share a global shape
and a placement tuple while assigning different global rows to each rank.
The DTensor fallback would pair unrelated local rows (zero-size shards even
defeat the local shape check via 0/1 broadcasting), so the fallback paths
now raise a deterministic error instead. All assertions here are pure
metadata checks raised identically on every rank -- no collective can be
left half-posted.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, validate_aligned_sharding

pytestmark = [pytest.mark.multigpu_static, pytest.mark.timeout(120)]


def _uneven_pair(mesh, device, cols=3, seed=7):
    r"""Two ShardTensors, same global shape and placements, misaligned rows.

    Global row count is ``ws + 1``. Operand ``a`` gives the extra row to
    rank 0 (sizes ``2, 1, 1, ...``); operand ``b`` gives it to the last
    rank (sizes ``1, ..., 1, 2``). At world size >= 2 the boundaries
    disagree while every piece of DTensor-visible metadata matches.
    """
    dm = DistributedManager()
    ws = mesh.size(0)
    n = ws + 1
    torch.manual_seed(seed)
    a_full = torch.randn(n, cols, device=device)
    b_full = torch.randn(n, cols, device=device)

    a_rows = [2] + [1] * (ws - 1)
    b_rows = [1] * (ws - 1) + [2]

    def build(full, rows):
        offset = sum(rows[: dm.rank])
        return ShardTensor.from_local(
            full[offset : offset + rows[dm.rank]].clone(),
            mesh,
            (Shard(0),),
            sharding_shapes={0: [(r, cols) for r in rows]},
            global_shape=(n, cols),
        )

    return build(a_full, a_rows), build(b_full, b_rows), a_full, b_full


def test_misaligned_add_raises(distributed_mesh):
    r"""The issue's reproducer shape: add must raise, not pair wrong rows."""
    a_s, b_s, _, _ = _uneven_pair(distributed_mesh, DistributedManager().device)

    with pytest.raises(RuntimeError, match="misaligned"):
        torch.add(a_s, b_s)


def test_misaligned_mul_raises(distributed_mesh):
    a_s, b_s, _, _ = _uneven_pair(distributed_mesh, DistributedManager().device)

    with pytest.raises(RuntimeError, match="misaligned"):
        torch.mul(a_s, b_s)


def test_aligned_uneven_add_passes(distributed_mesh):
    r"""Control: identical uneven boundaries still compute correct values."""
    dm = DistributedManager()
    ws = distributed_mesh.size(0)
    n = ws + 1
    torch.manual_seed(11)
    a_full = torch.randn(n, 3, device=dm.device)
    b_full = torch.randn(n, 3, device=dm.device)
    rows = [2] + [1] * (ws - 1)
    offset = sum(rows[: dm.rank])

    def build(full):
        return ShardTensor.from_local(
            full[offset : offset + rows[dm.rank]].clone(),
            distributed_mesh,
            (Shard(0),),
            sharding_shapes={0: [(r, 3) for r in rows]},
            global_shape=(n, 3),
        )

    result = torch.add(build(a_full), build(b_full))

    torch.testing.assert_close(result.full_tensor(), a_full + b_full)


def test_broadcast_columns_misaligned_raises(distributed_mesh):
    r"""Different global shapes but the SAME sharded extent still align-check:
    (n, 3) + (n, 1) pair rows directly, so misaligned rows must raise."""
    dm = DistributedManager()
    a_s, _, _, _ = _uneven_pair(distributed_mesh, dm.device)
    _, b1_s, _, _ = _uneven_pair(distributed_mesh, dm.device, cols=1, seed=8)

    with pytest.raises(RuntimeError, match="misaligned"):
        torch.add(a_s, b1_s)


def test_different_extents_not_flagged_as_misaligned(distributed_mesh):
    r"""Different global extents on the sharded dim are a shape problem, not
    boundary misalignment; the alignment check must not claim them."""
    dm = DistributedManager()
    ws = distributed_mesh.size(0)
    a_s, _, _, _ = _uneven_pair(distributed_mesh, dm.device)

    n_big = ws + 1 + ws
    rows = [3] + [2] * (ws - 1)  # sums to n_big
    torch.manual_seed(9)
    big = torch.randn(n_big, 3, device=dm.device)
    offset = sum(rows[: dm.rank])
    big_s = ShardTensor.from_local(
        big[offset : offset + rows[dm.rank]].clone(),
        distributed_mesh,
        (Shard(0),),
        sharding_shapes={0: [(r, 3) for r in rows]},
        global_shape=(n_big, 3),
    )

    with pytest.raises(Exception) as excinfo:
        torch.add(a_s, big_s)
    assert "misaligned" not in str(excinfo.value)


def test_validate_aligned_sharding_direct(distributed_mesh):
    r"""The unified API is callable on raw specs (the extension-path usage)."""
    dm = DistributedManager()
    a_s, b_s, _, _ = _uneven_pair(distributed_mesh, dm.device)

    # Same spec twice: aligned, no raise.
    validate_aligned_sharding([a_s._spec, a_s._spec], "test_op")

    with pytest.raises(RuntimeError, match="test_op.*misaligned"):
        validate_aligned_sharding([a_s._spec, b_s._spec], "test_op")


def test_sharded_with_replicated_passes(distributed_mesh):
    r"""A single sharded operand has nothing to misalign against."""
    dm = DistributedManager()
    a_s, _, a_full, _ = _uneven_pair(distributed_mesh, dm.device)
    torch.manual_seed(13)
    c = torch.randn(1, 3, device=dm.device)
    c_s = ShardTensor.from_local(c, distributed_mesh, (Replicate(),))

    result = torch.add(a_s, c_s)

    torch.testing.assert_close(result.full_tensor(), a_full + c)


# ---------------------------------------------------------------------------
# Smoke tests: custom handlers (which bypass the DTensor fallback) raise the
# same deterministic error on misaligned inputs.
# ---------------------------------------------------------------------------


def test_cross_misaligned_raises(distributed_mesh):
    r"""The cross handler validates boundaries before pairing locals."""
    dm = DistributedManager()
    a_s, b_s, _, _ = _uneven_pair(distributed_mesh, dm.device)

    with pytest.raises(RuntimeError, match="misaligned"):
        torch.linalg.cross(a_s, b_s)


def _uneven_seq_qkv(mesh, device, misaligned: bool):
    r"""(B, H, S, D) q/k/v with k and v sharded on the sequence dim.

    q is replicated; k and v use identical or shifted uneven boundaries.
    """
    dm = DistributedManager()
    ws = mesh.size(0)
    s = ws + 1
    torch.manual_seed(17)
    q = torch.randn(1, 2, s, 8, device=device)
    k_full = torch.randn(1, 2, s, 8, device=device)
    v_full = torch.randn(1, 2, s, 8, device=device)

    k_rows = [2] + [1] * (ws - 1)
    v_rows = ([1] * (ws - 1) + [2]) if misaligned else k_rows

    def build(full, rows):
        offset = sum(rows[: dm.rank])
        return ShardTensor.from_local(
            full[:, :, offset : offset + rows[dm.rank]].clone(),
            mesh,
            (Shard(2),),
            sharding_shapes={0: [(1, 2, r, 8) for r in rows]},
            global_shape=(1, 2, s, 8),
        )

    q_s = ShardTensor.from_local(q, mesh, (Replicate(),))
    return q_s, build(k_full, k_rows), build(v_full, v_rows)


def test_sdpa_misaligned_kv_raises(distributed_mesh):
    r"""SDPA pairs k/v positionally along the sequence dim in every path."""
    dm = DistributedManager()
    q_s, k_s, v_s = _uneven_seq_qkv(distributed_mesh, dm.device, misaligned=True)

    with pytest.raises(RuntimeError, match="misaligned"):
        torch.nn.functional.scaled_dot_product_attention(q_s, k_s, v_s)


def test_natten_misaligned_qkv_raises(distributed_mesh):
    r"""natten computes halo configs from q and applies them to k/v."""
    natten = pytest.importorskip("natten")  # noqa: F841
    from physicsnemo.nn.functional.natten import na2d

    dm = DistributedManager()
    ws = distributed_mesh.size(0)
    s = 4 * ws + 4
    torch.manual_seed(19)
    full = torch.randn(1, s, 16, 2, 8, device=dm.device)

    rows_a = [8] + [4] * (ws - 1)
    rows_b = [4] * (ws - 1) + [8]

    def build(rows):
        offset = sum(rows[: dm.rank])
        return ShardTensor.from_local(
            full[:, offset : offset + rows[dm.rank]].clone(),
            distributed_mesh,
            (Shard(1),),
            sharding_shapes={0: [(1, r, 16, 2, 8) for r in rows]},
            global_shape=tuple(full.shape),
        )

    q_s, k_s, v_s = build(rows_a), build(rows_b), build(rows_b)

    with pytest.raises(RuntimeError, match="misaligned"):
        na2d(q_s, k_s, v_s, kernel_size=3)


def test_scatter_add_misaligned_raises(distributed_mesh):
    r"""scatter_add with genuinely sharded, misaligned index/src raises.

    Without halo routing metadata the handler falls back to the DTensor
    fallback, whose alignment check covers the same pairing.
    """
    dm = DistributedManager()
    ws = distributed_mesh.size(0)
    n = ws + 1
    torch.manual_seed(23)

    acc = torch.zeros(n, 3, device=dm.device)
    acc_s = ShardTensor.from_local(acc, distributed_mesh, (Replicate(),))

    rows_a = [2] + [1] * (ws - 1)
    rows_b = [1] * (ws - 1) + [2]

    def build(full, rows):
        offset = sum(rows[: dm.rank])
        return ShardTensor.from_local(
            full[offset : offset + rows[dm.rank]].clone(),
            distributed_mesh,
            (Shard(0),),
            sharding_shapes={0: [(r, 3) for r in rows]},
            global_shape=(n, 3),
        )

    index = build(torch.zeros(n, 3, dtype=torch.int64, device=dm.device), rows_a)
    src = build(torch.randn(n, 3, device=dm.device), rows_b)

    with pytest.raises(RuntimeError, match="misaligned"):
        acc_s.scatter_add(0, index, src)
