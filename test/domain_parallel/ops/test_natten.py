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

r"""Tests for 1D, 2D, and 3D neighborhood attention on sharded tensors.

This module validates the correctness of :func:`physicsnemo.nn.functional.natten.na1d`,
:func:`physicsnemo.nn.functional.natten.na2d`, and
:func:`physicsnemo.nn.functional.natten.na3d` over sharded inputs, covering both
forward and backward passes. Sharding is performed over spatial dimensions which
correspond to ``Shard(1)``, ``Shard(2)``, etc. in the natten heads-last layout.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from test.conftest import requires_module

from .utils import collective_assert_close, sharded_to_local, validate_shard_tensor_spec


def _run_natten_check(
    na_func,
    distributed_mesh,
    spatial_shape,
    num_heads,
    head_dim,
    kernel_size,
    placements,
    backward,
):
    """Shared helper that tests a natten function over sharded tensors.

    Compares sharded forward/backward against local (unsharded) reference.
    """
    dm = DistributedManager()

    # (B, *spatial, heads, D)
    full_shape = (1, *spatial_shape, num_heads, head_dim)
    q = torch.randn(full_shape, device=dm.device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    sq = scatter_tensor(q, 0, distributed_mesh, placements, requires_grad=backward)
    sk = scatter_tensor(k, 0, distributed_mesh, placements, requires_grad=backward)
    sv = scatter_tensor(v, 0, distributed_mesh, placements, requires_grad=backward)

    # --- Forward: sharded path ---
    d_output = na_func(sq, sk, sv, kernel_size=kernel_size, dilation=1)

    # --- Forward: local reference ---
    if backward:
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)

    ref_output = na_func(q, k, v, kernel_size=kernel_size, dilation=1)

    validate_shard_tensor_spec(d_output)

    local_output = sharded_to_local(d_output)
    collective_assert_close(
        ref_output,
        local_output,
        atol=1e-4,
        rtol=1e-4,
        msg=f"{na_func.__name__} forward output mismatch",
    )

    if backward:
        d_output.mean().backward()
        ref_output.mean().backward()

        for name, shard_t, ref_t in [("q", sq, q), ("k", sk, k), ("v", sv, v)]:
            local_grad = sharded_to_local(shard_t.grad)
            collective_assert_close(
                ref_t.grad,
                local_grad,
                atol=1e-3,
                rtol=1e-3,
                msg=f"{na_func.__name__} backward {name}.grad mismatch",
            )


# ---------------------------------------------------------------------------
# 1D neighborhood attention  --  tensor layout: (B, L, heads, D)
# ---------------------------------------------------------------------------


@requires_module("natten")
class TestNA1D:
    """Tests for sharded 1D neighborhood attention."""

    @pytest.mark.multigpu_static
    @pytest.mark.parametrize("L", [32, 64])
    @pytest.mark.parametrize("num_heads", [4])
    @pytest.mark.parametrize("head_dim", [32])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("backward", [False, True])
    def test_na1d_shard_l(
        self, distributed_mesh, L, num_heads, head_dim, kernel_size, backward
    ):
        from physicsnemo.nn.functional.natten import na1d

        _run_natten_check(
            na1d,
            distributed_mesh,
            spatial_shape=(L,),
            num_heads=num_heads,
            head_dim=head_dim,
            kernel_size=kernel_size,
            placements=(Shard(1),),
            backward=backward,
        )


# ---------------------------------------------------------------------------
# 2D neighborhood attention  --  tensor layout: (B, H, W, heads, D)
# ---------------------------------------------------------------------------


@requires_module("natten")
class TestNA2D:
    """Tests for sharded 2D neighborhood attention."""

    @pytest.mark.multigpu_static
    @pytest.mark.parametrize("H", [16, 32])
    @pytest.mark.parametrize("W", [16])
    @pytest.mark.parametrize("num_heads", [4])
    @pytest.mark.parametrize("head_dim", [32])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("backward", [False, True])
    def test_na2d_shard_h(
        self, distributed_mesh, H, W, num_heads, head_dim, kernel_size, backward
    ):
        from physicsnemo.nn.functional.natten import na2d

        _run_natten_check(
            na2d,
            distributed_mesh,
            spatial_shape=(H, W),
            num_heads=num_heads,
            head_dim=head_dim,
            kernel_size=kernel_size,
            placements=(Shard(1),),
            backward=backward,
        )

    @pytest.mark.multigpu_static
    @pytest.mark.parametrize("H", [16])
    @pytest.mark.parametrize("W", [16, 32])
    @pytest.mark.parametrize("num_heads", [4])
    @pytest.mark.parametrize("head_dim", [32])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("backward", [False, True])
    def test_na2d_shard_w(
        self, distributed_mesh, H, W, num_heads, head_dim, kernel_size, backward
    ):
        from physicsnemo.nn.functional.natten import na2d

        _run_natten_check(
            na2d,
            distributed_mesh,
            spatial_shape=(H, W),
            num_heads=num_heads,
            head_dim=head_dim,
            kernel_size=kernel_size,
            placements=(Shard(2),),
            backward=backward,
        )


# ---------------------------------------------------------------------------
# 3D neighborhood attention  --  tensor layout: (B, X, Y, Z, heads, D)
# ---------------------------------------------------------------------------


@requires_module("natten")
class TestNA3D:
    """Tests for sharded 3D neighborhood attention."""

    @pytest.mark.multigpu_static
    @pytest.mark.parametrize("X", [8, 16])
    @pytest.mark.parametrize("Y", [8])
    @pytest.mark.parametrize("Z", [8])
    @pytest.mark.parametrize("num_heads", [4])
    @pytest.mark.parametrize("head_dim", [32])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("backward", [False, True])
    def test_na3d_shard_x(
        self, distributed_mesh, X, Y, Z, num_heads, head_dim, kernel_size, backward
    ):
        from physicsnemo.nn.functional.natten import na3d

        _run_natten_check(
            na3d,
            distributed_mesh,
            spatial_shape=(X, Y, Z),
            num_heads=num_heads,
            head_dim=head_dim,
            kernel_size=kernel_size,
            placements=(Shard(1),),
            backward=backward,
        )

    @pytest.mark.multigpu_static
    @pytest.mark.parametrize("X", [8])
    @pytest.mark.parametrize("Y", [8, 16])
    @pytest.mark.parametrize("Z", [8])
    @pytest.mark.parametrize("num_heads", [4])
    @pytest.mark.parametrize("head_dim", [32])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("backward", [False, True])
    def test_na3d_shard_y(
        self, distributed_mesh, X, Y, Z, num_heads, head_dim, kernel_size, backward
    ):
        from physicsnemo.nn.functional.natten import na3d

        _run_natten_check(
            na3d,
            distributed_mesh,
            spatial_shape=(X, Y, Z),
            num_heads=num_heads,
            head_dim=head_dim,
            kernel_size=kernel_size,
            placements=(Shard(2),),
            backward=backward,
        )

    @pytest.mark.multigpu_static
    @pytest.mark.parametrize("X", [8])
    @pytest.mark.parametrize("Y", [8])
    @pytest.mark.parametrize("Z", [8, 16])
    @pytest.mark.parametrize("num_heads", [4])
    @pytest.mark.parametrize("head_dim", [32])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("backward", [False, True])
    def test_na3d_shard_z(
        self, distributed_mesh, X, Y, Z, num_heads, head_dim, kernel_size, backward
    ):
        from physicsnemo.nn.functional.natten import na3d

        _run_natten_check(
            na3d,
            distributed_mesh,
            spatial_shape=(X, Y, Z),
            num_heads=num_heads,
            head_dim=head_dim,
            kernel_size=kernel_size,
            placements=(Shard(3),),
            backward=backward,
        )
