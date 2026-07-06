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

r"""Integration tests for plain-tensor auto-promotion.

When a plain ``torch.Tensor`` (e.g. an all-gathered / replicated model weight)
meets a sharded ``ShardTensor`` inside an intercepted op, ``ShardTensor``
auto-promotes it to a ``Replicate`` distributed tensor on the sharded
argument's mesh. These tests verify the three ``TensorPromotionMode`` behaviors
(``WARN`` warns, ``SILENT`` is quiet, ``DISABLED`` refuses to mix) and that the
promoted computation is numerically correct.

These tests require multiple GPUs (``--multigpu-static``).
"""

import warnings

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, TensorPromotionMode, scatter_tensor


@pytest.fixture(autouse=True)
def restore_promotion_mode():
    r"""Save and restore the global ShardTensor promotion mode around each test."""
    previous = ShardTensor.get_promotion_mode()
    try:
        yield
    finally:
        ShardTensor.set_promotion_mode(previous)


def _shape_and_placements(mesh):
    r"""Return a (global_shape, placements) pair valid for the given mesh.

    The last dimension is left unsharded so it can broadcast against a plain
    1D weight, which is what triggers promotion.
    """
    global_shape = (2, 2 * 3 * 4 * 5 * 7, 2 * 3 * 4 * 5 * 7)
    placements = [Shard(1)]
    if mesh.ndim > 1:
        placements.append(Shard(2))
    return global_shape, tuple(placements)


def _make_sharded_activation(mesh):
    r"""Scatter a deterministic activation across the mesh (sharded, no grad)."""
    dm = DistributedManager()
    global_shape, placements = _shape_and_placements(mesh)

    torch.manual_seed(12345)
    full_input = torch.randn(global_shape, device=dm.device)

    shard_x = scatter_tensor(
        full_input,
        0,
        mesh,
        placements,
        global_shape=torch.Size(global_shape),
        dtype=full_input.dtype,
        requires_grad=False,
    )
    # full_input is identical on every rank (fixed seed) and equals the data
    # scattered from src=0, so it doubles as the single-process reference.
    return shard_x, full_input


def _plain_weight(global_shape):
    r"""Deterministic plain weight broadcastable against the last activation dim."""
    dm = DistributedManager()
    torch.manual_seed(999)
    return torch.randn(global_shape[-1], device=dm.device)


def run_promotion_forward(mesh, mode):
    r"""Mix a sharded activation with a plain weight under ``mode`` and check math."""
    shard_x, full_input = _make_sharded_activation(mesh)
    global_shape, _ = _shape_and_placements(mesh)
    weight = _plain_weight(global_shape)

    reference = full_input * weight

    ShardTensor.set_promotion_mode(mode)

    if mode is TensorPromotionMode.WARN:
        with pytest.warns(UserWarning, match="auto-promoting"):
            out = shard_x * weight
    else:  # SILENT
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = shard_x * weight

    assert isinstance(out, ShardTensor)
    assert torch.allclose(out.full_tensor(), reference, atol=1e-5)


def run_promotion_disabled(mesh):
    r"""With promotion DISABLED, mixing a plain non-scalar tensor must not promote."""
    shard_x, _ = _make_sharded_activation(mesh)
    global_shape, _ = _shape_and_placements(mesh)
    weight = _plain_weight(global_shape)

    ShardTensor.set_promotion_mode(TensorPromotionMode.DISABLED)

    # No promotion happens (so no warning); the underlying DTensor routing
    # refuses to mix a plain non-scalar tensor with distributed data.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(Exception):
            _ = (shard_x * weight).full_tensor()


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
@pytest.mark.parametrize("mode", [TensorPromotionMode.WARN, TensorPromotionMode.SILENT])
def test_promotion_forward_1d(distributed_mesh, mode):
    run_promotion_forward(distributed_mesh, mode)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
@pytest.mark.parametrize("mode", [TensorPromotionMode.WARN, TensorPromotionMode.SILENT])
def test_promotion_forward_2d(distributed_mesh_2d, mode):
    run_promotion_forward(distributed_mesh_2d, mode)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_promotion_disabled_1d(distributed_mesh):
    run_promotion_disabled(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_promotion_disabled_2d(distributed_mesh_2d):
    run_promotion_disabled(distributed_mesh_2d)
