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
Tests that verify RingSDPA (overlapping) produces the same results as
RingSDPABlocking (reference), both forward and backward.

Also tests the end-to-end path through F.scaled_dot_product_attention
on ShardTensors, confirming it matches a single-GPU reference.

Run with:
    torchrun --nproc-per-node 2 -m pytest --multigpu-static \
        test/domain_parallel/ops/test_ring_sdpa_overlap.py
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.domain_parallel.shard_utils.attention_patches import (
    RingSDPA,
    RingSDPABlocking,
)
from physicsnemo.domain_parallel.shard_utils.ring import RingPassingConfig

from .utils import collective_assert_close, numerical_shard_tensor_check

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SDPAWrapper(torch.nn.Module):
    """Thin wrapper so numerical_shard_tensor_check can call SDPA as a module."""

    def forward(self, q, k, v, **kwargs):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, **kwargs)


def _generate_qkv(
    batch_size, num_heads, seq_len, head_dim, device, dtype=torch.float32
):
    """Generate random Q, K, V tensors on the given device."""
    shape = (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    return q, k, v


# ---------------------------------------------------------------------------
# Forward-only correctness: RingSDPA vs RingSDPABlocking
# ---------------------------------------------------------------------------


@pytest.mark.multigpu_static
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [256])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("head_dim", [32, 64])
def test_ring_sdpa_forward_matches_blocking(
    distributed_mesh,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
):
    """RingSDPA.forward() should produce the same output as RingSDPABlocking.forward()."""
    dm = DistributedManager()
    mesh = distributed_mesh
    local_group = mesh.get_group(0)
    local_size = dist.get_world_size(group=local_group)

    if seq_len % local_size != 0:
        pytest.skip(f"seq_len {seq_len} not divisible by world_size {local_size}")

    local_seq = seq_len // local_size
    q, k, v = _generate_qkv(batch_size, num_heads, local_seq, head_dim, dm.device)

    ring_config = RingPassingConfig(
        mesh_dim=0,
        mesh_size=local_size,
        communication_method="p2p",
    )

    attn_args = {"dropout_p": 0.0, "is_causal": False, "scale": None}

    out_blocking = RingSDPABlocking.apply(
        q,
        k,
        v,
        None,
        mesh,
        ring_config,
        attn_args,
    )
    out_overlap = RingSDPA.apply(
        q,
        k,
        v,
        None,
        mesh,
        ring_config,
        attn_args,
    )

    collective_assert_close(
        out_overlap,
        out_blocking,
        atol=1e-4,
        rtol=1e-4,
        msg="RingSDPA forward vs RingSDPABlocking forward",
    )


# ---------------------------------------------------------------------------
# Forward + backward correctness: RingSDPA vs RingSDPABlocking
# ---------------------------------------------------------------------------


@pytest.mark.multigpu_static
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [256])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("head_dim", [32])
def test_ring_sdpa_backward_matches_blocking(
    distributed_mesh,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
):
    """RingSDPA backward gradients should match RingSDPABlocking gradients."""
    dm = DistributedManager()
    mesh = distributed_mesh
    local_group = mesh.get_group(0)
    local_size = dist.get_world_size(group=local_group)

    if seq_len % local_size != 0:
        pytest.skip(f"seq_len {seq_len} not divisible by world_size {local_size}")

    local_seq = seq_len // local_size

    ring_config = RingPassingConfig(
        mesh_dim=0,
        mesh_size=local_size,
        communication_method="p2p",
    )
    attn_args = {"dropout_p": 0.0, "is_causal": False, "scale": None}

    # --- Blocking path ---
    q_b, k_b, v_b = _generate_qkv(
        batch_size,
        num_heads,
        local_seq,
        head_dim,
        dm.device,
    )
    q_b.requires_grad_(True)
    k_b.requires_grad_(True)
    v_b.requires_grad_(True)

    out_b = RingSDPABlocking.apply(q_b, k_b, v_b, None, mesh, ring_config, attn_args)
    loss_b = out_b.mean()
    loss_b.backward()

    # --- Overlapping path (same input data) ---
    q_o = q_b.detach().clone().requires_grad_(True)
    k_o = k_b.detach().clone().requires_grad_(True)
    v_o = v_b.detach().clone().requires_grad_(True)

    out_o = RingSDPA.apply(q_o, k_o, v_o, None, mesh, ring_config, attn_args)
    loss_o = out_o.mean()
    loss_o.backward()

    # Compare outputs
    collective_assert_close(
        out_o,
        out_b,
        atol=1e-4,
        rtol=1e-4,
        msg="backward: forward output mismatch",
    )

    # Compare gradients
    collective_assert_close(
        q_o.grad,
        q_b.grad,
        atol=1e-4,
        rtol=1e-4,
        msg="backward: grad_q mismatch",
    )
    collective_assert_close(
        k_o.grad,
        k_b.grad,
        atol=1e-4,
        rtol=1e-4,
        msg="backward: grad_k mismatch",
    )
    collective_assert_close(
        v_o.grad,
        v_b.grad,
        atol=1e-4,
        rtol=1e-4,
        msg="backward: grad_v mismatch",
    )


# ---------------------------------------------------------------------------
# End-to-end through sdpa_wrapper (F.scaled_dot_product_attention)
# ---------------------------------------------------------------------------


@pytest.mark.multigpu_static
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [256, 512])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_dim", [32])
@pytest.mark.parametrize("backward", [False, True])
def test_sdpa_e2e_sharded_vs_reference(
    distributed_mesh,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    backward,
):
    """F.scaled_dot_product_attention on ShardTensors should match single-GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (batch_size, num_heads, seq_len, head_dim)

    q = torch.randn(shape, device=dm.device)
    k = torch.randn(shape, device=dm.device)
    v = torch.randn(shape, device=dm.device)

    placements = (Shard(2),)  # shard along sequence dim

    sq = scatter_tensor(q, 0, distributed_mesh, placements, requires_grad=backward)
    sk = scatter_tensor(k, 0, distributed_mesh, placements, requires_grad=backward)
    sv = scatter_tensor(v, 0, distributed_mesh, placements, requires_grad=backward)

    module = SDPAWrapper()
    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sq, sk, sv],
        {},
        check_grads=backward,
        atol=1e-4,
        rtol=1e-4,
    )
