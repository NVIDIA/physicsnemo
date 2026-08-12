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
Test scaled dot product attention with shard tensor inputs.

Sharding is supported only over the sequence dimension.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor

from .utils import numerical_shard_tensor_check


class SDPAWrapper(torch.nn.Module):
    """
    TESTING ONLY

    This is a thin wrapper for SDPA to enable the test.

    """

    def __init__(
        self, dropout_p: float = 0.0, is_causal: bool = False, scale: float = None
    ) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.scale = scale

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, **kwargs)


def generate_sequence_data(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    *,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Generate a random sequence-like tensor
    """

    return torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)


@pytest.mark.multigpu_static
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [256, 512])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_dim", [32, 64])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize(
    "is_causal",
    [
        False,
    ],
)
@pytest.mark.parametrize(
    "scale",
    [
        None,
    ],
)
@pytest.mark.parametrize("backward", [False, True])
def test_sdpa_sequence_parallel(
    distributed_mesh,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    dropout,
    is_causal,
    scale,
    backward,
):
    """Test basic scaled dot product attention with various configurations"""

    # Skip test configurations that are not supported
    if dropout > 0.0:
        pytest.xfail("Dropout > 0 is not currently supported in sharded SDPA")

    if is_causal:
        pytest.xfail("Causal SDPA is not currently supported in sharded SDPA")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()

    q = generate_sequence_data(batch_size, seq_len, num_heads, head_dim).to(dm.device)
    k = generate_sequence_data(batch_size, seq_len, num_heads, head_dim).to(dm.device)
    v = generate_sequence_data(batch_size, seq_len, num_heads, head_dim).to(dm.device)

    placements = (Shard(2),)

    # Distribute q, k, v:
    sq = scatter_tensor(q, 0, distributed_mesh, placements, requires_grad=backward)
    sk = scatter_tensor(k, 0, distributed_mesh, placements, requires_grad=backward)
    sv = scatter_tensor(v, 0, distributed_mesh, placements, requires_grad=backward)

    module = SDPAWrapper(dropout_p=dropout, is_causal=is_causal, scale=scale)

    numerical_shard_tensor_check(
        distributed_mesh, module, [sq, sk, sv], {}, check_grads=backward
    )


# ---------------------------------------------------------------------------
# Mixed-placement SDPA, as used by FLARE and GALE_FA:
#   pass 1: sdpa(G_replicated, k_sharded, v_sharded)  -- latent tokens attend
#           to the sharded point cloud via partial-softmax (logsumexp)
#           accumulation over the KV shards. Output follows q: Replicate.
#   pass 2: sdpa(q_sharded, K_replicated, V_replicated) -- embarrassingly
#           parallel over the query shards. Output follows q: Shard(2).
# ---------------------------------------------------------------------------


def _assert_placements(expected_placements):
    def check(output):
        assert output._spec.placements == expected_placements, (
            f"expected output placements {expected_placements}, "
            f"got {output._spec.placements}"
        )

    return check


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len_q", [64])
@pytest.mark.parametrize("seq_len_kv", [256])
@pytest.mark.parametrize("backward", [False, True])
def test_sdpa_replicated_q_sharded_kv(
    distributed_mesh, batch_size, seq_len_q, seq_len_kv, backward
):
    r"""FLARE pass 1: replicated queries attending to a sharded point cloud.

    The output has q's (replicated) sequence axis, so it must come back
    ``Replicate``; grads must come back with each input's own placement
    (``numerical_shard_tensor_check`` asserts grad spec == input spec).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    num_heads, head_dim = 8, 32
    dm = DistributedManager()

    q = generate_sequence_data(batch_size, seq_len_q, num_heads, head_dim).to(dm.device)
    k = generate_sequence_data(batch_size, seq_len_kv, num_heads, head_dim).to(
        dm.device
    )
    v = generate_sequence_data(batch_size, seq_len_kv, num_heads, head_dim).to(
        dm.device
    )

    sq = scatter_tensor(q, 0, distributed_mesh, (Replicate(),), requires_grad=backward)
    sk = scatter_tensor(k, 0, distributed_mesh, (Shard(2),), requires_grad=backward)
    sv = scatter_tensor(v, 0, distributed_mesh, (Shard(2),), requires_grad=backward)

    numerical_shard_tensor_check(
        distributed_mesh,
        SDPAWrapper(),
        [sq, sk, sv],
        {},
        check_grads=backward,
        atol=1e-4,
        rtol=1e-4,
        output_check_fn=_assert_placements((Replicate(),)),
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len_q", [256])
@pytest.mark.parametrize("seq_len_kv", [64])
@pytest.mark.parametrize("backward", [False, True])
def test_sdpa_sharded_q_replicated_kv(
    distributed_mesh, batch_size, seq_len_q, seq_len_kv, backward
):
    r"""FLARE pass 2: sharded queries attending to replicated keys/values.

    Embarrassingly parallel over the query shards -- no communication in
    forward. The output must come back ``Shard(2)`` (following q).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    num_heads, head_dim = 8, 32
    dm = DistributedManager()

    q = generate_sequence_data(batch_size, seq_len_q, num_heads, head_dim).to(dm.device)
    k = generate_sequence_data(batch_size, seq_len_kv, num_heads, head_dim).to(
        dm.device
    )
    v = generate_sequence_data(batch_size, seq_len_kv, num_heads, head_dim).to(
        dm.device
    )

    sq = scatter_tensor(q, 0, distributed_mesh, (Shard(2),), requires_grad=backward)
    sk = scatter_tensor(k, 0, distributed_mesh, (Replicate(),), requires_grad=backward)
    sv = scatter_tensor(v, 0, distributed_mesh, (Replicate(),), requires_grad=backward)

    numerical_shard_tensor_check(
        distributed_mesh,
        SDPAWrapper(),
        [sq, sk, sv],
        {},
        check_grads=backward,
        atol=1e-4,
        rtol=1e-4,
        output_check_fn=_assert_placements((Shard(2),)),
    )


def test_blockwise_logsumexp_fold_matches_full_sdpa():
    r"""Single-rank numerics: the log-space accumulation building blocks
    (``add_log_sumexp`` / ``stable_signed_accumulate``) folded over KV blocks
    must reproduce full SDPA.

    This is the invariant the replicated-Q/sharded-KV path relies on: each
    rank owns one KV block, and one log-space fold over the mesh recovers
    the full attention output.
    """
    from physicsnemo.domain_parallel.shard_utils.attention_patches import (
        add_log_sumexp,
        stable_signed_accumulate,
    )

    torch.manual_seed(3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch, heads, s_q, s_kv, dim, n_blocks = 2, 4, 16, 64, 32, 4

    q = torch.randn(batch, heads, s_q, dim, device=device)
    k = torch.randn(batch, heads, s_kv, dim, device=device)
    v = torch.randn(batch, heads, s_kv, dim, device=device)

    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    log_global_output = None
    sign_global_output = None
    global_log_sumexp = None
    for k_block, v_block in zip(k.chunk(n_blocks, dim=2), v.chunk(n_blocks, dim=2)):
        output, log_sumexp = torch.ops.aten._scaled_dot_product_efficient_attention(
            q,
            k_block.contiguous(),
            v_block.contiguous(),
            attn_bias=None,
            compute_log_sumexp=True,
        )[:2]
        log_sumexp = log_sumexp.unsqueeze(-1)
        log_output = torch.log(torch.abs(output))
        sign_output = torch.sign(output)
        log_global_output, sign_global_output = stable_signed_accumulate(
            log_global_output, sign_global_output, log_output, sign_output, log_sumexp
        )
        global_log_sumexp = add_log_sumexp(global_log_sumexp, log_sumexp)

    folded = sign_global_output * torch.exp(log_global_output - global_log_sumexp)
    torch.testing.assert_close(folded, reference, atol=1e-4, rtol=1e-4)
