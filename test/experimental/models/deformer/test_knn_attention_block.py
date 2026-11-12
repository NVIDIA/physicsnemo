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
"""The kNN attention block: parameterization, precision rules and
input validation (CPU)."""

import pytest
import torch

from physicsnemo.experimental.models.deformer.fused_knn_attention import (
    knn_attention,
)
from physicsnemo.experimental.models.deformer.local_attention import (
    KNNAttentionBlock,
)


def _inputs(n=40, k=6, dim=32, edge_dim=8, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g, dtype=dtype)
    index = torch.randint(0, n, (n, k), generator=g)
    index[:, 0] = torch.arange(n)
    mask = torch.rand(n, k, generator=g) > 0.25
    mask[:, 0] = True
    index = torch.where(mask, index, index[:, :1])  # padded slots repeat self
    edge_latent = torch.randn(n, k, edge_dim, generator=g, dtype=dtype)
    return x, index, mask, edge_latent


def _block(fused, backend="torch", seed=1, **options):
    torch.manual_seed(seed)
    return KNNAttentionBlock(
        32, 4, 8, mlp_ratio=2, fused=fused, fused_backend=backend, **options
    )


def test_block_parameterization_and_shapes():
    block = _block(False)
    names = {name for name, _ in block.named_parameters()}
    assert {
        "query.weight",
        "key.weight",
        "key.bias",
        "value.weight",
        "edge_bias.weight",
        "update_mlp.0.weight",
        "update_mlp.3.weight",
    } <= names
    # the value and edge-bias projections carry no bias (neither would receive
    # a gradient, see the block's docstring)
    assert not {"value.bias", "edge_bias.bias"} & names
    # update MLP input: LN(x) + message + edge message + lse per head
    assert block.update_mlp[0].in_features == 2 * 32 + 4 * 8 + 4
    x, index, mask, edge_latent = _inputs()
    out = block(x, index, mask, edge_latent)
    assert out.shape == x.shape and torch.isfinite(out).all()
    out.square().sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    with pytest.raises(ValueError, match="fused_backend"):
        _block(False, backend="cuda")


def test_value_projection_sees_the_raw_stream():
    block = _block(False)
    x = torch.randn(10, 32) * 50 + 3  # far from unit scale
    normalized, _, _, value = block._projections(x)
    torch.testing.assert_close(value, block.value(x))
    assert not torch.allclose(value, block.value(normalized))


def test_logit_bias_is_float32_under_autocast():
    block = _block(False)
    edge_latent = torch.randn(5, 6, 8)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert block._logit_bias(edge_latent).dtype == torch.float32


@pytest.mark.parametrize("fused", [False, True])
def test_bias_runs_in_float32_with_half_precision_parameters(fused):
    """A block converted with ``.bfloat16()`` still projects the bias in float32."""
    block = _block(fused).bfloat16()
    x, index, mask, edge_latent = _inputs(dtype=torch.bfloat16)
    assert block._logit_bias(edge_latent).dtype == torch.float32
    out = block(x, index, mask, edge_latent)
    assert out.shape == x.shape and out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()
    reference = _block(False).bfloat16()(x, index, mask, edge_latent)
    torch.testing.assert_close(out.float(), reference.float(), atol=5e-2, rtol=5e-2)


def test_gather_logits_are_float32_under_autocast(monkeypatch):
    """The gather path's softmax statistics must not be re-dispatched to bf16."""
    block = _block(False)
    x, index, mask, edge_latent = _inputs()
    seen = []
    original = torch.softmax

    def spy(input, dim, *args, **kwargs):
        seen.append(input.dtype)
        return original(input, dim, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", spy)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        block(x, index, mask, edge_latent)
    assert seen == [torch.float32]


def test_kernel_returns_detached_lse():
    x, index, mask, edge_latent = _inputs()
    q = torch.randn(40, 4, 8, requires_grad=True)
    k = torch.randn(40, 4, 8, requires_grad=True)
    v = torch.randn(40, 4, 8, requires_grad=True)
    bias = torch.randn(40, 6, 4)
    out, edge_out, lse = knn_attention(
        q,
        k,
        v,
        bias,
        edge_latent,
        index,
        mask,
        8**-0.5,
        backend="torch",
        return_lse=True,
    )
    assert lse.shape == (40, 4) and not lse.requires_grad and torch.isfinite(lse).all()
    logits = torch.einsum("nhd,nkhd->nhk", q, k[index]) * 8**-0.5 + bias.permute(
        0, 2, 1
    )
    logits = logits.masked_fill(~mask[:, None], -torch.inf)
    torch.testing.assert_close(
        lse, torch.logsumexp(logits, -1).detach(), rtol=1e-5, atol=1e-5
    )
    assert (
        len(
            knn_attention(
                q, k, v, bias, edge_latent, index, mask, 8**-0.5, backend="torch"
            )
        )
        == 2
    )
    out.sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None


def test_block_rejects_mismatched_inputs():
    block = _block(False)
    x, index, mask, edge_latent = _inputs()
    with pytest.raises(ValueError, match="Expected x of shape"):
        block(x[:, :-1], index, mask, edge_latent)
    with pytest.raises(ValueError, match="Expected int64 index"):
        block(x, index[:-1], mask[:-1], edge_latent[:-1])
    with pytest.raises(ValueError, match="Expected int64 index"):
        block(x, index.to(torch.int32), mask, edge_latent)
    with pytest.raises(ValueError, match="Expected bool mask"):
        block(x, index, mask.float(), edge_latent)
    with pytest.raises(ValueError, match="Expected edge_latent of shape"):
        block(x, index, mask, edge_latent[..., :-1])
