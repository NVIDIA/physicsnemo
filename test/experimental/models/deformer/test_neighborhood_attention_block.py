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
"""Second-design options of the neighborhood attention block (CPU)."""

import pytest
import torch

from physicsnemo.experimental.models.deformer.fused_neighborhood_attention import (
    neighborhood_attention,
)
from physicsnemo.experimental.models.deformer.local_attention import (
    LocalNeighborhood,
    NeighborhoodAttentionBlock,
    NeighborhoodAttentionProcessor,
    fold_legacy_block_biases,
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
    return NeighborhoodAttentionBlock(
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
    # the value and edge-bias biases never received a gradient and are gone
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


def test_fused_torch_backend_matches_gather_path():
    """Both execution paths implement the same block, forward and backward."""
    torch.manual_seed(0)
    x, index, mask, edge_latent = _inputs(dtype=torch.float64)
    gather = _block(False).double()
    fused = _block(True).double()
    fused.load_state_dict(gather.state_dict())
    x_g = x.clone().requires_grad_()
    x_f = x.clone().requires_grad_()
    e_g = edge_latent.clone().requires_grad_()
    e_f = edge_latent.clone().requires_grad_()
    out_g = gather(x_g, index, mask, e_g)
    out_f = fused(x_f, index, mask, e_f)
    torch.testing.assert_close(out_f, out_g, rtol=1e-9, atol=1e-9)
    (out_g * torch.arange(1, 33)).sum().backward()
    (out_f * torch.arange(1, 33)).sum().backward()
    torch.testing.assert_close(x_f.grad, x_g.grad, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(e_f.grad, e_g.grad, rtol=1e-8, atol=1e-9)
    for (name, pg), (_, pf) in zip(gather.named_parameters(), fused.named_parameters()):
        torch.testing.assert_close(pf.grad, pg.grad, rtol=1e-8, atol=1e-9, msg=name)


def _legacy_block(block, seed=3):
    """A copy of ``block`` with the biases earlier revisions gave value and edge_bias."""
    import copy

    legacy = copy.deepcopy(block)
    generator = torch.Generator().manual_seed(seed)
    for name in ("value", "edge_bias"):
        linear = getattr(legacy, name)
        biased = torch.nn.Linear(linear.in_features, linear.out_features)
        with torch.no_grad():
            biased.weight.copy_(linear.weight)
            biased.bias.copy_(torch.randn(linear.out_features, generator=generator))
        setattr(legacy, name, biased)
    return legacy


@pytest.mark.parametrize("fused", [False, True])
def test_fold_legacy_block_biases_reproduces_the_biased_block(fused):
    """Dropping the two biases and folding the lse shift changes no output."""
    block = _block(fused)
    legacy = _legacy_block(block)
    legacy_state = legacy.state_dict()
    assert {"value.bias", "edge_bias.bias"} <= set(legacy_state)
    folded = fold_legacy_block_biases(legacy_state)
    assert set(folded) == set(block.state_dict())
    migrated = _block(fused, seed=7)
    migrated.load_state_dict(folded, strict=True)
    x, index, mask, edge_latent = _inputs()
    with torch.no_grad():
        expected = legacy(x, index, mask, edge_latent)
        actual = migrated(x, index, mask, edge_latent)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    # the fold is exact: folded from float64 parameters, the two agree to
    # float64 rounding (above, the folded bias was stored in float32)
    import copy

    legacy64 = copy.deepcopy(legacy).double()
    migrated64 = _block(fused, seed=7).double()
    migrated64.load_state_dict(
        fold_legacy_block_biases(legacy64.state_dict()), strict=True
    )
    with torch.no_grad():
        expected64 = legacy64(x.double(), index, mask, edge_latent.double())
        actual64 = migrated64(x.double(), index, mask, edge_latent.double())
    torch.testing.assert_close(actual64, expected64, rtol=1e-12, atol=1e-12)


def test_fold_legacy_block_biases_touches_only_the_two_keys_per_block():
    processor = NeighborhoodAttentionProcessor(2, 32, 4, edge_input_dim=4, edge_dim=8)
    state = processor.state_dict()
    legacy = dict(state)
    for prefix in ("blocks.0", "blocks.1"):
        legacy[f"{prefix}.value.bias"] = torch.randn(32)
        legacy[f"{prefix}.edge_bias.bias"] = torch.randn(4)
    folded = fold_legacy_block_biases(legacy)
    assert set(folded) == set(state)
    assert len(legacy) - len(folded) == 4
    for key, value in state.items():
        if key.endswith("update_mlp.0.bias"):
            assert not torch.equal(folded[key], value)
        else:
            assert torch.equal(folded[key], value), key
    # a state dict without the legacy keys passes through unchanged
    again = fold_legacy_block_biases(folded)
    assert set(again) == set(folded)
    assert all(torch.equal(again[key], folded[key]) for key in folded)


def test_difference_values_vanish_for_a_constant_field():
    """With identical neighbors the difference message is exactly zero."""
    block = _block(False)
    x, index, mask, edge_latent = _inputs()
    constant = x[:1].expand_as(x).contiguous()
    normalized, query, key, value = block._projections(constant)
    # Reproduce the message path only: aggregated minus own value.
    n, k = index.shape
    flat = index.reshape(-1)
    gathered = value.index_select(0, flat).view(n, k, 4, 8)
    weights = torch.softmax(
        torch.zeros(n, 4, k).masked_fill(~mask[:, None], -torch.inf), -1
    )
    aggregated = torch.einsum("nhk,nkhd->nhd", weights, gathered).reshape(n, 32)
    torch.testing.assert_close(
        aggregated - value, torch.zeros_like(value), atol=1e-6, rtol=0
    )


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
    out, edge_out, lse = neighborhood_attention(
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
            neighborhood_attention(
                q, k, v, bias, edge_latent, index, mask, 8**-0.5, backend="torch"
            )
        )
        == 2
    )
    out.sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None


def test_processor_checkpoints_match_plain_execution():
    torch.manual_seed(2)
    processor = NeighborhoodAttentionProcessor(
        2, 32, 4, edge_input_dim=4, edge_dim=8, checkpoint_segments=2
    ).train()
    x, index, mask, _ = _inputs()
    neighborhood = LocalNeighborhood(index, mask, torch.randn(40, 6, 4))
    y = processor(x.clone().requires_grad_(), neighborhood)
    assert y.shape == (40, 32)
    y.sum().backward()
    processor.eval()
    with torch.no_grad():
        z = processor(x, neighborhood)
    torch.testing.assert_close(y.detach(), z, rtol=1e-5, atol=1e-6)


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
