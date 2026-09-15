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

import pytest
import torch

from physicsnemo.nn import (
    GALE,
    GALE_FA,
    GALEBlock,
)
from test.conftest import requires_module

# =============================================================================
# GALE (Geometry-Aware Latent Embeddings) Attention Tests
# =============================================================================


def test_gale_forward_basic(device):
    """Test GALE attention layer forward pass without context."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    slice_num = 8
    batch_size = 2
    n_tokens = 100

    gale = GALE(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=dim_head,  # Must match dim_head for cross attention
    ).to(device)

    # Single input tensor wrapped in tuple
    x = torch.randn(batch_size, n_tokens, dim).to(device)

    outputs = gale((x,), context=None)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, dim)
    assert not torch.isnan(outputs[0]).any()


def test_gale_forward_with_context(device):
    """Test GALE attention layer forward pass with cross-attention context."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    slice_num = 8
    batch_size = 2
    n_tokens = 100
    context_tokens = 32
    context_dim = dim_head

    gale = GALE(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
    ).to(device)

    x = torch.randn(batch_size, n_tokens, dim).to(device)
    context = torch.randn(batch_size, heads, context_tokens, context_dim).to(device)

    outputs = gale((x,), context=context)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, dim)
    assert not torch.isnan(outputs[0]).any()


def test_gale_forward_multiple_inputs(device):
    """Test GALE attention layer with multiple input tensors."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    slice_num = 8
    batch_size = 2
    n_tokens_1 = 100
    n_tokens_2 = 150
    context_dim = dim_head

    gale = GALE(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
    ).to(device)

    x1 = torch.randn(batch_size, n_tokens_1, dim).to(device)
    x2 = torch.randn(batch_size, n_tokens_2, dim).to(device)

    outputs = gale((x1, x2), context=None)

    assert len(outputs) == 2
    assert outputs[0].shape == (batch_size, n_tokens_1, dim)
    assert outputs[1].shape == (batch_size, n_tokens_2, dim)
    assert not torch.isnan(outputs[0]).any()
    assert not torch.isnan(outputs[1]).any()


# =============================================================================
# GALE_FA Attention Tests
# =============================================================================


@requires_module("transformer_engine>=2.14.0")
@pytest.mark.parametrize("attention_type", ["GALE", "GALE_FA"])
def test_gale_te_uses_only_concrete_output_dropout(device, attention_type):
    """Test TE attention leaves dropout to the shared ConcreteDropout layer."""
    if device == "cpu":
        pytest.skip("Transformer Engine requires CUDA")

    attention_cls = GALE if attention_type == "GALE" else GALE_FA
    attention = attention_cls(
        dim=64,
        heads=4,
        dim_head=16,
        dropout=0.25,
        use_te=True,
        context_dim=16,
        concrete_dropout=True,
    ).to(device)

    assert attention.attn_fn.attention_dropout == 0.0
    assert torch.allclose(attention.out_dropout.p, torch.tensor(0.25, device=device))


def test_gale_fa_forward_basic(device):
    """Test GALE_FA attention layer pass without context."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    n_global_queries = 8
    batch_size = 2
    n_tokens = 100

    gale_fa = GALE_FA(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        n_global_queries=n_global_queries,
        use_te=False,
        context_dim=dim_head,  # Must match dim_head for cross attention
    ).to(device)

    # Single input tensor wrapped in tuple
    x = torch.randn(batch_size, n_tokens, dim).to(device)

    outputs = gale_fa((x,), context=None)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, dim)
    assert not torch.isnan(outputs[0]).any()


def test_gale_fa_forward_with_context(device):
    """Test GALE_FA attention layer with cross-attention context."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    n_global_queries = 8
    batch_size = 2
    n_tokens = 100
    context_tokens = 32
    context_dim = dim_head

    gale_fa = GALE_FA(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        n_global_queries=n_global_queries,
        use_te=False,
        context_dim=context_dim,
    ).to(device)

    x = torch.randn(batch_size, n_tokens, dim).to(device)
    context = torch.randn(batch_size, heads, context_tokens, context_dim).to(device)

    outputs = gale_fa((x,), context=context)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, dim)
    assert not torch.isnan(outputs[0]).any()


def test_gale_fa_forward_multiple_inputs(device):
    """Test GALE_FA attention layer with multiple input tensors."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    n_global_queries = 8
    batch_size = 2
    n_tokens_1 = 100
    n_tokens_2 = 150
    context_dim = dim_head

    gale_fa = GALE_FA(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        n_global_queries=n_global_queries,
        use_te=False,
        context_dim=context_dim,
    ).to(device)

    x1 = torch.randn(batch_size, n_tokens_1, dim).to(device)
    x2 = torch.randn(batch_size, n_tokens_2, dim).to(device)

    outputs = gale_fa((x1, x2), context=None)

    assert len(outputs) == 2
    assert outputs[0].shape == (batch_size, n_tokens_1, dim)
    assert outputs[1].shape == (batch_size, n_tokens_2, dim)
    assert not torch.isnan(outputs[0]).any()
    assert not torch.isnan(outputs[1]).any()


# =============================================================================
# concat_project state mixing mode
# =============================================================================


def test_gale_concat_project_forward(device):
    """Test GALE with state_mixing_mode='concat_project' and cross-attention context."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    slice_num = 8
    batch_size = 2
    n_tokens = 100
    context_tokens = 32
    context_dim = dim_head

    gale = GALE(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
        state_mixing_mode="concat_project",
    ).to(device)

    x = torch.randn(batch_size, n_tokens, dim).to(device)
    context = torch.randn(batch_size, heads, context_tokens, context_dim).to(device)

    outputs = gale((x,), context=context)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, dim)
    assert not torch.isnan(outputs[0]).any()


def test_gale_fa_concat_project_forward(device):
    """Test GALE_FA with state_mixing_mode='concat_project' and cross-attention context."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    n_global_queries = 8
    batch_size = 2
    n_tokens = 100
    context_tokens = 32
    context_dim = dim_head

    gale_fa = GALE_FA(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        n_global_queries=n_global_queries,
        use_te=False,
        context_dim=context_dim,
        state_mixing_mode="concat_project",
    ).to(device)

    x = torch.randn(batch_size, n_tokens, dim).to(device)
    context = torch.randn(batch_size, heads, context_tokens, context_dim).to(device)

    outputs = gale_fa((x,), context=context)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, dim)
    assert not torch.isnan(outputs[0]).any()


# =============================================================================
# GALE_FA context placement and per-source gated blending
# =============================================================================


def test_gale_fa_context_placement_default_unchanged(device):
    """Test that the context_placement option adds no parameters and keeps the default path."""
    torch.manual_seed(42)
    gale_fa_implicit = GALE_FA(dim=64, heads=4, dim_head=16, context_dim=16).to(device)
    torch.manual_seed(42)
    gale_fa_points = GALE_FA(
        dim=64, heads=4, dim_head=16, context_dim=16, context_placement="points"
    ).to(device)
    gale_fa_implicit.eval()
    gale_fa_points.eval()

    assert gale_fa_implicit.context_placement == "points"
    assert gale_fa_implicit.context_source_dims is None
    implicit_state = gale_fa_implicit.state_dict()
    points_state = gale_fa_points.state_dict()
    assert set(implicit_state) == set(points_state)
    for name, tensor in implicit_state.items():
        assert torch.equal(tensor, points_state[name])

    x = torch.randn(2, 100, 64).to(device)
    context = torch.randn(2, 4, 32, 16).to(device)
    assert torch.equal(
        gale_fa_implicit((x,), context)[0], gale_fa_points((x,), context)[0]
    )


def test_gale_fa_latents_placement_forward(device):
    """Test the latent-bottleneck context read: shape preserved, context changes the output."""
    torch.manual_seed(42)
    gale_fa = GALE_FA(
        dim=64,
        heads=4,
        dim_head=16,
        n_global_queries=8,
        context_dim=16,
        context_placement="latents",
    ).to(device)
    gale_fa.eval()

    x = torch.randn(2, 100, 64).to(device)
    context = torch.randn(2, 4, 32, 16).to(device)

    out_ctx = gale_fa((x,), context)
    out_plain = gale_fa((x,), None)
    assert len(out_ctx) == 1
    assert out_ctx[0].shape == (2, 100, 64)
    assert not torch.isnan(out_ctx[0]).any()
    assert not torch.allclose(out_ctx[0], out_plain[0])


def test_gale_fa_latents_none_context_matches_self_path(device):
    """Test that the latents placement without a context matches the points placement."""
    torch.manual_seed(42)
    gale_fa_points = GALE_FA(dim=64, heads=4, dim_head=16, context_dim=16).to(device)
    torch.manual_seed(42)
    gale_fa_latents = GALE_FA(
        dim=64, heads=4, dim_head=16, context_dim=16, context_placement="latents"
    ).to(device)
    gale_fa_points.eval()
    gale_fa_latents.eval()

    x = torch.randn(2, 100, 64).to(device)
    assert torch.equal(gale_fa_points((x,), None)[0], gale_fa_latents((x,), None)[0])


def test_gale_fa_latents_gradient_flow(device):
    """Test gradient flow to the cross projections through the latent context read."""
    torch.manual_seed(42)
    gale_fa = GALE_FA(
        dim=32, heads=4, dim_head=8, context_dim=8, context_placement="latents"
    ).to(device)
    x = torch.randn(2, 20, 32, device=device, requires_grad=True)
    context = torch.randn(2, 4, 6, 8, device=device, requires_grad=True)

    out = gale_fa((x,), context)
    out[0].sum().backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert context.grad is not None
    assert not torch.isnan(context.grad).any()
    for proj in (gale_fa.cross_q, gale_fa.cross_k, gale_fa.cross_v):
        assert proj.weight.grad is not None
        assert not torch.isnan(proj.weight.grad).any()
    assert gale_fa.state_mixing.grad is not None


@pytest.mark.parametrize("context_placement", ["points", "latents"])
def test_gale_fa_source_gate_forward(device, context_placement):
    """Test the per-source gated blend at both context placements."""
    torch.manual_seed(42)
    gale_fa = GALE_FA(
        dim=64,
        heads=4,
        dim_head=16,
        n_global_queries=8,
        context_dim=32,
        context_placement=context_placement,
        context_source_dims=(24, 8),
    ).to(device)
    gale_fa.eval()

    x = torch.randn(2, 100, 64).to(device)
    context = torch.randn(2, 4, 10, 32).to(device)

    out_ctx = gale_fa((x,), context)
    out_plain = gale_fa((x,), None)
    assert out_ctx[0].shape == (2, 100, 64)
    assert not torch.isnan(out_ctx[0]).any()
    assert not torch.allclose(out_ctx[0], out_plain[0])


def test_gale_fa_source_gate_uniform_at_init(device):
    """Test that zero-initialized gate logits give a uniform blend over the streams."""
    torch.manual_seed(42)
    gale_fa = GALE_FA(
        dim=64,
        heads=4,
        dim_head=16,
        context_dim=32,
        context_placement="latents",
        context_source_dims=(24, 8),
    ).to(device)

    assert gale_fa.context_source_dims == (24, 8)
    assert gale_fa.source_gate_logits.shape == (3, 16)
    assert torch.equal(gale_fa.source_gate_logits, torch.zeros(3, 16, device=device))
    alpha = torch.softmax(gale_fa.source_gate_logits, dim=0)
    assert torch.allclose(alpha, torch.full((3, 16), 1.0 / 3.0, device=device))


@pytest.mark.parametrize("context_placement", ["points", "latents"])
def test_gale_fa_source_gate_gradient_flow(device, context_placement):
    """Test per-source gradient flow: each value projection and each gate row."""
    torch.manual_seed(42)
    gale_fa = GALE_FA(
        dim=32,
        heads=4,
        dim_head=8,
        context_dim=16,
        context_placement=context_placement,
        context_source_dims=(10, 6),
    ).to(device)
    x = torch.randn(2, 20, 32, device=device, requires_grad=True)
    context = torch.randn(2, 4, 6, 16, device=device, requires_grad=True)

    out = gale_fa((x,), context)
    out[0].sum().backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert context.grad is not None
    assert not torch.isnan(context.grad).any()
    for proj in (gale_fa.cross_q, gale_fa.cross_k, *gale_fa.cross_v_sources):
        assert proj.weight.grad is not None
        assert not torch.isnan(proj.weight.grad).any()
        assert proj.weight.grad.abs().sum() > 0
    gate_grad = gale_fa.source_gate_logits.grad
    assert gate_grad is not None
    assert (gate_grad.abs().sum(dim=1) > 0).all()


@pytest.mark.parametrize("context_placement", ["points", "latents"])
def test_gale_fa_single_source_gate_matches_state_mixing_at_init(
    device, context_placement
):
    """Test the documented single-source semantics at initialization.

    With one source, zero gate logits blend the two streams 50/50, exactly like
    the "weighted" state mixing at initialization (sigmoid(0) = 0.5), so the two
    blends agree once the value projections share weights.
    """
    torch.manual_seed(42)
    gale_fa_mixed = GALE_FA(
        dim=64,
        heads=4,
        dim_head=16,
        context_dim=16,
        context_placement=context_placement,
    ).to(device)
    torch.manual_seed(42)
    gale_fa_gated = GALE_FA(
        dim=64,
        heads=4,
        dim_head=16,
        context_dim=16,
        context_placement=context_placement,
        context_source_dims=(16,),
    ).to(device)
    gale_fa_mixed.eval()
    gale_fa_gated.eval()

    gale_fa_gated.load_state_dict(gale_fa_mixed.state_dict(), strict=False)
    with torch.no_grad():
        gale_fa_gated.cross_v_sources[0].weight.copy_(gale_fa_mixed.cross_v.weight)
        gale_fa_gated.cross_v_sources[0].bias.copy_(gale_fa_mixed.cross_v.bias)

    x = torch.randn(2, 100, 64).to(device)
    context = torch.randn(2, 4, 32, 16).to(device)
    out_mixed = gale_fa_mixed((x,), context)[0]
    out_gated = gale_fa_gated((x,), context)[0]
    assert torch.allclose(out_mixed, out_gated, atol=1e-6, rtol=1e-6)


def test_gale_fa_context_placement_invalid():
    """Test that construction rejects an unknown context placement."""
    with pytest.raises(ValueError, match="context_placement"):
        GALE_FA(dim=64, heads=4, dim_head=16, context_placement="global")


def test_gale_fa_context_source_dims_invalid():
    """Test that construction rejects malformed context source widths."""
    with pytest.raises(ValueError, match="must sum to context_dim"):
        GALE_FA(
            dim=64, heads=4, dim_head=16, context_dim=16, context_source_dims=(10, 4)
        )
    with pytest.raises(ValueError, match="must sum to context_dim"):
        GALE_FA(dim=64, heads=4, dim_head=16, context_source_dims=(16,))
    with pytest.raises(ValueError, match="non-empty tuple of positive"):
        GALE_FA(dim=64, heads=4, dim_head=16, context_dim=16, context_source_dims=())
    with pytest.raises(ValueError, match="non-empty tuple of positive"):
        GALE_FA(
            dim=64, heads=4, dim_head=16, context_dim=16, context_source_dims=(20, -4)
        )


@requires_module("transformer_engine>=2.14.0")
@pytest.mark.parametrize(
    "context_placement,context_source_dims",
    [("latents", None), ("points", (10, 6)), ("latents", (10, 6))],
)
def test_gale_fa_te_context_options_forward_backward(
    device, context_placement, context_source_dims
):
    """Test the TE backend across the context placement and blending options."""
    if device == "cpu":
        pytest.skip("Transformer Engine requires CUDA")

    torch.manual_seed(42)
    gale_fa = GALE_FA(
        dim=64,
        heads=4,
        dim_head=16,
        n_global_queries=7,
        use_te=True,
        context_dim=16,
        context_placement=context_placement,
        context_source_dims=context_source_dims,
    ).to(device)
    x = torch.randn(2, 19, 64, device=device, requires_grad=True)
    context = torch.randn(2, 4, 5, 16, device=device, requires_grad=True)

    out = gale_fa((x,), context)
    assert out[0].shape == x.shape
    assert not torch.isnan(out[0]).any()

    out[0].sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert context.grad is not None
    assert not torch.isnan(context.grad).any()


# =============================================================================
# GALEBlock Tests
# =============================================================================


@pytest.mark.parametrize("attention_type", ["GALE", "GALE_FA"])
def test_gale_block_forward(device, attention_type):
    """Test GALEBlock transformer block forward pass (GALE and GALE_FA)."""
    torch.manual_seed(42)

    hidden_dim = 64
    n_head = 4
    batch_size = 2
    n_tokens = 100
    slice_num = 8
    context_dim = hidden_dim // n_head

    block = GALEBlock(
        num_heads=n_head,
        hidden_dim=hidden_dim,
        dropout=0.0,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
        attention_type=attention_type,
    ).to(device)

    x = torch.randn(batch_size, n_tokens, hidden_dim).to(device)
    context = torch.randn(batch_size, n_head, slice_num, context_dim).to(device)

    outputs = block((x,), global_context=context)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, hidden_dim)
    assert not torch.isnan(outputs[0]).any()


@pytest.mark.parametrize("attention_type", ["GALE", "GALE_FA"])
def test_gale_block_multiple_inputs(device, attention_type):
    """Test GALEBlock with multiple input tensors and attention type (GALE and GALE_FA)."""
    torch.manual_seed(42)

    hidden_dim = 64
    n_head = 4
    batch_size = 2
    n_tokens_1 = 100
    n_tokens_2 = 150
    slice_num = 8
    context_dim = hidden_dim // n_head

    block = GALEBlock(
        num_heads=n_head,
        hidden_dim=hidden_dim,
        dropout=0.0,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
        attention_type=attention_type,
    ).to(device)

    x1 = torch.randn(batch_size, n_tokens_1, hidden_dim).to(device)
    x2 = torch.randn(batch_size, n_tokens_2, hidden_dim).to(device)
    context = torch.randn(batch_size, n_head, slice_num, context_dim).to(device)

    outputs = block((x1, x2), global_context=context)

    assert len(outputs) == 2
    assert outputs[0].shape == (batch_size, n_tokens_1, hidden_dim)
    assert outputs[1].shape == (batch_size, n_tokens_2, hidden_dim)


@pytest.mark.parametrize("attention_type", ["GALE", "GALE_FA"])
def test_gale_block_concat_project(device, attention_type):
    """Test GALEBlock with state_mixing_mode='concat_project'."""
    torch.manual_seed(42)

    hidden_dim = 64
    n_head = 4
    batch_size = 2
    n_tokens = 100
    slice_num = 8
    context_dim = hidden_dim // n_head

    block = GALEBlock(
        num_heads=n_head,
        hidden_dim=hidden_dim,
        dropout=0.0,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
        attention_type=attention_type,
        state_mixing_mode="concat_project",
    ).to(device)

    x = torch.randn(batch_size, n_tokens, hidden_dim).to(device)
    context = torch.randn(batch_size, n_head, slice_num, context_dim).to(device)

    outputs = block((x,), global_context=context)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, hidden_dim)
    assert not torch.isnan(outputs[0]).any()


def test_gale_block_context_options_forward(device):
    """Test that GALEBlock forwards the context options to its GALE_FA attention."""
    torch.manual_seed(42)
    block_default = GALEBlock(
        num_heads=4,
        hidden_dim=64,
        dropout=0.0,
        slice_num=8,
        context_dim=16,
        attention_type="GALE_FA",
    ).to(device)
    torch.manual_seed(42)
    block_latents = GALEBlock(
        num_heads=4,
        hidden_dim=64,
        dropout=0.0,
        slice_num=8,
        context_dim=16,
        attention_type="GALE_FA",
        context_placement="latents",
        context_source_dims=(10, 6),
    ).to(device)
    block_default.eval()
    block_latents.eval()

    assert block_latents.Attn.context_placement == "latents"
    assert block_latents.Attn.context_source_dims == (10, 6)

    x = torch.randn(2, 100, 64).to(device)
    context = torch.randn(2, 4, 8, 16).to(device)
    out_default = block_default((x,), global_context=context)
    out_latents = block_latents((x,), global_context=context)
    assert out_latents[0].shape == (2, 100, 64)
    assert not torch.isnan(out_latents[0]).any()
    assert not torch.allclose(out_default[0], out_latents[0])


def test_gale_block_context_options_default_unchanged(device):
    """Test that omitting the context options leaves GALEBlock unchanged."""
    torch.manual_seed(42)
    block_implicit = GALEBlock(
        num_heads=4,
        hidden_dim=64,
        dropout=0.0,
        slice_num=8,
        context_dim=16,
        attention_type="GALE_FA",
    ).to(device)
    torch.manual_seed(42)
    block_explicit = GALEBlock(
        num_heads=4,
        hidden_dim=64,
        dropout=0.0,
        slice_num=8,
        context_dim=16,
        attention_type="GALE_FA",
        context_placement="points",
        context_source_dims=None,
    ).to(device)
    block_implicit.eval()
    block_explicit.eval()

    implicit_state = block_implicit.state_dict()
    explicit_state = block_explicit.state_dict()
    assert set(implicit_state) == set(explicit_state)
    for name, tensor in implicit_state.items():
        assert torch.equal(tensor, explicit_state[name])

    x = torch.randn(2, 100, 64).to(device)
    context = torch.randn(2, 4, 8, 16).to(device)
    assert torch.equal(
        block_implicit((x,), global_context=context)[0],
        block_explicit((x,), global_context=context)[0],
    )


def test_gale_block_context_options_invalid():
    """Test that GALEBlock rejects context options with GALE and propagates GALE_FA errors."""
    with pytest.raises(ValueError, match="GALE_FA"):
        GALEBlock(
            num_heads=4,
            hidden_dim=64,
            dropout=0.0,
            context_dim=16,
            attention_type="GALE",
            context_placement="latents",
        )
    with pytest.raises(ValueError, match="GALE_FA"):
        GALEBlock(
            num_heads=4,
            hidden_dim=64,
            dropout=0.0,
            context_dim=16,
            attention_type="GALE",
            context_source_dims=(8, 8),
        )
    with pytest.raises(ValueError, match="context_placement"):
        GALEBlock(
            num_heads=4,
            hidden_dim=64,
            dropout=0.0,
            context_dim=16,
            attention_type="GALE_FA",
            context_placement="global",
        )
    with pytest.raises(ValueError, match="must sum to context_dim"):
        GALEBlock(
            num_heads=4,
            hidden_dim=64,
            dropout=0.0,
            context_dim=16,
            attention_type="GALE_FA",
            context_source_dims=(10, 4),
        )
