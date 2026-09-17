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
    GALE_FPP,
    FLAREPlusPlus,
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
# GALE_FPP Attention Tests
# =============================================================================


def test_gale_fpp_without_context_matches_standalone(device):
    """The context-free backend is exactly the standalone FLARE++ layer."""
    torch.manual_seed(42)
    standalone = FLAREPlusPlus(
        dim=32,
        heads=4,
        dim_head=8,
        n_global_queries=6,
        dropout=0.0,
    ).to(device)
    backend = GALE_FPP(
        dim=32,
        heads=4,
        dim_head=8,
        n_global_queries=6,
        dropout=0.0,
        context_dim=0,
    ).to(device)
    backend.load_state_dict(standalone.state_dict(), strict=True)
    x = torch.randn(2, 19, 32, device=device)

    torch.testing.assert_close(backend((x,), None)[0], standalone(x))


@pytest.mark.parametrize("state_mixing_mode", ["weighted", "concat_project"])
def test_gale_fpp_context_matches_reference(device, state_mixing_mode):
    """The GeoTransolver adapter adds only context attention and mixing."""
    torch.manual_seed(7)
    backend = GALE_FPP(
        dim=24,
        heads=3,
        dim_head=8,
        n_global_queries=5,
        context_dim=6,
        state_mixing_mode=state_mixing_mode,
    ).to(device)
    x = torch.randn(2, 17, 24, device=device)
    context = torch.randn(2, 3, 7, 6, device=device)

    actual = backend((x,), context)[0]
    self_output, physical_keys = backend._compute_attention(x)
    context_k, context_v = backend.context_kv(context).chunk(2, dim=-1)
    cross_output = torch.nn.functional.scaled_dot_product_attention(
        backend.cross_q(physical_keys),
        context_k,
        context_v,
        scale=backend.scale,
    )
    if state_mixing_mode == "weighted":
        weight = torch.sigmoid(backend.state_mixing)
        mixed = weight * self_output + (1.0 - weight) * cross_output
    else:
        mixed = backend.concat_project(torch.cat((self_output, cross_output), dim=-1))
    expected = backend._project_output(mixed)

    torch.testing.assert_close(actual, expected)


def test_gale_fpp_multiple_inputs_backward(device):
    """FLARE++ context attention supports multiple streams and gradients."""
    backend = GALE_FPP(
        dim=16,
        heads=2,
        dim_head=8,
        n_global_queries=4,
        context_dim=5,
    ).to(device)
    x1 = torch.randn(2, 11, 16, device=device, requires_grad=True)
    x2 = torch.randn(2, 13, 16, device=device, requires_grad=True)
    context = torch.randn(2, 2, 6, 5, device=device, requires_grad=True)

    outputs = backend((x1, x2), context)
    sum(output.square().mean() for output in outputs).backward()

    assert [output.shape for output in outputs] == [(2, 11, 16), (2, 13, 16)]
    assert x1.grad is not None
    assert x2.grad is not None
    assert context.grad is not None
    assert torch.isfinite(context.grad).all()


def test_gale_fpp_eval_is_rng_free(device):
    """GALE_FPP with context and Concrete Dropout has deterministic eval."""
    torch.manual_seed(45)
    attention = GALE_FPP(
        dim=32,
        heads=4,
        dim_head=8,
        n_global_queries=6,
        dropout=0.4,
        context_dim=8,
        concrete_dropout=True,
    ).to(device)
    x = torch.randn(2, 31, 32, device=device)
    context = torch.randn(2, 4, 6, 8, device=device)

    attention.eval()
    model_device = next(attention.parameters()).device
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = (
        torch.cuda.get_rng_state(model_device).clone()
        if model_device.type == "cuda"
        else None
    )
    with torch.no_grad():
        outputs_1 = attention((x,), context=context)
        outputs_2 = attention((x,), context=context)

    assert len(outputs_1) == len(outputs_2)
    assert all(
        torch.equal(output_1, output_2)
        for output_1, output_2 in zip(outputs_1, outputs_2, strict=True)
    )
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    if cuda_rng_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(model_device), cuda_rng_before)


def test_gale_fpp_torch_compile_fullgraph(device):
    """The context-enabled backend supports full-graph compilation."""
    backend = GALE_FPP(
        dim=16,
        heads=2,
        dim_head=8,
        n_global_queries=4,
        context_dim=5,
    ).to(device)
    x = torch.randn(2, 11, 16, device=device)
    context = torch.randn(2, 2, 6, 5, device=device)
    expected = backend((x,), context)
    compile_backend = "inductor" if str(device).startswith("cuda") else "aot_eager"
    compiled = torch.compile(backend, backend=compile_backend, fullgraph=True)

    torch.testing.assert_close(compiled((x,), context)[0], expected[0])


def test_gale_fpp_validates_configuration():
    """GALE_FPP rejects unsupported and inconsistent configurations."""
    with pytest.raises(ValueError, match="does not support Transformer Engine"):
        GALE_FPP(dim=16, heads=2, dim_head=8, use_te=True)
    with pytest.raises(ValueError, match="state_mixing_mode"):
        GALE_FPP(dim=16, heads=2, dim_head=8, state_mixing_mode="invalid")

    backend = GALE_FPP(dim=16, heads=2, dim_head=8, context_dim=0)
    with pytest.raises(ValueError, match="context_dim=0"):
        backend((torch.randn(1, 5, 16),), torch.randn(1, 2, 3, 4))


def test_gale_block_rejects_transolver_plus_with_flare_plus_plus():
    """The FLARE++ backend cannot be combined with Transolver++ slicing."""
    with pytest.raises(ValueError, match="GALE_FPP.*requires plus=False"):
        GALEBlock(
            num_heads=2,
            hidden_dim=16,
            dropout=0.0,
            plus=True,
            attention_type="GALE_FPP",
        )


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
# GALEBlock Tests
# =============================================================================


@pytest.mark.parametrize("attention_type", ["GALE", "GALE_FA", "GALE_FPP"])
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


@pytest.mark.parametrize("attention_type", ["GALE", "GALE_FA", "GALE_FPP"])
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


@pytest.mark.parametrize("attention_type", ["GALE", "GALE_FA", "GALE_FPP"])
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
