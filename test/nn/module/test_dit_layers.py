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

from physicsnemo.nn.module.dit_layers import (
    ConvDetokenizer,
    DiTBlock,
    Natten2DSelfAttention,
    ProjReshape2DDetokenizer,
    TimmSelfAttention,
)
from physicsnemo.nn.module.rope import (
    apply_rotary_pos_emb,
    build_axial_rope_cos_sin_2d,
)
from test import common
from test.conftest import requires_module

# --- TimmSelfAttention tests ---


def test_timm_self_attention_is_causal(device):
    # Default attention is bidirectional; the causal flag is stored on the module.
    bidirectional = TimmSelfAttention(hidden_size=32, num_heads=4).to(device)
    assert bidirectional.is_causal is False

    causal = TimmSelfAttention(hidden_size=32, num_heads=4, is_causal=True).to(device)
    assert causal.is_causal is True

    # Perturbing the final position must not change earlier outputs under a causal
    # mask (each position only attends to itself and earlier ones), while the
    # bidirectional module leaks the perturbation into every position.
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32, device=device)
    perturbed = x.clone()
    perturbed[:, -1] += 100.0
    with torch.no_grad():
        before = causal(x)
        after = causal(perturbed)
        bi_before = bidirectional(x)
        bi_after = bidirectional(perturbed)

    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1.0e-6)
    assert not torch.allclose(before[:, -1], after[:, -1])
    assert not torch.allclose(bi_before[:, :-1], bi_after[:, :-1])

    # attn_mask and is_causal=True are mutually exclusive (an explicit mask plus
    # the causal flag is an illegal combination for the underlying SDPA call).
    mask = torch.ones(6, 6, dtype=torch.bool, device=device).tril()
    with pytest.raises(ValueError):
        causal(x, attn_mask=mask)


# --- DiTBlock tests ---


@torch.no_grad()
def test_ditblock_forward_accuracy_timm(device):
    if device == "cpu":
        pytest.skip("CUDA only")

    torch.manual_seed(0)
    hidden_size = 128
    num_heads = 4
    B, T = 2, 16

    block = (
        DiTBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_backend="timm",
            layernorm_backend="torch",
        )
        .to(device)
        .eval()
    )

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)

    y = block(x, c)
    assert y.shape == (B, T, hidden_size)

    assert common.validate_tensor_accuracy(
        y,
        file_name="nn/module/data/ditblock_timm_output.pth",
    )


@torch.no_grad()
@requires_module(["natten"])
def test_ditblock_forward_accuracy_natten(device, pytestconfig):
    if device == "cpu":
        pytest.skip("natten not available on CPU")

    torch.manual_seed(0)
    hidden_size = 64
    num_heads = 4
    B, H, W = 2, 8, 8
    T = H * W

    block = (
        DiTBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_backend="natten2d",
            layernorm_backend="torch",
            attn_kernel=3,
        )
        .to(device)
        .eval()
    )

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)

    y = block(x, c, attn_kwargs={"latent_hw": (H, W)})
    assert y.shape == (B, T, hidden_size)

    assert common.validate_tensor_accuracy(
        y,
        file_name="nn/module/data/ditblock_natten_output.pth",
    )


@torch.no_grad()
@requires_module(["natten"])
def test_ditblock_natten_rope_and_mask_token_forward(device, pytestconfig):
    """natten2d_rope runs through DiTBlock; the mask token alters only the
    masked tokens' contribution and is a no-op when no tokens are invalid."""
    if device == "cpu":
        pytest.skip("natten not available on CPU")

    torch.manual_seed(0)
    hidden_size, num_heads = 64, 4
    B, H, W = 2, 8, 8
    T = H * W

    block = (
        DiTBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_backend="natten2d_rope",
            layernorm_backend="torch",
            attn_kernel=3,
            latent_hw=(H, W),
            use_mask_token=True,
        )
        .to(device)
        .eval()
    )
    # Confirm the RoPE backend was selected and exposes its precomputed tables.
    assert hasattr(block.attention, "rope_cos")
    assert block.attention.rope_cos.shape == (H, W, hidden_size // num_heads)

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)

    # An all-valid mask must match passing no mask at all (mask token init zero
    # would also match, but we trained none here; pass explicit zeros).
    all_valid = torch.zeros(T, dtype=torch.bool, device=device)
    y_none = block(x, c, attn_kwargs={"latent_hw": (H, W)})
    y_valid = block(
        x, c, attn_kwargs={"latent_hw": (H, W), "invalid_token_mask": all_valid}
    )
    assert y_none.shape == (B, T, hidden_size)
    assert torch.allclose(y_none, y_valid, atol=1e-5)

    # Marking tokens invalid (with a non-zero mask token) must change the output.
    torch.nn.init.normal_(block.attention.mask_token)
    invalid = all_valid.clone()
    invalid[[0, 9, 33]] = True
    y_masked = block(
        x, c, attn_kwargs={"latent_hw": (H, W), "invalid_token_mask": invalid}
    )
    assert not torch.allclose(y_none, y_masked, atol=1e-5)

    # A per-sample (B, T) mask applies a distinct pattern to each sample: mask
    # sample 0 only and the result must match the shared (T,) mask on sample 0
    # while sample 1 is left unmasked (equals the no-mask output).
    per_sample = torch.zeros(B, T, dtype=torch.bool, device=device)
    per_sample[0] = invalid
    y_per_sample = block(
        x, c, attn_kwargs={"latent_hw": (H, W), "invalid_token_mask": per_sample}
    )
    assert torch.allclose(y_per_sample[0], y_masked[0], atol=1e-5)
    assert torch.allclose(y_per_sample[1], y_none[1], atol=1e-5)


@torch.no_grad()
@requires_module(["transformer_engine"])
def test_ditblock_forward_accuracy_transformer_engine(device, pytestconfig):
    if device == "cpu":
        pytest.skip("Skipping DiT checkpoint test on CPU since TE is CUDA-only")

    torch.manual_seed(0)
    hidden_size = 128
    num_heads = 8
    B, T = 2, 32

    block = (
        DiTBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_backend="transformer_engine",
            layernorm_backend="torch",
        )
        .to(device)
        .eval()
    )

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)

    y = block(x, c)
    assert y.shape == (B, T, hidden_size)

    assert common.validate_tensor_accuracy(
        y,
        file_name="nn/module/data/ditblock_te_output.pth",
    )


def test_ditblock_exceptions(device):
    hidden_size = 32
    num_heads = 4
    B, T = 2, 8
    block = (
        DiTBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_backend="timm",
            layernorm_backend="torch",
            intermediate_dropout=True,
        )
        .to(device)
        .train()
    )

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)
    with pytest.raises(ValueError):
        _ = block(x, c, p_dropout=torch.tensor([0.5], device=device))

    try:
        import natten  # noqa: F401
    except Exception:
        pytest.skip("natten not available; skipping natten exception subtest")

    hidden_size = 64
    num_heads = 4
    B, T = 2, 64
    nat_block = DiTBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        attention_backend="natten2d",
        layernorm_backend="torch",
        attn_kernel=3,
    ).to(device)

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)
    with pytest.raises(TypeError):
        _ = nat_block(x, c)  # missing required attn_kwargs: latent_hw


def test_ditblock_intermediate_dropout_scalar_and_per_sample(device):
    torch.manual_seed(123)
    hidden_size = 64
    num_heads = 4
    B, T = 3, 16
    block = DiTBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        attention_backend="timm",
        layernorm_backend="torch",
        intermediate_dropout=True,
    ).to(device)

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)

    # Eval mode: dropout should be a no-op regardless of p_dropout
    block.eval()
    y_no = block(x, c, p_dropout=None)
    y_ps = block(x, c, p_dropout=0.7)
    assert torch.allclose(y_no, y_ps, atol=0.0)

    # Train mode: deterministic under fixed seed
    block.train()
    torch.manual_seed(999)
    y1 = block(x, c, p_dropout=0.5)
    torch.manual_seed(999)
    y2 = block(x, c, p_dropout=0.5)
    assert torch.allclose(y1, y2, atol=0.0)

    # Per-sample dropout requires p shaped [B]
    p = torch.tensor([0.1] * B, device=device)
    _ = block(x, c, p_dropout=p)  # should run


# --- RoPE helper / mask-token tests (CPU-friendly: no NATTEN kernel needed) ---


@torch.no_grad()
def test_build_axial_rope_cos_sin_2d_shape_and_validation():
    h, w, head_dim = 5, 7, 16
    cos, sin = build_axial_rope_cos_sin_2d(h, w, head_dim, theta=10000.0)
    assert cos.shape == (h, w, head_dim)
    assert sin.shape == (h, w, head_dim)
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32
    # Adjacent channels (2i, 2i+1) share a frequency, so their cos/sin match.
    assert torch.allclose(cos[..., 0::2], cos[..., 1::2])
    assert torch.allclose(sin[..., 0::2], sin[..., 1::2])
    # head_dim must be divisible by 4.
    with pytest.raises(ValueError):
        build_axial_rope_cos_sin_2d(h, w, head_dim=6)


@torch.no_grad()
def test_rope_rotation_matches_complex_rotation_and_preserves_norm():
    """The rotate-half formulation must equal the canonical per-pair 2D rotation
    (real-valued rotation matrix / complex multiply) and preserve pair norms."""
    torch.manual_seed(0)
    h, w, head_dim = 4, 6, 16
    cos, sin = build_axial_rope_cos_sin_2d(h, w, head_dim, theta=10000.0)
    q = torch.randn(2, 3, h, w, head_dim)  # (B, heads, h, w, head_dim)

    q_rot = apply_rotary_pos_emb(q, cos, sin)

    # Canonical rotation on each adjacent (even, odd) pair:
    #   even' = even*cos - odd*sin ;  odd' = even*sin + odd*cos
    cos_pair = cos[..., 0::2]
    sin_pair = sin[..., 0::2]
    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    expected_even = q_even * cos_pair - q_odd * sin_pair
    expected_odd = q_even * sin_pair + q_odd * cos_pair

    assert torch.allclose(q_rot[..., 0::2], expected_even, atol=1e-6)
    assert torch.allclose(q_rot[..., 1::2], expected_odd, atol=1e-6)

    # A rotation preserves the norm of each channel pair.
    pair_norm_in = q_even.square() + q_odd.square()
    pair_norm_out = q_rot[..., 0::2].square() + q_rot[..., 1::2].square()
    assert torch.allclose(pair_norm_in, pair_norm_out, atol=1e-5)


@torch.no_grad()
def test_mask_token_arithmetic_matches_where():
    """The mask-token splice (x*(1-alpha) + mask_token*alpha) must be identical
    to torch.where for a boolean mask."""
    torch.manual_seed(0)
    hidden_size, num_heads = 32, 4
    B, H, W = 2, 4, 4
    N = H * W

    attn = Natten2DSelfAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        attn_kernel=3,
        use_mask_token=True,
    )
    # Give the mask token a non-trivial value so the splice is observable.
    torch.nn.init.normal_(attn.mask_token)

    x = torch.randn(B, N, hidden_size)
    invalid = torch.zeros(N, dtype=torch.bool)
    invalid[[1, 5, 12]] = True

    spliced = attn._apply_mask_token(x, invalid)
    expected = torch.where(invalid.view(1, -1, 1), attn.mask_token.to(x.dtype), x)
    assert torch.equal(spliced, expected)

    # When the module has no mask token, the input is returned unchanged.
    attn_no_mask = Natten2DSelfAttention(hidden_size, num_heads, attn_kernel=3)
    assert attn_no_mask.mask_token is None
    assert torch.equal(attn_no_mask._apply_mask_token(x, invalid), x)


# --- ConvDetokenizer tests ---


@torch.no_grad()
def test_conv_detokenizer_shape():
    """Output shape matches ProjReshape2DDetokenizer."""
    torch.manual_seed(0)
    B, H, W, P = 2, 16, 16, 4
    out_channels, hidden_size = 3, 64
    input_size = (H, W)
    patch_size = (P, P)
    L = (H // P) * (W // P)

    x_tokens = torch.randn(B, L, hidden_size)
    c = torch.randn(B, hidden_size)

    ref = ProjReshape2DDetokenizer(input_size, patch_size, out_channels, hidden_size)
    ref.initialize_weights()
    det = ConvDetokenizer(input_size, patch_size, out_channels, hidden_size)
    det.initialize_weights()

    assert det(x_tokens, c).shape == ref(x_tokens, c).shape == (B, out_channels, H, W)


@torch.no_grad()
def test_conv_detokenizer_zero_init_identity():
    """At init the conv residual is zero, so ConvDetokenizer == ProjReshape2DDetokenizer."""
    torch.manual_seed(0)
    B, H, W, P = 2, 16, 16, 4
    out_channels, hidden_size = 3, 64
    input_size = (H, W)
    patch_size = (P, P)
    L = (H // P) * (W // P)

    x_tokens = torch.randn(B, L, hidden_size)
    c = torch.randn(B, hidden_size)

    ref = ProjReshape2DDetokenizer(input_size, patch_size, out_channels, hidden_size)
    ref.initialize_weights()

    det = ConvDetokenizer(input_size, patch_size, out_channels, hidden_size)
    det.initialize_weights()
    # Copy inner proj weights so the two modules are identical pre-smoothing.
    det.proj.load_state_dict(ref.state_dict())

    assert torch.allclose(det(x_tokens, c), ref(x_tokens, c))
