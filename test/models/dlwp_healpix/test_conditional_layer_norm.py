import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import torch
from _cln_reference import ConditionalLayerNormReference
from physicsnemo.models.dlwp_healpix.layers.normalization import ConditionalLayerNorm


def _make_old_cln(condition_shape, channel_depth, **kwargs):
    """Instantiate the reference (old) implementation."""
    return ConditionalLayerNormReference(
        condition_shape=condition_shape, channel_depth=channel_depth, **kwargs
    ).cuda()


def _make_new_cln(condition_shape, channel_depth, **kwargs):
    """Instantiate the optimized (new) implementation."""
    return ConditionalLayerNorm(
        condition_shape=condition_shape, channel_depth=channel_depth, **kwargs
    ).cuda()


def _copy_old_to_new(old_cln, new_cln):
    """Copy old separate gamma/beta MLP weights into new fused MLP using block-diagonal structure.

    Old: gamma_mlp and beta_mlp each have hidden_dims [h1, h2] and output C.
    New: gamma_beta_mlp has hidden_dims [2*h1, 2*h2] and output 2*C.

    Layer 0 (condition_shape → 2*h1): vertical cat of gamma/beta weights.
    Layer i>0 (2*h_{i-1} → 2*h_i or 2*C): block-diagonal [[gamma, 0], [0, beta]].
    Biases: always concatenated.
    """
    old_sd = old_cln.state_dict()

    # Collect Linear layer indices from the old gamma MLP
    gamma_linear_indices = sorted({
        int(k.split(".")[1])
        for k in old_sd if k.startswith("gamma_mlp.") and k.endswith(".weight")
    })
    first_layer_idx = gamma_linear_indices[0]

    new_sd = {}
    for key in old_sd:
        if key.startswith("norm."):
            new_sd[key] = old_sd[key]

    for idx in gamma_linear_indices:
        for param in ("weight", "bias"):
            gamma_val = old_sd[f"gamma_mlp.{idx}.{param}"]
            beta_val = old_sd[f"beta_mlp.{idx}.{param}"]
            fused_key = f"gamma_beta_mlp.{idx}.{param}"

            if param == "bias":
                new_sd[fused_key] = torch.cat([gamma_val, beta_val], dim=0)
            elif idx == first_layer_idx:
                # First layer: shared input dim, just cat along output dim
                new_sd[fused_key] = torch.cat([gamma_val, beta_val], dim=0)
            else:
                # Block-diagonal: [[gamma, 0], [0, beta]]
                out_old, in_old = gamma_val.shape
                zeros = torch.zeros_like(gamma_val)
                new_sd[fused_key] = torch.cat([
                    torch.cat([gamma_val, zeros], dim=1),
                    torch.cat([zeros, beta_val], dim=1),
                ], dim=0)

    new_cln.load_state_dict(new_sd)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_cond", [1, 2, 4])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("scale_center", [0.0, 1.0])
def test_old_vs_new_forward(n_cond, channels_last, scale_center):
    """Verify optimized CLN matches reference implementation with block-diagonal weight mapping."""
    C, H, W = 128, 16, 16
    cond_shape = 32
    B_nf = n_cond * 12

    torch.manual_seed(42)
    old_cln = _make_old_cln(cond_shape, C, scale_center=scale_center)

    new_cln = _make_new_cln(cond_shape, C, scale_center=scale_center)
    _copy_old_to_new(old_cln, new_cln)

    x = torch.randn(B_nf, C, H, W, device="cuda")
    cond = torch.randn(n_cond, cond_shape, device="cuda")

    if channels_last:
        x = x.to(memory_format=torch.channels_last)

    with torch.no_grad():
        out_old = old_cln(x, cond)
        out_new = new_cln(x, cond)

    assert out_old.shape == out_new.shape
    assert torch.allclose(out_old, out_new, atol=1e-5, rtol=1e-4), \
        f"Max diff: {(out_old - out_new).abs().max().item()}"

    if channels_last:
        assert out_new.is_contiguous(memory_format=torch.channels_last), \
            "Output should preserve channels_last format"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
def test_old_vs_new_backward(channels_last):
    """Verify gradients match between old and new implementations."""
    C, H, W = 64, 8, 8
    cond_shape = 16
    n_cond = 2
    B_nf = n_cond * 12

    torch.manual_seed(42)
    old_cln = _make_old_cln(cond_shape, C)
    new_cln = _make_new_cln(cond_shape, C)
    _copy_old_to_new(old_cln, new_cln)

    x_base = torch.randn(B_nf, C, H, W, device="cuda")
    cond_base = torch.randn(n_cond, cond_shape, device="cuda")

    if channels_last:
        x_base = x_base.to(memory_format=torch.channels_last)

    x_old = x_base.clone().detach().requires_grad_(True)
    cond_old = cond_base.clone().detach().requires_grad_(True)
    x_new = x_base.clone().detach().requires_grad_(True)
    cond_new = cond_base.clone().detach().requires_grad_(True)

    out_old = old_cln(x_old, cond_old)
    out_old.sum().backward()

    out_new = new_cln(x_new, cond_new)
    out_new.sum().backward()

    assert torch.allclose(x_old.grad, x_new.grad, atol=1e-4, rtol=1e-3), \
        f"Input grad max diff: {(x_old.grad - x_new.grad).abs().max().item()}"

    assert torch.allclose(cond_old.grad, cond_new.grad, atol=1e-4, rtol=1e-3), \
        f"Cond grad max diff: {(cond_old.grad - cond_new.grad).abs().max().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
def test_init_cln_to_zero_matches_layer_norm(channels_last):
    """With scale_center=1.0 and init_cln_to_zero=True, CLN should behave like plain LayerNorm."""
    C, H, W = 64, 8, 8
    n_cond = 2
    B_nf = n_cond * 12

    torch.manual_seed(42)
    cln = _make_new_cln(32, C, scale_center=1.0, init_cln_to_zero=True)
    plain_ln = torch.nn.LayerNorm(C, elementwise_affine=False).cuda()

    x = torch.randn(B_nf, C, H, W, device="cuda")
    cond = torch.randn(n_cond, 32, device="cuda")

    if channels_last:
        x = x.to(memory_format=torch.channels_last)

    with torch.no_grad():
        out_cln = cln(x, cond)
        x_nhwc = x.permute(0, 2, 3, 1)
        out_ln = plain_ln(x_nhwc).permute(0, 3, 1, 2)

    assert torch.allclose(out_cln, out_ln, atol=1e-5, rtol=1e-4), \
        f"Max diff: {(out_cln - out_ln).abs().max().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
def test_backward_gradients(channels_last):
    """Verify gradients flow through CLN and are finite."""
    C, H, W = 64, 8, 8
    cond_shape = 16
    n_cond = 2
    B_nf = n_cond * 12

    torch.manual_seed(42)
    cln = _make_new_cln(cond_shape, C)

    x = torch.randn(B_nf, C, H, W, device="cuda")
    cond = torch.randn(n_cond, cond_shape, device="cuda")

    if channels_last:
        x = x.to(memory_format=torch.channels_last)

    x = x.requires_grad_(True)
    cond = cond.requires_grad_(True)

    out = cln(x, cond)
    out.sum().backward()

    assert x.grad is not None, "No gradient for input x"
    assert cond.grad is not None, "No gradient for conditions"
    assert torch.isfinite(x.grad).all(), "Non-finite input gradients"
    assert torch.isfinite(cond.grad).all(), "Non-finite condition gradients"

    for name, p in cln.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backward_channels_last_matches_contiguous():
    """Verify channels_last and contiguous inputs produce the same gradients."""
    C, H, W = 64, 8, 8
    cond_shape = 16
    n_cond = 2
    B_nf = n_cond * 12

    torch.manual_seed(42)
    cln = _make_new_cln(cond_shape, C)

    x_base = torch.randn(B_nf, C, H, W, device="cuda")
    cond_base = torch.randn(n_cond, cond_shape, device="cuda")

    # Contiguous path
    x_cont = x_base.clone().detach().requires_grad_(True)
    cond_cont = cond_base.clone().detach().requires_grad_(True)
    out_cont = cln(x_cont, cond_cont)
    out_cont.sum().backward()

    cln.zero_grad()

    # Channels-last path
    x_cl = x_base.clone().detach().to(memory_format=torch.channels_last).requires_grad_(True)
    cond_cl = cond_base.clone().detach().requires_grad_(True)
    out_cl = cln(x_cl, cond_cl)
    out_cl.sum().backward()

    assert torch.allclose(out_cont, out_cl, atol=1e-5, rtol=1e-4), \
        f"Output max diff: {(out_cont - out_cl).abs().max().item()}"
    assert torch.allclose(x_cont.grad, x_cl.grad, atol=1e-5, rtol=1e-4), \
        f"Input grad max diff: {(x_cont.grad - x_cl.grad).abs().max().item()}"
    assert torch.allclose(cond_cont.grad, cond_cl.grad, atol=1e-5, rtol=1e-4), \
        f"Cond grad max diff: {(cond_cont.grad - cond_cl.grad).abs().max().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_load_old_checkpoint():
    """Verify new CLN can load old-format state dict via _load_from_state_dict."""
    C, cond_shape = 64, 16
    torch.manual_seed(42)
    old_cln = _make_old_cln(cond_shape, C)
    old_sd = old_cln.state_dict()

    new_cln = _make_new_cln(cond_shape, C)
    new_cln.load_state_dict(old_sd, strict=False)

    # Verify outputs match after loading old checkpoint
    x = torch.randn(12, C, 8, 8, device="cuda")
    cond = torch.randn(1, cond_shape, device="cuda")

    with torch.no_grad():
        out_old = old_cln(x, cond)
        out_new = new_cln(x, cond)

    assert out_new.shape == (12, C, 8, 8)
    assert torch.isfinite(out_new).all()
    assert torch.allclose(out_old, out_new, atol=1e-5, rtol=1e-4), \
        f"Max diff after loading old checkpoint: {(out_old - out_new).abs().max().item()}"
