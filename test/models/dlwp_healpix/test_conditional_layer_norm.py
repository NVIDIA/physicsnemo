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

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import torch
from _cln_reference import ConditionalLayerNormReference

from physicsnemo.models.dlwp_healpix.layers import normalization
from physicsnemo.models.dlwp_healpix.layers.normalization import ConditionalLayerNorm


def _make_reference_cln(condition_shape, channel_depth, device="cpu", **kwargs):
    """Instantiate the reference implementation."""
    return ConditionalLayerNormReference(
        condition_shape=condition_shape, channel_depth=channel_depth, **kwargs
    ).to(device)


def _make_optimized_cln(condition_shape, channel_depth, device="cpu", **kwargs):
    """Instantiate the optimized implementation."""
    return ConditionalLayerNorm(
        condition_shape=condition_shape, channel_depth=channel_depth, **kwargs
    ).to(device)


def _copy_reference_to_optimized(reference_cln, optimized_cln):
    """Copy the reference implementation's separate gamma/beta MLP weights into the
    optimized fused MLP using block-diagonal structure.

    Reference: gamma_mlp and beta_mlp each have hidden_dims [h1, h2] and output C.
    Optimized: gamma_beta_mlp has hidden_dims [2*h1, 2*h2] and output 2*C.

    Layer 0 (condition_shape → 2*h1): vertical cat of gamma/beta weights.
    Layer i>0 (2*h_{i-1} → 2*h_i or 2*C): block-diagonal [[gamma, 0], [0, beta]].
    Biases: always concatenated.
    """
    reference_sd = reference_cln.state_dict()

    # Collect Linear layer indices from the reference gamma MLP
    gamma_linear_indices = sorted(
        {
            int(k.split(".")[1])
            for k in reference_sd
            if k.startswith("gamma_mlp.") and k.endswith(".weight")
        }
    )
    first_layer_idx = gamma_linear_indices[0]

    optimized_sd = {}
    for key in reference_sd:
        if key.startswith("norm."):
            optimized_sd[key] = reference_sd[key]

    for idx in gamma_linear_indices:
        for param in ("weight", "bias"):
            gamma_val = reference_sd[f"gamma_mlp.{idx}.{param}"]
            beta_val = reference_sd[f"beta_mlp.{idx}.{param}"]
            fused_key = f"gamma_beta_mlp.{idx}.{param}"

            if param == "bias":
                optimized_sd[fused_key] = torch.cat([gamma_val, beta_val], dim=0)
            elif idx == first_layer_idx:
                # First layer: shared input dim, just cat along output dim
                optimized_sd[fused_key] = torch.cat([gamma_val, beta_val], dim=0)
            else:
                # Block-diagonal: [[gamma, 0], [0, beta]]
                out_dim, in_dim = gamma_val.shape
                zeros = torch.zeros_like(gamma_val)
                optimized_sd[fused_key] = torch.cat(
                    [
                        torch.cat([gamma_val, zeros], dim=1),
                        torch.cat([zeros, beta_val], dim=1),
                    ],
                    dim=0,
                )

    for key, value in reference_sd.items():
        if not key.startswith("gamma_mlp."):
            continue
        layer_key = key[len("gamma_mlp.") :]
        parts = layer_key.split(".", 1)
        if len(parts) != 2 or parts[1] in ("weight", "bias"):
            continue
        optimized_sd[f"gamma_beta_mlp.{layer_key}"] = value

    optimized_cln.load_state_dict(optimized_sd)


def _assert_forward_is_finite(
    cln, channel_depth, condition_shape, device, n_cond=1, n_faces=12, height=4, width=4
):
    """Run a minimal forward pass and assert the output is finite."""
    x = torch.randn(n_cond * n_faces, channel_depth, height, width, device=device)
    cond = torch.randn(n_cond, condition_shape, device=device)
    out = cln(x, cond)
    assert torch.isfinite(out).all()
    return out


@pytest.mark.parametrize("n_cond", [1, 2, 4])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("scale_center", [0.0, 1.0])
def test_optimized_vs_reference_forward(device, n_cond, channels_last, scale_center):
    """Verify optimized CLN matches reference implementation with block-diagonal weight mapping."""
    C, H, W = 128, 16, 16
    cond_shape = 32
    B_nf = n_cond * 12

    torch.manual_seed(42)
    reference_cln = _make_reference_cln(
        cond_shape, C, device=device, scale_center=scale_center
    )

    optimized_cln = _make_optimized_cln(
        cond_shape, C, device=device, scale_center=scale_center
    )
    _copy_reference_to_optimized(reference_cln, optimized_cln)

    x = torch.randn(B_nf, C, H, W, device=device)
    cond = torch.randn(n_cond, cond_shape, device=device)

    if channels_last:
        x = x.to(memory_format=torch.channels_last)

    with torch.no_grad():
        out_ref = reference_cln(x, cond)
        out_opt = optimized_cln(x, cond)

    assert out_ref.shape == out_opt.shape
    assert torch.allclose(out_ref, out_opt, atol=1e-5, rtol=1e-4), (
        f"Max diff: {(out_ref - out_opt).abs().max().item()}"
    )

    if channels_last:
        assert out_opt.is_contiguous(memory_format=torch.channels_last), (
            "Output should preserve channels_last format"
        )


@pytest.mark.parametrize("channels_last", [False, True])
def test_optimized_vs_reference_backward(device, channels_last):
    """Verify gradients match between the optimized implementation and the
    reference implementation."""
    C, H, W = 64, 8, 8
    cond_shape = 16
    n_cond = 2
    B_nf = n_cond * 12

    torch.manual_seed(42)
    reference_cln = _make_reference_cln(cond_shape, C, device=device)
    optimized_cln = _make_optimized_cln(cond_shape, C, device=device)
    _copy_reference_to_optimized(reference_cln, optimized_cln)

    x_base = torch.randn(B_nf, C, H, W, device=device)
    cond_base = torch.randn(n_cond, cond_shape, device=device)

    if channels_last:
        x_base = x_base.to(memory_format=torch.channels_last)

    x_ref = x_base.clone().detach().requires_grad_(True)
    cond_ref = cond_base.clone().detach().requires_grad_(True)
    x_opt = x_base.clone().detach().requires_grad_(True)
    cond_opt = cond_base.clone().detach().requires_grad_(True)

    out_ref = reference_cln(x_ref, cond_ref)
    out_ref.sum().backward()

    out_opt = optimized_cln(x_opt, cond_opt)
    out_opt.sum().backward()

    assert torch.allclose(x_ref.grad, x_opt.grad, atol=1e-4, rtol=1e-3), (
        f"Input grad max diff: {(x_ref.grad - x_opt.grad).abs().max().item()}"
    )

    assert torch.allclose(cond_ref.grad, cond_opt.grad, atol=1e-4, rtol=1e-3), (
        f"Cond grad max diff: {(cond_ref.grad - cond_opt.grad).abs().max().item()}"
    )


@pytest.mark.parametrize("channels_last", [False, True])
def test_init_cln_to_zero_matches_layer_norm(device, channels_last):
    """With scale_center=1.0 and init_cln_to_zero=True, CLN should behave like plain LayerNorm."""
    C, H, W = 64, 8, 8
    n_cond = 2
    B_nf = n_cond * 12

    torch.manual_seed(42)
    cln = _make_optimized_cln(
        32, C, device=device, scale_center=1.0, init_cln_to_zero=True
    )
    plain_ln = torch.nn.LayerNorm(C, elementwise_affine=False).to(device)

    x = torch.randn(B_nf, C, H, W, device=device)
    cond = torch.randn(n_cond, 32, device=device)

    if channels_last:
        x = x.to(memory_format=torch.channels_last)

    with torch.no_grad():
        out_cln = cln(x, cond)
        x_nhwc = x.permute(0, 2, 3, 1)
        out_ln = plain_ln(x_nhwc).permute(0, 3, 1, 2)

    assert torch.allclose(out_cln, out_ln, atol=1e-5, rtol=1e-4), (
        f"Max diff: {(out_cln - out_ln).abs().max().item()}"
    )


@pytest.mark.parametrize("channels_last", [False, True])
def test_backward_gradients(device, channels_last):
    """Verify gradients flow through CLN and are finite."""
    C, H, W = 64, 8, 8
    cond_shape = 16
    n_cond = 2
    B_nf = n_cond * 12

    torch.manual_seed(42)
    cln = _make_optimized_cln(cond_shape, C, device=device)

    x = torch.randn(B_nf, C, H, W, device=device)
    cond = torch.randn(n_cond, cond_shape, device=device)

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


def test_backward_channels_last_matches_contiguous(device):
    """Verify channels_last and contiguous inputs produce the same gradients."""
    C, H, W = 64, 8, 8
    cond_shape = 16
    n_cond = 2
    B_nf = n_cond * 12

    torch.manual_seed(42)
    cln = _make_optimized_cln(cond_shape, C, device=device)

    x_base = torch.randn(B_nf, C, H, W, device=device)
    cond_base = torch.randn(n_cond, cond_shape, device=device)

    # Contiguous path
    x_cont = x_base.clone().detach().requires_grad_(True)
    cond_cont = cond_base.clone().detach().requires_grad_(True)
    out_cont = cln(x_cont, cond_cont)
    out_cont.sum().backward()

    cln.zero_grad()

    # Channels-last path
    x_cl = (
        x_base.clone()
        .detach()
        .to(memory_format=torch.channels_last)
        .requires_grad_(True)
    )
    cond_cl = cond_base.clone().detach().requires_grad_(True)
    out_cl = cln(x_cl, cond_cl)
    out_cl.sum().backward()

    assert torch.allclose(out_cont, out_cl, atol=1e-5, rtol=1e-4), (
        f"Output max diff: {(out_cont - out_cl).abs().max().item()}"
    )
    assert torch.allclose(x_cont.grad, x_cl.grad, atol=1e-5, rtol=1e-4), (
        f"Input grad max diff: {(x_cont.grad - x_cl.grad).abs().max().item()}"
    )
    assert torch.allclose(cond_cont.grad, cond_cl.grad, atol=1e-5, rtol=1e-4), (
        f"Cond grad max diff: {(cond_cont.grad - cond_cl.grad).abs().max().item()}"
    )


@pytest.mark.parametrize("hidden_dims", [[], [64]])
def test_optimized_vs_reference_forward_hidden_dims_variation(device, hidden_dims):
    """Verify the block-diagonal weight mapping generalizes to MLP depths other
    than the default two hidden layers, including the degenerate zero-hidden-layer
    case (a single Linear directly from ``condition_shape`` to the output).
    """
    C, H, W = 32, 8, 8
    cond_shape = 16
    n_cond = 2
    B_nf = n_cond * 12

    torch.manual_seed(42)
    reference_cln = _make_reference_cln(
        cond_shape, C, device=device, mlp_hidden_dims=hidden_dims
    )
    optimized_cln = _make_optimized_cln(
        cond_shape, C, device=device, mlp_hidden_dims=hidden_dims
    )
    _copy_reference_to_optimized(reference_cln, optimized_cln)

    x = torch.randn(B_nf, C, H, W, device=device)
    cond = torch.randn(n_cond, cond_shape, device=device)

    with torch.no_grad():
        out_ref = reference_cln(x, cond)
        out_opt = optimized_cln(x, cond)

    assert torch.allclose(out_ref, out_opt, atol=1e-5, rtol=1e-4), (
        f"Max diff: {(out_ref - out_opt).abs().max().item()}"
    )


def test_norm_op_apex_raises_when_unavailable():
    """``norm_op='apex'`` must raise an informative ImportError when the
    ``apex`` package isn't installed, rather than silently falling back or
    failing with an unrelated error later on. The error is raised by
    ``OptionalImport`` when the fused-norm symbol is accessed.
    """
    if importlib.util.find_spec("apex") is not None:
        pytest.skip("apex is installed in this environment")

    with pytest.raises(ImportError, match="Missing optional dependency: apex"):
        ConditionalLayerNorm(condition_shape=16, channel_depth=8, norm_op="apex")


def test_norm_op_invalid_value_leaves_norm_unset():
    """``norm_op`` outside {"torch", "apex"} is not validated; document the
    current behavior of silently skipping norm construction rather than
    raising, so a regression (e.g. an accidental typo check) is caught.
    """
    cln = ConditionalLayerNorm(condition_shape=16, channel_depth=8, norm_op="bogus")
    assert not hasattr(cln, "norm")


def test_norm_op_apex_available_uses_fused_layer_norm(monkeypatch):
    """When apex *is* available, ``norm_op='apex'`` must construct a
    ``FusedLayerNorm`` instead of ``torch.nn.LayerNorm``. Exercised via a
    stand-in ``apex.normalization`` module since apex isn't a hard dependency
    of this environment.
    """

    class _FakeFusedLayerNorm(torch.nn.Module):
        def __init__(self, channel_depth, elementwise_affine=False):
            super().__init__()
            self.channel_depth = channel_depth
            self.elementwise_affine = elementwise_affine

        def forward(self, x):
            return x

    class _FakeApexNormalization:
        FusedLayerNorm = _FakeFusedLayerNorm

    monkeypatch.setattr(normalization, "apex_normalization", _FakeApexNormalization)

    cln = ConditionalLayerNorm(condition_shape=16, channel_depth=8, norm_op="apex")
    assert isinstance(cln.norm, _FakeFusedLayerNorm)
    assert cln.norm.channel_depth == 8


def test_make_mlp_skips_activation_when_falsy():
    """``_make_mlp`` only inserts an activation module between hidden Linear
    layers when ``activation`` is truthy; verify the skip path directly since
    the public constructor always coerces ``None`` to ``nn.Identity()``
    (itself truthy), making this otherwise unreachable through ``__init__``.
    """
    cln = ConditionalLayerNorm(condition_shape=16, channel_depth=8, mlp_hidden_dims=[])
    mlp = cln._make_mlp(in_dim=4, hidden_dims=[8, 8], out_dim=2, activation=False)
    assert all(isinstance(layer, torch.nn.Linear) for layer in mlp)
    assert len(mlp) == 3  # one Linear per hidden dim, plus the output Linear


def test_cln_affine_eager_matches_compiled(monkeypatch, device):
    """``_cln_affine`` is wrapped in ``@torch.compile``, which bypasses
    coverage tracing of its body; run it once via the (functionally
    identical) eager callable dynamo wraps to confirm equivalence and
    exercise the underlying implementation directly.
    """
    eager_cln_affine = normalization._cln_affine._torchdynamo_orig_callable

    C, H, W = 16, 4, 4
    cond_shape = 8
    n_cond = 2
    n_faces = 12

    torch.manual_seed(0)
    x_norm = torch.randn(n_cond * n_faces, H, W, C, device=device)
    gamma_raw = torch.randn(n_cond, C, device=device)
    beta = torch.randn(n_cond, C, device=device)
    scale_center = 1.0

    compiled_out = normalization._cln_affine(
        x_norm, gamma_raw, beta, scale_center, n_faces
    )
    eager_out = eager_cln_affine(x_norm, gamma_raw, beta, scale_center, n_faces)

    assert torch.allclose(compiled_out, eager_out)

    # also drive it end-to-end through ConditionalLayerNorm.forward with the
    # module-level name patched to the eager callable
    monkeypatch.setattr(normalization, "_cln_affine", eager_cln_affine)
    cln = ConditionalLayerNorm(
        condition_shape=cond_shape, channel_depth=C, mlp_hidden_dims=[]
    ).to(device)
    _assert_forward_is_finite(
        cln, C, cond_shape, device, n_cond=n_cond, n_faces=n_faces, height=H, width=W
    )


def test_load_new_format_checkpoint_roundtrip(device):
    """A state dict already in the fused ``gamma_beta_mlp`` format (i.e. saved
    from a ``ConditionalLayerNorm`` instance, with no legacy ``gamma_mlp``/
    ``beta_mlp`` keys) should load directly with no remapping and reproduce
    identical outputs.
    """
    C, cond_shape = 32, 16
    torch.manual_seed(42)
    source = _make_optimized_cln(cond_shape, C, device=device)
    state_dict = source.state_dict()
    assert not any(k.startswith("gamma_mlp.") for k in state_dict)

    target = _make_optimized_cln(cond_shape, C, device=device)
    missing, unexpected = target.load_state_dict(state_dict, strict=True)
    assert not missing
    assert not unexpected

    x = torch.randn(12, C, 4, 4, device=device)
    cond = torch.randn(1, cond_shape, device=device)
    with torch.no_grad():
        out_source = source(x, cond)
        out_target = target(x, cond)

    assert torch.allclose(out_source, out_target, atol=1e-6, rtol=1e-5)
