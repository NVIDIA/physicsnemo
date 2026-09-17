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

import copy
from pathlib import Path

import pytest
import torch

from physicsnemo.core.module import Module
from physicsnemo.experimental.models.strata import Strata, StrataTransformer3D
from physicsnemo.experimental.models.strata.coords import (
    build_axial_token_coords,
    build_stereographic_token_coords,
)
from physicsnemo.experimental.models.strata.layers import (
    DepthwiseConv,
    Natten3DSelfAttention,
    StrataPixel3DBlock,
)
from test.common import validate_checkpoint
from test.conftest import requires_module

_DATA = Path(__file__).parent / "data"


def _make_pos(b: int, h: int, w: int) -> torch.Tensor:
    """Deterministic (B, 2, H, W) latitude / longitude grid in radians."""
    lat = torch.linspace(-1.0, 1.0, h).reshape(1, h, 1).expand(b, h, w)
    lon = torch.linspace(0.0, 1.5, w).reshape(1, 1, w).expand(b, h, w)
    return torch.stack([lat, lon], dim=1).contiguous()


def _seed_params(model: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Fill all parameters with reproducible random values (MOD-008b pattern).

    StrataTransformer3D zero-initializes its output head, so a freshly constructed model
    produces all-zero outputs; seeding the parameters gives meaningful,
    reproducible forward outputs for non-regression and checkpoint tests.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=param.dtype))
    return model


# --------------------------------------------------------------------------- #
# Non-regression fixtures. Each builder returns (model, args). Goldens store
# {args, state_dict, y}; the test loads state_dict + args and compares y, so it
# is independent of init/RNG changes across PyTorch versions. All use the CPU-
# reproducible full-attention path (attn_kernel=-1).
# --------------------------------------------------------------------------- #
def _build_transformer_axial():
    model = StrataTransformer3D(
        in_channels=4,
        input_shape=(4, 8, 8),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=3,
        attn_kernel=-1,
        do_alt_depthwise_attn=True,
        gated_attention=True,
        rope_mode="axial",
    )
    gen = torch.Generator().manual_seed(11)
    return _seed_params(model, seed=10), (torch.randn(2, 4, 4, 8, 8, generator=gen),)


def _build_transformer_stereo():
    model = StrataTransformer3D(
        in_channels=3,
        input_shape=(6, 8, 8),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        attn_kernel=-1,
        qk_norm=True,
        rope_mode="stereographic",
    )
    gen = torch.Generator().manual_seed(21)
    return _seed_params(model, seed=20), (
        torch.randn(2, 3, 6, 8, 8, generator=gen),
        _make_pos(2, 8, 8),
    )


def _build_strata_pixelproj():
    model = Strata(
        backbone_config=dict(
            in_channels=4,
            input_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
        ),
        embed_dim_pixel=16,
        num_layers_pixel=2,
        num_heads_pixel=2,
        attn_kernel_pixel=-1,
        adaln_mode="pixel_proj",
    )
    gen = torch.Generator().manual_seed(31)
    return _seed_params(model, seed=30), (torch.randn(2, 4, 4, 8, 8, generator=gen),)


def _build_strata_bilinear():
    model = Strata(
        backbone_config=dict(
            in_channels=4,
            input_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
        ),
        embed_dim_pixel=16,
        num_layers_pixel=3,
        num_heads_pixel=2,
        attn_kernel_pixel=-1,
        adaln_mode="bilinear_dw",
        first_block_only_adaln=True,
    )
    gen = torch.Generator().manual_seed(41)
    return _seed_params(model, seed=40), (torch.randn(2, 4, 4, 8, 8, generator=gen),)


def _build_strata_bilinear_pd2():
    # Vertical patch size 2 -> backbone depth 2, pixel depth 4, so bilinear_dw
    # takes the trilinear depth-upsample path (not the d == sd 2D fallback).
    model = Strata(
        backbone_config=dict(
            in_channels=4,
            input_shape=(4, 8, 8),
            patch_size=(2, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
        ),
        embed_dim_pixel=16,
        num_layers_pixel=2,
        num_heads_pixel=2,
        attn_kernel_pixel=-1,
        adaln_mode="bilinear_dw",
    )
    gen = torch.Generator().manual_seed(51)
    return _seed_params(model, seed=50), (torch.randn(2, 4, 4, 8, 8, generator=gen),)


# name -> (builder, golden path). Drives the non-regression test and the golden
# generator (data/_generate_strata_goldens.py).
_FIXTURE_REGISTRY = [
    ("transformer_axial", _build_transformer_axial, _DATA / "transformer_axial.pth"),
    ("transformer_stereo", _build_transformer_stereo, _DATA / "transformer_stereo.pth"),
    ("strata_pixelproj", _build_strata_pixelproj, _DATA / "strata_pixelproj.pth"),
    ("strata_bilinear", _build_strata_bilinear, _DATA / "strata_bilinear.pth"),
    (
        "strata_bilinear_pd2",
        _build_strata_bilinear_pd2,
        _DATA / "strata_bilinear_pd2.pth",
    ),
]


# --------------------------------------------------------------------------- #
# Constructor / attribute tests (MOD-008a)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "config", ["default", "custom"], ids=["with_defaults", "with_custom_args"]
)
def test_transformer_constructor(config):
    """StrataTransformer3D constructor and public attributes."""
    if config == "default":
        model = StrataTransformer3D(in_channels=4)
        assert model.out_channels == 4  # defaults to in_channels
        assert model.embed_dim == 768
        assert model.num_heads == 8
        assert model.num_layers == 12
        assert model.patch_size == (1, 1, 1)
        assert model.rope_mode == "none"
        assert model.input_shape == (16, 64, 64)
    else:
        model = StrataTransformer3D(
            in_channels=3,
            out_channels=5,
            input_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            embed_dim=64,
            num_heads=8,
            num_layers=4,
            attn_kernel=-1,
            rope_mode="stereographic",
        )
        assert model.out_channels == 5
        assert model.embed_dim == 64
        assert model.num_layers == 4
        assert model.patch_size == (1, 2, 2)
        assert model.rope_mode == "stereographic"

    assert isinstance(model, Module), (
        "StrataTransformer3D should inherit physicsnemo.Module"
    )
    assert hasattr(model, "meta")
    assert len(model.blocks) == model.num_layers
    # Axial mode caches static cos/sin buffers; stereographic builds per forward.
    assert hasattr(model, "_rope_cos") == (model.rope_mode == "axial")


@pytest.mark.parametrize(
    "config", ["default", "custom"], ids=["with_defaults", "with_custom_args"]
)
def test_strata_constructor(config):
    """Strata constructor and public attributes."""
    backbone_config = dict(
        in_channels=4,
        input_shape=(4, 8, 8),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        attn_kernel=-1,
    )
    if config == "default":
        model = Strata(backbone_config=backbone_config)
        assert model.embed_dim_pixel == 128
        assert model.num_layers_pixel == 4
        assert model.adaln_mode == "pixel_proj"
        assert model.first_block_only_adaln is False
        # All blocks inject conditioning when not first-block-only.
        assert all(isinstance(b, StrataPixel3DBlock) for b in model.pixel_blocks)
    else:
        model = Strata(
            backbone_config=backbone_config,
            embed_dim_pixel=16,
            num_layers_pixel=3,
            num_heads_pixel=2,
            attn_kernel_pixel=-1,
            adaln_mode="bilinear_dw",
            first_block_only_adaln=True,
        )
        assert model.embed_dim_pixel == 16
        assert model.num_layers_pixel == 3
        assert model.adaln_mode == "bilinear_dw"
        # First-block-only: exactly one conditioning block, the rest plain.
        assert isinstance(model.pixel_blocks[0], StrataPixel3DBlock)
        assert sum(isinstance(b, StrataPixel3DBlock) for b in model.pixel_blocks) == 1

    assert isinstance(model, Module), "Strata should inherit physicsnemo.Module"
    assert isinstance(model.backbone, StrataTransformer3D)
    # The backbone stage is built headless (include_head=False): no output head,
    # only forward_tokens is used.
    assert model.backbone.final_layer is None
    assert not any("final_layer" in n for n, _ in model.backbone.named_parameters())
    assert len(model.pixel_blocks) == model.num_layers_pixel


def test_transformer_invalid_args():
    """Constructor validation for incompatible arguments."""
    with pytest.raises(ValueError):  # embed_dim not divisible by num_heads
        StrataTransformer3D(in_channels=4, embed_dim=30, num_heads=4)
    with pytest.raises(ValueError):  # head_dim not divisible by 4 with RoPE
        StrataTransformer3D(in_channels=4, embed_dim=8, num_heads=4, rope_mode="axial")
    with pytest.raises(ValueError):  # bad rope_mode
        StrataTransformer3D(in_channels=4, rope_mode="bogus")


@pytest.mark.parametrize("adaln_mode", ["pixel_proj", "bilinear_dw"])
def test_strata_supports_vertical_patch_gt_one(adaln_mode):
    """Both AdaLN modes support a backbone vertical patch size > 1; bilinear_dw
    trilinearly upsamples the depth axis (no longer restricted to patch_vert=1)."""
    torch.manual_seed(0)
    model = Strata(
        backbone_config=dict(
            in_channels=4,
            input_shape=(4, 8, 8),
            patch_size=(2, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=1,
            attn_kernel=-1,
        ),
        embed_dim_pixel=16,
        num_heads_pixel=2,
        attn_kernel_pixel=-1,
        adaln_mode=adaln_mode,
    )
    x = torch.randn(2, 4, 4, 8, 8)
    out = model(x)
    assert out.shape == (2, 4, 4, 8, 8)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Forward shape tests (CPU + CUDA via the SDPA path)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rope_mode", ["none", "axial", "stereographic"])
def test_transformer_forward_shape(device, rope_mode):
    """StrataTransformer3D forward produces the correct output shape on the SDPA path."""
    torch.manual_seed(0)
    b, c, d, h, w = 2, 4, 4, 8, 8
    model = StrataTransformer3D(
        in_channels=c,
        input_shape=(d, h, w),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        attn_kernel=-1,
        rope_mode=rope_mode,
    ).to(device)
    x = torch.randn(b, c, d, h, w, device=device)
    pos = _make_pos(b, h, w).to(device) if rope_mode == "stereographic" else None
    out = model(x, pos)
    assert out.shape == (b, c, d, h, w)


@pytest.mark.parametrize("adaln_mode", ["pixel_proj", "bilinear_dw"])
def test_strata_forward_shape(device, adaln_mode):
    """Strata forward produces the correct output shape on the SDPA path."""
    torch.manual_seed(0)
    b, c, d, h, w = 2, 4, 4, 8, 8
    model = Strata(
        backbone_config=dict(
            in_channels=c,
            input_shape=(d, h, w),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
        ),
        embed_dim_pixel=16,
        num_layers_pixel=2,
        num_heads_pixel=2,
        attn_kernel_pixel=-1,
        adaln_mode=adaln_mode,
    ).to(device)
    x = torch.randn(b, c, d, h, w, device=device)
    assert model(x).shape == (b, c, d, h, w)


# --------------------------------------------------------------------------- #
# Non-regression tests (MOD-008b)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,builder,golden",
    _FIXTURE_REGISTRY,
    ids=[n for n, _, _ in _FIXTURE_REGISTRY],
)
def test_non_regression(name, builder, golden):
    """Forward outputs match the committed golden (loaded params + inputs)."""
    if not golden.exists():
        pytest.skip(
            f"golden {golden.name} missing; run "
            f"test/experimental/models/strata/data/_generate_strata_goldens.py"
        )
    data = torch.load(golden)
    model, _ = builder()
    model.load_state_dict(data["state_dict"])
    model.eval()
    with torch.no_grad():
        y = model(*data["args"])
    # 2e-3 (not 1e-4): the bilinear_dw variants run through DepthwiseConv, and
    # conv2d results differ across float backends. The committed goldens are
    # CPU-generated and match bit-exactly on that backend, but a different
    # torch/oneDNN build (CI) or device diverges -- measured up to ~5e-3 on a
    # CUDA backend, ~1e-3 on a differing CPU build. 6e-3/6e-3 covers that; it is
    # float non-determinism, not a regression (a real regression shifts this
    # deep transformer's outputs by far more).
    assert torch.allclose(y, data["y"], atol=6e-3, rtol=6e-3)


# --------------------------------------------------------------------------- #
# Checkpoint tests (MOD-008c)
# --------------------------------------------------------------------------- #
def test_transformer_checkpoint(device):
    """StrataTransformer3D save/load/from_checkpoint reproduce the forward output."""
    torch.manual_seed(0)
    kwargs = dict(
        in_channels=4,
        input_shape=(4, 8, 8),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=3,
        attn_kernel=-1,
        do_alt_depthwise_attn=True,
        rope_mode="axial",
    )
    model_1 = _seed_params(StrataTransformer3D(**kwargs), seed=1).to(device)
    model_2 = _seed_params(StrataTransformer3D(**kwargs), seed=2).to(device)
    x = torch.randn(2, 4, 4, 8, 8, device=device)
    assert validate_checkpoint(model_1, model_2, (x,))


@pytest.mark.parametrize("adaln_mode", ["pixel_proj", "bilinear_dw"])
def test_strata_checkpoint(device, adaln_mode):
    """Strata save/load/from_checkpoint reproduce the forward output.

    Covers ``bilinear_dw`` too, so the shadowed-``forward`` (chunked
    :class:`DepthwiseConv`) survives a full ``.mdlus`` round-trip.
    """
    torch.manual_seed(0)
    kwargs = dict(
        backbone_config=dict(
            in_channels=4,
            input_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
        ),
        embed_dim_pixel=16,
        num_layers_pixel=2,
        num_heads_pixel=2,
        attn_kernel_pixel=-1,
        adaln_mode=adaln_mode,
    )
    model_1 = _seed_params(Strata(**kwargs), seed=1).to(device)
    model_2 = _seed_params(Strata(**kwargs), seed=2).to(device)
    x = torch.randn(2, 4, 4, 8, 8, device=device)
    assert validate_checkpoint(model_1, model_2, (x,))


# --------------------------------------------------------------------------- #
# 3D neighborhood-attention (NATTEN) tests (CUDA + natten only)
# --------------------------------------------------------------------------- #
@requires_module(["natten"])
def test_transformer_natten_forward(device):
    """StrataTransformer3D forward on the NA3D path (NATTEN is CUDA-only)."""
    if device == "cpu":
        pytest.skip("natten neighborhood attention is not available on CPU")
    torch.manual_seed(0)
    b, c, d, h, w = 2, 4, 4, 8, 8
    model = StrataTransformer3D(
        in_channels=c,
        input_shape=(d, h, w),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=3,
        attn_kernel=3,
        do_alt_depthwise_attn=True,
        gated_attention=True,
        rope_mode="stereographic",
    ).to(device)
    x = torch.randn(b, c, d, h, w, device=device)
    pos = _make_pos(b, h, w).to(device)
    out = model(x, pos)
    assert out.shape == (b, c, d, h, w)
    assert torch.isfinite(out).all()


@requires_module(["natten"])
def test_strata_natten_forward(device):
    """Strata forward on the NA3D path (NATTEN is CUDA-only)."""
    if device == "cpu":
        pytest.skip("natten neighborhood attention is not available on CPU")
    torch.manual_seed(0)
    b, c, d, h, w = 2, 4, 4, 8, 8
    model = Strata(
        backbone_config=dict(
            in_channels=c,
            input_shape=(d, h, w),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=3,
        ),
        embed_dim_pixel=16,
        num_layers_pixel=2,
        num_heads_pixel=2,
        attn_kernel_pixel=3,
        adaln_mode="bilinear_dw",
    ).to(device)
    x = torch.randn(b, c, d, h, w, device=device)
    out = model(x)
    assert out.shape == (b, c, d, h, w)
    assert torch.isfinite(out).all()


def test_natten3d_attention_kernel_triple():
    """Natten3DSelfAttention accepts int and per-axis tuple kernels; rejects bad length."""
    attn_int = Natten3DSelfAttention(dim=32, num_heads=4, attn_kernel=3)
    assert attn_int.attn_kernel == 3
    attn_tuple = Natten3DSelfAttention(dim=32, num_heads=4, attn_kernel=(3, 5, 5))
    assert attn_tuple.attn_kernel == (3, 5, 5)
    # A malformed (non-length-3) tuple is rejected eagerly at construction.
    with pytest.raises(ValueError):
        Natten3DSelfAttention(dim=32, num_heads=4, attn_kernel=(3, 5))


# --------------------------------------------------------------------------- #
# Gradient-flow tests: every trainable parameter must participate in the
# forward graph (a parameter left with ``grad is None`` after backward is dead /
# disconnected). For Strata this also confirms the backbone stage receives
# gradients through the pixel stage.
# --------------------------------------------------------------------------- #
def _assert_all_params_receive_grad(model):
    missing = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing, f"parameters received no gradient: {missing}"


def test_transformer_backward_all_params_receive_gradients(device):
    """All StrataTransformer3D parameters receive a gradient (no dead params)."""
    torch.manual_seed(0)
    model = _seed_params(
        StrataTransformer3D(
            in_channels=4,
            input_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
            do_alt_depthwise_attn=True,
            gated_attention=True,
            rope_mode="stereographic",
        ),
        seed=1,
    ).to(device)
    x = torch.randn(2, 4, 4, 8, 8, device=device)
    pos = _make_pos(2, 8, 8).to(device)
    model(x, pos).pow(2).mean().backward()
    _assert_all_params_receive_grad(model)


def test_strata_backward_all_params_receive_gradients(device):
    """All Strata parameters (incl. the backbone stage) receive a gradient."""
    torch.manual_seed(0)
    model = _seed_params(
        Strata(
            backbone_config=dict(
                in_channels=4,
                input_shape=(4, 8, 8),
                patch_size=(1, 2, 2),
                embed_dim=32,
                num_heads=4,
                num_layers=2,
                attn_kernel=-1,
            ),
            embed_dim_pixel=16,
            num_layers_pixel=2,
            num_heads_pixel=2,
            attn_kernel_pixel=-1,
            adaln_mode="bilinear_dw",
        ),
        seed=1,
    ).to(device)
    x = torch.randn(2, 4, 4, 8, 8, device=device)
    model(x).pow(2).mean().backward()
    _assert_all_params_receive_grad(model)
    # The backbone stage must be reached through the pixel stage.
    assert any(
        n.startswith("backbone.") and p.grad is not None and p.grad.any()
        for n, p in model.named_parameters()
    )


# --------------------------------------------------------------------------- #
# Component-level unit test: DepthwiseConv
# --------------------------------------------------------------------------- #
@torch.no_grad()
def test_depthwise_conv_is_depthwise_and_preserves_shape():
    """DepthwiseConv is grouped per-channel, preserves shape, and rejects groups."""
    conv = DepthwiseConv(8, kernel_size=5, padding=2)
    assert conv.groups == 8  # one group per channel == depthwise
    x = torch.randn(2, 8, 6, 6)
    out = conv(x)
    assert out.shape == (2, 8, 6, 6) and torch.isfinite(out).all()
    with pytest.raises(ValueError):
        DepthwiseConv(8, kernel_size=3, groups=2)


@torch.no_grad()
def test_depthwise_conv_chunked_matches_plain():
    """The chunked torch.vmap path (used by bilinear_dw) matches the plain conv."""
    torch.manual_seed(0)
    # chunk_size triggers the vmapped implementation; copy its weights to a plain
    # DepthwiseConv (standard nn.Conv2d forward) and check the outputs agree.
    chunked = DepthwiseConv(8, kernel_size=3, padding=1, chunk_size=4)
    plain = DepthwiseConv(8, kernel_size=3, padding=1)
    plain.load_state_dict(chunked.state_dict())
    x = torch.randn(2, 8, 6, 6)
    assert torch.allclose(chunked(x), plain(x), atol=1e-5)


@torch.no_grad()
def test_depthwise_conv_chunked_deepcopy_uses_own_weights():
    """A deep-copied chunked DepthwiseConv must use its OWN parameters.

    Regression: the chunked forward reads ``self.weight`` / ``self.bias`` live
    rather than closing over the original module, so EMA / ``AveragedModel`` /
    snapshot paths (which ``copy.deepcopy`` the model) stay correct. If the copy
    aliased the source's weights, zeroing the copy would have no effect.
    """
    torch.manual_seed(0)
    conv = DepthwiseConv(4, kernel_size=3, padding=1, chunk_size=2)
    clone = copy.deepcopy(conv)
    clone.weight.zero_()
    clone.bias.zero_()
    x = torch.randn(1, 4, 8, 8)
    # All-zero weight+bias => output must be exactly zero, and must NOT equal the
    # original's output (which would mean the clone aliased the source weights).
    assert torch.count_nonzero(clone(x)) == 0
    assert not torch.equal(clone(x), conv(x))


@torch.no_grad()
def test_depthwise_conv_chunked_no_bias_matches_and_moves(device):
    """bias=False chunked conv matches the plain conv and survives a device move.

    Regression: the synthetic zero-bias must be built from live module state at
    forward time; if captured at construction it stays on the original device
    after ``.to(device)`` and the chunked forward errors on a cross-device call.
    """
    torch.manual_seed(0)
    chunked = DepthwiseConv(8, kernel_size=3, padding=1, bias=False, chunk_size=4).to(
        device
    )
    plain = DepthwiseConv(8, kernel_size=3, padding=1, bias=False).to(device)
    plain.load_state_dict(chunked.state_dict())
    x = torch.randn(2, 8, 6, 6, device=device)
    assert torch.allclose(chunked(x), plain(x), atol=1e-5)


@torch.no_grad()
def test_build_axial_token_coords():
    """Integer (row, col) grid, row-major, tiled across depth."""
    d, h, w = 2, 3, 4
    coords = build_axial_token_coords(d, h, w)
    assert coords.shape == (d * h * w, 2)
    per_depth = coords.reshape(d, h * w, 2)
    assert torch.equal(per_depth[0], per_depth[1])  # same grid per depth level
    assert torch.equal(per_depth[0, 0], torch.tensor([0.0, 0.0]))
    assert torch.equal(per_depth[0, -1], torch.tensor([float(h - 1), float(w - 1)]))


@torch.no_grad()
def test_build_stereographic_token_coords():
    """Pooled patch coords, finite, depth-tiled; length_scale must be positive."""
    b, h, w = 2, 8, 8
    pos = _make_pos(b, h, w)
    coords = build_stereographic_token_coords(pos, (2, 2), d_patch=3, length_scale=0.1)
    assert coords.shape == (b, 3 * 4 * 4, 2)  # (h//2)*(w//2)=16 horizontal, x3 depth
    assert torch.isfinite(coords).all()
    blk = coords.reshape(b, 3, 4 * 4, 2)
    assert torch.equal(blk[:, 0], blk[:, 1])  # horizontal block tiled across depth

    # Value / non-degeneracy: a zeroed or constant projection (which collapses RoPE
    # to the identity) or a scrambled patch-pooling must fail these -- a shape-only
    # check would not. _make_pos has latitude increasing down rows and longitude
    # increasing across columns, so after projection the North coord (y) must
    # increase with the patch-row index and the East coord (x) with the column.
    assert coords.abs().max() > 1.0  # not collapsed to ~0
    grid = blk[0, 0].reshape(4, 4, 2)  # (h_patch, w_patch, 2), batch 0, depth 0
    y_by_row = grid[..., 1].mean(dim=1)  # North, averaged over columns
    x_by_col = grid[..., 0].mean(dim=0)  # East, averaged over rows
    assert torch.all(y_by_row[1:] > y_by_row[:-1])
    assert torch.all(x_by_col[1:] > x_by_col[:-1])

    # length_scale magnitude: coords scale as 1/length_scale, so doubling it
    # exactly halves them. Pins the normalization exponent (the >1.0 and
    # monotonicity checks above are blind to the scale magnitude).
    coords_1x = build_stereographic_token_coords(
        pos, (2, 2), d_patch=3, length_scale=1.0
    )
    coords_2x = build_stereographic_token_coords(
        pos, (2, 2), d_patch=3, length_scale=2.0
    )
    assert torch.allclose(coords_2x, coords_1x / 2, atol=1e-6)

    with pytest.raises(ValueError):
        build_stereographic_token_coords(pos, (2, 2), d_patch=3, length_scale=0.0)


# --------------------------------------------------------------------------- #
# Shape-variation: depth / horizontal / vertical-patch combinations
# --------------------------------------------------------------------------- #
@torch.no_grad()
@pytest.mark.parametrize(
    "shape,patch",
    [((4, 8, 8), (1, 2, 2)), ((6, 8, 8), (2, 2, 2)), ((2, 16, 8), (1, 4, 2))],
    ids=["pd1", "pd2", "anisotropic"],
)
def test_transformer_forward_varied_shapes(shape, patch):
    """Forward preserves shape across depth / horizontal / vertical-patch combos."""
    torch.manual_seed(0)
    d, h, w = shape
    model = StrataTransformer3D(
        in_channels=3,
        input_shape=shape,
        patch_size=patch,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        attn_kernel=-1,
        rope_mode="axial",
    )
    x = torch.randn(2, 3, d, h, w)
    assert model(x).shape == (2, 3, d, h, w)


# --------------------------------------------------------------------------- #
# torch.compile, bf16 autocast, and activation checkpointing
# --------------------------------------------------------------------------- #
@torch.no_grad()
def test_transformer_torch_compile_matches_eager():
    """torch.compile produces the same output as eager (SDPA path)."""
    torch.manual_seed(0)
    model = _seed_params(
        StrataTransformer3D(
            in_channels=4,
            input_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
        ),
        seed=1,
    ).eval()
    x = torch.randn(2, 4, 4, 8, 8)
    eager = model(x)
    compiled = torch.compile(model, fullgraph=False)(x)
    assert torch.allclose(eager, compiled, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_transformer_bf16_autocast_forward(device):
    """The model runs under bf16 autocast (and accepts bf16_mixed) with finite output."""
    torch.manual_seed(0)
    model = _seed_params(
        StrataTransformer3D(
            in_channels=4,
            input_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
            bf16_mixed=True,
        ),
        seed=1,
    ).to(device)
    x = torch.randn(2, 4, 4, 8, 8, device=device)
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.autocast(dev_type, dtype=torch.bfloat16):
        out = model(x)
    assert torch.isfinite(out.float()).all()


def _make_transformer_default():
    return StrataTransformer3D(
        in_channels=4,
        input_shape=(4, 8, 8),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        attn_kernel=-1,
    )


@torch.no_grad()
def test_transformer_include_head_toggle():
    """include_head=False omits the head: forward returns tokens, not a field."""
    torch.manual_seed(0)
    kw = dict(
        in_channels=3,
        input_shape=(4, 8, 8),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        attn_kernel=-1,
    )
    full = StrataTransformer3D(**kw).eval()
    headless = StrataTransformer3D(**kw, include_head=False).eval()
    x = torch.randn(2, 3, 4, 8, 8)
    assert full.final_layer is not None
    assert full(x).shape == (2, 3, 4, 8, 8)  # decoded field
    # Headless: no output head, forward returns post-block tokens (d*h*w, embed_dim).
    assert headless.final_layer is None
    assert headless(x).shape == (2, 4 * 4 * 4, 32)
    assert not any("final_layer" in n for n, _ in headless.named_parameters())


@torch.no_grad()
@pytest.mark.parametrize(
    "make",
    [_make_transformer_default, lambda: _build_strata_bilinear()[0]],
    ids=["transformer", "strata"],
)
def test_output_head_supports_pure_bf16_cast(make, device):
    """A model cast wholesale to bf16 (model.bfloat16()) runs without a dtype crash.

    Distinct from the bf16_mixed autocast path (weights stay fp32): here the
    parameters are genuinely bf16. The fp32-forced output head must match its
    input to the (bf16) weight dtype, else F.linear raises
    "mat1 and mat2 must have the same dtype". The Strata case also exercises
    the bf16 DepthwiseConv (bilinear_dw) path.
    """
    torch.manual_seed(0)
    model = make().to(device).bfloat16().eval()
    x = torch.randn(2, 4, 4, 8, 8, device=device, dtype=torch.bfloat16)
    out = model(x)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


@requires_module(["natten"])
@torch.no_grad()
def test_transformer_natten_bf16_cast(device):
    """A pure bf16 cast also survives the real NA3D (NATTEN) attention path.

    The other bf16-cast test uses the SDPA fallback (``attn_kernel=-1``); this one
    exercises the CUDA-only NATTEN neighborhood-attention kernel (``attn_kernel>0``)
    together with ``do_alt_depthwise_attn`` (depth-axis blocks) and
    ``gated_attention`` under genuine bf16 weights, confirming the NA3D forward,
    the depth-axis attention, the gate, and the fp32 output head all run in bf16.
    """
    if device == "cpu":
        pytest.skip("natten neighborhood attention is not available on CPU")
    torch.manual_seed(0)
    model = (
        StrataTransformer3D(
            in_channels=4,
            input_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=4,
            attn_kernel=3,  # > 0 -> real NA3D windowed attention
            do_alt_depthwise_attn=True,
            gated_attention=True,
        )
        .to(device)
        .bfloat16()
        .eval()
    )
    x = torch.randn(2, 4, 4, 8, 8, device=device, dtype=torch.bfloat16)
    out = model(x)
    assert out.dtype == torch.bfloat16
    assert out.shape == (2, 4, 4, 8, 8) and torch.isfinite(out.float()).all()


def test_transformer_activation_checkpointing_matches(device):
    """activation_checkpointing reproduces the non-checkpointed output and grads.

    Uses depth-axis-alternating blocks + stereographic RoPE so per-block
    ``rope_tables`` differ (depth blocks get ``None``): this exercises the
    closure's default-arg capture, which a homogeneous config could not.
    """
    torch.manual_seed(0)
    kwargs = dict(
        in_channels=4,
        input_shape=(4, 8, 8),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=3,
        attn_kernel=-1,
        do_alt_depthwise_attn=True,
        rope_mode="stereographic",
    )
    plain = _seed_params(
        StrataTransformer3D(**kwargs, activation_checkpointing=False), seed=1
    ).to(device)
    ckpt = _seed_params(
        StrataTransformer3D(**kwargs, activation_checkpointing=True), seed=1
    ).to(device)
    # Checkpointing only engages in train mode (drop rates default to 0, so the
    # forward is still deterministic and comparable to the plain model).
    plain.train()
    ckpt.train()
    assert ckpt._should_checkpoint_block(0), (
        "checkpointing must be active in train mode"
    )
    x = torch.randn(2, 4, 4, 8, 8, device=device)
    pos = _make_pos(2, 8, 8).to(device)
    y_plain = plain(x, pos)
    y_ckpt = ckpt(x, pos)
    assert torch.allclose(y_plain, y_ckpt, atol=1e-5)
    y_plain.pow(2).mean().backward()
    y_ckpt.pow(2).mean().backward()
    for (n, p_plain), (_, p_ckpt) in zip(
        plain.named_parameters(), ckpt.named_parameters()
    ):
        assert torch.allclose(p_plain.grad, p_ckpt.grad, atol=1e-4), n


# --------------------------------------------------------------------------- #
# Strata RoPE: the backbone stage (via backbone_config["rope_mode"]) and the
# pixel stage (rope_mode_pixel) are INDEPENDENT — every combination must work.
# Strata.forward routes `pos` to both stages. In particular a stereographic
# pixel stage must not depend on the backbone stage also being stereographic
# (the pixel coords must not dereference a possibly-None backbone RoPE module).
# --------------------------------------------------------------------------- #
@torch.no_grad()
@pytest.mark.parametrize(
    "sem_rope,pix_rope",
    [
        ("none", "none"),
        ("none", "axial"),
        ("none", "stereographic"),  # stereographic pixel stage, no backbone RoPE
        ("stereographic", "none"),
        ("axial", "stereographic"),
        ("stereographic", "stereographic"),
    ],
)
def test_strata_forward_rope_modes(device, sem_rope, pix_rope):
    """Backbone-stage and pixel-stage RoPE are independent across all combos."""
    torch.manual_seed(0)
    b, c, d, h, w = 2, 4, 4, 8, 8
    model = Strata(
        backbone_config=dict(
            in_channels=c,
            input_shape=(d, h, w),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
            rope_mode=sem_rope,
        ),
        embed_dim_pixel=16,
        num_layers_pixel=2,
        num_heads_pixel=2,
        attn_kernel_pixel=-1,
        adaln_mode="pixel_proj",
        rope_mode_pixel=pix_rope,
    ).to(device)
    x = torch.randn(b, c, d, h, w, device=device)
    needs_pos = "stereographic" in (sem_rope, pix_rope)
    pos = _make_pos(b, h, w).to(device) if needs_pos else None
    out = model(x, pos)
    assert out.shape == (b, c, d, h, w)
    assert torch.isfinite(out).all()
    # Axial pixel RoPE caches cos/sin buffers; stereographic builds them per
    # forward and "none" builds nothing.
    assert hasattr(model, "_rope_cos_pixel") == (pix_rope == "axial")


@torch.no_grad()
def test_strata_validates_pos_shape():
    """Strata.forward rejects a wrong-shaped pos even when only the pixel stage
    uses stereographic RoPE (the backbone, here rope_mode='none', won't check it)."""
    torch.manual_seed(0)
    model = Strata(
        backbone_config=dict(
            in_channels=3,
            input_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
        ),
        embed_dim_pixel=16,
        num_layers_pixel=2,
        num_heads_pixel=2,
        attn_kernel_pixel=-1,
        adaln_mode="pixel_proj",
        rope_mode_pixel="stereographic",
    ).eval()
    x = torch.randn(2, 3, 4, 8, 8)
    bad_pos = torch.randn(2, 2, 4, 8)  # wrong H/W; should be (2, 2, 8, 8)
    with pytest.raises(ValueError):
        model(x, bad_pos)


@torch.no_grad()
def test_forward_rejects_wrong_ndim():
    """Both models reject a non-5D input with a clean ValueError (not a cryptic
    unpack/conv error). Guards the StrataTransformer3D.forward and Strata.forward
    ndim checks."""
    backbone_cfg = dict(
        in_channels=3,
        input_shape=(4, 8, 8),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        attn_kernel=-1,
    )
    bad = torch.randn(2, 3, 8, 8)  # 4D: missing the depth axis
    transformer = StrataTransformer3D(**backbone_cfg).eval()
    with pytest.raises(ValueError):
        transformer(bad)
    model = Strata(
        backbone_config=backbone_cfg,
        embed_dim_pixel=16,
        num_layers_pixel=2,
        num_heads_pixel=2,
        attn_kernel_pixel=-1,
    ).eval()
    with pytest.raises(ValueError):
        model(bad)


@torch.no_grad()
@pytest.mark.parametrize("adaln_mode", ["pixel_proj", "bilinear_dw"])
def test_strata_identity_init(adaln_mode):
    """A freshly built Strata starts as an identity residual: the pixel blocks'
    attn/MLP linears are Xavier-initialized (matching the backbone), but the
    AdaLN projections and the output head are zero, so the forward is all-zero."""
    torch.manual_seed(0)
    model = Strata(
        backbone_config=dict(
            in_channels=4,
            input_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
        ),
        embed_dim_pixel=16,
        num_layers_pixel=2,
        num_heads_pixel=2,
        attn_kernel_pixel=-1,
        adaln_mode=adaln_mode,
    ).eval()
    # The pixel blocks' attn/MLP linears get the backbone's Xavier init, which
    # *zeros* the linear bias -- whereas PyTorch's default nn.Linear init leaves a
    # nonzero uniform bias. So a zeroed qkv bias proves the blocks were actually
    # initialized by initialize_weights (a weight!=0 check can't: default init is
    # nonzero too, so it would pass even if the Xavier pass were removed).
    qkv = model.pixel_blocks[0].attn.qkv
    assert qkv.bias is not None and (qkv.bias == 0).all()
    # ...but AdaLN-zero + zero output head => exact identity-residual (zero) output.
    assert (model.pixel_final_layer.linear.weight == 0).all()
    assert (model(torch.randn(2, 4, 4, 8, 8)) == 0).all()


@pytest.mark.parametrize("adaln_mode", ["pixel_proj", "bilinear_dw"])
def test_strata_activation_checkpointing_matches(device, adaln_mode):
    """Strata pixel-block checkpointing reproduces the non-checkpointed output/grads.

    ``first_block_only_adaln=True`` makes the pixel stack heterogeneous (one
    conditioning ``StrataPixel3DBlock`` + plain ``StrataTransformer3DBlock``s), so checkpointing
    exercises both closure branches; ``bilinear_dw`` additionally checkpoints the
    block that captures ``s_cond_bilinear``.
    """
    torch.manual_seed(0)
    kwargs = dict(
        backbone_config=dict(
            in_channels=4,
            input_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
        ),
        embed_dim_pixel=16,
        num_layers_pixel=3,
        num_heads_pixel=2,
        attn_kernel_pixel=-1,
        adaln_mode=adaln_mode,
        first_block_only_adaln=True,
    )
    plain = _seed_params(
        Strata(**kwargs, activation_checkpointing_pixel=False), seed=1
    ).to(device)
    ckpt = _seed_params(
        Strata(**kwargs, activation_checkpointing_pixel=True), seed=1
    ).to(device)
    # Checkpointing only engages in train mode (drop rates default to 0).
    plain.train()
    ckpt.train()
    assert ckpt._should_checkpoint_pixel_block(0), "checkpointing must be active"
    x = torch.randn(2, 4, 4, 8, 8, device=device)
    y_plain = plain(x)
    y_ckpt = ckpt(x)
    assert torch.allclose(y_plain, y_ckpt, atol=1e-5)
    y_plain.pow(2).mean().backward()
    y_ckpt.pow(2).mean().backward()
    # Looser grad tolerance than StrataTransformer3D: gradients through the chunked-vmap
    # DepthwiseConv (bilinear_dw) recomputed under checkpointing accumulate
    # ~1e-4 float noise on CUDA. The forward output above still matches exactly.
    for (n, p_plain), (_, p_ckpt) in zip(
        plain.named_parameters(), ckpt.named_parameters()
    ):
        assert torch.allclose(p_plain.grad, p_ckpt.grad, atol=1e-3, rtol=1e-3), n


def _make_axial_transformer(input_shape):
    return StrataTransformer3D(
        in_channels=3,
        input_shape=input_shape,
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        attn_kernel=-1,
        rope_mode="axial",
    ).eval()


@torch.no_grad()
def test_transformer_set_tile_size_axial():
    """set_tile_size rebuilds the axial RoPE buffers to match a fresh model.

    The buffers are compared against a model constructed directly at the new
    size, not just by token count: a wrong row/col assignment keeps the count
    but changes the table, which a shape-only check would miss.
    """
    torch.manual_seed(0)
    model = _make_axial_transformer((4, 8, 8))
    assert model(torch.randn(2, 3, 4, 8, 8)).shape == (2, 3, 4, 8, 8)
    # Re-tile to a taller grid and compare to a model built directly at (4, 16, 8).
    model.set_tile_size(height=16, width=8)
    fresh = _make_axial_transformer((4, 16, 8))
    assert model._rope_cos.shape[0] == 4 * (16 // 2) * (8 // 2)
    assert torch.equal(model._rope_cos, fresh._rope_cos)
    assert torch.equal(model._rope_sin, fresh._rope_sin)
    out = model(torch.randn(2, 3, 4, 16, 8))
    assert out.shape == (2, 3, 4, 16, 8) and torch.isfinite(out).all()
    # A tile not divisible by the patch size is rejected (mirrors the
    # constructor's check), rather than silently truncated by the patch conv.
    with pytest.raises(ValueError):
        model.set_tile_size(height=15, width=8)  # 15 % patch_size[1] (2) != 0


@torch.no_grad()
@pytest.mark.parametrize("rope_mode", ["stereographic", "none"])
def test_transformer_set_tile_size_retiles_non_axial(rope_mode):
    """set_tile_size also re-tiles non-axial modes (the regional/global use case).

    stereographic / none build their RoPE per forward, so re-tiling only needs
    the expected input shape updated; a forward at the *new* tile must then run,
    and one at the old tile is correctly rejected.
    """
    torch.manual_seed(0)
    model = StrataTransformer3D(
        in_channels=3,
        input_shape=(4, 8, 8),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        attn_kernel=-1,
        rope_mode=rope_mode,
    ).eval()
    model.set_tile_size(height=16, width=8)
    assert model.input_shape == (4, 16, 8)  # expected tile updated for all modes
    x = torch.randn(2, 3, 4, 16, 8)
    pos = _make_pos(2, 16, 8) if rope_mode == "stereographic" else None
    out = model(x, pos) if pos is not None else model(x)
    assert out.shape == (2, 3, 4, 16, 8) and torch.isfinite(out).all()
    # The old tile is now correctly rejected.
    with pytest.raises(ValueError):
        old_pos = _make_pos(2, 8, 8) if rope_mode == "stereographic" else None
        model(torch.randn(2, 3, 4, 8, 8), old_pos) if old_pos is not None else model(
            torch.randn(2, 3, 4, 8, 8)
        )
