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

"""Tests for the FLARE and FLARE++ attention layers."""

import re
import subprocess
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.core.warnings import LegacyFeatureWarning
from physicsnemo.nn import FLARE, FLAREPlusPlus
from test.conftest import requires_module

_MIXERS = {
    "flare": (FLARE, {}),
    "flare_qk_norm": (FLARE, {"qk_norm": True}),
    "flarepp": (FLAREPlusPlus, {}),
    "flarepp_unshared": (
        FLAREPlusPlus,
        {"share_k0_v0": False, "k_norm": False, "q_fixed_norm": False},
    ),
}


def _projections(mixer: nn.Module) -> list[nn.Linear]:
    names = ("k0_proj", "v0_proj", "k_proj", "v_proj")
    return [getattr(mixer, n) for n in names if getattr(mixer, n, None) is not None]


@pytest.mark.parametrize("name", list(_MIXERS))
@pytest.mark.parametrize("use_rmsnorm", [True, False])
def test_forward_backward(device, name, use_rmsnorm):
    """Shapes, finite outputs and gradients to the input and every parameter."""
    cls, kwargs = _MIXERS[name]
    torch.manual_seed(0)
    mixer = cls(
        dim=32, heads=4, n_global_queries=6, use_rmsnorm=use_rmsnorm, **kwargs
    ).to(device)
    x = torch.randn(2, 19, 32, device=device, requires_grad=True)
    out = mixer(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for pname, param in mixer.named_parameters():
        assert param.grad is not None, pname
        assert torch.isfinite(param.grad).all(), pname


@pytest.mark.parametrize("name", list(_MIXERS))
def test_norm_type(name):
    """use_rmsnorm selects RMSNorm (eps 1e-6) or LayerNorm for the head norms."""
    cls, kwargs = _MIXERS[name]
    for use_rmsnorm, norm_cls in ((True, nn.RMSNorm), (False, nn.LayerNorm)):
        mixer = cls(dim=32, heads=4, use_rmsnorm=use_rmsnorm, **kwargs)
        norms = [
            m for m in mixer.modules() if isinstance(m, (nn.RMSNorm, nn.LayerNorm))
        ]
        assert all(type(m) is norm_cls for m in norms)
        assert norms or name == "flare"
        if use_rmsnorm:
            assert all(m.eps == 1e-6 for m in norms)


def test_flare_qk_norm_modules():
    """qk_norm adds affine per-head norms on latent queries and keys."""
    plain = FLARE(dim=32, heads=4)
    assert isinstance(plain.q_norm, nn.Identity)
    assert isinstance(plain.k_norm, nn.Identity)
    normed = FLARE(dim=32, heads=4, qk_norm=True)
    for norm in (normed.q_norm, normed.k_norm):
        assert isinstance(norm, nn.RMSNorm)
        assert norm.normalized_shape == (8,)
        assert norm.weight is not None


def test_flarepp_norm_modules():
    """FLARE++ norm layout: q0/k affine, k0/v0/q_fixed without affine."""
    mixer = FLAREPlusPlus(dim=32, heads=4)
    assert mixer.q0_norm.weight is not None
    assert mixer.k_norm.weight is not None
    for norm in (mixer.k0_norm, mixer.v0_norm, mixer.q_fixed_norm):
        assert isinstance(norm, nn.RMSNorm) and norm.weight is None
    assert mixer.v0_proj is None
    off = FLAREPlusPlus(dim=32, heads=4, k_norm=False, q_fixed_norm=False)
    assert isinstance(off.k_norm, nn.Identity)
    assert isinstance(off.q_fixed_norm, nn.Identity)


@pytest.mark.parametrize("name", list(_MIXERS))
def test_identity_plus_noise_init(name):
    """K/V projections start at identity plus small noise with zero bias."""
    cls, kwargs = _MIXERS[name]
    torch.manual_seed(0)
    mixer = cls(dim=64, heads=4, **kwargs)
    projections = _projections(mixer)
    assert projections
    for proj in projections:
        noise = proj.weight - torch.eye(64)
        assert noise.abs().max() < 0.15
        assert 0.015 < noise.std() < 0.025
        assert torch.all(proj.bias == 0)
        assert proj._skip_weight_init


def test_latent_init_and_shapes():
    """Latent queries are N(0, 0.02) with shape (heads, M, dim_head)."""
    torch.manual_seed(0)
    flare = FLARE(dim=128, heads=8, n_global_queries=64)
    assert flare.latent_q.shape == (8, 64, 16)
    assert 0.018 < flare.latent_q.std() < 0.022
    mixer = FLAREPlusPlus(dim=128, heads=8, n_global_queries=64, gate_logit_init=0.5)
    assert mixer.latent_q0.shape == mixer.latent_q_fixed.shape == (8, 64, 16)
    torch.testing.assert_close(mixer.gate_logit, torch.full((8,), 0.5))


def test_no_weight_decay():
    """Each mixer declares its latent parameters as no-weight-decay."""
    flare = FLARE(dim=32, heads=4)
    assert flare.no_weight_decay() == {"latent_q"}
    mixer = FLAREPlusPlus(dim=32, heads=4)
    assert mixer.no_weight_decay() == {"latent_q0", "latent_q_fixed", "gate_logit"}
    for module in (flare, mixer):
        params = dict(module.named_parameters(recurse=False))
        assert module.no_weight_decay() <= params.keys()


def test_flare_matches_reference():
    """FLARE equals the two-pass SDPA reference computation."""
    torch.manual_seed(0)
    flare = FLARE(dim=32, heads=4, n_global_queries=5, qk_norm=True).eval()
    x = torch.randn(2, 11, 32)

    def heads(t):
        return t.reshape(2, 11, 4, 8).transpose(1, 2)

    k = flare.k_norm(heads(flare.k_proj(x)))
    v = heads(flare.v_proj(x))
    q = flare.q_norm(flare.latent_q).expand(2, -1, -1, -1)
    z = F.scaled_dot_product_attention(q, k, v, scale=8**-0.5)
    y = F.scaled_dot_product_attention(k, q, z, scale=8**-0.5)
    expected = flare.out_proj(y.transpose(1, 2).reshape(2, 11, 32))
    torch.testing.assert_close(flare(x), expected)


def test_flarepp_gate_mixes_dynamic_queries():
    """A very negative gate logit reduces FLARE++ to fixed-query FLARE routing."""
    torch.manual_seed(0)
    mixer = FLAREPlusPlus(dim=32, heads=4, n_global_queries=5).eval()
    x = torch.randn(2, 11, 32)
    with torch.no_grad():
        mixer.gate_logit.fill_(-1e4)

    def heads(t):
        return t.reshape(2, 11, 4, 8).transpose(1, 2)

    k = mixer.k_norm(heads(mixer.k_proj(x)))
    v = heads(mixer.v_proj(x))
    q = mixer.q_fixed_norm(mixer.latent_q_fixed).expand(2, -1, -1, -1)
    z = F.scaled_dot_product_attention(q, k, v, scale=8**-0.5)
    y = F.scaled_dot_product_attention(k, q, z, scale=8**-0.5)
    expected = mixer.out_proj(y.transpose(1, 2).reshape(2, 11, 32))
    torch.testing.assert_close(mixer(x), expected)
    with torch.no_grad():
        mixer.gate_logit.fill_(0.25)
    assert not torch.allclose(mixer(x), expected)


@pytest.mark.parametrize("name", list(_MIXERS))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_autocast(device, name, dtype):
    """Half-precision autocast forward and backward stay finite."""
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 autocast is CUDA-only")
    cls, kwargs = _MIXERS[name]
    torch.manual_seed(0)
    mixer = cls(dim=32, heads=4, **kwargs).to(device)
    x = torch.randn(2, 19, 32, device=device)
    with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
        out = mixer(x)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    out.float().sum().backward()


@pytest.mark.parametrize("name", list(_MIXERS))
def test_low_precision_module(device, name):
    """The mixers run with half-precision parameters and inputs."""
    cls, kwargs = _MIXERS[name]
    mixer = cls(dim=32, heads=4, **kwargs).to(device=device, dtype=torch.bfloat16)
    out = mixer(torch.randn(2, 19, 32, device=device, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"dim": 32, "heads": 4, "dim_head": 16}, "heads \\* dim_head"),
        ({"dim": 30, "heads": 4}, "divisible"),
        ({"dim": 32, "heads": 4, "n_global_queries": 0}, "n_global_queries"),
    ],
)
@pytest.mark.parametrize("cls", [FLARE, FLAREPlusPlus])
def test_invalid_arguments(cls, kwargs, match):
    with pytest.raises(ValueError, match=match):
        cls(**kwargs)


def test_flarepp_invalid_gate():
    with pytest.raises(ValueError, match="gate_logit_init"):
        FLAREPlusPlus(dim=32, heads=4, gate_logit_init=float("nan"))


@pytest.mark.parametrize("name", list(_MIXERS))
def test_torch_compile(device, name):
    """torch.compile(fullgraph=True) matches eager."""
    cls, kwargs = _MIXERS[name]
    torch.manual_seed(0)
    mixer = cls(dim=32, heads=4, **kwargs).to(device).eval()
    x = torch.randn(2, 19, 32, device=device)
    expected = mixer(x)
    compiled = torch.compile(mixer, fullgraph=True)
    torch.testing.assert_close(compiled(x), expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("name", list(_MIXERS))
def test_use_te_requires_transformer_engine(name, monkeypatch):
    """use_te=True fails clearly when Transformer Engine is missing."""
    import physicsnemo.nn.module.flare_attention as fa

    cls, kwargs = _MIXERS[name]

    class _Missing:
        available = False

    monkeypatch.setattr(fa, "te", _Missing())
    with pytest.raises(ImportError, match="Transformer Engine"):
        cls(dim=32, heads=4, use_te=True, **kwargs)


@requires_module("transformer_engine>=2.14.0")
@pytest.mark.parametrize("name", list(_MIXERS))
@pytest.mark.parametrize("use_rmsnorm", [True, False])
def test_use_te_matches_pytorch(device, name, use_rmsnorm):
    """The Transformer Engine backend computes the same mixer as the SDPA backend."""
    if device == "cpu":
        pytest.skip("Transformer Engine requires CUDA")
    import transformer_engine.pytorch as te

    cls, kwargs = _MIXERS[name]
    torch.manual_seed(0)
    common = dict(dim=64, heads=4, n_global_queries=7, use_rmsnorm=use_rmsnorm)
    ref = cls(**common, **kwargs).to(device)
    mixer = cls(**common, use_te=True, **kwargs).to(device)
    assert mixer.use_te and isinstance(mixer.attn_fn, te.DotProductAttention)
    for proj in _projections(mixer):
        assert isinstance(proj, te.Linear) and proj._skip_weight_init
        noise = proj.weight.detach() - torch.eye(64, device=device)
        assert noise.abs().max() < 0.15

    # Parameter names are backend independent; TE adds only _extra_state.
    result = mixer.load_state_dict(ref.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert all(k.endswith("_extra_state") for k in result.missing_keys)

    x = torch.randn(2, 19, 64, device=device)
    x_ref, x_te = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    out_ref, out_te = ref(x_ref), mixer(x_te)
    torch.testing.assert_close(out_te, out_ref, atol=2e-3, rtol=2e-3)
    out_ref.square().mean().backward()
    out_te.square().mean().backward()
    torch.testing.assert_close(x_te.grad, x_ref.grad, atol=2e-3, rtol=2e-3)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = mixer(x)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), out_ref.detach(), atol=5e-2, rtol=5e-2)


def test_flare_attention_legacy_import_paths():
    """Test the import paths used before the move out of experimental."""
    # Drop the cached legacy module so the shim warning fires again.
    sys.modules.pop("physicsnemo.experimental.nn.flare_attention", None)

    with pytest.warns(
        LegacyFeatureWarning, match=re.escape("from physicsnemo.nn import FLARE")
    ):
        from physicsnemo.experimental.nn.flare_attention import (
            FLARE as LegacyModuleFLARE,
        )
    with pytest.warns(
        LegacyFeatureWarning, match=re.escape("from physicsnemo.nn import FLARE")
    ):
        from physicsnemo.experimental.nn import FLARE as LegacyPackageFLARE

    assert LegacyPackageFLARE is FLARE
    assert LegacyModuleFLARE is FLARE


def test_experimental_nn_import_does_not_warn():
    """Importing physicsnemo.experimental.nn alone must not raise the FLARE shim warning.

    Runs in a subprocess because a module body executes once per process: by the
    time this test runs, the modules are already cached, so an in-process check
    would pass regardless of what the package imports.
    """
    snippet = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import physicsnemo.experimental.nn\n"
        "leaked = [str(w.message) for w in caught if 'FLARE' in str(w.message)]\n"
        "assert not leaked, leaked\n"
    )
    subprocess.run(  # noqa: S603 - interpreter and snippet are test constants
        [sys.executable, "-c", snippet],
        check=True,
        capture_output=True,
        text=True,
    )
