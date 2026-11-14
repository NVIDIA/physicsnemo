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

"""Tests for FLARE attention layer."""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from physicsnemo.core.warnings import LegacyFeatureWarning
from physicsnemo.nn import FLARE, FLAREPlusPlus
from test.conftest import requires_module


def _load_or_create_output_reference(
    file_name: str, output: torch.Tensor
) -> torch.Tensor:
    """Load a local golden output, or create it and require a second run."""
    reference_path = Path(__file__).parent / "data" / file_name
    if not reference_path.exists():
        torch.save({"output": output.detach().cpu()}, reference_path)
        raise IOError(
            f"Golden output {reference_path} was missing and has been created; "
            "commit it and re-run the test."
        )
    reference = torch.load(reference_path, weights_only=True)
    return next(iter(reference.values())).to(output.device)


def test_flare_forward(device):
    """Test FLARE forward pass and output shape."""
    torch.manual_seed(42)
    flare = FLARE(dim=64, heads=4, dim_head=16, n_global_queries=32, use_te=False).to(
        device
    )
    x = torch.randn(2, 100, 64).to(device)
    out = flare(x)
    assert out.shape == (2, 100, 64)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("heads,dim_head", [(2, 32), (8, 8), (4, 16)])
def test_flare_configs(device, heads, dim_head):
    """Test FLARE with different head configurations."""
    torch.manual_seed(42)
    dim = heads * dim_head
    flare = FLARE(
        dim=dim, heads=heads, dim_head=dim_head, n_global_queries=16, use_te=False
    ).to(device)
    x = torch.randn(2, 50, dim).to(device)
    out = flare(x)
    assert out.shape == x.shape


@requires_module("transformer_engine>=2.14.0")
def test_flare_use_te_forward_backward(device):
    """Test TE cross-attention with unequal global and token sequence lengths."""
    if device == "cpu":
        pytest.skip("Transformer Engine requires CUDA")

    torch.manual_seed(42)
    flare = FLARE(
        dim=64,
        heads=4,
        dim_head=16,
        dropout=0.25,
        n_global_queries=7,
        use_te=True,
    ).to(device)
    x = torch.randn(2, 19, 64, device=device, requires_grad=True)

    assert flare.attn_fn.attention_dropout == 0.0
    assert flare.out_dropout.p == 0.25

    out = flare(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()

    out.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_flare_gradient_flow(device):
    """Test gradient flow through FLARE."""
    torch.manual_seed(42)
    flare = FLARE(dim=32, heads=4, dim_head=8, use_te=False).to(device)
    x = torch.randn(2, 20, 32, device=device, requires_grad=True)
    out = flare(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"dim": 64}, (8, 8, 64, 8**-0.5)),
        (
            {
                "dim": 24,
                "heads": 3,
                "dim_head": 8,
                "n_global_queries": 5,
                "attn_scale": 0.25,
            },
            (3, 8, 5, 0.25),
        ),
    ],
    ids=("defaults", "custom"),
)
def test_flare_plus_plus_constructor(kwargs, expected):
    """Default and custom constructor values are exposed consistently."""
    attention = FLAREPlusPlus(**kwargs)
    heads, dim_head, n_queries, scale = expected
    dim = kwargs["dim"]

    assert attention.dim == dim
    assert attention.heads == heads
    assert attention.dim_head == dim_head
    assert attention.n_global_queries == n_queries
    assert attention.scale == pytest.approx(scale)
    assert attention.latent_q0.shape == (heads, n_queries, dim_head)
    assert attention.latent_q_fixed.shape == (heads, n_queries, dim_head)
    assert attention.gate_logit.shape == (heads,)
    # Three input projections (synthesis K/V, physical K, physical V) plus
    # the output projection.
    assert attention.in_projection.out_features == 3 * dim
    projection_weights = (
        attention.in_projection.weight.numel() + attention.out_linear.weight.numel()
    )
    assert projection_weights == 4 * dim**2


def test_flare_plus_plus_initialization():
    """Seeds, gate, norms, and near-identity projections follow the v2 design."""
    torch.manual_seed(3)
    dim = 32
    attention = FLAREPlusPlus(dim=dim, heads=4, n_global_queries=7)

    torch.testing.assert_close(
        attention.gate_logit, torch.full((4,), FLAREPlusPlus._GATE_LOGIT_INIT)
    )
    for latent in (attention.latent_q0, attention.latent_q_fixed):
        assert latent.std().item() == pytest.approx(0.02, rel=0.25)
    for norm in (attention.q0_norm, attention.k_norm):
        assert torch.equal(norm.weight, torch.ones(8))
        assert torch.equal(norm.bias, torch.zeros(8))

    weight = attention.in_projection.weight.detach().view(3, dim, dim)
    noise = weight - torch.eye(dim)
    assert noise.abs().max().item() < 0.2
    assert noise.std().item() == pytest.approx(0.02, rel=0.25)
    assert torch.equal(attention.in_projection.bias, torch.zeros(3 * dim))


def test_flare_plus_plus_forward_accuracy(device):
    """A freshly initialized FLARE++ attention layer matches its golden output."""
    torch.manual_seed(1234)
    attention = FLAREPlusPlus(
        dim=24,
        heads=3,
        dim_head=8,
        n_global_queries=5,
    ).to(device)
    x = torch.randn(2, 17, 24).to(device)
    with torch.no_grad():
        output = attention(x)
    reference = _load_or_create_output_reference(
        "flare_plus_plus_attention_output.pth", output
    )

    torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-3)


def _reference_flare_plus_plus(attention, x):
    """Reference FLARE++ v2 mixer written out step by step."""
    batch, tokens, dim = x.shape
    heads, dim_head = attention.heads, attention.dim_head

    def split_heads(tensor):
        return tensor.reshape(batch, tokens, heads, dim_head).transpose(1, 2)

    def head_norm(tensor):
        return F.layer_norm(tensor, (dim_head,))

    k0_weight, k_weight, v_weight = attention.in_projection.weight.chunk(3, dim=0)
    k0_bias, k_bias, v_bias = attention.in_projection.bias.chunk(3, dim=0)
    k0 = head_norm(split_heads(F.linear(x, k0_weight, k0_bias)))
    k = attention.k_norm(split_heads(F.linear(x, k_weight, k_bias)))
    v = split_heads(F.linear(x, v_weight, v_bias))

    q0 = attention.q0_norm(attention.latent_q0).expand(batch, -1, -1, -1)
    q_dynamic = F.scaled_dot_product_attention(q0, k0, k0, scale=attention.scale)
    q_fixed = head_norm(attention.latent_q_fixed).expand(batch, -1, -1, -1)
    gate = torch.sigmoid(attention.gate_logit).view(1, heads, 1, 1)
    q = q_fixed + gate * q_dynamic

    z = F.scaled_dot_product_attention(q, k, v, scale=attention.scale)
    y = F.scaled_dot_product_attention(k, q, z, scale=attention.scale)
    y = y.transpose(1, 2).reshape(batch, tokens, dim)
    return attention.out_dropout(attention.out_linear(y)), q_dynamic


def test_flare_plus_plus_matches_paper_equations(device):
    """FLARE++ matches the per-block operation sequence of the v2 paper."""
    torch.manual_seed(7)
    attention = FLAREPlusPlus(
        dim=24,
        heads=3,
        dim_head=8,
        n_global_queries=5,
        dropout=0.0,
    ).to(device)
    attention.eval()
    # Move parameters away from their initial values so every term matters.
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    x = torch.randn(2, 11, 24, device=device)

    actual = attention(x)
    expected, _ = _reference_flare_plus_plus(attention, x)

    torch.testing.assert_close(actual, expected)


def test_flare_plus_plus_closed_gate_uses_fixed_queries(device):
    """With the gate closed, the synthesized queries do not affect the output."""
    torch.manual_seed(9)
    attention = FLAREPlusPlus(dim=16, heads=2, n_global_queries=4).to(device)
    attention.eval()
    with torch.no_grad():
        attention.gate_logit.fill_(-1e4)
    x = torch.randn(2, 13, 16, device=device)

    expected = attention(x)
    with torch.no_grad():
        attention.latent_q0.add_(torch.randn_like(attention.latent_q0))
    torch.testing.assert_close(attention(x), expected)


def test_flare_plus_plus_routing_is_input_conditioned(device):
    """Different samples synthesize different routing queries."""
    torch.manual_seed(11)
    attention = FLAREPlusPlus(
        dim=16,
        heads=2,
        dim_head=8,
        n_global_queries=4,
    ).to(device)
    x = torch.stack(
        (
            torch.randn(9, 16, device=device),
            torch.randn(9, 16, device=device),
        )
    )
    _, dynamic_queries = _reference_flare_plus_plus(attention, x)

    assert not torch.allclose(dynamic_queries[0], dynamic_queries[1])


def test_flare_plus_plus_eval_is_rng_free(device):
    """Standalone FLARE++ attention is deterministic during evaluation."""
    torch.manual_seed(43)
    attention = FLAREPlusPlus(
        dim=32,
        heads=4,
        dim_head=8,
        n_global_queries=7,
        dropout=0.4,
    ).to(device)
    x = torch.randn(2, 23, 32, device=device)

    attention.eval()
    model_device = next(attention.parameters()).device
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = (
        torch.cuda.get_rng_state(model_device).clone()
        if model_device.type == "cuda"
        else None
    )
    with torch.no_grad():
        output_1 = attention(x)
        output_2 = attention(x)

    assert torch.equal(output_1, output_2)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    if cuda_rng_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(model_device), cuda_rng_before)


def test_flare_plus_plus_gradient_flow(device):
    """Gradients reach the input and every learned parameter."""
    attention = FLAREPlusPlus(
        dim=32,
        heads=4,
        dim_head=8,
        n_global_queries=7,
    ).to(device)
    x = torch.randn(2, 20, 32, device=device, requires_grad=True)
    attention(x).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, parameter in attention.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_flare_plus_plus_validates_options():
    """Unsupported backends and invalid scales fail during construction."""
    with pytest.raises(ValueError, match="does not support Transformer Engine"):
        FLAREPlusPlus(dim=32, use_te=True)

    for scale in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="attn_scale"):
            FLAREPlusPlus(dim=32, attn_scale=scale)

    with pytest.raises(ValueError, match="divisible by heads"):
        FLAREPlusPlus(dim=30, heads=4)
    with pytest.raises(ValueError, match="heads \\* dim_head == dim"):
        FLAREPlusPlus(dim=32, heads=4, dim_head=16)


def test_flare_plus_plus_torch_compile_fullgraph(device):
    """FLARE++ can be captured by torch.compile with fullgraph enabled."""
    torch._dynamo.config.error_on_recompile = True
    attention = FLAREPlusPlus(
        dim=16,
        heads=2,
        dim_head=8,
        n_global_queries=4,
    ).to(device)
    x = torch.randn(2, 13, 16, device=device)
    expected = attention(x)
    backend = "inductor" if str(device).startswith("cuda") else "aot_eager"
    compiled = torch.compile(attention, backend=backend, fullgraph=True)

    actual = compiled(x)
    repeated = compiled(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(repeated, expected)


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
