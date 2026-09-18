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
    """Load a local golden output, creating it from this test when absent."""
    reference_path = Path(__file__).parent / "data" / file_name
    if not reference_path.exists():
        torch.save({"output": output.detach().cpu()}, reference_path)
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
        ({"dim": 64}, (8, 64, 64, 64**-0.5)),
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

    assert attention.dim == kwargs["dim"]
    assert attention.heads == heads
    assert attention.dim_head == dim_head
    assert attention.n_global_queries == n_queries
    assert attention.scale == pytest.approx(scale)
    assert attention.q_seed.shape == (1, heads, n_queries, dim_head)
    assert attention.in_projection.out_features == 4 * heads * dim_head
    assert attention.out_linear.in_features == heads * dim_head
    if heads * dim_head == kwargs["dim"]:
        projection_weights = (
            attention.in_projection.weight.numel() + attention.out_linear.weight.numel()
        )
        assert projection_weights == 5 * kwargs["dim"] ** 2


def test_flare_plus_plus_forward_accuracy(device):
    """A freshly initialized FLARE++ attention layer matches its golden output."""
    torch.manual_seed(1234)
    attention = FLAREPlusPlus(
        dim=24,
        heads=3,
        dim_head=8,
        n_global_queries=5,
    ).to(device)
    x = torch.randn(2, 17, 24, device=device)
    with torch.no_grad():
        output = attention(x)
    reference = _load_or_create_output_reference(
        "flare_plus_plus_attention_output.pth", output
    )

    torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-3)


def test_flare_plus_plus_matches_paper_equations(device):
    """FLARE++ matches the three-SDPA formulation from the paper."""
    torch.manual_seed(7)
    attention = FLAREPlusPlus(
        dim=24,
        heads=3,
        dim_head=8,
        n_global_queries=5,
        dropout=0.0,
    ).to(device)
    attention.eval()
    x = torch.randn(2, 11, 24, device=device)

    actual = attention(x)
    query_k, query_v, physical_k, physical_v = attention.in_projection(x).chunk(
        4, dim=-1
    )

    def split_heads(tensor):
        return tensor.reshape(2, 11, 3, 8).transpose(1, 2)

    query_k, query_v, physical_k, physical_v = map(
        split_heads, (query_k, query_v, physical_k, physical_v)
    )
    queries = F.scaled_dot_product_attention(
        attention.q_seed.expand(2, -1, -1, -1),
        query_k,
        query_v,
        scale=attention.scale,
    )
    routed_values = F.scaled_dot_product_attention(
        queries, physical_k, physical_v, scale=attention.scale
    )
    expected = F.scaled_dot_product_attention(
        physical_k, queries, routed_values, scale=attention.scale
    )
    expected = expected.transpose(1, 2).reshape(2, 11, 24)
    expected = attention.out_dropout(attention.out_linear(expected))

    torch.testing.assert_close(actual, expected)


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
            torch.zeros(9, 16, device=device),
            torch.ones(9, 16, device=device),
        )
    )
    query_k, query_v, _, _ = attention.in_projection(x).chunk(4, dim=-1)
    query_k = query_k.reshape(2, 9, 2, 8).transpose(1, 2)
    query_v = query_v.reshape(2, 9, 2, 8).transpose(1, 2)
    queries = F.scaled_dot_product_attention(
        attention.q_seed.expand(2, -1, -1, -1),
        query_k,
        query_v,
        scale=attention.scale,
    )

    assert not torch.allclose(queries[0], queries[1])


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
    """Gradients reach the input, learned seeds, and fused projections."""
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
    for parameter in (
        attention.q_seed,
        attention.in_projection.weight,
        attention.out_linear.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_flare_plus_plus_validates_options():
    """Unsupported backends and invalid scales fail during construction."""
    with pytest.raises(ValueError, match="does not support Transformer Engine"):
        FLAREPlusPlus(dim=32, use_te=True)

    for scale in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="attn_scale"):
            FLAREPlusPlus(dim=32, attn_scale=scale)


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
