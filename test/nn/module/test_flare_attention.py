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

import pytest
import torch

from physicsnemo.core.warnings import LegacyFeatureWarning
from physicsnemo.nn import FLARE
from test.conftest import requires_module


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


def test_flare_context_disabled_by_default(device):
    """Test that default construction creates no context parameters and rejects a context."""
    torch.manual_seed(42)
    flare = FLARE(dim=64, heads=4, dim_head=16, n_global_queries=32, use_te=False).to(
        device
    )
    assert flare.context_dim == 0
    assert not any("cross_" in name for name, _ in flare.named_parameters())

    x = torch.randn(2, 100, 64).to(device)
    out = flare(x)
    assert out.shape == (2, 100, 64)

    context = torch.randn(2, 4, 10, 8).to(device)
    with pytest.raises(ValueError, match="context_dim > 0"):
        flare(x, context)


def test_flare_context_changes_output(device):
    """Test that passing a context changes the output but preserves its shape."""
    torch.manual_seed(42)
    flare = FLARE(
        dim=64, heads=4, dim_head=16, n_global_queries=32, use_te=False, context_dim=8
    ).to(device)
    x = torch.randn(2, 100, 64).to(device)
    context = torch.randn(2, 4, 10, 8).to(device)

    out_ctx = flare(x, context)
    out_plain = flare(x)
    assert out_ctx.shape == (2, 100, 64)
    assert not torch.allclose(out_ctx, out_plain)


def test_flare_context_none_matches_disabled(device):
    """Test that an enabled layer without a context matches the disabled layer."""
    torch.manual_seed(42)
    flare_off = FLARE(
        dim=64, heads=4, dim_head=16, n_global_queries=32, use_te=False
    ).to(device)
    flare_on = FLARE(
        dim=64, heads=4, dim_head=16, n_global_queries=32, use_te=False, context_dim=8
    ).to(device)
    # Align the shared weights; only the cross projections are extra
    flare_on.load_state_dict(flare_off.state_dict(), strict=False)

    x = torch.randn(2, 100, 64).to(device)
    assert torch.equal(flare_on(x), flare_off(x))


def test_flare_context_zeroed_value_projection(device):
    """Test that zeroing the context-value projection makes the read a no-op."""
    torch.manual_seed(42)
    flare = FLARE(
        dim=64, heads=4, dim_head=16, n_global_queries=32, use_te=False, context_dim=8
    ).to(device)
    with torch.no_grad():
        flare.cross_v.weight.zero_()
        flare.cross_v.bias.zero_()

    x = torch.randn(2, 100, 64).to(device)
    context = torch.randn(2, 4, 10, 8).to(device)
    assert torch.equal(flare(x, context), flare(x))


def test_flare_context_invalid_shape(device):
    """Test that forward rejects mismatched context shapes."""
    torch.manual_seed(42)
    flare = FLARE(
        dim=64, heads=4, dim_head=16, n_global_queries=32, use_te=False, context_dim=8
    ).to(device)
    x = torch.randn(2, 50, 64).to(device)

    with pytest.raises(ValueError, match="Expected context"):
        flare(x, torch.randn(2, 4, 10, 4).to(device))  # wrong context_dim
    with pytest.raises(ValueError, match="Expected context"):
        flare(x, torch.randn(2, 2, 10, 8).to(device))  # wrong number of heads
    with pytest.raises(ValueError, match="Expected context"):
        flare(x, torch.randn(2, 4, 8).to(device))  # missing heads axis


def test_flare_context_gradient_flow(device):
    """Test gradient flow to the context projections and the context input."""
    torch.manual_seed(42)
    flare = FLARE(dim=32, heads=4, dim_head=8, use_te=False, context_dim=8).to(device)
    x = torch.randn(2, 20, 32, device=device, requires_grad=True)
    context = torch.randn(2, 4, 6, 8, device=device, requires_grad=True)

    out = flare(x, context)
    out.sum().backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert context.grad is not None
    assert not torch.isnan(context.grad).any()
    for proj in (flare.cross_q, flare.cross_k, flare.cross_v):
        assert proj.weight.grad is not None
        assert not torch.isnan(proj.weight.grad).any()


@requires_module("transformer_engine>=2.14.0")
def test_flare_use_te_context_forward_backward(device):
    """Test the TE-backend context read at the latent bottleneck."""
    if device == "cpu":
        pytest.skip("Transformer Engine requires CUDA")

    torch.manual_seed(42)
    flare = FLARE(
        dim=64,
        heads=4,
        dim_head=16,
        n_global_queries=7,
        use_te=True,
        context_dim=8,
    ).to(device)
    x = torch.randn(2, 19, 64, device=device, requires_grad=True)
    context = torch.randn(2, 4, 5, 8, device=device, requires_grad=True)

    out = flare(x, context)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()

    out.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert context.grad is not None
    assert not torch.isnan(context.grad).any()


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
