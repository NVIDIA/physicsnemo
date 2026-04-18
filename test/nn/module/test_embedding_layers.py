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

from typing import Any

import pytest
import torch

from physicsnemo.nn import (
    FourierEmbedding,
    OneHotEmbedding,
    PositionalEmbedding,
    SinusoidalTimestepEmbedding,
)
from test.common import validate_forward_accuracy

# ---------------------------------------------------------------------------
# PositionalEmbedding – reference-data accuracy (existing tests)
# ---------------------------------------------------------------------------

POSITIONAL_CONFIGS = [
    {
        "num_channels": 32,
        "max_positions": 10000,
        "endpoint": False,
        "learnable": False,
        "freq_embed_dim": None,
        "mlp_hidden_dim": None,
        "embed_fn": "cos_sin",
    },  # default
    {
        "num_channels": 128,
        "max_positions": 10000,
        "endpoint": False,
        "learnable": True,
        "freq_embed_dim": 128,
        "mlp_hidden_dim": 256,
        "embed_fn": "np_sin_cos",
    },
    {
        "num_channels": 128,
        "max_positions": 8192,
        "endpoint": True,
        "learnable": False,
        "freq_embed_dim": 128,
        "mlp_hidden_dim": 256,
        "embed_fn": "np_sin_cos",
    },
]


@pytest.mark.parametrize("config", POSITIONAL_CONFIGS)
@pytest.mark.parametrize("batch_size", [1, 4, 17])
def test_positional_embedding(device, config: dict[str, Any], batch_size):
    torch.manual_seed(7)
    target_device = torch.device(device)
    model = PositionalEmbedding(**config).to(target_device)
    model.eval()

    positions = torch.linspace(
        0,
        config["max_positions"] - 1,
        steps=batch_size,
        device=target_device,
        dtype=torch.float32,
    )

    def _fmt(value):
        return "none" if value is None else str(value)

    file_name = (
        "nn/module/data/"
        "positional_embedding_"
        f"c{config['num_channels']}_"
        f"max{config['max_positions']}_"
        f"endpoint{int(config['endpoint'])}_"
        f"learnable{int(config['learnable'])}_"
        f"freq{_fmt(config['freq_embed_dim'])}_"
        f"mlp{_fmt(config['mlp_hidden_dim'])}_"
        f"{config['embed_fn']}_"
        f"bs{batch_size}.pth"
    )

    # Tack this on for the test, since model is not a physicsnemo Module:
    model.device = target_device

    assert validate_forward_accuracy(
        model,
        (positions,),
        file_name=file_name,
        rtol=1e-3,
        atol=1e-3,
    )


# ---------------------------------------------------------------------------
# PositionalEmbedding – output shape & embed_fn ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("learnable", [False, True])
@pytest.mark.parametrize("embed_fn", ["cos_sin", "np_sin_cos"])
def test_positional_embedding_shape(device, learnable, embed_fn):
    num_channels = 64
    model = PositionalEmbedding(
        num_channels=num_channels,
        learnable=learnable,
        embed_fn=embed_fn,
    ).to(device)
    model.eval()

    x = torch.linspace(0, 100, steps=8, device=device)
    out = model(x)

    assert out.shape == (8, num_channels)


def test_positional_embedding_embed_fn_order(device):
    """cos_sin and np_sin_cos should produce different column ordering."""
    num_channels = 32
    model_cs = PositionalEmbedding(num_channels=num_channels, embed_fn="cos_sin").to(
        device
    )
    model_sc = PositionalEmbedding(num_channels=num_channels, embed_fn="np_sin_cos").to(
        device
    )
    # Both share the same freqs buffer – copy so comparison is fair.
    model_sc.load_state_dict(model_cs.state_dict())

    x = torch.tensor([1.0, 2.0], device=device)
    out_cs = model_cs(x)
    out_sc = model_sc(x)

    half = num_channels // 2
    # cos_sin: first half = cos, second half = sin
    # np_sin_cos: first half = sin, second half = cos
    torch.testing.assert_close(out_cs[:, :half], out_sc[:, half:])  # cos parts
    torch.testing.assert_close(out_cs[:, half:], out_sc[:, :half])  # sin parts


# ---------------------------------------------------------------------------
# PositionalEmbedding – checkpoint / state-dict behaviour
# ---------------------------------------------------------------------------


def test_positional_embedding_state_dict_roundtrip(device):
    """Save and reload state dict, outputs must match exactly."""
    cfg = dict(num_channels=64, learnable=True, freq_embed_dim=64, mlp_hidden_dim=128)
    torch.manual_seed(42)
    model_a = PositionalEmbedding(**cfg).to(device)
    model_a.eval()

    torch.manual_seed(99)
    model_b = PositionalEmbedding(**cfg).to(device)
    model_b.eval()

    x = torch.tensor([0.0, 0.5, 1.0], device=device)

    # Before loading, outputs should differ (different init seeds).
    out_a = model_a(x)
    out_b = model_b(x)
    assert not torch.allclose(out_a, out_b), "Differently-seeded models should differ"

    # Load model_a weights into model_b.
    model_b.load_state_dict(model_a.state_dict())
    out_b = model_b(x)
    torch.testing.assert_close(out_a, out_b)


def test_positional_embedding_missing_freqs_compat(device):
    """Old checkpoints may not contain 'freqs'; the pre-hook must fill it."""
    model = PositionalEmbedding(num_channels=64).to(device)
    model.eval()

    x = torch.tensor([0.0, 50.0, 100.0], device=device)
    expected = model(x)

    # Simulate an old checkpoint that has no 'freqs' key.
    old_state = {k: v.clone() for k, v in model.state_dict().items() if k != "freqs"}
    assert "freqs" not in old_state

    # Loading the old state dict should NOT raise.
    fresh = PositionalEmbedding(num_channels=64).to(device)
    fresh.eval()
    fresh.load_state_dict(old_state, strict=False)

    # The freqs buffer should still be present (filled by the hook).
    assert hasattr(fresh, "freqs") and fresh.freqs is not None

    # Because the hook copies the module's own (freshly-computed) freqs, and
    # both modules use the same deterministic formula, outputs should match.
    actual = fresh(x)
    torch.testing.assert_close(actual, expected)


def test_positional_embedding_freqs_present_in_checkpoint(device):
    """When freqs IS in the checkpoint, the hook should be a no-op."""
    torch.manual_seed(42)
    model = PositionalEmbedding(num_channels=64).to(device)
    model.eval()

    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    expected = model(x)

    state = model.state_dict()
    assert "freqs" in state

    # Load into a fresh model (different seed → different default freqs).
    torch.manual_seed(99)
    fresh = PositionalEmbedding(num_channels=64).to(device)
    fresh.eval()
    fresh.load_state_dict(state)

    actual = fresh(x)
    torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# FourierEmbedding
# ---------------------------------------------------------------------------


class TestFourierEmbedding:
    def test_output_shape(self, device):
        model = FourierEmbedding(num_channels=64).to(device)
        x = torch.randn(8, device=device)
        out = model(x)
        assert out.shape == (8, 64)

    def test_output_shape_odd_batch(self, device):
        model = FourierEmbedding(num_channels=32).to(device)
        x = torch.randn(1, device=device)
        out = model(x)
        assert out.shape == (1, 32)

    def test_deterministic(self, device):
        """Same input → same output (given fixed freqs buffer)."""
        model = FourierEmbedding(num_channels=64).to(device)
        model.eval()
        x = torch.tensor([0.5, 1.0], device=device)
        torch.testing.assert_close(model(x), model(x))

    def test_dtype_cast(self, device):
        """When input is float64 and amp_mode=False, freqs are cast to match."""
        model = FourierEmbedding(num_channels=32, amp_mode=False).to(device)
        x = torch.tensor([1.0], device=device, dtype=torch.float64)
        out = model(x)
        assert out.dtype == torch.float64

    def test_state_dict_roundtrip(self, device):
        torch.manual_seed(42)
        model_a = FourierEmbedding(num_channels=64).to(device)

        torch.manual_seed(99)
        model_b = FourierEmbedding(num_channels=64).to(device)

        x = torch.randn(4, device=device)
        out_a = model_a(x)
        out_b = model_b(x)
        assert not torch.allclose(out_a, out_b)

        model_b.load_state_dict(model_a.state_dict())
        torch.testing.assert_close(model_a(x), model_b(x))

    def test_scale_affects_output(self, device):
        """Different scale values should produce different embeddings."""
        torch.manual_seed(0)
        model_s1 = FourierEmbedding(num_channels=32, scale=1).to(device)
        torch.manual_seed(0)
        model_s16 = FourierEmbedding(num_channels=32, scale=16).to(device)

        x = torch.tensor([1.0], device=device)
        out_s1 = model_s1(x)
        out_s16 = model_s16(x)
        assert not torch.allclose(out_s1, out_s16)


# ---------------------------------------------------------------------------
# SinusoidalTimestepEmbedding
# ---------------------------------------------------------------------------


class TestSinusoidalTimestepEmbedding:
    def test_output_shape(self, device):
        model = SinusoidalTimestepEmbedding(num_channels=64).to(device)
        x = torch.randn(8, device=device)
        out = model(x)
        assert out.shape == (8, 64)

    def test_output_shape_multidim(self, device):
        """Input with extra dims should be flattened to (B, D)."""
        model = SinusoidalTimestepEmbedding(num_channels=32).to(device)
        x = torch.randn(4, 1, device=device)
        out = model(x)
        assert out.shape == (4, 32)

    def test_zero_input(self, device):
        """At t=0, cos terms should be 1 and sin terms should be 0."""
        model = SinusoidalTimestepEmbedding(num_channels=8).to(device)
        out = model(torch.zeros(1, device=device))
        half = 4
        torch.testing.assert_close(out[0, :half], torch.ones(half, device=device))
        torch.testing.assert_close(
            out[0, half:], torch.zeros(half, device=device), atol=1e-7, rtol=1e-5
        )

    def test_deterministic(self, device):
        model = SinusoidalTimestepEmbedding(num_channels=64).to(device)
        x = torch.tensor([0.0, 0.5, 1.0], device=device)
        torch.testing.assert_close(model(x), model(x))

    def test_state_dict_roundtrip(self, device):
        model_a = SinusoidalTimestepEmbedding(num_channels=64).to(device)
        model_b = SinusoidalTimestepEmbedding(num_channels=64).to(device)

        # Freqs are deterministic (not random), so outputs should already match
        # for identically-configured models.
        x = torch.tensor([1.0, 2.0], device=device)
        torch.testing.assert_close(model_a(x), model_b(x))

        # Round-trip through state_dict still works.
        model_b.load_state_dict(model_a.state_dict())
        torch.testing.assert_close(model_a(x), model_b(x))

    def test_different_channels_different_output(self, device):
        m32 = SinusoidalTimestepEmbedding(num_channels=32).to(device)
        m64 = SinusoidalTimestepEmbedding(num_channels=64).to(device)
        x = torch.tensor([1.0], device=device)
        assert m32(x).shape[1] != m64(x).shape[1]


# ---------------------------------------------------------------------------
# OneHotEmbedding
# ---------------------------------------------------------------------------


class TestOneHotEmbedding:
    def test_output_shape(self, device):
        model = OneHotEmbedding(num_channels=64).to(device)
        t = torch.rand(8, 1, device=device)
        out = model(t)
        assert out.shape == (8, 64)

    def test_boundary_t0(self, device):
        """At t=0, only the first channel should be active (=1)."""
        model = OneHotEmbedding(num_channels=16).to(device)
        out = model(torch.zeros(1, 1, device=device))
        assert out[0, 0].item() == pytest.approx(1.0)
        assert out[0, 1:].abs().sum().item() == pytest.approx(0.0)

    def test_boundary_t1(self, device):
        """At t=1, only the last channel should be active (=1)."""
        model = OneHotEmbedding(num_channels=16).to(device)
        out = model(torch.ones(1, 1, device=device))
        assert out[0, -1].item() == pytest.approx(1.0)
        assert out[0, :-1].abs().sum().item() == pytest.approx(0.0)

    def test_midpoint(self, device):
        """At t=0.5, the peak should be near the middle channel."""
        D = 17
        model = OneHotEmbedding(num_channels=D).to(device)
        out = model(torch.tensor([[0.5]], device=device))
        peak = out[0].argmax().item()
        assert peak == D // 2

    def test_sums_to_one_interior(self, device):
        """For interior timesteps, the non-zero entries sum to 1.0 (triangular basis)."""
        model = OneHotEmbedding(num_channels=32).to(device)
        t = torch.tensor([[0.25]], device=device)
        out = model(t)
        assert out.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_non_negative(self, device):
        """All outputs should be non-negative due to the clamp."""
        model = OneHotEmbedding(num_channels=64).to(device)
        t = torch.rand(100, 1, device=device)
        out = model(t)
        assert (out >= 0).all()

    def test_state_dict_roundtrip(self, device):
        model_a = OneHotEmbedding(num_channels=32).to(device)
        model_b = OneHotEmbedding(num_channels=32).to(device)

        # Deterministic (no random init), so should already agree.
        t = torch.tensor([[0.3]], device=device)
        torch.testing.assert_close(model_a(t), model_b(t))

        model_b.load_state_dict(model_a.state_dict())
        torch.testing.assert_close(model_a(t), model_b(t))

    def test_deterministic(self, device):
        model = OneHotEmbedding(num_channels=64).to(device)
        t = torch.tensor([[0.0], [0.5], [1.0]], device=device)
        torch.testing.assert_close(model(t), model(t))
