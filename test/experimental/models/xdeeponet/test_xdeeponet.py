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

"""Minimal, reviewer-preferred test suite for the xDeepONet family.

Covers, per `MOD-008a/b/c <../../CODING_STANDARDS/MODELS_IMPLEMENTATION.md>`_
and the PR #1576 human review:

- **Constructor + public attributes** (MOD-008a) — default and custom configs.
- **Forward non-regression** (MOD-008b) — compare a single forward pass
  against committed golden ``.pth`` fixtures.
- **Checkpoint round-trip** (MOD-008c) — ``save`` to ``.mdlus``, reload via
  :meth:`physicsnemo.Module.from_checkpoint`, and verify the loaded model
  reproduces the same output as the in-memory model.
- **Gradient flow** — backward pass produces non-None gradients on input
  and parameters.
- **torch.compile smoke** — wrapping the model in :func:`torch.compile`
  succeeds and produces shape-compatible output.

Broader shape / variant / error-path coverage (all 8 variants, both
decoder types, construction-time guards, Fourier code path, adaptive
pooling, etc.) lives on the ``pr/neural-operator-factory`` branch as
``examples/reservoir_simulation/neural_operator_factory/tests/test_xdeeponet_upstream.py``
for local regression when preparing future NOF upstream PRs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from physicsnemo import Module
from physicsnemo.experimental.models.xdeeponet import (
    DeepONet,
    DeepONet3D,
    DeepONet3DWrapper,
    DeepONetWrapper,
)

_DATA_DIR = Path(__file__).parent / "data"
_GOLDEN_2D = _DATA_DIR / "xdeeponet_wrapper_2d_v1.pth"
_GOLDEN_3D = _DATA_DIR / "xdeeponet_wrapper_3d_v1.pth"
_SEED = 0

# Minimal branch/trunk configs chosen for (a) small tensor shapes so the
# golden files stay tiny and the tests run in well under a second each,
# and (b) exercising the spatial-branch + UNet + MLP-decoder path, which
# is the most common user-facing configuration.
_BRANCH_SPATIAL = {
    "encoder": {"type": "linear", "activation_fn": "relu"},
    "layers": {
        "num_fourier_layers": 0,
        "num_unet_layers": 1,
        "num_conv_layers": 0,
        "kernel_size": 3,
        "dropout": 0.0,
        "activation_fn": "relu",
    },
}
_TRUNK = {
    "input_type": "time",
    "hidden_width": 16,
    "num_layers": 2,
    "activation_fn": "tanh",
}


def _wrapper_2d() -> tuple[DeepONetWrapper, torch.Tensor]:
    """Build a deterministic 2D wrapper + matching input.

    Uses a fixed RNG seed so the materialised :class:`torch.nn.LazyLinear`
    weights (created on the first forward pass) are reproducible across
    runs.
    """
    torch.manual_seed(_SEED)
    model = DeepONetWrapper(
        padding=8,
        variant="u_deeponet",
        width=8,
        branch1_config=_BRANCH_SPATIAL,
        trunk_config=_TRUNK,
        decoder_type="mlp",
        decoder_width=8,
        decoder_layers=1,
    )
    x = torch.randn(1, 8, 8, 2, 2)
    return model, x


def _wrapper_3d() -> tuple[DeepONet3DWrapper, torch.Tensor]:
    """Build a deterministic 3D wrapper + matching input."""
    torch.manual_seed(_SEED)
    model = DeepONet3DWrapper(
        padding=8,
        variant="u_deeponet",
        width=8,
        branch1_config=_BRANCH_SPATIAL,
        trunk_config=_TRUNK,
        decoder_type="mlp",
        decoder_width=8,
        decoder_layers=1,
    )
    x = torch.randn(1, 8, 8, 8, 2, 2)
    return model, x


def _init_lazy(model, x) -> None:
    """Run one forward pass to materialise ``nn.LazyLinear`` parameters."""
    with torch.no_grad():
        model(x)


def _load_golden(path: Path) -> dict[str, torch.Tensor | dict]:
    """Load a golden fixture; fail loudly with a regeneration hint if missing.

    Fixtures under ``test/experimental/models/data/`` are committed
    alongside this file and updated deliberately when model numerics
    change (via the generator script
    ``test/experimental/models/data/_generate_xdeeponet_goldens.py``).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Golden fixture {path} is missing. Regenerate with "
            f"``python test/experimental/models/xdeeponet/data/"
            f"_generate_xdeeponet_goldens.py`` and commit the "
            f"resulting ``.pth`` files."
        )
    # Golden payload is {str -> Tensor | dict[str, Tensor]} so
    # ``weights_only=True`` is the safer default and avoids PyTorch 2.6's
    # FutureWarning on the legacy permissive path.
    return torch.load(path, weights_only=True)


# ----------------------------------------------------------------------
# Constructor + public attributes (MOD-008a)
# ----------------------------------------------------------------------


class TestDeepONetConstructor:
    """Constructor instantiates and exposes the documented public attributes."""

    @pytest.mark.parametrize(
        "config",
        [
            {"variant": "u_deeponet", "width": 8, "decoder_type": "mlp"},
            {"variant": "deeponet", "width": 16, "decoder_type": "conv"},
        ],
        ids=["default-ish", "custom"],
    )
    def test_deeponet_2d_core(self, config):
        """``DeepONet`` stores the constructor arguments on public attrs."""
        model = DeepONet(
            variant=config["variant"],
            width=config["width"],
            branch1_config=_BRANCH_SPATIAL,
            trunk_config=_TRUNK,
            decoder_type=config["decoder_type"],
            decoder_width=config["width"],
            decoder_layers=1,
        )
        assert model.variant == config["variant"]
        assert model.width == config["width"]
        assert model.decoder_type == config["decoder_type"]
        assert model.decoder_activation_fn == "relu"
        assert model.trunk is not None

    @pytest.mark.parametrize(
        "config",
        [
            {"variant": "u_deeponet", "width": 8, "decoder_type": "mlp"},
            {"variant": "deeponet", "width": 16, "decoder_type": "conv"},
        ],
        ids=["default-ish", "custom"],
    )
    def test_deeponet_3d_core(self, config):
        """``DeepONet3D`` stores the constructor arguments on public attrs."""
        model = DeepONet3D(
            variant=config["variant"],
            width=config["width"],
            branch1_config=_BRANCH_SPATIAL,
            trunk_config=_TRUNK,
            decoder_type=config["decoder_type"],
            decoder_width=config["width"],
            decoder_layers=1,
        )
        assert model.variant == config["variant"]
        assert model.width == config["width"]
        assert model.decoder_type == config["decoder_type"]
        assert model.decoder_activation_fn == "relu"
        assert model.trunk is not None

    @pytest.mark.parametrize(
        "config",
        [
            {"padding": 8, "variant": "u_deeponet", "trunk_input": "time"},
            {"padding": 16, "variant": "deeponet", "trunk_input": "grid"},
        ],
        ids=["default-ish", "custom"],
    )
    def test_wrapper_2d(self, config):
        """``DeepONetWrapper`` exposes padding / variant / trunk_input."""
        model = DeepONetWrapper(
            padding=config["padding"],
            variant=config["variant"],
            width=8,
            branch1_config=_BRANCH_SPATIAL,
            trunk_config={**_TRUNK, "input_type": config["trunk_input"]},
            decoder_type="mlp",
            decoder_width=8,
            decoder_layers=1,
        )
        assert model.padding == config["padding"]
        assert model.variant == config["variant"]
        assert model.trunk_input == config["trunk_input"]
        assert isinstance(model.model, DeepONet)

    @pytest.mark.parametrize(
        "config",
        [
            {"padding": 8, "variant": "u_deeponet", "trunk_input": "time"},
            {"padding": 16, "variant": "deeponet", "trunk_input": "grid"},
        ],
        ids=["default-ish", "custom"],
    )
    def test_wrapper_3d(self, config):
        """``DeepONet3DWrapper`` exposes padding / variant / trunk_input."""
        model = DeepONet3DWrapper(
            padding=config["padding"],
            variant=config["variant"],
            width=8,
            branch1_config=_BRANCH_SPATIAL,
            trunk_config={**_TRUNK, "input_type": config["trunk_input"]},
            decoder_type="mlp",
            decoder_width=8,
            decoder_layers=1,
        )
        assert model.padding == config["padding"]
        assert model.variant == config["variant"]
        assert model.trunk_input == config["trunk_input"]
        assert isinstance(model.model, DeepONet3D)


# ----------------------------------------------------------------------
# Forward non-regression against committed golden files (MOD-008b)
# ----------------------------------------------------------------------


class TestDeepONetNonRegression:
    """Forward output matches the committed golden fixture."""

    def test_wrapper_2d_matches_golden(self):
        """2D wrapper: loading fixed state_dict reproduces the stored output."""
        golden = _load_golden(_GOLDEN_2D)
        model, _ = _wrapper_2d()
        _init_lazy(model, golden["x"])
        model.load_state_dict(golden["state_dict"])
        with torch.no_grad():
            y = model(golden["x"])
        torch.testing.assert_close(y, golden["y"], rtol=1e-5, atol=1e-6)

    def test_wrapper_3d_matches_golden(self):
        """3D wrapper: loading fixed state_dict reproduces the stored output."""
        golden = _load_golden(_GOLDEN_3D)
        model, _ = _wrapper_3d()
        _init_lazy(model, golden["x"])
        model.load_state_dict(golden["state_dict"])
        with torch.no_grad():
            y = model(golden["x"])
        torch.testing.assert_close(y, golden["y"], rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# Checkpoint (.mdlus) round-trip (MOD-008c)
# ----------------------------------------------------------------------


class TestDeepONetCheckpoint:
    """``Module.save`` + ``Module.from_checkpoint`` round-trip.

    Verifies that :meth:`physicsnemo.Module.from_checkpoint` reconstructs a
    byte-identical model.  The loaded model's forward output is compared
    **against the committed golden fixture** — not against a second forward
    pass on the in-memory model — so the test fails if the serialized
    state is incomplete, corrupted, or silently re-initialised.

    PyTorch's :meth:`torch.nn.Module.load_state_dict` natively materialises
    :class:`torch.nn.LazyLinear` parameters from the saved tensors, so no
    ``_init_lazy`` call is needed on the reloaded model.
    """

    def _roundtrip(self, model, x, tmp_path):
        _init_lazy(model, x)
        ckpt = tmp_path / "model.mdlus"
        model.save(str(ckpt))
        loaded = Module.from_checkpoint(str(ckpt))
        with torch.no_grad():
            y_loaded = loaded(x)
        return loaded, y_loaded

    def test_wrapper_2d_roundtrip(self, tmp_path):
        """2D wrapper: reloaded output matches the committed golden."""
        golden = _load_golden(_GOLDEN_2D)
        model, _ = _wrapper_2d()
        loaded, y_loaded = self._roundtrip(model, golden["x"], tmp_path)
        assert type(loaded).__name__ == type(model).__name__
        assert loaded.padding == model.padding
        assert loaded.variant == model.variant
        assert loaded.trunk_input == model.trunk_input
        torch.testing.assert_close(y_loaded, golden["y"], rtol=1e-5, atol=1e-6)

    def test_wrapper_3d_roundtrip(self, tmp_path):
        """3D wrapper: reloaded output matches the committed golden."""
        golden = _load_golden(_GOLDEN_3D)
        model, _ = _wrapper_3d()
        loaded, y_loaded = self._roundtrip(model, golden["x"], tmp_path)
        assert type(loaded).__name__ == type(model).__name__
        assert loaded.padding == model.padding
        assert loaded.variant == model.variant
        assert loaded.trunk_input == model.trunk_input
        torch.testing.assert_close(y_loaded, golden["y"], rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# Gradient flow
# ----------------------------------------------------------------------


class TestDeepONetGradientFlow:
    """Backward pass produces non-None gradients on input and parameters."""

    def test_wrapper_2d_gradients(self):
        """Gradients flow through the 2D wrapper."""
        model, x = _wrapper_2d()
        _init_lazy(model, x)
        x = x.detach().requires_grad_(True)
        y = model(x)
        y.sum().backward()
        assert x.grad is not None
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert trainable, "model has no trainable parameters"
        assert any(p.grad is not None for p in trainable)


# ----------------------------------------------------------------------
# torch.compile smoke test
# ----------------------------------------------------------------------


class TestDeepONetCompile:
    """``torch.compile`` wraps the model without raising."""

    def test_wrapper_2d_compile(self):
        """Compiled model produces shape-compatible output vs eager."""
        model, x = _wrapper_2d()
        _init_lazy(model, x)
        with torch.no_grad():
            y_eager = model(x)
        compiled = torch.compile(model, fullgraph=False)
        with torch.no_grad():
            y_compiled = compiled(x)
        assert y_compiled.shape == y_eager.shape
        torch.testing.assert_close(y_compiled, y_eager, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
