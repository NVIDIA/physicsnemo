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

"""Test suite for the 4D xFNO operator (FNO4D) of the xDeepONet family.

Covers, per `MOD-008a/b/c <../../CODING_STANDARDS/MODELS_IMPLEMENTATION.md>`_,
mirroring the structure of ``test_xdeeponet.py``:

- **Constructor + public attributes** (MOD-008a) — default and custom configs.
- **Forward non-regression** (MOD-008b) — compare a single forward pass
  against committed golden ``.pth`` fixtures.
- **Checkpoint round-trip** (MOD-008c) — ``save`` to ``.mdlus``, reload via
  :meth:`physicsnemo.Module.from_checkpoint`, and verify the loaded model
  reproduces the committed golden output.
- **Gradient flow** — backward pass produces non-None gradients on input
  and parameters.
- **torch.compile smoke** — wrapping the model in :func:`torch.compile`
  (``fullgraph=False``) succeeds and matches eager numerically.
- **Time-axis extension** — wrapper ``target_times`` autoregressive
  forecast horizon ``K``.

The 3D FNO / Conv-FNO / U-FNO operators are intentionally *not* tested here:
they are not separate classes but configurations of
:class:`~physicsnemo.experimental.models.xdeeponet.DeepONet` (``trunk=None``
+ a :class:`~physicsnemo.experimental.models.xdeeponet.SpatialBranch`), and
are exercised by ``test_xdeeponet.py``.  ``FNO4D`` is the genuinely new 4D
operator the dimension-capped ``DeepONet`` core cannot express.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from physicsnemo import Module
from physicsnemo.experimental.models.xdeeponet import FNO4D, FNO4DWrapper

_DATA_DIR = Path(__file__).parent / "data"
_SEED = 0

# ----- Golden fixture paths ------------------------------------------------
#
# One ``.pth`` per scenario.  Filenames are versioned (``_v1``) so a new
# ``v2`` can land alongside an older fixture during a numerics transition.

_GOLDEN_FNO4D_CORE = _DATA_DIR / "xfno_fno4d_core_v1.pth"
_GOLDEN_FNO4D_WRAPPER = _DATA_DIR / "xfno_fno4d_wrapper_v1.pth"


# ----- Fixture builders ----------------------------------------------------
#
# Each builder returns ``(model, args)`` where ``args`` is the positional
# forward-argument tuple.  Inputs are kept tiny (1x4x4x4x4) so the golden
# files stay small and every test runs in well under a second.  ``FNO4D``
# uses ``ConvNdFCLayer`` (no ``LazyLinear``), so there are no lazy
# parameters to materialise.


def _fno4d_core() -> tuple[FNO4D, tuple[torch.Tensor, ...]]:
    """4D FNO core with coordinate features (the higher-dimension operator)."""
    torch.manual_seed(_SEED)
    model = FNO4D(
        in_channels=2,
        out_channels=1,
        width=8,
        modes1=2,
        modes2=2,
        modes3=2,
        modes4=2,
        num_fno_layers=2,
        lifting_layers=2,
        decoder_layers=1,
        decoder_width=16,
        coord_features=True,
    )
    x = torch.randn(1, 4, 4, 4, 4, 2)  # (B, X, Y, Z, T, C)
    return model, (x,)


def _fno4d_wrapper() -> tuple[FNO4DWrapper, tuple[torch.Tensor, ...]]:
    """4D FNO wrapper: per-dim auto-pad + crop."""
    torch.manual_seed(_SEED)
    model = FNO4DWrapper(
        modes1=2,
        modes2=2,
        modes3=2,
        modes4=2,
        width=8,
        in_channels=2,
        out_channels=1,
        num_fno_layers=2,
        lifting_layers=1,
        decoder_layers=1,
        decoder_width=16,
        coord_features=True,
        padding=0,
    )
    x = torch.randn(1, 4, 4, 4, 4, 2)  # (B, X, Y, Z, T_in, C)
    return model, (x,)


def _init_lazy(model, *args) -> None:
    """Run one forward pass (warmup; no lazy params, kept for symmetry)."""
    with torch.no_grad():
        model(*args)


def _load_golden(path: Path) -> dict[str, torch.Tensor | dict]:
    """Load a golden fixture; fail with a regen hint if missing.

    Fixtures under ``test/experimental/models/xdeeponet/data/`` are
    committed alongside this file and updated deliberately when model
    numerics intentionally change.  Regenerate with::

        python test/experimental/models/xdeeponet/data/\\
            _generate_xfno_goldens.py

    and commit the resulting ``.pth`` file.
    """
    if not path.exists():
        pytest.fail(
            f"Golden fixture {path.name} is missing. "
            f"Regenerate with "
            f"``python test/experimental/models/xdeeponet/data/"
            f"_generate_xfno_goldens.py`` and commit the "
            f"resulting ``.pth`` file."
        )
    return torch.load(path, weights_only=True)


def _golden_args(golden: dict) -> tuple[torch.Tensor, ...]:
    """Read positional forward arguments from a golden payload."""
    args = golden["args"]
    if isinstance(args, (list, tuple)):
        return tuple(args)
    return (args,)


# Registry of all (name, builder, golden-path) scenarios; consumed by the
# parameterised non-regression test below and by the golden generator
# script (``_generate_xfno_goldens.py``) so new scenarios are picked up in
# both places by adding one entry here.
_FIXTURE_REGISTRY = [
    ("fno4d_core", _fno4d_core, _GOLDEN_FNO4D_CORE),
    ("fno4d_wrapper", _fno4d_wrapper, _GOLDEN_FNO4D_WRAPPER),
]


# ----------------------------------------------------------------------
# Constructor + public attributes (MOD-008a)
# ----------------------------------------------------------------------


class TestFNO4DConstructor:
    """``FNO4D`` instantiates and exposes the documented public attributes."""

    @pytest.mark.parametrize(
        "coord_features",
        [True, False],
        ids=["coords", "no-coords"],
    )
    def test_fno4d_attrs(self, coord_features):
        """``FNO4D`` stores the constructor arguments on public attrs."""
        model = FNO4D(
            in_channels=2,
            out_channels=3,
            width=8,
            modes1=2,
            modes2=2,
            modes3=2,
            modes4=2,
            num_fno_layers=2,
            lifting_layers=2,
            decoder_layers=1,
            decoder_width=16,
            coord_features=coord_features,
        )
        assert model.in_channels == 2
        assert model.out_channels == 3
        assert model.width == 8
        assert model.modes1 == 2 and model.modes4 == 2
        assert model.num_fno_layers == 2
        assert model.coord_features is coord_features
        assert model.activation_fn_name == "gelu"

    def test_nonpositive_layers_rejected(self):
        """``num_fno_layers <= 0`` is rejected at construction."""
        with pytest.raises(ValueError, match="num_fno_layers must be positive"):
            FNO4D(in_channels=2, out_channels=1, num_fno_layers=0)


# ----------------------------------------------------------------------
# Forward non-regression against committed golden files (MOD-008b)
# ----------------------------------------------------------------------


class TestXFNONonRegression:
    """Forward output matches the committed golden fixture.

    Parameterised on the full :data:`_FIXTURE_REGISTRY` so adding a new
    scenario is a one-line addition (and a regenerated ``.pth``).
    """

    @pytest.mark.parametrize(
        "name, builder, golden_path",
        _FIXTURE_REGISTRY,
        ids=[entry[0] for entry in _FIXTURE_REGISTRY],
    )
    def test_matches_golden(self, name, builder, golden_path):
        """Forward output reproduces the stored golden output."""
        del name  # used only for the test ID
        golden = _load_golden(golden_path)
        args = _golden_args(golden)
        model, _ = builder()
        _init_lazy(model, *args)
        model.load_state_dict(golden["state_dict"])
        with torch.no_grad():
            y = model(*args)
        torch.testing.assert_close(y, golden["y"], rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# Checkpoint (.mdlus) round-trip (MOD-008c)
# ----------------------------------------------------------------------


class TestXFNOCheckpoint:
    """``Module.save`` + ``Module.from_checkpoint`` round-trip.

    Verifies that :meth:`physicsnemo.Module.from_checkpoint` reconstructs a
    model whose forward output matches the committed golden fixture — not a
    second forward pass on the in-memory model — so the test fails if the
    serialized state is incomplete, corrupted, or silently re-initialised.
    """

    def _roundtrip(self, model, args, tmp_path):
        _init_lazy(model, *args)
        ckpt = tmp_path / "model.mdlus"
        model.save(str(ckpt))
        loaded = Module.from_checkpoint(str(ckpt))
        with torch.no_grad():
            y_loaded = loaded(*args)
        return loaded, y_loaded

    def test_fno4d_roundtrip(self, tmp_path):
        """FNO4D: reloaded output matches the committed golden."""
        golden = _load_golden(_GOLDEN_FNO4D_CORE)
        args = _golden_args(golden)
        model, _ = _fno4d_core()
        loaded, y_loaded = self._roundtrip(model, args, tmp_path)
        assert type(loaded).__name__ == type(model).__name__
        assert loaded.width == model.width
        assert loaded.coord_features == model.coord_features
        torch.testing.assert_close(y_loaded, golden["y"], rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# Gradient flow
# ----------------------------------------------------------------------


class TestXFNOGradientFlow:
    """Backward pass produces non-None gradients on input and parameters."""

    def test_fno4d_gradients(self):
        """Gradients flow through FNO4D."""
        model, (x,) = _fno4d_core()
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


class TestXFNOCompile:
    """``torch.compile(fullgraph=False)`` wraps the model and matches eager.

    ``fullgraph=True`` is not asserted: the spectral convolutions use
    ``torch.fft`` (complex tensors), which introduces graph breaks under
    ``torch.compile`` on the torch versions exercised in CI.  The default
    production path (``fullgraph=False``) tolerates those breaks and is what
    we verify here.
    """

    def test_fno4d_compile(self):
        """FNO4D compiled model produces eager-equivalent output."""
        model, (x,) = _fno4d_core()
        _init_lazy(model, x)
        with torch.no_grad():
            y_eager = model(x)
        compiled = torch.compile(model, fullgraph=False)
        with torch.no_grad():
            y_compiled = compiled(x)
        assert y_compiled.shape == y_eager.shape
        torch.testing.assert_close(y_compiled, y_eager, rtol=1e-4, atol=1e-5)


# ----------------------------------------------------------------------
# Time-axis extension (autoregressive bundling)
# ----------------------------------------------------------------------


class TestXFNOTimeExtend:
    """Wrapper ``target_times`` autoregressive forecast-horizon extension.

    When ``target_times`` of length ``K != T_in`` is supplied, the time
    axis is right-replicate-padded so the inner operator sees at least
    ``T_in + K`` (and ``2 * modes_t``) timesteps; the output is cropped to
    the last ``K`` timesteps.
    """

    def test_fno4d_wrapper_extends_to_K(self):
        """FNO4DWrapper: output time-axis equals the requested horizon K."""
        model, (x,) = _fno4d_wrapper()
        _init_lazy(model, x)
        target_times = torch.linspace(0.5, 1.0, 6)  # K=6 != T_in=4
        with torch.no_grad():
            y = model(x, target_times=target_times)
        # x: (1, 4, 4, 4, 4, 2); squeezed output -> (1, 4, 4, 4, K=6)
        assert y.shape == (1, 4, 4, 4, 6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
