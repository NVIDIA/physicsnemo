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

"""Test suite for :class:`FNO4DWrapper` (4D xFNO) of the xDeepONet family.

Covers, per `MOD-008a/b/c <../../CODING_STANDARDS/MODELS_IMPLEMENTATION.md>`_,
mirroring the structure of ``test_xdeeponet.py``:

- **Constructor + public attributes** (MOD-008a) — verify the wrapper wires a
  4D :class:`physicsnemo.models.fno.FNO` and exposes its config.
- **Forward non-regression** (MOD-008b) — compare a single forward pass
  against a committed golden ``.pth`` fixture.
- **Checkpoint round-trip** (MOD-008c) — ``save`` to ``.mdlus``, reload via
  :meth:`physicsnemo.Module.from_checkpoint`, and verify the loaded model
  reproduces the committed golden output.
- **Gradient flow** — backward pass produces non-None gradients on input
  and parameters.
- **torch.compile smoke** — ``torch.compile(fullgraph=False)`` succeeds and
  matches eager numerically.
- **Time-axis extension** — wrapper ``target_times`` autoregressive forecast
  horizon ``K``.

The 4D FNO core itself is the library model
(:class:`physicsnemo.models.fno.FNO` with ``dimension=4``) and is tested by
its own suite; this file only covers the wrapper's added behavior.  The 3D
FNO / Conv-FNO / U-FNO operators are configurations of
:class:`~physicsnemo.experimental.models.xdeeponet.DeepONet` and are exercised
by ``test_xdeeponet.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from physicsnemo import Module
from physicsnemo.experimental.models.xdeeponet import FNO4DWrapper
from physicsnemo.models.fno import FNO

_DATA_DIR = Path(__file__).parent / "data"
_SEED = 0

# ----- Golden fixture paths ------------------------------------------------

_GOLDEN_FNO4D_WRAPPER = _DATA_DIR / "xfno_fno4d_wrapper_v1.pth"


# ----- Fixture builders ----------------------------------------------------
#
# The builder returns ``(model, args)``.  Inputs are kept tiny (1x4x4x4x4) so
# the golden file stays small and the test runs in well under a second.


def _fno4d_wrapper() -> tuple[FNO4DWrapper, tuple[torch.Tensor, ...]]:
    """4D FNO wrapper around the library ``FNO(dimension=4)``."""
    torch.manual_seed(_SEED)
    model = FNO4DWrapper(
        in_channels=2,
        out_channels=1,
        latent_channels=8,
        num_fno_layers=2,
        num_fno_modes=2,
        padding=0,
        decoder_layers=1,
        decoder_layer_size=16,
        coord_features=True,
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
# script (``_generate_xfno_goldens.py``).
_FIXTURE_REGISTRY = [
    ("fno4d_wrapper", _fno4d_wrapper, _GOLDEN_FNO4D_WRAPPER),
]


# ----------------------------------------------------------------------
# Constructor + public attributes (MOD-008a)
# ----------------------------------------------------------------------


class TestFNO4DWrapperConstructor:
    """``FNO4DWrapper`` wires a 4D ``FNO`` and exposes its config."""

    @pytest.mark.parametrize(
        "coord_features",
        [True, False],
        ids=["coords", "no-coords"],
    )
    def test_wrapper_attrs(self, coord_features):
        """The wrapper builds an inner ``FNO(dimension=4)`` and stores attrs."""
        model = FNO4DWrapper(
            in_channels=2,
            out_channels=3,
            latent_channels=8,
            num_fno_layers=2,
            num_fno_modes=[2, 2, 2, 3],
            padding=0,
            decoder_layers=1,
            decoder_layer_size=16,
            coord_features=coord_features,
        )
        assert isinstance(model.fno, FNO)
        assert model.fno.dimension == 4
        assert model.fno.in_channels == 2
        assert model.fno.coord_features is coord_features
        # The time-axis mode count is the last entry of num_fno_modes.
        assert model.time_modes == 3

    def test_time_modes_from_scalar(self):
        """A scalar ``num_fno_modes`` is used as the time-axis mode count."""
        model = FNO4DWrapper(in_channels=2, num_fno_modes=4, num_fno_layers=2)
        assert model.time_modes == 4


# ----------------------------------------------------------------------
# Forward non-regression against committed golden files (MOD-008b)
# ----------------------------------------------------------------------


class TestXFNONonRegression:
    """Forward output matches the committed golden fixture."""

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

    The reloaded model's forward output is compared against the committed
    golden fixture so the test fails if the serialized state is incomplete,
    corrupted, or silently re-initialised.
    """

    def test_wrapper_roundtrip(self, tmp_path):
        """FNO4DWrapper: reloaded output matches the committed golden."""
        golden = _load_golden(_GOLDEN_FNO4D_WRAPPER)
        args = _golden_args(golden)
        model, _ = _fno4d_wrapper()
        _init_lazy(model, *args)
        ckpt = tmp_path / "model.mdlus"
        model.save(str(ckpt))
        loaded = Module.from_checkpoint(str(ckpt))
        assert type(loaded).__name__ == type(model).__name__
        assert loaded.time_modes == model.time_modes
        with torch.no_grad():
            y_loaded = loaded(*args)
        torch.testing.assert_close(y_loaded, golden["y"], rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# Gradient flow
# ----------------------------------------------------------------------


class TestXFNOGradientFlow:
    """Backward pass produces non-None gradients on input and parameters."""

    def test_wrapper_gradients(self):
        """Gradients flow through FNO4DWrapper."""
        model, (x,) = _fno4d_wrapper()
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
    ``torch.compile``.  The default production path (``fullgraph=False``)
    tolerates those breaks and is what we verify here.
    """

    def test_wrapper_compile(self):
        """FNO4DWrapper compiled model produces eager-equivalent output."""
        model, (x,) = _fno4d_wrapper()
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
    axis is right-replicate-padded so the operator runs on at least
    ``T_in + K`` (and ``2 * time_modes``) timesteps; the output is cropped to
    the last ``K`` timesteps.
    """

    def test_wrapper_extends_to_K(self):
        """FNO4DWrapper: output time-axis equals the requested horizon K."""
        model, (x,) = _fno4d_wrapper()
        _init_lazy(model, x)
        target_times = torch.linspace(0.5, 1.0, 6)  # K=6 != T_in=4
        with torch.no_grad():
            y = model(x, target_times=target_times)
        # x: (1, 4, 4, 4, 4, 2); squeezed output -> (1, 4, 4, 4, K=6)
        assert y.shape == (1, 4, 4, 4, 6)

    def test_wrapper_K_equals_T_in(self):
        """FNO4DWrapper: K == T_in short-circuits the extension."""
        model, (x,) = _fno4d_wrapper()
        _init_lazy(model, x)
        target_times = torch.linspace(0.0, 1.0, 4)  # K == T_in == 4
        with torch.no_grad():
            y = model(x, target_times=target_times)
        assert y.shape == (1, 4, 4, 4, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
