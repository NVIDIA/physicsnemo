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

"""Test suite for the xDeepONet family.

Covers, per `MOD-008a/b/c <../../CODING_STANDARDS/MODELS_IMPLEMENTATION.md>`_:

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
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from physicsnemo import Module
from physicsnemo.experimental.models.xdeeponet import DeepONet, SpatialBranch
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.nn import get_activation

_DATA_DIR = Path(__file__).parent / "data"
_SEED = 0

# ----- Golden fixture paths ------------------------------------------------
#
# One ``.pth`` per scenario.  The fixture filenames are versioned (``_v1``)
# so a new ``v2`` can land alongside an older fixture during a numerics
# transition.

# Packed-input (auto_pad=True) scenarios.
_GOLDEN_PACKED_2D = _DATA_DIR / "xdeeponet_packed_2d_v1.pth"
_GOLDEN_PACKED_3D = _DATA_DIR / "xdeeponet_packed_3d_v1.pth"
_GOLDEN_PACKED_2D_FOURIER = _DATA_DIR / "xdeeponet_packed_2d_fourier_v1.pth"
_GOLDEN_PACKED_2D_MIONET = _DATA_DIR / "xdeeponet_packed_2d_mionet_v1.pth"
_GOLDEN_PACKED_2D_TEMPORAL = _DATA_DIR / "xdeeponet_packed_2d_temporal_v1.pth"
_GOLDEN_PACKED_2D_MULTICHANNEL = _DATA_DIR / "xdeeponet_packed_2d_multichannel_v1.pth"
# Trunkless packed-input (xFNO-style) scenarios.
_GOLDEN_XFNO_PACKED_3D = _DATA_DIR / "xdeeponet_xfno_packed_3d_v1.pth"
_GOLDEN_XFNO_PACKED_3D_EXTEND = _DATA_DIR / "xdeeponet_xfno_packed_3d_extend_v1.pth"
# Core-mode (auto_pad=False) fixture for the MLP-branch path.
_GOLDEN_CORE_2D_MLPBRANCH = _DATA_DIR / "xdeeponet_core_2d_mlpbranch_v1.pth"


# ----- Module builders -----------------------------------------------------
#
# DeepONet expects branch / trunk modules to be constructed and passed in
# directly.  These helpers produce minimal modules so the golden files
# stay tiny (test inputs are 1x8x8 or 1x8x8x8) and every test runs in
# well under a second.


def _make_unet_spatial_branch(dimension: int, width: int) -> SpatialBranch:
    """Spatial branch with a single UNet layer (U-DeepONet style)."""
    return SpatialBranch(
        dimension=dimension,
        in_channels=2,
        width=width,
        num_unet_layers=1,
        kernel_size=3,
        activation_fn="relu",
    )


def _make_fourier_spatial_branch(dimension: int, width: int) -> SpatialBranch:
    """Spatial branch with a single Fourier layer (Fourier-DeepONet style)."""
    return SpatialBranch(
        dimension=dimension,
        in_channels=2,
        width=width,
        num_fourier_layers=1,
        modes1=2,
        modes2=2,
        activation_fn="relu",
    )


def _make_mlp_branch(
    *,
    in_features: int,
    hidden_width: int,
    out_features: int,
    num_layers: int,
    activation_fn: str = "relu",
) -> nn.Module:
    """Flat MLP branch: ``num_layers`` activated linears in total.

    Composed as :class:`FullyConnected` (with ``num_layers - 1`` activated
    hidden layers + one unactivated projection) wrapped with a trailing
    activation so every linear is followed by an activation.
    """
    return nn.Sequential(
        FullyConnected(
            in_features=in_features,
            layer_size=hidden_width,
            out_features=out_features,
            num_layers=num_layers - 1,
            activation_fn=activation_fn,
        ),
        get_activation(activation_fn),
    )


def _make_trunk(
    *,
    in_features: int = 1,
    out_features: int,
    hidden_width: int = 16,
    num_layers: int = 2,
    activation_fn: str = "tanh",
    output_activation: bool = True,
) -> nn.Module:
    """Trunk MLP.

    A :class:`FullyConnected` produces ``num_layers`` activated hidden
    linears followed by a single unactivated projection
    (``hidden_width -> out_features``); when ``output_activation`` is
    true the projection is wrapped with a trailing activation.
    """
    trunk = FullyConnected(
        in_features=in_features,
        layer_size=hidden_width,
        out_features=out_features,
        num_layers=num_layers,
        activation_fn=activation_fn,
    )
    if output_activation:
        return nn.Sequential(trunk, get_activation(activation_fn))
    return trunk


# ----- Fixture builders ----------------------------------------------------


def _wrapper_2d() -> tuple[DeepONet, tuple[torch.Tensor, ...]]:
    """Packed-input 2D U-DeepONet builder."""
    torch.manual_seed(_SEED)
    model = DeepONet(
        branch1=_make_unet_spatial_branch(dimension=2, width=8),
        trunk=_make_trunk(out_features=8),
        dimension=2,
        width=8,
        decoder_type="mlp",
        decoder_width=8,
        decoder_layers=1,
        auto_pad=True,
        padding=8,
        trunk_input="time",
        variant="u_deeponet",
    )
    x = torch.randn(1, 8, 8, 2, 2)
    return model, (x,)


def _wrapper_3d() -> tuple[DeepONet, tuple[torch.Tensor, ...]]:
    """Packed-input 3D U-DeepONet builder."""
    torch.manual_seed(_SEED)
    model = DeepONet(
        branch1=_make_unet_spatial_branch(dimension=3, width=8),
        trunk=_make_trunk(out_features=8),
        dimension=3,
        width=8,
        decoder_type="mlp",
        decoder_width=8,
        decoder_layers=1,
        auto_pad=True,
        padding=8,
        trunk_input="time",
        variant="u_deeponet",
    )
    x = torch.randn(1, 8, 8, 8, 2, 2)
    return model, (x,)


def _wrapper_2d_fourier() -> tuple[DeepONet, tuple[torch.Tensor, ...]]:
    """Packed-input 2D Fourier-DeepONet builder (exercises SpectralConv2d)."""
    torch.manual_seed(_SEED)
    model = DeepONet(
        branch1=_make_fourier_spatial_branch(dimension=2, width=8),
        trunk=_make_trunk(out_features=8),
        dimension=2,
        width=8,
        decoder_type="mlp",
        decoder_width=8,
        decoder_layers=1,
        auto_pad=True,
        padding=8,
        trunk_input="time",
        variant="fourier_deeponet",
    )
    x = torch.randn(1, 8, 8, 2, 2)
    return model, (x,)


def _wrapper_2d_mionet() -> tuple[DeepONet, tuple[torch.Tensor, ...]]:
    """Packed-input 2D MIONet builder (exercises the dual-branch path)."""
    torch.manual_seed(_SEED)
    model = DeepONet(
        branch1=_make_unet_spatial_branch(dimension=2, width=8),
        branch2=_make_unet_spatial_branch(dimension=2, width=8),
        trunk=_make_trunk(out_features=8),
        dimension=2,
        width=8,
        decoder_type="mlp",
        decoder_width=8,
        decoder_layers=1,
        auto_pad=True,
        padding=8,
        trunk_input="time",
        variant="mionet",
    )
    x = torch.randn(1, 8, 8, 2, 2)
    x_branch2 = torch.randn(1, 8, 8, 2, 2)
    return model, (x, x_branch2)


def _wrapper_2d_temporal() -> tuple[DeepONet, tuple[torch.Tensor, ...]]:
    """Packed-input 2D builder exercising the ``temporal_projection`` decoder."""
    torch.manual_seed(_SEED)
    model = DeepONet(
        branch1=_make_unet_spatial_branch(dimension=2, width=8),
        trunk=_make_trunk(out_features=8),
        dimension=2,
        width=8,
        decoder_type="temporal_projection",
        decoder_width=8,
        decoder_layers=1,
        output_window=3,
        auto_pad=True,
        padding=8,
        trunk_input="time",
        variant="u_deeponet",
    )
    x = torch.randn(1, 8, 8, 2, 2)
    return model, (x,)


def _xfno_packed_3d() -> tuple[DeepONet, tuple[torch.Tensor, ...]]:
    """Packed-input trunkless 3D operator (xFNO / U-FNO style).

    No trunk MLP; the branch produces a spatial latent that the decoder
    projects to ``out_channels`` directly.  Auto-padding is on but
    ``time_modes`` is not set, so no time-axis-extend occurs.  The input
    is channels-last ``(B, *spatial, C)`` and the output is
    ``(B, *spatial, out_channels)``.
    """
    torch.manual_seed(_SEED)
    branch1 = SpatialBranch(
        dimension=3,
        in_channels=2,
        width=8,
        num_fourier_layers=2,
        num_unet_layers=1,
        modes1=2,
        modes2=2,
        modes3=2,
        kernel_size=3,
        activation_fn="relu",
        coord_features=True,
    )
    model = DeepONet(
        branch1=branch1,
        trunk=None,
        dimension=3,
        width=8,
        out_channels=1,
        decoder_type="mlp",
        decoder_width=8,
        decoder_layers=1,
        auto_pad=True,
        padding=8,
        variant="ufno",
    )
    x = torch.randn(1, 8, 8, 4, 2)  # (B, H, W, T_in, C)
    return model, (x,)


def _xfno_packed_3d_extend() -> tuple[DeepONet, tuple[torch.Tensor, ...]]:
    """Packed-input trunkless 3D operator with ``time_modes`` set.

    The returned ``args`` tuple contains only the input tensor.  Tests
    that need to drive the time-axis-extend feature pass
    ``target_times`` as a keyword argument when calling the model
    (not part of the standard fixture-registry contract).
    """
    torch.manual_seed(_SEED)
    branch1 = SpatialBranch(
        dimension=3,
        in_channels=2,
        width=8,
        num_fourier_layers=2,
        num_unet_layers=0,
        modes1=2,
        modes2=2,
        modes3=2,
        kernel_size=3,
        activation_fn="relu",
        coord_features=True,
    )
    model = DeepONet(
        branch1=branch1,
        trunk=None,
        dimension=3,
        width=8,
        out_channels=1,
        decoder_type="mlp",
        decoder_width=8,
        decoder_layers=1,
        auto_pad=True,
        padding=8,
        time_modes=2,
        variant="ufno",
    )
    x = torch.randn(1, 8, 8, 4, 2)  # (B, H, W, T_in=4, C)
    return model, (x,)


def _packed_2d_multichannel() -> tuple[DeepONet, tuple[torch.Tensor, ...]]:
    """Trunked packed-input 2D builder with ``out_channels=3``.

    Exercises the multi-channel-output path: the decoder's final layer
    maps width to ``out_channels=3`` and the output tensor's trailing
    dim is 3 (not squeezed).
    """
    torch.manual_seed(_SEED)
    model = DeepONet(
        branch1=_make_unet_spatial_branch(dimension=2, width=8),
        trunk=_make_trunk(out_features=8),
        dimension=2,
        width=8,
        out_channels=3,
        decoder_type="mlp",
        decoder_width=8,
        decoder_layers=1,
        auto_pad=True,
        padding=8,
        trunk_input="time",
        variant="u_deeponet",
    )
    x = torch.randn(1, 8, 8, 2, 2)
    return model, (x,)


def _core_2d_mlpbranch() -> tuple[DeepONet, tuple[torch.Tensor, ...]]:
    """Core-mode 2D builder exercising the MLP-branch (non-spatial) code path.

    The MLP branch consumes a flat ``(B, D_in)`` input rather than a
    packed spatial tensor; this scenario is built against the core
    forward (no ``auto_pad``).
    """
    torch.manual_seed(_SEED)
    model = DeepONet(
        branch1=_make_mlp_branch(
            in_features=4,
            hidden_width=16,
            out_features=8,
            num_layers=2,
        ),
        trunk=_make_trunk(out_features=8),
        dimension=2,
        width=8,
        decoder_type="mlp",
        decoder_width=8,
        decoder_layers=1,
        variant="deeponet",
    )
    x_branch1 = torch.randn(2, 4)  # (B, D_in)
    x_time = torch.linspace(0, 1, 3).unsqueeze(-1)  # (T, 1)
    return model, (x_branch1, x_time)


def _init_lazy(model, *args) -> None:
    """Run one forward pass to materialise ``nn.LazyLinear`` parameters."""
    with torch.no_grad():
        model(*args)


def _load_golden(path: Path) -> dict[str, torch.Tensor | dict]:
    """Load a golden fixture; ``pytest.skip`` with a regen hint if missing.

    Fixtures under ``test/experimental/models/xdeeponet/data/`` are
    committed alongside this file and updated deliberately when model
    numerics intentionally change.  When a fixture is missing — for
    example because a new scenario has been added but its ``.pth`` has
    not yet been generated and committed — the test is skipped (not
    failed) so CI remains green; regenerate with::

        python test/experimental/models/xdeeponet/data/\\
            _generate_xdeeponet_goldens.py

    and commit the resulting ``.pth`` file.
    """
    if not path.exists():
        pytest.skip(
            f"Golden fixture {path.name} is not yet committed. "
            f"Regenerate with "
            f"``python test/experimental/models/xdeeponet/data/"
            f"_generate_xdeeponet_goldens.py`` and commit the "
            f"resulting ``.pth`` file."
        )
    # Golden payload is {str -> Tensor | dict[str, Tensor]} so
    # ``weights_only=True`` is the safer default and avoids PyTorch 2.6's
    # FutureWarning on the permissive load path.
    return torch.load(path, weights_only=True)


# Registry of all (name, builder, golden-path) scenarios; consumed by the
# parameterised non-regression test below and by the golden generator
# script (``_generate_xdeeponet_goldens.py``) so new scenarios are picked
# up in both places by adding one entry here.
_FIXTURE_REGISTRY = [
    ("u_deeponet_packed_2d", _wrapper_2d, _GOLDEN_PACKED_2D),
    ("u_deeponet_packed_3d", _wrapper_3d, _GOLDEN_PACKED_3D),
    ("fourier_packed_2d", _wrapper_2d_fourier, _GOLDEN_PACKED_2D_FOURIER),
    ("mionet_packed_2d", _wrapper_2d_mionet, _GOLDEN_PACKED_2D_MIONET),
    ("temporal_packed_2d", _wrapper_2d_temporal, _GOLDEN_PACKED_2D_TEMPORAL),
    ("packed_2d_multichannel", _packed_2d_multichannel, _GOLDEN_PACKED_2D_MULTICHANNEL),
    ("xfno_packed_3d", _xfno_packed_3d, _GOLDEN_XFNO_PACKED_3D),
    ("xfno_packed_3d_extend", _xfno_packed_3d_extend, _GOLDEN_XFNO_PACKED_3D_EXTEND),
    ("mlpbranch_core_2d", _core_2d_mlpbranch, _GOLDEN_CORE_2D_MLPBRANCH),
]


# ----------------------------------------------------------------------
# Constructor + public attributes (MOD-008a)
# ----------------------------------------------------------------------


class TestDeepONetConstructor:
    """Constructor instantiates and exposes the documented public attributes."""

    @pytest.mark.parametrize(
        "config",
        [
            {"variant": "u_deeponet", "width": 8, "decoder_type": "mlp"},
            {"variant": "u_deeponet", "width": 16, "decoder_type": "conv"},
        ],
        ids=["default-ish", "custom"],
    )
    def test_deeponet_2d_core(self, config):
        """``DeepONet`` stores the constructor arguments on public attrs."""
        model = DeepONet(
            branch1=_make_unet_spatial_branch(dimension=2, width=config["width"]),
            trunk=_make_trunk(out_features=config["width"]),
            dimension=2,
            width=config["width"],
            decoder_type=config["decoder_type"],
            decoder_width=config["width"],
            decoder_layers=1,
            variant=config["variant"],
        )
        assert model.dimension == 2
        assert model.variant == config["variant"]
        assert model.width == config["width"]
        assert model.decoder_type == config["decoder_type"]
        assert model.decoder_activation_fn == "relu"
        assert model.trunk is not None

    @pytest.mark.parametrize(
        "config",
        [
            {"variant": "u_deeponet", "width": 8, "decoder_type": "mlp"},
            {"variant": "u_deeponet", "width": 16, "decoder_type": "conv"},
        ],
        ids=["default-ish", "custom"],
    )
    def test_deeponet_3d_core(self, config):
        """``DeepONet(dimension=3)`` stores the constructor arguments on public attrs."""
        model = DeepONet(
            branch1=_make_unet_spatial_branch(dimension=3, width=config["width"]),
            trunk=_make_trunk(out_features=config["width"]),
            dimension=3,
            width=config["width"],
            decoder_type=config["decoder_type"],
            decoder_width=config["width"],
            decoder_layers=1,
            variant=config["variant"],
        )
        assert model.dimension == 3
        assert model.variant == config["variant"]
        assert model.width == config["width"]
        assert model.decoder_type == config["decoder_type"]
        assert model.decoder_activation_fn == "relu"
        assert model.trunk is not None

    @pytest.mark.parametrize(
        "config",
        [
            {"padding": 8, "variant": "u_deeponet", "trunk_input": "time"},
            {"padding": 16, "variant": "u_deeponet", "trunk_input": "grid"},
        ],
        ids=["default-ish", "custom"],
    )
    def test_packed_2d(self, config):
        """``DeepONet(auto_pad=True)`` exposes padding / variant / trunk_input."""
        model = DeepONet(
            branch1=_make_unet_spatial_branch(dimension=2, width=8),
            trunk=_make_trunk(out_features=8),
            dimension=2,
            width=8,
            decoder_type="mlp",
            decoder_width=8,
            decoder_layers=1,
            auto_pad=True,
            padding=config["padding"],
            trunk_input=config["trunk_input"],
            variant=config["variant"],
        )
        assert model.auto_pad is True
        assert model.padding == config["padding"]
        assert model.variant == config["variant"]
        assert model.trunk_input == config["trunk_input"]

    @pytest.mark.parametrize(
        "config",
        [
            {"padding": 8, "variant": "u_deeponet", "trunk_input": "time"},
            {"padding": 16, "variant": "u_deeponet", "trunk_input": "grid"},
        ],
        ids=["default-ish", "custom"],
    )
    def test_packed_3d(self, config):
        """``DeepONet(dimension=3, auto_pad=True)`` exposes padding / variant / trunk_input."""
        model = DeepONet(
            branch1=_make_unet_spatial_branch(dimension=3, width=8),
            trunk=_make_trunk(out_features=8),
            dimension=3,
            width=8,
            decoder_type="mlp",
            decoder_width=8,
            decoder_layers=1,
            auto_pad=True,
            padding=config["padding"],
            trunk_input=config["trunk_input"],
            variant=config["variant"],
        )
        assert model.dimension == 3
        assert model.auto_pad is True
        assert model.padding == config["padding"]
        assert model.variant == config["variant"]
        assert model.trunk_input == config["trunk_input"]

    def test_simple_fourier_construction(self):
        """Direct DI construction with a Fourier branch + custom trunk.

        Sanity-check that hand-composing :class:`SpatialBranch` and
        :class:`physicsnemo.models.mlp.FullyConnected` modules into a
        :class:`DeepONet` produces a model with the expected attributes
        and that the passed-in module instances are preserved as
        submodules (not copied or rebuilt).
        """
        torch.manual_seed(_SEED)
        branch1 = SpatialBranch(
            dimension=2,
            in_channels=2,
            width=8,
            num_fourier_layers=1,
            modes1=2,
            modes2=2,
            activation_fn="relu",
        )
        trunk = FullyConnected(
            in_features=1,
            layer_size=16,
            out_features=8,
            num_layers=2,
            activation_fn="tanh",
        )
        model = DeepONet(
            branch1=branch1,
            trunk=trunk,
            dimension=2,
            width=8,
            decoder_type="mlp",
            decoder_width=8,
            decoder_layers=1,
            decoder_activation_fn="relu",
            variant="fourier_deeponet",
        )
        assert model.dimension == 2
        assert model.width == 8
        assert model.variant == "fourier_deeponet"
        assert model.auto_pad is False
        # branch1 is a SpatialBranch -> not the MLP-branch path
        assert model._branch1_is_mlp is False
        # trunk is preserved as the passed-in instance (not rebuilt)
        assert model.trunk is trunk
        assert model.branch1 is branch1


# ----------------------------------------------------------------------
# Forward non-regression against committed golden files (MOD-008b)
# ----------------------------------------------------------------------


def _golden_args(golden: dict) -> tuple[torch.Tensor, ...]:
    """Read positional forward arguments from a golden payload.

    Two on-disk schemas are recognised:

    - ``{"args": (tensor, ...), "y": ..., "state_dict": ...}`` (multi-arg)
    - ``{"x": tensor, "y": ..., "state_dict": ...}`` (single-input)
    """
    if "args" in golden:
        args = golden["args"]
        if isinstance(args, (list, tuple)):
            return tuple(args)
        return (args,)
    return (golden["x"],)


class TestDeepONetNonRegression:
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
        """Forward output reproduces the stored golden output bit-for-bit."""
        del name  # used only for the test ID
        golden = _load_golden(golden_path)
        args = _golden_args(golden)
        model, _ = builder()
        _init_lazy(model, *args)
        model.load_state_dict(golden["state_dict"])
        with torch.no_grad():
            y = model(*args)
        torch.testing.assert_close(y, golden["y"], rtol=1e-5, atol=1e-6)


class TestDeepONetTimeAxisExtend:
    """Time-axis-extend (xFNO-style autoregressive bundling).

    Exercises the trunkless packed-input forward path when
    ``time_modes`` is set and ``target_times`` is supplied at forward
    time.  Verifies that the output shape matches the requested forecast
    horizon ``K`` and that the spatial axes are cropped to the
    original input shape.
    """

    def test_predicts_K_future_steps(self):
        model, (x,) = _xfno_packed_3d_extend()
        _init_lazy(model, x)
        # Choose K different from T_in (4) to trigger the time-extend
        # code path.  K=6 should produce output with the last spatial
        # axis = K.
        target_times = torch.linspace(0.5, 1.0, 6)
        with torch.no_grad():
            y = model(x, target_times=target_times)
        # x: (1, 8, 8, 4, 2); output should be (1, 8, 8, K=6, out_channels=1).
        assert y.shape == (1, 8, 8, 6, 1)

    def test_K_equals_T_in_no_extend(self):
        model, (x,) = _xfno_packed_3d_extend()
        _init_lazy(model, x)
        # K == T_in (4): time-extend short-circuits; output keeps the
        # original time-axis length.
        target_times = torch.linspace(0.0, 1.0, 4)
        with torch.no_grad():
            y = model(x, target_times=target_times)
        assert y.shape == (1, 8, 8, 4, 1)


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

    Round-trip is exercised on the wrapper fixtures only; ``Module``
    save/load is class-level, so once it works on one variant it works on
    all of them.  Picking the 2D and 3D U-DeepONet wrappers because those
    are the most user-facing.
    """

    def _roundtrip(self, model, args, tmp_path):
        _init_lazy(model, *args)
        ckpt = tmp_path / "model.mdlus"
        model.save(str(ckpt))
        loaded = Module.from_checkpoint(str(ckpt))
        with torch.no_grad():
            y_loaded = loaded(*args)
        return loaded, y_loaded

    def test_wrapper_2d_roundtrip(self, tmp_path):
        """2D wrapper: reloaded output matches the committed golden."""
        golden = _load_golden(_GOLDEN_PACKED_2D)
        args = _golden_args(golden)
        model, _ = _wrapper_2d()
        loaded, y_loaded = self._roundtrip(model, args, tmp_path)
        assert type(loaded).__name__ == type(model).__name__
        assert loaded.padding == model.padding
        assert loaded.variant == model.variant
        assert loaded.trunk_input == model.trunk_input
        torch.testing.assert_close(y_loaded, golden["y"], rtol=1e-5, atol=1e-6)

    def test_wrapper_3d_roundtrip(self, tmp_path):
        """3D wrapper: reloaded output matches the committed golden."""
        golden = _load_golden(_GOLDEN_PACKED_3D)
        args = _golden_args(golden)
        model, _ = _wrapper_3d()
        loaded, y_loaded = self._roundtrip(model, args, tmp_path)
        assert type(loaded).__name__ == type(model).__name__
        assert loaded.padding == model.padding
        assert loaded.variant == model.variant
        assert loaded.trunk_input == model.trunk_input
        torch.testing.assert_close(y_loaded, golden["y"], rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# Gradient flow
# ----------------------------------------------------------------------


class TestDeepONetGradientFlow:
    """Backward pass produces non-None gradients on input and parameters.

    Tested for both the 2D and 3D wrappers since the 3D forward path
    performs different tensor reshapes (extra unsqueeze, deeper
    permutations) and could in principle fail to propagate gradients
    even when the 2D path works.
    """

    def test_wrapper_2d_gradients(self):
        """Gradients flow through the 2D wrapper."""
        model, (x,) = _wrapper_2d()
        _init_lazy(model, x)
        x = x.detach().requires_grad_(True)
        y = model(x)
        y.sum().backward()
        assert x.grad is not None
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert trainable, "model has no trainable parameters"
        assert any(p.grad is not None for p in trainable)

    def test_wrapper_3d_gradients(self):
        """Gradients flow through the 3D wrapper."""
        model, (x,) = _wrapper_3d()
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
    """``torch.compile`` wraps the model without raising.

    Two variants per dimensionality:

    - ``fullgraph=False`` (the default for production code): the model
      must compile end-to-end with graph breaks tolerated.  Output must
      match eager numerically.
    - ``fullgraph=True``: probes whether the entire forward is
      graph-capturable with no breaks at all.  Marked ``xfail`` with
      ``strict=False`` because graph-break behavior depends on jaxtyping
      decorators and dynamic shape paths in
      :func:`~physicsnemo.experimental.models.xdeeponet._padding.pad_spatial_right`
      that are evaluated under ``torch.compiler.is_compiling()`` guards.
      The test reports XPASS once we eliminate the breaks (e.g. by
      switching jaxtyping off during compile or by going dimensionally
      generic so shape paths constant-fold), at which point the xfail
      marker can be removed.
    """

    def test_wrapper_2d_compile(self):
        """2D compiled model produces shape-compatible output vs eager."""
        model, (x,) = _wrapper_2d()
        _init_lazy(model, x)
        with torch.no_grad():
            y_eager = model(x)
        compiled = torch.compile(model, fullgraph=False)
        with torch.no_grad():
            y_compiled = compiled(x)
        assert y_compiled.shape == y_eager.shape
        torch.testing.assert_close(y_compiled, y_eager, rtol=1e-4, atol=1e-5)

    def test_wrapper_3d_compile(self):
        """3D compiled model produces shape-compatible output vs eager."""
        model, (x,) = _wrapper_3d()
        _init_lazy(model, x)
        with torch.no_grad():
            y_eager = model(x)
        compiled = torch.compile(model, fullgraph=False)
        with torch.no_grad():
            y_compiled = compiled(x)
        assert y_compiled.shape == y_eager.shape
        torch.testing.assert_close(y_compiled, y_eager, rtol=1e-4, atol=1e-5)

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "Forward currently has graph breaks from jaxtyping shape "
            "decorators and dynamic spatial-padding shapes. Marked "
            "strict=False so an XPASS (no breaks) doesn't fail CI; if "
            "the test starts passing reliably, remove this marker."
        ),
    )
    def test_wrapper_2d_compile_fullgraph(self):
        """2D model compiles cleanly with ``fullgraph=True``."""
        model, (x,) = _wrapper_2d()
        _init_lazy(model, x)
        compiled = torch.compile(model, fullgraph=True)
        with torch.no_grad():
            compiled(x)

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "Forward currently has graph breaks from jaxtyping shape "
            "decorators and dynamic spatial-padding shapes. Marked "
            "strict=False so an XPASS (no breaks) doesn't fail CI; if "
            "the test starts passing reliably, remove this marker."
        ),
    )
    def test_wrapper_3d_compile_fullgraph(self):
        """3D model compiles cleanly with ``fullgraph=True``."""
        model, (x,) = _wrapper_3d()
        _init_lazy(model, x)
        compiled = torch.compile(model, fullgraph=True)
        with torch.no_grad():
            compiled(x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
