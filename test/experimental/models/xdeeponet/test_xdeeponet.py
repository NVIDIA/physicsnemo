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

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from physicsnemo import Module
from physicsnemo.core.meta import ModelMetaData
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
# Kitchen-sink scenarios: every major code path turned on simultaneously.
# - 2D variant: packed-input mode + ``temporal_projection`` decoder +
#   ``output_window > 1`` + ``trunk_input="grid"`` + multi-layer lift +
#   ``coord_features`` asymmetry across branches.
# - 3D variant: core mode (``auto_pad=False``) + ``decoder_type="conv"``
#   + mionet dual-branch + deeper trunk (3 layers, no output activation)
#   + ``lift_hidden_width`` set explicitly + a different activation
#   palette (celu / leaky_relu / elu / tanh).
# Together they exercise nearly every constructor knob the model exposes.
_GOLDEN_PACKED_2D_KITCHEN_SINK = _DATA_DIR / "xdeeponet_packed_2d_kitchen_sink_v1.pth"
_GOLDEN_CORE_3D_KITCHEN_SINK = _DATA_DIR / "xdeeponet_core_3d_kitchen_sink_v1.pth"
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
#
# Note on physicsnemo.Module compliance: every submodule passed into
# DeepONet as a constructor argument (branch1, branch2, trunk) must be a
# physicsnemo.Module instance, otherwise Module.save rejects the
# hierarchy at serialization time (see Module._save_process).  A bare
# nn.Sequential wrapper around a FullyConnected does not satisfy that
# contract, so :class:`_MLPWithTrailingActivation` below replaces the
# nn.Sequential pattern used by ``_make_trunk`` and ``_make_mlp_branch``.


@dataclass
class _MLPWithTrailingActivationMeta(ModelMetaData):
    """PhysicsNeMo metadata for the test-only MLP+activation wrapper."""


class _MLPWithTrailingActivation(Module):
    """Test-only physicsnemo.Module replacement for
    ``nn.Sequential(FullyConnected, get_activation(...))``.

    A plain ``nn.Sequential`` cannot be a DeepONet constructor arg
    because :meth:`Module._save_process` rejects ``torch.nn.Module``
    instances in ``_args``; this lightweight subclass satisfies the
    contract without changing forward semantics.  Production users
    wanting the same pattern should define their own
    :class:`physicsnemo.Module` subclass (or use
    :meth:`Module.from_torch` on a custom class).

    Parameters
    ----------
    in_features : int
        Input feature count, forwarded to :class:`FullyConnected`.
    layer_size : int
        Hidden width, forwarded to :class:`FullyConnected.layer_size`.
    out_features : int
        Output feature count, forwarded to :class:`FullyConnected`.
    num_layers : int
        Hidden-layer count, forwarded to :class:`FullyConnected.num_layers`.
    activation_fn : str
        Activation name used both as the FullyConnected hidden activation
        and as the trailing activation applied to the projection output.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape ``(..., in_features)``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape ``(..., out_features)`` after the trailing
        activation.
    """

    def __init__(
        self,
        *,
        in_features: int,
        layer_size: int,
        out_features: int,
        num_layers: int,
        activation_fn: str,
    ):
        super().__init__(meta=_MLPWithTrailingActivationMeta())
        self.fc = FullyConnected(
            in_features=in_features,
            layer_size=layer_size,
            out_features=out_features,
            num_layers=num_layers,
            activation_fn=activation_fn,
        )
        self.activation = get_activation(activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.fc(x))


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
) -> Module:
    """Flat MLP branch: ``num_layers`` activated linears in total.

    Composed as :class:`FullyConnected` (with ``num_layers - 1`` activated
    hidden layers + one unactivated projection) wrapped with a trailing
    activation so every linear is followed by an activation.  The
    wrapping is a :class:`_MLPWithTrailingActivation` instance so the
    branch is a :class:`physicsnemo.Module` and survives
    :meth:`DeepONet.save`.
    """
    return _MLPWithTrailingActivation(
        in_features=in_features,
        layer_size=hidden_width,
        out_features=out_features,
        num_layers=num_layers - 1,
        activation_fn=activation_fn,
    )


def _make_trunk(
    *,
    in_features: int = 1,
    out_features: int,
    hidden_width: int = 16,
    num_layers: int = 2,
    activation_fn: str = "tanh",
    output_activation: bool = True,
) -> Module:
    """Trunk MLP.

    A :class:`FullyConnected` produces ``num_layers`` activated hidden
    linears followed by a single unactivated projection
    (``hidden_width -> out_features``); when ``output_activation`` is
    true the projection is wrapped with a trailing activation.

    Both branches of the conditional return a :class:`physicsnemo.Module`
    so the trunk survives :meth:`DeepONet.save`.  With
    ``output_activation=False`` the bare :class:`FullyConnected` (already
    a physicsnemo.Module) is returned; otherwise a
    :class:`_MLPWithTrailingActivation` wraps the same FullyConnected
    semantics plus a trailing activation.
    """
    if output_activation:
        return _MLPWithTrailingActivation(
            in_features=in_features,
            layer_size=hidden_width,
            out_features=out_features,
            num_layers=num_layers,
            activation_fn=activation_fn,
        )
    return FullyConnected(
        in_features=in_features,
        layer_size=hidden_width,
        out_features=out_features,
        num_layers=num_layers,
        activation_fn=activation_fn,
    )


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
    )
    x = torch.randn(1, 8, 8, 2, 2)
    return model, (x,)


def _packed_2d_kitchen_sink() -> tuple[DeepONet, tuple[torch.Tensor, ...]]:
    """Kitchen-sink 2D builder exercising every major code path.

    Turns on every :class:`SpatialBranch` sub-stack (Fourier + UNet +
    Conv) on **both** primary and secondary branches, the mionet
    dual-branch Hadamard product, the ``temporal_projection`` decoder
    with ``output_window > 1``, multi-channel output
    (``out_channels=2``), ``trunk_input="grid"`` (trunk sees the full
    ``(x, y, t)`` coordinate), a multi-layer pointwise lift network on
    branch1, asymmetric ``coord_features`` and activation functions
    between the two branches, and the Sin trunk activation.

    This is the most complex single configuration the model exposes;
    if anything regresses across these knobs the recorded golden
    payload (or the companion :class:`TestDeepONetStress` checks) will
    surface the regression early.
    """
    torch.manual_seed(_SEED)
    # ``trunk_input="grid"`` in 2D reads the last ``dim+1 = 3`` channels
    # of ``x`` (the (x, y, t) coords) to build the trunk input.  Both
    # branches therefore see ``in_channels=3``; ``coord_features=True``
    # on branch1 lifts to 5 effective channels before the linear lift.
    branch1 = SpatialBranch(
        dimension=2,
        in_channels=3,
        width=8,
        num_fourier_layers=1,
        num_unet_layers=1,
        num_conv_layers=1,
        modes1=2,
        modes2=2,
        kernel_size=3,
        activation_fn="gelu",
        coord_features=True,
        lift_layers=2,
    )
    branch2 = SpatialBranch(
        dimension=2,
        in_channels=3,
        width=8,
        num_fourier_layers=1,
        num_unet_layers=1,
        num_conv_layers=1,
        modes1=2,
        modes2=2,
        kernel_size=3,
        activation_fn="silu",
        coord_features=False,
        lift_layers=1,
    )
    trunk = _make_trunk(
        in_features=3,
        out_features=8,
        hidden_width=16,
        num_layers=2,
        activation_fn="sin",
    )
    model = DeepONet(
        branch1=branch1,
        branch2=branch2,
        trunk=trunk,
        dimension=2,
        width=8,
        out_channels=2,
        decoder_type="temporal_projection",
        decoder_width=8,
        decoder_layers=2,
        decoder_activation_fn="tanh",
        output_window=3,
        auto_pad=True,
        padding=8,
        trunk_input="grid",
    )
    x = torch.randn(1, 8, 8, 2, 3)
    x_branch2 = torch.randn(1, 8, 8, 2, 3)
    return model, (x, x_branch2)


def _core_3d_kitchen_sink() -> tuple[DeepONet, tuple[torch.Tensor, ...]]:
    """Core-mode 3D mionet builder hitting code paths the other fixtures skip.

    Distinct from :func:`_packed_2d_kitchen_sink` along several axes
    that no other fixture exercises:

    - ``auto_pad=False`` (core mode) in 3D with a trunk and a spatial
      branch — the packed-input wrapper path is bypassed entirely;
      ``forward`` dispatches through the ``(x_branch1, x_time,
      x_branch2)`` core entry point.
    - ``decoder_type="conv"`` — exercises the convolutional decoder
      head (``Conv3dFCLayer`` stack with channel-first permute), which
      no other fixture covers.
    - 3D mionet — :func:`_wrapper_2d_mionet` covers the 2D path; this
      is the 3D counterpart.
    - Fourier + UNet + Conv sub-stacks composed on **both** 3D
      branches (``_wrapper_3d`` is UNet-only, ``_xfno_packed_3d`` has
      no Conv stack).
    - ``lift_layers=3`` with ``lift_hidden_width`` set explicitly on
      branch1 — exercises the multi-layer pointwise lift network with
      a custom hidden width.
    - Trunk with ``num_layers=3`` and ``output_activation=False`` — no
      other fixture builds a 3-layer trunk or skips the trailing
      activation wrapper.
    - Activation palette: ``celu`` (branch1), ``leaky_relu`` (branch2),
      ``tanh`` (trunk), ``elu`` (decoder).  None of these appear in
      another fixture.

    The 8x8x8 spatial input is chosen so the UNet sub-stack's pool
    chain doesn't collapse to a 1x1x1 BatchNorm input (training mode
    forbids that with batch_size=1).
    """
    torch.manual_seed(_SEED)
    branch1 = SpatialBranch(
        dimension=3,
        in_channels=2,
        width=8,
        num_fourier_layers=1,
        num_unet_layers=1,
        num_conv_layers=1,
        modes1=2,
        modes2=2,
        modes3=2,
        kernel_size=3,
        activation_fn="celu",
        coord_features=True,
        lift_layers=3,
        lift_hidden_width=12,
    )
    branch2 = SpatialBranch(
        dimension=3,
        in_channels=2,
        width=8,
        num_fourier_layers=1,
        num_unet_layers=1,
        num_conv_layers=1,
        modes1=2,
        modes2=2,
        modes3=2,
        kernel_size=3,
        activation_fn="leaky_relu",
        coord_features=True,
        lift_layers=2,
    )
    trunk = _make_trunk(
        in_features=1,
        out_features=8,
        hidden_width=12,
        num_layers=3,
        activation_fn="tanh",
        output_activation=False,
    )
    model = DeepONet(
        branch1=branch1,
        branch2=branch2,
        trunk=trunk,
        dimension=3,
        width=8,
        out_channels=2,
        decoder_type="conv",
        decoder_width=16,
        decoder_layers=2,
        decoder_activation_fn="elu",
        auto_pad=False,
    )
    x_branch1 = torch.randn(1, 8, 8, 8, 2)
    x_time = torch.linspace(0, 1, 3).unsqueeze(-1)
    x_branch2 = torch.randn(1, 8, 8, 8, 2)
    return model, (x_branch1, x_time, x_branch2)


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
    )
    x_branch1 = torch.randn(2, 4)  # (B, D_in)
    x_time = torch.linspace(0, 1, 3).unsqueeze(-1)  # (T, 1)
    return model, (x_branch1, x_time)


def _init_lazy(model, *args) -> None:
    """Run one forward pass to materialise ``nn.LazyLinear`` parameters."""
    with torch.no_grad():
        model(*args)


def _load_golden(path: Path) -> dict[str, torch.Tensor | dict]:
    """Load a golden fixture; fail with a regen hint if missing.

    Fixtures under ``test/experimental/models/xdeeponet/data/`` are
    committed alongside this file and updated deliberately when model
    numerics intentionally change.  Regenerate with::

        python test/experimental/models/xdeeponet/data/\\
            _generate_xdeeponet_goldens.py

    and commit the resulting ``.pth`` file.
    """
    if not path.exists():
        pytest.fail(
            f"Golden fixture {path.name} is missing. "
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
    ("kitchen_sink_packed_2d", _packed_2d_kitchen_sink, _GOLDEN_PACKED_2D_KITCHEN_SINK),
    ("kitchen_sink_core_3d", _core_3d_kitchen_sink, _GOLDEN_CORE_3D_KITCHEN_SINK),
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
            {"width": 8, "decoder_type": "mlp"},
            {"width": 16, "decoder_type": "conv"},
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
        )
        assert model.dimension == 2
        assert model.width == config["width"]
        assert model.decoder_type == config["decoder_type"]
        assert model.decoder_activation_fn == "relu"
        assert model.trunk is not None

    @pytest.mark.parametrize(
        "config",
        [
            {"width": 8, "decoder_type": "mlp"},
            {"width": 16, "decoder_type": "conv"},
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
        )
        assert model.dimension == 3
        assert model.width == config["width"]
        assert model.decoder_type == config["decoder_type"]
        assert model.decoder_activation_fn == "relu"
        assert model.trunk is not None

    @pytest.mark.parametrize(
        "config",
        [
            {"padding": 8, "trunk_input": "time"},
            {"padding": 16, "trunk_input": "grid"},
        ],
        ids=["default-ish", "custom"],
    )
    def test_packed_2d(self, config):
        """``DeepONet(auto_pad=True)`` exposes padding / trunk_input."""
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
        )
        assert model.auto_pad is True
        assert model.padding == config["padding"]
        assert model.trunk_input == config["trunk_input"]

    @pytest.mark.parametrize(
        "config",
        [
            {"padding": 8, "trunk_input": "time"},
            {"padding": 16, "trunk_input": "grid"},
        ],
        ids=["default-ish", "custom"],
    )
    def test_packed_3d(self, config):
        """``DeepONet(dimension=3, auto_pad=True)`` exposes padding / trunk_input."""
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
        )
        assert model.dimension == 3
        assert model.auto_pad is True
        assert model.padding == config["padding"]
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
        )
        assert model.dimension == 2
        assert model.width == 8
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
      graph-capturable with no breaks at all.  Jaxtyping shape
      decorators and the dynamic spatial-padding paths in
      :func:`~physicsnemo.experimental.models.xdeeponet._padding.pad_spatial_right`
      are evaluated under ``torch.compiler.is_compiling()`` guards so
      they constant-fold during compile; both 2D and 3D forward paths
      currently compile cleanly with no graph breaks across the
      torch versions exercised in CI and locally.  If a future torch
      update reintroduces breaks the assertion below will fail; re-add
      ``@pytest.mark.xfail(strict=False)`` until the breaks are fixed.
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

    def test_wrapper_2d_compile_fullgraph(self):
        """2D model compiles cleanly with ``fullgraph=True``."""
        model, (x,) = _wrapper_2d()
        _init_lazy(model, x)
        compiled = torch.compile(model, fullgraph=True)
        with torch.no_grad():
            compiled(x)

    def test_wrapper_3d_compile_fullgraph(self):
        """3D model compiles cleanly with ``fullgraph=True``."""
        model, (x,) = _wrapper_3d()
        _init_lazy(model, x)
        compiled = torch.compile(model, fullgraph=True)
        with torch.no_grad():
            compiled(x)


# ----------------------------------------------------------------------
# Stress test: kitchen-sink configuration
# ----------------------------------------------------------------------


class TestDeepONetStress:
    """Stress-test the kitchen-sink configurations end-to-end.

    The fixture-pinned non-regression checks on
    ``kitchen_sink_packed_2d`` and ``kitchen_sink_core_3d`` (above, in
    :class:`TestDeepONetNonRegression`) already verify numerics against
    committed goldens.  This class complements them by exercising the
    same configurations through three dynamic-behaviour checks per
    dimensionality, mirroring the structure of
    :class:`TestDeepONetGradientFlow` and :class:`TestDeepONetCompile`:

    - forward output shape matches the expected contract,
    - the backward pass populates gradients on every input tensor and
      on at least one trainable parameter,
    - ``torch.compile(fullgraph=False)`` produces eager-equivalent
      output (full-graph compile is probed on the simpler wrappers in
      :class:`TestDeepONetCompile`; skipping it here to keep the test
      runtime reasonable).

    The 2D kitchen-sink combines: Fourier + UNet + Conv sub-stacks on
    both branches, the mionet dual-branch Hadamard product, the
    ``temporal_projection`` decoder with ``output_window > 1``,
    multi-channel output, ``trunk_input="grid"``, a multi-layer
    pointwise lift on branch1, asymmetric ``coord_features`` and
    activation functions across branches, and the Sin trunk activation.

    The 3D kitchen-sink covers a deliberately disjoint set of code
    paths: ``auto_pad=False`` (core mode) with a 3D spatial branch,
    ``decoder_type="conv"``, 3D mionet, all three sub-stacks on both
    branches, ``lift_layers=3`` with ``lift_hidden_width`` set, a
    3-layer trunk with ``output_activation=False``, and a non-default
    activation palette (celu / leaky_relu / tanh / elu).

    If any code path among these regresses, this class flags it
    independently of any fixture-numerics drift.
    """

    def test_forward_shape_2d(self):
        """2D kitchen-sink: output shape matches ``(B, H, W, K, oc)``."""
        model, args = _packed_2d_kitchen_sink()
        _init_lazy(model, *args)
        with torch.no_grad():
            y = model(*args)
        assert y.shape == (1, 8, 8, 3, 2)

    def test_forward_shape_3d(self):
        """3D kitchen-sink: output shape matches ``(B, X, Y, Z, T, oc)``."""
        model, args = _core_3d_kitchen_sink()
        _init_lazy(model, *args)
        with torch.no_grad():
            y = model(*args)
        assert y.shape == (1, 8, 8, 8, 3, 2)

    def test_gradients_2d(self):
        """2D kitchen-sink: backward populates gradients on inputs and params."""
        model, args = _packed_2d_kitchen_sink()
        _init_lazy(model, *args)
        args = tuple(a.detach().requires_grad_(True) for a in args)
        y = model(*args)
        y.sum().backward()
        for i, a in enumerate(args):
            assert a.grad is not None, f"input arg[{i}] has no gradient"
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert trainable, "model has no trainable parameters"
        assert any(p.grad is not None for p in trainable)

    def test_gradients_3d(self):
        """3D kitchen-sink: backward populates gradients on inputs and params."""
        model, args = _core_3d_kitchen_sink()
        _init_lazy(model, *args)
        args = tuple(a.detach().requires_grad_(True) for a in args)
        y = model(*args)
        y.sum().backward()
        for i, a in enumerate(args):
            assert a.grad is not None, f"input arg[{i}] has no gradient"
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert trainable, "model has no trainable parameters"
        assert any(p.grad is not None for p in trainable)

    def test_compile_2d(self):
        """2D kitchen-sink: ``torch.compile(fullgraph=False)`` parity."""
        model, args = _packed_2d_kitchen_sink()
        _init_lazy(model, *args)
        with torch.no_grad():
            y_eager = model(*args)
        compiled = torch.compile(model, fullgraph=False)
        with torch.no_grad():
            y_compiled = compiled(*args)
        assert y_compiled.shape == y_eager.shape
        torch.testing.assert_close(y_compiled, y_eager, rtol=1e-4, atol=1e-5)

    def test_compile_3d(self):
        """3D kitchen-sink: ``torch.compile(fullgraph=False)`` parity."""
        model, args = _core_3d_kitchen_sink()
        _init_lazy(model, *args)
        with torch.no_grad():
            y_eager = model(*args)
        compiled = torch.compile(model, fullgraph=False)
        with torch.no_grad():
            y_compiled = compiled(*args)
        assert y_compiled.shape == y_eager.shape
        torch.testing.assert_close(y_compiled, y_eager, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
