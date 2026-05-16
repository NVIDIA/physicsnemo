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

"""Core xDeepONet architectures for 2D and 3D operator learning.

The xDeepONet family extends the original DeepONet with eight variants
that cover both single-input and multi-input operator learning, including
the Temporal Neural Operator (TNO) for autoregressive temporal bundling:

- ``deeponet``           — basic DeepONet (MLP branch).
- ``u_deeponet``         — UNet-enhanced spatial branch.
- ``fourier_deeponet``   — spectral (Fourier) spatial branch.
- ``conv_deeponet``      — plain convolutional spatial branch.
- ``hybrid_deeponet``    — Fourier + UNet + Conv spatial branch.
- ``mionet``             — two-branch multi-input operator network.
- ``fourier_mionet``     — MIONet with a Fourier spatial branch.
- ``tno``                — Temporal Neural Operator (branch2 = previous
  solution, autoregressive only).

The core :class:`DeepONet` class is dimension-generic: pass
``dimension=2`` for 2D spatial inputs ``(B, H, W, C)`` and ``dimension=3``
for 3D volumetric inputs ``(B, X, Y, Z, C)``.  Construction is the same
in both cases — a primary branch (``branch1``), an optional secondary
branch (``branch2`` for MIONet/TNO), a coordinate trunk, and a decoder —
with per-dimension primitives dispatched internally through a small
lookup table (see :data:`SpatialBranch._DIM_LAYERS` and ``_DIM_DEFAULTS``
in this module).

References
----------
- Lu, L. et al. (2021). "Learning nonlinear operators via DeepONet."
  *Nature Machine Intelligence*, 3, 218-229.
- Jin, P., Meng, S. & Lu, L. (2022). "MIONet: Learning multiple-input
  operators via tensor product." *SIAM J. Sci. Comp.*, 44(6), A3490-A3514.
- Diab, W. & Al Kobaisi, M. (2024). "U-DeepONet: U-Net enhanced deep
  operator network for geologic carbon sequestration."
  *Scientific Reports*, 14, 21298.
- Zhu, M. et al. (2023). "Fourier-DeepONet: Fourier-enhanced deep operator
  networks for full waveform inversion." arXiv:2305.17289.
- Diab, W. & Al Kobaisi, M. (2025). "Temporal neural operator for modeling
  time-dependent physical phenomena." *Scientific Reports*, 15.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.xdeeponet._padding import (
    compute_right_pad_to_multiple,
    pad_spatial_right,
)
from physicsnemo.experimental.models.xdeeponet.branches import SpatialBranch
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.nn import Conv2dFCLayer, Conv3dFCLayer, get_activation

# Type alias for the enumerated ``decoder_type`` parameter.  Annotating
# with ``Literal`` rather than bare ``str`` lets static type checkers
# and IDEs flag unknown values at the call site; the runtime
# ``.lower()`` normalization and ``ValueError`` guard below remain in
# place so mixed-case strings still flow through (Python does not
# enforce ``Literal`` at runtime).
_DecoderTypeStr = Literal["mlp", "conv", "temporal_projection"]

# Supported decoder types -- runtime view of the ``_DecoderTypeStr``
# alias.  Used by ``DeepONet.__init__`` to reject unknown decoder
# types at the API boundary instead of deferring to ``_build_decoder``
# and raising cryptically from deep inside construction.
_VALID_DECODER_TYPES = frozenset(get_args(_DecoderTypeStr))


@dataclass
class _DeepONetMetaData(ModelMetaData):
    """PhysicsNeMo model metadata for :class:`DeepONet`."""


# Per-dimension defaults referenced by the class docstring (input
# channels / modes defaults that users can copy into their own branch
# construction).  See ``branches._DIM_LAYERS`` for the matching
# per-dimension layer-class lookup table.
_DIM_DEFAULTS: dict[int, dict] = {
    2: {
        "default_in_channels": 12,
        "default_modes": (12, 12),
        "ConvNdFC": "Conv2dFCLayer",
    },
    3: {
        "default_in_channels": 11,
        "default_modes": (10, 10, 8),
        "ConvNdFC": "Conv3dFCLayer",
    },
}


# ---------------------------------------------------------------------------
# 2D DeepONet
# ---------------------------------------------------------------------------


class DeepONet(Module):
    r"""Dimension-generic xDeepONet core architecture for operator learning.

    Combines a primary spatial/MLP branch, an optional secondary branch
    (for MIONet/TNO variants), a coordinate trunk, and a decoder.  The
    branch outputs and trunk are combined via Hadamard product and then
    projected to the output by the decoder.

    The same class handles 2D inputs ``(B, H, W, C)`` and 3D inputs
    ``(B, X, Y, Z, C)``; the spatial dimensionality is selected through
    the :attr:`dimension` constructor argument and the per-dimension
    primitives (``SpectralConv*d``, ``Conv*dFCLayer``, ``Adaptive*Pool*d``,
    UNet adapters) are dispatched internally.

    Parameters
    ----------
    branch1 : torch.nn.Module
        Primary branch.  Spatial branches must be a :class:`SpatialBranch`
        instance (or subclass) and produce a channels-last output of shape
        :math:`(B, *spatial, \text{width})`.  Any other module is treated
        as an MLP branch and must consume a 2D input :math:`(B, D_{in})`
        and produce a 2D output :math:`(B, \text{width})`.
    trunk : torch.nn.Module, optional
        Trunk MLP.  Takes coordinate queries of shape :math:`(T, D_{in})`
        and produces :math:`(T, \text{width})`.  Set to ``None`` to build
        a trunkless operator (the branch output is fed directly to the
        decoder, skipping the branch-trunk Hadamard product).  This is
        the xFNO operator shape and is the recommended entry point for
        Fourier-only operators that don't need coordinate queries.
    branch2 : torch.nn.Module, optional
        Secondary branch for MIONet / TNO variants.  Must produce the same
        output rank as ``branch1`` (both spatial or both flat); the
        constructor rejects mixed configurations up front.
    dimension : Literal[2, 3], optional
        Spatial dimensionality of the inputs.  Must be ``2`` (default) or
        ``3``.
    width : int, optional
        Latent width.  Must match the output channel dim of ``branch1``,
        ``branch2`` (if any), and (when present) ``trunk``.
    out_channels : int, optional
        Number of output channels.  Default ``1``.  The decoder's final
        layer maps the latent width to ``out_channels``; the output
        tensor always carries an explicit trailing channel dim of size
        ``out_channels``.
    decoder_type : Literal["mlp", "conv", "temporal_projection"], optional
        Decoder choice: ``"mlp"`` queries the trunk at each target
        timestep and applies an MLP decoder; ``"conv"`` uses a
        convolutional decoder; ``"temporal_projection"`` queries the
        trunk once and projects the combined latent to K timesteps via a
        learned linear head for fast autoregressive bundling.
        ``"temporal_projection"`` requires a trunk.
        Mixed-case strings are accepted at runtime and lowercased.
    decoder_width : int, optional
        Decoder hidden width.
    decoder_layers : int, optional
        Decoder layer count.
    decoder_activation_fn : str, optional
        Activation function name for the decoder.  Resolved at decoder
        construction time via
        :func:`physicsnemo.nn.module.activations.get_activation`; see
        :data:`physicsnemo.nn.module.activations.ACT2FN` for the full
        set of supported names (e.g. ``"relu"``, ``"gelu"``, ``"silu"``,
        ``"tanh"``, ``"sin"``).  Default ``"relu"``.
    output_window : int, optional
        Output window length K for the ``"temporal_projection"`` decoder.
        When supplied the temporal head is constructed at ``__init__``, which
        produces a deterministic ``state_dict`` and makes checkpoint
        round-tripping straightforward.  When omitted,
        :meth:`set_output_window` must be called before the first forward
        pass.
    auto_pad : bool, optional
        When ``True`` (default ``False``) the packed-input forward path
        right-pads the spatial dims to a multiple of 8 (with a floor of
        ``padding``) before running the core operator and crops back
        afterwards.
    padding : int, optional
        Minimum right-side padding for the spatial dims when
        ``auto_pad=True``.  Rounded up to a multiple of 8.  Default ``8``.
    trunk_input : Literal["time", "grid"], optional
        Used by trunked packed-input mode to decide how trunk coordinates
        are extracted from the channel dimension of the packed input.
        ``"time"`` (default) treats the last channel as time;
        ``"grid"`` treats the last :math:`d+1` channels as
        ``(x, y, [z,] t)``.  Ignored in trunkless mode.
    time_modes : int, optional
        Enables xFNO-style time-axis autoregressive bundling.  Only
        meaningful in **trunkless packed-input mode** (``trunk=None``
        and ``auto_pad=True``).  When set and ``target_times`` of length
        :math:`K` is supplied at forward time, the last spatial axis is
        treated as the time axis and replicate-padded to
        :math:`\max(T_{in} + K, 2 \cdot \texttt{time\_modes})` before the
        branch runs, then cropped to the K future steps.  Must equal
        the Fourier-modes count along the time axis of the branch.
    Forward
    -------
    Six valid call conventions, dispatched by :attr:`auto_pad` and
    whether :attr:`trunk` is ``None``.  The output tensor's trailing
    dim is always :attr:`out_channels` (no implicit squeeze).

    +------------+-----------------+--------+--------------------------+---------------------------------+
    | ``auto_pad``| ``trunk``      | branch | call style               | output shape                    |
    +============+=================+========+==========================+=================================+
    | ``False``  | module          | spatial| ``model(x_b, x_t)``      | ``(B, *spatial, T, oc)``        |
    +------------+-----------------+--------+--------------------------+---------------------------------+
    | ``False``  | module          | mlp    | ``model(x_b, x_t)``      | ``(B, T, oc)``                  |
    +------------+-----------------+--------+--------------------------+---------------------------------+
    | ``True``   | module          | spatial| ``model(x)`` packed      | ``(B, *spatial, T, oc)``        |
    +------------+-----------------+--------+--------------------------+---------------------------------+
    | ``True``   | module + ``output_window=K``  | ``model(x)``  | ``(B, *spatial, K, oc)`` (temporal_projection) |
    +------------+-----------------+--------+--------------------------+---------------------------------+
    | ``False``  | ``None``        | spatial| ``model(x)``             | ``(B, *spatial, oc)``           |
    +------------+-----------------+--------+--------------------------+---------------------------------+
    | ``True``   | ``None``        | spatial| ``model(x)`` (+ optional ``target_times`` w/ time_modes set) | ``(B, *spatial', oc)`` |
    +------------+-----------------+--------+--------------------------+---------------------------------+

    In trunkless packed mode with ``time_modes`` set and ``target_times``
    of length K, the last spatial axis is sliced to the K future steps
    (:math:`[T_{in} : T_{in}+K]`); otherwise the spatial axes are
    cropped back to the original input shape.

    Notes
    -----
    The :class:`SpatialBranch` ``in_channels`` defaults to ``12`` for
    ``dimension=2`` and ``11`` for ``dimension=3``; default Fourier modes
    are ``(12, 12)`` and ``(10, 10, 8)`` respectively.

    Examples
    --------
    2D U-DeepONet (trunked):

    >>> import torch
    >>> from physicsnemo.experimental.models.xdeeponet import DeepONet, SpatialBranch
    >>> from physicsnemo.models.mlp import FullyConnected
    >>> branch1 = SpatialBranch(
    ...     dimension=2, in_channels=5, width=64,
    ...     num_unet_layers=1, kernel_size=3, activation_fn="tanh",
    ... )
    >>> trunk = FullyConnected(
    ...     in_features=1, layer_size=64, out_features=64,
    ...     num_layers=4, activation_fn="sin",
    ... )
    >>> model = DeepONet(
    ...     branch1=branch1, trunk=trunk,
    ...     dimension=2, width=64, out_channels=1,
    ...     decoder_type="mlp", decoder_width=64, decoder_layers=2,
    ... )
    >>> x_branch = torch.randn(2, 32, 32, 5)
    >>> x_time = torch.linspace(0, 1, 3).unsqueeze(-1)
    >>> out = model(x_branch, x_time)                  # (2, 32, 32, 3, 1)

    3D U-FNO (trunkless, packed input with auto_pad + time-axis-extend):

    >>> branch1 = SpatialBranch(
    ...     dimension=3, in_channels=2, width=32,
    ...     num_fourier_layers=4, num_unet_layers=0,
    ...     modes1=12, modes2=12, modes3=8,
    ...     coord_features=True,
    ... )
    >>> model = DeepONet(
    ...     branch1=branch1, trunk=None,
    ...     dimension=3, width=32, out_channels=1,
    ...     decoder_type="mlp", decoder_width=32, decoder_layers=2,
    ...     auto_pad=True, padding=8,
    ...     time_modes=8,                              # enables time-extend
    ... )
    >>> x = torch.randn(1, 32, 32, 4, 2)               # (B, H, W, T_in=4, C)
    >>> y = model(x)                                    # (1, 32, 32, 4, 1) -- predict same length
    >>> t_future = torch.linspace(0.5, 1.0, 6)         # K=6 future steps
    >>> y_future = model(x, target_times=t_future)     # (1, 32, 32, 6, 1)
    """

    def __init__(
        self,
        branch1: nn.Module,
        *,
        trunk: nn.Module | None = None,
        branch2: nn.Module | None = None,
        dimension: Literal[2, 3] = 2,
        width: int = 64,
        out_channels: int = 1,
        decoder_type: _DecoderTypeStr = "mlp",
        decoder_width: int = 128,
        decoder_layers: int = 2,
        decoder_activation_fn: str = "relu",
        output_window: int | None = None,
        auto_pad: bool = False,
        padding: int = 8,
        trunk_input: Literal["time", "grid"] = "time",
        time_modes: int | None = None,
    ):
        super().__init__(meta=_DeepONetMetaData())

        if dimension not in _DIM_DEFAULTS:
            raise ValueError(
                f"DeepONet only supports dimension=2 or dimension=3, "
                f"got dimension={dimension!r}."
            )
        self.dimension = dimension

        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}.")
        self.out_channels = out_channels

        decoder_type_lc = decoder_type.lower()
        if decoder_type_lc not in _VALID_DECODER_TYPES:
            raise ValueError(
                f"Unknown decoder_type: {decoder_type!r}. Valid: "
                f"{sorted(_VALID_DECODER_TYPES)}."
            )
        self.decoder_type = decoder_type_lc
        self.decoder_activation_fn = decoder_activation_fn

        if trunk_input not in ("time", "grid"):
            raise ValueError(
                f"trunk_input must be 'time' or 'grid', got {trunk_input!r}."
            )
        self.trunk_input = trunk_input

        # Auto-padding: when enabled, the packed-input forward path
        # right-pads the spatial dims to a multiple of 8 (with a floor of
        # ``padding``) before running the core operator and crops back
        # afterwards.
        if padding < 0:
            raise ValueError(f"padding must be non-negative, got {padding}.")
        self.auto_pad = auto_pad
        # Round the padding up to a multiple of 8 so the UNet pooling
        # chain stays evenly divisible.  Stored even when
        # ``auto_pad=False`` so callers can inspect the value, but the
        # forward path only consults it when ``auto_pad=True``.
        self.padding = ((padding + 7) // 8) * 8 if padding % 8 != 0 else padding

        # Time-axis-extend: when set together with ``trunk=None`` and
        # ``auto_pad=True``, the packed-input forward path interprets the
        # last spatial axis as the time axis.  Given ``target_times`` of
        # length ``K`` it right-replicate-pads that axis to at least
        # ``max(T_in + K, 2 * time_modes)`` before running the branch,
        # then crops the output to the predicted ``K`` future steps.
        # Only meaningful in trunkless packed mode.
        if time_modes is not None and trunk is not None:
            raise ValueError(
                "time_modes is only meaningful when trunk is None "
                "(xFNO-style trunkless operators).  Drop time_modes, or "
                "set trunk=None."
            )
        if time_modes is not None and not auto_pad:
            raise ValueError(
                "time_modes requires auto_pad=True; the time-axis-extend "
                "feature is part of the packed-input forward path."
            )
        if time_modes is not None and time_modes < 1:
            raise ValueError(f"time_modes must be >= 1, got {time_modes}.")
        self.time_modes = time_modes

        self.width = width

        # Cached forward-time permute / ndim values; computed once at
        # construction so the forward path can stay dimension-agnostic
        # without rebuilding tuples on every call (and so torch.compile
        # sees them as Python constants per model instance).
        self._spatial_branch_ndim = dimension + 2
        self._spatial_axes = tuple(range(2, dimension + 2))
        # Trunked-mode permutes: ``combined`` has rank ``dimension + 3``
        # ``(B, T, *spatial, channels)``.
        self._mlp_decoder_permute = (0, *self._spatial_axes, 1, dimension + 2)
        self._conv_decoder_in_permute = (0, 1, dimension + 2, *self._spatial_axes)
        self._conv_decoder_out_permute = (
            0,
            *tuple(range(3, dimension + 3)),
            1,
            2,
        )
        # Trunkless-mode permutes: branch output has rank ``dimension + 2``
        # ``(B, *spatial, channels)``; the channel axis is moved to /
        # from position 1 for conv-decoder dispatch.
        self._trunkless_channel_first_permute = (
            0,
            dimension + 1,
            *range(1, dimension + 1),
        )
        self._trunkless_channel_last_permute = (
            0,
            *range(2, dimension + 2),
            1,
        )

        # Detect MLP vs spatial branches via instance check.  This drives
        # both the runtime forward dispatch (different unsqueeze / permute
        # paths for spatial-vs-MLP branch outputs) and the fail-fast
        # validation below.  A non-:class:`SpatialBranch` module is
        # assumed to produce a flat ``(B, width)`` output, matching the
        # MLP-branch shape contract.
        self._branch1_is_mlp = not isinstance(branch1, SpatialBranch)
        self.has_branch2 = branch2 is not None
        self._branch2_is_mlp = self.has_branch2 and not isinstance(
            branch2, SpatialBranch
        )

        # ``temporal_projection`` decoder only makes sense with a trunk
        # (it projects the trunk-queried combined latent to ``K`` output
        # timesteps via a learned linear head).  Without a trunk there is
        # no temporal-query axis to project from.
        if trunk is None and self.decoder_type == "temporal_projection":
            raise ValueError(
                "decoder_type='temporal_projection' requires a trunk; "
                "use decoder_type='mlp' or 'conv' for trunkless operators."
            )

        # ``temporal_projection`` and ``conv`` decoders need a spatial
        # ``combined`` tensor.  MLP-branch forward produces a flat 3D
        # tensor of shape (B, T, width), incompatible with both: ``conv``
        # crashes inside the decoder; ``temporal_projection`` silently
        # drops the temporal head.  Fail fast at construction.
        if self._branch1_is_mlp and self.decoder_type in (
            "temporal_projection",
            "conv",
        ):
            raise ValueError(
                f"decoder_type={self.decoder_type!r} is not supported with "
                "MLP branches.  Use decoder_type='mlp', or pass a "
                "SpatialBranch as branch1."
            )

        # Reject mixed (MLP branch1, spatial branch2): forward assumes
        # branch2's output has the same rank as branch1's, otherwise the
        # Hadamard product broadcasts nonsensically or raises a cryptic
        # dim-mismatch error.
        if self.has_branch2 and self._branch1_is_mlp and not self._branch2_is_mlp:
            raise ValueError(
                "When branch1 is an MLP branch, branch2 must also be an "
                "MLP branch (i.e. produce a 2D (B, width) output).  "
                "Swap branch1 and branch2, or pass a SpatialBranch as "
                "branch1."
            )

        # Reject MLP branch + auto_pad: packed-input mode assumes the
        # input has spatial axes to pad and (in trunked mode) a time
        # axis to strip.  MLP branches consume flat ``(B, D_in)`` input
        # and have neither.
        if self._branch1_is_mlp and auto_pad:
            raise ValueError(
                "auto_pad=True requires a SpatialBranch branch1 (the "
                "packed-input forward path operates on spatial dims).  "
                "Use auto_pad=False with an MLP branch."
            )

        # Register submodules.
        self.branch1 = branch1
        if self.has_branch2:
            self.branch2 = branch2
        # ``self.trunk`` is registered unconditionally (None or a module);
        # PyTorch handles None submodule attributes fine.
        self.trunk = trunk

        if self.decoder_type == "temporal_projection":
            self._temporal_projection = True
            self.decoder = self._build_decoder(
                width,
                width,
                decoder_layers,
                decoder_width,
                "mlp",
                decoder_activation_fn,
            )
            # Preferred path: construct the temporal head at __init__ so
            # state_dict keys are deterministic and checkpointing just works.
            # When ``output_window`` is not provided the user must call
            # :meth:`set_output_window` before the first forward pass.
            # The head projects to ``output_window * out_channels`` so a
            # multi-channel output is reshaped at the end.
            if output_window is not None:
                if output_window < 1:
                    raise ValueError(
                        f"output_window must be a positive integer, got {output_window}"
                    )
                self.temporal_head = nn.Linear(self.width, output_window * out_channels)
            else:
                self.temporal_head = None
        else:
            self._temporal_projection = False
            self.decoder = self._build_decoder(
                width,
                out_channels,
                decoder_layers,
                decoder_width,
                self.decoder_type,
                decoder_activation_fn,
            )

    @property
    def has_temporal_projection(self) -> bool:
        """Whether the model was constructed with the temporal-projection
        decoder (``decoder_type="temporal_projection"``).

        Public read-only view of the internal flag; preferred over reaching
        into the private attribute from outside the class.
        """
        return self._temporal_projection

    def set_output_window(self, K: int):
        """Create the temporal-projection head for K output timesteps.

        The head projects to ``K * out_channels`` so the trailing
        out-channels dim is preserved in the output.  Only effective
        when ``decoder_type="temporal_projection"``.
        """
        if self._temporal_projection:
            device = next(self.parameters()).device
            self.temporal_head = nn.Linear(self.width, K * self.out_channels).to(device)

    def _build_decoder(
        self,
        width: int,
        out_channels: int,
        num_layers: int,
        hidden_width: int,
        decoder_type: str,
        activation_fn: str,
    ) -> nn.Module:
        # Per-dimension dispatch for the spatial decoder.
        ConvNdFC = Conv2dFCLayer if self.dimension == 2 else Conv3dFCLayer

        if decoder_type == "mlp":
            if num_layers == 0:
                return nn.Linear(width, out_channels)
            return FullyConnected(
                width, hidden_width, out_channels, num_layers, activation_fn
            )

        elif decoder_type == "conv":
            if num_layers == 0:
                return ConvNdFC(width, out_channels)

            layers = []
            in_ch = width
            for _ in range(num_layers):
                layers.extend(
                    [ConvNdFC(in_ch, hidden_width), get_activation(activation_fn)]
                )
                in_ch = hidden_width
            layers.append(ConvNdFC(hidden_width, out_channels))
            return nn.Sequential(*layers)

        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(
        self,
        *args: Float[Tensor, "..."],
        x_branch2: Float[Tensor, "..."] | None = None,
        target_times: Float[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "..."]:
        """Forward pass.

        Dispatched by the :attr:`auto_pad` constructor flag:

        - **Packed mode** (``auto_pad=True``): ``model(x)`` (or
          ``model(x, x_branch2)`` for MIONet-style dual-branch variants).
          ``x`` has shape :math:`(B, *spatial, T, C)` with the time axis
          and trunk / grid coordinates encoded in the channel dimension.
          The model extracts the spatial branch input and trunk
          coordinates itself (using :attr:`trunk_input`), right-pads the
          spatial dims when :attr:`padding` is positive, runs the core
          operator, and crops the output back to the original spatial
          extent.  ``target_times`` (keyword) selects an explicit set of
          trunk query coordinates.

        - **Core mode** (``auto_pad=False``, the default):
          ``model(x_branch1, x_time)`` (or
          ``model(x_branch1, x_time, x_branch2)``).  Required for the
          MLP-branch code path (where there is no spatial axis to
          extract from) and for power users who assemble the trunk
          coordinates themselves.  ``target_times`` must be ``None``
          in this mode.

        ``x_branch2`` may be passed positionally (second arg in packed
        mode, third arg in core mode) or by keyword.
        """
        # Branch on the four (auto_pad, trunk-is-None) combinations.
        if self.auto_pad:
            if self.trunk is None:
                # Trunkless packed mode: ``model(x)`` only.
                if len(args) != 1:
                    raise TypeError(
                        f"In trunkless packed-input mode (auto_pad=True, "
                        f"trunk=None), forward expects exactly 1 positional "
                        f"tensor, got {len(args)}."
                    )
                if x_branch2 is not None:
                    raise TypeError("x_branch2 is not supported in trunkless mode.")
                return self._forward_packed_trunkless(
                    args[0], target_times=target_times
                )

            # Trunked packed mode.
            if len(args) == 1:
                return self._forward_packed(
                    args[0],
                    x_branch2=x_branch2,
                    target_times=target_times,
                )
            if len(args) == 2:
                if x_branch2 is not None:
                    raise TypeError(
                        "x_branch2 supplied both positionally and as a "
                        "keyword argument."
                    )
                return self._forward_packed(
                    args[0],
                    x_branch2=args[1],
                    target_times=target_times,
                )
            raise TypeError(
                f"In trunked packed-input mode (auto_pad=True, trunk!=None), "
                f"forward expects 1 or 2 positional tensors ((x,) or "
                f"(x, x_branch2)), got {len(args)}."
            )

        # Core mode (auto_pad=False).
        if target_times is not None:
            raise TypeError(
                "target_times is only valid in packed-input mode "
                "(construct DeepONet with auto_pad=True)."
            )
        if self.trunk is None:
            # Trunkless core mode: ``model(x)`` only.
            if len(args) != 1:
                raise TypeError(
                    f"In trunkless core mode (auto_pad=False, trunk=None), "
                    f"forward expects exactly 1 positional tensor, got "
                    f"{len(args)}."
                )
            if x_branch2 is not None:
                raise TypeError("x_branch2 is not supported in trunkless mode.")
            return self._forward_core(args[0], None, x_branch2=None)

        # Trunked core mode.
        if len(args) == 2:
            x_branch1, x_time = args
            b2 = x_branch2
        elif len(args) == 3:
            if x_branch2 is not None:
                raise TypeError(
                    "x_branch2 supplied both positionally and as a keyword argument."
                )
            x_branch1, x_time, b2 = args
        else:
            raise TypeError(
                f"In trunked core mode (auto_pad=False, trunk!=None), "
                f"forward expects 2 ((x_branch1, x_time)) or 3 "
                f"((x_branch1, x_time, x_branch2)) positional tensors, "
                f"got {len(args)}."
            )
        return self._forward_core(x_branch1, x_time, x_branch2=b2)

    def _forward_packed(
        self,
        x: Float[Tensor, "..."],
        *,
        x_branch2: Float[Tensor, "..."] | None = None,
        target_times: Float[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "..."]:
        """Trunked packed-input forward: unpack ``x`` and (optionally) auto-pad.

        ``x`` has shape :math:`(B, *spatial, T, C)`; this method:

        1. Optionally right-pads the spatial dims to a multiple of 8
           when :attr:`auto_pad` is ``True``.
        2. Extracts the spatial branch input as ``x[..., 0, :]`` (the
           ``T=0`` slice).
        3. Builds the trunk coordinates from ``x`` (or ``target_times``
           when provided) according to :attr:`trunk_input`.
        4. Runs :meth:`_forward_core` and crops back to the original
           spatial extent if auto-padding was applied.
        """
        dim = self.dimension
        expected_ndim = dim + 3  # batch + spatial + time + channels

        if not torch.compiler.is_compiling():
            if x.ndim != expected_ndim:
                spatial_doc = "H, W" if dim == 2 else "X, Y, Z"
                raise ValueError(
                    f"Packed-input mode (trunked) expects {expected_ndim}D "
                    f"input (B, {spatial_doc}, T, C), got {x.ndim}D tensor "
                    f"with shape {tuple(x.shape)}."
                )
            if target_times is not None and target_times.ndim not in (1, 2):
                raise ValueError(
                    f"Expected target_times to be 1D (K,) or 2D (K, 1), "
                    f"got {target_times.ndim}D tensor with shape "
                    f"{tuple(target_times.shape)}."
                )

        spatial_shape = x.shape[1 : 1 + dim]

        # Right-pad the spatial axes (always to a multiple of 8, with the
        # configured floor) when auto-padding is enabled.
        if self.auto_pad and self.padding > 0:
            pads = compute_right_pad_to_multiple(
                spatial_shape, multiple=8, min_right_pad=self.padding
            )
            x = pad_spatial_right(x, spatial_ndim=dim, right_pad=pads, mode="replicate")
            if x_branch2 is not None and x_branch2.dim() > 2:
                x_branch2 = pad_spatial_right(
                    x_branch2,
                    spatial_ndim=dim,
                    right_pad=pads,
                    mode="replicate",
                )

        # Strip the time axis -- spatial branch sees only the T=0 slice.
        # Index = (slice(None),) * (1 + dim) + (0, slice(None))
        idx_strip_T = (slice(None),) * (1 + dim) + (0, slice(None))
        x_spatial = x[idx_strip_T]
        # Symmetric handling for branch2: when it's also a packed
        # (B, *spatial, T, C) tensor, strip its time axis the same way so
        # the second spatial branch sees a 4D (B, *spatial, C) tensor.
        # 2D (B, D_in) and already-stripped (B, *spatial, C) inputs are
        # left untouched so MLP/spatial-branch consumers stay valid.
        if x_branch2 is not None and x_branch2.ndim == expected_ndim:
            x_branch2 = x_branch2[idx_strip_T]

        # Build the trunk input.  All paths produce a (T_out, in_features) tensor.
        if target_times is not None:
            if self.trunk_input == "grid":
                t_vals = (
                    target_times
                    if target_times.dim() == 1
                    else target_times.squeeze(-1)
                )
                # Spatial coords of point [0, 0, ..., 0, t=0]: the
                # ``dim`` channels preceding the time channel.
                idx_spatial = (0,) * (2 + dim) + (slice(-(dim + 1), -1),)
                spatial = x[idx_spatial]
                spatial_exp = spatial.unsqueeze(0).expand(t_vals.shape[0], -1)
                x_trunk = torch.cat([spatial_exp, t_vals.unsqueeze(-1)], dim=-1)
            else:
                x_trunk = (
                    target_times
                    if target_times.dim() == 2
                    else target_times.unsqueeze(-1)
                )
        elif self.trunk_input == "grid":
            # Sweep all T values at the first spatial point; keep last
            # ``dim+1`` channels.
            idx_grid_over_time = (
                (0,) * (1 + dim) + (slice(None),) + (slice(-(dim + 1), None),)
            )
            x_trunk = x[idx_grid_over_time]
        else:
            # Time-only coords at the first spatial point.
            idx_time_over_time = (0,) * (1 + dim) + (slice(None), -1)
            x_trunk = x[idx_time_over_time].unsqueeze(-1)

        out = self._forward_core(x_spatial, x_trunk, x_branch2=x_branch2)
        # out: (B, *padded_spatial, T_out, out_channels)

        # Crop back to original spatial extent when auto-padding shifted
        # the padded dims out beyond ``spatial_shape``.  Trailing two
        # axes (T_out, out_channels) are preserved in full.
        if self.auto_pad and self.padding > 0:
            crop_idx = (
                (slice(None),)
                + tuple(slice(0, s) for s in spatial_shape)
                + (slice(None), slice(None))
            )
            out = out[crop_idx]
        return out

    def _forward_packed_trunkless(
        self,
        x: Float[Tensor, "..."],
        *,
        target_times: Float[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "..."]:
        """Trunkless packed-input forward (xFNO-style operator).

        ``x`` is channels-last ``(B, *spatial, C)``.  Steps:

        1. **Time-axis extension** (only when ``self.time_modes is not
           None`` and ``target_times`` is provided with length
           :math:`K \\neq T_{in}`): replicate-pad the last spatial axis
           to :math:`\\max(T_{in} + K, 2 \\cdot \\texttt{time\\_modes})`.
        2. **Spatial padding** (when ``self.auto_pad`` and
           ``self.padding > 0``): right-pad all spatial dims to a
           multiple of 8 with a floor of ``self.padding``.
        3. Run the trunkless core forward (branch + decoder).
        4. **Crop** the output back to the original spatial shape.  In
           the time-axis-extend case, the last spatial axis is sliced to
           :math:`[T_{in} : T_{in} + K]` (the predicted future steps);
           otherwise it's sliced to :math:`[:T_{in}]`.

        Output shape: ``(B, *spatial_or_K, out_channels)``.
        """
        dim = self.dimension
        expected_ndim = dim + 2  # batch + spatial + channels

        if not torch.compiler.is_compiling():
            if x.ndim != expected_ndim:
                spatial_doc = "H, W" if dim == 2 else "X, Y, Z"
                raise ValueError(
                    f"Packed-input mode (trunkless) expects {expected_ndim}D "
                    f"input (B, {spatial_doc}, C), got {x.ndim}D tensor "
                    f"with shape {tuple(x.shape)}."
                )
            if target_times is not None:
                if self.time_modes is None:
                    raise ValueError(
                        "target_times provided but the model was constructed "
                        "without time_modes; nothing to extend.  Either pass "
                        "time_modes=N at construction (xFNO-style autoregressive "
                        "bundling) or omit target_times."
                    )
                if target_times.ndim not in (1, 2):
                    raise ValueError(
                        f"Expected target_times to be 1D (K,) or 2D (K, 1), "
                        f"got {target_times.ndim}D tensor with shape "
                        f"{tuple(target_times.shape)}."
                    )

        original_spatial = x.shape[1 : 1 + dim]

        # Time-axis extension (xFNO autoregressive bundling).  The last
        # spatial axis is the time axis by convention.  ``K`` is the
        # number of future steps to predict; when ``K == T_in`` (or
        # ``target_times`` is absent) no extension happens and the
        # output covers the original time axis.
        k_future: int | None = None
        if self.time_modes is not None and target_times is not None:
            k_candidate = target_times.shape[0]
            t_in = original_spatial[-1]
            if k_candidate != t_in:
                k_future = k_candidate
                desired_t = t_in + k_future
                min_t = max(desired_t, 2 * self.time_modes)
                extra = min_t - t_in
                time_pad = (0,) * (dim - 1) + (extra,)
                x = pad_spatial_right(
                    x, spatial_ndim=dim, right_pad=time_pad, mode="replicate"
                )

        # Spatial padding to a multiple of 8 (after any time extension).
        if self.auto_pad and self.padding > 0:
            current_spatial = x.shape[1 : 1 + dim]
            pads = compute_right_pad_to_multiple(
                current_spatial, multiple=8, min_right_pad=self.padding
            )
            x = pad_spatial_right(x, spatial_ndim=dim, right_pad=pads, mode="replicate")

        # Trunkless core forward.
        out = self._forward_core(x, None, x_branch2=None)
        # out: (B, *padded_spatial, out_channels)

        # Crop back.  When time-extending: the last spatial axis is
        # sliced to the K future steps; other spatial axes to original.
        if k_future is not None:
            t_in = original_spatial[-1]
            crop = (
                (slice(None),)
                + tuple(slice(0, s) for s in original_spatial[:-1])
                + (slice(t_in, t_in + k_future),)
                + (slice(None),)  # out_channels axis
            )
        else:
            crop = (
                (slice(None),)
                + tuple(slice(0, s) for s in original_spatial)
                + (slice(None),)  # out_channels axis
            )
        return out[crop]

    def _forward_core(
        self,
        x_branch1: Float[Tensor, "..."],
        x_time: Float[Tensor, "..."] | None,
        x_branch2: Float[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "..."]:
        """Raw branch + (optional) trunk + decoder forward.

        ``x_branch1`` is either 2D ``(B, D_in)`` (MLP branches) or
        ``(dimension + 2)``-D channels-last spatial input.  ``x_time`` is
        1D ``(T,)``, 2D ``(T, D_trunk)``, or ``None`` (trunkless
        operator).  Called by :meth:`forward` in core mode and internally
        by :meth:`_forward_packed`.

        Output shape:

        - Spatial branch + trunk: ``(B, *spatial, T, out_channels)``
        - Spatial branch + trunkless: ``(B, *spatial, out_channels)``
        - Spatial branch + ``temporal_projection``:
          ``(B, *spatial, output_window, out_channels)``
        - MLP branch + trunk: ``(B, T, out_channels)``
        - MLP branch + trunkless: ``(B, out_channels)``
        """
        spatial_ndim = self._spatial_branch_ndim

        if not torch.compiler.is_compiling():
            if x_branch1.ndim not in (2, spatial_ndim):
                spatial_shape_doc = (
                    "(B, H, W, C)" if self.dimension == 2 else "(B, X, Y, Z, C)"
                )
                raise ValueError(
                    f"Expected x_branch1 to be 2D (B, D_in) for MLP branches "
                    f"or {spatial_ndim}D {spatial_shape_doc} for spatial "
                    f"branches, got {x_branch1.ndim}D tensor with shape "
                    f"{tuple(x_branch1.shape)}"
                )
            if x_time is not None and x_time.ndim not in (1, 2):
                raise ValueError(
                    f"Expected x_time to be 1D (T,) or 2D (T, D), got "
                    f"{x_time.ndim}D tensor with shape {tuple(x_time.shape)}"
                )
            if self.has_branch2 and x_branch2 is None:
                raise ValueError(
                    "branch2 is configured but x_branch2 was not provided "
                    "to forward()."
                )

        b1_out = self.branch1(x_branch1)

        if self.has_branch2:
            if x_branch2 is None:
                raise ValueError("x_branch2 required for mionet/tno variants")
            b2_out = self.branch2(x_branch2)

        # ---- Trunkless path (xFNO-style operator) ----------------------
        if self.trunk is None:
            if b1_out.dim() == spatial_ndim:
                # Spatial branch: combine with optional branch2 directly.
                combined = b1_out
                if self.has_branch2:
                    combined = combined * b2_out
                return self._apply_decoder_trunkless(combined)
            # MLP branch: flat ``(B, width)`` -> ``(B, out_channels)``.
            combined = b1_out
            if self.has_branch2:
                combined = combined * b2_out
            return self.decoder(combined)

        # ---- Trunked path ----------------------------------------------
        if x_time.dim() == 1:
            x_time = x_time.unsqueeze(-1)
        trunk_out = self.trunk(x_time)

        if b1_out.dim() == spatial_ndim:  # Spatial branch path
            if self._temporal_projection:
                # Broadcast a single trunk value across every spatial point:
                # trunk_single: (1, width) -> (1, 1, ..., 1, width) with
                # ``dimension`` spatial singleton axes inserted at position 1.
                trunk_single = trunk_out[0:1]
                trunk_exp = trunk_single
                for _ in range(self.dimension):
                    trunk_exp = trunk_exp.unsqueeze(1)
                combined = b1_out * trunk_exp
                if self.has_branch2:
                    if b2_out.dim() == spatial_ndim:
                        combined = combined * b2_out
                    else:
                        b2_exp = b2_out
                        for _ in range(self.dimension):
                            b2_exp = b2_exp.unsqueeze(1)
                        combined = combined * b2_exp
                combined = self.decoder(combined)
                if self.temporal_head is None:
                    raise RuntimeError(
                        "decoder_type='temporal_projection' requires either "
                        "output_window to be provided at construction time, "
                        "or set_output_window(K) to be called before forward."
                    )
                # temporal_head: width -> (output_window * out_channels);
                # reshape splits the trailing dim into (output_window,
                # out_channels) so the final shape is
                # (B, *spatial, output_window, out_channels).
                head_out = self.temporal_head(combined)
                head_shape = head_out.shape
                return head_out.reshape(*head_shape[:-1], -1, self.out_channels)

            # Insert a time axis at position 1 in the branch output:
            # (B, *spatial, width) -> (B, 1, *spatial, width)
            b1_out = b1_out.unsqueeze(1)
            # Insert a batch axis at position 0 and ``dimension`` spatial
            # singleton axes at position 2, giving (1, T, *1..*1, width).
            trunk_out = trunk_out.unsqueeze(0)
            for _ in range(self.dimension):
                trunk_out = trunk_out.unsqueeze(2)

            if self.has_branch2:
                if b2_out.dim() == spatial_ndim:
                    b2_out = b2_out.unsqueeze(1)
                else:
                    b2_out = b2_out.unsqueeze(1)
                    for _ in range(self.dimension):
                        b2_out = b2_out.unsqueeze(2)
                combined = b1_out * b2_out * trunk_out
            else:
                combined = b1_out * trunk_out

            # ``combined`` is now (B, T, *spatial, width).
            if self.decoder_type == "mlp":
                # Decoder maps width -> out_channels:
                # (B, T, *spatial, out_channels).  Move T from position 1
                # to the second-to-last so result is
                # (B, *spatial, T, out_channels).
                return self.decoder(combined).permute(*self._mlp_decoder_permute)

            # ``conv`` decoder: needs channel-first (B*T, width, *spatial)
            # input, returns (B*T, out_channels, *spatial).
            shape = combined.shape
            batch_size, n_t = shape[0], shape[1]
            spatial_shape = shape[2:-1]
            ch = shape[-1]
            combined = combined.permute(*self._conv_decoder_in_permute).reshape(
                batch_size * n_t, ch, *spatial_shape
            )
            decoded = self.decoder(combined)
            # decoded: (B*T, out_channels, *spatial)
            decoded = decoded.reshape(
                batch_size, n_t, self.out_channels, *spatial_shape
            )
            # -> (B, *spatial, T, out_channels)
            return decoded.permute(*self._conv_decoder_out_permute)

        # MLP branch + trunk path (no spatial axes).
        b1_out = b1_out.unsqueeze(1)
        trunk_out = trunk_out.unsqueeze(0)
        if self.has_branch2:
            combined = b1_out * b2_out.unsqueeze(1) * trunk_out
        else:
            combined = b1_out * trunk_out
        return self.decoder(combined)

    def _apply_decoder_trunkless(
        self,
        branch_out: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        """Apply the decoder to a trunkless spatial branch output.

        ``branch_out`` is channels-last ``(B, *spatial, width)``; the
        returned tensor is channels-last ``(B, *spatial, out_channels)``.
        """
        if self.decoder_type == "mlp":
            # MLP decoder acts pointwise on the last axis; no permute needed.
            return self.decoder(branch_out)
        # ``conv`` decoder operates channels-first.
        cf = branch_out.permute(*self._trunkless_channel_first_permute)
        cf = self.decoder(cf)
        return cf.permute(*self._trunkless_channel_last_permute)


__all__ = [
    "DeepONet",
]
