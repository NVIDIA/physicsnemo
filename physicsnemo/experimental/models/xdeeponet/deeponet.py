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

The core :class:`DeepONet` (2D) and :class:`DeepONet3D` (3D) classes are
dimension-specific but share the same construction pattern: a primary branch
(``branch1``), an optional secondary branch (``branch2`` for MIONet/TNO),
a coordinate trunk, and a decoder.

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
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.xdeeponet.branches import (
    MLPBranch,
    SpatialBranch,
    SpatialBranch3D,
    TrunkNet,
)
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.nn import Conv2dFCLayer, Conv3dFCLayer, get_activation

# All xDeepONet variants supported by both 2D and 3D cores.  Defined once
# at module scope so the two classes share a single source of truth; each
# class still exposes it as the ``VALID_VARIANTS`` class attribute for a
# stable public API.
_VALID_VARIANTS = (
    "deeponet",
    "u_deeponet",
    "fourier_deeponet",
    "conv_deeponet",
    "hybrid_deeponet",
    "mionet",
    "fourier_mionet",
    "tno",
)

# Variants that require a secondary branch (branch2).  Used by the core
# DeepONet / DeepONet3D __init__ to validate branch2_config up-front so
# multi-branch variants cannot silently degrade to single-branch models.
_DUAL_BRANCH_VARIANTS = frozenset({"mionet", "fourier_mionet", "tno"})

# Supported decoder types.  Used by the core DeepONet / DeepONet3D
# __init__ to reject unknown decoder types at the API boundary instead
# of deferring to ``_build_decoder`` and raising cryptically from deep
# inside construction.
_VALID_DECODER_TYPES = frozenset({"mlp", "conv", "temporal_projection"})


@dataclass
class _DeepONetMetaData(ModelMetaData):
    """PhysicsNeMo model metadata for :class:`DeepONet`."""


@dataclass
class _DeepONet3DMetaData(ModelMetaData):
    """PhysicsNeMo model metadata for :class:`DeepONet3D`."""


# ---------------------------------------------------------------------------
# Branch config helpers
# ---------------------------------------------------------------------------


def _normalize_branch_config(config: dict) -> dict:
    """Normalize a branch config to the nested encoder/layers format.

    Supports two input formats:

    **New (nested)** format::

        {
            "encoder": {"type": "linear", "activation_fn": "tanh", ...},
            "layers":  {"num_fourier_layers": 1, "num_unet_layers": 1, ...},
            "internal_resolution": [H, W],
        }

    **Old (flat)** format (auto-converted for backward compatibility)::

        {
            "encoder": "spatial",        # or "mlp"
            "num_fourier_layers": 1,
            "num_unet_layers": 1,
            "activation_fn": "tanh",
            ...
        }

    Returns a dict in the new nested format.
    """
    if "encoder" not in config:
        return config

    enc = config["encoder"]
    if not isinstance(enc, str):
        return config

    enc_type_str = str(enc).lower()
    cfg = dict(config)
    cfg.pop("encoder")

    encoder_keys = {"hidden_width", "num_layers"}
    layer_keys = {
        "num_fourier_layers",
        "num_unet_layers",
        "num_conv_layers",
        "modes1",
        "modes2",
        "modes3",
        "kernel_size",
        "dropout",
    }

    activation = cfg.pop("activation_fn", "sin")
    internal_res = cfg.pop("internal_resolution", None)
    in_channels = cfg.pop("in_channels", None)
    # The legacy 'unet_impl' key is silently dropped: only the library UNet
    # (physicsnemo.models.unet.UNet) is supported in the experimental package.
    cfg.pop("unet_impl", None)

    encoder_dict = {
        "type": "mlp" if enc_type_str == "mlp" else "linear",
        "activation_fn": activation,
    }
    for k in encoder_keys:
        if k in cfg:
            encoder_dict[k] = cfg.pop(k)

    layers_dict = {"activation_fn": activation}
    for k in layer_keys:
        if k in cfg:
            layers_dict[k] = cfg.pop(k)

    result = {"encoder": encoder_dict, "layers": layers_dict}
    if internal_res is not None:
        result["internal_resolution"] = internal_res
    if in_channels is not None:
        result["in_channels"] = in_channels

    return result


class _SinActivation(nn.Module):
    """Module wrapper around :func:`torch.sin` for use inside ``nn.Sequential``.

    ``physicsnemo.nn.get_activation`` does not register ``"sin"`` in its
    activation table; branch modules in ``branches.py`` work around this
    by storing ``torch.sin`` as a bare callable and invoking it directly
    in ``forward``.  That pattern does not compose with ``nn.Sequential``
    (which requires ``nn.Module`` instances), so this thin wrapper is used
    whenever a sin activation needs to slot into a ``Sequential`` pipeline.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply elementwise sine."""
        return torch.sin(x)


def _build_conv_encoder(width: int, enc_config: dict) -> nn.Module:
    """Build a multi-layer pointwise encoder replacing the default LazyLinear lift.

    Operates in channels-last format ``(B, *spatial, C)``.  Each layer is a
    :class:`torch.nn.Linear` with activation — equivalent to a 1x1 convolution
    applied independently at every spatial point.
    """
    num_layers = enc_config.get("num_layers", 1)
    activation_fn = enc_config.get("activation_fn", "relu")

    # ``get_activation`` does not know about ``"sin"``; use the module
    # wrapper defined above when the user explicitly requests it, so
    # config parity with the branch encoders is preserved.
    if activation_fn.lower() == "sin":
        act = _SinActivation()
    else:
        act = get_activation(activation_fn)

    if num_layers <= 1:
        return nn.LazyLinear(width)

    hidden_width = enc_config.get("hidden_width", width // 2)
    layers_list = [nn.LazyLinear(hidden_width), act]
    for _ in range(num_layers - 2):
        layers_list.extend([nn.Linear(hidden_width, hidden_width), act])
    layers_list.append(nn.Linear(hidden_width, width))
    return nn.Sequential(*layers_list)


# ---------------------------------------------------------------------------
# 2D DeepONet
# ---------------------------------------------------------------------------


class DeepONet(Module):
    r"""2D xDeepONet core architecture for operator learning.

    Combines a primary spatial/MLP branch, an optional secondary branch
    (for MIONet/TNO variants), a coordinate trunk, and a decoder.  The
    branch outputs and trunk are combined via Hadamard product and then
    projected to the output by the decoder.

    Parameters
    ----------
    variant : str
        One of the eight supported variants (see :data:`VALID_VARIANTS`).
    width : int
        Latent width.
    branch1_config : dict, optional
        Primary branch configuration.  See module docstring for schema.
    branch2_config : dict, optional
        Secondary branch configuration, required for the ``"mionet"``,
        ``"fourier_mionet"``, and ``"tno"`` variants.
    trunk_config : dict, optional
        Trunk network configuration.
    decoder_type : str, optional
        One of ``"mlp"`` (queries the trunk at each target timestep and
        applies an MLP decoder), ``"conv"`` (uses a convolutional decoder),
        or ``"temporal_projection"`` (queries the trunk once and projects
        the combined latent to K timesteps via a learned linear head for
        fast autoregressive bundling).
    decoder_width : int, optional
        Decoder hidden width.
    decoder_layers : int, optional
        Decoder layer count.
    decoder_activation_fn : str, optional
        Activation function name for the decoder.
    output_window : int, optional
        Output window length K for the ``"temporal_projection"`` decoder.
        When supplied the temporal head is constructed at ``__init__``, which
        produces a deterministic ``state_dict`` and makes checkpoint
        round-tripping straightforward.  When omitted,
        :meth:`set_output_window` must be called before the first forward
        pass.

    Forward
    -------
    x_branch1 : torch.Tensor
        Primary input of shape :math:`(B, H, W, C)` for spatial branches or
        :math:`(B, D_{in})` for MLP branches.
    x_time : torch.Tensor
        Query coordinates of shape :math:`(T,)` or
        :math:`(T, D_{\text{trunk}})`.
    x_branch2 : torch.Tensor, optional
        Secondary branch input for MIONet/TNO variants.

    Outputs
    -------
    torch.Tensor
        Operator output of shape :math:`(B, H, W, T)` for spatial branches
        or :math:`(B, T)` for MLP branches.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.xdeeponet import DeepONet
    >>> model = DeepONet(
    ...     variant="u_deeponet",
    ...     width=64,
    ...     branch1_config={
    ...         "encoder": {"type": "linear"},
    ...         "layers": {"num_unet_layers": 1, "kernel_size": 3},
    ...     },
    ...     trunk_config={"hidden_width": 64, "num_layers": 4},
    ... )
    >>> x_branch = torch.randn(2, 32, 32, 5)   # (B, H, W, C)
    >>> x_time = torch.linspace(0, 1, 3).unsqueeze(-1)   # (T, 1)
    >>> out = model(x_branch, x_time)          # (2, 32, 32, 3)
    """

    VALID_VARIANTS = _VALID_VARIANTS

    def __init__(
        self,
        variant: str = "u_deeponet",
        width: int = 64,
        branch1_config: Dict[str, Any] = None,
        branch2_config: Dict[str, Any] = None,
        trunk_config: Dict[str, Any] = None,
        decoder_type: str = "mlp",
        decoder_width: int = 128,
        decoder_layers: int = 2,
        decoder_activation_fn: str = "relu",
        output_window: Optional[int] = None,
    ):
        super().__init__(meta=_DeepONetMetaData())

        self.variant = variant.lower()
        self.width = width
        self.decoder_type = decoder_type.lower()
        self.decoder_activation_fn = decoder_activation_fn

        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant: {variant}. Valid: {self.VALID_VARIANTS}"
            )

        if self.decoder_type not in _VALID_DECODER_TYPES:
            raise ValueError(
                f"Unknown decoder_type: {decoder_type!r}. Valid: "
                f"{sorted(_VALID_DECODER_TYPES)}."
            )

        if self.variant in _DUAL_BRANCH_VARIANTS and branch2_config is None:
            raise ValueError(
                f"variant='{self.variant}' requires branch2_config to be "
                f"provided.  Dual-branch variants: "
                f"{sorted(_DUAL_BRANCH_VARIANTS)}."
            )

        branch1_config = branch1_config or {}
        trunk_config = trunk_config or {}

        self.branch1 = self._build_branch(branch1_config, width)

        # Reject MLP-branch configurations paired with a decoder that
        # needs a spatial (4D / 5D) ``combined`` tensor.  The MLP-branch
        # forward path produces a 3D tensor of shape (B, T, width) and:
        #   * ``temporal_projection`` silently drops the temporal head
        #     (wrong shape, no error);
        #   * ``conv`` crashes inside the decoder's ``Conv2d`` /
        #     ``Conv3d`` with PyTorch's generic "Expected 3D or 4D
        #     input" message, with no hint that the real cause is a
        #     config mismatch.
        # Fail fast here instead.
        if isinstance(self.branch1, MLPBranch) and self.decoder_type in (
            "temporal_projection",
            "conv",
        ):
            raise ValueError(
                f"decoder_type={self.decoder_type!r} is not supported with "
                "MLP branches.  Use decoder_type='mlp', or configure a "
                "SpatialBranch for branch1 (set num_unet_layers, "
                "num_fourier_layers, or num_conv_layers > 0 in "
                "branch1_config)."
            )

        self.has_branch2 = branch2_config is not None
        if self.has_branch2:
            self.branch2 = self._build_branch(branch2_config, width)

            # Forward assumes branch2's output has the same rank as
            # branch1's.  Mixing an MLPBranch (2D output (B, width)) with
            # a SpatialBranch (4D / 5D output) would either broadcast
            # nonsensically or raise a cryptic dim-mismatch error in the
            # Hadamard product.  Reject the mixed configuration here.
            if isinstance(self.branch1, MLPBranch) and not isinstance(
                self.branch2, MLPBranch
            ):
                raise ValueError(
                    "When branch1 is an MLPBranch, branch2 must also be "
                    "an MLPBranch (i.e. produce a 2D (B, width) output). "
                    "Swap branch1 and branch2, or configure branch1 as "
                    "a SpatialBranch."
                )

        self.trunk = TrunkNet(
            in_features=trunk_config.get("in_features", 1),
            out_features=width,
            hidden_width=trunk_config.get("hidden_width", 128),
            num_layers=trunk_config.get("num_layers", 6),
            activation_fn=trunk_config.get("activation_fn", "sin"),
            output_activation=trunk_config.get("output_activation", True),
        )

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
            # If ``output_window`` is not provided the user must call
            # :meth:`set_output_window` before the first forward pass; this
            # path is kept for backwards compatibility but produces a
            # state_dict whose structure depends on when the method is called.
            if output_window is not None:
                if output_window < 1:
                    raise ValueError(
                        f"output_window must be a positive integer, got {output_window}"
                    )
                self.temporal_head = nn.Linear(self.width, output_window)
            else:
                self.temporal_head = None
        else:
            self._temporal_projection = False
            self.decoder = self._build_decoder(
                width,
                1,
                decoder_layers,
                decoder_width,
                self.decoder_type,
                decoder_activation_fn,
            )

    def set_output_window(self, K: int):
        """Create the temporal-projection head for K output timesteps.

        Only effective when ``decoder_type="temporal_projection"``.
        """
        if self._temporal_projection:
            device = next(self.parameters()).device
            self.temporal_head = nn.Linear(self.width, K).to(device)

    def _build_branch(self, config: dict, width: int) -> nn.Module:
        config = _normalize_branch_config(config)
        enc = config.get("encoder", {})
        layers = config.get("layers", {})

        enc_type = enc.get("type", "linear")
        enc_activation = enc.get("activation_fn", "sin")

        has_layers = (
            layers.get("num_fourier_layers", 0)
            + layers.get("num_unet_layers", 0)
            + layers.get("num_conv_layers", 0)
        ) > 0

        if enc_type == "mlp" and not has_layers:
            return MLPBranch(
                out_features=width,
                hidden_width=enc.get("hidden_width", 64),
                num_layers=enc.get("num_layers", 3),
                activation_fn=enc_activation,
            )

        layer_activation = layers.get("activation_fn", enc_activation)
        branch = SpatialBranch(
            in_channels=config.get("in_channels", 12),
            width=width,
            num_fourier_layers=layers.get("num_fourier_layers", 0),
            num_unet_layers=layers.get("num_unet_layers", 0),
            num_conv_layers=layers.get("num_conv_layers", 0),
            modes1=layers.get("modes1", 12),
            modes2=layers.get("modes2", 12),
            kernel_size=layers.get("kernel_size", 3),
            dropout=layers.get("dropout", 0.0),
            activation_fn=layer_activation,
            internal_resolution=config.get("internal_resolution", None),
        )
        if enc_type == "conv":
            branch.lift = _build_conv_encoder(width, enc)
        return branch

    def _build_decoder(
        self,
        width: int,
        out_channels: int,
        num_layers: int,
        hidden_width: int,
        decoder_type: str,
        activation_fn: str,
    ) -> nn.Module:
        if decoder_type == "mlp":
            if num_layers == 0:
                return nn.Linear(width, out_channels)
            return FullyConnected(
                width, hidden_width, out_channels, num_layers, activation_fn
            )

        elif decoder_type == "conv":
            if num_layers == 0:
                return Conv2dFCLayer(width, out_channels)

            layers = []
            in_ch = width
            for _ in range(num_layers):
                layers.extend(
                    [Conv2dFCLayer(in_ch, hidden_width), get_activation(activation_fn)]
                )
                in_ch = hidden_width
            layers.append(Conv2dFCLayer(hidden_width, out_channels))
            return nn.Sequential(*layers)

        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(
        self,
        x_branch1: Float[Tensor, "..."],
        x_time: Float[Tensor, "..."],
        x_branch2: Optional[Float[Tensor, "..."]] = None,
    ) -> Float[Tensor, "..."]:
        """Forward pass through the DeepONet.

        See class docstring for input/output shapes.  ``x_branch1`` accepts
        either 2D ``(B, D_in)`` (MLP branches) or 4D ``(B, H, W, C)``
        (spatial branches); ``x_time`` accepts 1D ``(T,)`` or 2D
        ``(T, D_trunk)``, so the jaxtyping annotation is the unconstrained
        ``"..."`` shape.  Strict shape validation is performed at the top
        of this method under a :func:`torch.compiler.is_compiling` guard.
        """
        if not torch.compiler.is_compiling():
            if x_branch1.ndim not in (2, 4):
                raise ValueError(
                    f"Expected x_branch1 to be 2D (B, D_in) for MLP branches "
                    f"or 4D (B, H, W, C) for spatial branches, got "
                    f"{x_branch1.ndim}D tensor with shape "
                    f"{tuple(x_branch1.shape)}"
                )
            if x_time.ndim not in (1, 2):
                raise ValueError(
                    f"Expected x_time to be 1D (T,) or 2D (T, D), got "
                    f"{x_time.ndim}D tensor with shape {tuple(x_time.shape)}"
                )
            if self.has_branch2 and x_branch2 is None:
                raise ValueError(
                    f"variant='{self.variant}' requires x_branch2 but got None"
                )

        if x_time.dim() == 1:
            x_time = x_time.unsqueeze(-1)

        b1_out = self.branch1(x_branch1)

        if self.has_branch2:
            if x_branch2 is None:
                raise ValueError("x_branch2 required for mionet/tno variants")
            b2_out = self.branch2(x_branch2)

        trunk_out = self.trunk(x_time)

        if b1_out.dim() == 4:  # Spatial branch
            if self._temporal_projection:
                trunk_single = trunk_out[0:1]
                trunk_exp = trunk_single.unsqueeze(1).unsqueeze(2)
                combined = b1_out * trunk_exp
                if self.has_branch2:
                    if b2_out.dim() == 4:
                        combined = combined * b2_out
                    else:
                        combined = combined * b2_out.unsqueeze(1).unsqueeze(2)
                combined = self.decoder(combined)
                if self.temporal_head is None:
                    raise RuntimeError(
                        "decoder_type='temporal_projection' requires either "
                        "output_window to be provided at construction time, "
                        "or set_output_window(K) to be called before forward."
                    )
                combined = self.temporal_head(combined)
                return combined

            b1_out = b1_out.unsqueeze(1)
            trunk_out = trunk_out.unsqueeze(0).unsqueeze(2).unsqueeze(3)

            if self.has_branch2:
                if b2_out.dim() == 4:
                    b2_out = b2_out.unsqueeze(1)
                else:
                    b2_out = b2_out.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                combined = b1_out * b2_out * trunk_out
            else:
                combined = b1_out * trunk_out

            if self.decoder_type == "mlp":
                return self.decoder(combined).squeeze(-1).permute(0, 2, 3, 1)

            B, T, H, W, C = combined.shape
            combined = combined.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
            return self.decoder(combined).reshape(B, T, H, W).permute(0, 2, 3, 1)

        else:  # MLP branch
            b1_out = b1_out.unsqueeze(1)
            trunk_out = trunk_out.unsqueeze(0)

            if self.has_branch2:
                combined = b1_out * b2_out.unsqueeze(1) * trunk_out
            else:
                combined = b1_out * trunk_out

            return self.decoder(combined).squeeze(-1)

    def count_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 3D DeepONet
# ---------------------------------------------------------------------------


class DeepONet3D(Module):
    r"""3D xDeepONet core architecture for volumetric operator learning.

    See :class:`DeepONet` for parameter semantics.  The 3D variant operates
    on volumetric inputs and uses :class:`SpatialBranch3D` for spatial
    branches.

    Parameters
    ----------
    variant : str
        One of the eight supported variants (see :data:`VALID_VARIANTS`).

    Forward
    -------
    x_branch1 : torch.Tensor
        Primary input of shape :math:`(B, X, Y, Z, C)` for spatial branches
        or :math:`(B, D_{in})` for MLP branches.
    x_time : torch.Tensor
        Query coordinates of shape :math:`(T,)` or
        :math:`(T, D_{\text{trunk}})`.
    x_branch2 : torch.Tensor, optional
        Secondary branch input for MIONet/TNO variants.

    Outputs
    -------
    torch.Tensor
        Operator output of shape :math:`(B, X, Y, Z, T)` for spatial
        branches or :math:`(B, T)` for MLP branches.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.xdeeponet import DeepONet3D
    >>> model = DeepONet3D(
    ...     variant="u_deeponet",
    ...     width=64,
    ...     branch1_config={
    ...         "encoder": {"type": "linear"},
    ...         "layers": {"num_unet_layers": 1, "kernel_size": 3},
    ...     },
    ...     trunk_config={"hidden_width": 64, "num_layers": 4},
    ... )
    >>> x_branch = torch.randn(1, 16, 16, 16, 5)   # (B, X, Y, Z, C)
    >>> x_time = torch.linspace(0, 1, 2).unsqueeze(-1)
    >>> out = model(x_branch, x_time)          # (1, 16, 16, 16, 2)
    """

    VALID_VARIANTS = _VALID_VARIANTS

    def __init__(
        self,
        variant: str = "u_deeponet",
        width: int = 64,
        branch1_config: Dict[str, Any] = None,
        branch2_config: Dict[str, Any] = None,
        trunk_config: Dict[str, Any] = None,
        decoder_type: str = "mlp",
        decoder_width: int = 128,
        decoder_layers: int = 2,
        decoder_activation_fn: str = "relu",
        output_window: Optional[int] = None,
    ):
        super().__init__(meta=_DeepONet3DMetaData())

        self.variant = variant.lower()
        self.width = width
        self.decoder_type = decoder_type.lower()
        self.decoder_activation_fn = decoder_activation_fn

        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant: {variant}. Valid: {self.VALID_VARIANTS}"
            )

        if self.decoder_type not in _VALID_DECODER_TYPES:
            raise ValueError(
                f"Unknown decoder_type: {decoder_type!r}. Valid: "
                f"{sorted(_VALID_DECODER_TYPES)}."
            )

        if self.variant in _DUAL_BRANCH_VARIANTS and branch2_config is None:
            raise ValueError(
                f"variant='{self.variant}' requires branch2_config to be "
                f"provided.  Dual-branch variants: "
                f"{sorted(_DUAL_BRANCH_VARIANTS)}."
            )

        branch1_config = branch1_config or {}
        trunk_config = trunk_config or {}

        self.branch1 = self._build_branch(branch1_config, width)

        # Reject MLP-branch configurations paired with a decoder that
        # needs a spatial (4D / 5D) ``combined`` tensor.  The MLP-branch
        # forward path produces a 3D tensor of shape (B, T, width) and:
        #   * ``temporal_projection`` silently drops the temporal head
        #     (wrong shape, no error);
        #   * ``conv`` crashes inside the decoder's ``Conv2d`` /
        #     ``Conv3d`` with PyTorch's generic "Expected 3D or 4D
        #     input" message, with no hint that the real cause is a
        #     config mismatch.
        # Fail fast here instead.
        if isinstance(self.branch1, MLPBranch) and self.decoder_type in (
            "temporal_projection",
            "conv",
        ):
            raise ValueError(
                f"decoder_type={self.decoder_type!r} is not supported with "
                "MLP branches.  Use decoder_type='mlp', or configure a "
                "SpatialBranch for branch1 (set num_unet_layers, "
                "num_fourier_layers, or num_conv_layers > 0 in "
                "branch1_config)."
            )

        self.has_branch2 = branch2_config is not None
        if self.has_branch2:
            self.branch2 = self._build_branch(branch2_config, width)

            # Forward assumes branch2's output has the same rank as
            # branch1's.  Mixing an MLPBranch (2D output (B, width)) with
            # a SpatialBranch (4D / 5D output) would either broadcast
            # nonsensically or raise a cryptic dim-mismatch error in the
            # Hadamard product.  Reject the mixed configuration here.
            if isinstance(self.branch1, MLPBranch) and not isinstance(
                self.branch2, MLPBranch
            ):
                raise ValueError(
                    "When branch1 is an MLPBranch, branch2 must also be "
                    "an MLPBranch (i.e. produce a 2D (B, width) output). "
                    "Swap branch1 and branch2, or configure branch1 as "
                    "a SpatialBranch."
                )

        self.trunk = TrunkNet(
            in_features=trunk_config.get("in_features", 1),
            out_features=width,
            hidden_width=trunk_config.get("hidden_width", 128),
            num_layers=trunk_config.get("num_layers", 6),
            activation_fn=trunk_config.get("activation_fn", "sin"),
            output_activation=trunk_config.get("output_activation", True),
        )

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
            # If ``output_window`` is not provided the user must call
            # :meth:`set_output_window` before the first forward pass; this
            # path is kept for backwards compatibility but produces a
            # state_dict whose structure depends on when the method is called.
            if output_window is not None:
                if output_window < 1:
                    raise ValueError(
                        f"output_window must be a positive integer, got {output_window}"
                    )
                self.temporal_head = nn.Linear(self.width, output_window)
            else:
                self.temporal_head = None
        else:
            self._temporal_projection = False
            self.decoder = self._build_decoder(
                width,
                1,
                decoder_layers,
                decoder_width,
                self.decoder_type,
                decoder_activation_fn,
            )

    def set_output_window(self, K: int):
        """Create the temporal-projection head for K output timesteps.

        Only effective when ``decoder_type="temporal_projection"``.
        """
        if self._temporal_projection:
            device = next(self.parameters()).device
            self.temporal_head = nn.Linear(self.width, K).to(device)

    def _build_branch(self, config: dict, width: int) -> nn.Module:
        config = _normalize_branch_config(config)
        enc = config.get("encoder", {})
        layers = config.get("layers", {})

        enc_type = enc.get("type", "linear")
        enc_activation = enc.get("activation_fn", "sin")

        has_layers = (
            layers.get("num_fourier_layers", 0)
            + layers.get("num_unet_layers", 0)
            + layers.get("num_conv_layers", 0)
        ) > 0

        if enc_type == "mlp" and not has_layers:
            return MLPBranch(
                out_features=width,
                hidden_width=enc.get("hidden_width", 64),
                num_layers=enc.get("num_layers", 3),
                activation_fn=enc_activation,
            )

        layer_activation = layers.get("activation_fn", enc_activation)
        branch = SpatialBranch3D(
            in_channels=config.get("in_channels", 11),
            width=width,
            num_fourier_layers=layers.get("num_fourier_layers", 0),
            num_unet_layers=layers.get("num_unet_layers", 0),
            num_conv_layers=layers.get("num_conv_layers", 0),
            modes1=layers.get("modes1", 10),
            modes2=layers.get("modes2", 10),
            modes3=layers.get("modes3", 8),
            kernel_size=layers.get("kernel_size", 3),
            dropout=layers.get("dropout", 0.0),
            activation_fn=layer_activation,
            internal_resolution=config.get("internal_resolution", None),
        )
        if enc_type == "conv":
            branch.lift = _build_conv_encoder(width, enc)
        return branch

    def _build_decoder(
        self,
        width: int,
        out_channels: int,
        num_layers: int,
        hidden_width: int,
        decoder_type: str,
        activation_fn: str,
    ) -> nn.Module:
        if decoder_type == "mlp":
            if num_layers == 0:
                return nn.Linear(width, out_channels)
            return FullyConnected(
                width, hidden_width, out_channels, num_layers, activation_fn
            )

        elif decoder_type == "conv":
            if num_layers == 0:
                return Conv3dFCLayer(width, out_channels)

            layers = []
            in_ch = width
            for _ in range(num_layers):
                layers.extend(
                    [Conv3dFCLayer(in_ch, hidden_width), get_activation(activation_fn)]
                )
                in_ch = hidden_width
            layers.append(Conv3dFCLayer(hidden_width, out_channels))
            return nn.Sequential(*layers)

        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(
        self,
        x_branch1: Float[Tensor, "..."],
        x_time: Float[Tensor, "..."],
        x_branch2: Optional[Float[Tensor, "..."]] = None,
    ) -> Float[Tensor, "..."]:
        """Forward pass through the 3D DeepONet.

        See class docstring for input/output shapes.  ``x_branch1`` accepts
        either 2D ``(B, D_in)`` (MLP branches) or 5D ``(B, X, Y, Z, C)``
        (spatial branches); ``x_time`` accepts 1D ``(T,)`` or 2D
        ``(T, D_trunk)``.  Strict shape validation is performed at the top
        of this method under a :func:`torch.compiler.is_compiling` guard.
        """
        if not torch.compiler.is_compiling():
            if x_branch1.ndim not in (2, 5):
                raise ValueError(
                    f"Expected x_branch1 to be 2D (B, D_in) for MLP branches "
                    f"or 5D (B, X, Y, Z, C) for spatial branches, got "
                    f"{x_branch1.ndim}D tensor with shape "
                    f"{tuple(x_branch1.shape)}"
                )
            if x_time.ndim not in (1, 2):
                raise ValueError(
                    f"Expected x_time to be 1D (T,) or 2D (T, D), got "
                    f"{x_time.ndim}D tensor with shape {tuple(x_time.shape)}"
                )
            if self.has_branch2 and x_branch2 is None:
                raise ValueError(
                    f"variant='{self.variant}' requires x_branch2 but got None"
                )

        if x_time.dim() == 1:
            x_time = x_time.unsqueeze(-1)

        b1_out = self.branch1(x_branch1)

        if self.has_branch2:
            if x_branch2 is None:
                raise ValueError("x_branch2 required for mionet/tno variants")
            b2_out = self.branch2(x_branch2)

        trunk_out = self.trunk(x_time)

        if b1_out.dim() == 5:  # Spatial branch
            if self._temporal_projection:
                trunk_single = trunk_out[0:1]
                trunk_exp = trunk_single.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                combined = b1_out * trunk_exp
                if self.has_branch2:
                    if b2_out.dim() == 5:
                        combined = combined * b2_out
                    else:
                        combined = combined * b2_out.unsqueeze(1).unsqueeze(
                            2
                        ).unsqueeze(3)
                combined = self.decoder(combined)
                if self.temporal_head is None:
                    raise RuntimeError(
                        "decoder_type='temporal_projection' requires either "
                        "output_window to be provided at construction time, "
                        "or set_output_window(K) to be called before forward."
                    )
                combined = self.temporal_head(combined)
                return combined

            b1_out = b1_out.unsqueeze(1)
            trunk_out = trunk_out.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

            if self.has_branch2:
                if b2_out.dim() == 5:
                    b2_out = b2_out.unsqueeze(1)
                else:
                    b2_out = b2_out.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                combined = b1_out * b2_out * trunk_out
            else:
                combined = b1_out * trunk_out

            if self.decoder_type == "mlp":
                return self.decoder(combined).squeeze(-1).permute(0, 2, 3, 4, 1)

            B, T, X, Y, Z, C = combined.shape
            combined = combined.permute(0, 1, 5, 2, 3, 4).reshape(B * T, C, X, Y, Z)
            return self.decoder(combined).reshape(B, T, X, Y, Z).permute(0, 2, 3, 4, 1)

        else:  # MLP branch
            b1_out = b1_out.unsqueeze(1)
            trunk_out = trunk_out.unsqueeze(0)

            if self.has_branch2:
                combined = b1_out * b2_out.unsqueeze(1) * trunk_out
            else:
                combined = b1_out * trunk_out

            return self.decoder(combined).squeeze(-1)

    def count_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = [
    "DeepONet",
    "DeepONet3D",
]
