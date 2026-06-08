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

"""xFNO 4D operator (3D space + time) for the xDeepONet family.

This module adds the 4-dimensional xFNO operator alongside
:class:`~physicsnemo.experimental.models.xdeeponet.DeepONet`:

- :class:`FNO4D` — a pure-FNO architecture for 4-dimensional input domains
  (three spatial dimensions plus time).  Unlike the 3D operators (which are
  expressed as :class:`~physicsnemo.experimental.models.xdeeponet.DeepONet`
  with a :class:`~physicsnemo.experimental.models.xdeeponet.SpatialBranch`
  and can use U-Net / Conv skip-branches), no U-Net or Conv skip-branches
  are available in 4D — PyTorch has no native ``nn.Conv4d``, so the 4D
  variant operates exclusively on dimension-agnostic primitives:
  :class:`~physicsnemo.nn.SpectralConv4d` for the spectral operation and
  :class:`~physicsnemo.nn.ConvNdFCLayer` for the lifting / pointwise /
  decoder operations.
- :class:`FNO4DWrapper` — the recommended public entry point, adding
  automatic right-side padding of the spectral dimensions and optional
  autoregressive temporal extension.

Architecture
------------
1. **Optional coordinate channels.**  When ``coord_features=True`` (the
   default), four channels carrying normalized :math:`(x, y, z, t)`
   coordinates in :math:`[0, 1]` are concatenated to the input before the
   lift, so the operator has access to position information.
2. **Lifting** projects the per-point feature vector to the latent width
   using a stack of :class:`~physicsnemo.nn.ConvNdFCLayer` layers.
3. **Fourier block sequence** stacks ``num_fno_layers`` Fourier layers
   (``SpectralConv4d`` + 1x1x1x1 conv via ``ConvNdKernel1Layer``).  All
   layers except the last apply ``activation_fn`` after the spectral +
   pointwise sum; the final layer omits the activation so the decoder
   sees a raw residual.
4. **Decoder** projects the latent width back to ``out_channels`` with a
   :class:`~physicsnemo.nn.ConvNdFCLayer` stack.

Migration notes vs. the original NOF source
-------------------------------------------
This module is a behavior-preserving refactor of the ``FNO4D`` class from
``examples/reservoir_simulation/neural_operator_factory/models/xfno.py``
on the Neural Operator Factory (NOF) feature branch.  Every option is
preserved verbatim — same name, same default, same semantics.  Output is
**bit-identical** to the NOF source under matching seed / config / input.

References
----------
- Li, Z. et al. (2021). *Fourier Neural Operator for Parametric Partial
  Differential Equations.* ICLR.
- Wen, G., Li, Z., Azizzadenesheli, K., Anandkumar, A., & Benson, S. M.
  (2022). *U-FNO -- An enhanced Fourier neural operator-based deep-learning
  model for multiphase flow.* Advances in Water Resources, 163, 104180.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.xdeeponet._padding import (
    compute_right_pad_to_multiple_per_dim,
    pad_spatial_right,
)
from physicsnemo.nn import (
    ConvNdFCLayer,
    ConvNdKernel1Layer,
    SpectralConv4d,
    get_activation,
)


@dataclass
class _FNO4DMetaData(ModelMetaData):
    """PhysicsNeMo model metadata for :class:`FNO4D`."""


@dataclass
class _FNO4DWrapperMetaData(ModelMetaData):
    """PhysicsNeMo model metadata for :class:`FNO4DWrapper`."""


class FNO4D(Module):
    r"""4D Fourier Neural Operator for volumetric (3D space + time) problems.

    Operates on channel-last 6D tensors of shape
    :math:`(B, X, Y, Z, T, C_{in})` and produces output of shape
    :math:`(B, X, Y, Z, T, C_{out})`.  Padding to align spatial-temporal
    dimensions to the spectral block requirements is the wrapper's
    responsibility
    (see :class:`~physicsnemo.experimental.models.xdeeponet.FNO4DWrapper`).

    Pure FNO only: U-Net and Conv skip-branches are not available in 4D
    because PyTorch has no native ``nn.Conv4d``.  Only the
    dimension-agnostic :class:`~physicsnemo.nn.SpectralConv4d`,
    :class:`~physicsnemo.nn.ConvNdKernel1Layer`, and
    :class:`~physicsnemo.nn.ConvNdFCLayer` are used.

    Parameters
    ----------
    in_channels : int
        Number of input channels :math:`C_{in}`.
    out_channels : int
        Number of output channels :math:`C_{out}`.
    width : int, optional
        Latent channel dimension (default ``32``).
    modes1 : int, optional
        Number of Fourier modes along the first spatial axis ``X``
        (default ``8``).
    modes2 : int, optional
        Number of Fourier modes along ``Y`` (default ``8``).
    modes3 : int, optional
        Number of Fourier modes along ``Z`` (default ``6``).
    modes4 : int, optional
        Number of Fourier modes along ``T`` (default ``6``).
    num_fno_layers : int, optional
        Number of Fourier layers (default ``4``).
    activation_fn : str, optional
        Activation function name applied after each Fourier block (except
        the last) and inside the lifting / decoder stacks (default
        ``"gelu"``).
    lifting_layers : int, optional
        Number of layers in the lifting network (default ``2``).
        ``num_layers == 1`` collapses to a single
        :class:`~physicsnemo.nn.ConvNdFCLayer`.
    decoder_layers : int, optional
        Number of hidden layers in the decoder (default ``1``).  ``0``
        produces a single :class:`~physicsnemo.nn.ConvNdFCLayer`
        projection from ``width`` to ``out_channels``.
    decoder_width : int, optional
        Hidden width in the decoder (default ``128``).
    coord_features : bool, optional
        Whether to concatenate normalized :math:`(x, y, z, t)` coordinate
        channels to the input before the lift (default ``True``).  When
        ``True`` the lifting input has ``in_channels + 4`` channels.

    Forward
    -------
    x : torch.Tensor
        Input of shape :math:`(B, X, Y, Z, T, C_{in})`.

    Outputs
    -------
    torch.Tensor
        Output of shape :math:`(B, X, Y, Z, T, C_{out})`.

    Notes
    -----
    Output is bit-identical to the NOF source :class:`FNO4D` under matching
    seed / config / input.  No API changes relative to NOF.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.xdeeponet import FNO4D
    >>> model = FNO4D(
    ...     in_channels=2,
    ...     out_channels=1,
    ...     width=8,
    ...     modes1=2, modes2=2, modes3=2, modes4=2,
    ...     num_fno_layers=2,
    ...     lifting_layers=1,
    ...     decoder_layers=1,
    ...     decoder_width=16,
    ...     coord_features=True,
    ... )
    >>> x = torch.randn(1, 4, 4, 4, 4, 2)   # (B, X, Y, Z, T, C_in)
    >>> y = model(x)                        # (1, 4, 4, 4, 4, 1)
    >>> tuple(y.shape)
    (1, 4, 4, 4, 4, 1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 32,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 6,
        modes4: int = 6,
        num_fno_layers: int = 4,
        activation_fn: str = "gelu",
        lifting_layers: int = 2,
        decoder_layers: int = 1,
        decoder_width: int = 128,
        coord_features: bool = True,
    ):
        super().__init__(meta=_FNO4DMetaData())

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.num_fno_layers = num_fno_layers
        self.coord_features = coord_features
        self.activation_fn_name = activation_fn
        self.activation_fn = get_activation(activation_fn)

        if num_fno_layers <= 0:
            raise ValueError(f"num_fno_layers must be positive, got {num_fno_layers}.")

        # When coord_features is enabled the lift sees four extra channels
        # carrying normalized (x, y, z, t) positions.
        lift_in_channels = in_channels + 4 if coord_features else in_channels

        # Submodule construction order is preserved verbatim from the NOF
        # source: lift_network -> spectral_convs -> conv_1x1s -> decoder.
        # Keeping this order keeps state_dict keys and the RNG-init sequence
        # bit-identical to NOF.
        self.lift_network = self._build_lifting_network(
            lift_in_channels, width, lifting_layers
        )

        self.spectral_convs = nn.ModuleList()
        self.conv_1x1s = nn.ModuleList()
        for _ in range(num_fno_layers):
            self.spectral_convs.append(
                SpectralConv4d(self.width, self.width, modes1, modes2, modes3, modes4)
            )
            self.conv_1x1s.append(ConvNdKernel1Layer(self.width, self.width))

        self.decoder = self._build_decoder_network(
            width, out_channels, decoder_layers, decoder_width
        )

    # ---------------------------------------------------------------- build helpers

    def _build_lifting_network(
        self,
        in_channels: int,
        width: int,
        num_layers: int,
    ) -> nn.Module:
        r"""Construct the lifting network using
        :class:`~physicsnemo.nn.ConvNdFCLayer`.

        Parameters
        ----------
        in_channels : int
            Input feature dimension (already includes the four coord
            channels when ``coord_features=True``).
        width : int
            Target latent width.
        num_layers : int
            Number of layers in the lift; ``1`` collapses to a single
            :class:`~physicsnemo.nn.ConvNdFCLayer`.

        Returns
        -------
        torch.nn.Module
            The lifting submodule.
        """
        if num_layers == 1:
            return ConvNdFCLayer(in_channels, width)

        layers_list: list[nn.Module] = []
        hidden_width = width // 2
        layers_list.append(ConvNdFCLayer(in_channels, hidden_width))
        layers_list.append(self.activation_fn)
        for _ in range(num_layers - 2):
            layers_list.append(ConvNdFCLayer(hidden_width, hidden_width))
            layers_list.append(self.activation_fn)
        layers_list.append(ConvNdFCLayer(hidden_width, width))
        return nn.Sequential(*layers_list)

    def _build_decoder_network(
        self,
        width: int,
        out_channels: int,
        num_layers: int,
        hidden_width: int,
    ) -> nn.Module:
        r"""Construct the decoder network using
        :class:`~physicsnemo.nn.ConvNdFCLayer`.

        Parameters
        ----------
        width : int
            Latent input width from the FNO block sequence.
        out_channels : int
            Output channel count.
        num_layers : int
            Number of hidden layers.  ``0`` produces a single
            :class:`~physicsnemo.nn.ConvNdFCLayer` projection.
        hidden_width : int
            Hidden width for the decoder layers.

        Returns
        -------
        torch.nn.Module
            The decoder submodule.
        """
        if num_layers == 0:
            return ConvNdFCLayer(width, out_channels)

        layers_list: list[nn.Module] = []
        in_ch = width
        for _ in range(num_layers):
            layers_list.append(ConvNdFCLayer(in_ch, hidden_width))
            layers_list.append(self.activation_fn)
            in_ch = hidden_width
        layers_list.append(ConvNdFCLayer(hidden_width, out_channels))
        return nn.Sequential(*layers_list)

    def _create_meshgrid(self, shape: list, device: torch.device) -> Tensor:
        r"""Build a 4D coordinate meshgrid normalized to :math:`[0, 1]`.

        Parameters
        ----------
        shape : list[int]
            Shape of the channel-first input ``(B, C, X, Y, Z, T)``.
        device : torch.device
            Device on which to construct the coordinate tensors.

        Returns
        -------
        torch.Tensor
            Coordinate tensor of shape :math:`(B, 4, X, Y, Z, T)` carrying the
            normalized :math:`(x, y, z, t)` positions in :math:`[0, 1]`, ready
            to concatenate to the channel-first input before the lift.
        """
        bsize = shape[0]
        size_x, size_y, size_z, size_t = shape[2], shape[3], shape[4], shape[5]

        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_t = torch.linspace(0, 1, size_t, dtype=torch.float32, device=device)

        grid_x, grid_y, grid_z, grid_t = torch.meshgrid(
            grid_x, grid_y, grid_z, grid_t, indexing="ij"
        )

        grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(bsize, 1, -1, -1, -1, -1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(bsize, 1, -1, -1, -1, -1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).expand(bsize, 1, -1, -1, -1, -1)
        grid_t = grid_t.unsqueeze(0).unsqueeze(0).expand(bsize, 1, -1, -1, -1, -1)

        return torch.cat((grid_x, grid_y, grid_z, grid_t), dim=1)

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        x: Float[Tensor, "b x y z t c"],
    ) -> Float[Tensor, "b x_out y_out z_out t_out c_out"]:
        r"""Forward pass through the 4D FNO.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape :math:`(B, X, Y, Z, T, C_{in})`.

        Returns
        -------
        torch.Tensor
            Output of shape :math:`(B, X, Y, Z, T, C_{out})`.
        """
        if not torch.compiler.is_compiling():
            if x.ndim != 6:
                raise ValueError(
                    f"Expected x to be 6D (B, X, Y, Z, T, C_in), got "
                    f"{x.ndim}D tensor with shape {tuple(x.shape)}."
                )
            if x.shape[-1] != self.in_channels:
                raise ValueError(
                    f"Expected x last dim (C_in) == {self.in_channels}, got "
                    f"{x.shape[-1]} (full shape {tuple(x.shape)})."
                )

        # Convert to channel-first for the spectral/conv stack.
        x = x.permute(0, 5, 1, 2, 3, 4)  # (B, C_in, X, Y, Z, T)

        # Optionally concatenate normalized (x, y, z, t) coord channels.
        if self.coord_features:
            coord_feat = self._create_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)  # (B, C_in+4, X, Y, Z, T)

        # Lifting.
        x = self.lift_network(x)  # (B, width, X, Y, Z, T)

        # Fourier layers: spectral + 1x1x1x1.  Activation after every layer
        # except the last (matches the NOF source).
        for layer_idx in range(self.num_fno_layers):
            x1 = self.spectral_convs[layer_idx](x)
            x2 = self.conv_1x1s[layer_idx](x)
            if layer_idx < self.num_fno_layers - 1:
                x = self.activation_fn(x1 + x2)
            else:
                x = x1 + x2

        # Decoder.
        x = self.decoder(x)  # (B, C_out, X, Y, Z, T)

        # Convert back to channel-last.
        x = x.permute(0, 2, 3, 4, 5, 1)  # (B, X, Y, Z, T, C_out)
        return x


class FNO4DWrapper(Module):
    r"""4D xFNO wrapper with automatic spatial-temporal padding.

    Wraps :class:`FNO4D`.  Pads the spectral dimensions
    :math:`(X, Y, Z, T)` of the input to a multiple of ``8``
    (per-dimension minimum padding via ``padding``), runs the inner
    :class:`FNO4D`, and crops the output back to the original spatial
    shape.  Optionally extends the time axis to a forecast horizon ``K``
    via ``target_times``.

    The wrapper always returns a 5D tensor by squeezing the trailing
    channel dimension, matching the original NOF behavior.  This is only
    meaningful when ``out_channels == 1``; with ``out_channels > 1`` the
    squeeze is a no-op and the returned tensor is 6D (callers should
    construct :class:`FNO4D` directly in that case).

    Parameters
    ----------
    modes1 : int
        Number of Fourier modes along ``X``.
    modes2 : int
        Number of Fourier modes along ``Y``.
    modes3 : int
        Number of Fourier modes along ``Z``.
    modes4 : int
        Number of Fourier modes along ``T``.
    width : int
        Latent channel dimension.
    in_channels : int, optional
        Number of input channels (default ``11``, matching NOF).
    out_channels : int, optional
        Number of output channels (default ``1``).
    num_fno_layers : int, optional
        Number of Fourier layers (default ``4``).
    activation_fn : str, optional
        Activation function name (default ``"gelu"``).
    lifting_layers : int, optional
        Number of layers in the lifting network (default ``2``).
    decoder_layers : int, optional
        Number of hidden layers in the decoder (default ``1``).
    decoder_width : int, optional
        Hidden width in the decoder (default ``128``).
    coord_features : bool, optional
        Whether to concatenate normalized :math:`(x, y, z, t)` coord
        channels to the input before the lift (default ``True``).
    padding : int or Sequence[int], optional
        Minimum right-side padding for the spectral dimensions
        :math:`(X, Y, Z, T)`.  An ``int`` is broadcast to all four
        dimensions; a shorter sequence is right-padded with zeros and
        truncated to length 4.  Default ``8``.

    Forward
    -------
    x : torch.Tensor
        Input of shape :math:`(B, X, Y, Z, T_{in}, C_{in})`.
    target_times : torch.Tensor, optional
        Explicit target time coordinates of shape :math:`(K,)` or
        :math:`(K, 1)`.  When provided and :math:`K \neq T_{in}` the time
        axis is right-padded so the inner :class:`FNO4D` operates on at
        least :math:`T_{in} + K` (and at least :math:`2 \cdot modes_4`)
        timesteps; the output is cropped to the last :math:`K` timesteps.

    Outputs
    -------
    torch.Tensor
        Output of shape :math:`(B, X, Y, Z, T_{out})` when
        ``out_channels == 1`` (after the trailing-channel squeeze), or
        :math:`(B, X, Y, Z, T_{out}, C_{out})` otherwise.  :math:`T_{out}
        = K` when ``target_times`` is provided, else :math:`T_{in}`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.xdeeponet import FNO4DWrapper
    >>> model = FNO4DWrapper(
    ...     modes1=2, modes2=2, modes3=2, modes4=2,
    ...     width=8,
    ...     in_channels=2, out_channels=1,
    ...     num_fno_layers=2,
    ...     lifting_layers=1,
    ...     decoder_layers=1, decoder_width=16,
    ...     coord_features=True,
    ...     padding=0,
    ... )
    >>> x = torch.randn(1, 4, 4, 4, 4, 2)   # (B, X, Y, Z, T_in, C_in)
    >>> y = model(x)
    >>> tuple(y.shape)
    (1, 4, 4, 4, 4)
    """

    def __init__(
        self,
        modes1: int,
        modes2: int,
        modes3: int,
        modes4: int,
        width: int,
        in_channels: int = 11,
        out_channels: int = 1,
        num_fno_layers: int = 4,
        activation_fn: str = "gelu",
        lifting_layers: int = 2,
        decoder_layers: int = 1,
        decoder_width: int = 128,
        coord_features: bool = True,
        padding: Union[int, Sequence[int]] = 8,
    ):
        super().__init__(meta=_FNO4DWrapperMetaData())

        # Normalize padding to a length-4 list (X, Y, Z, T).  A shorter
        # sequence is right-padded with zeros; a longer one is truncated
        # (matches NOF behavior).
        if isinstance(padding, int):
            if padding < 0:
                raise ValueError(f"padding must be non-negative, got {padding}.")
            self.padding = [padding, padding, padding, padding]
        else:
            pad_list = list(padding)
            if any(p < 0 for p in pad_list):
                raise ValueError(
                    f"padding entries must be non-negative, got {pad_list}."
                )
            self.padding = (pad_list + [0] * (4 - len(pad_list)))[:4]

        # Cached so the time-extension branch in forward can size the
        # right-pad without re-inspecting the inner model.
        self.time_modes = modes4

        self.fno4d = FNO4D(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            modes4=modes4,
            num_fno_layers=num_fno_layers,
            activation_fn=activation_fn,
            lifting_layers=lifting_layers,
            decoder_layers=decoder_layers,
            decoder_width=decoder_width,
            coord_features=coord_features,
        )

    def forward(
        self,
        x: Float[Tensor, "batch x_dim y_dim z_dim time channels"],
        target_times: Optional[Float[Tensor, "..."]] = None,
    ) -> Float[Tensor, "..."]:
        r"""Forward pass with automatic padding and optional time extension.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape :math:`(B, X, Y, Z, T_{in}, C_{in})`.
        target_times : torch.Tensor, optional
            Explicit target time coordinates of shape :math:`(K,)` or
            :math:`(K, 1)`.  When provided and :math:`K \neq T_{in}` the
            time axis is extended via right-replicate padding.

        Returns
        -------
        torch.Tensor
            See the class docstring for shape semantics.
        """
        if not torch.compiler.is_compiling():
            if x.ndim != 6:
                raise ValueError(
                    f"Expected x to be 6D (B, X, Y, Z, T, C_in), got "
                    f"{x.ndim}D tensor with shape {tuple(x.shape)}."
                )

        x0, y0, z0, t_in = x.shape[1], x.shape[2], x.shape[3], x.shape[4]

        # Optional time-axis extension to the forecast horizon K.
        K = target_times.shape[0] if target_times is not None else None
        if K is not None and K != t_in:
            desired_t = t_in + K
            min_t = max(desired_t, 2 * self.time_modes)
            extra = min_t - t_in
            x = pad_spatial_right(
                x,
                spatial_ndim=4,
                right_pad=(0, 0, 0, extra),
                mode="replicate",
            )
            t_padded = x.shape[4]
        else:
            K = None
            t_padded = t_in

        # Right-pad (X, Y, Z, T) to a multiple of 8 with per-dim minima.
        pad_x, pad_y, pad_z, pad_t = compute_right_pad_to_multiple_per_dim(
            (x0, y0, z0, t_padded), multiple=8, min_right_pad=self.padding
        )
        x = pad_spatial_right(
            x,
            spatial_ndim=4,
            right_pad=(pad_x, pad_y, pad_z, pad_t),
            mode="replicate",
        )

        # Inner core operator.
        x = self.fno4d(x)

        # Crop back to the requested output shape.
        if K is not None:
            x = x[:, :x0, :y0, :z0, t_in : t_in + K, :]
        else:
            x = x[:, :x0, :y0, :z0, :t_in, :]

        # Squeeze the trailing channel dim when out_channels == 1
        # (preserves NOF behavior; no-op for out_channels > 1).
        return x.squeeze(-1)


__all__ = [
    "FNO4D",
    "FNO4DWrapper",
]
