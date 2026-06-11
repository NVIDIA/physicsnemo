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

"""4D xFNO wrapper (3D space + time) for the xDeepONet family.

Provides :class:`FNO4DWrapper`, built on the library model
:class:`physicsnemo.models.fno.FNO` (``dimension=4``, backed by
:class:`physicsnemo.nn.module.fno_layers.FNO4DEncoder`).  The wrapper adds
**autoregressive time-axis extension**: given an explicit forecast horizon via
``target_times``, the time axis is right-replicate-padded so the operator
predicts ``K`` future steps, and the output is cropped back to those ``K``
steps.  It also adopts the channel-last :math:`(B, X, Y, Z, T, C)`
input/output convention and squeezes a trailing unit channel.

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
from typing import List, Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.xdeeponet._padding import pad_spatial_right
from physicsnemo.models.fno import FNO


@dataclass
class _FNO4DWrapperMetaData(ModelMetaData):
    """PhysicsNeMo model metadata for :class:`FNO4DWrapper`."""


class FNO4DWrapper(Module):
    r"""4D FNO wrapper with autoregressive time-axis extension.

    Wraps the library :class:`physicsnemo.models.fno.FNO` (``dimension=4``)
    and adds autoregressive time-axis forecasting on top of it.  The inner
    ``FNO`` handles the 4D spectral encoding, optional coordinate features,
    domain padding, and decoding; this wrapper adds:

    1. a channel-last :math:`(B, X, Y, Z, T, C)` input/output convention
       (the inner ``FNO`` is channel-first :math:`(B, C, X, Y, Z, T)`);
    2. **time-axis extension** — when ``target_times`` of length :math:`K`
       is supplied, the time axis is right-replicate-padded so the operator
       runs on at least :math:`\max(T_{in} + K, 2\,m_T)` steps (where
       :math:`m_T` is the time-axis Fourier-mode count) and the output is
       cropped to the last :math:`K` steps;
    3. a trailing unit-channel squeeze.

    Parameters
    ----------
    in_channels : int
        Number of input channels :math:`C_{in}`.
    out_channels : int, optional
        Number of output channels :math:`C_{out}` (default ``1``).
    latent_channels : int, optional
        Latent channel dimension of the inner ``FNO`` (default ``32``).
    num_fno_layers : int, optional
        Number of spectral convolution layers (default ``4``).
    num_fno_modes : int or list[int], optional
        Number of Fourier modes kept per axis.  An ``int`` is broadcast to all
        four axes; a list must be ``[m_X, m_Y, m_Z, m_T]``.  The last entry is
        the time-axis mode count used to size the time extension.  Default
        ``16``.
    padding : int, optional
        Domain padding for the inner ``FNO`` spectral convolutions
        (default ``8``).
    padding_type : str, optional
        Padding type for the inner ``FNO`` (default ``"constant"``).
    activation_fn : str, optional
        Activation function name for the inner ``FNO`` (default ``"gelu"``).
    decoder_layers : int, optional
        Number of decoder layers in the inner ``FNO`` (default ``1``).
    decoder_layer_size : int, optional
        Decoder hidden width in the inner ``FNO`` (default ``32``).
    decoder_activation_fn : str, optional
        Decoder activation for the inner ``FNO`` (default ``"silu"``).
    coord_features : bool, optional
        Whether the inner ``FNO`` concatenates normalized coordinate channels
        (default ``True``).

    Forward
    -------
    x : torch.Tensor
        Input of shape :math:`(B, X, Y, Z, T_{in}, C_{in})`.
    target_times : torch.Tensor, optional
        Explicit target time coordinates of shape :math:`(K,)` or
        :math:`(K, 1)`.  When provided and :math:`K \neq T_{in}` the time
        axis is extended to a :math:`K`-step forecast horizon.

    Outputs
    -------
    torch.Tensor
        Output of shape :math:`(B, X, Y, Z, T_{out})` when
        ``out_channels == 1`` (after the trailing-channel squeeze), or
        :math:`(B, X, Y, Z, T_{out}, C_{out})` otherwise.  :math:`T_{out} = K`
        when ``target_times`` is provided, else :math:`T_{in}`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.xdeeponet import FNO4DWrapper
    >>> model = FNO4DWrapper(
    ...     in_channels=2,
    ...     out_channels=1,
    ...     latent_channels=8,
    ...     num_fno_layers=2,
    ...     num_fno_modes=2,
    ...     padding=0,
    ...     decoder_layers=1,
    ...     decoder_layer_size=16,
    ...     coord_features=True,
    ... )
    >>> x = torch.randn(1, 4, 4, 4, 4, 2)   # (B, X, Y, Z, T_in, C_in)
    >>> y = model(x)
    >>> tuple(y.shape)
    (1, 4, 4, 4, 4)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        latent_channels: int = 32,
        num_fno_layers: int = 4,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: int = 8,
        padding_type: str = "constant",
        activation_fn: str = "gelu",
        decoder_layers: int = 1,
        decoder_layer_size: int = 32,
        decoder_activation_fn: str = "silu",
        coord_features: bool = True,
    ):
        super().__init__(meta=_FNO4DWrapperMetaData())

        # Time-axis Fourier-mode count, used to size the autoregressive
        # extension so the spectral conv along time stays well-defined.
        if isinstance(num_fno_modes, (list, tuple)):
            self.time_modes = int(num_fno_modes[-1])
        else:
            self.time_modes = int(num_fno_modes)

        self.fno = FNO(
            in_channels=in_channels,
            out_channels=out_channels,
            dimension=4,
            latent_channels=latent_channels,
            num_fno_layers=num_fno_layers,
            num_fno_modes=num_fno_modes,
            padding=padding,
            padding_type=padding_type,
            activation_fn=activation_fn,
            decoder_layers=decoder_layers,
            decoder_layer_size=decoder_layer_size,
            decoder_activation_fn=decoder_activation_fn,
            coord_features=coord_features,
        )

    def forward(
        self,
        x: Float[Tensor, "batch x_dim y_dim z_dim time channels"],
        target_times: Optional[Float[Tensor, "..."]] = None,
    ) -> Float[Tensor, "..."]:
        r"""Forward pass with optional autoregressive time-axis extension.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape :math:`(B, X, Y, Z, T_{in}, C_{in})`.
        target_times : torch.Tensor, optional
            Explicit target time coordinates of shape :math:`(K,)` or
            :math:`(K, 1)`.  When provided and :math:`K \neq T_{in}` the time
            axis is extended via right-replicate padding to a ``K``-step
            forecast horizon.

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

        t_in = x.shape[4]

        # Optional time-axis extension to the forecast horizon K.  The inner
        # FNO handles its own domain padding for the spectral dims, so the
        # only padding the wrapper performs is this deliberate time extension.
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
        else:
            K = None

        # Channel-last (B, X, Y, Z, T, C) -> channel-first (B, C, X, Y, Z, T)
        # for the library FNO, then back again.
        x = x.permute(0, 5, 1, 2, 3, 4)
        y = self.fno(x)
        y = y.permute(0, 2, 3, 4, 5, 1)  # (B, X, Y, Z, T, C_out)

        # Crop the time axis: to the K future steps when extending, else to
        # the original input length.  (Spatial dims are preserved by FNO.)
        if K is not None:
            y = y[:, :, :, :, t_in : t_in + K, :]
        else:
            y = y[:, :, :, :, :t_in, :]

        # Squeeze the trailing channel dim when out_channels == 1
        # (preserves NOF behavior; no-op for out_channels > 1).
        return y.squeeze(-1)


__all__ = [
    "FNO4DWrapper",
]
