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
r"""Self-contained building blocks for the spherical transformer (S2Transformer).

Only the layers required by :class:`~physicsnemo.experimental.models.s2transformer.s2transformer.S2Transformer`
are provided here (stochastic-depth ``DropPath`` and a channel-first ``MLP``),
so the model has no dependency on any external ``common``/``makani`` layer
package at import or run time.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


@torch.compile(fullgraph=False)
def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    r"""Drop paths (stochastic depth) per sample, used in residual blocks.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape :math:`(B, ...)`.
    drop_prob : float, optional, default=0.0
        Probability of dropping the residual path for a given sample.
    training : bool, optional, default=False
        Whether the module is in training mode. When ``False`` the input is
        returned unchanged.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as ``x``.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    r"""Stochastic-depth (timm-style) ``DropPath`` module.

    Parameters
    ----------
    drop_prob : float, optional, default=None
        Probability of dropping the residual path for a given sample.
    """

    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    r"""Two-layer channel-first (NCHW) or ``traditional`` linear MLP.

    Parameters
    ----------
    in_features : int
        Number of input channels/features.
    hidden_features : int, optional, default=None
        Number of hidden channels/features. Defaults to ``in_features``.
    out_features : int, optional, default=None
        Number of output channels/features. Defaults to ``in_features``.
    act_layer : nn.Module, optional, default=nn.GELU
        Activation layer constructor.
    output_bias : bool, optional, default=True
        Whether the output projection uses a bias.
    input_format : str, optional, default="nchw"
        Either ``"nchw"`` (uses 1x1 convolutions) or ``"traditional"`` (uses
        linear layers).
    drop_rate : float, optional, default=0.0
        Dropout rate applied after the activation and after the output.
    drop_type : str, optional, default="iid"
        Either ``"iid"`` (elementwise ``Dropout``) or ``"features"``
        (channel-wise ``Dropout2d``, ``nchw`` only).
    checkpointing : bool, optional, default=False
        Whether to apply activation checkpointing to the forward pass.
    gain : float, optional, default=1.0
        Gain used to scale the variance of the output-projection weights.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        output_bias: bool = True,
        input_format: str = "nchw",
        drop_rate: float = 0.0,
        drop_type: str = "iid",
        checkpointing: bool = False,
        gain: float = 1.0,
    ):
        super().__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if input_format == "nchw":
            fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
        elif input_format == "traditional":
            fc1 = nn.Linear(in_features, hidden_features, bias=True)
        else:
            raise NotImplementedError(f"Unsupported input_format='{input_format}'")

        nn.init.normal_(fc1.weight, mean=0.0, std=math.sqrt(2.0 / in_features))
        nn.init.constant_(fc1.bias, 0.0)

        act = act_layer()

        if input_format == "traditional" and drop_type == "features":
            raise NotImplementedError(
                "input_format='traditional' is incompatible with drop_type='features'"
            )

        if input_format == "nchw":
            fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=output_bias)
        else:
            fc2 = nn.Linear(hidden_features, out_features, bias=output_bias)

        nn.init.normal_(fc2.weight, mean=0.0, std=math.sqrt(gain / hidden_features))
        if fc2.bias is not None:
            nn.init.constant_(fc2.bias, 0.0)

        if drop_rate > 0.0:
            if drop_type == "iid":
                drop = nn.Dropout(drop_rate)
            elif drop_type == "features":
                drop = nn.Dropout2d(drop_rate)
            else:
                raise NotImplementedError(f"Unsupported drop_type='{drop_type}'")
        else:
            drop = nn.Identity()

        self.fwd = nn.Sequential(fc1, act, drop, fc2, drop)

    @torch.compiler.disable(recursive=False)
    def _checkpoint_forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self.fwd, x, use_reentrant=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._checkpoint_forward(x) if self.checkpointing else self.fwd(x)
