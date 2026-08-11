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

"""
Implementation of the Deep Learning Weather Prediction (DLWP) normalization on the HEALPix mesh.

This class contains the implementation of the Deep Learning Weather Prediction (DLWP) normalization on the HEALPix mesh.
"""

from typing import List

import torch

from physicsnemo.core.version_check import OptionalImport

# ``apex`` is an optional accelerator (not a core dependency). Import it lazily
# via ``OptionalImport``. If ``apex`` is missing, ``OptionalImport`` raises
# an ``ImportError`` with an install hint at that point.
apex_normalization = OptionalImport("apex.normalization")


@torch.compile
def _cln_affine(x_norm, gamma_raw, beta, scale_center, n_faces):
    """Fused affine transform: expand gamma/beta across faces and apply to normalized input."""
    C = gamma_raw.shape[-1]
    gamma = (
        (scale_center + gamma_raw)
        .unsqueeze(1)
        .expand(-1, n_faces, -1)
        .reshape(-1, 1, 1, C)
    )
    beta = beta.unsqueeze(1).expand(-1, n_faces, -1).reshape(-1, 1, 1, C)
    return gamma * x_norm + beta


class ConditionalLayerNorm(torch.nn.Module):
    """LayerNorm whose affine (gamma/beta) parameters are predicted from a conditional input field.

    Normalizes the input over the channel dimension with no learnable affine
    parameters, then predicts per-channel scale and shift from ``conditions``
    via a fused MLP and applies them, optionally recentered by
    ``scale_center``.
    """

    def __init__(
        self,
        condition_shape: int,
        channel_depth: int,
        mlp_hidden_dims: List[int] = [128, 128],
        activation: torch.nn.Module = None,
        n_faces: int = 12,
        norm_op: str = "torch",
        init_cln_to_zero: bool = False,
        scale_center: float = 0.0,
    ):
        """
        Conditional LayerNorm with MLP-based conditioning.

        Parameters
        ----------
        condition_shape : int
            Shape of the conditioning input.
        channel_depth : int
            Number of channels in the input tensor.
        mlp_hidden_dims : List[int]
            Hidden layer sizes for MLPs predicting gamma and beta.
        activation : DictConfig
            Activation function configuration for the MLPs.
        n_faces : int
            Number of faces in the Healpix grid, used for reshaping.
        norm_op : str
            "torch" for torch.nn.LayerNorm, "apex" for apex FusedLayerNorm.
        init_cln_to_zero : bool = False
            If True, initialize the last layer of the MLPs to zero.
            At the start of training, the noise will be ignored
        scale_center : float = 0.0
            Center of the scale parameter. Set to 1.0 and use `init_cln_to_zero=True`
            to make CLN behave like standard LayerNorm at initialization.
        """
        super().__init__()
        self.condition_shape = condition_shape
        self.channel_depth = channel_depth
        self.hidden_dims = mlp_hidden_dims
        self.activation = activation if activation is not None else torch.nn.Identity()
        self.gamma_beta_mlp = self._make_mlp(
            self.condition_shape,
            [2 * h for h in self.hidden_dims],
            2 * self.channel_depth,
            self.activation,
        )
        self.n_faces = n_faces
        self.scale_center = scale_center

        if init_cln_to_zero:
            self.gamma_beta_mlp[-1].weight.data.zero_()
            self.gamma_beta_mlp[-1].bias.data.zero_()

        if norm_op == "torch":
            self.norm = torch.nn.LayerNorm(channel_depth, elementwise_affine=False)
        elif norm_op == "apex":
            self.norm = apex_normalization.FusedLayerNorm(
                channel_depth, elementwise_affine=False
            )

    def _make_mlp(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        activation: torch.nn.Module,
    ) -> torch.nn.Sequential:
        """Helper function that creates the MLP for the conditional layer normalization.

        Parameters
        ----------
        in_dim: int
            The input dimension
        hidden_dims: List[int]
            The hidden dimensions
        out_dim: int
            The output dimension
        activation: torch.nn.Module
            The activation function

        Returns
        -------
        torch.nn.Sequential
            The MLP
        """
        layers = []
        for hdim in hidden_dims:
            layers.append(torch.nn.Linear(in_dim, hdim))
            if activation:
                layers.append(activation)
            in_dim = hdim
        layers.append(torch.nn.Linear(in_dim, out_dim))
        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape: (B, C, H, W)
        conditions : torch.Tensor
            Conditioning tensor of shape (B*n_cond, cond_dim)

        Returns
        -------
        torch.Tensor
            Normalized and conditioned tensor of shape: (B, C, H, W)
        """

        is_channels_last = x.is_contiguous(memory_format=torch.channels_last)

        # LayerNorm on last dim: permute to (B, H, W, C)
        x_nhwc = x.permute(0, 2, 3, 1)
        if not is_channels_last:
            x_nhwc = x_nhwc.contiguous()
        x_norm = self.norm(x_nhwc)

        # Fused gamma/beta MLP: single forward pass, then split
        gamma_beta = self.gamma_beta_mlp(conditions)  # (B*n_cond, 2*C)
        gamma_raw, beta = gamma_beta.chunk(2, dim=-1)  # each (B*n_cond, C)

        # Fused affine: expand across faces + scale_center + multiply + add
        result = _cln_affine(x_norm, gamma_raw, beta, self.scale_center, self.n_faces)

        # Return to NCHW logical layout, preserving channels_last memory format if input was
        if is_channels_last:
            return result.permute(0, 3, 1, 2).contiguous(
                memory_format=torch.channels_last
            )
        else:
            return result.permute(0, 3, 1, 2)
