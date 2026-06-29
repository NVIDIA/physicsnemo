# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Sequence, List

import torch as th
from omegaconf import DictConfig

try:
    from apex.normalization import FusedLayerNorm
    _APEX_AVAILABLE = True
except ImportError:
    _APEX_AVAILABLE = False


@th.compile
def _cln_affine(x_norm, gamma_raw, beta, scale_center, n_faces):
    """Fused affine transform: expand gamma/beta across faces and apply to normalized input."""
    C = gamma_raw.shape[-1]
    gamma = (scale_center + gamma_raw).unsqueeze(1).expand(-1, n_faces, -1).reshape(-1, 1, 1, C)
    beta = beta.unsqueeze(1).expand(-1, n_faces, -1).reshape(-1, 1, 1, C)
    return gamma * x_norm + beta


class ConditionalLayerNorm(th.nn.Module):
    def __init__(
        self,
        condition_shape: int,
        channel_depth: int,
        mlp_hidden_dims: List[int] = [128, 128],
        activation: th.nn.Module = None,
        eps: float = 1e-5,
        n_faces: int = 12,
        norm_op:str = "torch",
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
        eps : float
            Numerical stability constant.
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
        self.eps = eps
        self.condition_shape = condition_shape
        self.channel_depth = channel_depth
        self.hidden_dims = mlp_hidden_dims
        self.activation = activation if activation is not None else th.nn.Identity()
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
            self.norm = th.nn.LayerNorm(channel_depth, elementwise_affine=False)
        elif norm_op == "apex":
            if not _APEX_AVAILABLE:
                raise ImportError("Apex FusedLayerNorm requested but apex is not available, please install it from https://github.com/NVIDIA/apex")
            self.norm = FusedLayerNorm(channel_depth, elementwise_affine=False)

    def _make_mlp(self, in_dim: int, hidden_dims: List[int], out_dim: int, activation: th.nn.Module) -> th.nn.Sequential:

        layers = []
        for hdim in hidden_dims:
            layers.append(th.nn.Linear(in_dim, hdim))
            if activation:
                layers.append(activation)
            in_dim = hdim
        layers.append(th.nn.Linear(in_dim, out_dim))
        return th.nn.Sequential(*layers)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Backward compatibility: merge old separate gamma_mlp/beta_mlp into fused gamma_beta_mlp.

        Old MLPs had hidden_dims [h1, h2, ...] and output dim C.
        New fused MLP has hidden_dims [2*h1, 2*h2, ...] and output dim 2*C.

        For the first Linear (condition_shape → 2*h1), we vertically concatenate:
            new_weight = cat([gamma_weight, beta_weight], dim=0)

        For subsequent Linear layers (2*h_i → 2*h_{i+1} or 2*h_last → 2*C),
        we build a block-diagonal weight matrix:
            new_weight = [[gamma_weight, 0          ],
                          [0,            beta_weight]]

        Biases are always concatenated: cat([gamma_bias, beta_bias]).
        """
        gamma_prefix = prefix + "gamma_mlp."
        beta_prefix = prefix + "beta_mlp."
        fused_prefix = prefix + "gamma_beta_mlp."

        has_old_keys = any(k.startswith(gamma_prefix) for k in state_dict)

        if has_old_keys:
            # Collect all Linear layer indices from the old gamma MLP
            gamma_layer_indices = set()
            for k in state_dict:
                if k.startswith(gamma_prefix):
                    layer_key = k[len(gamma_prefix):]
                    parts = layer_key.split(".")
                    if len(parts) == 2 and parts[1] in ("weight", "bias"):
                        gamma_layer_indices.add(int(parts[0]))
            first_layer_idx = min(gamma_layer_indices)

            keys_to_remove = []
            keys_to_add = {}

            for k in list(state_dict.keys()):
                if k.startswith(gamma_prefix):
                    layer_key = k[len(gamma_prefix):]  # e.g. "0.weight"
                    parts = layer_key.split(".")
                    if len(parts) != 2 or parts[1] not in ("weight", "bias"):
                        continue
                    idx = int(parts[0])
                    param_type = parts[1]
                    fused_key = fused_prefix + layer_key
                    beta_key = beta_prefix + layer_key

                    if beta_key not in state_dict:
                        continue

                    gamma_val = state_dict[k]
                    beta_val = state_dict[beta_key]

                    if param_type == "bias":
                        # Biases are always concatenated
                        keys_to_add[fused_key] = th.cat([gamma_val, beta_val], dim=0)
                    elif idx == first_layer_idx:
                        # First layer: input dim is shared (condition_shape),
                        # just concatenate along output dim
                        keys_to_add[fused_key] = th.cat([gamma_val, beta_val], dim=0)
                    else:
                        # Hidden→hidden or hidden→output: block-diagonal
                        # gamma_val: (out_old, in_old), beta_val: (out_old, in_old)
                        # result: (2*out_old, 2*in_old)
                        out_old, in_old = gamma_val.shape
                        zeros = th.zeros(out_old, in_old, dtype=gamma_val.dtype, device=gamma_val.device)
                        keys_to_add[fused_key] = th.cat([
                            th.cat([gamma_val, zeros], dim=1),
                            th.cat([zeros, beta_val], dim=1),
                        ], dim=0)

                    keys_to_remove.append(k)
                    if beta_key not in keys_to_remove:
                        keys_to_remove.append(beta_key)

            for k in keys_to_remove:
                if k in state_dict:
                    del state_dict[k]
            state_dict.update(keys_to_add)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: th.Tensor, conditions: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        x : th.Tensor
            Input tensor of shape: (B, C, H, W)
        conditions : th.Tensor
            Conditioning tensor of shape (B*n_cond, cond_dim)

        Returns
        -------
        th.Tensor
            Normalized and conditioned tensor of shape: (B, C, H, W)
        """

        is_channels_last = x.is_contiguous(memory_format=th.channels_last)

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
            return result.permute(0, 3, 1, 2).contiguous(memory_format=th.channels_last)
        else:
            return result.permute(0, 3, 1, 2)
