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

import torch
from torch import Tensor, nn
from torch.nn import LayerNorm

from physicsnemo.nn.module.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from physicsnemo.nn.module.transformer_decoder import (
    DecoderOnlyLayer,
    TransformerDecoder,
)


class Sequence_Model(torch.nn.Module):
    r"""Decoder-only multi-head attention architecture.
    Parameters
    ----------
    input_dim : int
        Latent feature dimension :math:`D` for each time step
        (typically ``#pivotal_positions * output_decode_dim``).
    input_context_dim : int
        Number of physical context features.
    dist : Any
        Distributed manager or device wrapper containing the target ``device``.
    dropout_rate : float, optional, default=0.0
        Dropout value used in the attention decoder.
    num_layers_decoder : int, optional, default=3
        Number of decoder-only transformer layers.
    num_heads : int, optional, default=8
        Number of attention heads in the decoder.
    dim_feedforward_scale : int, optional, default=4
        Scale factor for the decoder MLP (FFN) hidden dimension, i.e.
        hidden size is ``dim_feedforward_scale * input_dim``.
    num_layers_context_encoder : int, optional, default=2
        Number of MLP layers for encoding the physical context.
    num_layers_input_encoder : int, optional, default=2
        Number of MLP layers for encoding input sequence features.
    num_layers_output_encoder : int, optional, default=2
        Number of MLP layers for encoding output sequence features.
    activation : str, optional, default="gelu"
        Activation function used in the decoder (``"relu"`` or ``"gelu"``).

    Forward
    -------
    x : torch.Tensor
        Input sequence tensor of shape :math:`(B, T, D)` with batch-first layout.
    context : torch.Tensor, optional
        Optional physical context. When provided it is encoded and concatenated
        along the temporal axis; it should broadcast or match to a shape compatible
        with :math:`(B, 1, D)` after encoding.

    Returns
    -------
    torch.Tensor
        Predicted sequence tensor of shape :math:`(B, T-1, D)`. The first token is
        treated as a prompt and excluded from the returned sequence.

    Notes
    -----
    Reference: `Predicting physics in mesh-reduced space with temporal attention <https://arxiv.org/pdf/2201.09113>`_
    """

    def __init__(
        self,
        input_dim: int,
        input_context_dim: int,
        dist,
        dropout_rate: float = 0.0000,
        num_layers_decoder: int = 3,
        num_heads: int = 8,
        dim_feedforward_scale: int = 4,
        num_layers_context_encoder: int = 2,
        num_layers_input_encoder: int = 2,
        num_layers_output_encoder: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.dist = dist
        decoder_layer = DecoderOnlyLayer(
            input_dim,
            num_heads,
            dim_feedforward_scale * input_dim,
            dropout_rate,
            activation,
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
            bias=True,
        )
        decoder_norm = LayerNorm(input_dim, eps=1e-5, bias=True)
        self.decoder = TransformerDecoder(
            decoder_layer, num_layers_decoder, decoder_norm
        )

        self.input_dim = input_dim
        self.input_context_dim = input_context_dim

        self.input_encoder = MeshGraphMLP(
            input_dim,
            output_dim=input_dim,
            hidden_dim=input_dim * 2,
            hidden_layers=num_layers_input_encoder,
            activation_fn=nn.ReLU(),
            norm_type="LayerNorm",
            recompute_activation=False,
        )
        self.output_encoder = MeshGraphMLP(
            input_dim,
            output_dim=input_dim,
            hidden_dim=input_dim * 2,
            hidden_layers=num_layers_output_encoder,
            activation_fn=nn.ReLU(),
            norm_type=None,
            recompute_activation=False,
        )
        self.context_encoder = MeshGraphMLP(
            input_context_dim,
            output_dim=input_dim,
            hidden_dim=input_dim * 2,
            hidden_layers=num_layers_context_encoder,
            activation_fn=nn.ReLU(),
            norm_type="LayerNorm",
            recompute_activation=False,
        )

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
    ) -> Tensor:
        if not torch.compiler.is_compiling():
            if x.ndim != 3 or x.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected tensor of shape (B, T, {self.input_dim}) but got tensor of shape {tuple(x.shape)}"
                )
            if context is not None:
                if context.ndim != 3 or context.shape[-1] != self.input_context_dim:
                    raise ValueError(
                        f"Expected context shape (B, T_c, {self.input_context_dim}) but got tensor of shape {tuple(context.shape)}"
                    )

        if context is not None:
            context = self.context_encoder(context)
            x = torch.cat([context, x], dim=1)

        x = self.input_encoder(x)
        tgt_mask = self.generate_square_subsequent_mask(
            x.size()[1], device=self.dist.device
        )
        output = self.decoder(x, tgt_mask=tgt_mask)
        output = self.output_encoder(output)
        return output[:, 1:]

    @torch.no_grad()
    def sample(self, z0: Tensor, step_size: int, context: Tensor | None = None):
        r"""Autoregressively sample a sequence starting from ``z0``.

        Parameters
        ----------
        z0 : torch.Tensor
            Initial prompt sequence of shape :math:`(B, T_0, D)`.
        step_size : int
            Number of future steps to generate.
        context : torch.Tensor, optional
            Optional physical context passed to :meth:`forward`.

        Returns
        -------
        torch.Tensor
            Concatenated sequence including the prompt and generated steps, of shape
            :math:`(B, T_0 + \mathrm{step\_size}, D)`.
        """

        z = z0
        for _ in range(step_size):
            prediction = self.forward(z, context)[:, -1].unsqueeze(1)
            z = torch.concat([z, prediction], dim=1)
        return z

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),
        dtype: torch.dtype = torch.get_default_dtype(),
    ) -> Tensor:
        r"""Generate a causal (future-masking) square attention mask.

        Parameters
        ----------
        sz : int
            Sequence length :math:`T`.
        device : torch.device, optional
            Target device for the mask tensor.
        dtype : torch.dtype, optional
            Data type for the mask tensor.

        Returns
        -------
        torch.Tensor
            Causal mask of shape :math:`(T, T)` with ``-inf`` above the main diagonal.
        """
        return torch.triu(
            torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
            diagonal=1,
        )
