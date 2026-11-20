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

r"""Encoder, decoder and merge utilities for the Cross_Unet PV-power model."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange

from physicsnemo.experimental.models.pv_power.attention import (
    AttentionLayer,
    ParallelTwoStageAttentionLayer,
    TwoStageAttentionLayer,
)


class SegMerging(nn.Module):
    r"""Token-wise segment merging that halves the patch axis (linear).

    Parameters
    ----------
    d_model : int
        Per-token hidden dimension :math:`D`.
    win_size : int
        Number of consecutive segments combined into one.
    """

    def __init__(self, d_model: int, win_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = nn.LayerNorm(win_size * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x``: (B, ts_d, seg_num, d_model).
        _, _, seg_num, _ = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)
        seg_to_merge = [x[:, :, i :: self.win_size, :] for i in range(self.win_size)]
        x = torch.cat(seg_to_merge, -1)
        x = self.norm(x)
        return self.linear_trans(x)


class CNNMerging(nn.Module):
    r"""Convolutional segment merging using a 1D conv with stride ``win_size``.

    Parameters
    ----------
    d_model : int
        Per-token hidden dimension.
    win_size : int
        Conv stride (segments merged per output token).
    """

    def __init__(self, d_model: int, win_size: int) -> None:
        super().__init__()
        self.win_size = win_size
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=win_size,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x``: (B, ts_d, seg_num, d_model) -> (B, ts_d, seg_num // win, d_model).
        batch_size, ts_d, seg_num, d_model = x.shape
        x = x.permute(0, 1, 3, 2).reshape(batch_size * ts_d, d_model, seg_num)
        x = self.conv(x)
        new_seg_num = x.shape[-1]
        return (
            x.reshape(batch_size, ts_d, d_model, new_seg_num)
            .permute(0, 1, 3, 2)
            .contiguous()
        )


class ScaleBlock(nn.Module):
    r"""One U-Net level: optional segment merging followed by attention layers.

    Parameters
    ----------
    win_size : int
        Patch-axis downsampling factor for this level. Use ``1`` to skip the
        merge layer entirely.
    n_vars : int
        Total channels :math:`C` flowing through the model.
    seg_num : int
        Number of patches at this scale (input to the attention layers).
    d_model : int
        Per-token hidden dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward inner width inside attention layers.
    dropout : float
        Dropout probability.
    depth : int, optional, default=1
        Number of attention layers stacked at this level.
    merge_kind : Literal["seg_merge", "cnn_merge"], optional, default="seg_merge"
        Selects the merge layer used to compress consecutive segments.
    attention_kind : Literal["two_stage", "parallel"], optional, default="two_stage"
        Selects between :class:`TwoStageAttentionLayer` (channel mixing via
        correlation) and :class:`ParallelTwoStageAttentionLayer` (channel
        mixing via self-attention).
    swap_corr_axis : bool, optional, default=False
        Forwarded to :class:`TwoStageAttentionLayer`.
    """

    def __init__(
        self,
        win_size: int,
        n_vars: int,
        seg_num: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        depth: int = 1,
        merge_kind: Literal["seg_merge", "cnn_merge"] = "seg_merge",
        attention_kind: Literal["two_stage", "parallel"] = "two_stage",
        swap_corr_axis: bool = False,
    ) -> None:
        super().__init__()
        if win_size > 1:
            if merge_kind == "seg_merge":
                self.merge_layer: nn.Module | None = SegMerging(d_model, win_size)
            elif merge_kind == "cnn_merge":
                self.merge_layer = CNNMerging(d_model, win_size)
            else:
                raise ValueError(
                    f"Unknown merge_kind {merge_kind!r}; expected 'seg_merge' or 'cnn_merge'."
                )
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList()
        for _ in range(depth):
            if attention_kind == "two_stage":
                self.encode_layers.append(
                    TwoStageAttentionLayer(
                        n_vars=n_vars,
                        d_model=d_model,
                        n_heads=n_heads,
                        d_ff=d_ff,
                        dropout=dropout,
                        swap_corr_axis=swap_corr_axis,
                    )
                )
            elif attention_kind == "parallel":
                self.encode_layers.append(
                    ParallelTwoStageAttentionLayer(
                        n_vars=n_vars,
                        d_model=d_model,
                        n_heads=n_heads,
                        d_ff=d_ff,
                        dropout=dropout,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown attention_kind {attention_kind!r}; "
                    "expected 'two_stage' or 'parallel'."
                )

    def forward(
        self, x: torch.Tensor, corr: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.merge_layer is not None:
            x = self.merge_layer(x)
        atten_new: list[torch.Tensor] = []
        for layer in self.encode_layers:
            x, atten_new = layer(x, corr)
        return x, atten_new


class BottleneckLayer(nn.Module):
    r"""Self-attention + feed-forward block applied at the encoder bottleneck.

    Parameters
    ----------
    d_model : int
        Per-token hidden dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward inner width.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x``: (B, ts_d, seg_num, d_model).
        batch_size, ts_d, seg_num, d_model = x.shape
        x_flat = x.reshape(batch_size * ts_d, seg_num, d_model)
        attn_output, _ = self.self_attn(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + self.dropout(attn_output))
        x_flat = self.norm2(x_flat + self.dropout(self.feed_forward(x_flat)))
        return x_flat.reshape(batch_size, ts_d, seg_num, d_model)


class Encoder(nn.Module):
    r"""Stack of :class:`ScaleBlock` levels followed by a bottleneck.

    Parameters
    ----------
    scale_blocks : list of ScaleBlock
        Encoder levels in coarse-to-fine order.
    d_model : int
        Per-token hidden dimension.
    n_heads : int
        Number of attention heads in the bottleneck layer.
    d_ff : int
        Feed-forward inner width in the bottleneck layer.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        scale_blocks: list[ScaleBlock],
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encode_blocks = nn.ModuleList(scale_blocks)
        self.bottleneck = BottleneckLayer(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
        )

    def forward(
        self, x: torch.Tensor, corr: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        # ``x``: (B, n_vars, seg_num, d_model).
        encode_x: list[torch.Tensor] = [x]
        atten_weights: list[list[torch.Tensor]] = []
        for block in self.encode_blocks:
            x, attns = block(x, corr)
            encode_x.append(x)
            atten_weights.append(attns)
        encode_x.append(self.bottleneck(x))
        return encode_x, atten_weights


class DecoderLayer(nn.Module):
    r"""One decoder level: self-attention, encoder cross-attention, and prediction head.

    Parameters
    ----------
    self_attention : nn.Module
        Self-attention layer that consumes the decoder state and channel
        correlation matrix (typically :class:`TwoStageAttentionLayer` or
        :class:`ParallelTwoStageAttentionLayer`).
    cross_attention : AttentionLayer
        Cross-attention layer used to attend to one encoder level.
    seg_len : int
        Length of one output segment in the time domain (used to size the
        per-segment linear prediction head).
    d_model : int
        Per-token hidden dimension.
    dropout : float, optional, default=0.1
        Dropout probability.
    """

    def __init__(
        self,
        self_attention: nn.Module,
        cross_attention: AttentionLayer,
        seg_len: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(
        self, x: torch.Tensor, cross: torch.Tensor, corr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ``x``: (B, ts_d, out_seg_num, d_model); ``cross``: same as ``x`` from
        # the encoder side (or the bottleneck output).
        batch = x.shape[0]
        x, _ = self.self_attention(x, corr)
        x = rearrange(x, "b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model")
        cross = rearrange(
            cross, "b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model"
        )
        cross_attn_out = self.cross_attention(x, cross, cross)
        x = self.norm1(x + self.dropout(cross_attn_out))
        x = self.norm2(x + self.mlp(x))
        dec_output = rearrange(
            x,
            "(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model",
            b=batch,
        )
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(
            layer_predict,
            "b out_d seg_num seg_len -> b (out_d seg_num) seg_len",
        )
        return dec_output, layer_predict


class Decoder(nn.Module):
    r"""Stack of :class:`DecoderLayer` blocks producing a multi-segment forecast.

    Parameters
    ----------
    layers : list of DecoderLayer
        Decoder levels in coarse-to-fine order. The first layer attends to
        the encoder bottleneck when ``use_bottleneck`` is True.
    use_bottleneck : bool, optional, default=True
        If True, the encoder's bottleneck output is consumed by the deepest
        decoder layer (and skip connections start from one level shallower).
    """

    def __init__(self, layers: list[DecoderLayer], use_bottleneck: bool = True) -> None:
        super().__init__()
        self.decode_layers = nn.ModuleList(layers)
        self.use_bottleneck = use_bottleneck

    def forward(
        self, x: torch.Tensor, cross_all: list[torch.Tensor], corr: torch.Tensor
    ) -> torch.Tensor:
        if self.use_bottleneck:
            bottleneck_output = cross_all[-1]
            cross = cross_all[:-1]
        else:
            bottleneck_output = None
            cross = cross_all

        ts_d = x.shape[1]
        layer_num = len(self.decode_layers)
        # Initialise ``final_predict`` from the first layer; ``Decoder`` is built
        # with at least one decoder layer (``e_layers + 1 >= 2``).
        final_predict = torch.zeros(0, dtype=x.dtype, device=x.device)
        for i, layer in enumerate(self.decode_layers):
            if self.use_bottleneck and i == 0:
                cross_in = bottleneck_output + x[:, :, : bottleneck_output.size(2), :]
            else:
                cross_in = cross[layer_num - i - 1] if self.use_bottleneck else cross[i]
            x, layer_predict = layer(x, cross_in, corr)
            final_predict = layer_predict if i == 0 else final_predict + layer_predict

        return rearrange(
            final_predict,
            "b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d",
            out_d=ts_d,
        )
