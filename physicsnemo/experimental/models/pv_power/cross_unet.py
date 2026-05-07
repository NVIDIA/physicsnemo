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

r"""Cross_Unet photovoltaic-power forecasting model.

Adapted from the upstream open-source PV-power benchmark suite (Apache-2.0,
2023 Yunhao Zhang & Junchi Yan). The PhysicsNeMo port refactors the upstream
``Model(configs)`` API to explicit typed kwargs and drops dead modules while
preserving the upstream channel-correlation construction: correlations are
computed from historical weather channels, the full historical target window,
and the current primary target channel, then reduced to the model's
:math:`C \times C` channel-mixing grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from torch.func import vmap

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.pv_power.attention import (
    AttentionLayer,
    ParallelTwoStageAttentionLayer,
    TwoStageAttentionLayer,
)
from physicsnemo.experimental.models.pv_power.embedding import PatchEmbedding
from physicsnemo.experimental.models.pv_power.encoder_decoder import (
    Decoder,
    DecoderLayer,
    Encoder,
    ScaleBlock,
)


@dataclass
class CrossUnetMetaData(ModelMetaData):
    """Metadata for :class:`CrossUnet`."""

    # Optimization
    jit: bool = False  # vmap + dynamic ceil-padding break TorchScript trace
    cuda_graphs: bool = False  # dynamic per-call seg padding
    amp_cpu: bool = False
    amp_gpu: bool = True
    bf16: bool = True
    torch_fx: bool = False
    # Inference
    onnx: bool = False
    onnx_runtime: bool = False
    # Physics informed
    func_torch: bool = True  # uses torch.func.vmap internally
    auto_grad: bool = True


def _pearson_correlation(sample: Tensor) -> Tensor:
    r"""Per-sample Pearson correlation of the channel axis of ``sample``.

    Parameters
    ----------
    sample : torch.Tensor
        Single-sample slice of shape :math:`(L, C)`.

    Returns
    -------
    torch.Tensor
        Channel-correlation matrix of shape :math:`(C, C)`.
    """
    return torch.corrcoef(sample.T)


class CrossUnet(Module):
    r"""Cross_Unet: hierarchical patch-Transformer for PV-power forecasting.

    The model embeds a multi-channel time series as patches, applies a
    U-Net-style encoder/decoder of two-stage (time + channel) attention
    blocks, and conditions the attention on a Pearson-correlation matrix
    computed from the historical weather and target channels. It is designed
    for short-to-medium-horizon photovoltaic power forecasting at 15-minute
    cadence (the default ``seg_len=12`` corresponds to 3-hour patches), but
    works for any tabular multivariate forecasting problem with a single
    designated *target* channel.

    Channels are split into two semantic groups:

    * ``target_channels``: the target time series plus any history channels
      that should be reconstructed. The **last** channel of ``x_enc`` is the
      primary signal whose correlation with the other channels conditions
      the attention.
    * ``weather_channels``: exogenous weather forecast features that are
      concatenated to ``x_enc`` before the patch embedding (set to ``0`` to
      disable the weather branch entirely).

    Internally the network operates on ``total_channels = target_channels +
    weather_channels`` channels. The channel-correlation matrix is computed
    from one extra correlation input channel: the full historical target
    window plus the current primary target channel.

    Adapted from the upstream `PV-power Cross_Unet
    <https://github.com/Z-Yh1/PV-power>`_ (Apache-2.0).

    Parameters
    ----------
    target_channels : int
        Number of target/history channels :math:`C_{tgt}` (must be
        :math:`\geq 1`). The last channel of ``x_enc`` is the primary signal
        whose feature correlations condition the attention.
    weather_channels : int, optional, default=0
        Number of exogenous weather channels :math:`C_{wx}` concatenated to
        ``x_enc``. Set to ``0`` to disable the weather branch.
    seq_len : int, optional, default=96
        Input window length :math:`L`.
    pred_len : int, optional, default=16
        Forecast horizon :math:`H`.
    seg_len : int, optional, default=12
        Patch length along the time axis. Defaults to ``12`` (= 3 h at
        15-min cadence).
    e_layers : int, optional, default=3
        Number of encoder/decoder levels (the decoder has ``e_layers + 1``
        levels including the bottleneck-fed level).
    d_model : int, optional, default=128
        Per-token hidden dimension :math:`D`.
    n_heads : int, optional, default=4
        Number of attention heads.
    d_ff : int, optional, default=256
        Feed-forward inner width inside attention layers.
    dropout : float, optional, default=0.0
        Dropout probability shared across attention and MLP layers.
    nonlinear_correlation_proj : bool, optional, default=False
        If True, run the channel-correlation vector through a small learned
        nonlinear projection before broadcasting it into a mixing matrix.
    swap_corr_axis : bool, optional, default=False
        If True, multiply the correlation matrix on the right of the
        channel-flattened activations. See
        :class:`~physicsnemo.experimental.models.pv_power.attention.TwoStageAttentionLayer`.
    merge_kind : Literal["seg_merge", "cnn_merge"], optional, default="seg_merge"
        Selects the segment-downsampling layer used between encoder levels.
    attention_kind : Literal["two_stage", "parallel"], optional, default="two_stage"
        Selects between correlation-conditioned attention (``"two_stage"``)
        and pure self-attention along both axes (``"parallel"``).
    use_bottleneck_in_decoder : bool, optional, default=True
        If True, the encoder bottleneck output is fused into the deepest
        decoder level (U-Net skip-connection variant).

    Forward
    -------
    x_enc : torch.Tensor
        Target window of shape :math:`(B, L, C_{tgt})`. The last channel
        is treated as the primary forecast target.
    w_enc : torch.Tensor or None
        Weather forecast window of shape :math:`(B, L, C_{wx})`. May be
        ``None`` only when ``weather_channels == 0``.
    seq_w_nwp_hist : torch.Tensor or None
        Historical weather signal of shape :math:`(B, L, C_{wx})` used to
        compute the channel-correlation matrix. May be ``None`` only when
        ``weather_channels == 0``.
    seq_x_hist : torch.Tensor
        Historical target/history channels of shape
        :math:`(B, L, C_{tgt})`. Concatenated with ``seq_w_nwp_hist`` and
        the last channel of ``x_enc`` to produce a
        ``(B, L, C_{tgt} + C_{wx} + 1)`` matrix on which Pearson
        correlations are evaluated.

    Outputs
    -------
    torch.Tensor
        Forecast tensor of shape :math:`(B, H, C_{tgt} + C_{wx})`. The last
        ``target_channels`` columns correspond to the target channels.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.pv_power import CrossUnet
    >>> model = CrossUnet(
    ...     target_channels=4, weather_channels=3,
    ...     seq_len=96, pred_len=16, seg_len=12,
    ...     e_layers=2, d_model=32, n_heads=4, d_ff=64,
    ... )
    >>> x_enc = torch.randn(2, 96, 4)
    >>> w_enc = torch.randn(2, 96, 3)
    >>> hist_w = torch.randn(2, 96, 3)
    >>> hist_x = torch.randn(2, 96, 4)
    >>> out = model(x_enc, w_enc, hist_w, hist_x)
    >>> out.shape
    torch.Size([2, 16, 7])
    """

    def __init__(
        self,
        target_channels: int,
        weather_channels: int = 0,
        seq_len: int = 96,
        pred_len: int = 16,
        seg_len: int = 12,
        e_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.0,
        nonlinear_correlation_proj: bool = False,
        swap_corr_axis: bool = False,
        merge_kind: Literal["seg_merge", "cnn_merge"] = "seg_merge",
        attention_kind: Literal["two_stage", "parallel"] = "two_stage",
        use_bottleneck_in_decoder: bool = True,
    ) -> None:
        if target_channels < 1:
            raise ValueError(f"target_channels must be >= 1, got {target_channels}.")
        if weather_channels < 0:
            raise ValueError(f"weather_channels must be >= 0, got {weather_channels}.")
        if seq_len <= 0 or pred_len <= 0 or seg_len <= 0:
            raise ValueError(
                f"seq_len/pred_len/seg_len must be positive, got "
                f"({seq_len}, {pred_len}, {seg_len})."
            )
        if e_layers < 1:
            raise ValueError(f"e_layers must be >= 1, got {e_layers}.")

        super().__init__(meta=CrossUnetMetaData())

        self.target_channels = target_channels
        self.weather_channels = weather_channels
        self.total_channels = target_channels + weather_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.seg_len = seg_len
        self.e_layers = e_layers
        self.d_model = d_model
        self.dropout_p = dropout
        self.nonlinear_correlation_proj = nonlinear_correlation_proj
        self.use_weather = weather_channels > 0

        # Patch / segment grid sizes (mirrors upstream: pad to a multiple of seg_len).
        win_size = 2
        self.win_size = win_size
        pad_in_len = ceil(seq_len / seg_len) * seg_len
        pad_out_len = ceil(pred_len / seg_len) * seg_len
        in_seg_num = pad_in_len // seg_len
        out_seg_num = ceil(in_seg_num / (win_size ** (e_layers - 1)))
        dec_seg_num = pad_out_len // seg_len
        # Decoder must be at least as long as the encoder bottleneck only when
        # the bottleneck skip-add is enabled in ``Decoder.forward``.
        if use_bottleneck_in_decoder and dec_seg_num < out_seg_num:
            raise ValueError(
                f"pred_len={pred_len} (decoder segments={dec_seg_num}) is shorter "
                f"than the encoder bottleneck (segments={out_seg_num}) implied by "
                f"seq_len={seq_len}, seg_len={seg_len}, e_layers={e_layers}. "
                f"Either increase pred_len, decrease e_layers, or decrease seg_len."
            )
        self.in_seg_num = in_seg_num
        self.out_seg_num = out_seg_num

        # Embedding stack.
        self.enc_value_embedding = PatchEmbedding(
            d_model=d_model,
            patch_len=seg_len,
            stride=seg_len,
            padding=pad_in_len - seq_len,
            dropout=0.0,
        )
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, self.total_channels, in_seg_num, d_model)
        )
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, self.total_channels, pad_out_len // seg_len, d_model)
        )
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder.
        scale_blocks = [
            ScaleBlock(
                win_size=1 if level == 0 else win_size,
                n_vars=self.total_channels,
                seg_num=in_seg_num
                if level == 0
                else ceil(in_seg_num / win_size**level),
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                merge_kind=merge_kind,
                attention_kind=attention_kind,
                swap_corr_axis=swap_corr_axis,
            )
            for level in range(e_layers)
        ]
        self.encoder = Encoder(
            scale_blocks=scale_blocks,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Decoder (one extra level: e_layers + 1 layers).
        decoder_layers: list[DecoderLayer] = []
        for _ in range(e_layers + 1):
            if attention_kind == "two_stage":
                self_attn: nn.Module = TwoStageAttentionLayer(
                    n_vars=self.total_channels,
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    swap_corr_axis=swap_corr_axis,
                )
            else:
                self_attn = ParallelTwoStageAttentionLayer(
                    n_vars=self.total_channels,
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
            cross_attn = AttentionLayer(
                d_model=d_model,
                n_heads=n_heads,
                attention_dropout=dropout,
            )
            decoder_layers.append(
                DecoderLayer(
                    self_attention=self_attn,
                    cross_attention=cross_attn,
                    seg_len=seg_len,
                    d_model=d_model,
                    dropout=dropout,
                )
            )
        self.decoder = Decoder(
            layers=decoder_layers,
            use_bottleneck=use_bottleneck_in_decoder,
        )

        # Optional non-linear correlation projection.
        if nonlinear_correlation_proj:
            corr_dim = self.total_channels
            self.channel_proj1 = nn.Sequential(
                nn.Linear(corr_dim, corr_dim * 4, bias=False),
                nn.Sigmoid(),
                nn.Linear(corr_dim * 4, corr_dim, bias=False),
            )
            self.channel_proj2 = nn.Sequential(
                nn.Linear(corr_dim * corr_dim, corr_dim * corr_dim * 4, bias=False),
                nn.Sigmoid(),
                nn.Linear(corr_dim * corr_dim * 4, corr_dim * corr_dim, bias=False),
            )

    def _compute_channel_correlation(self, samples: Tensor) -> Tensor:
        r"""Pearson correlations reduced to a source-style mixing matrix.

        Parameters
        ----------
        samples : torch.Tensor
            Tensor of shape :math:`(B, L, C + 1)` whose final channel is the
            primary target signal used as the correlation reference.

        Returns
        -------
        torch.Tensor
            Mixing matrix of shape :math:`(B, C, C)` for the model channels.
        """
        b, _, c = samples.shape
        if c <= 1:
            return torch.ones(b, 1, 1, dtype=samples.dtype, device=samples.device)

        # Pearson correlation matrix per sample, shape (B, C, C).
        corr = vmap(_pearson_correlation)(samples)
        # Pull "feature -> target" column (excluding the target's diagonal).
        corr_to_target = corr[:, :-1, -1]  # (B, C - 1)
        corr_to_target = torch.nan_to_num(
            corr_to_target, nan=0.0, posinf=0.0, neginf=0.0
        )
        corr_to_target = corr_to_target.clamp(min=0.0)
        corr_dim = c - 1

        if self.nonlinear_correlation_proj:
            # Project, broadcast, project, softmax.
            v = corr_to_target.unsqueeze(-1)  # (B, C - 1, 1)
            v = self.channel_proj1(v.permute(0, 2, 1)).permute(0, 2, 1)
            v = v.repeat(1, 1, corr_dim)  # (B, C - 1, C - 1)
            flat = v.view(b, corr_dim * corr_dim)
            flat = self.channel_proj2(flat)
            mixing = F.softmax(flat, dim=-1).view(b, corr_dim, corr_dim)
        else:
            # Source-style rank-1 mixing: each feature-target softmax weight is
            # broadcast across the destination-channel axis.
            row = F.softmax(corr_to_target, dim=-1)  # (B, C - 1)
            mixing = row.unsqueeze(-1).repeat(1, 1, corr_dim)

        return mixing

    def _forecast(self, x: Tensor, corr: Tensor) -> Tensor:
        # ``x``: (B, L, C); ``corr``: (B, C, C).
        x_emb, n_vars = self.enc_value_embedding(x.permute(0, 2, 1))
        x_emb = rearrange(
            x_emb, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars
        )
        x_emb = x_emb + self.enc_pos_embedding
        x_emb = self.pre_norm(x_emb)
        enc_out, _ = self.encoder(x_emb, corr)
        dec_in = repeat(
            self.dec_pos_embedding,
            "b ts_d l d -> (repeat b) ts_d l d",
            repeat=x_emb.shape[0],
        )
        return self.decoder(dec_in, enc_out, corr)

    def forward(
        self,
        x_enc: Float[Tensor, "batch seq_len target_channels"],
        w_enc: Optional[Float[Tensor, "batch seq_len weather_channels"]],
        seq_w_nwp_hist: Optional[Float[Tensor, "batch seq_len weather_channels"]],
        seq_x_hist: Float[Tensor, "batch seq_len target_channels"],
    ) -> Float[Tensor, "batch pred_len total_channels"]:
        # Validate input shapes (skip under torch.compile per MOD-005).
        if not torch.compiler.is_compiling():
            self._validate_forward_inputs(x_enc, w_enc, seq_w_nwp_hist, seq_x_hist)

        # Build the correlation-conditioning input: weather history + target
        # history + full target history + the primary target channel from the
        # current window.
        if self.use_weather and w_enc is not None and seq_w_nwp_hist is not None:
            corr_input = torch.cat(
                [seq_w_nwp_hist, seq_x_hist, x_enc[:, :, -1:]], dim=-1
            )
            x = torch.cat([w_enc, x_enc], dim=-1)
        else:
            corr_input = torch.cat([seq_x_hist, x_enc[:, :, -1:]], dim=-1)
            x = x_enc

        corr = self._compute_channel_correlation(corr_input)
        dec_out = self._forecast(x, corr)
        return dec_out[:, : self.pred_len, :]

    def _validate_forward_inputs(
        self,
        x_enc: Tensor,
        w_enc: Optional[Tensor],
        seq_w_nwp_hist: Optional[Tensor],
        seq_x_hist: Tensor,
    ) -> None:
        batch = x_enc.shape[0]
        if x_enc.shape != (batch, self.seq_len, self.target_channels):
            raise ValueError(
                f"Expected x_enc of shape (B, {self.seq_len}, {self.target_channels}) "
                f"but got tensor of shape {tuple(x_enc.shape)}."
            )
        if seq_x_hist.shape != (batch, self.seq_len, self.target_channels):
            raise ValueError(
                f"Expected seq_x_hist of shape "
                f"(B, {self.seq_len}, {self.target_channels}) "
                f"but got tensor of shape {tuple(seq_x_hist.shape)}."
            )
        if self.use_weather:
            if w_enc is None or seq_w_nwp_hist is None:
                raise ValueError(
                    "w_enc and seq_w_nwp_hist are required when "
                    f"weather_channels={self.weather_channels} > 0."
                )
            if w_enc.shape != (batch, self.seq_len, self.weather_channels):
                raise ValueError(
                    f"Expected w_enc of shape (B, {self.seq_len}, "
                    f"{self.weather_channels}) but got tensor of shape "
                    f"{tuple(w_enc.shape)}."
                )
            if seq_w_nwp_hist.shape != (batch, self.seq_len, self.weather_channels):
                raise ValueError(
                    f"Expected seq_w_nwp_hist of shape (B, {self.seq_len}, "
                    f"{self.weather_channels}) but got tensor of shape "
                    f"{tuple(seq_w_nwp_hist.shape)}."
                )
