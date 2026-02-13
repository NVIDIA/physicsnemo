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
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.models.dit import DiT
from physicsnemo.experimental.models.healda.point_embed import MultiSensorObsEmbedder


@dataclass
class HealDAMetaData(ModelMetaData):
    """Metadata for HealDA model."""

    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    bf16: bool = True
    onnx: bool = False
    func_torch: bool = False
    auto_grad: bool = False


class HealDA(Module):
    r"""
    HealDA model that combines HEALPix tokenizers, observation embedders, and a DiT backbone.

    Parameters
    ----------
    in_channels : int
        Number of input conditioning channels (e.g. static conditioning channels
        like orography and land-sea mask).
    out_channels : int
        Number of output channels.
    nchannel_per_sensor : list[int]
        Number of channels for each sensor, in sensor order.
    nplatform_per_sensor : list[int]
        Number of platforms for each sensor, in sensor order.
    sensor_names : list[str], optional
        Human-readable names for each sensor, in sensor order. Passed to
        :class:`~physicsnemo.experimental.models.healda.point_embed.MultiSensorObsEmbedder`
        as ``ModuleDict`` keys. Defaults to ``["sensor_0", "sensor_1", ...]``.
    hidden_size : int, optional, default=1024
        Transformer hidden dimension.
    num_layers : int, optional, default=24
        Number of transformer blocks.
    num_heads : int, optional, default=16
        Number of attention heads.
    mlp_ratio : float, optional, default=4.0
        MLP hidden dim multiplier.
    level_in : int, optional, default=6
        HEALPix input resolution level.
    level_model : int, optional, default=5
        HEALPix model resolution level after patching.
    time_length : int, optional, default=1
        Number of time windows.
    embed_dim : int, optional, default=32
        Tokenization dimension for observation embedder.
    meta_dim : int, optional, default=28
        Dimension of float metadata features for observation embedder.
    fusion_dim : int, optional, default=512
        Output dimension after sensor fusion.
    obs_gradient_checkpointing : bool, optional, default=False
        If ``True``, enables gradient checkpointing in the observation embedder.
    compile_obs_embedder : bool, optional, default=False
        If ``True``, applies ``torch.compile`` to the observation embedder's forward method.
    qk_norm_type : Literal["RMSNorm", "LayerNorm"], optional, default="RMSNorm"
        QK normalization type. ``None`` disables QK normalization.
    qk_norm_affine : bool, optional, default=True
        Whether QK normalization layers use learnable affine parameters (timm backend only).
    drop_path : float, optional, default=0.0
        Maximum DropPath rate for stochastic depth. Linearly schedule from 0
        (first layer) to ``drop_path`` (last layer).
    dropout : float, optional, default=0.0
        Dropout rate for projection and MLP layers.
    diffusion_conditioning : bool, optional, default=True
        If ``True``, creates an EDM-style conditioning embedder that maps the
        noise timestep (and optionally class labels) to a conditioning vector
        fed into the AdaLN modulation layers. If ``False``, no conditioning
        embedder is created and AdaLN MLP reduces to learnable biases.
    condition_dim : int, optional, default=0
        Dimension of class-label condition vectors. Only used when
        ``diffusion_conditioning`` is ``True``. If 0, no class-label input is
        used (noise-only conditioning). If > 0, an additional linear layer
        projects the label vector into the noise embedding before the MLP.
    legacy_condition_bias : bool, optional, default=True
        If ``True`` and ``condition_dim == 0``, still creates a bias-only
        linear layer in the conditioning embedder.
    condition_embed_dim : int, optional, default=None
        Output dimension of the conditioning embedder, which feeds into the
        AdaLN modulation layers in each DiTBlock and the detokenizer. If
        ``None``, defaults to ``4 * hidden_size``. Ignored when
        ``diffusion_conditioning`` is ``False``.
    noise_channels : int, optional, default=None
        Intermediate embedding dimension of the noise timestep inside the
        EDM conditioning embedder. If ``None``, defaults to ``hidden_size``.
        Ignored when ``diffusion_conditioning`` is ``False``.
    condition_dropout : float, optional, default=0.0
        Dropout rate for condition vectors during training. Only relevant when
        ``condition_dim > 0``.
    attention_backend : str, optional, default="transformer_engine"
        Attention backend to use.
    layernorm_backend : str, optional, default="apex"
        LayerNorm backend to use.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, T, N_{pix})` where
        :math:`C_{in}` is ``in_channels``.
        and :math:`N_{pix} = 12 \times 4^{\mathrm{level\_in}}`.
    t : torch.Tensor
        Diffusion timestep (noise level) tensor of shape :math:`(B,)`.
    obs : torch.Tensor
        Flattened scalar observation values of shape :math:`(N_{obs},)`,
        concatenated across all sensors, batch elements, and time windows.
    float_metadata : torch.Tensor
        Per-observation float metadata of shape :math:`(N_{obs}, M)` (e.g. features based
        on observation time, latitude, longitude, scan angles, etc.).
    pix : torch.Tensor
        Spatial pixel index for each observation of shape :math:`(N_{obs},)`,
        mapping observations onto the HEALPix grid.
    local_channel : torch.Tensor
        Sensor-local channel id for each observation of shape
        :math:`(N_{obs},)` (e.g. which channel/variable within a sensor).
    local_platform : torch.Tensor
        Sensor-local platform id for each observation of shape
        :math:`(N_{obs},)` (e.g. which satellite).
    obs_type : torch.Tensor
        Observation type id for each observation of shape
        :math:`(N_{obs},)`, used by the tokenizer embedding table.
    offsets : torch.Tensor
        Cumulative exclusive-end row offsets of shape :math:`(S, B, T)` into
        the flattened observation tensors. ``offsets[s, b, t]`` gives the end
        index for sensor ``s``, batch element ``b``, time window ``t``.
        See :class:`~physicsnemo.experimental.models.healda.point_embed.MultiSensorObsEmbedder`
        for more details.
    second_of_day : torch.Tensor
        Second-of-day tensor of shape :math:`(B, T)` for calendar embedding
        in the HEALPix tokenizer.
    day_of_year : torch.Tensor
        Day-of-year tensor of shape :math:`(B, T)` for calendar embedding
        in the HEALPix tokenizer.
    class_labels : torch.Tensor, optional
        Condition vectors of shape :math:`(B, \text{condition\_dim})` for the
        EDM conditioning embedder. Required when ``diffusion_conditioning``
        is ``True`` and ``condition_dim > 0``. When ``condition_dim == 0``
        and ``legacy_condition_bias`` is ``True``, pass empty tensor with shape
        ``(B, 0)`` to use the learned bias term.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, T, N_{pix})`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nchannel_per_sensor: list[int],
        nplatform_per_sensor: list[int],
        sensor_names: list[str] | None = None,
        hidden_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        level_in: int = 6,
        level_model: int = 5,
        time_length: int = 1,
        embed_dim: int = 32,
        meta_dim: int = 28,
        fusion_dim: int = 512,
        obs_gradient_checkpointing: bool = False,
        compile_obs_embedder: bool = False,
        qk_norm_type: Literal["RMSNorm", "LayerNorm"] | None = "RMSNorm",
        qk_norm_affine: bool = False,
        drop_path: float = 0.0,
        dropout: float = 0.0,
        diffusion_conditioning: bool = True,
        condition_dim: int = 0,
        condition_embed_dim: int | None = None,
        noise_channels: int | None = None,
        legacy_condition_bias: bool = True,
        condition_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        attention_backend: str = "timm",
        layernorm_backend: str = "torch",
    ):
        super().__init__(meta=HealDAMetaData())

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.level_in = level_in
        self.level_model = level_model
        self.time_length = time_length
        self.diffusion_conditioning = diffusion_conditioning
        self.condition_dim = condition_dim
        self.fusion_dim = fusion_dim
        self.qk_norm_type = qk_norm_type
        self.qk_norm_affine = qk_norm_affine
        self.attention_backend = attention_backend

        self.npix = 12 * 4**level_in

        # Observation encoder (embeds obs and concatenates with state)
        self.obs_embedder = MultiSensorObsEmbedder(
            nchannel_per_sensor=nchannel_per_sensor,
            nplatform_per_sensor=nplatform_per_sensor,
            sensor_names=sensor_names,
            embed_dim=embed_dim,
            meta_dim=meta_dim,
            fusion_dim=fusion_dim,
            gradient_checkpointing=obs_gradient_checkpointing,
            torch_compile=compile_obs_embedder,
        )
        tokenizer_in_channels = in_channels + fusion_dim

        npix_coarse = 12 * 4**level_model
        attn_kwargs = {"qk_norm_type": qk_norm_type} if qk_norm_type else {}
        if qk_norm_type:
            # TE backend doesn't support qk_norm_affine=False,
            # raises warning and ignores it
            attn_kwargs["qk_norm_affine"] = qk_norm_affine
        if attention_backend == "transformer_engine":
            attn_kwargs["qkv_format"] = "bshd"

        # Skip attention-weight dropout and final-layer dropout in the DiTBlock MLP.
        block_kwargs = {
            "proj_drop_rate": dropout,
            "mlp_drop_rate": dropout,
            "norm_eps": norm_eps,
            "final_mlp_dropout": False,
        }

        if diffusion_conditioning:
            if condition_embed_dim is None:
                condition_embed_dim = 4 * hidden_size
            if noise_channels is None:
                noise_channels = hidden_size
            conditioning_embedder = "edm"
            conditioning_embedder_kwargs = {
                "emb_channels": condition_embed_dim,
                "noise_channels": noise_channels,
                "condition_dropout": condition_dropout,
                "legacy_condition_bias": legacy_condition_bias,
            }
        else:
            condition_embed_dim = 0
            conditioning_embedder = "zero"
            conditioning_embedder_kwargs = {}

        patch_size = 2 ** (level_in - level_model)
        self.dit = DiT(
            input_size=(npix_coarse * time_length,),  # ignored by hpx tokenizer
            in_channels=tokenizer_in_channels,
            patch_size=(patch_size, patch_size), # ignored by hpx tokenizer
            tokenizer="hpx_patch_embed",
            detokenizer="hpx_patch_detokenizer",
            out_channels=out_channels,
            hidden_size=hidden_size,
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attention_backend=attention_backend,
            layernorm_backend=layernorm_backend,
            condition_dim=condition_dim,
            conditioning_embedder=conditioning_embedder,
            conditioning_embedder_kwargs=conditioning_embedder_kwargs,
            drop_path_rates=[
                drop_path * i / max(1, num_layers - 1) for i in range(num_layers)
            ],
            attn_kwargs=attn_kwargs,
            block_kwargs=block_kwargs,
            tokenizer_kwargs={"level_fine": level_in, "level_coarse": level_model},
            detokenizer_kwargs={
                "level_coarse": level_model,
                "level_fine": level_in,
                "time_length": time_length,
                "condition_dim": condition_embed_dim,
            },
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch in_channels time npix"],
        t: Float[torch.Tensor, "batch"],
        obs: Float[torch.Tensor, "nobs"],
        float_metadata: Float[torch.Tensor, "nobs meta_dim"],
        pix: Int[torch.Tensor, "nobs"],
        local_channel: Int[torch.Tensor, "nobs"],
        local_platform: Int[torch.Tensor, "nobs"],
        obs_type: Int[torch.Tensor, "nobs"],
        offsets: Int[torch.Tensor, "sensors batch time"],
        second_of_day: Float[torch.Tensor, "batch time"],
        day_of_year: Float[torch.Tensor, "batch time"],
        class_labels: Float[torch.Tensor, "batch condition_dim"] | None = None,
    ) -> Float[torch.Tensor, "batch out_channels time npix"]:
        # Embed observations onto HEALPix grid
        obs_embedding = self.obs_embedder(
            obs=obs,
            float_metadata=float_metadata,
            pix=pix,
            local_channel=local_channel,
            local_platform=local_platform,
            obs_type=obs_type,
            offsets=offsets,
            npix=self.npix,
        ) # (b, c, t, npix)
        x = torch.cat([x, obs_embedding], dim=1)

        return self.dit(
            x,
            t,
            condition=class_labels,
            tokenizer_kwargs={
                "second_of_day": second_of_day,
                "day_of_year": day_of_year,
            },
        )
