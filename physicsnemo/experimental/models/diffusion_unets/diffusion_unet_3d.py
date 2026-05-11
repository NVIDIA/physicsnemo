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
from typing import List, Literal, cast

import numpy as np
import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch.nn.functional import silu
from torch.utils.checkpoint import checkpoint

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.nn import Conv3D, GroupNorm3D, UNetBlock3D
from physicsnemo.nn import (
    FourierEmbedding,
    Linear,
    PositionalEmbedding,
)


@dataclass
class MetaData(ModelMetaData):
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class DiffusionUNet3D(Module):
    r"""
    3D U-Net diffusion backbone for volumetric data.

    Implements the :class:`~physicsnemo.diffusion.base.DiffusionModel` protocol
    and can be used directly with preconditioners, losses, and samplers from
    :mod:`physicsnemo.diffusion`. Conceptually a 3D counterpart of
    :class:`~physicsnemo.models.diffusion_unets.SongUNet`; refer to that class
    for the underlying architectural overview.

    Based on the architecture described in `Diff-SPORT: Diffusion-based Sensor
    Placement Optimization and Reconstruction of Turbulent flows in urban
    environments <https://arxiv.org/abs/2506.00214>`_.

    Parameters
    ----------
    x_channels : int
        Number of channels :math:`C_x` in the input/output state
        :math:`\mathbf{x}`.
        The output has the same number of channels as the input.
    vol_cond_channels : int, optional, default=0
        Number of channels :math:`C_{cond,v}` in the optional volume-based
        conditioning. When non-zero, a volume condition tensor of shape
        :math:`(B, C_{cond,v}, D, H, W)` may be passed via ``condition["volume"]``;
        it is concatenated channel-wise to ``x`` before the first convolution. Set to ``0`` for no volume conditioning.
    vec_cond_dim : int, optional, default=0
        Dimension :math:`D_v` of the optional vector-valued condition.
        When non-zero, a condition tensor of shape
        :math:`(B, D_v)` may be passed via ``condition["vector"]``. The vector
        condition is mapped through a linear layer and added to the diffusion time
        embedding; the resulting embedding then conditions all 3D U-Net blocks
        via adaptive group norm.
    num_levels : int, optional, default=4
        Number of encoder/decoder levels. ``len(channel_mult)`` must equal this.
    model_channels : int, optional, default=128
        Base channel count at the first U-Net level.
    channel_mult : list[int], optional, default=[1, 2, 2, 2]
        Per-level channel multipliers. Channels at level :math:`l` equal
        ``channel_mult[l] * model_channels``. Length must equal ``num_levels``.
    channel_mult_emb : int, optional, default=4
        Multiplier for the conditioning embedding dimension:
        ``emb_channels = model_channels * channel_mult_emb``.
    num_blocks : int, optional, default=4
        Number of 3D U-Net blocks per level. The decoder has ``num_blocks + 1``
        blocks per level for the extra skip connection.
    attention_levels : list[int], optional, default=[]
        0-indexed encoder levels at which to apply 3D self-attention. Level 0
        is the outermost (highest resolution). All values must be in
        ``[0, num_levels)``.
    dropout : float, optional, default=0.10
        Dropout probability inside the 3D U-Net blocks.
    embedding_type : Literal["positional", "fourier", "zero"], optional, default="positional"
        Embedding type used for both the diffusion time and (when present) the
        vector condition. ``"positional"`` is the DDPM++ style, ``"fourier"`` is
        the NCSN++ style, and ``"zero"`` replaces the time embedding by a zero
        buffer and disables vector conditioning (so ``vec_cond_dim`` must be
        ``0``). Volume conditioning is independent of ``embedding_type`` since
        it is concatenated channel-wise to ``x`` before the first convolution.
    channel_mult_noise : int, optional, default=1
        Multiplier for the noise-level embedding dimension:
        ``noise_channels = model_channels * channel_mult_noise``.
    encoder_type : Literal["standard", "skip", "residual"], optional, default="standard"
        Encoder architecture variant (``"standard"`` = DDPM++,
        ``"residual"`` = NCSN++, ``"skip"`` = skip connections).
    decoder_type : Literal["standard", "skip"], optional, default="standard"
        Decoder architecture variant.
    resample_filter : list[int], optional, default=[1, 1]
        1D coefficients for the separable up/downsampling filter. The 3D filter
        is constructed as their outer product, normalized to sum to 1. Use
        ``[1, 1]`` for bilinear (DDPM++) or ``[1, 3, 3, 1]`` for bicubic
        (NCSN++) resampling.
    checkpoint_level : int, optional, default=0
        Gradient checkpointing aggressiveness. Higher values checkpoint more
        layers, trading memory for compute. ``0`` disables checkpointing.
    bottleneck_attention : bool, optional, default=True
        If ``True``, applies 3D self-attention at the innermost bottleneck block.
        Set to ``False`` for faster inference without bottleneck attention.
    activation : Literal["silu", "gelu"], optional, default="silu"
        Activation function used inside the 3D U-Net blocks.

    Forward
    -------
    x : torch.Tensor
        Input state of shape :math:`(B, C_x, D, H, W)`. Spatial dimensions must
        be powers of 2 or multiples of :math:`2^{\text{num_levels}-1}`.
    t : torch.Tensor
        Batched diffusion time (or noise level) of shape :math:`(B,)`.
    condition : TensorDict or None, optional, default=None
        Conditioning information. ``None`` for unconditional models. Otherwise
        a :class:`~tensordict.TensorDict` with a subset of:

        - ``"vector"``: tensor of shape :math:`(B, D_v)` (requires
          ``vec_cond_dim > 0``).
        - ``"volume"``: tensor of shape :math:`(B, C_v, D, H, W)` (requires
          ``vol_cond_channels > 0`` and matching spatial dimensions).

        Any other key raises ``ValueError``.

    Outputs
    -------
    torch.Tensor
        Output of shape :math:`(B, C_x, D, H, W)`. The channels match
        :math:`C_x` so the model can be used as any predictor type
        (:math:`\mathbf{x}_0`, :math:`\boldsymbol{\epsilon}`, score,
        velocity, etc.); the interpretation depends on the predictor / loss.

    Raises
    ------
    ValueError
        If ``len(channel_mult) != num_levels``.
    ValueError
        If any value in ``attention_levels`` is outside ``[0, num_levels)``.
    ValueError
        If ``embedding_type == "zero"`` is combined with non-zero
        ``vec_cond_dim``.

    See Also
    --------
    :class:`~physicsnemo.models.diffusion_unets.SongUNet` : 2D counterpart.
    :class:`~physicsnemo.diffusion.base.DiffusionModel` : Protocol this model
        implements.

    Examples
    --------
    Unconditional model on a non-cubic grid:

    >>> import torch
    >>> from physicsnemo.experimental.models.diffusion_unets import DiffusionUNet3D
    >>> model = DiffusionUNet3D(
    ...     x_channels=4, num_levels=2,
    ...     model_channels=16, channel_mult=[1, 2], num_blocks=1,
    ... )
    >>> x = torch.randn(2, 4, 4, 12, 16)
    >>> out = model(x, torch.randn(2))
    >>> out.shape
    torch.Size([2, 4, 4, 12, 16])

    Conditional model with vector and volume conditioning:

    >>> from tensordict import TensorDict
    >>> model = DiffusionUNet3D(
    ...     x_channels=4, vol_cond_channels=2, vec_cond_dim=8,
    ...     num_levels=2, model_channels=16, channel_mult=[1, 2], num_blocks=1,
    ... )
    >>> cond = TensorDict(
    ...     {"vector": torch.randn(2, 8), "volume": torch.randn(2, 2, 4, 12, 16)},
    ...     batch_size=[2],
    ... )
    >>> out = model(x, torch.randn(2), condition=cond)
    >>> out.shape
    torch.Size([2, 4, 4, 12, 16])

    Larger conditional model with custom encoder/decoder, attention at level 1,
    NCSN++-style filter, no bottleneck attention, and gelu activation:

    >>> model = DiffusionUNet3D(
    ...     x_channels=2, vol_cond_channels=1, vec_cond_dim=4,
    ...     num_levels=3, model_channels=16, channel_mult=[1, 2, 2], num_blocks=2,
    ...     attention_levels=[1], encoder_type="residual", decoder_type="skip",
    ...     resample_filter=[1, 3, 3, 1], bottleneck_attention=False,
    ...     activation="gelu",
    ... )
    >>> x = torch.randn(2, 2, 4, 12, 16)
    >>> cond = TensorDict(
    ...     {"vector": torch.randn(2, 4), "volume": torch.randn(2, 1, 4, 12, 16)},
    ...     batch_size=[2],
    ... )
    >>> out = model(x, torch.randn(2), condition=cond)
    >>> out.shape
    torch.Size([2, 2, 4, 12, 16])
    """

    def __init__(
        self,
        x_channels: int,
        vol_cond_channels: int = 0,
        vec_cond_dim: int = 0,
        num_levels: int = 4,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 2, 2],
        channel_mult_emb: int = 4,
        num_blocks: int = 4,
        attention_levels: List[int] = [],
        dropout: float = 0.10,
        embedding_type: Literal["fourier", "positional", "zero"] = "positional",
        channel_mult_noise: int = 1,
        encoder_type: Literal["standard", "skip", "residual"] = "standard",
        decoder_type: Literal["standard", "skip"] = "standard",
        resample_filter: List[int] = [1, 1],
        checkpoint_level: int = 0,
        bottleneck_attention: bool = True,
        activation: Literal["silu", "gelu"] = "silu",
    ):
        if len(channel_mult) != num_levels:
            raise ValueError(
                f"len(channel_mult) must equal num_levels, got "
                f"len(channel_mult)={len(channel_mult)} and num_levels={num_levels}"
            )

        if any(not (0 <= lvl < num_levels) for lvl in attention_levels):
            raise ValueError(
                f"All values in attention_levels must be in [0, num_levels="
                f"{num_levels}), got {attention_levels}"
            )

        if embedding_type == "zero" and vec_cond_dim > 0:
            raise ValueError(
                "embedding_type='zero' disables the conditioning embedding; "
                "vec_cond_dim must be 0 in that case "
                f"(got vec_cond_dim={vec_cond_dim})."
            )

        super().__init__(meta=MetaData())

        self.x_channels = x_channels
        self.vol_cond_channels = vol_cond_channels
        self.vec_cond_dim = vec_cond_dim
        self.embedding_type = embedding_type
        self.num_levels = num_levels
        self._input_shape_mult = 2 ** (num_levels - 1)
        self.checkpoint_level = checkpoint_level

        emb_channels = model_channels * channel_mult_emb
        self.emb_channels = emb_channels
        noise_channels = model_channels * channel_mult_noise

        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))

        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            activation=activation,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        if self.embedding_type != "zero":
            self.map_noise = (
                PositionalEmbedding(num_channels=noise_channels, endpoint=True)
                if embedding_type == "positional"
                else FourierEmbedding(num_channels=noise_channels)
            )
            self.map_condition = (
                Linear(in_features=vec_cond_dim, out_features=noise_channels, **init)
                if vec_cond_dim > 0
                else None
            )
            self.map_layer0 = Linear(
                in_features=noise_channels, out_features=emb_channels, **init
            )
            self.map_layer1 = Linear(
                in_features=emb_channels, out_features=emb_channels, **init
            )
        else:
            # FSDP-compatible zero buffer; persistent=False keeps it out of state_dict
            self.register_buffer(
                "zero_emb", torch.zeros(1, emb_channels), persistent=False
            )
            self.map_condition = None

        # Encoder
        self.enc = torch.nn.ModuleDict()
        cout = x_channels + vol_cond_channels
        caux = x_channels + vol_cond_channels
        for level, mult in enumerate(channel_mult):
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"l{level}_conv"] = Conv3D(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"l{level}_down"] = UNetBlock3D(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"l{level}_aux_down"] = Conv3D(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"l{level}_aux_skip"] = Conv3D(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == "residual":
                    self.enc[f"l{level}_aux_residual"] = Conv3D(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = level in attention_levels
                self.enc[f"l{level}_block{idx}"] = UNetBlock3D(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )

        skips = [
            block.out_channels
            for name, block in self.enc.items()
            if "aux" not in name
        ]

        # Decoder
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            if level == len(channel_mult) - 1:
                self.dec[f"l{level}_in0"] = UNetBlock3D(
                    in_channels=cout,
                    out_channels=cout,
                    attention=bottleneck_attention,
                    **block_kwargs,
                )
                self.dec[f"l{level}_in1"] = UNetBlock3D(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"l{level}_up"] = UNetBlock3D(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and level in attention_levels
                self.dec[f"l{level}_block{idx}"] = UNetBlock3D(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"l{level}_aux_up"] = Conv3D(
                        in_channels=x_channels,
                        out_channels=x_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                    )
                self.dec[f"l{level}_aux_norm"] = GroupNorm3D(
                    num_channels=cout, eps=1e-6
                )
                self.dec[f"l{level}_aux_conv"] = Conv3D(
                    in_channels=cout, out_channels=x_channels, kernel=3, **init_zero
                )

    def forward(
        self,
        x: Float[torch.Tensor, "B C_x D H W"],
        t: Float[torch.Tensor, " B"],
        condition: TensorDict | None = None,
    ) -> Float[torch.Tensor, "B C_x D H W"]:

        # Tensor shape validation
        if not torch.compiler.is_compiling():
            if x.ndim != 5:
                raise ValueError(
                    f"Expected x to be a 5D tensor, "
                    f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
                )

            B, _, D, H, W = x.shape

            if x.shape[1] != self.x_channels:
                raise ValueError(
                    f"Expected x to have {self.x_channels} channels (x_channels), "
                    f"got {x.shape[1]}"
                )

            for d in (D, H, W):
                is_power_of_2 = (d & (d - 1)) == 0 and d > 0
                if not (
                    (is_power_of_2 and d < self._input_shape_mult)
                    or (d % self._input_shape_mult == 0)
                ):
                    raise ValueError(
                        f"Input spatial dimensions (D, H, W)={(D, H, W)} must be "
                        f"powers of 2 or multiples of 2**(num_levels-1)="
                        f"{self._input_shape_mult}"
                    )

            if t.shape != (B,):
                raise ValueError(
                    f"Expected t to have shape ({B},), got {tuple(t.shape)}"
                )

            if condition is not None:
                valid_keys = {"vector", "volume"}
                extra_keys = set(condition.keys()) - valid_keys
                if extra_keys:
                    raise ValueError(
                        f"Unexpected condition keys: {extra_keys}. "
                        f"Allowed keys: {valid_keys}"
                    )

                vector_cond = condition.get("vector", None)
                volume_cond = condition.get("volume", None)

                if vector_cond is not None:
                    if self.embedding_type == "zero":
                        raise ValueError(
                            "condition['vector'] cannot be used with "
                            "embedding_type='zero'."
                        )
                    if self.vec_cond_dim == 0:
                        raise ValueError(
                            "condition['vector'] provided but vec_cond_dim=0"
                        )
                    if vector_cond.shape != (B, self.vec_cond_dim):
                        raise ValueError(
                            f"Expected condition['vector'] to have shape "
                            f"{(B, self.vec_cond_dim)}, got {tuple(vector_cond.shape)}"
                        )

                if volume_cond is not None:
                    if self.vol_cond_channels == 0:
                        raise ValueError(
                            "condition['volume'] provided but vol_cond_channels=0"
                        )
                    if volume_cond.shape != (B, self.vol_cond_channels, D, H, W):
                        raise ValueError(
                            f"Expected condition['volume'] to have shape "
                            f"{(B, self.vol_cond_channels, D, H, W)}, got {tuple(volume_cond.shape)}"
                        )

        # Extract condition components (no isinstance under torch.compile)
        if condition is not None:
            vector_cond = condition.get("vector", None)
            volume_cond = condition.get("volume", None)
        else:
            vector_cond = None
            volume_cond = None

        # Prepend volume condition channels to x
        if volume_cond is not None:
            x = torch.cat([x, volume_cond], dim=1)  # (B, C_x + C_v, D, H, W)

        # Compute conditioning embedding from t and optional vector_cond
        if self.embedding_type != "zero":
            emb = self.map_noise(t)
            emb_shape = emb.shape
            # Swap sin/cos halves to match the DDPM++ convention
            emb = emb.reshape(emb.shape[0], 2, -1)
            emb = torch.concat([emb[:, 1:], emb[:, :1]], dim=1).reshape(*emb_shape)
            if self.map_condition is not None and vector_cond is not None:
                emb = emb + self.map_condition(
                    vector_cond * np.sqrt(self.map_condition.in_features)
                )
            emb = silu(self.map_layer0(emb))
            emb = silu(self.map_layer1(emb))
        else:
            emb = self.zero_emb.repeat(t.shape[0], 1)

        # Gradient-checkpointing threshold from current spatial extent
        max_dim = max(x.shape[-3], x.shape[-2], x.shape[-1])
        threshold = (max_dim >> self.checkpoint_level) + 1

        # Encoder: progressively downsample and cache skip connections
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                # Normalize by 1/sqrt(2) to preserve activation variance
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            elif "_conv" in name:
                x = block(x)
                skips.append(x)
            else:
                if isinstance(block, UNetBlock3D):
                    if max(x.shape[-3], x.shape[-2], x.shape[-1]) > threshold:
                        x = checkpoint(block, x, emb, use_reentrant=False)
                    else:
                        x = block(x, emb)
                else:
                    x = block(x)
                skips.append(x)

        # Decoder: progressively upsample and merge skip connections.
        out = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                out = block(out)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                out = tmp if out is None else tmp + out
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                cur_max = max(x.shape[-3], x.shape[-2], x.shape[-1])
                if (cur_max > threshold and "_block" in name) or (
                    cur_max > (threshold / 2) and "_up" in name
                ):
                    x = checkpoint(block, x, emb, use_reentrant=False)
                else:
                    x = block(x, emb)

        return cast(torch.Tensor, out)
