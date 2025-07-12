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

import contextlib
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Set, Union

import numpy as np
import nvtx
import torch
from torch.nn.functional import silu
from torch.utils.checkpoint import checkpoint

from physicsnemo.models.diffusion import (
    Conv2d,
    FourierEmbedding,
    GroupNorm,
    Linear,
    PositionalEmbedding,
    UNetBlock,
)
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

# ------------------------------------------------------------------------------
# Backbone architectures
# ------------------------------------------------------------------------------


@dataclass
class MetaData(ModelMetaData):
    name: str = "SongUNet"
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


class SongUNet(Module):
    r"""
    This architecture is a diffusion backbone for 2D image generation.
    It is a reimplementation of the `DDPM++
    <https://proceedings.mlr.press/v139/nichol21a.html>`_ and `NCSN++ <https://arxiv.org/abs/2011.13456>`_
    architectures, which are U-Net variants
    with optional self-attention, embeddings, and encoder-decoder components.

    This model supports conditional and unconditional setups, as well as several
    options for various internal architectural choices such as encoder and decoder
    type, embedding type, etc., making it flexible and adaptable to different tasks
    and configurations.

    This architecture supports conditioning on the noise level (called *noise labels*),
    as well as on additional vector-valued labels (called *class labels*) and (optional)
    vector-valued augmentation labels. The conditioning mechanism relies on addition
    of the conditioning embeddings in the U-Net blocks of the encoder. To condition
    on images, the simplest mechanism is to concatenate the image to the input
    before passing it to the SongUNet.

    The model first applies a mapping operation to generate embeddings for all
    the conditioning inputs (the noise level, the class labels, and the
    optional augmentation labels).

    Then, at each level in the U-Net encoder, a sequence of blocks is applied:

    • A first block downsamples the feature map resolution by a factor of 2
      (odd resolutions are floored). This block does not change the number of
      channels.

    • A sequence of ``num_blocks`` U-Net blocks are applied, each with a different
      number of channels. These blocks do not change the feature map
      resolution, but they multiply the number of channels by a factor
      specified in ``channel_mult``.
      If required, the U-Net blocks also apply self-attention at the specified
      resolutions.

    • At the end of the level, the feature map is cached to be used in a skip
      connection in the decoder.

    The decoder is a mirror of the encoder, with the same number of levels and
    the same number of blocks per level. It multiplies the feature map resolution
    by a factor of 2 at each level.

    Parameters
    -----------
    img_resolution : Union[List[int, int], int]
        The resolution of the input/output image. Can be a single int :math:`H` for
        square images or a list :math:`[H, W]` for rectangular images.

        *Note:* This parameter is only used as a convenience to build the
        network. In practice, the model can still be used with images of
        different resolutions. The only exception to this rule is when
        ``additive_pos_embed`` is True, in which case the resolution of the latent
        state :math:`\mathbf{x}` must match ``img_resolution``.
    in_channels : int
        Number of channels :math:`C_{in}` in the input image. May include channels from both
        the latent state and additional channels when conditioning on images.
        For an unconditional model, this should be equal to ``out_channels``.
    out_channels : int
        Number of channels :math:`C_{out}` in the output image. Should be equal to the number
        of channels :math:`C_{\mathbf{x}}` in the latent state.
    label_dim : int, optional, default=0
        Dimension of the vector-valued ``class_labels`` conditioning; 0
        indicates no conditioning on class labels.
    augment_dim : int, optional, default=0
        Dimension of the vector-valued `augment_labels` conditioning; 0 means
        no conditioning on augmentation labels.
    model_channels : int, optional, default=128
        Base multiplier for the number of channels accross the entire network.
    channel_mult : List[int], optional, default=[1, 2, 2, 2]
        Multipliers for the number of channels at every level in
        the encoder and decoder. The length of ``channel_mult`` determines the
        number of levels in the U-Net. At level ``i``, the number of channel in
        the feature map is ``channel_mult[i] * model_channels``.
    channel_mult_emb : int, optional, default=4
        Multiplier for the number of channels in the embedding vector. The
        embedding vector has ``model_channels * channel_mult_emb`` channels.
    num_blocks : int, optional, default=4
        Number of U-Net blocks at each level.
    attn_resolutions : List[int], optional, default=[16]
        Resolutions of the levels at which self-attention layers are applied.
        Note that the feature map resolution must match exactly the value
        provided in `attn_resolutions` for the self-attention layers to be
        applied.
    dropout : float, optional, default=0.10
        Dropout probability applied to intermediate activations within the
        U-Net blocks.
    label_dropout : float, optional, default=0.0
        Dropout probability applied to the `class_labels`. Typically used for
        classifier-free guidance.
    embedding_type : Literal["fourier", "positional", "zero"], optional, default="positional"
        Diffusion timestep embedding type: 'positional' for DDPM++, 'fourier'
        for NCSN++, 'zero' for none.
    channel_mult_noise : int, optional, default=1
        Multiplier for the number of channels in the noise level embedding. The
        noise level embedding vector has ``model_channels * channel_mult_noise`` channels.
    encoder_type : Literal["standard", "skip", "residual"], optional, default="standard"
        Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++, 'skip' for skip connections.
    decoder_type : Literal["standard", "skip"], optional, default="standard"
        Decoder architecture: 'standard' or 'skip' for skip connections.
    resample_filter : List[int], optional, default=[1, 1]
        Resampling filter coefficients applied in the U-Net blocks
        convolutions: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    checkpoint_level : int, optional, default=0
        Number of levels that should use gradient checkpointing. Only levels at
        which the feature map resolution is large enough will be checkpointed
        (0 disables checkpointing, higher values means more layers are checkpointed).
        Higher values trade memory for computation.
    additive_pos_embed : bool, optional, default=False
        If ``True``, adds a learnable positional embedding after the first convolution layer.
        Used in StormCast model.

        *Note:* Those positional embeddings encode spatial position information
        of the image pixels, unlike the ``embedding_type`` parameter which encodes
        temporal information about the diffusion process. In that sense it is a
        simpler version of the positional embedding used in
        :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd`.
    use_apex_gn : bool, optional, default=False
        A flag indicating whether we want to use Apex GroupNorm for NHWC layout.
        Apex needs to be installed for this to work. Need to set this as False on cpu.
    act : str, optional, default=None
        The activation function to use when fusing activation with GroupNorm.
        Required when ``use_apex_gn`` is ``True``.
    profile_mode : bool, optional, default=False
        A flag indicating whether to enable all nvtx annotations during
        profiling.
    amp_mode : bool, optional, default=False
        A flag indicating whether mixed-precision (AMP) training is enabled.


    Forward
    -------
    x : torch.Tensor
        The input image of shape :math:`(B, C_{in}, H_{in}, W_{in})`. In
        general ``x`` is the channel-wise concatenation of the latent state
        :math:`\mathbf{x}` and additional images used for conditioning. For an
        unconditional model, ``x`` is simply the latent state
        :math:`\mathbf{x}`.

        *Note:* :math:`H_{in}` and :math:`W_{in}` do not need to match
        :math:`H` and :math:`W` defined in ``img_resolution``, except when
        ``additive_pos_embed`` is ``True``. In that case, the resolution of
        ``x`` must match ``img_resolution``.
    noise_labels : torch.Tensor
        The noise labels of shape :math:`(B,)`. Used for conditioning on
        the diffusion noise level.
    class_labels : torch.Tensor
        The class labels of shape :math:`(B, \text{label_dim})`. Used for
        conditioning on any vector-valued quantity. Can pass ``None`` when
        ``label_dim`` is 0.
    augment_labels : torch.Tensor, optional, default=None
        The augmentation labels of shape :math:`(B, \text{augment_dim})`. Used
        for conditioning on any additional vector-valued quantity. Can pass
        ``None`` when ``augment_dim`` is 0.

    Outputs
    -------
    torch.Tensor
        The denoised latent state of shape :math:`(B, C_{out}, H_{in}, W_{in})`.


    .. important::
        • The terms *noise levels* (or *noise labels*) are used to refer to the diffusion time-step, as these are conceptually equivalent.
        • The terms *labels* and *classes* originate from the original paper and EDM repository,
          where this architecture was used for class-conditional image generation. While these terms
          suggest class-based conditioning, the architecture can actually be conditioned on any vector-valued
          conditioning.
        • The term *positional embedding* used in the `embedding_type` parameter
          also comes from the original paper and EDM repository. Here,
          *positional* refers to the diffusion time-step, similar to how position is used in transformer
          architectures. Despite the name, these embeddings encode temporal information about the
          diffusion process rather than spatial position information.
        • Limitations on input image resolution: for a model that has :math:`N` levels,
          the latent state :math:`\mathbf{x}` must have resolution that is a multiple of :math:`2^N` in each dimension.
          This is due to a limitation in the decoder that does not support shape mismatch
          in the residual connections from the encoder to the decoder. For images that do not match
          this requirement, it is recommended to interpolate your data on a grid of the required resolution
          beforehand.

    Example
    --------
    >>> model = SongUNet(img_resolution=16, in_channels=2, out_channels=2)
    >>> noise_labels = torch.randn([1])
    >>> class_labels = torch.randint(0, 1, (1, 1))
    >>> input_image = torch.ones([1, 2, 16, 16])
    >>> output_image = model(input_image, noise_labels, class_labels)
    >>> output_image.shape
    torch.Size([1, 2, 16, 16])
    """

    # Arguments of the __init__ method that can be overridden with the
    # ``Module.from_checkpoint`` method.
    _overridable_args: Set[str] = {"use_apex_gn", "act"}

    def __init__(
        self,
        img_resolution: Union[List[int], int],
        in_channels: int,
        out_channels: int,
        label_dim: int = 0,
        augment_dim: int = 0,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 2, 2],
        channel_mult_emb: int = 4,
        num_blocks: int = 4,
        attn_resolutions: List[int] = [16],
        dropout: float = 0.10,
        label_dropout: float = 0.0,
        embedding_type: Literal["fourier", "positional", "zero"] = "positional",
        channel_mult_noise: int = 1,
        encoder_type: Literal["standard", "skip", "residual"] = "standard",
        decoder_type: Literal["standard", "skip"] = "standard",
        resample_filter: List[int] = [1, 1],
        checkpoint_level: int = 0,
        additive_pos_embed: bool = False,
        use_apex_gn: bool = False,
        act: str = "silu",
        profile_mode: bool = False,
        amp_mode: bool = False,
    ):
        valid_embedding_types = ["fourier", "positional", "zero"]
        if embedding_type not in valid_embedding_types:
            raise ValueError(
                f"Invalid embedding_type: {embedding_type}. Must be one of {valid_embedding_types}."
            )

        valid_encoder_types = ["standard", "skip", "residual"]
        if encoder_type not in valid_encoder_types:
            raise ValueError(
                f"Invalid encoder_type: {encoder_type}. Must be one of {valid_encoder_types}."
            )

        valid_decoder_types = ["standard", "skip"]
        if decoder_type not in valid_decoder_types:
            raise ValueError(
                f"Invalid decoder_type: {decoder_type}. Must be one of {valid_decoder_types}."
            )

        super().__init__(meta=MetaData())
        self.label_dropout = label_dropout
        self.embedding_type = embedding_type
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
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
            use_apex_gn=use_apex_gn,
            act=act,
            fused_conv_bias=True,
            profile_mode=profile_mode,
            amp_mode=amp_mode,
        )
        self.profile_mode = profile_mode
        self.amp_mode = amp_mode

        # for compatibility with older versions that took only 1 dimension
        self.img_resolution = img_resolution
        if isinstance(img_resolution, int):
            self.img_shape_y = self.img_shape_x = img_resolution
        else:
            self.img_shape_y = img_resolution[0]
            self.img_shape_x = img_resolution[1]

        # set the threshold for checkpointing based on image resolution
        self.checkpoint_threshold = (self.img_shape_y >> checkpoint_level) + 1

        # Optional additive learned positition embed after the first conv
        self.additive_pos_embed = additive_pos_embed
        if self.additive_pos_embed:
            self.spatial_emb = torch.nn.Parameter(
                torch.randn(1, model_channels, self.img_shape_y, self.img_shape_x)
            )
            torch.nn.init.trunc_normal_(self.spatial_emb, std=0.02)

        # Mapping.
        if self.embedding_type != "zero":
            self.map_noise = (
                PositionalEmbedding(
                    num_channels=noise_channels, endpoint=True, amp_mode=amp_mode
                )
                if embedding_type == "positional"
                else FourierEmbedding(num_channels=noise_channels, amp_mode=amp_mode)
            )
            self.map_label = (
                Linear(
                    in_features=label_dim,
                    out_features=noise_channels,
                    amp_mode=amp_mode,
                    **init,
                )
                if label_dim
                else None
            )
            self.map_augment = (
                Linear(
                    in_features=augment_dim,
                    out_features=noise_channels,
                    bias=False,
                    amp_mode=amp_mode,
                    **init,
                )
                if augment_dim
                else None
            )
            self.map_layer0 = Linear(
                in_features=noise_channels,
                out_features=emb_channels,
                amp_mode=amp_mode,
                **init,
            )
            self.map_layer1 = Linear(
                in_features=emb_channels,
                out_features=emb_channels,
                amp_mode=amp_mode,
                **init,
            )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = self.img_shape_y >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin,
                    out_channels=cout,
                    kernel=3,
                    fused_conv_bias=True,
                    amp_mode=amp_mode,
                    **init,
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = Conv2d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                        amp_mode=amp_mode,
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=1,
                        fused_conv_bias=True,
                        amp_mode=amp_mode,
                        **init,
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        fused_conv_bias=True,
                        amp_mode=amp_mode,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = self.img_shape_y >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                        amp_mode=amp_mode,
                    )
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(
                    num_channels=cout,
                    eps=1e-6,
                    use_apex_gn=use_apex_gn,
                    amp_mode=amp_mode,
                )
                self.dec[f"{res}x{res}_aux_conv"] = Conv2d(
                    in_channels=cout,
                    out_channels=out_channels,
                    kernel=3,
                    fused_conv_bias=True,
                    amp_mode=amp_mode,
                    **init_zero,
                )

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        with (
            nvtx.annotate(message="SongUNet", color="blue")
            if self.profile_mode
            else contextlib.nullcontext()
        ):
            if self.embedding_type != "zero":
                # Mapping.
                emb = self.map_noise(noise_labels)
                emb = (
                    emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
                )  # swap sin/cos
                if self.map_label is not None:
                    tmp = class_labels
                    if self.training and self.label_dropout:
                        tmp = tmp * (
                            torch.rand([x.shape[0], 1], device=x.device)
                            >= self.label_dropout
                        ).to(tmp.dtype)
                    emb = emb + self.map_label(
                        tmp * np.sqrt(self.map_label.in_features)
                    )
                if self.map_augment is not None and augment_labels is not None:
                    emb = emb + self.map_augment(augment_labels)
                emb = silu(self.map_layer0(emb))
                emb = silu(self.map_layer1(emb))
            else:
                emb = torch.zeros(
                    (noise_labels.shape[0], self.emb_channels), device=x.device
                )

            # Encoder.
            skips = []
            aux = x
            for name, block in self.enc.items():
                with (
                    nvtx.annotate(f"SongUNet encoder: {name}", color="blue")
                    if self.profile_mode
                    else contextlib.nullcontext()
                ):
                    if "aux_down" in name:
                        aux = block(aux)
                    elif "aux_skip" in name:
                        x = skips[-1] = x + block(aux)
                    elif "aux_residual" in name:
                        x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
                    elif "_conv" in name:
                        x = block(x)
                        if self.additive_pos_embed:
                            x = x + self.spatial_emb.to(dtype=x.dtype)
                        skips.append(x)
                    else:
                        # For UNetBlocks check if we should use gradient checkpointing
                        if isinstance(block, UNetBlock):
                            if x.shape[-1] > self.checkpoint_threshold:
                                # self.checkpoint = checkpoint?
                                # else: self.checkpoint  = lambda(block,x,emb:block(x,emb))
                                x = checkpoint(block, x, emb)
                            else:
                                # AssertionError: Only support NHWC layout.
                                x = block(x, emb)
                        else:
                            x = block(x)
                        skips.append(x)

            # Decoder.
            aux = None
            tmp = None
            for name, block in self.dec.items():
                with (
                    nvtx.annotate(f"SongUNet decoder: {name}", color="blue")
                    if self.profile_mode
                    else contextlib.nullcontext()
                ):
                    if "aux_up" in name:
                        aux = block(aux)
                    elif "aux_norm" in name:
                        tmp = block(x)
                    elif "aux_conv" in name:
                        tmp = block(silu(tmp))
                        aux = tmp if aux is None else tmp + aux
                    else:
                        if x.shape[1] != block.in_channels:
                            x = torch.cat([x, skips.pop()], dim=1)
                        # check for checkpointing on decoder blocks and up sampling blocks
                        if (
                            x.shape[-1] > self.checkpoint_threshold and "_block" in name
                        ) or (
                            x.shape[-1] > (self.checkpoint_threshold / 2)
                            and "_up" in name
                        ):
                            x = checkpoint(block, x, emb)
                        else:
                            x = block(x, emb)
            return aux


# ------------------------------------------------------------------------------
# Specialized architectures
# ------------------------------------------------------------------------------


class SongUNetPosEmbd(SongUNet):
    r"""This specialized architecture extends
    :class:`~physicsnemo.models.diffusion.song_unet.SongUNet` with positional
    embeddings that encode global spatial coordinates of the pixels.

    This model supports the same type of conditioning as the base SongUNet, and
    can be in addition conditioned on the positional embeddings. Conditioning on
    the positional embeddings is performed with a channel-wise concatenation to
    the input image before the first layer of the U-Net. Multiple types of
    positional embeddings are supported. Positional embeddings are represented by
    a 2D grid of shape :math:`(C_{PE}, H, W)`, where :math:`H` and
    :math:`W` correspond to the ``img_resolution`` parameter.

    The following types of positional embeddings are
    supported:

    • learnable: uses a 2D grid of learnable parameters.

    • linear: uses a 2D rectilinear grid over the domain :math:`[-1, 1] \times
      [-1, 1]`.

    • sinusoidal: uses sinusoidal functions of the spatial coordinates, with
      possibly multiple frequency bands.

    • test: uses a 2D grid of integer indices, only used for testing.

    When the input image spatial resolution is smaller than the global
    positional embeddings, it is necessary to select a subset (or *patch*) of the embedding
    grid that correspond to the spatial locations of the input image pixels. The
    model provides two methods for selecting the subset of positional
    embeddings:

    1. Using a selector function. See :meth:`positional_embedding_selector` for
       details.

    2. Using global indices. See :meth:`positional_embedding_indexing` for
       details.

    If none of these are provided, the entire grid of positional embeddings is
    used and channel-wise concatenated to the input image.

    Most parameters are the same as in the parent class
    :class:`~physicsnemo.models.diffusion.song_unet.SongUNet`. Only the ones
    that differ are listed below.

    Parameters
    ----------
    img_resolution : Union[List[int, int], int]
        The resolution of the input/output image. Can be a single int for
        square images or a list :math:`[H, W]` for rectangular images.
        Used to set the resolution of the positional embedding grid. It must
        correspond to the spatial resolution of the *global* domain/image.
    in_channels : int
        Number of channels :math:`C_{in} + C_{PE}`, where :math:`C_{in}` is the
        number of channels in the image passed to the U-Net and :math:`C_{PE}`
        is the number of channels in the positional embedding grid.

        **Important:** in comparison to the base
        :class:`~physicsnemo.models.diffusion.song_unet.SongUNet`, this
        parameter should also include the number of channels in the positional
        embedding grid :math:`C_{PE}`.
    gridtype : Literal["sinusoidal", "learnable", "linear", "test"], optional, default="sinusoidal"
        Type of positional embedding to use. Controls how spatial pixels locations are encoded.
    N_grid_channels : int, optional, default=4
        Number of channels :math:`C_{PE}` in the positional embedding grid. For 'sinusoidal' must be 4 or
        multiple of 4. For 'linear' and 'test' must be 2. For 'learnable' can be any
        value.
    lead_time_mode : bool, optional, default=False
        Provided for convenience. It is recommended to use the architecture
        :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosLtEmbd`
        for a lead-time aware model.
    lead_time_channels : int, optional, default=None
        Provided for convenience. Refer to :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosLtEmbd`.
    lead_time_steps : int, optional, default=9
        Provided for convenience. Refer to :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosLtEmbd`.
    prob_channels : List[int], optional, default=[]
        Provided for convenience. Refer to :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosLtEmbd`.


    Forward
    -------
    x : torch.Tensor
        The input image of shape :math:`(B, C_{in}, H_{in}, W_{in})`,
        where :math:`H_{in}` and :math:`W_{in}` are the spatial dimensions of
        the input image (does not need to be the full image).
        In general ``x`` is the channel-wise concatenation of the latent state :math:`\mathbf{x}` and
        additional images used for conditioning. For an unconditional model, ``x``
        is simply the latent state :math:`\mathbf{x}`.

        *Note:* :math:`H_{in}` and :math:`W_{in}` do not need to match the
        ``img_resolution`` parameter, except when ``additive_pos_embed`` is ``True``.
        In all other cases, the resolution of ``x`` must be smaller than
        ``img_resolution``.
    noise_labels : torch.Tensor
        The noise labels of shape :math:`(B,)`. Used for conditioning on
        the diffusion noise level.
    class_labels : torch.Tensor
        The class labels of shape :math:`(B, \text{label_dim})`. Used for
        conditioning on any vector-valued quantity. Can pass ``None`` when
        ``label_dim`` is 0.
    global_index : torch.Tensor, optional, default=None
        The global indices of the positional embeddings to use. If neither
        ``global_index`` nor ``embedding_selector`` are provided, the entire
        positional embedding grid of shape :math:`(C_{PE}, H, W)` is used.
        In this case ``x`` must have the same spatial resolution as the positional
        embedding grid. See :meth:`positional_embedding_indexing` for details.
    embedding_selector : Callable, optional, default=None
        A function that selects the positional embeddings to use. See
        :meth:`positional_embedding_selector` for details.
    augment_labels : torch.Tensor, optional, default=None
        The augmentation labels of shape :math:`(B, \text{augment_dim})`. Used
        for conditioning on any additional vector-valued quantity. Can pass
        ``None`` when ``augment_dim`` is 0.

    Outputs
    -------
    torch.Tensor
        The output tensor of shape :math:`(B, C_{out}, H_{in}, W_{in})`.


    .. important::
        Unlike positional embeddings defined by ``embedding_type`` in the parent
        class :class:`~physicsnemo.models.diffusion.song_unet.SongUNet` that
        encode the diffusion time-step (or noise level), the positional embeddings in this
        specialized architecture encode global spatial coordinates of the
        pixels.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.diffusion.song_unet import SongUNetPosEmbd
    >>> from physicsnemo.utils.patching import GridPatching2D
    >>>
    >>> # Model initialization - in_channels must include both original input channels (2)
    >>> # and the positional embedding channels (N_grid_channels=4 by default)
    >>> model = SongUNetPosEmbd(img_resolution=16, in_channels=2+4, out_channels=2)
    >>> noise_labels = torch.randn([1])
    >>> class_labels = torch.randint(0, 1, (1, 1))
    >>> # The input has only the original 2 channels - positional embeddings are
    >>> # added automatically inside the forward method
    >>> input_image = torch.ones([1, 2, 16, 16])
    >>> output_image = model(input_image, noise_labels, class_labels)
    >>> output_image.shape
    torch.Size([1, 2, 16, 16])
    >>>
    >>> # Using a global index to select all positional embeddings
    >>> patching = GridPatching2D(img_shape=(16, 16), patch_shape=(16, 16))
    >>> global_index = patching.global_index(batch_size=1)
    >>> output_image = model(
    ...     input_image, noise_labels, class_labels,
    ...     global_index=global_index
    ... )
    >>> output_image.shape
    torch.Size([1, 2, 16, 16])
    >>>
    >>> # Using a custom embedding selector to select all positional embeddings
    >>> def patch_embedding_selector(emb):
    ...     return patching.apply(emb[None].expand(1, -1, -1, -1))
    >>> output_image = model(
    ...     input_image, noise_labels, class_labels,
    ...     embedding_selector=patch_embedding_selector
    ... )
    >>> output_image.shape
    torch.Size([1, 2, 16, 16])
    """

    def __init__(
        self,
        img_resolution: Union[List[int], int],
        in_channels: int,
        out_channels: int,
        label_dim: int = 0,
        augment_dim: int = 0,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 2, 2, 2],
        channel_mult_emb: int = 4,
        num_blocks: int = 4,
        attn_resolutions: List[int] = [28],
        dropout: float = 0.13,
        label_dropout: float = 0.0,
        embedding_type: str = "positional",
        channel_mult_noise: int = 1,
        encoder_type: str = "standard",
        decoder_type: str = "standard",
        resample_filter: List[int] = [1, 1],
        gridtype: Literal["sinusoidal", "learnable", "linear", "test"] = "sinusoidal",
        N_grid_channels: int = 4,
        checkpoint_level: int = 0,
        additive_pos_embed: bool = False,
        use_apex_gn: bool = False,
        act: str = "silu",
        profile_mode: bool = False,
        amp_mode: bool = False,
        lead_time_mode: bool = False,
        lead_time_channels: int = None,
        lead_time_steps: int = 9,
        prob_channels: List[int] = [],
    ):
        super().__init__(
            img_resolution,
            in_channels,
            out_channels,
            label_dim,
            augment_dim,
            model_channels,
            channel_mult,
            channel_mult_emb,
            num_blocks,
            attn_resolutions,
            dropout,
            label_dropout,
            embedding_type,
            channel_mult_noise,
            encoder_type,
            decoder_type,
            resample_filter,
            checkpoint_level,
            additive_pos_embed,
            use_apex_gn,
            act,
            profile_mode,
            amp_mode,
        )

        self.gridtype = gridtype
        self.N_grid_channels = N_grid_channels
        if self.gridtype == "learnable":
            self.pos_embd = self._get_positional_embedding()
        else:
            self.register_buffer("pos_embd", self._get_positional_embedding().float())
        self.lead_time_mode = lead_time_mode
        if self.lead_time_mode:
            self.lead_time_channels = lead_time_channels
            self.lead_time_steps = lead_time_steps
            self.lt_embd = self._get_lead_time_embedding()
            self.prob_channels = prob_channels
            if self.prob_channels:
                self.scalar = torch.nn.Parameter(
                    torch.ones((1, len(self.prob_channels), 1, 1))
                )

    def forward(
        self,
        x,
        noise_labels,
        class_labels,
        global_index: Optional[torch.Tensor] = None,
        embedding_selector: Optional[Callable] = None,
        augment_labels=None,
        lead_time_label=None,
    ):
        with (
            nvtx.annotate(message="SongUNetPosEmbd", color="blue")
            if self.profile_mode
            else contextlib.nullcontext()
        ):
            if embedding_selector is not None and global_index is not None:
                raise ValueError(
                    "Cannot provide both embedding_selector and global_index."
                )

            if x.dtype != self.pos_embd.dtype:
                self.pos_embd = self.pos_embd.to(x.dtype)

            # Append positional embedding to input conditioning
            if self.pos_embd is not None:
                # Select positional embeddings with a selector function
                if embedding_selector is not None:
                    selected_pos_embd = self.positional_embedding_selector(
                        x, embedding_selector, lead_time_label=lead_time_label
                    )
                # Select positional embeddings using global indices (selects all
                # embeddings if global_index is None)
                else:
                    selected_pos_embd = self.positional_embedding_indexing(
                        x, global_index=global_index, lead_time_label=lead_time_label
                    )
                x = torch.cat((x, selected_pos_embd), dim=1)

            out = super().forward(x, noise_labels, class_labels, augment_labels)

            if self.lead_time_mode:
                # if training mode, let crossEntropyLoss do softmax. The model outputs logits.
                # if eval mode, the model outputs probability
                if self.prob_channels and out.dtype != self.scalar.dtype:
                    self.scalar.data = self.scalar.data.to(out.dtype)
                if self.prob_channels and (not self.training):
                    out[:, self.prob_channels] = (
                        out[:, self.prob_channels] * self.scalar
                    ).softmax(dim=1)
                elif self.prob_channels and self.training:
                    out[:, self.prob_channels] = (
                        out[:, self.prob_channels] * self.scalar
                    )
            return out

    def positional_embedding_indexing(
        self,
        x: torch.Tensor,
        global_index: Optional[torch.Tensor] = None,
        lead_time_label=None,
    ) -> torch.Tensor:
        r"""Select positional embeddings using global indices.

        This method uses global indices to select specific subset of the
        positional embedding grid (called *patches*). If no indices are provided,
        the entire positional embedding grid is returned.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(P \times B, C, H_{in}, W_{in})`.
            Only used to determine batch size :math:`B` and device.
        global_index : Optional[torch.Tensor], default=None
            Tensor of shape :math:`(P, 2, H_{in}, W_{in})` that correspond to
            the patches to extract from the positional embedding grid.
            :math:`P` is the number of distinct patches in the input tensor ``x``.
            The channel dimension should contain :math:`j`, :math:`i` indices that
            should represent the indices of the pixels to extract from the embedding grid.

        Returns
        -------
        torch.Tensor
            Selected positional embeddings with shape :math:`(P \times B, C_{PE}, H_{in}, W_{in})`
            (same spatial resolution as ``global_index``) if ``global_index`` is provided.
            If ``global_index`` is None, the entire positional embedding grid
            is duplicated :math:`B` times and returned with shape :math:`(B, C_{PE}, H, W)`.

        Example
        -------
        >>> # Create global indices using patching utility:
        >>> from physicsnemo.utils.patching import GridPatching2D
        >>> patching = GridPatching2D(img_shape=(16, 16), patch_shape=(8, 8))
        >>> global_index = patching.global_index(batch_size=3)
        >>> print(global_index.shape)
        torch.Size([4, 2, 8, 8])

        Notes
        -----
            - This method is typically used in patch-based diffusion (or
              multi-diffusion), where a large input image is split into
              multiple patches. The batch dimension of the input tensor contains the patches.
              Patches are processed independently by the model, and the ``global_index`` parameter
              is used to select the grid of positional embeddings corresponding
              to each patch.
            - See this method from :class:`physicsnemo.utils.patching.BasePatching2D`
              for generating the ``global_index`` parameter:
              :meth:`~physicsnemo.utils.patching.BasePatching2D.global_index`.
        """
        # If no global indices are provided, select all embeddings and expand
        # to match the batch size of the input
        if x.dtype != self.pos_embd.dtype:
            self.pos_embd = self.pos_embd.to(x.dtype)

        if global_index is None:
            if self.lead_time_mode:
                selected_pos_embd = []
                if self.pos_embd is not None:
                    selected_pos_embd.append(
                        self.pos_embd[None].expand((x.shape[0], -1, -1, -1))
                    )
                if self.lt_embd is not None:
                    selected_pos_embd.append(
                        torch.reshape(
                            self.lt_embd[lead_time_label.int()],
                            (
                                x.shape[0],
                                self.lead_time_channels,
                                self.img_shape_y,
                                self.img_shape_x,
                            ),
                        )
                    )
                if len(selected_pos_embd) > 0:
                    selected_pos_embd = torch.cat(selected_pos_embd, dim=1)
            else:
                selected_pos_embd = self.pos_embd[None].expand(
                    (x.shape[0], -1, -1, -1)
                )  # (B, C_{PE}, H, W)

        else:
            P = global_index.shape[0]
            B = x.shape[0] // P
            H = global_index.shape[2]
            W = global_index.shape[3]

            global_index = torch.reshape(
                torch.permute(global_index, (1, 0, 2, 3)), (2, -1)
            )  # (P, 2, X, Y) to (2, P*X*Y)
            selected_pos_embd = self.pos_embd[
                :, global_index[0], global_index[1]
            ]  # (C_pe, P*X*Y)
            selected_pos_embd = torch.permute(
                torch.reshape(selected_pos_embd, (self.pos_embd.shape[0], P, H, W)),
                (1, 0, 2, 3),
            )  # (P, C_pe, X, Y)

            selected_pos_embd = selected_pos_embd.repeat(
                B, 1, 1, 1
            )  # (B*P, C_pe, X, Y)

            # Append positional and lead time embeddings to input conditioning
            if self.lead_time_mode:
                embeds = []
                if self.pos_embd is not None:
                    embeds.append(selected_pos_embd)  # reuse code below
                if self.lt_embd is not None:
                    lt_embds = self.lt_embd[
                        lead_time_label.int()
                    ]  # (B, self.lead_time_channels, self.img_shape_y, self.img_shape_x),

                    selected_lt_pos_embd = lt_embds[
                        :, :, global_index[0], global_index[1]
                    ]  # (B, C_lt, P*X*Y)
                    selected_lt_pos_embd = torch.reshape(
                        torch.permute(
                            torch.reshape(
                                selected_lt_pos_embd,
                                (B, self.lead_time_channels, P, H, W),
                            ),
                            (0, 2, 1, 3, 4),
                        ).contiguous(),
                        (B * P, self.lead_time_channels, H, W),
                    )  # (B*P, C_pe, X, Y)
                    embeds.append(selected_lt_pos_embd)

                if len(embeds) > 0:
                    selected_pos_embd = torch.cat(embeds, dim=1)

        return selected_pos_embd

    def positional_embedding_selector(
        self,
        x: torch.Tensor,
        embedding_selector: Callable[[torch.Tensor], torch.Tensor],
        lead_time_label=None,
    ) -> torch.Tensor:
        r"""Select positional embeddings using a selector function.

        Similar to :meth:`positional_embedding_indexing`, but instead uses a selector
        function to select the embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(P \times B, C, H_{in}, W_{in})`.
            Only used to determine batch size :math:`B`, dtype and device.
        embedding_selector : Callable
            Function that takes as input the entire embedding grid of shape
            :math:`(C_{PE}, H, W)` and returns selected embeddings with shape
            :math:`(P \times B, C_{PE}, H_{in}, W_{in})`.
            Each selected embedding should correspond to the portion of the embedding grid
            that corresponds to the batch element in ``x``.
            Typically this should be based on
            :meth:`physicsnemo.utils.patching.BasePatching2D.apply` method to
            maintain consistency with patch extraction.
        lead_time_label : Optional[torch.Tensor], default=None
            Tensor of shape :math:`(P,)` that corresponds to the lead-time label for each patch.
            Only used if ``lead_time_mode`` is True.

        Returns
        -------
        torch.Tensor
            A tensor of shape :math:`(P \times B, C_{PE} [+ C_{LT}], H_{in}, W_{in})`.
            :math:`C_{PE}` is the number of embedding channels in the positional embedding grid,
            and :math:`C_{LT}` is the number of embedding channels in the lead-time embedding grid.
            If ``lead_time_label`` is provided, the lead-time embedding channels are included.

        Notes
        -----
            - This method is typically used in patch-based diffusion (or
              multi-diffusion), where a large input image is split into
              multiple patches. The batch dimension of the input tensor contains the patches.
              Patches are processed independently by the model, and the ``embedding_selector`` function is used
              to select the grid of positional embeddings corresponding to each
              patch.
            - See this method from :class:`physicsnemo.utils.patching.BasePatching2D`
              for generating the ``embedding_selector`` parameter:
              :meth:`~physicsnemo.utils.patching.BasePatching2D.apply`

        Example
        -------
        >>> # Define a selector function with a patching utility:
        >>> from physicsnemo.utils.patching import GridPatching2D
        >>> patching = GridPatching2D(img_shape=(16, 16), patch_shape=(8, 8))
        >>> batch_size = 4
        >>> def embedding_selector(emb):
        ...     return patching.apply(emb[None].expand(batch_size, -1, -1, -1))
        >>>
        """
        if x.dtype != self.pos_embd.dtype:
            self.pos_embd = self.pos_embd.to(x.dtype)
        if lead_time_label is not None:
            # all patches share same lead_time_label
            embeddings = torch.cat(
                [self.pos_embd, self.lt_embd[lead_time_label[0].int()]]
            )
        else:
            embeddings = self.pos_embd
        return embedding_selector(embeddings)  # (B, N_pe, H, W)

    def _get_positional_embedding(self):
        if self.N_grid_channels == 0:
            return None
        elif self.gridtype == "learnable":
            grid = torch.nn.Parameter(
                torch.randn(self.N_grid_channels, self.img_shape_y, self.img_shape_x)
            )  # (N_grid_channels, img_shape_y, img_shape_x)
        elif self.gridtype == "linear":
            if self.N_grid_channels != 2:
                raise ValueError("N_grid_channels must be set to 2 for gridtype linear")
            x = np.meshgrid(np.linspace(-1, 1, self.img_shape_y))
            y = np.meshgrid(np.linspace(-1, 1, self.img_shape_x))
            grid_x, grid_y = np.meshgrid(y, x)
            grid = torch.from_numpy(
                np.stack((grid_x, grid_y), axis=0)
            )  # (2, img_shape_y, img_shape_x)
            grid.requires_grad = False
        elif self.gridtype == "sinusoidal" and self.N_grid_channels == 4:
            # print('sinusuidal grid added ......')
            x1 = np.meshgrid(np.sin(np.linspace(0, 2 * np.pi, self.img_shape_y)))
            x2 = np.meshgrid(np.cos(np.linspace(0, 2 * np.pi, self.img_shape_y)))
            y1 = np.meshgrid(np.sin(np.linspace(0, 2 * np.pi, self.img_shape_x)))
            y2 = np.meshgrid(np.cos(np.linspace(0, 2 * np.pi, self.img_shape_x)))
            grid_x1, grid_y1 = np.meshgrid(y1, x1)
            grid_x2, grid_y2 = np.meshgrid(y2, x2)
            grid = torch.squeeze(
                torch.from_numpy(
                    np.expand_dims(
                        np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis=0), axis=0
                    )
                )
            )  # (4, img_shape_y, img_shape_x)
            grid.requires_grad = False
        elif self.gridtype == "sinusoidal" and self.N_grid_channels != 4:
            if self.N_grid_channels % 4 != 0:
                raise ValueError("N_grid_channels must be a factor of 4")
            num_freq = self.N_grid_channels // 4
            freq_bands = 2.0 ** np.linspace(0.0, num_freq, num=num_freq)
            grid_list = []
            grid_x, grid_y = np.meshgrid(
                np.linspace(0, 2 * np.pi, self.img_shape_x),
                np.linspace(0, 2 * np.pi, self.img_shape_y),
            )
            for freq in freq_bands:
                for p_fn in [np.sin, np.cos]:
                    grid_list.append(p_fn(grid_x * freq))
                    grid_list.append(p_fn(grid_y * freq))
            grid = torch.from_numpy(
                np.stack(grid_list, axis=0)
            )  # (N_grid_channels, img_shape_y, img_shape_x)
            grid.requires_grad = False
        elif self.gridtype == "test" and self.N_grid_channels == 2:
            idx_x = torch.arange(self.img_shape_y)
            idx_y = torch.arange(self.img_shape_x)
            mesh_x, mesh_y = torch.meshgrid(idx_x, idx_y)
            grid = torch.stack((mesh_x, mesh_y), dim=0)  # (2, img_shape_y, img_shape_x)
        else:
            raise ValueError("Gridtype not supported.")
        return grid

    def _get_lead_time_embedding(self):
        if (self.lead_time_steps is None) or (self.lead_time_channels is None):
            return None
        grid = torch.nn.Parameter(
            torch.randn(
                self.lead_time_steps,
                self.lead_time_channels,
                self.img_shape_y,
                self.img_shape_x,
            )
        )  # (lead_time_steps, lead_time_channels, img_shape_y, img_shape_x)
        return grid


# TODO: the entire logic of the lead-time logic should be moved there. We
# should use subclass of the SongUNetPosEmbd class and specialize it for
# lead-time aware embeddings.
class SongUNetPosLtEmbd(SongUNetPosEmbd):
    r"""
    This specialized architecture extends
    :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd` with two
    additional capabilities:

    1. The model can be conditioned on lead-time labels. These labels encode
       *physical* time information, such as a forecasting horizon.

    2. Similarly to the parent ``SongUNetPosEmbd``, this model predicts
       regression targets, but it can also produce classification predictions.
        More precisely, some of the ouput channels are probability outputs, that
        are passed through a softmax activation function. This is useful for
        multi-task applications, where the objective is a combination of both
        regression and classification losses.

    The mechanism to condition on lead-time labels is implemented by:

    • First generating a grid of learnable lead-time embeddings of shape
      :math:`(\text{lead_time_steps}, C_{LT}, H, W)`. The spatial resolution of
      the lead-time embeddings is the same as the input/output image.

    • Then, given an input ``x``, select the lead-time embeddings that
      corresponds to the lead-times associated with the samples in the input
      ``x``.

    • Finally, concatenate channels-wise the selected lead-time embeddings and
      positional embeddings to the input ``x`` and pass them to the U-Net network.

    Most parameters are similar to the parent
    :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd`, at the
    exception of the ones listed below.

    Parameters
    -----------
    in_channels : int
        Number of channels :math:`C_{in} + C_{PE} + C_{LT}` in the image passed to the U-Net.

        *Important:* in comparison to the base :class:`~physicsnemo.models.diffusion.song_unet.SongUNet`,
        this parameter should also include the number of channels in the positional embedding grid
        :math:`C_{PE}` and the number of channels in the lead-time embedding grid
        :math:`C_{LT}`.
    lead_time_channels : int, optional, default=None
        Number of channels :math:`C_{LT}` in the lead time embedding. These are
        learned embeddings that encode *physical* time information.
    lead_time_steps : int, optional, default=9
        Number of discrete lead time steps to support. Each step gets its own
        learned embedding vector of shape :math:`(C_{LT}, H, W)`.
    prob_channels : List[int], optional, default=[]
        Indices of channels that are probability outputs (or *classification* predictions),
        In training mode, the model outputs logits for these probability
        channels, and in eval mode, the model applies a softmax to outputs the probabilities.

    Forward
    -------
    x : torch.Tensor
        The input image of shape :math:`(B, C_{in}, H_{in}, W_{in})`,
        where :math:`H_{in}` and :math:`W_{in}` are the spatial dimensions of
        the input image (does not need to be the full image).
    noise_labels : torch.Tensor
        The noise labels of shape :math:`(B,)`. Used for conditioning on
        the diffusion noise level.
    class_labels : torch.Tensor
        The class labels of shape :math:`(B, \text{label_dim})`. Used for
        conditioning on any vector-valued quantity. Can pass ``None`` when
        ``label_dim`` is 0.
    global_index : torch.Tensor, optional, default=None
        The global indices of the positional embeddings to use. See
        :meth:`positional_embedding_indexing` for details. If neither
        ``global_index`` nor ``embedding_selector`` are provided, the entire
        positional embedding grid is used.
    embedding_selector : Callable, optional, default=None
        A function that selects the positional embeddings to use. See
        :meth:`positional_embedding_selector` for details.
    augment_labels : torch.Tensor, optional, default=None
        The augmentation labels of shape :math:`(B, \text{augment_dim})`. Used
        for conditioning on any additional vector-valued quantity.
    lead_time_label : torch.Tensor, optional, default=None
        The lead-time labels of shape :math:`(B,)`. Used for selecting
        lead-time embeddings. It should contain the indices of the lead-time
        embeddings that correspond to the lead-time of each sample in the batch.

    Outputs
    -------
    torch.Tensor
        The output tensor of shape :math:`(B, C_{out}, H_{in}, W_{in})`.

    Notes
    -----
        - The lead-time embeddings differ from the diffusion time embeddings used in
          :class:`~physicsnemo.models.diffusion.song_unet.SongUNet` class, as they do not
          encode diffusion time-step but *physical forecast time*.

    Example
    --------
    >>> import torch
    >>> from physicsnemo.models.diffusion.song_unet import SongUNetPosLtEmbd
    >>> from physicsnemo.utils.patching import GridPatching2D
    >>>
    >>> # Model initialization - in_channels must include original input channels (2),
    >>> # positional embedding channels (N_grid_channels=4 by default) and
    >>> # lead time embedding channels (4)
    >>> model = SongUNetPosLtEmbd(
    ...     img_resolution=16, in_channels=2+4+4, out_channels=2,
    ...     lead_time_channels=4, lead_time_steps=9
    ... )
    >>> noise_labels = torch.randn([1])
    >>> class_labels = torch.randint(0, 1, (1, 1))
    >>> # The input has only the original 2 channels - positional embeddings and
    >>> # lead time embeddings are added automatically inside the forward method
    >>> input_image = torch.ones([1, 2, 16, 16])
    >>> lead_time_label = torch.tensor([3])
    >>> output_image = model(
    ...     input_image, noise_labels, class_labels,
    ...     lead_time_label=lead_time_label
    ... )
    >>> output_image.shape
    torch.Size([1, 2, 16, 16])
    >>>
    >>> # Using global_index to select all the positional and lead time embeddings
    >>> patching = GridPatching2D(img_shape=(16, 16), patch_shape=(16, 16))
    >>> global_index = patching.global_index(batch_size=1)
    >>> output_image = model(
    ...     input_image, noise_labels, class_labels,
    ...     lead_time_label=lead_time_label,
    ...     global_index=global_index
    ... )
    >>> output_image.shape
    torch.Size([1, 2, 16, 16])

    """

    def __init__(
        self,
        img_resolution: Union[List[int], int],
        in_channels: int,
        out_channels: int,
        label_dim: int = 0,
        augment_dim: int = 0,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 2, 2, 2],
        channel_mult_emb: int = 4,
        num_blocks: int = 4,
        attn_resolutions: List[int] = [28],
        dropout: float = 0.13,
        label_dropout: float = 0.0,
        embedding_type: str = "positional",
        channel_mult_noise: int = 1,
        encoder_type: str = "standard",
        decoder_type: str = "standard",
        resample_filter: List[int] = [1, 1],
        gridtype: str = "sinusoidal",
        N_grid_channels: int = 4,
        lead_time_channels: int = None,
        lead_time_steps: int = 9,
        prob_channels: List[int] = [],
        checkpoint_level: int = 0,
        additive_pos_embed: bool = False,
        use_apex_gn: bool = False,
        act: str = "silu",
        profile_mode: bool = False,
        amp_mode: bool = False,
    ):
        super().__init__(
            img_resolution,
            in_channels,
            out_channels,
            label_dim,
            augment_dim,
            model_channels,
            channel_mult,
            channel_mult_emb,
            num_blocks,
            attn_resolutions,
            dropout,
            label_dropout,
            embedding_type,
            channel_mult_noise,
            encoder_type,
            decoder_type,
            resample_filter,
            gridtype,
            N_grid_channels,
            checkpoint_level,
            additive_pos_embed,
            use_apex_gn,
            act,
            profile_mode,
            amp_mode,
            True,  # Note: lead_time_mode=True is enforced here
            lead_time_channels,
            lead_time_steps,
            prob_channels,
        )

    def forward(
        self,
        x,
        noise_labels,
        class_labels,
        lead_time_label=None,
        global_index: Optional[torch.Tensor] = None,
        embedding_selector: Optional[Callable] = None,
        augment_labels=None,
    ):
        return super().forward(
            x=x,
            noise_labels=noise_labels,
            class_labels=class_labels,
            global_index=global_index,
            embedding_selector=embedding_selector,
            augment_labels=augment_labels,
            lead_time_label=lead_time_label,
        )

    # Nothing else is re-implemented, because everything is already in the
    # parent SongUNetPosEmb
