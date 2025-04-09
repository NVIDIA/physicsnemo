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

"""
Model architectures used in the paper "Elucidating the Design Space of 
Diffusion-Based Generative Models".
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

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
    """
    Reimplementation of the DDPM++ and NCSN++ architectures, U-Net variants with
    optional self-attention, embeddings, and encoder-decoder components.

    This model supports conditional and unconditional setups, as well as several
    options for various internal architectural choices such as encoder and decoder
    type, embedding type, etc., making it flexible and adaptable to different tasks
    and configurations.

    Parameters
    -----------
    img_resolution : Union[List[int], int]
        The resolution of the input/output image. Can be a single int for square images
        or a list [height, width] for rectangular images.
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels in the output image.
    label_dim : int, optional
        Number of class labels; 0 indicates an unconditional model. By default 0.
    augment_dim : int, optional
        Dimensionality of augmentation labels; 0 means no augmentation. By default 0.
    model_channels : int, optional
        Base multiplier for the number of channels across the network. By default 128.
    channel_mult : List[int], optional
        Per-resolution multipliers for the number of channels. By default [1,2,2,2].
    channel_mult_emb : int, optional
        Multiplier for the dimensionality of the embedding vector. By default 4.
    num_blocks : int, optional
        Number of residual blocks per resolution. By default 4.
    attn_resolutions : List[int], optional
        Resolutions at which self-attention layers are applied. By default [16].
    dropout : float, optional
        Dropout probability applied to intermediate activations. By default 0.10.
    label_dropout : float, optional
        Dropout probability of class labels for classifier-free guidance. By default 0.0.
    embedding_type : str, optional
        Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++, 'zero' for none.
        By default 'positional'.
    channel_mult_noise : int, optional
        Timestep embedding size: 1 for DDPM++, 2 for NCSN++. By default 1.
    encoder_type : str, optional
        Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++, 'skip' for skip connections.
        By default 'standard'.
    decoder_type : str, optional
        Decoder architecture: 'standard' or 'skip' for skip connections. By default 'standard'.
    resample_filter : List[int], optional
        Resampling filter coefficients: [1,1] for DDPM++, [1,3,3,1] for NCSN++. By default [1,1].
    checkpoint_level : int, optional
        Number of layers that should use gradient checkpointing (0 disables checkpointing).
        Higher values trade memory for computation. By default 0.
    additive_pos_embed : bool, optional
        If True, adds a learned positional embedding after the first convolution layer.
        Used in StormCast model. By default False.

    Reference
    ----------
    Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.

    Note
    -----
    Equivalent to the original implementation by Song et al., available at
    https://github.com/yang-song/score_sde_pytorch

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
        embedding_type: str = "positional",
        channel_mult_noise: int = 1,
        encoder_type: str = "standard",
        decoder_type: str = "standard",
        resample_filter: List[int] = [1, 1],
        checkpoint_level: int = 0,
        additive_pos_embed: bool = False,
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
        )

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
                PositionalEmbedding(num_channels=noise_channels, endpoint=True)
                if embedding_type == "positional"
                else FourierEmbedding(num_channels=noise_channels)
            )
            self.map_label = (
                Linear(in_features=label_dim, out_features=noise_channels, **init)
                if label_dim
                else None
            )
            self.map_augment = (
                Linear(
                    in_features=augment_dim,
                    out_features=noise_channels,
                    bias=False,
                    **init,
                )
                if augment_dim
                else None
            )
            self.map_layer0 = Linear(
                in_features=noise_channels, out_features=emb_channels, **init
            )
            self.map_layer1 = Linear(
                in_features=emb_channels, out_features=emb_channels, **init
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
                    in_channels=cin, out_channels=cout, kernel=3, **init
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
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
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
                    )
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(
                    num_channels=cout, eps=1e-6
                )
                self.dec[f"{res}x{res}_aux_conv"] = Conv2d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
                )

    @nvtx.annotate(message="SongUNet", color="blue")
    def forward(self, x, noise_labels, class_labels, augment_labels=None):
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
                emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
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
            with nvtx.annotate(f"SongUNet encoder: {name}", color="blue"):
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
                            x = checkpoint(block, x, emb, use_reentrant=False)
                        else:
                            x = block(x, emb)
                    else:
                        x = block(x)
                    skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            with nvtx.annotate(f"SongUNet decoder: {name}", color="blue"):
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
                        x.shape[-1] > (self.checkpoint_threshold / 2) and "_up" in name
                    ):
                        x = checkpoint(block, x, emb, use_reentrant=False)
                    else:
                        x = block(x, emb)
        return aux


class SongUNetPosEmbd(SongUNet):
    """Extends SongUNet with positional embeddings.

    This model supports conditional and unconditional setups, as well as several
    options for various internal architectural choices such as encoder and decoder
    type, embedding type, etc., making it flexible and adaptable to different tasks
    and configurations.

    This model adds positional embeddings to the base SongUNet architecture. The embeddings
    can be selected using either a selector function or global indices, with the selector
    approach being more computationally efficient.

    The model provides two methods for selecting positional embeddings:

    1. Using a selector function (preferred method). See
       :meth:`positional_embedding_selector` for details.
    2. Using global indices. See :meth:`positional_embedding_indexing` for
       details.

    Parameters
    ----------
    img_resolution : Union[List[int], int]
        The resolution of the input/output image. Can be a single int for square images
        or a list [height, width] for rectangular images.
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels in the output image.
    label_dim : int, optional
        Number of class labels; 0 indicates an unconditional model. By default 0.
    augment_dim : int, optional
        Dimensionality of augmentation labels; 0 means no augmentation. By default 0.
    model_channels : int, optional
        Base multiplier for the number of channels across the network. By default 128.
    channel_mult : List[int], optional
        Per-resolution multipliers for the number of channels. By default [1,2,2,2,2].
    channel_mult_emb : int, optional
        Multiplier for the dimensionality of the embedding vector. By default 4.
    num_blocks : int, optional
        Number of residual blocks per resolution. By default 4.
    attn_resolutions : List[int], optional
        Resolutions at which self-attention layers are applied. By default [28].
    dropout : float, optional
        Dropout probability applied to intermediate activations. By default 0.13.
    label_dropout : float, optional
        Dropout probability of class labels for classifier-free guidance. By default 0.0.
    embedding_type : str, optional
        Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        By default 'positional'.
    channel_mult_noise : int, optional
        Timestep embedding size: 1 for DDPM++, 2 for NCSN++. By default 1.
    encoder_type : str, optional
        Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++, 'skip' for skip connections.
        By default 'standard'.
    decoder_type : str, optional
        Decoder architecture: 'standard' or 'skip' for skip connections. By default 'standard'.
    resample_filter : List[int], optional
        Resampling filter coefficients: [1,1] for DDPM++, [1,3,3,1] for NCSN++. By default [1,1].
    gridtype : str, optional
        Type of positional grid to use: 'sinusoidal', 'learnable', 'linear', or 'test'.
        Controls how positional information is encoded. By default 'sinusoidal'.
    N_grid_channels : int, optional
        Number of channels in the positional embedding grid. For 'sinusoidal' must be 4 or
        multiple of 4. For 'linear' must be 2. By default 4.
    checkpoint_level : int, optional
        Number of layers that should use gradient checkpointing (0 disables checkpointing).
        Higher values trade memory for computation. By default 0.

    Note
    -----
    Equivalent to the original implementation by Song et al., available at
    https://github.com/yang-song/score_sde_pytorch

    Example
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
        gridtype: str = "sinusoidal",
        N_grid_channels: int = 4,
        checkpoint_level: int = 0,
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
        )

        self.gridtype = gridtype
        self.N_grid_channels = N_grid_channels
        self.pos_embd = self._get_positional_embedding()

    @nvtx.annotate(message="SongUNet", color="blue")
    def forward(
        self,
        x,
        noise_labels,
        class_labels,
        global_index: Optional[torch.Tensor] = None,
        embedding_selector: Optional[Callable] = None,
        augment_labels=None,
    ):
        if embedding_selector is not None and global_index is not None:
            raise ValueError(
                "Cannot provide both embedding_selector and global_index. "
                "embedding_selector is the preferred approach for better efficiency."
            )

        # Append positional embedding to input conditioning
        if self.pos_embd is not None:
            # Select positional embeddings with a selector function
            if embedding_selector is not None:
                selected_pos_embd = self.positional_embedding_selector(
                    x, embedding_selector
                )
            # Select positional embeddings using global indices (selects all
            # embeddings if global_index is None)
            else:
                selected_pos_embd = self.positional_embedding_indexing(x, global_index)
            x = torch.cat((x, selected_pos_embd), dim=1)

        return super().forward(x, noise_labels, class_labels, augment_labels)

    def positional_embedding_indexing(
        self,
        x: torch.Tensor,
        global_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Select positional embeddings using global indices.

        This method either uses global indices to select specific embeddings or expands
        the embeddings for the full input when no indices are provided.

        Typically used in patch-based training, where the batch dimension
        contains multiple patches extracted from a larger image.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W), used to determine batch size
            and device.
        global_index : Optional[torch.Tensor]
            Optional tensor of indices for selecting embeddings. These should
            correspond to the spatial indices of the batch elements in the
            input tensor x. When provided, should have shape (B, 2, H, W) where
            the second dimension contains y,x coordinates (indices of the
            positional embedding grid).

        Returns
        -------
        torch.Tensor
            Selected positional embeddings with shape:
            - If global_index provided: (B, N_pe, H, W)
            - If global_index is None: (B, N_pe, H_pe, W_pe)
            where N_pe is the number of positional embedding channels, and H_pe
            and W_pe are the height and width of the positional embedding grid.

        Example
        -------
        >>> # Create global indices using patching utility:
        >>> from physicsnemo.utils.patching import GridPatching2D
        >>> patching = GridPatching2D(img_shape=(16, 16), patch_shape=(8, 8))
        >>> global_index = patching.global_index(batch_size=3)
        >>> print(global_index.shape)
        torch.Size([12, 2, 8, 8])

        See Also
        --------
        :meth:`physicsnemo.utils.patching.RandomPatching2D.global_index`
            For generating random patch indices.
        :meth:`physicsnemo.utils.patching.GridPatching2D.global_index`
            For generating deterministic grid-based patch indices.
            See these methods for possible ways to generate the global_index parameter.
        """
        # If no global indices are provided, select all embeddings and expand
        # to match the batch size of the input
        if global_index is None:
            return (
                self.pos_embd.to(x.dtype)
                .to(x.device)[None]
                .expand((x.shape[0], -1, -1, -1))
            )  # (B, N_pe, H, W)

        B = global_index.shape[0]
        H = global_index.shape[2]
        W = global_index.shape[3]
        global_index = torch.reshape(
            torch.permute(global_index, (1, 0, 2, 3)), (2, -1)
        )  # (B, 2, H, W) to (2, B*H*W)
        # Use advanced indexing to select the positional embeddings based on
        # their y-x coordinates
        selected_pos_embd = self.pos_embd.to(x.device)[
            :, global_index[0], global_index[1]
        ]  # (N_pe, B*H*W)
        selected_pos_embd = (
            torch.permute(
                torch.reshape(selected_pos_embd, (self.pos_embd.shape[0], B, H, W)),
                (1, 0, 2, 3),
            )
            .to(x.device)
            .to(x.dtype)
        )  # (B, N_pe, H, W)
        return selected_pos_embd

    def positional_embedding_selector(
        self,
        x: torch.Tensor,
        embedding_selector: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Select positional embeddings using a selector function.

        Similar to positional_embedding_indexing, but uses a selector function
        to select the embeddings. This method provides a more efficient way to
        select embeddings for batches of data.
        Typically used with patch-based processing, where the batch dimension
        contains multiple patches extracted from a larger image.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W) only used to determine dtype and
            device.
        embedding_selector : Callable
            Function that takes as input an embedding tensor of shape (N_pe,
            H_pe, W_pe) and returns selected embeddings with shape (batch_size, N_pe, H, W).
            Each selected embedding should correspond to the positional
            information of each batch element in x.
            For patch-based processing, typically this should be based on
            :meth:`physicsnemo.utils.patching.BasePatching2D.apply` method to
            maintain consistency with patch extraction.

        Returns
        -------
        torch.Tensor
            Selected positional embeddings with shape (B, N_pe, H, W)
            where N_pe is the number of positional embedding channels.

        Example
        -------
        >>> # Define a selector function with a patching utility:
        >>> from physicsnemo.utils.patching import GridPatching2D
        >>> patching = GridPatching2D(img_shape=(16, 16), patch_shape=(8, 8))
        >>> batch_size = 4
        >>> def embedding_selector(emb):
        ...     return patching.apply(emb[None].expand(batch_size, -1, -1, -1))
        >>>

        See Also
        --------
        :meth:`physicsnemo.utils.patching.BasePatching2D.apply`
            For the base patching method typically used in embedding_selector.
        """
        return embedding_selector(
            self.pos_embd.to(x.dtype).to(x.device)
        )  # (B, N_pe, H, W)

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


# TODO: Lots of stuff in common with SongUNetPosEmbd. Should inherit from it
# instead of SongUNet
class SongUNetPosLtEmbd(SongUNet):
    """
    This model is adapted from SongUNetPosEmbd, with the incorporation of lead-time aware
    embeddings. The lead-time embedding is activated by setting the
    `lead_time_channels` and `lead_time_steps` parameters.

    Like SongUNetPosEmbd, this model provides two methods for selecting positional embeddings:
    1. Using a selector function (preferred method). See
       :meth:`positional_embedding_selector` for details.
    2. Using global indices. See :meth:`positional_embedding_indexing` for
       details.

    Parameters
    -----------
    img_resolution : Union[List[int], int]
        The resolution of the input/output image. Can be a single int for square images
        or a list [height, width] for rectangular images.
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels in the output image.
    label_dim : int, optional
        Number of class labels; 0 indicates an unconditional model. By default 0.
    augment_dim : int, optional
        Dimensionality of augmentation labels; 0 means no augmentation. By default 0.
    model_channels : int, optional
        Base multiplier for the number of channels across the network. By default 128.
    channel_mult : List[int], optional
        Per-resolution multipliers for the number of channels. By default [1,2,2,2,2].
    channel_mult_emb : int, optional
        Multiplier for the dimensionality of the embedding vector. By default 4.
    num_blocks : int, optional
        Number of residual blocks per resolution. By default 4.
    attn_resolutions : List[int], optional
        Resolutions at which self-attention layers are applied. By default [28].
    dropout : float, optional
        Dropout probability applied to intermediate activations. By default 0.13.
    label_dropout : float, optional
        Dropout probability of class labels for classifier-free guidance. By default 0.0.
    embedding_type : str, optional
        Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        By default 'positional'.
    channel_mult_noise : int, optional
        Timestep embedding size: 1 for DDPM++, 2 for NCSN++. By default 1.
    encoder_type : str, optional
        Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++, 'skip' for skip connections.
        By default 'standard'.
    decoder_type : str, optional
        Decoder architecture: 'standard' or 'skip' for skip connections. By default 'standard'.
    resample_filter : List[int], optional
        Resampling filter coefficients: [1,1] for DDPM++, [1,3,3,1] for NCSN++. By default [1,1].
    gridtype : str, optional
        Type of positional grid to use: 'sinusoidal', 'learnable', 'linear', or 'test'.
        Controls how positional information is encoded. By default 'sinusoidal'.
    N_grid_channels : int, optional
        Number of channels in the positional embedding grid. For 'sinusoidal' must be 4 or
        multiple of 4. For 'linear' must be 2. By default 4.
    lead_time_channels : int, optional
        Number of channels in the lead time embedding. These are learned embeddings that
        encode temporal forecast information. By default None.
    lead_time_steps : int, optional
        Number of discrete lead time steps to support. Each step gets its own learned
        embedding vector. By default 9.
    prob_channels : List[int], optional
        Indices of probability output channels that should use softmax activation.
        Used for classification outputs. By default empty list.
    checkpoint_level : int, optional
        Number of layers that should use gradient checkpointing (0 disables checkpointing).
        Higher values trade memory for computation. By default 0.

    Note
    -----
    Equivalent to the original implementation by Song et al., available at
    https://github.com/yang-song/score_sde_pytorch

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
    >>>
    >>> # Using custom embedding selector to select all the positional and lead time embeddings
    >>> def patch_embedding_selector(emb):
    ...     return patching.apply(emb[None].expand(1, -1, -1, -1))
    >>> output_image = model(
    ...     input_image, noise_labels, class_labels,
    ...     lead_time_label=lead_time_label,
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
        gridtype: str = "sinusoidal",
        N_grid_channels: int = 4,
        lead_time_channels: int = None,
        lead_time_steps: int = 9,
        prob_channels: List[int] = [],
        checkpoint_level: int = 0,
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
        )

        self.gridtype = gridtype
        self.N_grid_channels = N_grid_channels
        self.pos_embd = self._get_positional_embedding()
        self.lead_time_channels = lead_time_channels
        self.lead_time_steps = lead_time_steps
        self.lt_embd = self._get_lead_time_embedding()
        self.prob_channels = prob_channels
        if self.prob_channels:
            self.scalar = torch.nn.Parameter(
                torch.ones((1, len(self.prob_channels), 1, 1))
            )

    @nvtx.annotate(message="SongUNet", color="blue")
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
        if embedding_selector is not None and global_index is not None:
            raise ValueError(
                "Cannot provide both embedding_selector and global_index. "
                "embedding_selector is the preferred approach for better efficiency."
            )

        # Append positional and lead time embeddings to input conditioning
        embeds = []
        if self.pos_embd is not None:
            embeds.append(self.pos_embd.to(x.device))
        if self.lt_embd is not None:
            embeds.append(
                torch.reshape(
                    self.lt_embd[lead_time_label.int()],
                    (self.lead_time_channels, self.img_shape_y, self.img_shape_x),
                ).to(x.device)
            )
        if len(embeds) > 0:
            embeds = torch.cat(embeds, dim=0)
            # Select embeddings using either selector function or global indices
            if embedding_selector is not None:
                selected_pos_embd = self.positional_embedding_selector(
                    x, embeds, embedding_selector
                )
            else:
                selected_pos_embd = self.positional_embedding_indexing(
                    x, embeds, global_index
                )
            x = torch.cat((x, selected_pos_embd), dim=1)

        out = super().forward(x, noise_labels, class_labels, augment_labels)

        # if training mode, let crossEntropyLoss do softmax. The model outputs logits.
        # if eval mode, the model outputs probability
        all_channels = list(range(out.shape[1]))  # [0, 1, 2, ..., 10]
        scalar_channels = [
            item for item in all_channels if item not in self.prob_channels
        ]
        if self.prob_channels and (not self.training):
            out_final = torch.cat(
                (
                    out[:, scalar_channels],
                    (out[:, self.prob_channels] * self.scalar).softmax(dim=1),
                ),
                dim=1,
            )
        elif self.prob_channels and self.training:
            out_final = torch.cat(
                (out[:, scalar_channels], (out[:, self.prob_channels] * self.scalar)),
                dim=1,
            )
        else:
            out_final = out
        return out_final

    def positional_embedding_indexing(
        self,
        x: torch.Tensor,
        embeds: torch.Tensor,
        global_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Select positional embeddings using global indices.

        This method either uses global indices to select specific embeddings or expands
        the embeddings for the full input when no indices are provided.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W), used to determine batch size
            and device.
        embeds : torch.Tensor
            Combined positional and lead time embeddings tensor.
        global_index : Optional[torch.Tensor]
            Optional tensor of indices for selecting embeddings. These should
            correspond to the spatial indices of the batch elements in the
            input tensor x. When provided, should have shape (B, 2, H, W) where
            the second dimension contains y,x coordinates.

        Returns
        -------
        torch.Tensor
            Selected embeddings with shape (B, N_pe, H, W) where N_pe is the
            total number of embedding channels (positional + lead time).

        Example
        -------
        >>> # Create global indices using patching utility:
        >>> from physicsnemo.utils.patching import GridPatching2D
        >>> patching = GridPatching2D(img_shape=(16, 16), patch_shape=(8, 8))
        >>> global_index = patching.global_index(batch_size=1)
        >>> global_index.shape
        torch.Size([4, 2, 8, 8])

        See Also
        --------
        :meth:`physicsnemo.utils.patching.RandomPatching2D.global_index`
            For generating random patch indices.
        :meth:`physicsnemo.utils.patching.GridPatching2D.global_index`
            For generating deterministic grid-based patch indices.
        See these methods for possible ways to generate the global_index parameter.
        """
        if global_index is None:
            return (
                embeds.to(x.dtype).to(x.device)[None].expand((x.shape[0], -1, -1, -1))
            )

        B = global_index.shape[0]
        H = global_index.shape[2]
        W = global_index.shape[3]
        global_index = torch.reshape(
            torch.permute(global_index, (1, 0, 2, 3)), (2, -1)
        )  # (B, 2, H, W) to (2, B*H*W)
        selected_embeds = embeds.to(x.device)[:, global_index[0], global_index[1]]
        selected_embeds = (
            torch.permute(
                torch.reshape(selected_embeds, (embeds.shape[0], B, H, W)),
                (1, 0, 2, 3),
            )
            .to(x.device)
            .to(x.dtype)
        )  # (B, N_pe, H, W)
        return selected_embeds

    def positional_embedding_selector(
        self,
        x: torch.Tensor,
        embeds: torch.Tensor,
        embedding_selector: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Select positional embeddings using a selector function.

        Similar to positional_embedding_indexing, but uses a selector function
        to select the embeddings. This method provides a more efficient way to
        select embeddings for batches of data.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W) used to determine batch
            size and device.
        embeds : torch.Tensor
            Combined positional and lead time embeddings tensor of shape
            (N_pe, H_pe, W_pe) where N_pe is the total number of embedding
            channels.
        embedding_selector : Callable
            Function that takes as input an embedding tensor of shape (N_pe,
            H_pe, W_pe) and returns selected embeddings with shape (B, N_pe, H, W).
            Each selected embedding should correspond to the positional
            information of each batch element in x.
            For patch-based processing, typically this should be based on
            :meth:`physicsnemo.utils.patching.BasePatching2D.apply` method to
            maintain consistency with patch extraction.

        Returns
        -------
        torch.Tensor
            Selected embeddings with shape (B, N_pe, H, W) where N_pe is the
            total number of embedding channels (positional + lead time).

        Example
        -------
        >>> # Define a selector function with a patching utility:
        >>> from physicsnemo.utils.patching import GridPatching2D
        >>> patching = GridPatching2D(img_shape=(16, 16), patch_shape=(8, 8))
        >>> batch_size = 4
        >>> def embedding_selector(emb):
        ...     return patching.apply(emb[None].expand(batch_size, -1, -1, -1))
        >>>

        See Also
        --------
        :meth:`physicsnemo.utils.patching.BasePatching2D.apply`
            For the base patching method typically used in embedding_selector.
        """
        return embedding_selector(embeds.to(x.dtype).to(x.device))  # (B, N_pe, H, W)

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
