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
from typing import List

import numpy as np
import torch
from jaxtyping import Float
from torch.nn.functional import silu

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn import (
    Conv2d,
    Linear,
    PositionalEmbedding,
    UNetBlock,
    get_group_norm,
)

from ._utils import _recursive_property

# ------------------------------------------------------------------------------
# Backbone architectures
# ------------------------------------------------------------------------------


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


# NOTE: this module can actually be replicated as a special case of the
# SongUnet class (with very minior extension of the SongUnet class). We should
# consider inheriting the more general SongUnet class here.
class DhariwalUNet(Module):
    r"""
    This architecture is a diffusion backbone for 2D image generation. It
    reimplements the `ADM architecture <https://arxiv.org/abs/2105.05233>`_, a U-Net variant, with optional
    self-attention.

    It is highly similar to the U-Net backbone defined in
    :class:`~physicsnemo.models.diffusion_unets.SongUNet`, and only differs
    in a few aspects:

    • The embedding conditioning mechanism relies on adaptive scaling of the
      group normalization layers within the U-Net blocks.

    • The parameters initialization follows Kaiming uniform initialization.

    Parameters
    -----------
    img_resolution :int
        The resolution :math:`H = W` of the input/output image. Assumes square images.

        *Note:* This parameter is only used as a convenience to build the
        network. In practice, the model can still be used with images of
        different resolutions.
    in_channels : int
        Number of channels :math:`C_{in}` in the input image. May include channels from both the
        latent state :math:`\mathbf{x}` and additional channels when conditioning on images. For an
        unconditional model, this should be equal to ``out_channels``.
    out_channels : int
        Number of channels :math:`C_{out}` in the output image. Should be equal to the number
        of channels :math:`C_{\mathbf{x}}` in the latent state.
    label_dim : int, optional, default=0
        Dimension of the vector-valued ``class_labels`` conditioning; 0
        indicates no conditioning on class labels.
    augment_dim : int, optional, default=0
        Dimension of the vector-valued ``augment_labels`` conditioning; 0 means
        no conditioning on augmentation labels.
    model_channels : int, optional, default=128
        Base multiplier for the number of channels accross the entire network.
    channel_mult : List[int], optional, default=[1,2,2,2]
        Multipliers for the number of channels at every level in
        the encoder and decoder. The length of ``channel_mult`` determines the
        number of levels in the U-Net. At level ``i``, the number of channel in
        the feature map is ``channel_mult[i] * model_channels``.
    channel_mult_emb : int, optional, default=4
        Multiplier for the number of channels in the embedding vector. The
        embedding vector has ``model_channels * channel_mult_emb`` channels.
    num_blocks : int, optional, default=3
        Number of U-Net blocks at each level.
    attn_resolutions : List[int], optional, default=[16]
        Resolutions of the levels at which self-attention layers are applied.
        Note that the feature map resolution must match exactly the value
        provided in ``attn_resolutions`` for the self-attention layers to be
        applied.
    dropout : float, optional, default=0.10
        Dropout probability applied to intermediate activations within the
        U-Net blocks.
    label_dropout : float, optional, default=0.0
        Dropout probability applied to the ``class_labels``. Typically used for
        classifier-free guidance.


    Forward
    -------
    x : torch.Tensor
        The input tensor of shape :math:`(B, C_{in}, H_{in}, W_{in})`. In general ``x``
        is the channel-wise concatenation of the latent state :math:`\mathbf{x}`
        and additional images used for conditioning. For an unconditional
        model, ``x`` is simply the latent state :math:`\mathbf{x}`.
    noise_labels : torch.Tensor
        The noise labels of shape :math:`(B,)`. Used for conditioning on
        the noise level.
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
    torch.Tensor:
        The denoised latent state of shape :math:`(B, C_{out}, H_{in}, W_{in})`.


    Examples
    --------
    >>> model = DhariwalUNet(img_resolution=16, in_channels=2, out_channels=2)
    >>> noise_labels = torch.randn([1])
    >>> class_labels = torch.randint(0, 1, (1, 1))  # noqa: N806
    >>> input_image = torch.ones([1, 2, 16, 16])  # noqa: N806
    >>> output_image = model(input_image, noise_labels, class_labels)  # noqa: N806
    """

    def __init__(
        self,
        img_resolution: int,
        in_channels: int,
        out_channels: int,
        label_dim: int = 0,
        augment_dim: int = 0,
        model_channels: int = 192,
        channel_mult: List[int] = [1, 2, 3, 4],
        channel_mult_emb: int = 4,
        num_blocks: int = 3,
        attn_resolutions: List[int] = [32, 16, 8],
        dropout: float = 0.10,
        label_dropout: float = 0.0,
    ):
        super().__init__(meta=MetaData())
        self.label_dim = label_dim
        self.augment_dim = augment_dim
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(
            init_mode="kaiming_uniform",
            init_weight=np.sqrt(1 / 3),
            init_bias=np.sqrt(1 / 3),
        )
        init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
        block_kwargs = dict(
            emb_channels=emb_channels,
            channels_per_head=64,
            dropout=dropout,
            init=init,
            init_zero=init_zero,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = (
            Linear(
                in_features=augment_dim,
                out_features=model_channels,
                bias=False,
                **init_zero,
            )
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(
            in_features=model_channels, out_features=emb_channels, **init
        )
        self.map_layer1 = Linear(
            in_features=emb_channels, out_features=emb_channels, **init
        )
        self.map_label = (
            Linear(
                in_features=label_dim,
                out_features=emb_channels,
                bias=False,
                init_mode="kaiming_normal",
                init_weight=np.sqrt(label_dim),
            )
            if label_dim
            else None
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
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
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        self.out_norm = get_group_norm(num_channels=cout)
        self.out_conv = Conv2d(
            in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
        )

        # Set properties recursively on submodules
        self.profile_mode = False
        self.amp_mode = False

    # Properties that are recursively set on submodules
    profile_mode = _recursive_property(
        "profile_mode", bool, "Should be set to ``True`` to enable profiling."
    )
    amp_mode = _recursive_property(
        "amp_mode",
        bool,
        "Should be set to ``True`` to enable automatic mixed precision.",
    )

    def forward(
        self,
        x: Float[torch.Tensor, "B C_in H_in W_in"],
        noise_labels: Float[torch.Tensor, " B_or_1"],
        class_labels: Float[torch.Tensor, "B label_dim"] | None = None,
        augment_labels: Float[torch.Tensor, "B augment_dim"] | None = None,
    ) -> Float[torch.Tensor, "B C_out H_in W_in"]:
        # Input validation
        if not torch.compiler.is_compiling():
            batch_size = x.shape[0]

            if x.ndim != 4:
                raise ValueError(
                    f"Expected 'x' to be a 4D tensor (B, C_in, H, W), "
                    f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
                )

            if noise_labels.ndim != 1 or noise_labels.shape[0] not in (
                batch_size,
                1,
            ):
                raise ValueError(
                    f"Expected 'noise_labels' shape ({batch_size},) or (1,), "
                    f"got {tuple(noise_labels.shape)}"
                )
            if (
                self.label_dim > 0
                and class_labels is not None
                and class_labels.shape != (batch_size, self.label_dim)
            ):
                raise ValueError(
                    f"Expected 'class_labels' shape ({batch_size}, {self.label_dim}), "
                    f"got {tuple(class_labels.shape)}"
                )

            if (
                self.augment_dim > 0
                and augment_labels is not None
                and augment_labels.shape != (batch_size, self.augment_dim)
            ):
                raise ValueError(
                    f"Expected 'augment_labels' shape ({batch_size}, {self.augment_dim}), "
                    f"got {tuple(augment_labels.shape)}"
                )

        # Compute conditioning embeddings from noise, class, and augment labels
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (
                    torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder: progressively downsample and cache skip connections
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder: progressively upsample and merge skip connections
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)

        # Final normalization and projection to output channels
        x = self.out_conv(silu(self.out_norm(x)))
        return x
