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


"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch.nn.functional import silu
import einops
from typing import Optional

# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Linear(torch.nn.Module):
    """
    Fully-connected layer.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class Conv2d(torch.nn.Module):
    """
    Convolutional layer with optional up/downsampling.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=[1, 1],
        fused_resample=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(x.dtype)
            if self.resample_filter is not None
            else None
        )
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(
                x,
                f.tile([self.out_channels, 1, 1, 1]),
                groups=self.out_channels,
                stride=2,
            )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class GroupNorm(torch.nn.Module):
    """
    Group normalization.
    """

    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x


class AttentionOp(torch.autograd.Function):
    """
    Attention weight computation, i.e., softmax(Q^T * K).
    Performs all computation using FP32, but uses the original datatype for
    inputs/outputs/gradients to conserve memory.
    """

    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum(
                "ncq,nck->nqk",
                q.to(torch.float32),
                (k / np.sqrt(k.shape[1])).to(torch.float32),
            )
            .softmax(dim=2)
            .to(q.dtype)
        )
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32),
            output=w.to(torch.float32),
            dim=2,
            input_dtype=torch.float32,
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(
            q.dtype
        ) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(
            k.dtype
        ) / np.sqrt(k.shape[1])
        return dq, dk


class UNetBlock(torch.nn.Module):
    """
    Unified U-Net block with optional up/downsampling and self-attention.
    Represents the union of all features employed by the DDPM++, NCSN++, and
    ADM architectures.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=[1, 1],
        resample_proj=False,
        adaptive_scale=True,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else num_heads
            if num_heads is not None
            else out_channels // channels_per_head
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            **init,
        )
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
            )

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(
                    x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
                .unbind(2)
            )
            w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


class PositionalEmbedding(torch.nn.Module):
    """
    Timestep embedding used in the DDPM++ and ADM architectures.
    """

    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierEmbedding(torch.nn.Module):
    """
    Fourier embedding used in the NCSN++ architecture.
    """

    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class SongUNet(torch.nn.Module):
    """
    Reimplementation of the DDPM++ and NCSN++ architectures from the paper
    "Score-Based Generative Modeling through Stochastic Differential
    Equations". Equivalent to the original implementation by Song et al.,
    available at https://github.com/yang-song/score_sde_pytorch
    """

    def __init__(
        self,
        img_resolution,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=128,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            2,
            2,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=4,  # Number of residual blocks per resolution.
        attn_resolutions=[16],  # List of resolutions with self-attention.
        dropout=0.10,  # Dropout probability of intermediate activations.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        embedding_type="positional",  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type="standard",  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type="standard",  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter=[
            1,
            1,
        ],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        spatial_embedding=None,  # None or 'add' or 'concat'
        hrrr_resolution=(512, 640),
    ):
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]

        super().__init__()
        self.label_dropout = label_dropout
        if spatial_embedding in ["add", "concat"]:
            self.spatial_pos_embed = torch.nn.Parameter(
                torch.randn(1, 1, hrrr_resolution[0], hrrr_resolution[1])
            )
            self.spatial_embedding = spatial_embedding
        else:
            self.spatial_embedding = None
        emb_channels = model_channels * channel_mult_emb
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

        # Mapping.
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
                in_features=augment_dim, out_features=noise_channels, bias=False, **init
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
            res = img_resolution >> level
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

    def forward(self, x, noise_labels, class_labels, augment_labels=None):

        emb = self.map_noise(noise_labels)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (
                    torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
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
                x = block(x, emb)

        return aux


class EDMPrecond(torch.nn.Module):
    """
    Preconditioned diffusion model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        img_resolution,  # Image resolution.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model

    def forward(
        self,
        x,
        sigma,
        condition=None,
        class_labels=None,
        force_fp32=False,
        **model_kwargs,
    ):
        """
        Args:
            x: The latent data (to be denoised). shape [batch_size, target_channels, w, h]
            class_labels: optional, shape [batch_size, label_dim]
            condition: optional, the conditional inputs,
                shape=[batch_size, condition_channel, w, h]
            **model_kwargs: passed on to self.model.

        Returns:
            D_x: shape [batch_size, out_channels, w, h]

        """
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        arg = c_in * x

        if condition is not None:
            arg = torch.cat([arg, condition], dim=1)

        F_x = self.model(
            arg.to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------


def get_preconditioned_architecture(
    name: str,
    resolution: int,
    target_channels: int,
    conditional_channels: int = 0,
    label_dim: int = 0,
    spatial_embedding: Optional[str] = None,
    hrrr_resolution: tuple = (512, 640),
    attn_resolutions: list = [],
):
    """

    Args:
        resolution (int): _description_
        target_channels: The number of channels in the target
        conditional_channels: The number of channels in the conditioning
        label_dim: size of label data
        sigma_min:  Defaults to 0.
        sigma_max: Defaults to float("inf").
        sigma_data:  Defaults to 0.5.

    Returns:
        EDMPrecond: a wrapped torch module net(x+n, sigma, condition, class_labels) -> x
    """
    if name == "ddpmpp-cwb-v0":
        model = SongUNet(
            img_resolution=resolution,
            in_channels=target_channels + conditional_channels,
            out_channels=target_channels,
            label_dim=label_dim,
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[1, 2, 2, 2, 2],
            attn_resolutions=attn_resolutions,
            spatial_embedding=spatial_embedding,
            hrrr_resolution=hrrr_resolution,
        )
    else:
        raise ValueError(f'Invalid architecture name "{name}"')

    return EDMPrecond(
        model=model,
        img_resolution=resolution,
        label_dim=label_dim,
    )
