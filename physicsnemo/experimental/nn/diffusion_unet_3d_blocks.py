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

import math
from typing import Any, Dict, List, Literal

import torch
from einops import rearrange
from jaxtyping import Float
from torch.nn.functional import dropout, scaled_dot_product_attention, silu

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn.module.fully_connected_layers import Linear
from physicsnemo.nn.module.utils.weight_init import _weight_init


class GroupNorm3D(Module):
    r"""
    Group Normalization for 5D tensors :math:`(B, C, D, H, W)`.

    Divides the channel dimension into groups and normalizes within each group
    independently. During training, uses ``torch.nn.functional.group_norm``.
    During inference, uses a manual implementation compatible with channels-last
    memory layouts.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor.
    num_groups : int, optional, default=32
        Target number of groups. Adjusted downward if
        ``num_channels // num_groups < min_channels_per_group``.
    min_channels_per_group : int, optional, default=4
        Minimum channels allowed per group.
    eps : float, optional, default=1e-5
        Epsilon for numerical stability.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, D, H, W)`.

    Outputs
    -------
    torch.Tensor
        Normalized tensor of shape :math:`(B, C, D, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.nn import GroupNorm3D
    >>> gn = GroupNorm3D(num_channels=32)
    >>> x = torch.randn(2, 32, 4, 12, 16)
    >>> y = gn(x)
    >>> y.shape
    torch.Size([2, 32, 4, 12, 16])
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        min_channels_per_group: int = 4,
        eps: float = 1e-5,
    ):
        super().__init__(meta=ModelMetaData())
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(
        self, x: Float[torch.Tensor, "B C D H W"]
    ) -> Float[torch.Tensor, "B C D H W"]:
        if self.training:
            x = torch.nn.functional.group_norm(
                x,
                num_groups=self.num_groups,
                weight=self.weight.to(x.dtype),
                bias=self.bias.to(x.dtype),
                eps=self.eps,
            )
        else:
            # Manual implementation that supports channels-last memory layout
            dtype = x.dtype
            x = x.float()
            x = rearrange(x, "b (g c) d h w -> b g c d h w", g=self.num_groups)
            mean = x.mean(dim=[2, 3, 4, 5], keepdim=True)
            var = x.var(dim=[2, 3, 4, 5], keepdim=True)
            x = (x - mean) * (var + self.eps).rsqrt()
            x = rearrange(x, "b g c d h w -> b (g c) d h w")
            x = x * rearrange(self.weight, "c -> 1 c 1 1 1") + rearrange(
                self.bias, "c -> 1 c 1 1 1"
            )
            x = x.to(dtype)
        return x


class Conv3D(Module):
    r"""
    3D convolution with optional fused up/downsampling.

    Implements a 3D convolution with optional 2x upsampling or downsampling via
    separable bilinear/bicubic filters. When a convolution weight is present
    (``kernel > 0``), resampling is fused with the convolution for efficiency.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel : int
        Convolution kernel size applied uniformly across all spatial dimensions.
        Set to 0 to apply resampling only (no learned convolution).
    bias : bool, optional, default=True
        Whether to include a learnable bias.
    up : bool, optional, default=False
        Apply 2x upsampling. Cannot be ``True`` simultaneously with ``down``.
    down : bool, optional, default=False
        Apply 2x downsampling. Cannot be ``True`` simultaneously with ``up``.
    resample_filter : list[int], optional, default=[1, 1]
        1D coefficients for the separable up/downsampling filter. The 3D filter
        is constructed as their outer product, normalized so it sums to 1.
        Use ``[1, 1]`` for bilinear resampling or ``[1, 3, 3, 1]`` for bicubic.
        Must be a non-empty list of positive integers.
    init_mode : Literal["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"], optional, default="kaiming_normal"
        Weight initialization mode.
    init_weight : float, optional, default=1.0
        Multiplier applied to the initialized weight tensor.
    init_bias : float, optional, default=0.0
        Multiplier applied to the initialized bias tensor.

    Raises
    ------
    ValueError
        If both ``up`` and ``down`` are ``True``, or if ``resample_filter`` is
        empty / contains non-positive values when ``up`` or ``down`` is ``True``.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, D, H, W)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, D', H', W')`. The spatial
        dimensions are doubled (``up=True``), halved (``down=True``), or unchanged.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.nn import Conv3D
    >>> conv = Conv3D(in_channels=4, out_channels=8, kernel=3)
    >>> x = torch.randn(2, 4, 4, 12, 16)
    >>> y = conv(x)
    >>> y.shape
    torch.Size([2, 8, 4, 12, 16])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        bias: bool = True,
        up: bool = False,
        down: bool = False,
        resample_filter: List[int] = [1, 1],
        init_mode: Literal[
            "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"
        ] = "kaiming_normal",
        init_weight: float = 1.0,
        init_bias: float = 0.0,
    ):
        if up and down:
            raise ValueError("Both 'up' and 'down' cannot be True at the same time.")
        if (up or down) and (
            not resample_filter or any(v <= 0 for v in resample_filter)
        ):
            raise ValueError(
                f"resample_filter must be a non-empty list of positive integers "
                f"when up=True or down=True, got {resample_filter}"
            )

        super().__init__(meta=ModelMetaData())
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down

        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel * kernel,
            fan_out=out_channels * kernel * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                _weight_init(
                    (out_channels, in_channels, kernel, kernel, kernel), **init_kwargs
                )
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(
                _weight_init((out_channels,), **init_kwargs) * init_bias
            )
            if kernel and bias
            else None
        )

        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = (f.ger(f).unsqueeze(2) * f.view(1, 1, -1)).unsqueeze(0).unsqueeze(
            1
        ) / f.sum().pow(3)
        self.register_buffer("resample_filter", f.contiguous() if up or down else None)

    def forward(
        self, x: Float[torch.Tensor, "B C_in D H W"]
    ) -> Float[torch.Tensor, "B C_out D_out H_out W_out"]:
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(x.dtype)
            if self.resample_filter is not None
            else None
        )
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.up and w is not None:
            # Fused upsample + conv
            x = torch.nn.functional.conv_transpose3d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            x = torch.nn.functional.conv3d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.down and w is not None:
            # Fused conv + downsample
            x = torch.nn.functional.conv3d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv3d(
                x,
                f.tile([self.out_channels, 1, 1, 1, 1]),
                groups=self.out_channels,
                stride=2,
            )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose3d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv3d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:
                x = torch.nn.functional.conv3d(x, w, padding=w_pad)

        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1, 1))
        return x


class UNetAttention3D(Module):
    r"""
    Multi-head 3D self-attention block.

    Applies group normalization followed by multi-head self-attention with a
    residual connection. Operates on volumetric feature maps of shape
    :math:`(B, C, D, H, W)`, flattening the spatial dimensions for the
    attention operation.

    Parameters
    ----------
    out_channels : int
        Number of channels :math:`C` in the input and output feature maps.
        Must be divisible by ``num_heads``.
    num_heads : int
        Number of attention heads. Must be a positive integer.
    eps : float, optional, default=1e-5
        Epsilon for numerical stability in :class:`GroupNorm3D`.
    init_zero : dict, optional, default={'init_weight': 0}
        Initialization kwargs with near-zero weights for the output projection.
    init_attn : dict or None, optional, default=None
        Initialization kwargs for the QKV projection. Defaults to ``init`` if ``None``.
    init : dict, optional, default={}
        Initialization kwargs for linear and convolutional layers.

    Raises
    ------
    ValueError
        If ``num_heads`` is not a positive integer, or ``out_channels`` is not
        divisible by ``num_heads``.

    Forward
    -------
    x : torch.Tensor
        Input feature map of shape :math:`(B, C, D, H, W)`.

    Outputs
    -------
    torch.Tensor
        Output feature map of shape :math:`(B, C, D, H, W)`, identical to input shape.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.nn import UNetAttention3D
    >>> attn = UNetAttention3D(out_channels=32, num_heads=4)
    >>> x = torch.randn(2, 32, 4, 12, 16)
    >>> y = attn(x)
    >>> y.shape
    torch.Size([2, 32, 4, 12, 16])
    """

    def __init__(
        self,
        *,
        out_channels: int,
        num_heads: int,
        eps: float = 1e-5,
        init_zero: Dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
        init: Dict[str, Any] = dict(),
    ) -> None:
        super().__init__(meta=ModelMetaData())
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {num_heads}"
            )
        if out_channels % num_heads != 0:
            raise ValueError(
                f"out_channels must be divisible by num_heads, "
                f"got out_channels={out_channels} and num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.norm = GroupNorm3D(num_channels=out_channels, eps=eps)
        self.qkv = Conv3D(
            in_channels=out_channels,
            out_channels=out_channels * 3,
            kernel=1,
            **(init_attn if init_attn is not None else init),
        )
        self.proj = Conv3D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=1,
            **init_zero,
        )

    def forward(
        self, x: Float[torch.Tensor, "B C D H W"]
    ) -> Float[torch.Tensor, "B C D H W"]:
        x1 = self.qkv(self.norm(x))  # (B, 3C, D, H, W)

        # Reshape for multi-head attention over flattened spatial dims D*H*W
        qkv = (
            x1.reshape(x.shape[0], self.num_heads, x.shape[1] // self.num_heads, 3, -1)
        ).permute(0, 1, 4, 3, 2)  # (B, num_heads, D*H*W, 3, C//num_heads)
        q, k, v = (qkv[..., i, :] for i in range(3))

        attn = scaled_dot_product_attention(
            q, k, v, scale=1 / math.sqrt(k.shape[-1])
        )  # (B, num_heads, D*H*W, C//num_heads)

        attn = attn.transpose(-1, -2)  # (B, num_heads, C//num_heads, D*H*W)
        return self.proj(attn.reshape(*x.shape)).add_(x)  # residual, (B, C, D, H, W)


class UNetBlock3D(Module):
    r"""
    Residual U-Net block for 3D volumetric inputs with an external embedding input.

    Applies a residual block with optional up/downsampling and self-attention,
    conditioned on an external vector input :math:`\mathbf{e}` via an affine
    transformation on intermediate features. The architecture combines elements
    from the DDPM++, NCSN++, and ADM U-Net designs and is suitable for any
    backbone that needs a conditioned 3D residual block.

    Parameters
    ----------
    in_channels : int
        Number of input channels :math:`C_{in}`.
    out_channels : int
        Number of output channels :math:`C_{out}`.
    emb_channels : int
        Dimension :math:`C_{emb}` of the external embedding vector :math:`\mathbf{e}`.
        :math:`\mathbf{e}` is broadcast spatially and consumed by the affine
        conditioning step. It can be any vector-valued input (e.g. a diffusion-time
        embedding, a sinusoidal positional code, a learned class embedding, etc.).
    up : bool, optional, default=False
        Apply 2x upsampling to the feature map in the first convolution.
    down : bool, optional, default=False
        Apply 2x downsampling to the feature map in the first convolution.
    attention : bool, optional, default=False
        Apply 3D self-attention after the residual branch.
    num_heads : int or None, optional, default=None
        Number of attention heads when ``attention=True``. Defaults to 1 if ``None``.
        Ignored when ``attention=False``.
    dropout : float, optional, default=0.0
        Dropout probability applied before the second convolution.
    skip_scale : float, optional, default=1.0
        Scale factor applied to the residual output and (if attention is enabled)
        to the attention residual.
    eps : float, optional, default=1e-5
        Epsilon for :class:`GroupNorm3D` normalization layers.
    resample_filter : list[int], optional, default=[1, 1]
        1D filter coefficients for up/downsampling. Passed to :class:`Conv3D`.
    resample_proj : bool, optional, default=False
        Use a :math:`1 \times 1 \times 1` projection in the skip path when
        the number of channels or the resolution changes.
    adaptive_scale : bool, optional, default=True
        If ``True``, apply FiLM-style scale-and-shift affine conditioning.
        If ``False``, apply additive shift only.
    activation : Literal["silu", "gelu"], optional, default="silu"
        Activation function applied after normalization layers.
    init : dict, optional, default={}
        Weight initialization kwargs for convolutions and linear layers.
    init_zero : dict, optional, default={'init_weight': 0}
        Weight initialization kwargs with near-zero weights for the output convolution.
    init_attn : dict or None, optional, default=None
        Weight initialization kwargs for the attention QKV projection. Defaults to
        ``init`` if ``None``.

    Forward
    -------
    x : torch.Tensor
        Input feature map of shape :math:`(B, C_{in}, D, H, W)`.
    emb : torch.Tensor
        External vector input of shape :math:`(B, C_{emb})`. Used for affine
        conditioning of intermediate features.

    Outputs
    -------
    torch.Tensor
        Output feature map of shape :math:`(B, C_{out}, D', H', W')` where
        :math:`D', H', W'` are halved (``down=True``), doubled (``up=True``),
        or equal to the input spatial dimensions.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.nn import UNetBlock3D
    >>> block = UNetBlock3D(in_channels=8, out_channels=16, emb_channels=32)
    >>> x = torch.randn(2, 8, 4, 12, 16)
    >>> emb = torch.randn(2, 32)
    >>> y = block(x, emb)
    >>> y.shape
    torch.Size([2, 16, 4, 12, 16])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        up: bool = False,
        down: bool = False,
        attention: bool = False,
        num_heads: int | None = None,
        dropout: float = 0.0,
        skip_scale: float = 1.0,
        eps: float = 1e-5,
        resample_filter: List[int] = [1, 1],
        resample_proj: bool = False,
        adaptive_scale: bool = True,
        activation: Literal["silu", "gelu"] = "silu",
        init: Dict[str, Any] = dict(),
        init_zero: Dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
    ):
        super().__init__(meta=ModelMetaData())

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.attention = attention
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.act = silu if activation == "silu" else torch.nn.functional.gelu

        self.norm0 = GroupNorm3D(num_channels=in_channels, eps=eps)
        self.conv0 = Conv3D(
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
        self.norm1 = GroupNorm3D(num_channels=out_channels, eps=eps)
        self.conv1 = Conv3D(
            in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv3D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.attention:
            self.attn = UNetAttention3D(
                out_channels=out_channels,
                num_heads=num_heads if num_heads is not None else 1,
                eps=eps,
                init=init,
                init_zero=init_zero,
                init_attn=init_attn,
            )

    def forward(
        self,
        x: Float[torch.Tensor, "B C_in D H W"],
        emb: Float[torch.Tensor, "B C_emb"],
    ) -> Float[torch.Tensor, "B C_out D_out H_out W_out"]:
        orig = x

        # First norm + conv (with optional up/down)
        x = self.conv0(self.act(self.norm0(x)))

        # Affine conditioning from emb, broadcast over spatial dims
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = self.act(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = self.act(self.norm1(x.add_(params)))

        # Second conv with dropout and residual connection
        x = self.conv1(dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        # Optional self-attention with residual scaling
        if self.attention:
            x = self.attn(x)
            x = x * self.skip_scale

        return x
