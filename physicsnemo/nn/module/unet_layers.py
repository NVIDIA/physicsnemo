# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
from typing import Any, Dict, List, Set

import nvtx
import torch
from torch.nn.functional import silu

from physicsnemo.nn.module.attention_layers import UNetAttention as Attention
from physicsnemo.nn.module.conv_layers import Conv2d
from physicsnemo.nn.module.fully_connected_layers import Linear
from physicsnemo.nn.module.group_norm import get_group_norm
from physicsnemo.nn.module.utils.utils import _validate_amp


class UNetBlock(torch.nn.Module):
    """
    Unified U-Net block with optional up/downsampling and self-attention. Represents
    the union of all features employed by the DDPM++, NCSN++, and ADM architectures.

    Parameters:
    -----------
    in_channels : int
        Number of input channels :math:`C_{in}`.
    out_channels : int
        Number of output channels :math:`C_{out}`.
    emb_channels : int
        Number of embedding channels :math:`C_{emb}`.
    up : bool, optional, default=False
        If True, applies upsampling in the forward pass.
    down : bool, optional, default=False
        If True, applies downsampling in the forward pass.
    attention : bool, optional, default=False
        If True, enables the self-attention mechanism in the block.
    num_heads : int, optional, default=None
        Number of attention heads. If None, defaults to :math:`C_{out} / 64`.
    channels_per_head : int, optional, default=64
        Number of channels per attention head.
    dropout : float, optional, default=0.0
        Dropout probability.
    skip_scale : float, optional, default=1.0
        Scale factor applied to skip connections.
    eps : float, optional, default=1e-5
        Epsilon value used for normalization layers.
    resample_filter : List[int], optional, default=``[1, 1]``
        Filter for resampling layers.
    resample_proj : bool, optional, default=False
        If True, resampling projection is enabled.
    adaptive_scale : bool, optional, default=True
        If True, uses adaptive scaling in the forward pass.
    init : dict, optional, default=``{}``
        Initialization parameters for convolutional and linear layers.
    init_zero : dict, optional, default=``{'init_weight': 0}``
        Initialization parameters with zero weights for certain layers.
    init_attn : dict, optional, default=``None``
        Initialization parameters specific to attention mechanism layers.
        Defaults to ``init`` if not provided.
    use_apex_gn : bool, optional, default=False
        A boolean flag indicating whether we want to use Apex GroupNorm for NHWC layout.
        Need to set this as False on cpu.
    act : str, optional, default=None
        The activation function to use when fusing activation with GroupNorm.
    fused_conv_bias: bool, optional, default=False
        A boolean flag indicating whether bias will be passed as a parameter of conv2d.
    profile_mode: bool, optional, default=False
        A boolean flag indicating whether to enable all nvtx annotations during profiling.
    amp_mode : bool, optional, default=False
        A boolean flag indicating whether mixed-precision (AMP) training is
        enabled.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, H, W)`, where :math:`B` is batch
        size, :math:`C_{in}` is ``in_channels``, and :math:`H, W` are spatial
        dimensions.
    emb : torch.Tensor
        Embedding tensor of shape :math:`(B, C_{emb})`, where :math:`B` is batch
        size, and :math:`C_{emb}` is ``emb_channels``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, H, W)`, where :math:`B` is batch
        size, :math:`C_{out}` is ``out_channels``, and :math:`H, W` are spatial
        dimensions.
    """

    # NOTE: these attributes have specific usage in old checkpoints, do not
    # reuse them!
    _reserved_attributes: Set[str] = set(["norm2", "qkv", "proj"])

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        up: bool = False,
        down: bool = False,
        attention: bool = False,
        num_heads: int | None = None,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        skip_scale: float = 1.0,
        eps: float = 1e-5,
        resample_filter: List[int] = [1, 1],
        resample_proj: bool = False,
        adaptive_scale: bool = True,
        init: Dict[str, Any] = dict(),
        init_zero: Dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
        use_apex_gn: bool = False,
        act: str = "silu",
        fused_conv_bias: bool = False,
        profile_mode: bool = False,
        amp_mode: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else (
                num_heads
                if num_heads is not None
                else out_channels // channels_per_head
            )
        )
        self.attention = attention
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.profile_mode = profile_mode
        self.amp_mode = amp_mode
        self.norm0 = get_group_norm(
            num_channels=in_channels,
            eps=eps,
            use_apex_gn=use_apex_gn,
            act=act,
            amp_mode=amp_mode,
        )
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            amp_mode=amp_mode,
            **init,
        )
        if self.adaptive_scale:
            self.norm1 = get_group_norm(
                num_channels=out_channels,
                eps=eps,
                use_apex_gn=use_apex_gn,
                amp_mode=amp_mode,
            )
        else:
            self.norm1 = get_group_norm(
                num_channels=out_channels,
                eps=eps,
                use_apex_gn=use_apex_gn,
                act=act,
                amp_mode=amp_mode,
            )
        self.conv1 = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=3,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
            **init_zero,
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            fused_conv_bias = fused_conv_bias if kernel != 0 else False
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                fused_conv_bias=fused_conv_bias,
                amp_mode=amp_mode,
                **init,
            )

        if self.attention:
            self.attn = Attention(
                out_channels=out_channels,
                num_heads=self.num_heads,
                eps=eps,
                init_zero=init_zero,
                init_attn=init_attn,
                init=init,
                use_apex_gn=use_apex_gn,
                amp_mode=amp_mode,
                fused_conv_bias=fused_conv_bias,
            )
        else:
            self.attn = None
        # A hook to migrate legacy attention module
        self.register_load_state_dict_pre_hook(self._migrate_attention_module)

    def forward(self, x, emb):
        with (
            nvtx.annotate(message="UNetBlock", color="purple")
            if self.profile_mode
            else contextlib.nullcontext()
        ):
            orig = x
            x = self.conv0(self.norm0(x))
            params = self.affine(emb).unsqueeze(2).unsqueeze(3)
            _validate_amp(self.amp_mode)
            if not self.amp_mode:
                if params.dtype != x.dtype:
                    params = params.to(x.dtype)  # type: ignore

            if self.adaptive_scale:
                scale, shift = params.chunk(chunks=2, dim=1)
                x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
            else:
                x = self.norm1(x.add_(params))

            x = self.conv1(
                torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            )
            x = x.add_(self.skip(orig) if self.skip is not None else orig)
            x = x * self.skip_scale

            if self.attn:
                x = self.attn(x)
                x = x * self.skip_scale
            return x

    def __setattr__(self, name, value):
        """Prevent setting attributes with reserved names.

        Parameters
        ----------
        name : str
            Attribute name.
        value : Any
            Attribute value.
        """
        if name in getattr(self.__class__, "_reserved_attributes", set()):
            raise AttributeError(f"Attribute '{name}' is reserved and cannot be set.")
        super().__setattr__(name, value)

    @staticmethod
    def _migrate_attention_module(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """``load_state_dict`` pre-hook that handles legacy checkpoints that
        stored attention layers at root.

        The earliest versions of ``UNetBlock`` stored the attention-layer
        parameters directly on the block using attribute names contained in
        ``_reserved_attributes``.  These have since been moved under the
        dedicated ``attn`` sub-module.  This helper migrates the parameter
        names so that older checkpoints can still be loaded.
        """

        _mapping = {
            f"{prefix}norm2.weight": f"{prefix}attn.norm.weight",
            f"{prefix}norm2.bias": f"{prefix}attn.norm.bias",
            f"{prefix}qkv.weight": f"{prefix}attn.qkv.weight",
            f"{prefix}qkv.bias": f"{prefix}attn.qkv.bias",
            f"{prefix}proj.weight": f"{prefix}attn.proj.weight",
            f"{prefix}proj.bias": f"{prefix}attn.proj.bias",
        }

        for old_key, new_key in _mapping.items():
            if old_key in state_dict:
                # NOTE: Only migrate if destination key not already present to
                # avoid accidental overwriting when both are present.
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)
                else:
                    raise ValueError(
                        f"Checkpoint contains both legacy and new keys for {old_key}"
                    )
