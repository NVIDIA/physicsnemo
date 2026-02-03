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

import importlib

import torch
from einops import rearrange
from torch.nn.functional import elu, gelu, leaky_relu, relu, sigmoid, silu, tanh

from physicsnemo.nn.module.utils.utils import _validate_amp

# Import apex GroupNorm if installed only
_is_apex_available = False
if torch.cuda.is_available():
    try:
        apex_gn_module = importlib.import_module("apex.contrib.group_norm")
        ApexGroupNorm = getattr(apex_gn_module, "GroupNorm")
        _is_apex_available = True
    except ImportError:
        pass


def _compute_groupnorm_groups(
    num_channels: int,
    num_groups: int = 32,
    min_channels_per_group: int = 4,
) -> int:
    """
    Compute the number of groups for GroupNorm based on the number of channels
    and the minimum number of channels per group.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor.
    num_groups : int, optional, default=32
        Desired number of groups to divide the input channels.
        This might be adjusted based on the ``min_channels_per_group``.
    min_channels_per_group : int, optional, default=4
        Minimum channels required per group. This ensures that no group has fewer
        channels than this number.

    Returns
    -------
    int
        The number of groups to use for GroupNorm.
    """
    num_groups: int = min(
        num_groups,
        (num_channels + min_channels_per_group - 1) // min_channels_per_group,
    )
    if num_channels % num_groups != 0:
        raise ValueError(
            "num_channels must be divisible by num_groups or min_channels_per_group"
        )
    return num_groups


def get_group_norm(
    num_channels: int,
    num_groups: int = 32,
    min_channels_per_group: int = 4,
    eps: float = 1e-5,
    use_apex_gn: bool = False,
    act: str | None = None,
    amp_mode: bool = False,
) -> torch.nn.Module:
    """
    Utility function to get the GroupNorm layer, either from apex or from torch.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor.
    num_groups : int, optional, default=32
        Desired number of groups to divide the input channels.
        This might be adjusted based on the ``min_channels_per_group``.
    min_channels_per_group : int, optional, default=4
        Minimum channels required per group. This ensures that no group has fewer
        channels than this number.
    eps : float, optional, default=1e-5
        A small number added to the variance to prevent division by zero.
    use_apex_gn : bool, optional, default=False
        A boolean flag indicating whether we want to use Apex GroupNorm for NHWC layout.
        Need to set this as False on cpu.
    act : str, optional, default=None
        The activation function to use when fusing activation with GroupNorm.
    amp_mode : bool, optional, default=False
        A boolean flag indicating whether mixed-precision (AMP) training is enabled.

    Returns
    -------
    torch.nn.Module
        The GroupNorm layer. If ``use_apex_gn`` is ``True``, returns an
        ApexGroupNorm layer, otherwise returns an instance of
        :class:`~physicsnemo.nn.GroupNorm`.

    .. note::

    If ``num_channels`` is not divisible by ``num_groups``, the actual number
    of groups might be adjusted to satisfy the ``min_channels_per_group``
    condition.
    """
    if use_apex_gn and not _is_apex_available:
        raise ValueError("'apex' is not installed, set `use_apex_gn=False`")

    act: str | None = act.lower() if act else act
    if use_apex_gn:
        # adjust number of groups to be consistent with GroupNorm
        num_groups: int = _compute_groupnorm_groups(
            num_channels, num_groups, min_channels_per_group
        )
        return ApexGroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=True,
            act=act,
        )
    else:
        return GroupNorm(
            num_channels=num_channels,
            num_groups=num_groups,
            min_channels_per_group=min_channels_per_group,
            eps=eps,
            act=act,
            amp_mode=amp_mode,
        )


class GroupNorm(torch.nn.Module):
    """
    A custom Group Normalization layer implementation.

    Group Normalization (GN) divides the channels of the input tensor into groups and
    normalizes the features within each group independently. It does not require the
    batch size as in Batch Normalization, making it suitable for batch sizes of any size
    or even for batch-free scenarios.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor.
    num_groups : int, optional, default=32
        Desired number of groups to divide the input channels.
        This might be adjusted based on the ``min_channels_per_group``.
    min_channels_per_group : int, optional, default=4
        Minimum channels required per group. This ensures that no group has fewer
        channels than this number.
    eps : float, optional, default=1e-5
        A small number added to the variance to prevent division by zero.
    use_apex_gn : bool, optional, default=False
        Deprecated. Please use
        :func:`~physicsnemo.nn.get_group_norm` instead.
    fused_act : bool, optional, default=False
        Deprecated. Please use
        :func:`~physicsnemo.nn.get_group_norm` instead.
    act : str, optional, default=None
        The activation function to use when fusing activation with GroupNorm.
    amp_mode : bool, optional, default=False
        A boolean flag indicating whether mixed-precision (AMP) training is
        enabled.

    Forward
    -------
    x : torch.Tensor
        4-D input tensor of shape :math:`(B, C, H, W)`, where :math:`B` is batch
        size, :math:`C` is ``num_channels``, and :math:`H, W` are spatial
        dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of the same shape as input: :math:`(B, C, H, W)`.

    .. note::

    If ``num_channels`` is not divisible by ``num_groups``, the actual number of
    groups might be adjusted to satisfy the ``min_channels_per_group`` condition.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        min_channels_per_group: int = 4,
        eps: float = 1e-5,
        use_apex_gn: bool = False,
        fused_act: bool = False,
        act: str | None = None,
        amp_mode: bool = False,
    ):
        super().__init__()
        # backwards compatibility warnings
        if use_apex_gn:
            raise ValueError(
                "'use_apex_gn' is deprecated. Please use 'get_group_norm' to enable "
                "Apex-based group norm."
            )
        if fused_act:
            raise ValueError(
                "'fused_act' is deprecated and only supported for Apex-based group norm. "
                "Please use `get_group_norm` to enable fused activations."
            )

        # initialize groupnorm
        self.num_groups: int = _compute_groupnorm_groups(
            num_channels, num_groups, min_channels_per_group
        )
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        self.act = act.lower() if act else act
        self.act_fn = None
        if self.act is not None:
            self.act_fn = self.get_activation_function()
        self.amp_mode = amp_mode

    def forward(self, x):
        weight, bias = self.weight, self.bias
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if weight.dtype != x.dtype:
                weight = self.weight.to(x.dtype)
            if bias.dtype != x.dtype:
                bias = self.bias.to(x.dtype)

        if self.training:
            # Use default torch implementation of GroupNorm for training
            # This does not support channels last memory format
            x = torch.nn.functional.group_norm(
                x,
                num_groups=self.num_groups,
                weight=weight,
                bias=bias,
                eps=self.eps,
            )
        else:
            # Use custom GroupNorm implementation that supports channels last
            # memory layout for inference
            x = rearrange(x, "b (g c) h w -> b g c h w", g=self.num_groups)

            mean = x.mean(dim=[2, 3, 4], keepdim=True)
            var = x.var(dim=[2, 3, 4], keepdim=True)

            x = (x - mean) * (var + self.eps).rsqrt()
            x = rearrange(x, "b g c h w -> b (g c) h w")

            weight = rearrange(weight, "c -> 1 c 1 1")
            bias = rearrange(bias, "c -> 1 c 1 1")
            x = x * weight + bias

        if self.act_fn is not None:
            x = self.act_fn(x)
        return x

    def get_activation_function(self):
        """
        Get activation function given string input
        """

        activation_map = {
            "silu": silu,
            "relu": relu,
            "leaky_relu": leaky_relu,
            "sigmoid": sigmoid,
            "tanh": tanh,
            "gelu": gelu,
            "elu": elu,
        }

        act_fn = activation_map.get(self.act, None)
        if act_fn is None:
            raise ValueError(f"Unknown activation function: {self.act}")
        return act_fn
