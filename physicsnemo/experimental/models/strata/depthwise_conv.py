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

r"""Depthwise 2D convolution with a ``torch.vmap`` fallback for very large tensors.

``torch.nn.functional.conv2d`` has an internal element-count limit that a
depthwise convolution over a high-resolution pixel grid (as used by Strata's
``bilinear_dw`` adaptive-layer-norm path) can exceed. :class:`DepthwiseConv`
optionally chunks the convolution with :func:`torch.vmap` to stay under that
limit.
"""

from __future__ import annotations

import warnings
from functools import partial

import torch
import torch.nn as nn

__all__ = ["DepthwiseConv"]


def _apply_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    stride,
    padding,
    dilation,
    padding_mode,
) -> torch.Tensor:
    r"""Apply a single-sample 2D convolution (helper for :func:`torch.vmap`).

    Parameters
    ----------
    x : torch.Tensor
        Single-sample input of shape :math:`(C, H, W)`.
    weight : torch.Tensor
        Convolution weight of shape :math:`(C, 1, k_h, k_w)`.
    bias : torch.Tensor
        Bias of shape :math:`(C,)`.
    stride, padding, dilation : tuple
        Standard convolution parameters.
    padding_mode : str
        Padding mode; non-``"zeros"`` modes are applied explicitly with
        :func:`torch.nn.functional.pad`.

    Returns
    -------
    torch.Tensor
        Convolved single-sample output of shape :math:`(C, H', W')`.
    """
    x = x.unsqueeze(0)
    w = weight.unsqueeze(0)
    bias = bias.unsqueeze(0)
    if padding_mode != "zeros":
        pad_h, pad_w = padding
        x = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=padding_mode)
        padding = (0, 0)
    return torch.nn.functional.conv2d(
        x, w, bias=bias, stride=stride, padding=padding, dilation=dilation
    )[0]


def _build_chunked_depthwise_conv(conv: nn.Conv2d, chunk_size: int = 4):
    r"""Build a chunked ``torch.vmap`` depthwise convolution callable.

    The returned callable takes ``(x, weight, bias)`` explicitly and captures
    only the static convolution configuration (stride / padding / dilation /
    padding mode) -- **not** the module or its parameters. The forward pass
    threads ``self.weight`` / ``self.bias`` in live, so the callable stays
    correct across ``deepcopy`` (e.g. EMA / ``AveragedModel``) and ``.to(device)``.

    Parameters
    ----------
    conv : torch.nn.Conv2d
        A depthwise convolution (``groups == out_channels``); used only to read
        its static configuration.
    chunk_size : int, optional, default=4
        Channel chunk size for the inner :func:`torch.vmap`.

    Returns
    -------
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
        A function mapping ``(x, weight, bias)`` with
        :math:`x \in (B, C, H, W)` to :math:`(B, C, H', W')`.
    """
    if conv.groups != conv.out_channels:
        raise ValueError("only works with depthwise convolution")

    # Inner vmap over channels (chunked); outer vmap over the batch dim with the
    # weight / bias broadcast (``in_dims=(0, None, None)``). This is equivalent
    # to mapping ``func(x[i], weight, bias)`` over the batch, but keeps weight
    # and bias as explicit arguments rather than closed-over module state.
    per_sample = torch.vmap(
        partial(
            _apply_conv2d,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            padding_mode=conv.padding_mode,
        ),
        chunk_size=chunk_size,
    )
    return torch.vmap(per_sample, in_dims=(0, None, None))


class DepthwiseConv(torch.nn.Conv2d):
    r"""Depthwise 2D convolution that chunks large inputs via ``torch.vmap``.

    A :class:`torch.nn.Conv2d` with ``groups == channels``. When ``chunk_size``
    is provided, the forward pass uses a chunked :func:`torch.vmap`
    implementation (see :func:`_build_chunked_depthwise_conv`) that avoids the
    conv2d element-count limit for very large tensors; otherwise it falls back
    to the standard convolution and warns if a single chunk would exceed the
    limit.

    Parameters
    ----------
    channels : int
        Number of input / output channels (the convolution is depthwise).
    *args
        Positional arguments forwarded to :class:`torch.nn.Conv2d` (e.g.
        ``kernel_size``).
    chunk_size : int, optional, default=None
        Channel chunk size for the ``torch.vmap`` path. If ``None``, the
        standard convolution is used.
    **kwargs
        Keyword arguments forwarded to :class:`torch.nn.Conv2d`. A ``groups``
        argument is rejected (the convolution is always depthwise).

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, H, W)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C, H', W')`.
    """

    def __init__(self, channels: int, *args, chunk_size: int | None = None, **kwargs):
        if "groups" in kwargs:
            raise ValueError("DepthwiseConv does not accept a groups argument")
        super().__init__(channels, channels, *args, **kwargs, groups=channels)
        self.chunk_size = chunk_size
        # Build the chunked callable once from the static conv configuration. It
        # does not capture ``self`` or the parameters, so ``forward`` can thread
        # ``self.weight`` / ``self.bias`` in live -- this keeps the module correct
        # after ``deepcopy`` (EMA / ``AveragedModel``) and ``.to(device)``, unlike
        # binding a ``self``-capturing closure to ``self.forward``.
        self._chunked_conv = (
            _build_chunked_depthwise_conv(self, chunk_size)
            if chunk_size is not None
            else None
        )

    def forward(self, x, *args, **kwargs):
        if self._chunked_conv is not None:
            # Chunked path: read parameters live so a deep-copied module uses its
            # own (possibly relocated / re-initialized) weights, not the source's.
            bias = self.bias
            if bias is None:
                bias = torch.zeros(
                    self.out_channels, device=self.weight.device, dtype=self.weight.dtype
                )
            return self._chunked_conv(x, self.weight, bias)

        # Standard (non-chunked) path: warn if conv2d would exceed its 32-bit
        # indexing limit (INT_MAX elements), then defer to nn.Conv2d. The limit is
        # element-count, not bytes -- it trips identically for fp32 and bf16.
        if x.numel() > torch.iinfo(torch.int32).max:
            warnings.warn(
                f"conv2d input has {x.numel()} elements (> 2**31 - 1, the 32-bit "
                f"indexing limit), so it will raise a RuntimeError "
                f"(canUse32BitIndexMath). Set the chunk_size option to enable "
                f"chunking, which keeps each conv2d call under the limit.",
                stacklevel=2,
            )

        return super().forward(x, *args, **kwargs)
