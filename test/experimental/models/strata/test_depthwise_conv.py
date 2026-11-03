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

import pytest
import torch

import physicsnemo.experimental.models.strata.layers as layers
from physicsnemo.experimental.models.strata.layers import DepthwiseConv


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "limit,shape,padding_mode,padding,stride,dilation,area",
    [
        (144, (5, 2, 4, 4), "replicate", 1, 1, 1, 36),
        (72, (2, 4, 4, 4), "replicate", 1, 1, 1, 36),
        (1000, (2, 4, 4, 4), "replicate", 1, 1, 1, 36),
        (64, (2, 4, 4, 4), "zeros", 3, 1, 1, 64),
        (162, (3, 3, 5, 7), "replicate", (2, 1), (1, 2), (2, 1), 81),
    ],
)
def test_auto_depthwise_outputs_gradients_and_index_bound(
    monkeypatch,
    device,
    bias,
    limit,
    shape,
    padding_mode,
    padding,
    stride,
    dilation,
    area,
):
    r"""Auto chunks agree with grouped Conv2d and bound padded/output elements."""
    monkeypatch.setattr(layers, "_MAX_CONV_ELEMENTS", limit)
    reference = torch.nn.Conv2d(
        shape[1],
        shape[1],
        3,
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=shape[1],
        padding_mode=padding_mode,
        bias=bias,
    ).to(device=device, dtype=torch.float64)
    conv = DepthwiseConv(
        shape[1],
        3,
        padding=padding,
        stride=stride,
        dilation=dilation,
        padding_mode=padding_mode,
        bias=bias,
        chunk_size="auto",
    ).to(device=device, dtype=torch.float64)
    conv.load_state_dict(reference.state_dict())
    x = torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    expected = reference(reference_x)
    original_build = layers._build_chunked_depthwise_conv
    calls = []

    def checked_build(conv, chunk_size, *, batch_chunk_size=None):
        batch = shape[0] if batch_chunk_size is None else batch_chunk_size
        channels = shape[1] if chunk_size is None else chunk_size
        assert batch * channels * area <= limit
        calls.append((chunk_size, batch_chunk_size))
        return original_build(conv, chunk_size, batch_chunk_size=batch_chunk_size)

    monkeypatch.setattr(layers, "_build_chunked_depthwise_conv", checked_build)
    actual = conv(x)
    torch.testing.assert_close(actual, expected)
    assert len(calls) == 1
    if limit == 1000:
        assert calls == [(None, None)]
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(conv.weight.grad, reference.weight.grad)
    if bias:
        torch.testing.assert_close(conv.bias.grad, reference.bias.grad)


def test_auto_depthwise_rejects_oversized_plane(monkeypatch):
    r"""Even one padded channel must fit the indexing bound."""
    monkeypatch.setattr(layers, "_MAX_CONV_ELEMENTS", 35)
    conv = DepthwiseConv(2, 3, padding=1, padding_mode="replicate", chunk_size="auto")
    with pytest.raises(ValueError, match="one spatial plane"):
        conv(torch.ones(2, 2, 4, 4))


@pytest.mark.parametrize("chunk_size", ["auto", 2])
def test_strata_conditioning_chunk_mode(chunk_size):
    r"""Pixel blocks propagate automatic and explicit conditioning chunk sizes."""
    block = layers.StrataPixel3DBlock(
        dim=8,
        cond_dim=8,
        num_heads=1,
        pixels_per_patch=1,
        adaln_mode="bilinear_dw",
        chunk_size_grouped_conv=chunk_size,
    )
    assert block.adaln_bilinear_dw_conv.chunk_size == chunk_size


@pytest.mark.parametrize("chunk_size", ["auto", 2])
@pytest.mark.parametrize("padding_mode", ["zeros", "replicate"])
@pytest.mark.parametrize(
    "padding,kernel_size,dilation",
    [("same", 3, 1), ("same", 4, 1), ("same", 4, 2), ("valid", 3, 1)],
)
def test_depthwise_string_padding(
    monkeypatch, device, chunk_size, padding_mode, padding, kernel_size, dilation
):
    r"""String padding agrees with Conv2d, including asymmetric same padding."""
    monkeypatch.setattr(layers, "_MAX_CONV_ELEMENTS", 160)
    reference = torch.nn.Conv2d(
        3,
        3,
        kernel_size,
        padding=padding,
        dilation=dilation,
        padding_mode=padding_mode,
        groups=3,
    ).to(device=device, dtype=torch.float64)
    conv = DepthwiseConv(
        3,
        kernel_size,
        padding=padding,
        dilation=dilation,
        padding_mode=padding_mode,
        chunk_size=chunk_size,
    ).to(device=device, dtype=torch.float64)
    conv.load_state_dict(reference.state_dict())
    x = torch.randn(2, 3, 5, 7, device=device, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    expected = reference(reference_x)
    actual = conv(x)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(conv.weight.grad, reference.weight.grad)
    torch.testing.assert_close(conv.bias.grad, reference.bias.grad)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "padding,padding_mode", [(1, "zeros"), ("same", "zeros"), ("same", "replicate")]
)
def test_auto_depthwise_empty_batch(device, bias, padding, padding_mode):
    r"""Empty batches preserve Conv2d output shape, dtype, and zero gradients."""
    reference = torch.nn.Conv2d(
        3,
        3,
        3,
        padding=padding,
        padding_mode=padding_mode,
        groups=3,
        bias=bias,
    ).to(device=device, dtype=torch.float64)
    conv = DepthwiseConv(
        3,
        3,
        padding=padding,
        padding_mode=padding_mode,
        bias=bias,
        chunk_size="auto",
    ).to(device=device, dtype=torch.float64)
    conv.load_state_dict(reference.state_dict())
    x = torch.empty(0, 3, 5, 7, device=device, dtype=torch.float64, requires_grad=True)
    actual = conv(x)
    torch.testing.assert_close(actual, reference(x.detach()))
    actual.sum().backward()
    assert x.grad.shape == x.shape
    torch.testing.assert_close(conv.weight.grad, torch.zeros_like(conv.weight))
    if bias:
        torch.testing.assert_close(conv.bias.grad, torch.zeros_like(conv.bias))
