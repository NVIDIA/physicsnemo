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

import numpy as np
import pytest
import torch

from physicsnemo.nn.module.hpx import (
    HEALPixAvgPool,
    HEALPixLayer,
    HEALPixMaxPool,
    HEALPixPadding,
    HEALPixPatchDetokenizer,
    HEALPixPatchTokenizer,
)
from physicsnemo.nn.module.hpx.padding import (
    HEALPixFoldFaces,
    HEALPixPaddingv2,
    HEALPixUnfoldFaces,
)
from physicsnemo.nn.module.hpx.tokenizer import (
    CalendarEmbedding,
)
from test import common
from test.conftest import requires_module
from test.nn.module.healpix_helpers import MulX, distinct_face_tensor


@pytest.fixture
def test_data():
    def generate_test_data(faces=12, channels=2, img_size=16, device="cpu"):
        test = torch.eye(img_size, device=device)
        test = test[(None,) * 2]
        return test.expand([faces, channels, -1, -1])

    return generate_test_data


def _padded_faces(pad_func, size, device):
    """Run ``pad_func`` on a distinct-face pattern and return the padded
    output alongside a per-face lookup of the (unpadded) input, so
    correctness tests can slice out expected neighbor regions."""
    pattern = distinct_face_tensor(size=size, device=device)
    out = pad_func(pattern)
    faces = {i: pattern[i, 0] for i in range(12)}
    return out, faces


def _arange_grid(size, offset, device):
    """A square, value-distinguishable tensor for isolated corner-blend
    tests, offset so the two operands of tl()/br() never overlap in value."""
    return offset + torch.arange(
        size * size, device=device, dtype=torch.float32
    ).reshape(size, size)


def _make_pad_func(pad_cls, padding, device):
    """Construct and move a ``HEALPixPadding``/``HEALPixPaddingv2`` instance
    to ``device`` in one call, since every padding test needs this."""
    return pad_cls(padding=padding).to(device)


def _padded_shape(batch_faces, channels, size, padding):
    """Expected output shape after padding a ``(batch_faces, channels, size,
    size)`` tensor by ``padding`` on each side of the last two dims."""
    return torch.Size([batch_faces, channels, size + 2 * padding, size + 2 * padding])


@requires_module("earth2grid")
def test_HEALPixFoldFaces_initialization(device, pytestconfig):
    fold_func = HEALPixFoldFaces()
    assert isinstance(fold_func, HEALPixFoldFaces)


@requires_module("earth2grid")
def test_HEALPixFoldFaces_forward(device, pytestconfig):
    fold_func = HEALPixFoldFaces()

    tensor_size = torch.randint(low=2, high=4, size=(5,)).tolist()
    output_size = (tensor_size[0] * tensor_size[1], *tensor_size[2:])
    invar = torch.ones(*tensor_size, device=device)

    outvar = fold_func(invar)
    assert outvar.shape == output_size

    fold_func = HEALPixFoldFaces(enable_nhwc=True)
    assert fold_func(invar).shape == outvar.shape
    assert fold_func(invar).stride() != outvar.stride()


@requires_module("earth2grid")
def test_HEALPixFoldFaces_forward_correctness(device, pytestconfig):
    fold_func = HEALPixFoldFaces()

    batch, faces, channels, height, width = 2, 3, 2, 4, 5
    invar = torch.arange(
        batch * faces * channels * height * width, device=device, dtype=torch.float32
    ).reshape(batch, faces, channels, height, width)

    outvar = fold_func(invar)

    for b in range(batch):
        for f in range(faces):
            assert torch.equal(outvar[b * faces + f], invar[b, f])


@requires_module("earth2grid")
def test_HEALPixFoldFaces_forward_invalid_ndim(device, pytestconfig):
    fold_func = HEALPixFoldFaces()

    invar = torch.randn(2, 3, 4, device=device)  # only 3D, needs 5D
    with pytest.raises(ValueError, match="requires 5D tensor"):
        fold_func(invar)


@requires_module("earth2grid")
def test_HEALPixUnfoldFaces_forward_correctness(device, pytestconfig):
    num_faces = 12
    unfold_func = HEALPixUnfoldFaces(num_faces=num_faces)

    batch, channels, height, width = 2, 2, 4, 5
    invar = torch.arange(
        batch * num_faces * channels * height * width,
        device=device,
        dtype=torch.float32,
    ).reshape(batch * num_faces, channels, height, width)

    outvar = unfold_func(invar)

    for b in range(batch):
        for f in range(num_faces):
            assert torch.equal(outvar[b, f], invar[b * num_faces + f])


@requires_module("earth2grid")
def test_HEALPixUnfoldFaces_forward_invalid_ndim(device, pytestconfig):
    unfold_func = HEALPixUnfoldFaces(num_faces=12)

    invar = torch.randn(2, 3, 4, device=device)  # only 3D, needs 4D
    with pytest.raises(ValueError, match="requires 4D tensor"):
        unfold_func(invar)


@requires_module("earth2grid")
def test_HEALPixUnfoldFaces_forward_invalid_batch_size(device, pytestconfig):
    unfold_func = HEALPixUnfoldFaces(num_faces=12)

    # batch_faces=13 is not a multiple of num_faces=12
    invar = torch.randn(13, 2, 4, 4, device=device)
    with pytest.raises(ValueError, match="invalid batch size"):
        unfold_func(invar)


@requires_module("earth2grid")
def test_HEALPixFoldFaces_UnfoldFaces_roundtrip(device, pytestconfig):
    num_faces = 12
    fold_func = HEALPixFoldFaces()
    unfold_func = HEALPixUnfoldFaces(num_faces=num_faces)

    batch, channels, height, width = 3, 2, 4, 4
    invar = torch.randn(batch, num_faces, channels, height, width, device=device)

    folded = fold_func(invar)
    assert folded.shape == (batch * num_faces, channels, height, width)

    unfolded = unfold_func(folded)
    assert torch.equal(unfolded, invar)

    refolded = fold_func(unfolded)
    assert torch.equal(refolded, folded)


@requires_module("earth2grid")
def test_HEALPixUnfoldFaces_initialization(device, pytestconfig):
    unfold_func = HEALPixUnfoldFaces()
    assert isinstance(unfold_func, HEALPixUnfoldFaces)


@requires_module("earth2grid")
def test_HEALPixUnfoldFaces_forward(device, pytestconfig):
    num_faces = 12
    unfold_func = HEALPixUnfoldFaces()

    tensor_size = torch.randint(low=1, high=4, size=(4,)).tolist()
    output_size = (tensor_size[0], num_faces, *tensor_size[1:])

    tensor_size[0] *= num_faces
    invar = torch.ones(*tensor_size, device=device)

    outvar = unfold_func(invar)
    assert outvar.shape == output_size


@requires_module("earth2grid")
@pytest.mark.parametrize("padding", [2, 3, 4])
def test_HEALPixPadding_initialization(device, padding, pytestconfig):
    pad_func = HEALPixPadding(padding)
    assert isinstance(pad_func, HEALPixPadding)


@requires_module("earth2grid")
@pytest.mark.parametrize("padding", [2, 3, 4])
def test_HEALPixPadding_forward(device, padding, pytestconfig):
    num_faces = 12
    batch_size = 2
    pad_func = HEALPixPadding(padding)

    with pytest.raises(
        ValueError, match=("invalid value for 'padding', expected int > 0 but got 0")
    ):
        HEALPixPadding(0)

    hw_size = torch.randint(low=4, high=24, size=(1,)).tolist()
    c_size = torch.randint(low=3, high=7, size=(1,)).tolist()
    hw_size = np.asarray(hw_size + hw_size)

    tensor_size = (batch_size * num_faces, *c_size, *hw_size)
    invar = torch.rand(tensor_size, device=device)

    hw_padded_size = hw_size + (2 * padding)
    out_size = (batch_size * num_faces, *c_size, *hw_padded_size)

    outvar = pad_func(invar)
    assert outvar.shape == out_size


@requires_module("earth2grid")
def test_HEALPixPadding_forward_invalid_ndim(device, pytestconfig):
    pad_func = _make_pad_func(HEALPixPadding, 1, device)

    invar = torch.randn(2, 3, 4, device=device)  # only 3D, needs 4D
    with pytest.raises(ValueError, match="requires a 4D tensor"):
        pad_func(invar)


@requires_module("earth2grid")
@pytest.mark.parametrize(
    "pad_cls,batch_faces,channels",
    [(HEALPixPadding, 12, 2), (HEALPixPaddingv2, 24, 3)],
    ids=["HEALPixPadding", "HEALPixPaddingv2"],
)
def test_padding_forward_skips_validation_when_compiling(
    device, monkeypatch, pad_cls, batch_faces, channels, pytestconfig
):
    """When invoked from inside a compiled graph, ``torch.compiler.is_compiling()``
    reports ``True`` and shape validation (in ``HEALPixPadding``/``HEALPixFoldFaces``/
    ``HEALPixUnfoldFaces``) must be skipped without affecting correctness."""
    if pad_cls is HEALPixPaddingv2 and device == "cpu":
        pytest.skip("HEALPixPaddingv2 requires a CUDA device")

    padding = 2
    size = 4
    pad_func = _make_pad_func(pad_cls, padding, device)

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    invar = torch.randn(batch_faces, channels, size, size, device=device)
    outvar = pad_func(invar)

    assert outvar.shape == _padded_shape(batch_faces, channels, size, padding)


@requires_module("earth2grid")
@pytest.mark.parametrize(
    "pad_cls,batch_faces,channels",
    [(HEALPixPadding, 12, 2), (HEALPixPaddingv2, 24, 3)],
    ids=["HEALPixPadding", "HEALPixPaddingv2"],
)
def test_padding_forward_cuda_nvtx_skipped_without_cuda(
    device, monkeypatch, pad_cls, batch_faces, channels, pytestconfig
):
    """The nvtx range push/pop around the forward pass is gated on
    ``torch.cuda.is_available()``, independent of the tensor's actual
    device; verify the pass-through path when it reports unavailable."""
    if pad_cls is HEALPixPaddingv2 and device == "cpu":
        pytest.skip("HEALPixPaddingv2 requires a CUDA device")

    padding = 2
    size = 4
    pad_func = _make_pad_func(pad_cls, padding, device)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    invar = torch.randn(batch_faces, channels, size, size, device=device)
    outvar = pad_func(invar)

    assert outvar.shape == _padded_shape(batch_faces, channels, size, padding)


@requires_module("earth2grid")
def test_HEALPixPadding_forward_correctness_north(device, pytestconfig):
    """Face 0 is a northern-hemisphere face padded via ``pn``, which rotates
    its top/left neighbors. Verify against an independently computed expected
    tensor built from the same documented neighbor/rotation contract."""
    padding = 2
    size = 4
    d = (-2, -1)

    pad_func = _make_pad_func(HEALPixPadding, padding, device)
    out, f = _padded_faces(pad_func, size, device)

    p = padding
    center = torch.cat((f[1].rot90(1, d)[-p:, :], f[0], f[4][:p, :]), dim=0)
    left = torch.cat(
        (f[2].rot90(2, d)[-p:, -p:], f[3].rot90(-1, d)[:, -p:], f[3][:p, -p:]), dim=0
    )
    right = torch.cat((f[1][-p:, :p], f[5][:, :p], f[8][:p, :p]), dim=0)
    expected = torch.cat((left, center, right), dim=1)

    torch.testing.assert_close(out[0, 0], expected)


@requires_module("earth2grid")
def test_HEALPixPadding_forward_correctness_equator(device, pytestconfig):
    """Face 4 is an equatorial face padded via ``pe``, including the tl()/br()
    corner-blend helpers for its missing diagonal neighbors."""
    padding = 2
    size = 4

    pad_func = _make_pad_func(HEALPixPadding, padding, device)
    out, f = _padded_faces(pad_func, size, device)

    p = padding
    tl_corner = pad_func.tl(f[0], f[3])
    br_corner = pad_func.br(f[11], f[8])

    center = torch.cat((f[0][-p:, :], f[4], f[11][:p, :]), dim=0)
    left = torch.cat((tl_corner[-p:, -p:], f[3][:, -p:], f[7][:p, -p:]), dim=0)
    right = torch.cat((f[5][-p:, :p], f[8][:, :p], br_corner[:p, :p]), dim=0)
    expected = torch.cat((left, center, right), dim=1)

    torch.testing.assert_close(out[4, 0], expected)


@requires_module("earth2grid")
def test_HEALPixPadding_forward_correctness_south(device, pytestconfig):
    """Face 8 is a southern-hemisphere face padded via ``ps``, which rotates
    its bottom/right neighbors."""
    padding = 2
    size = 4
    d = (-2, -1)

    pad_func = _make_pad_func(HEALPixPadding, padding, device)
    out, f = _padded_faces(pad_func, size, device)

    p = padding
    center = torch.cat((f[5][-p:, :], f[8], f[11].rot90(1, d)[:p, :]), dim=0)
    left = torch.cat((f[0][-p:, -p:], f[4][:, -p:], f[11][:p, -p:]), dim=0)
    right = torch.cat(
        (f[9][-p:, :p], f[9].rot90(-1, d)[:, :p], f[10].rot90(2, d)[:p, :p]), dim=0
    )
    expected = torch.cat((left, center, right), dim=1)

    torch.testing.assert_close(out[8, 0], expected)


@requires_module("earth2grid")
def test_HEALPixPadding_tl_corner_blend(device, pytestconfig):
    """Directly verify the tl() diagonal-corner blend formula in isolation."""
    padding = 3
    size = 5
    pad_func = _make_pad_func(HEALPixPadding, padding, device)

    top = _arange_grid(size, offset=0, device=device)
    lft = _arange_grid(size, offset=1000, device=device)

    result = pad_func.tl(top, lft)

    p = padding
    expected = torch.zeros_like(top)[..., :p, :p]
    expected[..., -1, -1] = 0.5 * top[..., -1, 0] + 0.5 * lft[..., 0, -1]
    for i in range(1, p):
        expected[..., -i - 1, -i:] = top[..., -i - 1, :i]
        expected[..., -i:, -i - 1] = lft[..., :i, -i - 1]
        expected[..., -i - 1, -i - 1] = (
            0.5 * top[..., -i - 1, 0] + 0.5 * lft[..., 0, -i - 1]
        )

    torch.testing.assert_close(result, expected)


@requires_module("earth2grid")
def test_HEALPixPadding_br_corner_blend(device, pytestconfig):
    """Directly verify the br() diagonal-corner blend formula in isolation."""
    padding = 3
    size = 5
    pad_func = _make_pad_func(HEALPixPadding, padding, device)

    b = _arange_grid(size, offset=0, device=device)
    r = _arange_grid(size, offset=1000, device=device)

    result = pad_func.br(b, r)

    p = padding
    expected = torch.zeros_like(b)[..., :p, :p]
    expected[..., 0, 0] = 0.5 * b[..., 0, -1] + 0.5 * r[..., -1, 0]
    for i in range(1, p):
        expected[..., :i, i] = r[..., -i:, i]
        expected[..., i, :i] = b[..., i, -i:]
        expected[..., i, i] = 0.5 * b[..., i, -1] + 0.5 * r[..., -1, i]

    torch.testing.assert_close(result, expected)


@requires_module("earth2grid")
def test_HEALPixPaddingv2_forward(device, pytestconfig):
    """``HEALPixPaddingv2`` wraps the accelerated ``earth2grid`` padding kernel,
    which requires an actual CUDA device regardless of earth2grid's presence."""
    if device == "cpu":
        pytest.skip("HEALPixPaddingv2 requires a CUDA device")

    padding = 2
    size = 4
    pad_func = _make_pad_func(HEALPixPaddingv2, padding, device)

    invar = torch.randn(24, 3, size, size, device=device)
    outvar = pad_func(invar)

    assert outvar.shape == _padded_shape(24, 3, size, padding)


@requires_module("earth2grid")
@pytest.mark.parametrize("multiplier", [2, 3, 4])
def test_HEALPixLayer_initialization(device, multiplier, pytestconfig):
    layer = HEALPixLayer(layer=MulX, multiplier=multiplier)
    assert isinstance(layer, HEALPixLayer)


@requires_module("earth2grid")
@pytest.mark.parametrize("multiplier", [2, 3, 4])
def test_HEALPixLayer_forward(device, multiplier, pytestconfig):
    layer = HEALPixLayer(layer=MulX, multiplier=multiplier)

    kernel_size = 3
    dilation = 2
    in_channels = 4
    out_channels = 8

    tensor_size = torch.randint(low=2, high=4, size=(1,)).tolist()
    tensor_size = [24, in_channels, *tensor_size, *tensor_size]
    invar = torch.rand(tensor_size, device=device)
    outvar = layer(invar)

    assert common.compare_output(outvar, invar * multiplier)

    layer = HEALPixLayer(
        layer=torch.nn.Conv2d,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        device=device,
        dilation=dilation,
        enable_healpixpad=True,
        enable_nhwc=True,
    )

    expected_shape = [24, out_channels, tensor_size[-1], tensor_size[-1]]
    expected_shape = torch.Size(expected_shape)

    assert expected_shape == layer(invar).shape


@requires_module("earth2grid")
def test_MaxPool_initialization(device, pytestconfig):
    pooling = 2
    maxpool_block = HEALPixMaxPool(pooling=pooling).to(device)
    assert isinstance(maxpool_block, HEALPixMaxPool)


@requires_module("earth2grid")
def test_MaxPool_forward(device, test_data, pytestconfig):
    pooling = 2
    size = 16
    channels = 4
    maxpool_block = HEALPixMaxPool(pooling=pooling).to(device)

    invar = test_data(
        faces=1, channels=channels, img_size=(size * pooling), device=device
    )
    outvar = test_data(faces=1, channels=channels, img_size=size, device=device)

    assert common.compare_output(outvar, maxpool_block(invar))


@requires_module("earth2grid")
def test_AvgPool_initialization(device, pytestconfig):
    pooling = 2
    avgpool_block = HEALPixAvgPool(pooling=pooling).to(device)
    assert isinstance(avgpool_block, HEALPixAvgPool)


@requires_module("earth2grid")
def test_AvgPool_forward(device, test_data, pytestconfig):
    pooling = 2
    size = 32
    channels = 4
    avgpool_block = HEALPixAvgPool(pooling=pooling).to(device)

    invar = test_data(
        faces=1, channels=channels, img_size=(size * pooling), device=device
    )
    outvar = test_data(faces=1, channels=channels, img_size=size, device=device)

    outvar = outvar * 0.5

    assert common.compare_output(outvar, avgpool_block(invar))


@requires_module("earth2grid")
def test_hpx_patch_tokenizer_forward(device):
    """Test HEALPixPatchTokenizer forward pass."""
    torch.manual_seed(0)

    in_channels = 5
    hidden_size = 8
    level_fine = 2
    level_coarse = 1

    model = HEALPixPatchTokenizer(
        in_channels=in_channels,
        hidden_size=hidden_size,
        level_fine=level_fine,
        level_coarse=level_coarse,
    ).to(device)
    model.eval()

    b, t = 2, 1
    npix = 12 * 4**level_fine
    x = torch.randn(b, in_channels, t, npix).to(device)
    second_of_day = torch.tensor([[43200], [21600]], device=device)
    day_of_year = torch.tensor([[100], [200]], device=device)
    # Manually track device since not psn Module
    model.device = device

    assert common.validate_forward_accuracy(
        model,
        (x, second_of_day, day_of_year),
        file_name="nn/module/data/hpx_tokenizer_output.pth",
        atol=1e-3,  # Data on order of [1 to 0.1]
    )


@requires_module("earth2grid")
def test_calendar_embedding_shape_mismatch():
    """Test CalendarEmbedding raises on shape mismatch."""
    lon = torch.linspace(-180, 180, 10)
    model = CalendarEmbedding(lon=lon, embed_channels=4)

    day_of_year = torch.tensor([[100, 101]])
    second_of_day = torch.tensor([[43200]])

    with pytest.raises(ValueError):
        model(day_of_year=day_of_year, second_of_day=second_of_day)


# HealDA tokenizers
@requires_module("earth2grid")
def test_hpx_patch_detokenizer_forward(device):
    """Test HEALPixPatchDetokenizer forward pass."""
    torch.manual_seed(0)

    hidden_size = 8
    out_channels = 2
    level_coarse = 1
    level_fine = 2
    time_length = 1

    model = HEALPixPatchDetokenizer(
        hidden_size=hidden_size,
        out_channels=out_channels,
        level_coarse=level_coarse,
        level_fine=level_fine,
        time_length=time_length,
    ).to(device)
    model.eval()

    b = 2
    L = time_length * 12 * 4**level_coarse
    x = torch.randn(b, L, hidden_size).to(device)
    c = torch.randn(b, hidden_size).to(device)
    # Manually track device since not psn Module
    model.device = device

    assert common.validate_forward_accuracy(
        model,
        (x, c),
        file_name="nn/module/data/hpx_detokenizer_output.pth",
        atol=1e-3,  # Data on order of [1 to 0.1]
    )
