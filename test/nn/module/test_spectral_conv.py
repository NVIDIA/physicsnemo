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

from physicsnemo.models.fno import FNO
from physicsnemo.nn.module.spectral_layers import (
    SpectralConv1d,
    SpectralConv2d,
    SpectralConv3d,
    SpectralConv4d,
)

IN_CHANNELS = 3
OUT_CHANNELS = 4
CONV_CLASSES = {
    1: SpectralConv1d,
    2: SpectralConv2d,
    3: SpectralConv3d,
    4: SpectralConv4d,
}
CORNERS = {
    1: [()],
    2: [(False,), (True,)],
    3: [(False, False), (True, False), (False, True), (True, True)],
    4: [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
}


def _make_conv(dimension: int, modes=None) -> torch.nn.Module:
    modes = modes if modes is not None else [2] * dimension
    return CONV_CLASSES[dimension](IN_CHANNELS, OUT_CHANNELS, *modes)


def _make_input(dimension: int, size: int, device, dtype=torch.float32):
    return torch.randn(
        2, IN_CHANNELS, *([size] * dimension), device=device, dtype=dtype
    )


AMP_DTYPES = [torch.float16, torch.bfloat16]


def _skip_unsupported_amp(device, amp_dtype):
    # Only float16 is skipped on CPU, where these tests use bfloat16 autocast
    if not str(device).startswith("cuda") and amp_dtype == torch.float16:
        pytest.skip("CPU autocast is tested with bfloat16 only")


def _reference_forward(conv: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Spectral convolution written with slice accumulation into a full-size
    # spectrum, independent of the layer implementation.
    dimension = x.ndim - 2
    modes = [getattr(conv, f"modes{i + 1}") for i in range(dimension)]
    dims = list(range(-dimension, 0))
    x_ft = torch.fft.rfftn(x, dim=dims)
    out_ft = torch.zeros(
        x.shape[0],
        conv.out_channels,
        *x_ft.shape[2:],
        dtype=x_ft.dtype,
        device=x.device,
    )
    letters = "xyzt"[:dimension]
    for k, corner in enumerate(CORNERS[dimension]):
        index = [slice(None), slice(None)]
        for high, m in zip(corner, modes[:-1]):
            index.append(slice(-m, None) if high else slice(None, m))
        index.append(slice(None, modes[-1]))
        index = tuple(index)
        weights = torch.view_as_complex(getattr(conv, f"weights{k + 1}"))
        block = torch.einsum(
            f"bi{letters},io{letters}->bo{letters}", x_ft[index], weights
        )
        out_ft[index] = out_ft[index] + block
    return torch.fft.irfftn(out_ft, s=x.shape[2:], dim=dims)


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_spectral_conv_shape(dimension, device):
    conv = _make_conv(dimension).to(device)
    x = _make_input(dimension, 6, device)
    y = conv(x)
    assert y.shape == (2, OUT_CHANNELS, *([6] * dimension))
    assert y.dtype == torch.float32


@pytest.mark.parametrize(
    "sizes, modes",
    [
        ((16,), (5,)),
        ((15, 12), (4, 5)),
        ((8, 9), (4, 3)),  # 2 * modes == size, no gap between corners
        ((10, 9), (6, 3)),  # corners overlap along the first axis
        ((9, 8, 10), (3, 2, 4)),
        ((8, 8, 8), (5, 3, 3)),  # corners overlap along the first axis
        ((8, 6, 8), (3, 4, 3)),  # corners overlap along the second axis
        ((6, 7, 6, 8), (2, 3, 2, 3)),
        ((6, 6, 6, 6), (2, 2, 4, 2)),  # corners overlap along the third axis
    ],
)
def test_spectral_conv_matches_reference(sizes, modes, device):
    dimension = len(sizes)
    conv = _make_conv(dimension, modes).to(device)
    ref = _make_conv(dimension, modes).to(device)
    ref.load_state_dict(conv.state_dict())
    x = torch.randn(2, IN_CHANNELS, *sizes, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    y = conv(x)
    y_ref = _reference_forward(ref, x_ref)
    torch.testing.assert_close(y, y_ref)

    grad = torch.randn_like(y)
    y.backward(grad)
    y_ref.backward(grad)
    torch.testing.assert_close(x.grad, x_ref.grad)
    for p, p_ref in zip(conv.parameters(), ref.parameters()):
        torch.testing.assert_close(p.grad, p_ref.grad)


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
@pytest.mark.parametrize("size", [6, 8])
@pytest.mark.parametrize("amp_dtype", AMP_DTYPES)
def test_spectral_conv_autocast_reduced_precision_input(
    dimension, size, amp_dtype, device
):
    # Under autocast, the layer before a spectral convolution produces a
    # reduced-precision tensor. The spectral convolution must run on it in
    # float32 and match the float32 result exactly. Size 6 is not a power of
    # two, which cuFFT rejects in half precision; at size 8, a half-precision
    # FFT runs but the mode multiplication does not, and a bfloat16 FFT may
    # not be computed in float32.
    _skip_unsupported_amp(device, amp_dtype)
    device_type = torch.device(device).type
    conv = _make_conv(dimension).to(device)
    x = _make_input(dimension, size, device).to(amp_dtype).requires_grad_(True)
    with torch.autocast(device_type, dtype=amp_dtype):
        y = conv(x)
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, conv(x.detach().float()), rtol=0, atol=0)

    y.square().mean().backward()
    assert torch.isfinite(x.grad.float()).all()
    assert all(torch.isfinite(p.grad).all() for p in conv.parameters())


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
@pytest.mark.parametrize("amp_dtype", AMP_DTYPES)
def test_spectral_conv_autocast_float32_input(dimension, amp_dtype, device):
    # Regression guard: a float32 input under autocast gives the same output
    # as without autocast.
    _skip_unsupported_amp(device, amp_dtype)
    device_type = torch.device(device).type
    conv = _make_conv(dimension).to(device)
    x = _make_input(dimension, 8, device)
    reference = conv(x)
    with torch.autocast(device_type, dtype=amp_dtype):
        y = conv(x)
    torch.testing.assert_close(y, reference, rtol=0, atol=0)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("amp_dtype", AMP_DTYPES)
def test_fno_autocast_training_step(dimension, amp_dtype, device):
    _skip_unsupported_amp(device, amp_dtype)
    device_type = torch.device(device).type
    model = FNO(
        in_channels=2,
        out_channels=1,
        dimension=dimension,
        latent_channels=8,
        num_fno_layers=2,
        num_fno_modes=3,
        padding=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device_type, enabled=amp_dtype == torch.float16)
    x = torch.randn(2, 2, *([10] * dimension), device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    initial = [p.detach().clone() for p in params]
    with torch.autocast(device_type, dtype=amp_dtype):
        loss = model(x).float().square().mean()
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()

    # GradScaler skips the optimizer step when gradients are not finite, so
    # check the unscaled gradients and that the step updated every weight
    scaler.unscale_(optimizer)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    scaler.step(optimizer)
    scaler.update()
    assert all(torch.isfinite(p).all() for p in params)
    assert all(not torch.equal(p, p0) for p, p0 in zip(params, initial))


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("use_autocast", [False, True])
def test_spectral_conv_compile(dimension, use_autocast, device):
    device_type = torch.device(device).type
    # CPU autocast is tested with bfloat16 only
    amp_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
    conv = _make_conv(dimension).to(device)
    compiled = torch.compile(conv, fullgraph=True, backend="aot_eager")
    x = _make_input(dimension, 8, device)
    if use_autocast:
        x = x.to(amp_dtype)
    x = x.requires_grad_(True)
    x_compiled = x.detach().clone().requires_grad_(True)

    with torch.autocast(device_type, dtype=amp_dtype, enabled=use_autocast):
        y = conv(x)
        y_compiled = compiled(x_compiled)
    torch.testing.assert_close(y_compiled, y)
    y.sum().backward()
    y_compiled.sum().backward()
    torch.testing.assert_close(x_compiled.grad, x.grad)


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_spectral_conv_meta_device(dimension):
    conv = _make_conv(dimension).to("meta")
    y = conv(_make_input(dimension, 6, "meta"))
    assert y.shape == (2, OUT_CHANNELS, *([6] * dimension))


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_spectral_conv_float64(dimension, device):
    conv = _make_conv(dimension).to(device)
    x = _make_input(dimension, 6, device)
    reference = conv(x)

    conv = conv.double()
    y = conv(x.double())
    assert y.dtype == torch.float64
    torch.testing.assert_close(y.float(), reference, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spectral_conv_reset_parameters_keeps_device_and_dtype(
    dimension, dtype, device
):
    conv = _make_conv(dimension).to(device=device, dtype=dtype)
    conv.reset_parameters()
    for param in conv.parameters():
        assert param.device == torch.device(device)
        assert param.dtype == dtype
        assert param.requires_grad
