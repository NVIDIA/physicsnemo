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

import logging

import pytest
import torch
import torch.fft
import torch.nn as nn
import torch.onnx
import torch.onnx.utils

import physicsnemo.models.layers.fft as fft

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from typing import Tuple

from ort_utils import check_ort_version

from physicsnemo.deploy.onnx import export_to_onnx_stream, run_onnx_inference

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


@pytest.fixture
def test_data() -> Tensor:
    # Simple input with 3 signals which contain non-zero DC, real and imaginary parts.
    # fmt: off
    x = torch.tensor([
        [1.0,  0.0, -1.0, 0.0],
        [2.0,  0.0,  2.0, 0.0],
        [0.0, -1.0,  0.0, 1.0]
    ])
    # fmt: on
    # Return as NHW.
    return x.unsqueeze(0)


@pytest.fixture(params=[1, 2])
def test_data_2(request, test_data: Tensor) -> Tensor:
    num_c = request.param
    # To NHWC with identical channels.
    return test_data.tile(1, num_c, 1, 1).permute(0, 2, 3, 1)


@pytest.fixture(params=["forward", "backward", "ortho"])
def norm(request) -> str:
    return request.param


@pytest.mark.parametrize("dft_dim", [-1, 1])
def test_rfft_onnx_op(
    test_data: Tensor, norm: str, dft_dim: int, rtol: float = 1e-5, atol: float = 1e-5
):
    """Test RFFT onnx forward operation is consistent with torch rfft"""
    # Swap last dim with requested, if needed.
    x = test_data.transpose(-1, dft_dim)

    y_expected = torch.fft.rfft(x, dim=dft_dim, norm=norm)
    y_actual = fft.rfft(x, dim=dft_dim, norm=norm)

    assert torch.allclose(y_actual, y_expected, rtol, atol)


@check_ort_version()
@pytest.mark.parametrize("dft_dim", [-1, 1])
def test_rfft_ort_op(
    test_data: Tensor, norm: str, dft_dim: int, rtol: float = 1e-5, atol: float = 1e-5
):
    """Test RFFT onnx runtime operation is consistent with torch rfft"""
    x = test_data.transpose(-1, dft_dim)

    class CustomRfft(nn.Module):
        def forward(self, x):
            return fft.rfft(x, dim=dft_dim, norm=norm)

    model = CustomRfft()
    output = model(x)

    onnx_model = export_to_onnx_stream(model, x)
    output_ort = run_onnx_inference(onnx_model, (x,))
    assert len(output_ort) == 1
    output_onnx = torch.Tensor(output_ort[0])
    output_onnx = torch.view_as_complex(output_onnx)

    assert torch.allclose(output, output_onnx, rtol, atol)


@pytest.mark.parametrize("dft_dim", [(-2, -1), (1, 2)])
def test_rfft2_onnx_op(
    test_data_2: Tensor,
    norm: str,
    dft_dim: Tuple[int],
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    """Test 2D RFFT onnx forward operation is consistent with torch rfft2"""
    x = test_data_2
    # Swap dims from right to left.
    x = x.transpose(2, dft_dim[-1]).transpose(1, dft_dim[-2])

    y_expected = torch.fft.rfft2(x, dim=dft_dim, norm=norm)
    y_actual = fft.rfft2(x, dim=dft_dim, norm=norm)

    assert torch.allclose(y_actual, y_expected, rtol, atol)


@check_ort_version()
@pytest.mark.parametrize("dft_dim", [(-2, -1), (1, 2)])
def test_rfft2_ort_op(
    test_data_2: Tensor,
    norm: str,
    dft_dim: Tuple[int],
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    """Test 2D RFFT onnx runtime operation is consistent with torch rfft2"""
    x = test_data_2
    x = x.transpose(2, dft_dim[-1]).transpose(1, dft_dim[-2])

    class CustomRfft2(nn.Module):
        def forward(self, x):
            return fft.rfft2(x, dim=dft_dim, norm=norm)

    model = CustomRfft2()
    output = model(x)

    onnx_model = export_to_onnx_stream(model, x)
    output_ort = run_onnx_inference(onnx_model, (x,))
    assert len(output_ort) == 1
    output_onnx = torch.Tensor(output_ort[0])
    output_onnx = torch.view_as_complex(output_onnx)

    assert torch.allclose(output, output_onnx, rtol, atol)


@pytest.mark.parametrize("dft_dim", [-1, 1])
def test_irfft_onnx_op(
    test_data: Tensor, norm: str, dft_dim: int, rtol: float = 1e-5, atol: float = 1e-5
):
    """Test IRFFT onnx forward operation is consistent with torch irfft"""
    x = test_data.transpose(-1, dft_dim)

    y = fft.rfft(x, dim=dft_dim, norm=norm)
    x_actual = fft.irfft(y, dim=dft_dim, norm=norm)

    assert torch.allclose(x_actual, x, rtol, atol)


@check_ort_version()
@pytest.mark.parametrize("dft_dim", [-1, 1])
def test_irfft_ort_op(
    test_data: Tensor, norm: str, dft_dim: int, rtol: float = 1e-5, atol: float = 1e-5
):
    """Test IRFFT onnx runtime operation is consistent with torch irfft"""
    x = test_data.transpose(-1, dft_dim)
    x = fft.rfft(x, dim=dft_dim, norm=norm)

    class CustomIrfft(nn.Module):
        def forward(self, y):
            return fft.irfft(y, dim=dft_dim, norm=norm)

    model = CustomIrfft()
    output = model(x)

    x0 = torch.view_as_real(x)
    onnx_model = export_to_onnx_stream(model, x0)
    output_ort = run_onnx_inference(onnx_model, (x0,))
    assert len(output_ort) == 1
    output_onnx = torch.Tensor(output_ort[0])

    assert torch.allclose(output, output_onnx, rtol, atol)


@pytest.mark.parametrize("dft_dim", [(-2, -1), (1, 2)])
def test_irfft2_onnx_op(
    test_data_2: Tensor,
    norm: str,
    dft_dim: Tuple[int],
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    """Test 2D IRFFT onnx forward operation is consistent with torch irfft2"""
    x = test_data_2
    x = x.transpose(2, dft_dim[-1]).transpose(1, dft_dim[-2])

    y = fft.rfft2(x, dim=dft_dim, norm=norm)
    x_actual = fft.irfft2(y, dim=dft_dim, norm=norm)

    assert torch.allclose(x_actual, x, rtol, atol)


@check_ort_version()
@pytest.mark.parametrize("dft_dim", [(-2, -1), (1, 2)])
def test_irfft2_ort_op(
    test_data_2: Tensor,
    norm: str,
    dft_dim: Tuple[int],
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    """Test 2D IRFFT onnx runtime operation is consistent with torch irfft2"""
    x = test_data_2
    x = x.transpose(2, dft_dim[-1]).transpose(1, dft_dim[-2])
    x = fft.rfft2(x, dim=dft_dim, norm=norm)

    class CustomIrfft(nn.Module):
        def forward(self, y):
            return fft.irfft2(y, dim=dft_dim, norm=norm)

    model = CustomIrfft()
    output = model(x)

    x0 = torch.view_as_real(x)
    onnx_model = export_to_onnx_stream(model, x0)
    output_ort = run_onnx_inference(onnx_model, (x0,))
    assert len(output_ort) == 1
    output_onnx = torch.Tensor(output_ort[0])

    assert torch.allclose(output, output_onnx, rtol, atol)


@check_ort_version()
def test_roundtrip_ort(test_data_2: Tensor, rtol: float = 1e-5, atol: float = 1e-5):
    """Tests model with rfft2 and irfft2 combined in ORT session"""
    x = test_data_2

    class Roundtrip(nn.Module):
        def forward(self, x):
            y = fft.rfft2(x, dim=(1, 2), norm="backward")
            return fft.irfft2(y, dim=(1, 2), norm="backward")

    model = Roundtrip()
    output = model(x)

    onnx_model = export_to_onnx_stream(model, x)
    output_ort = run_onnx_inference(onnx_model, (x,))
    assert len(output_ort) == 1
    output_onnx = torch.Tensor(output_ort[0])

    assert torch.allclose(output, output_onnx, rtol, atol)


@check_ort_version()
def test_complex_ort_op(test_data: Tensor, rtol: float = 1e-5, atol: float = 1e-5):
    """Test ONNX compatible complex operations"""
    x = test_data

    class ComplexOps(nn.Module):
        def forward(self, x):
            res = fft.view_as_complex(x)
            return fft.real(res), fft.imag(res)

    # Stack along last dimension to get the tensor that mimics complex numbers.
    x_cpl = torch.stack((x, 2 * x), dim=-1)

    # Convert to PyTorch Complex dtype to get expected values.
    output = torch.view_as_complex(x_cpl)

    # Export to ONNX and run inference.
    model = ComplexOps()
    onnx_model = export_to_onnx_stream(model, x_cpl)
    ort_outputs = run_onnx_inference(onnx_model, (x_cpl,))
    assert len(ort_outputs) == 2

    output_onnx_real = torch.Tensor(ort_outputs[0])
    output_onnx_imag = torch.Tensor(ort_outputs[1])

    assert torch.allclose(output.real, output_onnx_real, rtol, atol)
    assert torch.allclose(output.imag, output_onnx_imag, rtol, atol)


def test_onnx_rfft_checks(test_data: Tensor):
    """ONNX rfft error checks work, padding is not supported for ONNX RFFT"""
    # Should test multiple dims, but this is good enough
    itest_data = torch.stack([test_data, test_data], dim=-1)
    try:
        fft._rfft_onnx(test_data, [-1, -1], dim=(-2, -1), norm="backward")
        raise AssertionError("ONNX RFFT should error outside ORT")
    except AssertionError:
        pass
    try:
        fft._irfft_onnx(itest_data, [-1, -1], dim=(-2, -1), norm="backward")
        raise AssertionError("ONNX IRFFT should error outside ORT")
    except AssertionError:
        pass
    try:
        fft._rfft_onnx(test_data, [-1, -1, -1], dim=(-2, -1), norm="backward")
        raise AssertionError(
            "ONNX RFFT should error if user gives size larger than RFFT dim"
        )
    except ValueError:
        pass

    try:
        fft._rfft_onnx(test_data, [16, 16], dim=(-2, -1), norm="backward")
        raise AssertionError("ONNX RFFT should RuntimeError if user attempts padding")
    except RuntimeError:
        pass

    try:
        fft._irfft_onnx(itest_data, [16, None], dim=(-2, -1), norm="backward")
        raise AssertionError("ONNX IRFFT should RuntimeError if user attempts padding")
    except RuntimeError:
        pass
