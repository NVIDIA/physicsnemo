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

from __future__ import annotations

import pytest
import torch

from physicsnemo.nn.functional import imag, irfft, irfft2, real, rfft, rfft2
from physicsnemo.nn.functional import view_as_complex as functional_view_as_complex
from physicsnemo.nn.functional.fourier_spectral import (
    IRFFT,
    IRFFT2,
    RFFT,
    RFFT2,
    Imag,
    Real,
    ViewAsComplex,
)

_FFT_FUNCTION_SPECS = (ViewAsComplex, Real, Imag, RFFT, RFFT2, IRFFT, IRFFT2)


# Validate the view_as_complex functional wrapper against torch behavior.
def test_view_as_complex_torch(device: str):
    signal = torch.randn(4, 128, 2, device=device, dtype=torch.float32)
    output = functional_view_as_complex(signal, implementation="torch")
    expected = torch.view_as_complex(signal)
    torch.testing.assert_close(output, expected)


# Validate real and imag wrappers against torch complex views.
def test_real_imag_torch(device: str):
    signal = torch.randn(8, 64, 2, device=device, dtype=torch.float32)
    complex_signal = torch.view_as_complex(signal)

    real_output = real(complex_signal, implementation="torch")
    imag_output = imag(complex_signal, implementation="torch")

    torch.testing.assert_close(real_output, complex_signal.real)
    torch.testing.assert_close(imag_output, complex_signal.imag)


# Validate 1D/2D FFT wrappers against torch.fft reference ops.
def test_rfft_torch(device: str):
    signal = torch.randn(4, 256, device=device, dtype=torch.float32)
    output = rfft(signal, n=256, dim=-1, norm=None, implementation="torch")
    expected = torch.fft.rfft(signal, n=256, dim=-1, norm=None)
    torch.testing.assert_close(output, expected)


def test_rfft2_torch(device: str):
    signal = torch.randn(4, 32, 64, device=device, dtype=torch.float32)
    output = rfft2(
        signal,
        s=(32, 64),
        dim=(-2, -1),
        norm=None,
        implementation="torch",
    )
    expected = torch.fft.rfft2(signal, s=(32, 64), dim=(-2, -1), norm=None)
    torch.testing.assert_close(output, expected)


# Validate inverse 1D/2D FFT wrappers against torch.fft reference ops.
def test_irfft_torch(device: str):
    signal = torch.randn(4, 256, device=device, dtype=torch.float32)
    spectrum = torch.fft.rfft(signal)
    output = irfft(spectrum, n=256, dim=-1, norm=None, implementation="torch")
    expected = torch.fft.irfft(spectrum, n=256, dim=-1, norm=None)
    torch.testing.assert_close(output, expected)


def test_irfft2_torch(device: str):
    signal = torch.randn(4, 32, 64, device=device, dtype=torch.float32)
    spectrum = torch.fft.rfft2(signal)
    output = irfft2(
        spectrum,
        s=(32, 64),
        dim=(-2, -1),
        norm=None,
        implementation="torch",
    )
    expected = torch.fft.irfft2(spectrum, s=(32, 64), dim=(-2, -1), norm=None)
    torch.testing.assert_close(output, expected)


# Validate benchmark input generation contract for all FFT functionals.
def test_fft_make_inputs_forward(device: str):
    for spec in _FFT_FUNCTION_SPECS:
        label, args, kwargs = next(iter(spec.make_inputs_forward(device=device)))
        assert isinstance(label, str)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        output = spec.dispatch(*args, implementation="torch", **kwargs)
        assert isinstance(output, torch.Tensor)


# Validate benchmark backward-input contract for all FFT functionals.
def test_fft_make_inputs_backward(device: str):
    for spec in _FFT_FUNCTION_SPECS:
        label, args, kwargs = next(iter(spec.make_inputs_backward(device=device)))
        assert isinstance(label, str)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)

        differentiable_args = [
            arg for arg in args if torch.is_tensor(arg) and arg.requires_grad
        ]
        assert differentiable_args, (
            f"{spec.__name__} backward inputs must include requires_grad tensors"
        )
        grad_input = differentiable_args[0]

        output = spec.dispatch(*args, implementation="torch", **kwargs)
        if torch.is_complex(output):
            output.real.sum().backward()
        else:
            output.sum().backward()

        assert grad_input.grad is not None


# Validate API-level error handling for invalid view_as_complex input layout.
def test_fft_error_handling(device: str):
    with pytest.raises(RuntimeError):
        functional_view_as_complex(
            torch.randn(4, 128, device=device, dtype=torch.float32),
            implementation="torch",
        )
