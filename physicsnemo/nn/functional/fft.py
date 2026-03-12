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

import math
from typing import List, Optional, Tuple

import torch
import torch.fft
import torch.onnx
from torch import Tensor
from torch.autograd import Function

from physicsnemo.core.function_spec import FunctionSpec


def _rfft_onnx(
    input: Tensor, s: Optional[Tuple[Optional[int]]], dim: Tuple[int], norm: str
) -> Tensor:
    if s is not None:
        _check_padding_rfft(s, dim, input.size())

    ndim = len(dim)
    if ndim not in [1, 2]:
        raise ValueError(ndim)

    perm = not _is_last_dims(dim, input.ndim)

    if perm:
        perm_in, perm_out = _create_axes_perm(input.ndim, dim)
        # Add a dimension to account for complex output.
        perm_out.append(len(perm_out))
        # Transpose -> RFFT -> Transpose (inverse).
        input = input.permute(perm_in)

    rfft_func = OnnxRfft if ndim == 1 else OnnxRfft2
    output = rfft_func.apply(input)

    output = _scale_output_forward(output, norm, input.size(), ndim)

    if perm:
        output = output.permute(perm_out)

    return output


def _irfft_onnx(
    input: Tensor, s: Optional[Tuple[Optional[int]]], dim: Tuple[int], norm: str
) -> Tensor:
    if s is not None:
        _check_padding_irfft(s, dim, input.size())

    ndim = len(dim)
    if ndim not in [1, 2]:
        raise ValueError(ndim)

    # Whether to permute axes when DFT axis is not the last.
    perm = not _is_last_dims(dim, input.ndim)

    if perm:
        # Do not include last dimension (input is complex).
        perm_in, perm_out = _create_axes_perm(input.ndim - 1, dim)
        # Add a dimension to account for complex input.
        perm_in.append(len(perm_in))
        # Transpose -> IRFFT -> Transpose (inverse).
        input = input.permute(perm_in)

    irfft_func = OnnxIrfft if ndim == 1 else OnnxIrfft2
    output = irfft_func.apply(input)

    output = _scale_output_backward(output, norm, input.size(), ndim)

    if perm:
        output = output.permute(perm_out)

    return output


def _contrib_rfft(g: torch.Graph, input: torch.Value, ndim: int) -> torch.Value:
    if ndim not in [1, 2]:
        raise ValueError(ndim)

    # See https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Rfft
    output = g.op(
        "com.microsoft::Rfft",
        input,
        normalized_i=0,
        onesided_i=1,
        signal_ndim_i=ndim,
    )

    return output


def _contrib_irfft(g: torch.Graph, input: torch.Value, ndim: int) -> torch.Value:
    if ndim not in [1, 2]:
        raise ValueError(ndim)

    # See https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Irfft
    output = g.op(
        "com.microsoft::Irfft",
        input,
        normalized_i=0,
        onesided_i=1,
        signal_ndim_i=ndim,
    )

    return output


def _is_last_dims(dim: Tuple[int], inp_ndim: int) -> bool:
    ndim = len(dim)
    for i, idim in enumerate(dim):
        # This takes care of both positive and negative axis indices.
        if idim % inp_ndim != inp_ndim - ndim + i:
            return False
    return True


def _check_padding_rfft(
    sizes: Tuple[Optional[int]], dim: Tuple[int], inp_sizes: Tuple[int]
) -> None:
    if len(sizes) != len(dim):
        raise ValueError(f"{sizes}, {dim}")
    for i, s in enumerate(sizes):
        if s is None or s < 0:
            continue
        # Current Contrib RFFT does not support pad/trim yet.
        if s != inp_sizes[dim[i]]:
            raise RuntimeError(
                "Padding/trimming is not yet supported, "
                f"got sizes {sizes}, DFT dims {dim}, "
                f"input dims {inp_sizes}."
            )


def _check_padding_irfft(
    sizes: Tuple[Optional[int]], dim: Tuple[int], inp_sizes: Tuple[int]
) -> None:
    if len(sizes) != len(dim):
        raise ValueError(f"{sizes}, {dim}")
    # All but last dims must be equal to input dims.
    for i, s in enumerate(sizes[:-1]):
        if s is None or s < 0:
            continue
        # Current Contrib RFFT does not support pad/trim yet.
        if s != inp_sizes[dim[i]]:
            raise RuntimeError(
                "Padding/trimming is not yet supported, "
                f"got sizes {sizes}, DFT dims {dim}, "
                f"input dims {inp_sizes}."
            )
    # Check last dim.
    s = sizes[-1]
    if s is not None and s > 0:
        expected_size = 2 * (inp_sizes[dim[-1]] - 1)
        if s != expected_size:
            raise RuntimeError(
                f"Padding/trimming is not yet supported, got sizes {sizes}"
                f", DFT dims {dim}, input dims {inp_sizes}"
                f", expected last size {expected_size}."
            )


def _create_axes_perm(ndim: int, dims: Tuple[int]) -> Tuple[List[int], List[int]]:
    """Creates permuted axes indices for RFFT/IRFFT operators."""
    perm_in = list(range(ndim))
    perm_out = list(perm_in)
    # Move indices to the right to make 'dims' as innermost dimensions.
    for i in range(-1, -(len(dims) + 1), -1):
        perm_in[dims[i]], perm_in[i] = perm_in[i], perm_in[dims[i]]
    # Move indices to the left to restore original shape.
    for i in range(-len(dims), 0):
        perm_out[dims[i]], perm_out[i] = perm_out[i], perm_out[dims[i]]

    return perm_in, perm_out


def _scale_output_forward(
    output: Tensor, norm: str, sizes: torch.Size, ndim: int
) -> Tensor:
    """Scales the RFFT output according to norm parameter."""

    norm = "backward" if norm is None else norm
    if norm not in ["forward", "backward", "ortho"]:
        raise ValueError(norm)

    # No normalization for "backward" in RFFT ops.
    if norm in ["forward", "ortho"]:
        # Assuming DFT dimensions are the last. This is required by the current Contrib ops,
        # so the axes permutation of the input is done accordingly.
        dft_size = math.prod(sizes[-ndim:]).float()
        denom = torch.sqrt(dft_size) if norm == "ortho" else dft_size
        output = output / denom

    return output


def _scale_output_backward(
    output: Tensor, norm: str, sizes: torch.Size, ndim: int
) -> Tensor:
    """Scales the IRFFT output according to norm parameter."""

    norm = "backward" if norm is None else norm
    if norm not in ["forward", "backward", "ortho"]:
        raise ValueError(norm)

    # Things get interesting here: Contrib IRFFT op uses cuFFT cufftXtExec
    # followed by a custom CUDA kernel (`_Normalize`) which always performs
    # normalization (division by N) which means "norm" is essentially
    # always "backward" here. So we need to cancel this normalization
    # when norm is "forward" or "ortho".
    if norm in ["forward", "ortho"]:
        # Last dimension is complex numbers representation.
        # Second-to-last dim corresponds to last dim in RFFT transform.
        # This is required by the current Contrib ops,
        # so the axes permutation of the input is done previously.
        if not len(sizes) >= ndim + 1:
            raise ValueError
        dft_size = math.prod(sizes[-(ndim + 1) : -2])
        dft_size *= 2 * (sizes[-2] - 1)
        dft_size = dft_size.float()
        # Since cuFFT scales by 1/dft_size, replace this scale with appropriate one.
        scale = dft_size if norm == "forward" else torch.sqrt(dft_size)
        output = scale * output

    return output


class OnnxRfft(Function):
    """Auto-grad function to mimic rfft for ONNX exporting.

    Note
    ----
    Should only be called during an ONNX export.
    """

    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            raise ValueError("Must be called only during ONNX export.")

        # We need to mimic the behavior of Contrib RFFT which assumes
        # DFT of last dim and no normalization.
        y = torch.fft.rfft(input, dim=-1, norm="backward")
        return torch.view_as_real(y)

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        """Symbolic representation for onnx graph."""
        return _contrib_rfft(g, input, ndim=1)


class OnnxRfft2(Function):
    """Auto-grad function to mimic rfft2 for ONNX exporting.

    Note
    ----
    Should only be called during an ONNX export.
    """

    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            raise AssertionError("Must be called only during ONNX export.")

        # We need to mimic the behavior of Contrib RFFT which assumes
        # DFT of last dims and no normalization.
        y = torch.fft.rfft2(input, dim=(-2, -1), norm="backward")
        return torch.view_as_real(y)

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        """Symbolic representation for onnx graph."""
        return _contrib_rfft(g, input, ndim=2)


class OnnxIrfft(Function):
    """Auto-grad function to mimic irfft for ONNX exporting.

    Note
    ----
    Should only be called during an ONNX export.
    """

    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            raise ValueError("Must be called only during ONNX export.")

        # We need to mimic the behavior of Contrib IRFFT which assumes
        # DFT of last dim and 1/n normalization.
        return torch.fft.irfft(torch.view_as_complex(input), dim=-1, norm="backward")

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        """Symbolic representation for onnx graph."""
        return _contrib_irfft(g, input, ndim=1)


class OnnxIrfft2(Function):
    """Auto-grad function to mimic irfft2 for ONNX exporting.

    Note
    ----
    Should only be called during an ONNX export.
    """

    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            raise AssertionError("Must be called only during ONNX export.")

        # We need to mimic the behavior of Contrib IRFFT which assumes
        # DFT of last dims and 1/n normalization.
        return torch.fft.irfft2(
            torch.view_as_complex(input), dim=(-2, -1), norm="backward"
        )

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        """Symbolic representation for onnx graph."""
        return _contrib_irfft(g, input, ndim=2)


class ViewAsComplex(FunctionSpec):
    """ONNX-compatible view of real-valued tensors as complex tensors.

    Parameters
    ----------
    input : torch.Tensor
        Real tensor with a final dimension of size 2 storing real/imag parts.
    implementation : {"torch"} or None
        Implementation to use. When ``None``, dispatch selects the available
        implementation.
    """

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(input: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            return torch.view_as_complex(input)

        # Just return the input unchanged - during ONNX export
        # there will be no complex type.
        if input.size(-1) != 2:
            raise ValueError
        return input

    @classmethod
    def make_inputs(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        cases = [
            ("small", 4096),
            ("medium", 16384),
            ("large", 65536),
        ]
        for label, size in cases:
            signal = torch.randn(4, size, 2, device=device)
            yield (f"{label}-batch4-length{size}-realimag2", (signal,), {})


class Real(FunctionSpec):
    """ONNX-compatible view of the real component from complex tensors.

    Parameters
    ----------
    input : torch.Tensor
        Complex tensor.
    implementation : {"torch"} or None
        Implementation to use. When ``None``, dispatch selects the available
        implementation.
    """

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(input: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            return input.real

        # There is no complex type during ONNX export, so assuming
        # complex numbers are represented as if after `view_as_real`.
        if input.size(-1) != 2:
            raise ValueError
        return input[..., 0]

    @classmethod
    def make_inputs(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        cases = [
            ("small", 4096),
            ("medium", 16384),
            ("large", 65536),
        ]
        for label, size in cases:
            signal = torch.randn(4, size, 2, device=device)
            complex_signal = torch.view_as_complex(signal)
            yield (f"{label}-batch4-length{size}-complex", (complex_signal,), {})


class Imag(FunctionSpec):
    """ONNX-compatible view of the imaginary component from complex tensors.

    Parameters
    ----------
    input : torch.Tensor
        Complex tensor.
    implementation : {"torch"} or None
        Implementation to use. When ``None``, dispatch selects the available
        implementation.
    """

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(input: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            return input.imag

        # There is no complex type during ONNX export, so assuming
        # complex numbers are represented as if after `view_as_real`.
        if input.size(-1) != 2:
            raise ValueError(input.size(-1))
        return input[..., 1]

    @classmethod
    def make_inputs(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        cases = [
            ("small", 4096),
            ("medium", 16384),
            ("large", 65536),
        ]
        for label, size in cases:
            signal = torch.randn(4, size, 2, device=device)
            complex_signal = torch.view_as_complex(signal)
            yield (f"{label}-batch4-length{size}-complex", (complex_signal,), {})


class RFFT(FunctionSpec):
    """ONNX-compatible 1D real FFT.

    Parameters
    ----------
    input : torch.Tensor
        Real input tensor.
    n : int, optional
        Signal length along the FFT dimension.
    dim : int, optional
        Dimension along which to take the FFT.
    norm : str, optional
        Normalization mode (``"forward"``, ``"backward"``, or ``"ortho"``).
    implementation : {"torch"} or None
        Implementation to use. When ``None``, dispatch selects the available
        implementation.
    """

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(
        input: Tensor,
        n: int | None = None,
        dim: int = -1,
        norm: str | None = None,
    ) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            return torch.fft.rfft(input, n=n, dim=dim, norm=norm)

        if not isinstance(dim, int):
            raise TypeError()
        return _rfft_onnx(input, (n,), (dim,), norm)

    @classmethod
    def make_inputs(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        cases = [
            ("small", 4096),
            ("medium", 16384),
            ("large", 65536),
        ]
        for label, size in cases:
            signal = torch.randn(4, size, device=device)
            yield (f"{label}-batch4-signal-length{size}", (signal,), {"n": size})


class RFFT2(FunctionSpec):
    """ONNX-compatible 2D real FFT.

    Parameters
    ----------
    input : torch.Tensor
        Real input tensor.
    s : tuple[int, int], optional
        Signal size in the transformed dimensions.
    dim : tuple[int, int], optional
        Dimensions along which to take the FFT.
    norm : str, optional
        Normalization mode (``"forward"``, ``"backward"``, or ``"ortho"``).
    implementation : {"torch"} or None
        Implementation to use. When ``None``, dispatch selects the available
        implementation.
    """

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(
        input: Tensor,
        s: tuple[int, int] | None = None,
        dim: tuple[int, int] = (-2, -1),
        norm: str | None = None,
    ) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            return torch.fft.rfft2(input, s=s, dim=dim, norm=norm)

        if not (isinstance(dim, tuple) and len(dim) == 2):
            raise ValueError()
        return _rfft_onnx(input, s, dim, norm)

    @classmethod
    def make_inputs(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        cases = [
            ("small", 128, 128),
            ("medium", 256, 256),
            ("large", 512, 512),
        ]
        for label, height, width in cases:
            signal = torch.randn(4, height, width, device=device)
            yield (
                f"{label}-batch4-height{height}-width{width}",
                (signal,),
                {"s": (height, width)},
            )


class IRFFT(FunctionSpec):
    """ONNX-compatible inverse 1D real FFT.

    Parameters
    ----------
    input : torch.Tensor
        Complex input tensor in the frequency domain.
    n : int, optional
        Signal length along the inverse FFT dimension.
    dim : int, optional
        Dimension along which to take the inverse FFT.
    norm : str, optional
        Normalization mode (``"forward"``, ``"backward"``, or ``"ortho"``).
    implementation : {"torch"} or None
        Implementation to use. When ``None``, dispatch selects the available
        implementation.
    """

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(
        input: Tensor,
        n: int | None = None,
        dim: int = -1,
        norm: str | None = None,
    ) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            return torch.fft.irfft(input, n=n, dim=dim, norm=norm)

        if not isinstance(dim, int):
            raise TypeError()
        return _irfft_onnx(input, (n,), (dim,), norm)

    @classmethod
    def make_inputs(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        cases = [
            ("small", 4096),
            ("medium", 16384),
            ("large", 65536),
        ]
        for label, size in cases:
            signal = torch.randn(4, size, device=device)
            spectrum = torch.fft.rfft(signal)
            yield (
                f"{label}-batch4-spectrum-length{size}",
                (spectrum,),
                {"n": size},
            )


class IRFFT2(FunctionSpec):
    """ONNX-compatible inverse 2D real FFT.

    Parameters
    ----------
    input : torch.Tensor
        Complex input tensor in the frequency domain.
    s : tuple[int, int], optional
        Signal size in the transformed dimensions.
    dim : tuple[int, int], optional
        Dimensions along which to take the inverse FFT.
    norm : str, optional
        Normalization mode (``"forward"``, ``"backward"``, or ``"ortho"``).
    implementation : {"torch"} or None
        Implementation to use. When ``None``, dispatch selects the available
        implementation.
    """

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(
        input: Tensor,
        s: tuple[int, int] | None = None,
        dim: tuple[int, int] = (-2, -1),
        norm: str | None = None,
    ) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            return torch.fft.irfft2(input, s=s, dim=dim, norm=norm)

        if not (isinstance(dim, tuple) and len(dim) == 2):
            raise ValueError()
        return _irfft_onnx(input, s, dim, norm)

    @classmethod
    def make_inputs(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        cases = [
            ("small", 128, 128),
            ("medium", 256, 256),
            ("large", 512, 512),
        ]
        for label, height, width in cases:
            signal = torch.randn(4, height, width, device=device)
            spectrum = torch.fft.rfft2(signal)
            yield (
                f"{label}-batch4-spectrum-height{height}-width{width}",
                (spectrum,),
                {"s": (height, width)},
            )


rfft = RFFT.make_function("rfft")
rfft2 = RFFT2.make_function("rfft2")
irfft = IRFFT.make_function("irfft")
irfft2 = IRFFT2.make_function("irfft2")
view_as_complex = ViewAsComplex.make_function("view_as_complex")
real = Real.make_function("real")
imag = Imag.make_function("imag")


__all__ = [
    "RFFT",
    "RFFT2",
    "IRFFT",
    "IRFFT2",
    "ViewAsComplex",
    "Real",
    "Imag",
    "rfft",
    "rfft2",
    "irfft",
    "irfft2",
    "view_as_complex",
    "real",
    "imag",
]
