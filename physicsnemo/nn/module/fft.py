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

import math
from typing import List, Optional, Tuple

import torch
import torch.fft
import torch.onnx
from torch import Tensor
from torch.autograd import Function

# Note 1: for DFT operators, the less verbose way of registering an operator is via
# `register_custom_op_symbolic`. However, it does not currently work due to
# torch.fft.rfft* functions returning Complex type which is not yet supported in ONNX.

# Note 2:
# - current ONNX Contrib implementation does not support configurable normalization, so
#   "normalized" must be 0, the normalization is done outside of Contrib ops.
#   See also comments in `_scale_output_backward` function for more details.
# - "onesided" is not configurable either - must be set to 1.
# - Contrib implementation requires DFT dimensions to be the last ones,
#   otherwise axes permutation is required.
# See:
# https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cuda/math/fft_ops.h#L19


def rfft(
    input: Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> Tensor:
    """ONNX compatable method to compute the 1d Fourier transform of real-valued input.

    Parameters
    ----------
    input : Tensor
        Real input tensor
    n : Optional[int], optional
        Signal strength, by default None
    dim : int, optional
        Dimension along which to take the real FFT, by default -1
    norm : Optional[str], optional
        Normalization mode with options "forward", "backward and "ortho". When set to None,
        normalization will default to backward (no normalization), by default None

    Note
    ----
    The function is equivalent to `torch.fft.rfft` when not running in ONNX export mode
    """
    if not torch.onnx.is_in_onnx_export():
        return torch.fft.rfft(input, n=n, dim=dim, norm=norm)

    if not isinstance(dim, int):
        raise TypeError()
    return _rfft_onnx(input, (n,), (dim,), norm)


def rfft2(
    input: Tensor,
    s: Optional[Tuple[int]] = None,
    dim: Tuple[int] = (-2, -1),
    norm: Optional[str] = None,
) -> Tensor:
    """ONNX compatable method to compute the 2d Fourier transform of real-valued input.

    Parameters
    ----------
    input : Tensor
        Real input tensor
    s : Optional[Tuple[int]], optional
        Signal size in the transformed dimensions, by default None
    dim : Tuple[int], optional
        Dimensions along which to take the real 2D FFT, by default (-2, -1)
    norm : Optional[str], optional
        Normalization mode with options "forward", "backward" and "ortho". When set to None,
        normalization will default to backward (normalize by 1/n), by default None

    Note
    ----
    The function is equivalent to `torch.fft.rfft2` when not running in ONNX export mode
    """
    if not torch.onnx.is_in_onnx_export():
        return torch.fft.rfft2(input, s=s, dim=dim, norm=norm)

    if not (isinstance(dim, tuple) and len(dim) == 2):
        raise ValueError()
    return _rfft_onnx(input, s, dim, norm)


def irfft(
    input: Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> Tensor:
    """ONNX compatable method to compute the inverse of `rfft`.

    Parameters
    ----------
    input : Tensor
        Real input tensor
    n : Optional[int], optional
        Signal strength, by default None
    dim : int, optional
        Dimension along which to take the real IFFT, by default -1
    norm : Optional[str], optional
        Normalization mode with options "forward", "backward" and "ortho". When set to None,
        normalization will default to backward (no normalization), by default None

    Note
    ----
    The function is equivalent to `torch.fft.irfft` when not running in ONNX export mode
    """
    if not torch.onnx.is_in_onnx_export():
        return torch.fft.irfft(input, n=n, dim=dim, norm=norm)

    if not isinstance(dim, int):
        raise TypeError()
    return _irfft_onnx(input, (n,), (dim,), norm)


def irfft2(
    input: Tensor,
    s: Optional[Tuple[int]] = None,
    dim: Tuple[int] = (-2, -1),
    norm: Optional[str] = None,
) -> Tensor:
    """ONNX compatable method to compute the inverse of `rfft2`.

    Parameters
    ----------
    input : Tensor
        Real input tensor
    s : Optional[Tuple[int]], optional
        Signal size in the transformed dimensions, by default None
    dim : Tuple[int], optional
        Dimensions along which to take the real 2D IFFT, by default (-2, -1)
    norm : Optional[str], optional
        Normalization mode with options "forward", "backward" and "ortho". When set to None,
        normalization will default to backward (normalize by 1/n), by default None

    Note
    ----
    The function is equivalent to `torch.fft.irfft2` when not running in ONNX export mode
    """
    if not torch.onnx.is_in_onnx_export():
        return torch.fft.irfft2(input, s=s, dim=dim, norm=norm)

    if not (isinstance(dim, tuple) and len(dim) == 2):
        raise ValueError()
    return _irfft_onnx(input, s, dim, norm)


def view_as_complex(input: Tensor) -> Tensor:
    """ONNX compatable method to view input as complex tensor

    Parameters
    ----------
    input : Tensor
        The input Tensor

    Note
    ----
    The function is equivalent to `torch.view_as_complex` when not running in ONNX export mode

    Raises
    ------
    AssertionError
        If input tensor shape is not [...,2] during ONNX runtime where the last dimension
        denotes the real / imaginary tensors
    """
    if not torch.onnx.is_in_onnx_export():
        return torch.view_as_complex(input)

    # Just return the input unchanged - during ONNX export
    # there will be no complex type.
    if input.size(-1) != 2:
        raise ValueError
    return input


def real(input: Tensor) -> Tensor:
    """ONNX compatable method to view input as real tensor

    Parameters
    ----------
    input : Tensor
        The input Tensor

    Note
    ----
    The function is equivalent to `input.real` when not running in ONNX export mode

    Raises
    ------
    AssertionError
        If input tensor shape is not [...,2] during ONNX runtime where the last dimension
        denotes the real / imaginary tensors
    """
    if not torch.onnx.is_in_onnx_export():
        return input.real

    # There is no complex type during ONNX export, so assuming
    # complex numbers are represented as if after `view_as_real`.
    if input.size(-1) != 2:
        raise ValueError()
    return input[..., 0]


def imag(input: Tensor) -> Tensor:
    """ONNX compatable method to view input as imaginary tensor

    Parameters
    ----------
    input : Tensor
        The input Tensor

    Note
    ----
    The function is equivalent to `input.imag` when not running in ONNX export mode

    Raises
    ------
    AssertionError
        If input tensor shape is not [...,2] during ONNX runtime  where the last dimension
        denotes the real / imaginary tensors
    """
    if not torch.onnx.is_in_onnx_export():
        return input.imag

    # There is no complex type during ONNX export, so assuming
    # complex numbers are represented as if after `view_as_real`.
    if input.size(-1) != 2:
        raise ValueError(input.size(-1))
    return input[..., 1]


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
                f"Padding/trimming is not yet supported, "
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
                f"Padding/trimming is not yet supported, "
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
    """Auto-grad function to mimic rfft for ONNX exporting

    Note
    ----
    Should only be called during an ONNX export
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
        """Symbolic representation for onnx graph"""
        return _contrib_rfft(g, input, ndim=1)


class OnnxRfft2(Function):
    """Auto-grad function to mimic rfft2 for ONNX exporting

    Note
    ----
    Should only be called during an ONNX export
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
        """Symbolic representation for onnx graph"""
        return _contrib_rfft(g, input, ndim=2)


class OnnxIrfft(Function):
    """Auto-grad function to mimic irfft for ONNX exporting

    Note
    ----
    Should only be called during an ONNX export
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
        """Symbolic representation for onnx graph"""
        return _contrib_irfft(g, input, ndim=1)


class OnnxIrfft2(Function):
    """Auto-grad function to mimic irfft2 for ONNX exporting.

    Note
    ----
    Should only be called during an ONNX export
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
        """Symbolic representation for onnx graph"""
        return _contrib_irfft(g, input, ndim=2)
