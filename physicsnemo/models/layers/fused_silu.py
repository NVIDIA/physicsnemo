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

import functools
import logging
from typing import Tuple

import torch
from torch.autograd import Function

logger = logging.getLogger(__name__)

try:
    import nvfuser
    from nvfuser import DataType, FusionDefinition
except ImportError:
    logger.error(
        "An error occured. Either nvfuser is not installed or the version is "
        "incompatible. Please retry after installing correct version of nvfuser. "
        "The new version of nvfuser should be available in PyTorch container version "
        ">= 23.10. "
        "https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html. "
        "If using a source install method, please refer nvFuser repo for installation "
        "guidelines https://github.com/NVIDIA/Fuser.",
    )
    raise

_torch_dtype_to_nvfuser = {
    torch.double: DataType.Double,
    torch.float: DataType.Float,
    torch.half: DataType.Half,
    torch.int: DataType.Int,
    torch.int32: DataType.Int32,
    torch.bool: DataType.Bool,
    torch.bfloat16: DataType.BFloat16,
    torch.cfloat: DataType.ComplexFloat,
    torch.cdouble: DataType.ComplexDouble,
}


@functools.lru_cache(maxsize=None)
def silu_backward_for(
    fd: FusionDefinition,
    dtype: torch.dtype,
    dim: int,
    size: torch.Size,
    stride: Tuple[int, ...],
):  # pragma: no cover
    """
    nvfuser frontend implmentation of SiLU backward as a fused kernel and with
    activations recomputation

    Parameters
    ----------
    fd : FusionDefition
        nvFuser's FusionDefition class
    dtype : torch.dtype
        Data type to use for the implementation
    dim : int
        Dimension of the input tensor
    size : torch.Size
        Size of the input tensor
    stride : Tuple[int, ...]
        Stride of the input tensor
    """
    try:
        dtype = _torch_dtype_to_nvfuser[dtype]
    except KeyError:
        raise TypeError("Unsupported dtype")

    x = fd.define_tensor(
        shape=[-1] * dim,
        contiguity=nvfuser.compute_contiguity(size, stride),
        dtype=dtype,
    )
    one = fd.define_constant(1.0)

    # y = sigmoid(x)
    y = fd.ops.sigmoid(x)
    # z = sigmoid(x)
    grad_input = fd.ops.mul(y, fd.ops.add(one, fd.ops.mul(x, fd.ops.sub(one, y))))

    grad_input = fd.ops.cast(grad_input, dtype)

    fd.add_output(grad_input)


@functools.lru_cache(maxsize=None)
def silu_double_backward_for(
    fd: FusionDefinition,
    dtype: torch.dtype,
    dim: int,
    size: torch.Size,
    stride: Tuple[int, ...],
):  # pragma: no cover
    """
    nvfuser frontend implmentation of SiLU double backward as a fused kernel and with
    activations recomputation

    Parameters
    ----------
    fd : FusionDefition
        nvFuser's FusionDefition class
    dtype : torch.dtype
        Data type to use for the implementation
    dim : int
        Dimension of the input tensor
    size : torch.Size
        Size of the input tensor
    stride : Tuple[int, ...]
        Stride of the input tensor
    """
    try:
        dtype = _torch_dtype_to_nvfuser[dtype]
    except KeyError:
        raise TypeError("Unsupported dtype")

    x = fd.define_tensor(
        shape=[-1] * dim,
        contiguity=nvfuser.compute_contiguity(size, stride),
        dtype=dtype,
    )
    one = fd.define_constant(1.0)

    # y = sigmoid(x)
    y = fd.ops.sigmoid(x)
    # dy = y * (1 - y)
    dy = fd.ops.mul(y, fd.ops.sub(one, y))
    # z = 1 + x * (1 - y)
    z = fd.ops.add(one, fd.ops.mul(x, fd.ops.sub(one, y)))
    # term1 = dy * z
    term1 = fd.ops.mul(dy, z)

    # term2 = y * ((1 - y) - x * dy)
    term2 = fd.ops.mul(y, fd.ops.sub(fd.ops.sub(one, y), fd.ops.mul(x, dy)))

    grad_input = fd.ops.add(term1, term2)

    grad_input = fd.ops.cast(grad_input, dtype)

    fd.add_output(grad_input)


@functools.lru_cache(maxsize=None)
def silu_triple_backward_for(
    fd: FusionDefinition,
    dtype: torch.dtype,
    dim: int,
    size: torch.Size,
    stride: Tuple[int, ...],
):  # pragma: no cover
    """
    nvfuser frontend implmentation of SiLU triple backward as a fused kernel and with
    activations recomputation

    Parameters
    ----------
    fd : FusionDefition
        nvFuser's FusionDefition class
    dtype : torch.dtype
        Data type to use for the implementation
    dim : int
        Dimension of the input tensor
    size : torch.Size
        Size of the input tensor
    stride : Tuple[int, ...]
        Stride of the input tensor
    """
    try:
        dtype = _torch_dtype_to_nvfuser[dtype]
    except KeyError:
        raise TypeError("Unsupported dtype")

    x = fd.define_tensor(
        shape=[-1] * dim,
        contiguity=nvfuser.compute_contiguity(size, stride),
        dtype=dtype,
    )
    one = fd.define_constant(1.0)
    two = fd.define_constant(2.0)

    # y = sigmoid(x)
    y = fd.ops.sigmoid(x)
    # dy = y * (1 - y)
    dy = fd.ops.mul(y, fd.ops.sub(one, y))
    # ddy = (1 - 2y) * dy
    ddy = fd.ops.mul(fd.ops.sub(one, fd.ops.mul(two, y)), dy)
    # term1 = ddy * (2 + x - 2xy)
    term1 = fd.ops.mul(
        ddy, fd.ops.sub(fd.ops.add(two, x), fd.ops.mul(two, fd.ops.mul(x, y)))
    )

    # term2 = dy * (1 - 2 (y + x * dy))
    term2 = fd.ops.mul(
        dy, fd.ops.sub(one, fd.ops.mul(two, fd.ops.add(y, fd.ops.mul(x, dy))))
    )

    grad_input = fd.ops.add(term1, term2)

    grad_input = fd.ops.cast(grad_input, dtype)

    fd.add_output(grad_input)


class FusedSiLU(Function):
    """
    Fused SiLU activation implementation using nvfuser for a custom fused backward
    with activation recomputation
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward method for SiLU activation

        Parameters
        ----------
        ctx :
            torch context
        x :
            input tensor

        Returns
        -------
        output activation
        """
        ctx.save_for_backward(x)
        return torch.nn.functional.silu(x)

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        """
        Backward method for SiLU activation

        Parameters
        ----------
        ctx :
            torch context
        grad_output :
            output gradients

        Returns
        -------
        input gradients
        """
        (x,) = ctx.saved_tensors
        return FusedSiLU_deriv_1.apply(x) * grad_output


class FusedSiLU_deriv_1(Function):
    """
    Fused SiLU first derivative implementation using nvfuser
    with activation recomputation
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        with FusionDefinition() as fd:
            silu_backward_for(fd, x.dtype, x.dim(), x.size(), x.stride())
        out = fd.execute([x])[0]
        return out

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        (x,) = ctx.saved_tensors
        return FusedSiLU_deriv_2.apply(x) * grad_output


class FusedSiLU_deriv_2(Function):
    """
    Fused SiLU second derivative implementation using nvfuser
    with activation recomputation
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        with FusionDefinition() as fd:
            silu_double_backward_for(fd, x.dtype, x.dim(), x.size(), x.stride())
        out = fd.execute([x])[0]
        return out

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        (x,) = ctx.saved_tensors
        return FusedSiLU_deriv_3.apply(x) * grad_output


class FusedSiLU_deriv_3(Function):
    """
    Fused SiLU third derivative implementation using nvfuser
    with activation recomputation
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        with FusionDefinition() as fd:
            silu_triple_backward_for(fd, x.dtype, x.dim(), x.size(), x.stride())
        out = fd.execute([x])[0]
        return out

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        (x,) = ctx.saved_tensors
        y = torch.sigmoid(x)
        dy = y * (1 - y)
        ddy = (1 - 2 * y) * dy
        dddy = (1 - 2 * y) * ddy - 2 * dy * dy
        z = 1 - 2 * (y + x * dy)
        term1 = dddy * (2 + x - 2 * x * y)
        term2 = 2 * ddy * z
        term3 = dy * (-2) * (2 * dy + x * ddy)
        return (term1 + term2 + term3) * grad_output
