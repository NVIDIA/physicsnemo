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

from typing import Any

import torch
from torch.overrides import handle_torch_function, has_torch_function

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.core.version_check import OptionalImport, get_installed_version

_natten = OptionalImport("natten")


def _make_qkv(
    shape: tuple[int, ...],
    device: torch.device | str,
    *,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create query/key/value tensors for NAT benchmark inputs."""

    q = torch.randn(shape, device=device, requires_grad=requires_grad)
    k = torch.randn(shape, device=device, requires_grad=requires_grad)
    v = torch.randn(shape, device=device, requires_grad=requires_grad)
    return q, k, v


class _NeighborhoodAttention(FunctionSpec):
    """Shared behavior for NATTEN-backed function specs."""

    _NATTEN_REQUIREMENT = "natten>=0.21.5"

    @classmethod
    def dispatch(cls, *args, **kwargs):
        try:
            return super().dispatch(*args, **kwargs)
        except ImportError as exc:
            installed_version = get_installed_version("natten")
            if installed_version is not None:
                raise ImportError(
                    f"{cls._NATTEN_REQUIREMENT} is required for {cls.__name__}, "
                    f"but found natten {installed_version}"
                ) from exc
            try:
                _natten.functional
            except ImportError as missing_exc:
                raise missing_exc from exc
            raise ImportError(
                f"No available NATTEN implementation found for {cls.__name__}. "
                f"Expected {cls._NATTEN_REQUIREMENT}; verify that the installed "
                "natten package exposes the required functional backend."
            ) from exc


class NeighborhoodAttention1D(_NeighborhoodAttention):
    """Compute 1D neighborhood attention through the NATTEN backend."""

    _BENCHMARK_CASES = (
        ("small-l64-h2-d32-k3", (1, 64, 2, 32), 3, 1),
        ("medium-l256-h4-d32-k5", (1, 256, 4, 32), 5, 1),
        ("large-l1024-h8-d64-k7-d2", (1, 1024, 8, 64), 7, 2),
    )

    @FunctionSpec.register(
        name="natten",
        required_imports=(_NeighborhoodAttention._NATTEN_REQUIREMENT,),
        rank=0,
        baseline=True,
    )
    def natten_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kernel_size: int,
        dilation: int = 1,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the 1D NATTEN backend implementation."""
        return _natten.functional.na1d(
            q, k, v, kernel_size, dilation=dilation, **kwargs
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield labeled forward benchmark cases for 1D neighborhood attention."""
        for label, shape, kernel_size, dilation in cls._BENCHMARK_CASES:
            yield (
                label,
                (*_make_qkv(shape, device), kernel_size),
                {"dilation": dilation},
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield differentiable benchmark cases for 1D neighborhood attention."""
        for label, shape, kernel_size, dilation in cls._BENCHMARK_CASES:
            yield (
                label,
                (*_make_qkv(shape, device, requires_grad=True), kernel_size),
                {"dilation": dilation},
            )

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare 1D neighborhood-attention outputs against a reference."""
        torch.testing.assert_close(output, reference)


class NeighborhoodAttention2D(_NeighborhoodAttention):
    """Compute 2D neighborhood attention through the NATTEN backend."""

    _BENCHMARK_CASES = (
        ("small-32x32-h2-d32-k3", (1, 32, 32, 2, 32), 3, 1),
        ("medium-64x64-h4-d32-k5", (1, 64, 64, 4, 32), 5, 1),
        ("large-128x128-h8-d64-k7-d2", (1, 128, 128, 8, 64), 7, 2),
    )

    @FunctionSpec.register(
        name="natten",
        required_imports=(_NeighborhoodAttention._NATTEN_REQUIREMENT,),
        rank=0,
        baseline=True,
    )
    def natten_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kernel_size: int,
        dilation: int = 1,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the 2D NATTEN backend implementation."""
        return _natten.functional.na2d(
            q, k, v, kernel_size, dilation=dilation, **kwargs
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield labeled forward benchmark cases for 2D neighborhood attention."""
        for label, shape, kernel_size, dilation in cls._BENCHMARK_CASES:
            yield (
                label,
                (*_make_qkv(shape, device), kernel_size),
                {"dilation": dilation},
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield differentiable benchmark cases for 2D neighborhood attention."""
        for label, shape, kernel_size, dilation in cls._BENCHMARK_CASES:
            yield (
                label,
                (*_make_qkv(shape, device, requires_grad=True), kernel_size),
                {"dilation": dilation},
            )

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare 2D neighborhood-attention outputs against a reference."""
        torch.testing.assert_close(output, reference)


class NeighborhoodAttention3D(_NeighborhoodAttention):
    """Compute 3D neighborhood attention through the NATTEN backend."""

    _BENCHMARK_CASES = (
        ("small-8x8x8-h2-d16-k3", (1, 8, 8, 8, 2, 16), 3, 1),
        ("medium-16x16x16-h4-d32-k5", (1, 16, 16, 16, 4, 32), 5, 1),
        ("large-32x32x32-h4-d32-k7", (1, 32, 32, 32, 4, 32), 7, 1),
    )

    @FunctionSpec.register(
        name="natten",
        required_imports=(_NeighborhoodAttention._NATTEN_REQUIREMENT,),
        rank=0,
        baseline=True,
    )
    def natten_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kernel_size: int,
        dilation: int = 1,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the 3D NATTEN backend implementation."""
        return _natten.functional.na3d(
            q, k, v, kernel_size, dilation=dilation, **kwargs
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield labeled forward benchmark cases for 3D neighborhood attention."""
        for label, shape, kernel_size, dilation in cls._BENCHMARK_CASES:
            yield (
                label,
                (*_make_qkv(shape, device), kernel_size),
                {"dilation": dilation},
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield differentiable benchmark cases for 3D neighborhood attention."""
        for label, shape, kernel_size, dilation in cls._BENCHMARK_CASES:
            yield (
                label,
                (*_make_qkv(shape, device, requires_grad=True), kernel_size),
                {"dilation": dilation},
            )

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare 3D neighborhood-attention outputs against a reference."""
        torch.testing.assert_close(output, reference)


# Keep the FunctionSpec-produced callables private and expose a second public
# wrapper layer so NAT can first route ShardTensor inputs through PyTorch's
# ``__torch_function__`` protocol. This preserves the domain-parallel halo
# exchange path before falling back to normal FunctionSpec backend dispatch.
# TODO: Generalize this pattern in FunctionSpec once there is a broader design
# for functionals that need tensor-subclass dispatch before backend selection.
_na1d = NeighborhoodAttention1D.make_function("na1d")
_na2d = NeighborhoodAttention2D.make_function("na2d")
_na3d = NeighborhoodAttention3D.make_function("na3d")


def na1d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    r"""Compute 1D neighborhood attention, with ``__torch_function__`` dispatch.

    This is a thin wrapper around :func:`natten.functional.na1d` that enables
    automatic dispatch through PyTorch's ``__torch_function__`` protocol. When
    called with a tensor subclass (e.g. ``ShardTensor``), the registered handler
    is invoked instead of the underlying natten implementation.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape :math:`(B, L, \text{heads}, D)`.
    k : torch.Tensor
        Key tensor of shape :math:`(B, L, \text{heads}, D)`.
    v : torch.Tensor
        Value tensor of shape :math:`(B, L, \text{heads}, D)`.
    kernel_size : int
        Size of the attention kernel window.
    dilation : int, default=1
        Dilation factor for the attention kernel.
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`natten.functional.na1d`
        (e.g. ``is_causal``, ``scale``).

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as ``q``.
    """
    if has_torch_function((q, k, v)):
        return handle_torch_function(
            na1d,
            (q, k, v),
            q,
            k,
            v,
            kernel_size,
            dilation=dilation,
            **kwargs,
        )
    return _na1d(q, k, v, kernel_size, dilation=dilation, **kwargs)


def na2d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    r"""Compute 2D neighborhood attention, with ``__torch_function__`` dispatch.

    This is a thin wrapper around :func:`natten.functional.na2d` that enables
    automatic dispatch through PyTorch's ``__torch_function__`` protocol. When
    called with a tensor subclass (e.g. ``ShardTensor``), the registered handler
    is invoked instead of the underlying natten implementation.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape :math:`(B, H, W, \text{heads}, D)`.
    k : torch.Tensor
        Key tensor of shape :math:`(B, H, W, \text{heads}, D)`.
    v : torch.Tensor
        Value tensor of shape :math:`(B, H, W, \text{heads}, D)`.
    kernel_size : int
        Size of the attention kernel window.
    dilation : int, default=1
        Dilation factor for the attention kernel.
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`natten.functional.na2d`
        (e.g. ``is_causal``, ``scale``).

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as ``q``.
    """
    if has_torch_function((q, k, v)):
        return handle_torch_function(
            na2d,
            (q, k, v),
            q,
            k,
            v,
            kernel_size,
            dilation=dilation,
            **kwargs,
        )
    return _na2d(q, k, v, kernel_size, dilation=dilation, **kwargs)


def na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    r"""Compute 3D neighborhood attention, with ``__torch_function__`` dispatch.

    This is a thin wrapper around :func:`natten.functional.na3d` that enables
    automatic dispatch through PyTorch's ``__torch_function__`` protocol. When
    called with a tensor subclass (e.g. ``ShardTensor``), the registered handler
    is invoked instead of the underlying natten implementation.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape :math:`(B, X, Y, Z, \text{heads}, D)`.
    k : torch.Tensor
        Key tensor of shape :math:`(B, X, Y, Z, \text{heads}, D)`.
    v : torch.Tensor
        Value tensor of shape :math:`(B, X, Y, Z, \text{heads}, D)`.
    kernel_size : int
        Size of the attention kernel window.
    dilation : int, default=1
        Dilation factor for the attention kernel.
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`natten.functional.na3d`
        (e.g. ``is_causal``, ``scale``).

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as ``q``.
    """
    if has_torch_function((q, k, v)):
        return handle_torch_function(
            na3d,
            (q, k, v),
            q,
            k,
            v,
            kernel_size,
            dilation=dilation,
            **kwargs,
        )
    return _na3d(q, k, v, kernel_size, dilation=dilation, **kwargs)


__all__ = [
    "NeighborhoodAttention1D",
    "NeighborhoodAttention2D",
    "NeighborhoodAttention3D",
    "na1d",
    "na2d",
    "na3d",
]
