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

from collections.abc import Sequence

import torch

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import spectral_grid_laplacian_torch


class SpectralGridLaplacian(FunctionSpec):
    r"""Compute a periodic scalar Laplacian with Fourier differentiation.

    The Laplacian is evaluated directly in Fourier space using the multiplier
    :math:`-\lvert k \rvert^2`.

    Parameters
    ----------
    field : torch.Tensor
        Scalar field on a periodic uniform grid with shape ``(n0,)``,
        ``(n0, n1)``, or ``(n0, n1, n2)``.
    lengths : float | Sequence[float], optional
        Physical domain lengths per spatial axis. A scalar applies the same
        length to every axis.
    implementation : {"torch"} or None
        Implementation to use. When ``None``, dispatch selects the available
        implementation.

    Returns
    -------
    torch.Tensor
        Scalar Laplacian with the same shape as ``field``.
    """

    _BENCHMARK_CASES = (
        ("1d-n4096", (4096,), 2.0),
        ("2d-512x512", (512, 512), (2.0, 1.5)),
        ("3d-128x128x128", (128, 128, 128), (2.0, 1.5, 1.25)),
    )
    _BACKWARD_CASES = (
        ("1d-grad-n1024", (1024,), 2.0),
        ("2d-grad-256x256", (256, 256), (2.0, 1.5)),
        ("3d-grad-64x64x64", (64, 64, 64), (2.0, 1.5, 1.25)),
    )

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(
        field: torch.Tensor,
        lengths: float | Sequence[float] = 1.0,
    ) -> torch.Tensor:
        """Dispatch spectral Laplacian to the PyTorch backend."""
        return spectral_grid_laplacian_torch(field=field, lengths=lengths)

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark input cases."""
        device = torch.device(device)
        for label, shape, lengths in cls._BENCHMARK_CASES:
            yield (
                label,
                (_make_periodic_scalar_field(shape, device=device),),
                {"lengths": lengths},
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative backward benchmark input cases."""
        device = torch.device(device)
        for label, shape, lengths in cls._BACKWARD_CASES:
            field = (
                _make_periodic_scalar_field(shape, device=device)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            yield label, (field,), {"lengths": lengths}


def _make_periodic_scalar_field(
    shape: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Construct a smooth periodic scalar field for benchmark cases."""
    axes = tuple(
        torch.arange(size, device=device, dtype=torch.float32) / float(size)
        for size in shape
    )
    mesh = torch.meshgrid(*axes, indexing="ij")
    field = torch.sin(2.0 * torch.pi * mesh[0])
    for axis, coordinates in enumerate(mesh[1:], start=1):
        amplitude = 0.5**axis
        if axis % 2 == 0:
            field = field + amplitude * torch.sin(2.0 * torch.pi * coordinates)
        else:
            field = field + amplitude * torch.cos(2.0 * torch.pi * coordinates)
    return field


spectral_grid_laplacian = SpectralGridLaplacian.make_function("spectral_grid_laplacian")

__all__ = ["SpectralGridLaplacian", "spectral_grid_laplacian"]
