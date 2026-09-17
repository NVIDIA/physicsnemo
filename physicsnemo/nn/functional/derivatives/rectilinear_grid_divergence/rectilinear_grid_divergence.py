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

from ._torch_impl import rectilinear_grid_divergence_torch
from ._warp_impl import rectilinear_grid_divergence_warp


def _periodic_rectilinear_coordinates(
    shape: tuple[int, ...],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], tuple[float, ...]]:
    amps = (0.04, 0.03, 0.02)
    coords: list[torch.Tensor] = []
    for axis, n in enumerate(shape):
        s = torch.linspace(0.0, 1.0, n + 1, device=device)[:-1]
        coords.append(
            (s + amps[axis] * torch.sin(2.0 * torch.pi * s)).to(torch.float32)
        )
    return tuple(coords), tuple(1.0 for _ in shape)


def _make_vector_field(
    shape: tuple[int, ...],
    coordinates: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if len(shape) == 1:
        (x0,) = coordinates
        return torch.sin(2.0 * torch.pi * x0).unsqueeze(0).to(torch.float32)

    if len(shape) == 2:
        x0, x1 = coordinates
        xx, yy = torch.meshgrid(x0, x1, indexing="ij")
        return torch.stack(
            (
                torch.sin(2.0 * torch.pi * xx) * torch.cos(2.0 * torch.pi * yy),
                torch.cos(2.0 * torch.pi * xx) * torch.sin(2.0 * torch.pi * yy),
            ),
            dim=0,
        ).to(torch.float32)

    x0, x1, x2 = coordinates
    xx, yy, zz = torch.meshgrid(x0, x1, x2, indexing="ij")
    return torch.stack(
        (
            torch.sin(2.0 * torch.pi * xx) * torch.cos(2.0 * torch.pi * yy),
            torch.cos(2.0 * torch.pi * yy) * torch.sin(2.0 * torch.pi * zz),
            torch.sin(2.0 * torch.pi * zz) * torch.cos(2.0 * torch.pi * xx),
        ),
        dim=0,
    ).to(torch.float32)


class RectilinearGridDivergence(FunctionSpec):
    r"""Compute periodic divergence on rectilinear grids with nonuniform spacing.

    This functional evaluates the divergence of a channel-first vector field on
    a 1D/2D/3D periodic rectilinear grid. Each coordinate axis may have
    independent nonuniform spacing, and the derivative along each axis uses the
    same second-order central-difference stencil as
    :func:`physicsnemo.nn.functional.rectilinear_grid_gradient`.

    Parameters
    ----------
    vector_field : torch.Tensor
        Channel-first vector field with shape ``(dim, *grid_shape)`` where
        ``dim`` is 1, 2, or 3.
    coordinates : Sequence[torch.Tensor]
        Per-axis coordinate tensors matching ``grid_shape``. Each tensor must
        be rank-1, strictly increasing, and on the same device as
        ``vector_field``.
    periods : float | Sequence[float] | None, optional
        Period length per axis. If ``None``, each period is inferred from the
        coordinate span plus the first spacing.
    implementation : {"warp", "torch"} or None
        Explicit backend selection. When ``None``, dispatch selects by rank.

    Returns
    -------
    torch.Tensor
        Scalar divergence field with shape ``grid_shape``.
    """

    _BENCHMARK_CASES = (
        ("1d-n8192", (8192,)),
        ("2d-384x384", (384, 384)),
        ("3d-96x96x96", (96, 96, 96)),
    )

    _COMPARE_ATOL = 5e-2
    _COMPARE_RTOL = 5e-2
    _COMPARE_BACKWARD_ATOL = 5e-2
    _COMPARE_BACKWARD_RTOL = 5e-2

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        vector_field: torch.Tensor,
        coordinates: Sequence[torch.Tensor],
        periods: float | Sequence[float] | None = None,
    ) -> torch.Tensor:
        """Dispatch rectilinear divergence to the Warp backend."""
        return rectilinear_grid_divergence_warp(
            vector_field=vector_field,
            coordinates=coordinates,
            periods=periods,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        vector_field: torch.Tensor,
        coordinates: Sequence[torch.Tensor],
        periods: float | Sequence[float] | None = None,
    ) -> torch.Tensor:
        """Dispatch rectilinear divergence to eager PyTorch."""
        return rectilinear_grid_divergence_torch(
            vector_field=vector_field,
            coordinates=coordinates,
            periods=periods,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark and parity input cases."""
        device = torch.device(device)
        for label, shape in cls._BENCHMARK_CASES:
            coordinates, periods = _periodic_rectilinear_coordinates(shape, device)
            vector_field = _make_vector_field(shape, coordinates)
            yield label, (vector_field, coordinates), {"periods": periods}

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative backward benchmark and parity input cases."""
        device = torch.device(device)
        backward_cases = (
            ("1d-grad-n4096", (4096,)),
            ("2d-grad-256x256", (256, 256)),
            ("3d-grad-64x64x64", (64, 64, 64)),
        )
        for label, shape in backward_cases:
            coordinates, periods = _periodic_rectilinear_coordinates(shape, device)
            vector_field = _make_vector_field(shape, coordinates)
            yield (
                label,
                (vector_field.detach().clone().requires_grad_(True), coordinates),
                {"periods": periods},
            )

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare forward outputs across implementations."""
        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_ATOL,
            rtol=cls._COMPARE_RTOL,
        )

    @classmethod
    def compare_backward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare backward gradients across implementations."""
        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_BACKWARD_ATOL,
            rtol=cls._COMPARE_BACKWARD_RTOL,
        )


rectilinear_grid_divergence = RectilinearGridDivergence.make_function(
    "rectilinear_grid_divergence"
)


__all__ = ["RectilinearGridDivergence", "rectilinear_grid_divergence"]
