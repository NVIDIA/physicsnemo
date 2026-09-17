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

from ._torch_impl import rectilinear_grid_curl_torch
from ._warp_impl import rectilinear_grid_curl_warp


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
    if len(shape) == 2:
        x0, x1 = coordinates
        xx, yy = torch.meshgrid(x0, x1, indexing="ij")
        return torch.stack(
            (
                torch.sin(2.0 * torch.pi * yy) + 0.25 * torch.cos(2.0 * torch.pi * xx),
                torch.sin(2.0 * torch.pi * xx) - 0.2 * torch.cos(2.0 * torch.pi * yy),
            ),
            dim=0,
        ).to(torch.float32)

    x0, x1, x2 = coordinates
    xx, yy, zz = torch.meshgrid(x0, x1, x2, indexing="ij")
    return torch.stack(
        (
            torch.sin(2.0 * torch.pi * yy) + 0.25 * torch.cos(2.0 * torch.pi * zz),
            torch.sin(2.0 * torch.pi * zz) - 0.2 * torch.cos(2.0 * torch.pi * xx),
            torch.sin(2.0 * torch.pi * xx) + 0.1 * torch.cos(2.0 * torch.pi * yy),
        ),
        dim=0,
    ).to(torch.float32)


class RectilinearGridCurl(FunctionSpec):
    r"""Compute periodic curl on rectilinear grids with nonuniform spacing.

    This functional evaluates the curl of a channel-first vector field on a
    2D or 3D periodic rectilinear grid. In 2D, the scalar out-of-plane curl
    ``d(v_y)/dx - d(v_x)/dy`` is returned. In 3D, the returned tensor is a
    channel-first vector field with the same shape as the input.

    Parameters
    ----------
    vector_field : torch.Tensor
        Channel-first vector field with shape ``(2, n0, n1)`` or
        ``(3, n0, n1, n2)``.
    coordinates : Sequence[torch.Tensor]
        Per-axis coordinate tensors matching the grid dimensions. Each tensor
        must be rank-1, strictly increasing, and on the same device as
        ``vector_field``.
    periods : float | Sequence[float] | None, optional
        Period length per axis. If ``None``, each period is inferred from the
        coordinate span plus the first spacing.
    implementation : {"warp", "torch"} or None
        Explicit backend selection. When ``None``, dispatch selects by rank.

    Returns
    -------
    torch.Tensor
        Scalar curl field for 2D inputs or channel-first vector curl field for
        3D inputs.

    Examples
    --------
    Compute the scalar curl of a two-dimensional periodic vector field:

    >>> import torch
    >>> x = torch.linspace(0.0, 1.0, 9)[:-1]
    >>> y = torch.linspace(0.0, 1.0, 7)[:-1]
    >>> xx, yy = torch.meshgrid(x, y, indexing="ij")
    >>> vector_field = torch.stack(
    ...     (torch.zeros_like(xx), torch.sin(2.0 * torch.pi * xx))
    ... )
    >>> curl = rectilinear_grid_curl(
    ...     vector_field,
    ...     (x, y),
    ...     periods=(1.0, 1.0),
    ...     implementation="torch",
    ... )
    >>> curl.shape
    torch.Size([8, 6])
    """

    _BENCHMARK_CASES = (
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
        """Dispatch rectilinear curl to the Warp backend."""
        return rectilinear_grid_curl_warp(
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
        """Dispatch rectilinear curl to eager PyTorch."""
        return rectilinear_grid_curl_torch(
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


rectilinear_grid_curl = RectilinearGridCurl.make_function("rectilinear_grid_curl")


__all__ = ["RectilinearGridCurl", "rectilinear_grid_curl"]
