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

from typing import Sequence

import torch

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import electric_field_update_torch
from ._warp_impl import electric_field_update_warp


class ElectricFieldUpdate(FunctionSpec):
    r"""Update the electric field for one 3D Yee FDTD timestep with periodic boundaries.

    This operator updates the electric field :math:`\mathbf{E}^{n+1}` from
    :math:`\mathbf{E}^n`, :math:`\mathbf{H}^n`, material parameters, and an optional
    impressed current :math:`\mathbf{J}`.

    The component-wise update is:

    .. math::

       \mathbf{E}^{n+1} = \mathbf{c}_{ee} \odot \mathbf{E}^{n}
       + \mathbf{c}_{eh} \odot (\nabla \times \mathbf{H}^{n})
       + \mathbf{c}_{ej} \odot \mathbf{J}

    with Yee-cell averaged material coefficients:

    .. math::

       \mathbf{c}_{ee} = \frac{2\boldsymbol{\varepsilon} - \boldsymbol{\sigma}_e \Delta t}
       {2\boldsymbol{\varepsilon} + \boldsymbol{\sigma}_e \Delta t}

    .. math::

       \mathbf{c}_{eh} =
       \frac{2\Delta t}
       {\Delta\mathbf{x} \odot (2\boldsymbol{\varepsilon} + \boldsymbol{\sigma}_e \Delta t)}

    .. math::

       \mathbf{c}_{ej} =
       \frac{-2\Delta t}
       {2\boldsymbol{\varepsilon} + \boldsymbol{\sigma}_e \Delta t}

    where :math:`\odot` is elementwise multiplication.

    Parameters
    ----------
    electric_field : torch.Tensor
        Electric field with shape ``(3, nx, ny, nz)`` and dtype ``torch.float32``.
    magnetic_field : torch.Tensor
        Magnetic field with shape ``(3, nx, ny, nz)`` and dtype ``torch.float32``.
    eps : float | torch.Tensor
        Permittivity field. May be:
        - a scalar ``float`` for uniform material,
        - a tensor of shape ``(nx, ny, nz)``, or
        - a tensor of shape ``(1, nx, ny, nz)``.
        Tensor inputs must use dtype ``torch.float32``.
    sigma_e : float | torch.Tensor
        Electric conductivity field with the same shape rules as ``eps``.
        Tensor inputs must use dtype ``torch.float32``.
    spacing : torch.Tensor | Sequence[float]
        Cell spacing ``(dx, dy, dz)``.
    dt : float
        Time step.
    impressed_current : torch.Tensor | None, optional
        Optional impressed current with shape ``(3, jx, jy, jz)``.
    impressed_current_offset : torch.Tensor | Sequence[int], optional
        Integer offset mapping ``impressed_current`` into the electric grid.
    inplace : bool, optional
        If ``True``, update ``electric_field`` in-place and return it.
        In-place mode is non-differentiable and raises when any input requires
        gradients.
    implementation : {"warp", "torch"} or None
        Implementation to use. When ``None``, dispatch selects by rank.

    Returns
    -------
    torch.Tensor
        Updated electric field with shape ``(3, nx, ny, nz)``.

    Notes
    -----
    - Boundary conditions are periodic in all three spatial dimensions.
    - This functional currently supports ``torch.float32`` tensors.
    - Gradients with respect to ``spacing`` are not supported.
    - Warp and torch implementations are intended to match numerically for
      forward and backward in out-of-place mode.
    """

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        electric_field: torch.Tensor,
        magnetic_field: torch.Tensor,
        eps: float | torch.Tensor,
        sigma_e: float | torch.Tensor,
        spacing: torch.Tensor | Sequence[float],
        dt: float,
        impressed_current: torch.Tensor | None = None,
        impressed_current_offset: torch.Tensor | Sequence[int] = (0, 0, 0),
        inplace: bool = False,
    ) -> torch.Tensor:
        return electric_field_update_warp(
            electric_field,
            magnetic_field,
            eps,
            sigma_e,
            spacing,
            dt,
            impressed_current=impressed_current,
            impressed_current_offset=impressed_current_offset,
            inplace=inplace,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        electric_field: torch.Tensor,
        magnetic_field: torch.Tensor,
        eps: float | torch.Tensor,
        sigma_e: float | torch.Tensor,
        spacing: torch.Tensor | Sequence[float],
        dt: float,
        impressed_current: torch.Tensor | None = None,
        impressed_current_offset: torch.Tensor | Sequence[int] = (0, 0, 0),
        inplace: bool = False,
    ) -> torch.Tensor:
        return electric_field_update_torch(
            electric_field,
            magnetic_field,
            eps,
            sigma_e,
            spacing,
            dt,
            impressed_current=impressed_current,
            impressed_current_offset=impressed_current_offset,
            inplace=inplace,
        )

    @classmethod
    def make_inputs(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)

        cases = [
            ("scalar-scalar-no-current", 256, False, (0, 0, 0), "scalar-scalar"),
            ("scalar-scalar-with-current", 256, True, (-8, 7, 5), "scalar-scalar"),
            ("scalar-field-no-current", 256, False, (0, 0, 0), "scalar-field"),
            ("scalar-field-with-current", 256, True, (9, -6, 4), "scalar-field"),
            ("field-scalar-no-current", 256, False, (0, 0, 0), "field-scalar"),
            ("field-scalar-with-current", 256, True, (-5, 11, -7), "field-scalar"),
            ("field-field-no-current", 256, False, (0, 0, 0), "field-field"),
            ("field-field-with-current", 256, True, (13, -9, 6), "field-field"),
        ]

        for seed, (label, grid_n, use_current, offset, material_mode) in enumerate(
            cases
        ):
            generator = torch.Generator(device=device)
            generator.manual_seed(2026 + seed)

            electric_field = torch.randn(
                3,
                grid_n,
                grid_n,
                grid_n,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            magnetic_field = torch.randn(
                3,
                grid_n,
                grid_n,
                grid_n,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            if material_mode == "scalar-scalar":
                eps = 2.5
                sigma_e = 0.01
            elif material_mode == "scalar-field":
                eps = 2.5
                sigma_e = torch.empty(
                    grid_n,
                    grid_n,
                    grid_n,
                    device=device,
                    dtype=torch.float32,
                ).uniform_(0.0, 0.06, generator=generator)
            elif material_mode == "field-scalar":
                eps = torch.empty(
                    grid_n,
                    grid_n,
                    grid_n,
                    device=device,
                    dtype=torch.float32,
                ).uniform_(1.0, 6.0, generator=generator)
                sigma_e = 0.01
            else:
                eps = torch.empty(
                    grid_n,
                    grid_n,
                    grid_n,
                    device=device,
                    dtype=torch.float32,
                ).uniform_(1.0, 6.0, generator=generator)
                sigma_e = torch.empty(
                    grid_n,
                    grid_n,
                    grid_n,
                    device=device,
                    dtype=torch.float32,
                ).uniform_(0.0, 0.06, generator=generator)
            spacing = torch.tensor(
                [0.010, 0.011, 0.013],
                device=device,
                dtype=torch.float32,
            )
            dt = 0.00125

            kwargs = {
                "impressed_current": None,
                "impressed_current_offset": offset,
                "inplace": False,
            }
            if use_current:
                current_n = max(grid_n // 2, 2)
                kwargs["impressed_current"] = torch.randn(
                    3,
                    current_n,
                    current_n,
                    current_n,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )

            yield (
                (
                    f"{label}-grid{grid_n}^3-material={material_mode}-periodic-dt{dt:.5f}"
                ),
                (
                    electric_field,
                    magnetic_field,
                    eps,
                    sigma_e,
                    spacing,
                    dt,
                ),
                kwargs,
            )

    @classmethod
    def compare(
        cls,
        output: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        torch.testing.assert_close(output, reference, atol=5e-5, rtol=1e-4)


electric_field_update = ElectricFieldUpdate.make_function("electric_field_update")


__all__ = ["ElectricFieldUpdate", "electric_field_update"]
