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


# Normalize one material input to a spatial float32 field on the target device.
def _normalize_material_field(
    material: float | torch.Tensor,
    name: str,
    spatial_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    if isinstance(material, (int, float)):
        return torch.full(
            spatial_shape,
            float(material),
            device=device,
            dtype=torch.float32,
        )

    if not isinstance(material, torch.Tensor):
        raise TypeError(f"{name} must be a float or torch.Tensor")
    if material.dtype != torch.float32:
        raise TypeError(f"{name} tensor must be float32")
    if material.device != device:
        raise ValueError(f"{name} tensor must be on the same device as electric_field")

    if material.ndim == 4:
        if material.shape[0] != 1:
            raise ValueError(
                f"{name} with 4 dimensions must have shape (1, nx, ny, nz)"
            )
        field = material[0]
    elif material.ndim == 3:
        field = material
    else:
        raise ValueError(f"{name} must have shape (nx, ny, nz) or (1, nx, ny, nz)")

    if tuple(field.shape) != spatial_shape:
        raise ValueError(f"{name} spatial shape must match electric_field")
    return field


# Validate tensor layout and guard unsupported in-place autograd usage.
def _validate_common_inputs(
    electric_field: torch.Tensor,
    magnetic_field: torch.Tensor,
    eps: float | torch.Tensor,
    sigma_e: float | torch.Tensor,
    spacing: torch.Tensor | Sequence[float],
    impressed_current: torch.Tensor | None,
    inplace: bool,
) -> None:
    if electric_field.dtype != torch.float32:
        raise TypeError("electric_field must be float32")
    if magnetic_field.dtype != torch.float32:
        raise TypeError("magnetic_field must be float32")

    if electric_field.ndim != 4 or electric_field.shape[0] != 3:
        raise ValueError("electric_field must have shape (3, nx, ny, nz)")
    if magnetic_field.ndim != 4 or magnetic_field.shape[0] != 3:
        raise ValueError("magnetic_field must have shape (3, nx, ny, nz)")
    if electric_field.shape[1:] != magnetic_field.shape[1:]:
        raise ValueError("electric_field and magnetic_field must share spatial shape")

    if electric_field.device != magnetic_field.device:
        raise ValueError("electric_field and magnetic_field must be on the same device")

    spatial_shape = tuple(electric_field.shape[1:])
    for material, name in ((eps, "eps"), (sigma_e, "sigma_e")):
        if isinstance(material, (int, float)):
            continue
        if not isinstance(material, torch.Tensor):
            raise TypeError(f"{name} must be a float or torch.Tensor")
        if material.dtype != torch.float32:
            raise TypeError(f"{name} tensor must be float32")
        if material.device != electric_field.device:
            raise ValueError(
                f"{name} tensor must be on the same device as electric_field"
            )
        if material.ndim == 4:
            if material.shape[0] != 1:
                raise ValueError(
                    f"{name} with 4 dimensions must have shape (1, nx, ny, nz)"
                )
            if tuple(material.shape[1:]) != spatial_shape:
                raise ValueError(f"{name} spatial shape must match electric_field")
        elif material.ndim == 3:
            if tuple(material.shape) != spatial_shape:
                raise ValueError(f"{name} spatial shape must match electric_field")
        else:
            raise ValueError(f"{name} must have shape (nx, ny, nz) or (1, nx, ny, nz)")

    if isinstance(spacing, torch.Tensor):
        if spacing.device != electric_field.device:
            raise ValueError("spacing tensor must be on the same device as electric_field")
        if spacing.requires_grad:
            raise ValueError("spacing gradients are not supported")

    if impressed_current is not None:
        if impressed_current.dtype != torch.float32:
            raise TypeError("impressed_current must be float32")
        if impressed_current.ndim != 4 or impressed_current.shape[0] != 3:
            raise ValueError("impressed_current must have shape (3, nx, ny, nz)")
        if impressed_current.device != electric_field.device:
            raise ValueError(
                "impressed_current must be on the same device as electric_field"
            )

    if inplace:
        needs_grad = [
            electric_field.requires_grad,
            magnetic_field.requires_grad,
            isinstance(eps, torch.Tensor) and eps.requires_grad,
            isinstance(sigma_e, torch.Tensor) and sigma_e.requires_grad,
        ]
        if impressed_current is not None:
            needs_grad.append(impressed_current.requires_grad)
        if any(needs_grad):
            raise ValueError(
                "inplace=True is not supported when any input requires gradients"
            )


# Normalize impressed-current offset to a 3-tuple of ints.
def _normalize_offset(offset: torch.Tensor | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(offset, torch.Tensor):
        if offset.numel() != 3:
            raise ValueError("impressed_current_offset must have 3 elements")
        return tuple(int(v) for v in offset.detach().cpu().flatten().tolist())

    if len(offset) != 3:
        raise ValueError("impressed_current_offset must have 3 elements")
    return int(offset[0]), int(offset[1]), int(offset[2])


# Normalize spacing to a device tensor of shape (3,).
def _as_spacing_tensor(
    spacing: torch.Tensor | Sequence[float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(spacing, torch.Tensor):
        spacing_tensor = spacing.to(device=device, dtype=dtype)
    else:
        spacing_tensor = torch.tensor(tuple(spacing), device=device, dtype=dtype)

    if spacing_tensor.numel() != 3:
        raise ValueError("spacing must contain exactly 3 values")
    return spacing_tensor.reshape(3)


# Compute Yee-cell electric update coefficients from spatial material fields.
def _material_coefficients(
    eps: float | torch.Tensor,
    sigma_e: float | torch.Tensor,
    spacing: torch.Tensor,
    dt: float,
    spatial_shape: tuple[int, int, int],
    device: torch.device,
    expand_scalar_to_spatial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps_is_scalar = isinstance(eps, (int, float))
    sigma_is_scalar = isinstance(sigma_e, (int, float))

    eps_x: torch.Tensor | float
    eps_y: torch.Tensor | float
    eps_z: torch.Tensor | float
    sigma_x: torch.Tensor | float
    sigma_y: torch.Tensor | float
    sigma_z: torch.Tensor | float

    if eps_is_scalar:
        eps_x = float(eps)
        eps_y = float(eps)
        eps_z = float(eps)
    else:
        eps_field = _normalize_material_field(eps, "eps", spatial_shape, device)
        eps_0_0_1 = torch.roll(eps_field, shifts=(1, 1, 0), dims=(0, 1, 2))
        eps_0_1_0 = torch.roll(eps_field, shifts=(1, 0, 1), dims=(0, 1, 2))
        eps_0_1_1 = torch.roll(eps_field, shifts=(1, 0, 0), dims=(0, 1, 2))
        eps_1_0_0 = torch.roll(eps_field, shifts=(0, 1, 1), dims=(0, 1, 2))
        eps_1_0_1 = torch.roll(eps_field, shifts=(0, 1, 0), dims=(0, 1, 2))
        eps_1_1_0 = torch.roll(eps_field, shifts=(0, 0, 1), dims=(0, 1, 2))
        eps_1_1_1 = eps_field
        eps_x = (eps_1_1_1 + eps_1_1_0 + eps_1_0_1 + eps_1_0_0) * 0.25
        eps_y = (eps_1_1_1 + eps_1_1_0 + eps_0_1_1 + eps_0_1_0) * 0.25
        eps_z = (eps_1_1_1 + eps_1_0_1 + eps_0_1_1 + eps_0_0_1) * 0.25

    if sigma_is_scalar:
        sigma_x = float(sigma_e)
        sigma_y = float(sigma_e)
        sigma_z = float(sigma_e)
    else:
        sigma_e_field = _normalize_material_field(
            sigma_e, "sigma_e", spatial_shape, device
        )
        sigma_0_0_1 = torch.roll(sigma_e_field, shifts=(1, 1, 0), dims=(0, 1, 2))
        sigma_0_1_0 = torch.roll(sigma_e_field, shifts=(1, 0, 1), dims=(0, 1, 2))
        sigma_0_1_1 = torch.roll(sigma_e_field, shifts=(1, 0, 0), dims=(0, 1, 2))
        sigma_1_0_0 = torch.roll(sigma_e_field, shifts=(0, 1, 1), dims=(0, 1, 2))
        sigma_1_0_1 = torch.roll(sigma_e_field, shifts=(0, 1, 0), dims=(0, 1, 2))
        sigma_1_1_0 = torch.roll(sigma_e_field, shifts=(0, 0, 1), dims=(0, 1, 2))
        sigma_1_1_1 = sigma_e_field
        sigma_x = (sigma_1_1_1 + sigma_1_1_0 + sigma_1_0_1 + sigma_1_0_0) * 0.25
        sigma_y = (sigma_1_1_1 + sigma_1_1_0 + sigma_0_1_1 + sigma_0_1_0) * 0.25
        sigma_z = (sigma_1_1_1 + sigma_1_0_1 + sigma_0_1_1 + sigma_0_0_1) * 0.25

    if eps_is_scalar and sigma_is_scalar:
        eps_tensor = torch.tensor(
            [eps_x, eps_y, eps_z],  # type: ignore[list-item]
            device=device,
            dtype=torch.float32,
        ).view(3, 1, 1, 1)
        sigma_tensor = torch.tensor(
            [sigma_x, sigma_y, sigma_z],  # type: ignore[list-item]
            device=device,
            dtype=torch.float32,
        ).view(3, 1, 1, 1)
    else:
        if eps_is_scalar:
            assert isinstance(sigma_x, torch.Tensor)
            assert isinstance(sigma_y, torch.Tensor)
            assert isinstance(sigma_z, torch.Tensor)
            eps_x = torch.full_like(sigma_x, float(eps_x))
            eps_y = torch.full_like(sigma_y, float(eps_y))
            eps_z = torch.full_like(sigma_z, float(eps_z))
        if sigma_is_scalar:
            assert isinstance(eps_x, torch.Tensor)
            assert isinstance(eps_y, torch.Tensor)
            assert isinstance(eps_z, torch.Tensor)
            sigma_x = torch.full_like(eps_x, float(sigma_x))
            sigma_y = torch.full_like(eps_y, float(sigma_y))
            sigma_z = torch.full_like(eps_z, float(sigma_z))

        eps_tensor = torch.stack((eps_x, eps_y, eps_z), dim=0)  # type: ignore[arg-type]
        sigma_tensor = torch.stack((sigma_x, sigma_y, sigma_z), dim=0)  # type: ignore[arg-type]

    dt_tensor = torch.tensor(
        dt,
        device=eps_tensor.device,
        dtype=eps_tensor.dtype,
    )
    denom = 2.0 * eps_tensor + sigma_tensor * dt_tensor
    c_ee = (2.0 * eps_tensor - sigma_tensor * dt_tensor) / denom

    spacing_view = spacing.view(3, 1, 1, 1)
    c_eh = (2.0 * dt_tensor) / (spacing_view * denom)
    c_ej = (-2.0 * dt_tensor) / denom

    if eps_is_scalar and sigma_is_scalar and expand_scalar_to_spatial:
        nx, ny, nz = spatial_shape
        c_ee = c_ee.expand(3, nx, ny, nz).contiguous()
        c_eh = c_eh.expand(3, nx, ny, nz).contiguous()
        c_ej = c_ej.expand(3, nx, ny, nz).contiguous()

    return c_ee, c_eh, c_ej
