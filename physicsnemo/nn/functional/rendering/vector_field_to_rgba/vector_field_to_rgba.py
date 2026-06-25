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

import torch

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import vector_field_to_rgba_torch
from ._warp_impl import vector_field_to_rgba_warp


class VectorFieldToRGBA(FunctionSpec):
    """Map vector magnitude and LIC values to an RGBA volume.

    Args:
        vector_field: Vector field with shape ``(nx, ny, nz, 3)``.
        lic_field: LIC scalar field with shape ``(nx, ny, nz)``.
        vmin: Vector magnitude mapped to the bottom of the transfer function.
        vmax: Vector magnitude mapped to the top of the transfer function.
        max_opacity: Maximum output alpha in ``[0, 1]``.
        lic_threshold: LIC values below this threshold are transparent.
        implementation: Explicit implementation name. ``"warp"`` is preferred;
            ``"torch"`` is available as a portable fallback.

    Returns:
        ``uint8`` RGBA volume with shape ``(nx, ny, nz, 4)``.
    """

    @FunctionSpec.register(name="warp", required_imports=("warp>=1.0.0",), rank=0)
    def warp_forward(
        vector_field: torch.Tensor,
        lic_field: torch.Tensor,
        vmin: float,
        vmax: float,
        max_opacity: float = 0.8,
        lic_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Run the Warp implementation for ``vector_field_to_rgba``."""
        return vector_field_to_rgba_warp(
            vector_field=vector_field,
            lic_field=lic_field,
            vmin=vmin,
            vmax=vmax,
            max_opacity=max_opacity,
            lic_threshold=lic_threshold,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        vector_field: torch.Tensor,
        lic_field: torch.Tensor,
        vmin: float,
        vmax: float,
        max_opacity: float = 0.8,
        lic_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Run the PyTorch implementation for ``vector_field_to_rgba``."""
        return vector_field_to_rgba_torch(
            vector_field=vector_field,
            lic_field=lic_field,
            vmin=vmin,
            vmax=vmax,
            max_opacity=max_opacity,
            lic_threshold=lic_threshold,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield benchmark inputs for vector RGBA transfer."""
        device = torch.device(device)
        coords = torch.linspace(-1.0, 1.0, 16, device=device)
        x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
        vector_field = torch.stack([-y, x, 0.25 * torch.ones_like(z)], dim=-1)
        lic_field = torch.full((16, 16, 16), 0.8, device=device)
        yield ("vortex16", (vector_field, lic_field, 0.0, 1.5), {})

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare Warp and PyTorch transfer outputs."""
        torch.testing.assert_close(output, reference, atol=1, rtol=0)


vector_field_to_rgba = VectorFieldToRGBA.make_function("vector_field_to_rgba")

__all__ = ["VectorFieldToRGBA", "vector_field_to_rgba"]
