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

from ._torch_impl import scalar_field_to_rgba_torch
from ._warp_impl import scalar_field_to_rgba_warp


class ScalarFieldToRGBA(FunctionSpec):
    """Map a scalar volume to an RGBA transfer-function volume.

    Args:
        field: Scalar volume with shape ``(nx, ny, nz)``.
        vmin: Scalar value mapped to the bottom of the transfer function.
        vmax: Scalar value mapped to the top of the transfer function.
        max_opacity: Maximum output alpha in ``[0, 1]``.
        opacity_threshold: Normalized values below this threshold are transparent.
        implementation: Explicit implementation name. ``"warp"`` is preferred;
            ``"torch"`` is available as a portable fallback.

    Returns:
        ``uint8`` RGBA volume with shape ``(nx, ny, nz, 4)``.
    """

    @FunctionSpec.register(name="warp", required_imports=("warp>=1.0.0",), rank=0)
    def warp_forward(
        field: torch.Tensor,
        vmin: float,
        vmax: float,
        max_opacity: float = 0.8,
        opacity_threshold: float = 0.1,
    ) -> torch.Tensor:
        """Run the Warp implementation for ``scalar_field_to_rgba``."""
        return scalar_field_to_rgba_warp(
            field=field,
            vmin=vmin,
            vmax=vmax,
            max_opacity=max_opacity,
            opacity_threshold=opacity_threshold,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        field: torch.Tensor,
        vmin: float,
        vmax: float,
        max_opacity: float = 0.8,
        opacity_threshold: float = 0.1,
    ) -> torch.Tensor:
        """Run the PyTorch implementation for ``scalar_field_to_rgba``."""
        return scalar_field_to_rgba_torch(
            field=field,
            vmin=vmin,
            vmax=vmax,
            max_opacity=max_opacity,
            opacity_threshold=opacity_threshold,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield benchmark inputs for scalar RGBA transfer."""
        device = torch.device(device)
        coords = torch.linspace(0.0, 1.0, 24, device=device)
        x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
        field = torch.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2)
        yield ("grid24", (field, 0.0, 0.9), {"max_opacity": 0.75})

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare Warp and PyTorch transfer outputs."""
        torch.testing.assert_close(output, reference, atol=1, rtol=0)


scalar_field_to_rgba = ScalarFieldToRGBA.make_function("scalar_field_to_rgba")

__all__ = ["ScalarFieldToRGBA", "scalar_field_to_rgba"]
