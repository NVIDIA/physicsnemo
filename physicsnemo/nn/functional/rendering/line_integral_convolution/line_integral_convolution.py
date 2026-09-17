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

from ._warp_impl import line_integral_convolution_warp


class LineIntegralConvolution(FunctionSpec):
    """Compute a 3D line integral convolution field.

    Args:
        vector_field: Vector field with shape ``(nx, ny, nz, 3)``.
        seed: Scalar seed/noise field with shape ``(nx, ny, nz)``.
        step_size: Integration step size in grid-index units.
        num_steps: Number of integration steps in each direction.
        contrast: Contrast multiplier around ``0.5`` for the output LIC field.
        implementation: Explicit implementation name. Currently only ``"warp"``
            is registered.

    Returns:
        LIC scalar field with shape ``(nx, ny, nz)`` and values in ``[0, 1]``.
    """

    @FunctionSpec.register(
        name="warp", required_imports=("warp>=1.0.0",), rank=0, baseline=True
    )
    def warp_forward(
        vector_field: torch.Tensor,
        seed: torch.Tensor,
        step_size: float = 0.5,
        num_steps: int = 20,
        contrast: float = 1.4,
    ) -> torch.Tensor:
        """Run the Warp implementation for ``line_integral_convolution``."""
        return line_integral_convolution_warp(
            vector_field=vector_field,
            seed=seed,
            step_size=step_size,
            num_steps=num_steps,
            contrast=contrast,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield benchmark inputs for line integral convolution."""
        device = torch.device(device)
        coords = torch.linspace(-1.0, 1.0, 16, device=device)
        x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
        vector_field = torch.stack([-y, x, 0.25 * torch.ones_like(z)], dim=-1)
        seed = torch.rand(16, 16, 16, device=device)
        yield ("vortex16", (vector_field, seed), {"num_steps": 8})


line_integral_convolution = LineIntegralConvolution.make_function(
    "line_integral_convolution"
)

__all__ = ["LineIntegralConvolution", "line_integral_convolution"]
