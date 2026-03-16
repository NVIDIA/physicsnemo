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

"""
Random mesh augmentations (on-the-fly randomizations). Mesh -> Mesh.
"""

from __future__ import annotations

import torch

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.transformations.geometric import scale, translate


@register()
class RandomScaleMesh(MeshTransform):
    r"""Random uniform scale of mesh. Scale factor is sampled per __call__."""

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.9, 1.1),
        transform_point_data: bool = False,
        transform_cell_data: bool = False,
        transform_global_data: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.scale_range = scale_range
        self.transform_point_data = transform_point_data
        self.transform_cell_data = transform_cell_data
        self.transform_global_data = transform_global_data
        self._generator = generator

    def __call__(self, mesh: Mesh) -> Mesh:
        low, high = self.scale_range
        factor = low + (high - low) * torch.rand(
            1, device=mesh.points.device, generator=self._generator
        ).item()
        return scale(
            mesh,
            factor,
            transform_point_data=self.transform_point_data,
            transform_cell_data=self.transform_cell_data,
            transform_global_data=self.transform_global_data,
        )

    def extra_repr(self) -> str:
        return f"scale_range={self.scale_range}"


@register()
class RandomTranslateMesh(MeshTransform):
    r"""Random translation of mesh. Offset is sampled per __call__."""

    def __init__(
        self,
        max_offset: float | tuple[float, float, float] = 0.1,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if isinstance(max_offset, (int, float)):
            max_offset = (max_offset, max_offset, max_offset)
        self.max_offset = max_offset
        self._generator = generator

    def __call__(self, mesh: Mesh) -> Mesh:
        n = mesh.n_spatial_dims
        if isinstance(self.max_offset, (int, float)):
            scales = (self.max_offset,) * n
        else:
            scales = tuple(self.max_offset[i] for i in range(n))
        offset = torch.tensor(
            [
                (torch.rand(1, generator=self._generator).item() * 2 - 1) * s
                for s in scales
            ],
            device=mesh.points.device,
            dtype=mesh.points.dtype,
        )
        return translate(mesh, offset)

    def extra_repr(self) -> str:
        return f"max_offset={self.max_offset}"
