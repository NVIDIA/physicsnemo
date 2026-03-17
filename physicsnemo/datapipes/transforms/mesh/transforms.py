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
Deterministic mesh transforms (Mesh -> Mesh).
"""

from __future__ import annotations

from typing import Literal

import torch

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.datapipes.transforms.subsample import poisson_sample_indices_fixed
from physicsnemo.mesh import Mesh


@register()
class ScaleMesh(MeshTransform):
    r"""Scale mesh geometry (and optionally point/cell/global data) by a uniform factor."""

    def __init__(
        self,
        factor: float | torch.Tensor,
        transform_point_data: bool = False,
        transform_cell_data: bool = False,
        transform_global_data: bool = False,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.transform_point_data = transform_point_data
        self.transform_cell_data = transform_cell_data
        self.transform_global_data = transform_global_data

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh.scale(
            self.factor,
            transform_point_data=self.transform_point_data,
            transform_cell_data=self.transform_cell_data,
            transform_global_data=self.transform_global_data,
        )

    def extra_repr(self) -> str:
        return f"factor={self.factor}"


@register()
class TranslateMesh(MeshTransform):
    r"""Translate mesh geometry by a vector."""

    def __init__(self, vector: torch.Tensor | list[float]) -> None:
        super().__init__()
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=torch.float32)
        self.vector = vector

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh.translate(self.vector.to(mesh.points.device))

    def extra_repr(self) -> str:
        return f"vector={self.vector.tolist()}"


@register()
class RotateMesh(MeshTransform):
    r"""Rotate mesh geometry (and optionally point/cell/global data) about an axis."""

    def __init__(
        self,
        angle: float,
        axis: torch.Tensor | list | tuple | Literal["x", "y", "z"] | None = None,
        center: torch.Tensor | list | tuple | None = None,
        transform_point_data: bool = False,
        transform_cell_data: bool = False,
        transform_global_data: bool = False,
    ) -> None:
        super().__init__()
        self.angle = angle
        self.axis = axis
        self.center = center
        self.transform_point_data = transform_point_data
        self.transform_cell_data = transform_cell_data
        self.transform_global_data = transform_global_data

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh.rotate(
            self.angle,
            axis=self.axis,
            center=self.center,
            transform_point_data=self.transform_point_data,
            transform_cell_data=self.transform_cell_data,
            transform_global_data=self.transform_global_data,
        )

    def extra_repr(self) -> str:
        parts = [f"angle={self.angle}"]
        if self.axis is not None:
            parts.append(f"axis={self.axis}")
        if self.center is not None:
            parts.append(f"center={self.center}")
        return ", ".join(parts)


@register()
class CenterMesh(MeshTransform):
    r"""Translate mesh so its center of mass is at the origin."""

    def __init__(self, use_area_weighting: bool = True) -> None:
        super().__init__()
        self.use_area_weighting = use_area_weighting

    def __call__(self, mesh: Mesh) -> Mesh:
        if self.use_area_weighting and mesh.n_cells > 0:
            areas = mesh.cell_areas  # (n_cells,)
            centroids = mesh.cell_centroids  # (n_cells, n_spatial_dims)
            total_area = areas.sum()
            com = (centroids * areas.unsqueeze(-1)).sum(dim=0) / total_area
        else:
            com = mesh.points.mean(dim=0)
        return mesh.translate(-com)

    def extra_repr(self) -> str:
        return f"use_area_weighting={self.use_area_weighting}"


@register()
class SubsampleMesh(MeshTransform):
    r"""Subsample a mesh to a fixed number of cells and/or points."""

    def __init__(
        self,
        n_cells: int | None = None,
        n_points: int | None = None,
    ) -> None:
        super().__init__()
        if n_cells is None and n_points is None:
            raise ValueError("At least one of n_cells or n_points must be specified.")
        self.n_cells = n_cells
        self.n_points = n_points

    def _random_indices(self, total: int, k: int, device: torch.device) -> torch.Tensor:
        if total <= k:
            return torch.arange(total, device=device)
        if total > 2**24:
            return poisson_sample_indices_fixed(total, k, device=device)
        return torch.randperm(total, device=device)[:k]

    def __call__(self, mesh: Mesh) -> Mesh:
        if self.n_cells is not None and mesh.n_cells > self.n_cells:
            indices = self._random_indices(
                mesh.n_cells, self.n_cells, mesh.cells.device
            )
            mesh = mesh.slice_cells(indices)

        if self.n_points is not None and mesh.n_points > self.n_points:
            indices = self._random_indices(
                mesh.n_points, self.n_points, mesh.points.device
            )
            mesh = mesh.slice_points(indices)

        return mesh

    def extra_repr(self) -> str:
        parts = []
        if self.n_cells is not None:
            parts.append(f"n_cells={self.n_cells}")
        if self.n_points is not None:
            parts.append(f"n_points={self.n_points}")
        return ", ".join(parts)
