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
Small mesh transforms used by the surface dataset pipelines.

- :class:`DropDegenerateCells` checks the current coordinates for collapsed
  or non-finite cells using the same geometric measures as centroid conversion.

Recipe-local module registered into the global datapipe component
registry so components can be referenced via ``${dp:...}`` in Hydra
YAML configs.

Import this module before Hydra instantiation to register the components.
"""

from __future__ import annotations

from warnings import warn

import torch

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry import compute_cell_areas


@register()
class DropDegenerateCells(MeshTransform):
    r"""Drop cells with non-finite or degenerate current geometry.

    Recompute geometric measures from the current coordinates, in the mesh's
    dtype, using the same area routine as ``Mesh.cell_areas``. Its direct
    triangle area calculation preserves thin valid faces without Gram
    cancellation. Cached areas are ignored: centering, rotation, and scaling
    can collapse a face through rounding. Cells whose area is zero or
    non-finite in the mesh's dtype cannot supply usable quadrature weights.

    Place this last in the transform chain so it sees the same coordinates
    the model will. Meshes without rejected cells pass through unchanged.
    Only cells and their associated data are sliced; vertices are retained.
    """

    def __call__(self, mesh: Mesh) -> Mesh:
        if mesh.n_cells == 0:
            return mesh
        cell_points = mesh.points[mesh.cells]
        edges = cell_points[:, 1:] - cell_points[:, :1]
        finite_points = torch.isfinite(cell_points).all(dim=(-2, -1))
        areas = compute_cell_areas(edges)
        keep = finite_points & torch.isfinite(areas) & (areas > 0)
        n_bad = int((~keep).sum())
        if n_bad == 0:
            return mesh
        warn(
            f"DropDegenerateCells: dropping {n_bad} cell(s) with "
            "non-finite or degenerate geometry"
        )
        return mesh.slice_cells(keep)
