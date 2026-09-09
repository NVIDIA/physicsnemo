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

- :class:`ComputeFreestreamDirection` writes the unit freestream direction
  into ``global_data`` as a new leaf.  ``NonDimensionalizeByMetadata``
  scales fields and geometry but never ``global_data``, so this is the
  only nondimensional form of the freestream a model can read.
- :class:`DropDegenerateCells` checks the current coordinates for collapsed
  or non-finite cells. For 3D surface triangles, a direct cross product
  retains thin valid faces even when the float32 Gram area cancels to zero.

Recipe-local module registered into the global datapipe component
registry so components can be referenced via ``${dp:...}`` in Hydra
YAML configs.

Import this module before Hydra instantiation to register the components.
"""

from __future__ import annotations

from warnings import warn

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.geometry import compute_cell_areas


@register()
class ComputeFreestreamDirection(MeshTransform):
    r"""Write the unit freestream direction into ``global_data``.

    Stores ``global_data[output_field] = U / |U|`` computed from
    ``global_data[velocity_field]``.  The physical vector is left in place
    so force integration and inference-side re-dimensionalization keep
    reading it.

    Place this before ``CenterMesh`` so the rotation augmentation (which is
    inserted after ``CenterMesh`` and rotates ``global_data`` vectors)
    rotates the direction together with the geometry.

    On a ``DomainMesh`` the direction is read from and written to the
    domain-level ``global_data``.

    Parameters
    ----------
    velocity_field : str
        ``global_data`` key of the freestream velocity vector.
    output_field : str
        ``global_data`` key to write the unit direction to.
    """

    def __init__(
        self,
        velocity_field: str = "U_inf",
        output_field: str = "U_inf_dir",
    ) -> None:
        super().__init__()
        self.velocity_field = velocity_field
        self.output_field = output_field

    def _with_direction(self, global_data: TensorDict) -> TensorDict:
        if self.velocity_field not in global_data.keys():
            raise KeyError(
                f"ComputeFreestreamDirection: {self.velocity_field!r} not found "
                f"in global_data (available: {sorted(global_data.keys())!r})."
            )
        velocity = global_data[self.velocity_field]
        norm = torch.linalg.vector_norm(velocity)
        if not torch.isfinite(norm) or norm <= 0.0:
            raise ValueError(
                f"ComputeFreestreamDirection: |{self.velocity_field}| must be "
                f"finite and positive, got {norm.item()!r}."
            )
        new_gd = global_data.clone()
        new_gd[self.output_field] = velocity / norm
        return new_gd

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh.with_data(global_data=self._with_direction(mesh.global_data))

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Write the direction into the domain-level ``global_data``."""
        return DomainMesh(
            interior=domain.interior,
            boundaries=domain.boundaries,
            global_data=self._with_direction(domain.global_data),
        )

    def extra_repr(self) -> str:
        return f"{self.output_field} = {self.velocity_field} / |{self.velocity_field}|"


@register()
class DropDegenerateCells(MeshTransform):
    r"""Drop cells with non-finite or degenerate current geometry.

    For 3D surface triangles, use a direct cross product in float64 to
    avoid cancellation in the Gram area formula. Other simplex dimensions
    use their geometric measure recomputed in float64. Cached areas are
    ignored: this check concerns the coordinates after centering, rotation,
    and scaling, which can collapse a face through rounding.

    Place this last in the transform chain so it sees the same coordinates
    the model will. Meshes without rejected cells pass through unchanged.
    Only cells and their associated data are sliced; vertices are retained.
    """

    def __call__(self, mesh: Mesh) -> Mesh:
        if mesh.n_cells == 0:
            return mesh
        cell_points = mesh.points[mesh.cells].to(torch.float64)
        edges = cell_points[:, 1:] - cell_points[:, :1]
        finite_points = torch.isfinite(cell_points).all(dim=(-2, -1))
        if mesh.n_manifold_dims == 2 and mesh.n_spatial_dims == 3:
            area_vectors = torch.linalg.cross(edges[:, 0], edges[:, 1])
            # A nonzero component suffices: squaring a tiny area vector to
            # compute its norm could underflow and discard a valid face.
            keep = (
                finite_points
                & torch.isfinite(area_vectors).all(dim=-1)
                & (area_vectors != 0).any(dim=-1)
            )
        else:
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
