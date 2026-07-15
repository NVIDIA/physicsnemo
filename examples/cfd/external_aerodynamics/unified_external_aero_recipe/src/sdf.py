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
SDF (Signed Distance Field) pipeline transforms for volume meshes.

Provides a transform that computes SDF + normals from a boundary surface
onto interior volume points, and a cleanup transform to drop temporary
boundaries before TensorDict conversion.

These work with ``DomainMeshReader``'s ``extra_boundaries`` parameter,
which loads a sibling STL mesh at full resolution alongside the domain
mesh.  The SDF transform reads the injected boundary, computes the
signed distance field, and writes results into ``interior.point_data``.

Recipe-local module registered into the global datapipe component
registry so components can be referenced via ``${dp:...}`` in Hydra
YAML configs.

Import this module before Hydra instantiation to register the components.
"""

from __future__ import annotations

import torch

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.spatial.sdf import signed_distance_field


@register()
class ComputeSDFFromBoundary(MeshTransform):
    r"""Compute SDF and optionally normals from a boundary surface onto interior points.

    Reads the surface mesh from ``domain.boundaries[boundary_name]`` and
    evaluates the signed distance field at every interior point using
    :func:`physicsnemo.mesh.spatial.sdf.signed_distance_field`,
    a mesh-native, pure-PyTorch implementation backed by a torch BVH.

    The computed SDF is stored as a scalar field ``(N, 1)`` in
    ``interior.point_data[sdf_field]``.  If ``normals_field`` is set,
    approximate surface normals ``(N, 3)`` are also stored, computed as
    the normalized direction from each query point to its closest point
    on the surface.  Points essentially *on* the surface (boundary-layer
    points at sub-micron wall distances) instead use the oriented normal
    of the hit face, since at those distances the closest-point direction
    is float32 rounding noise.

    Parameters
    ----------
    boundary_name : str
        Key of the boundary mesh to use as the SDF surface.
    sdf_field : str
        Name for the SDF field in ``interior.point_data``.
    normals_field : str or None
        Optional name for the normals field.  ``None`` to skip.
    use_winding_number : bool
        Whether to use winding-number sign computation.  Required for
        non-watertight meshes; slightly slower.
    """

    def __init__(
        self,
        boundary_name: str = "stl_geometry",
        sdf_field: str = "sdf",
        normals_field: str | None = "sdf_normals",
        *,
        use_winding_number: bool = True,
    ) -> None:
        super().__init__()
        self.boundary_name = boundary_name
        self.sdf_field = sdf_field
        self.normals_field = normals_field
        self.use_winding_number = use_winding_number

    def __call__(self, mesh: Mesh) -> Mesh:
        # Single-mesh path is not meaningful for SDF (we need a separate
        # surface mesh).  Pass through unchanged.
        return mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Compute SDF from the boundary surface onto interior points.

        Parameters
        ----------
        domain : DomainMesh
            Must contain a boundary named ``self.boundary_name`` with
            triangle cells.

        Returns
        -------
        DomainMesh
            Domain with SDF (and optionally normals) injected into
            ``interior.point_data``.
        """
        if self.boundary_name not in domain.boundaries:
            raise KeyError(
                f"Boundary {self.boundary_name!r} not found. "
                f"Available: {domain.boundary_names}"
            )

        surface = domain.boundaries[self.boundary_name]

        if surface.n_cells == 0:
            raise ValueError(
                f"Boundary {self.boundary_name!r} has no cell connectivity "
                f"(required for SDF computation)"
            )

        query_points = domain.interior.points.float()

        sdf_values, closest_points, hit_faces = signed_distance_field(
            surface,
            query_points,
            use_sign_winding_number=self.use_winding_number,
        )

        # Build updated point_data with SDF (N, 1)
        new_pd = domain.interior.point_data.clone()
        new_pd[self.sdf_field] = sdf_values.unsqueeze(-1)

        # Optionally compute approximate normals from closest-point direction
        if self.normals_field is not None:
            normals = query_points - closest_points

            # For points essentially on the surface -- boundary-layer points
            # at sub-micron wall distances -- (query - closest) is float32
            # rounding noise (or exactly zero) and its direction is
            # meaningless. Substitute the oriented normal of the hit face:
            # the exact limit of the closest-point direction at the wall.
            # Sign-align with the SDF so the rare interior point keeps
            # pointing into the body like its neighbors (the SDF treats
            # on-surface as outside, so dist == 0 gets the outward normal).
            # The band is scale-aware: the closest point carries rounding
            # noise ~ eps * |coordinate|, so an absolute cutoff under-covers
            # geometry far from the origin. 128 eps (~1.5e-5 per unit
            # coordinate) clears that noise floor with a wide margin, and
            # widening the band is free because the substitute is exact.
            # Computed unconditionally and selected with a mask rather than
            # branching on ``near_surface.any()`` -- that host readback would
            # stall the prefetch stream.
            dist = torch.norm(normals, dim=-1)
            coord_scale = query_points.abs().amax(dim=-1).clamp(min=1.0)
            near_surface = dist < (128.0 * torch.finfo(torch.float32).eps * coord_scale)
            face_normals = surface.cell_normals.to(query_points.dtype)[hit_faces]
            # A degenerate (zero-area) hit face has no meaningful normal --
            # ``cell_normals`` returns a zero vector for it. Keep the raw
            # closest-point direction there instead of substituting zeros.
            face_normal_ok = (face_normals * face_normals).sum(-1) > 0.5
            oriented = torch.where(
                (sdf_values >= 0).unsqueeze(-1), face_normals, -face_normals
            )
            normals = torch.where(
                (near_surface & face_normal_ok).unsqueeze(-1), oriented, normals
            )

            # Normalize to unit vectors
            norm = torch.norm(normals, dim=-1, keepdim=True).clamp(min=1e-8)
            normals = normals / norm
            new_pd[self.normals_field] = normals

        new_interior = Mesh(
            points=domain.interior.points,
            cells=domain.interior.cells,
            point_data=new_pd,
            cell_data=domain.interior.cell_data,
            global_data=domain.interior.global_data,
        )

        return DomainMesh(
            interior=new_interior,
            boundaries=domain.boundaries,
            global_data=domain.global_data,
        )

    def extra_repr(self) -> str:
        parts = [
            f"boundary={self.boundary_name!r}",
            f"sdf_field={self.sdf_field!r}",
        ]
        if self.normals_field:
            parts.append(f"normals_field={self.normals_field!r}")
        parts.append(f"winding_number={self.use_winding_number}")
        return ", ".join(parts)


@register()
class DropBoundary(MeshTransform):
    r"""Remove one or more boundaries from a :class:`DomainMesh`.

    Useful for stripping temporary data (e.g. a full-resolution STL
    boundary injected for SDF computation) before downstream transforms
    like ``MeshToTensorDict`` that would otherwise serialize the large
    surface into the output TensorDict.

    Parameters
    ----------
    names : list[str]
        Boundary names to remove.
    """

    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self.names = set(names)

    def __call__(self, mesh: Mesh) -> Mesh:
        # Single-mesh path: nothing to drop.
        return mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Remove the named boundaries from the domain.

        Parameters
        ----------
        domain : DomainMesh
            Input domain mesh.

        Returns
        -------
        DomainMesh
            Domain mesh without the dropped boundaries.
        """
        return DomainMesh(
            interior=domain.interior,
            boundaries=domain.boundaries.exclude(*self.names),
            global_data=domain.global_data,
        )

    def extra_repr(self) -> str:
        return f"names={sorted(self.names)}"
