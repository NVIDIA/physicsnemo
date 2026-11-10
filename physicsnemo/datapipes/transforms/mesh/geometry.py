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

"""Capture geometry independently of field sampling and compute boundary distances."""

from __future__ import annotations

import math

import torch

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.nn.functional.geometry import signed_distance_field


@register()
class CaptureGeometry(MeshTransform):
    """Retain a full geometry cloud before centering and query subsampling.

    Store cell centroids and their bounding-box midpoint in global fields
    ``geometry_points`` and ``geometry_center``. Surface meshes also retain
    aligned ``query_coordinates`` and ``query_normals`` in cell data. Domains
    retain interior ``query_coordinates`` in point data and use the named
    boundary for geometry. Domain boundaries are copied to detach disk mappings.

    Parameters
    ----------
    boundary_name : str, default="vehicle"
        Boundary supplying geometry for DomainMesh inputs.
    """

    def __init__(self, boundary_name: str = "vehicle") -> None:
        super().__init__()
        self.boundary_name = boundary_name

    def _capture(self, surface, global_data):
        centers = surface.cell_centroids.as_subclass(torch.Tensor)
        if centers.shape[0] == 0:
            raise ValueError("Geometry must contain cells")
        data = global_data.clone()
        data["geometry_points"] = centers
        data["geometry_center"] = (centers.amin(0) + centers.amax(0)) * 0.5
        return centers, data

    def __call__(self, mesh: Mesh) -> Mesh:
        """Preserve surface geometry and aligned physical query features."""
        centers, data = self._capture(mesh, mesh.global_data)
        fields = mesh.cell_data.clone()
        fields["query_coordinates"] = centers
        fields["query_normals"] = mesh.cell_normals.as_subclass(torch.Tensor)
        return mesh.with_data(cell_data=fields, global_data=data)

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Preserve full boundary geometry separately from interior queries."""
        _, data = self._capture(
            domain.boundaries[self.boundary_name], domain.global_data
        )
        fields = domain.interior.point_data.clone()
        fields["query_coordinates"] = domain.interior.points.as_subclass(torch.Tensor)
        return DomainMesh(
            interior=domain.interior.with_data(point_data=fields),
            boundaries={name: mesh.clone() for name, mesh in domain.boundaries.items()},
            global_data=data,
        )


@register()
class ComputeUnsignedDistance(MeshTransform):
    """Compute distance and closest-surface directions in the current length units.

    Parameters
    ----------
    boundary_name : str, default="vehicle"
        Triangular boundary used for nearest-surface queries.
    coordinate_scale : sequence of three positive floats or None, default=None
        Optional round trip through the frame captured by ``CaptureGeometry``.
        This reproduces distances evaluated from normalized model coordinates.
    direction_threshold : float, default=1e-6
        Directions are zero at or below this distance to avoid amplifying roundoff.

    Outputs are ``sdf`` (unsigned despite the conventional field name) and
    ``sdf_normals`` in the interior point data. Apply before length scaling
    when physical distances are required.
    """

    def __init__(
        self,
        boundary_name: str = "vehicle",
        coordinate_scale=None,
        direction_threshold: float = 1e-6,
    ) -> None:
        super().__init__()
        if coordinate_scale is not None and (
            len(coordinate_scale) != 3
            or any(not math.isfinite(x) or x <= 0 for x in coordinate_scale)
        ):
            raise ValueError(
                "coordinate_scale must contain three positive finite values"
            )
        if not math.isfinite(direction_threshold) or direction_threshold < 0:
            raise ValueError("direction_threshold must be finite and nonnegative")
        self.boundary_name = boundary_name
        self.coordinate_scale = (
            tuple(coordinate_scale) if coordinate_scale is not None else None
        )
        self.direction_threshold = direction_threshold

    def __call__(self, mesh: Mesh) -> Mesh:
        """Require a domain with a separate boundary surface."""
        raise TypeError("ComputeUnsignedDistance requires a DomainMesh")

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Evaluate unsigned distances without changing query coordinates."""
        surface = domain.boundaries[self.boundary_name]
        queries = domain.interior.points.as_subclass(torch.Tensor)
        if self.coordinate_scale is not None:
            center = domain.global_data["geometry_center"]
            scale = queries.new_tensor(self.coordinate_scale)
            queries = ((queries - center) / scale) * scale + center
        _, closest, _ = signed_distance_field(
            surface.points.as_subclass(torch.Tensor),
            surface.cells.as_subclass(torch.Tensor).flatten(),
            queries,
            use_sign_winding_number=False,
        )
        delta = queries - closest
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        fields = domain.interior.point_data.clone()
        fields["sdf"] = distance
        fields["sdf_normals"] = torch.where(
            distance > self.direction_threshold, delta / distance.clamp_min(1e-30), 0
        )
        return DomainMesh(
            interior=domain.interior.with_data(point_data=fields),
            boundaries=domain.boundaries,
            global_data=domain.global_data,
        )
