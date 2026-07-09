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

"""Unit tests for `src/sdf.py` (ComputeSDFFromBoundary).

The critical contract is the near-wall normals: boundary-layer volume points
sit at sub-micron wall distances, where the naive normalized
``query - closest_point`` direction is float32 rounding noise. The transform
must substitute the oriented hit-face normal there -- a previous
centroid-direction fallback produced normals *tangent* to the surface for
1-3% of DrivAerML training points, concentrated exactly in the boundary
layer.
"""

from __future__ import annotations

import torch
from sdf import ComputeSDFFromBoundary

from physicsnemo.mesh import DomainMesh, Mesh

# A closed, elongated box: x in (0, 10), y in (-0.5, 0.5), z in (0, 1),
# 12 outward-wound triangles. Near the far end of the top face the direction
# away from the box centroid is nearly tangent to the face (z-component
# ~0.1), so the old centroid fallback is maximally wrong there while the
# winding sign stays well-defined (closed surface), mirroring a car fender.
_BOX_VERTICES = torch.tensor(
    [
        [0.0, -0.5, 0.0],
        [10.0, -0.5, 0.0],
        [10.0, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, -0.5, 1.0],
        [10.0, -0.5, 1.0],
        [10.0, 0.5, 1.0],
        [0.0, 0.5, 1.0],
    ],
    dtype=torch.float32,
)
_BOX_FACES = torch.tensor(
    [
        [0, 2, 1],
        [0, 3, 2],  # bottom, -z
        [4, 5, 6],
        [4, 6, 7],  # top, +z
        [0, 1, 5],
        [0, 5, 4],  # -y side
        [3, 7, 6],
        [3, 6, 2],  # +y side
        [0, 4, 7],
        [0, 7, 3],  # -x side
        [1, 2, 6],
        [1, 6, 5],  # +x side
    ],
    dtype=torch.int64,
)


def _domain_with_interior(interior_points: torch.Tensor) -> DomainMesh:
    """Wrap query points and the box surface into a DomainMesh."""
    return DomainMesh(
        interior=Mesh(points=interior_points),
        boundaries={"stl_geometry": Mesh(points=_BOX_VERTICES, cells=_BOX_FACES)},
        global_data={},
    )


def test_sdf_normals_near_wall_use_face_normal():
    """Near-wall normals equal the oriented face normal, not tangent junk."""
    torch.manual_seed(0)
    n_query = 500
    # Points hovering over the far end of the top face, at log-distributed
    # wall distances from 1e-8 (deep boundary layer, below any
    # float32-meaningful separation) to 1e-3.
    q_xy = torch.rand(n_query, 2) * torch.tensor([0.9, 0.8]) + torch.tensor([9.0, -0.4])
    wall_dist = 10 ** (torch.rand(n_query) * 5 - 8)
    interior = torch.stack([q_xy[:, 0], q_xy[:, 1], 1.0 + wall_dist], -1).float()
    domain = _domain_with_interior(interior)

    transform = ComputeSDFFromBoundary(
        boundary_name="stl_geometry",
        sdf_field="sdf",
        normals_field="sdf_normals",
        use_winding_number=True,
    )
    result = transform.apply_to_domain(domain)

    sdf = result.interior.point_data["sdf"]
    normals = result.interior.point_data["sdf_normals"]
    assert sdf.shape == (n_query, 1)
    assert normals.shape == (n_query, 3)
    # All points are outside (or, for the sub-float32-resolution wall
    # distances, exactly on) the closed box.
    assert torch.all(sdf >= 0)

    # Unit vectors, and the normal must be +z at ALL wall distances. The old
    # centroid fallback returned the direction away from the box centroid
    # (z-component ~0.1, i.e. nearly tangent) for points below 1e-6.
    torch.testing.assert_close(
        normals.norm(dim=-1), torch.ones(n_query), atol=1e-4, rtol=0.0
    )
    assert torch.all(normals[:, 2] > 0.99)


def test_sdf_normals_far_points_keep_closest_point_direction():
    """Far from the surface the normals stay the closest-point direction."""
    # Off the +x end (closest point on the box rim/edge) and above the middle
    # (closest point on the top face).
    off = torch.tensor([[12.0, 0.0, 2.0], [5.0, 0.0, 3.0]], dtype=torch.float32)
    domain = _domain_with_interior(off)

    result = ComputeSDFFromBoundary().apply_to_domain(domain)
    normals = result.interior.point_data["sdf_normals"]

    closest = torch.tensor([[10.0, 0.0, 1.0], [5.0, 0.0, 1.0]])
    expected = torch.nn.functional.normalize(off - closest, dim=-1)
    torch.testing.assert_close(normals, expected, atol=1e-5, rtol=1e-5)


def test_sdf_normals_degenerate_hit_face_stays_finite():
    """A near-wall query hitting a degenerate face must not get a junk normal.

    A zero-area face has a zero ``cell_normals`` entry, so the face-normal
    substitution must be skipped for it (keeping the raw closest-point
    direction) rather than writing zero/NaN normals into the training data.
    """
    # The box plus a degenerate "antenna" face: a repeated-vertex face
    # spanning the segment (10, 0, 1) -> (11, 0, 1), sticking off the box so
    # it is genuinely the nearest surface for queries above it.
    vertices = torch.cat(
        [_BOX_VERTICES, torch.tensor([[10.0, 0.0, 1.0], [11.0, 0.0, 1.0]])]
    )
    faces = torch.cat([_BOX_FACES, torch.tensor([[8, 9, 9]])])
    # Near-wall above the antenna midpoint, plus a regular top-face point.
    interior = torch.tensor(
        [[10.5, 0.0, 1.0 + 1e-7], [5.0, 0.0, 1.0 + 1e-7]], dtype=torch.float32
    )
    domain = DomainMesh(
        interior=Mesh(points=interior),
        boundaries={"stl_geometry": Mesh(points=vertices, cells=faces)},
        global_data={},
    )

    result = ComputeSDFFromBoundary().apply_to_domain(domain)
    normals = result.interior.point_data["sdf_normals"]

    assert torch.isfinite(normals).all()
    # The regular near-wall point still gets the exact face normal.
    torch.testing.assert_close(
        normals[1], torch.tensor([0.0, 0.0, 1.0]), atol=1e-4, rtol=0.0
    )
