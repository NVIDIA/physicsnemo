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

"""Tests for the MeshToDomainMesh transform."""

import pytest
import torch

from physicsnemo.datapipes.transforms.mesh import MeshToDomainMesh
from physicsnemo.mesh import DomainMesh, Mesh

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_triangle_mesh_3d() -> Mesh:
    """A 3-D surface mesh with 2 triangles, mixed cell_data and point_data.

    cell_data has both target-like fields ("C_p", "C_f") and feature-like
    fields ("normals"); point_data has one field ("vertex_label").
    """
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    return Mesh(
        points=points,
        cells=cells,
        cell_data={
            "C_p": torch.tensor([0.5, -0.3], dtype=torch.float32),
            "C_f": torch.tensor(
                [[0.01, 0.0, 0.0], [-0.005, 0.002, 0.0]],
                dtype=torch.float32,
            ),
            "normals": torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
        },
        point_data={
            "vertex_label": torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        },
        global_data={
            "U_inf": torch.tensor([30.0, 0.0, 0.0], dtype=torch.float32),
        },
    )


def _point_cloud_mesh_3d(n_points: int = 5) -> Mesh:
    """A 3-D point cloud (no cells) with one point_data field."""
    return Mesh(
        points=torch.randn(n_points, 3),
        point_data={
            "phi": torch.randn(n_points),
        },
    )


def _domain_mesh_3d() -> DomainMesh:
    """A minimal 3-D DomainMesh (interior point cloud + one boundary)."""
    interior = Mesh(
        points=torch.randn(10, 3),
        point_data={"target": torch.randn(10)},
    )
    boundary = Mesh(
        points=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        cells=torch.tensor([[0, 1, 2]], dtype=torch.int64),
    )
    return DomainMesh(interior=interior, boundaries={"wall": boundary})


# ---------------------------------------------------------------------------
# Construction / repr
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_args(self):
        t = MeshToDomainMesh()
        assert t._cell_data_targets == []
        assert t._point_data_targets == []
        assert t._interior_points == "cell_centroids"
        assert t._boundary_name == "vehicle"

    def test_invalid_interior_points_raises(self):
        with pytest.raises(ValueError, match="interior_points must be one of"):
            MeshToDomainMesh(interior_points="something_else")

    def test_repr_contains_args(self):
        t = MeshToDomainMesh(
            cell_data_targets=["C_p", "C_f"],
            interior_points="cell_centroids",
            boundary_name="vehicle",
        )
        r = repr(t)
        assert "cell_data_targets=['C_p', 'C_f']" in r
        assert "interior_points='cell_centroids'" in r
        assert "boundary_name='vehicle'" in r


# ---------------------------------------------------------------------------
# (cell_data_targets, interior_points='cell_centroids') corner
# ---------------------------------------------------------------------------


class TestCellCentroidsCorner:
    """Default surface-style conversion: predict at cell centroids."""

    def test_basic_shape_and_keys(self):
        mesh = _two_triangle_mesh_3d()
        transform = MeshToDomainMesh(
            cell_data_targets=["C_p", "C_f"],
            interior_points="cell_centroids",
            boundary_name="vehicle",
        )
        domain = transform(mesh)

        assert isinstance(domain, DomainMesh)
        ### Interior is a Mesh[0, 3] point cloud at the cell centroids.
        assert domain.interior.n_points == mesh.n_cells == 2
        assert domain.interior.n_cells == 0
        assert domain.interior.n_spatial_dims == 3
        ### Boundary keeps the original triangulated geometry.
        assert "vehicle" in domain.boundaries.keys()
        assert domain.boundaries["vehicle"].n_cells == 2

    def test_interior_points_are_cell_centroids(self):
        mesh = _two_triangle_mesh_3d()
        transform = MeshToDomainMesh(cell_data_targets=["C_p"])
        domain = transform(mesh)
        ### Cell centroid of triangle [0,1,2] = mean of the three points.
        expected = mesh.cell_centroids
        assert torch.allclose(domain.interior.points, expected)

    def test_targets_moved_to_interior_point_data(self):
        mesh = _two_triangle_mesh_3d()
        transform = MeshToDomainMesh(cell_data_targets=["C_p", "C_f"])
        domain = transform(mesh)
        interior_keys = set(domain.interior.point_data.keys())
        assert interior_keys == {"C_p", "C_f"}
        assert torch.allclose(domain.interior.point_data["C_p"], mesh.cell_data["C_p"])
        assert torch.allclose(domain.interior.point_data["C_f"], mesh.cell_data["C_f"])

    def test_non_target_cell_data_stays_on_boundary(self):
        mesh = _two_triangle_mesh_3d()
        transform = MeshToDomainMesh(cell_data_targets=["C_p", "C_f"])
        domain = transform(mesh)
        boundary = domain.boundaries["vehicle"]
        assert "normals" in boundary.cell_data.keys()
        assert torch.allclose(boundary.cell_data["normals"], mesh.cell_data["normals"])

    def test_targets_removed_from_boundary_cell_data(self):
        mesh = _two_triangle_mesh_3d()
        transform = MeshToDomainMesh(cell_data_targets=["C_p", "C_f"])
        domain = transform(mesh)
        boundary = domain.boundaries["vehicle"]
        assert "C_p" not in boundary.cell_data.keys()
        assert "C_f" not in boundary.cell_data.keys()

    def test_point_data_preserved_on_boundary(self):
        mesh = _two_triangle_mesh_3d()
        transform = MeshToDomainMesh(cell_data_targets=["C_p"])
        domain = transform(mesh)
        boundary = domain.boundaries["vehicle"]
        assert "vertex_label" in boundary.point_data.keys()
        assert torch.equal(
            boundary.point_data["vertex_label"], mesh.point_data["vertex_label"]
        )

    def test_global_data_passed_through(self):
        mesh = _two_triangle_mesh_3d()
        transform = MeshToDomainMesh(cell_data_targets=["C_p"])
        domain = transform(mesh)
        assert "U_inf" in domain.global_data.keys()
        assert torch.allclose(domain.global_data["U_inf"], mesh.global_data["U_inf"])

    def test_no_targets_yields_empty_interior_point_data(self):
        mesh = _two_triangle_mesh_3d()
        transform = MeshToDomainMesh(cell_data_targets=None)
        domain = transform(mesh)
        assert len(domain.interior.point_data.keys()) == 0
        ### All original cell_data should still be on the boundary.
        boundary_keys = set(domain.boundaries["vehicle"].cell_data.keys())
        assert boundary_keys == {"C_p", "C_f", "normals"}

    def test_custom_boundary_name(self):
        mesh = _two_triangle_mesh_3d()
        transform = MeshToDomainMesh(cell_data_targets=["C_p"], boundary_name="airfoil")
        domain = transform(mesh)
        assert "airfoil" in domain.boundaries.keys()
        assert "vehicle" not in domain.boundaries.keys()


# ---------------------------------------------------------------------------
# (point_data_targets, interior_points='vertices') corner
# ---------------------------------------------------------------------------


class TestVerticesCorner:
    """Predict at the input mesh's vertices, using point-centered targets."""

    def test_basic_shape_and_keys(self):
        mesh = _point_cloud_mesh_3d(n_points=5)
        transform = MeshToDomainMesh(
            point_data_targets=["phi"],
            interior_points="vertices",
            boundary_name="wall",
        )
        domain = transform(mesh)

        assert isinstance(domain, DomainMesh)
        assert domain.interior.n_points == mesh.n_points
        assert domain.interior.n_spatial_dims == 3
        assert torch.allclose(domain.interior.points, mesh.points)
        assert "phi" in domain.interior.point_data.keys()
        assert "phi" not in domain.boundaries["wall"].point_data.keys()

    def test_target_values_match(self):
        mesh = _point_cloud_mesh_3d(n_points=4)
        transform = MeshToDomainMesh(
            point_data_targets=["phi"], interior_points="vertices"
        )
        domain = transform(mesh)
        assert torch.allclose(domain.interior.point_data["phi"], mesh.point_data["phi"])


# ---------------------------------------------------------------------------
# Cross-corner: not implemented in v1
# ---------------------------------------------------------------------------


class TestCrossCornerRaises:
    def test_cell_data_targets_with_vertices_raises(self):
        mesh = _two_triangle_mesh_3d()
        transform = MeshToDomainMesh(
            cell_data_targets=["C_p"], interior_points="vertices"
        )
        with pytest.raises(NotImplementedError, match="cell_data_targets"):
            transform(mesh)

    def test_point_data_targets_with_cell_centroids_raises(self):
        mesh = _two_triangle_mesh_3d()
        transform = MeshToDomainMesh(
            point_data_targets=["vertex_label"], interior_points="cell_centroids"
        )
        with pytest.raises(NotImplementedError, match="point_data_targets"):
            transform(mesh)

    def test_point_cloud_with_cell_centroids_raises(self):
        mesh = _point_cloud_mesh_3d(n_points=10)
        transform = MeshToDomainMesh(
            cell_data_targets=["does_not_matter"],
            interior_points="cell_centroids",
        )
        with pytest.raises(NotImplementedError, match="n_cells"):
            transform(mesh)


# ---------------------------------------------------------------------------
# DomainMesh inputs (apply_to_domain): identity passthrough
# ---------------------------------------------------------------------------


class TestApplyToDomainIsIdentity:
    def test_returns_same_object(self):
        domain = _domain_mesh_3d()
        transform = MeshToDomainMesh(cell_data_targets=["irrelevant"])
        result = transform.apply_to_domain(domain)
        ### Identity passthrough -- same object, no copy.
        assert result is domain

    def test_does_not_mutate(self):
        domain = _domain_mesh_3d()
        transform = MeshToDomainMesh(cell_data_targets=["irrelevant"])
        original_interior_keys = set(domain.interior.point_data.keys())
        original_boundary_keys = set(domain.boundaries.keys())
        _ = transform.apply_to_domain(domain)
        assert set(domain.interior.point_data.keys()) == original_interior_keys
        assert set(domain.boundaries.keys()) == original_boundary_keys


# ---------------------------------------------------------------------------
# Pipeline composition / hydra registry
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_registered_in_hydra_resolver(self):
        ### Importing physicsnemo.datapipes registers the ${dp:...} resolver
        ### in OmegaConf; MeshToDomainMesh should be discoverable through it.
        from omegaconf import OmegaConf

        import physicsnemo.datapipes  # noqa: F401  -- side-effect import

        cfg = OmegaConf.create({"_target_": "${dp:MeshToDomainMesh}"})
        ### Resolving the interpolation should yield the fully-qualified path.
        target = cfg._target_
        assert target.endswith("MeshToDomainMesh")
