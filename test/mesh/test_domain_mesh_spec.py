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

"""Tests for parametric DomainMesh type specifications (DomainMesh[m, s] syntax)."""

import pytest
import torch

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh._mesh_spec import MeshDims

### Fixtures


@pytest.fixture
def tet_domain():
    """A DomainMesh with tet interior (m=3, s=3) and tri boundaries (m=2, s=3)."""
    interior = Mesh(
        points=torch.randn(20, 3),
        cells=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
    )
    wall = Mesh(
        points=torch.randn(10, 3),
        cells=torch.tensor([[0, 1, 2], [3, 4, 5]]),
    )
    inlet = Mesh(
        points=torch.randn(6, 3),
        cells=torch.tensor([[0, 1, 2]]),
    )
    return DomainMesh(
        interior=interior,
        boundaries={"wall": wall, "inlet": inlet},
    )


@pytest.fixture
def surface_domain():
    """A DomainMesh with tri interior (m=2, s=3) and edge boundaries (m=1, s=3)."""
    interior = Mesh(
        points=torch.randn(10, 3),
        cells=torch.tensor([[0, 1, 2], [3, 4, 5]]),
    )
    boundary = Mesh(
        points=torch.randn(4, 3),
        cells=torch.tensor([[0, 1], [2, 3]]),
    )
    return DomainMesh(
        interior=interior,
        boundaries={"outer": boundary},
    )


@pytest.fixture
def point_cloud_domain():
    """A DomainMesh with point cloud interior (m=0, s=3) and no boundaries."""
    return DomainMesh(interior=Mesh(points=torch.randn(50, 3)))


@pytest.fixture
def point_cloud_with_boundaries():
    """A DomainMesh with point cloud interior and triangle boundaries.

    This is a valid configuration: a volumetric point cloud with boundary
    surface patches (manifold dim check is skipped when interior has no cells).
    """
    interior = Mesh(points=torch.randn(100, 3))
    wall = Mesh(
        points=torch.randn(6, 3),
        cells=torch.tensor([[0, 1, 2], [3, 4, 5]]),
    )
    return DomainMesh(
        interior=interior,
        boundaries={"wall": wall},
    )


### __class_getitem__ syntax


class TestClassGetitemSyntax:
    """Tests for DomainMesh[m, s] subscript syntax and validation."""

    def test_concrete_both(self):
        spec = DomainMesh[3, 3]
        assert repr(spec) == "DomainMesh[3, 3]"

    def test_concrete_zero_manifold(self):
        spec = DomainMesh[0, 3]
        assert repr(spec) == "DomainMesh[0, 3]"

    def test_concrete_equal_dims(self):
        spec = DomainMesh[2, 2]
        assert repr(spec) == "DomainMesh[2, 2]"

    def test_ellipsis_spatial(self):
        spec = DomainMesh[2, ...]
        assert repr(spec) == "DomainMesh[2, ...]"

    def test_ellipsis_manifold(self):
        spec = DomainMesh[..., 3]
        assert repr(spec) == "DomainMesh[..., 3]"

    def test_ellipsis_both(self):
        spec = DomainMesh[..., ...]
        assert repr(spec) == "DomainMesh[..., ...]"

    def test_symbolic(self):
        spec = DomainMesh["n-1", "n"]
        assert repr(spec) == "DomainMesh['n-1', 'n']"

    def test_single_param_raises(self):
        with pytest.raises(TypeError, match="requires exactly 2 parameters"):
            DomainMesh[3]

    def test_three_params_raises(self):
        with pytest.raises(TypeError, match="requires exactly 2 parameters"):
            DomainMesh[1, 2, 3]

    def test_manifold_exceeds_spatial_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            DomainMesh[4, 3]

    def test_negative_manifold_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            DomainMesh[-1, 3]

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="int, str, or None"):
            DomainMesh[2.5, 3]


### Caching and identity


class TestCaching:
    """Tests that parametrized DomainMesh types are cached and share identity."""

    def test_same_concrete_is_identical(self):
        assert DomainMesh[3, 3] is DomainMesh[3, 3]

    def test_same_partial_is_identical(self):
        assert DomainMesh[2, ...] is DomainMesh[2, ...]
        assert DomainMesh[..., 3] is DomainMesh[..., 3]

    def test_same_symbolic_is_identical(self):
        assert DomainMesh["n-1", "n"] is DomainMesh["n-1", "n"]

    def test_different_specs_are_distinct(self):
        assert DomainMesh[3, 3] is not DomainMesh[2, 3]
        assert DomainMesh[3, 3] is not DomainMesh[3, ...]

    def test_domain_mesh_spec_distinct_from_mesh_spec(self):
        """DomainMesh[3, 3] and Mesh[3, 3] are different types."""
        assert DomainMesh[3, 3] is not Mesh[3, 3]


### isinstance checks


class TestIsinstance:
    """Tests for isinstance(dm, DomainMesh[m, s]) runtime dimension checks."""

    def test_concrete_match_tet(self, tet_domain):
        assert isinstance(tet_domain, DomainMesh[3, 3])

    def test_concrete_match_surface(self, surface_domain):
        assert isinstance(surface_domain, DomainMesh[2, 3])

    def test_concrete_match_point_cloud(self, point_cloud_domain):
        assert isinstance(point_cloud_domain, DomainMesh[0, 3])

    def test_concrete_mismatch_manifold(self, tet_domain):
        assert not isinstance(tet_domain, DomainMesh[2, 3])

    def test_concrete_mismatch_spatial(self, tet_domain):
        assert not isinstance(tet_domain, DomainMesh[3, 4])

    def test_partial_manifold_match(self, tet_domain):
        assert isinstance(tet_domain, DomainMesh[3, ...])

    def test_partial_manifold_mismatch(self, tet_domain):
        assert not isinstance(tet_domain, DomainMesh[2, ...])

    def test_partial_spatial_match(self, tet_domain):
        assert isinstance(tet_domain, DomainMesh[..., 3])

    def test_partial_spatial_mismatch(self, tet_domain):
        assert not isinstance(tet_domain, DomainMesh[..., 2])

    def test_unconstrained_always_matches(
        self, tet_domain, surface_domain, point_cloud_domain
    ):
        assert isinstance(tet_domain, DomainMesh[..., ...])
        assert isinstance(surface_domain, DomainMesh[..., ...])
        assert isinstance(point_cloud_domain, DomainMesh[..., ...])

    def test_non_domain_mesh_never_matches(self):
        assert not isinstance("not a domain mesh", DomainMesh[3, 3])
        assert not isinstance(42, DomainMesh[..., ...])

    def test_mesh_is_not_domain_mesh(self):
        """A bare Mesh should not match DomainMesh[m, s]."""
        mesh = Mesh(points=torch.randn(10, 3))
        assert not isinstance(mesh, DomainMesh[0, 3])

    def test_point_cloud_with_boundaries_matches_point_cloud_spec(
        self, point_cloud_with_boundaries
    ):
        """Point cloud domain with triangle boundaries matches DomainMesh[0, 3].

        Boundary manifold dims are not checked when interior m=0 because
        MeshDims(0, 3).boundary raises ValueError.
        """
        assert isinstance(point_cloud_with_boundaries, DomainMesh[0, 3])

    def test_wrong_boundary_dims_rejects(self):
        """DomainMesh with correct interior but wrong boundary dims should not match.

        Note: this construction itself raises ValueError with the new validation,
        so we must bypass __post_init__ to test isinstance in isolation. We test
        the validation error separately in TestManifoldDimValidation.
        """
        # Tet interior with edge boundaries (should be tri, not edge)
        interior = Mesh(
            points=torch.randn(10, 3),
            cells=torch.tensor([[0, 1, 2, 3]]),
        )
        edges = Mesh(
            points=torch.randn(4, 3),
            cells=torch.tensor([[0, 1], [2, 3]]),
        )
        # Construction now raises - tested in TestManifoldDimValidation
        with pytest.raises(ValueError, match="n_manifold_dims"):
            DomainMesh(interior=interior, boundaries={"wrong": edges})


### Derived type properties


class TestDerivedTypes:
    """Tests for .interior_type and .boundary_type navigation properties."""

    def test_interior_type_concrete(self):
        assert DomainMesh[3, 3].interior_type is Mesh[3, 3]

    def test_interior_type_surface(self):
        assert DomainMesh[2, 3].interior_type is Mesh[2, 3]

    def test_interior_type_partial(self):
        assert DomainMesh[2, ...].interior_type is Mesh[2, ...]

    def test_boundary_type_concrete(self):
        assert DomainMesh[3, 3].boundary_type is Mesh[2, 3]

    def test_boundary_type_surface(self):
        assert DomainMesh[2, 3].boundary_type is Mesh[1, 3]

    def test_boundary_type_chain(self):
        """DomainMesh[3, 3].boundary_type.boundary gives Mesh[1, 3]."""
        assert DomainMesh[3, 3].boundary_type.boundary is Mesh[1, 3]

    def test_boundary_type_partial_spatial(self):
        assert DomainMesh[2, ...].boundary_type is Mesh[1, ...]

    def test_boundary_type_zero_manifold_raises(self):
        with pytest.raises(ValueError, match="0-dimensional"):
            DomainMesh[0, 3].boundary_type

    def test_boundary_type_unconstrained_manifold_raises(self):
        with pytest.raises(TypeError, match="unconstrained"):
            DomainMesh[..., 3].boundary_type

    def test_symbolic_interior_type(self):
        spec = DomainMesh["n", "n+1"]
        assert spec.interior_type._mesh_dims == MeshDims("n", "n+1")

    def test_symbolic_boundary_type(self):
        spec = DomainMesh["n", "n+1"]
        assert spec.boundary_type._mesh_dims == MeshDims("n-1", "n+1")


### Manifold dimension validation


class TestManifoldDimValidation:
    """Tests for manifold dimension validation in __post_init__."""

    def test_tet_interior_with_tri_boundaries_ok(self):
        """Correct: tet interior (m=3), tri boundaries (m=2)."""
        interior = Mesh(
            points=torch.randn(10, 3),
            cells=torch.tensor([[0, 1, 2, 3]]),
        )
        wall = Mesh(
            points=torch.randn(6, 3),
            cells=torch.tensor([[0, 1, 2]]),
        )
        dm = DomainMesh(interior=interior, boundaries={"wall": wall})
        assert dm.n_boundaries == 1

    def test_tri_interior_with_edge_boundaries_ok(self):
        """Correct: tri interior (m=2), edge boundaries (m=1)."""
        interior = Mesh(
            points=torch.randn(10, 3),
            cells=torch.tensor([[0, 1, 2]]),
        )
        edge = Mesh(
            points=torch.randn(4, 3),
            cells=torch.tensor([[0, 1], [2, 3]]),
        )
        dm = DomainMesh(interior=interior, boundaries={"edge": edge})
        assert dm.n_boundaries == 1

    def test_tet_interior_with_tet_boundaries_raises(self):
        """Wrong: tet interior (m=3) with tet boundaries (m=3). Should be m=2."""
        interior = Mesh(
            points=torch.randn(10, 3),
            cells=torch.tensor([[0, 1, 2, 3]]),
        )
        bad_boundary = Mesh(
            points=torch.randn(8, 3),
            cells=torch.tensor([[0, 1, 2, 3]]),
        )
        with pytest.raises(ValueError, match="n_manifold_dims=2"):
            DomainMesh(interior=interior, boundaries={"bad": bad_boundary})

    def test_tri_interior_with_tri_boundaries_raises(self):
        """Wrong: tri interior (m=2) with tri boundaries (m=2). Should be m=1."""
        interior = Mesh(
            points=torch.randn(10, 3),
            cells=torch.tensor([[0, 1, 2]]),
        )
        bad_boundary = Mesh(
            points=torch.randn(6, 3),
            cells=torch.tensor([[0, 1, 2]]),
        )
        with pytest.raises(ValueError, match="n_manifold_dims=1"):
            DomainMesh(interior=interior, boundaries={"bad": bad_boundary})

    def test_tet_interior_with_edge_boundaries_raises(self):
        """Wrong: tet interior (m=3) with edge boundaries (m=1). Should be m=2."""
        interior = Mesh(
            points=torch.randn(10, 3),
            cells=torch.tensor([[0, 1, 2, 3]]),
        )
        edges = Mesh(
            points=torch.randn(4, 3),
            cells=torch.tensor([[0, 1], [2, 3]]),
        )
        with pytest.raises(ValueError, match="n_manifold_dims=2"):
            DomainMesh(interior=interior, boundaries={"bad": edges})

    def test_tet_interior_with_point_cloud_boundaries_raises(self):
        """Wrong: tet interior (m=3) with point cloud boundaries (m=0). Should be m=2."""
        interior = Mesh(
            points=torch.randn(10, 3),
            cells=torch.tensor([[0, 1, 2, 3]]),
        )
        points = Mesh(points=torch.randn(5, 3))
        with pytest.raises(ValueError, match="n_manifold_dims=2"):
            DomainMesh(interior=interior, boundaries={"bad": points})

    def test_point_cloud_interior_with_any_boundaries_ok(self):
        """Point cloud interior (m=0): manifold dim check is skipped."""
        interior = Mesh(points=torch.randn(50, 3))
        wall = Mesh(
            points=torch.randn(6, 3),
            cells=torch.tensor([[0, 1, 2]]),
        )
        dm = DomainMesh(interior=interior, boundaries={"wall": wall})
        assert dm.n_boundaries == 1

    def test_point_cloud_interior_no_boundaries_ok(self):
        """Point cloud interior with no boundaries is always valid."""
        dm = DomainMesh(interior=Mesh(points=torch.randn(50, 3)))
        assert dm.n_boundaries == 0

    def test_error_message_includes_boundary_name(self):
        """The error message should identify which boundary has wrong dims."""
        interior = Mesh(
            points=torch.randn(10, 3),
            cells=torch.tensor([[0, 1, 2, 3]]),
        )
        bad = Mesh(
            points=torch.randn(8, 3),
            cells=torch.tensor([[0, 1, 2, 3]]),
        )
        with pytest.raises(ValueError, match="'bad_bc'"):
            DomainMesh(interior=interior, boundaries={"bad_bc": bad})
