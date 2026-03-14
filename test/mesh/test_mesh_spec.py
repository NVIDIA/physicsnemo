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

"""Tests for parametric Mesh type specifications (Mesh[m, s] syntax)."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh._mesh_spec import MeshDims, _parse_dim_expr

### Fixtures


@pytest.fixture
def surface_3d():
    """A 2-manifold (triangle mesh) in 3D space."""
    return Mesh(
        points=torch.randn(10, 3),
        cells=torch.tensor([[0, 1, 2], [3, 4, 5]]),
    )


@pytest.fixture
def curve_3d():
    """A 1-manifold (edge mesh) in 3D space."""
    return Mesh(
        points=torch.randn(10, 3),
        cells=torch.tensor([[0, 1], [2, 3]]),
    )


@pytest.fixture
def point_cloud_3d():
    """A 0-manifold (point cloud) in 3D space."""
    return Mesh(points=torch.randn(10, 3))


@pytest.fixture
def surface_2d():
    """A 2-manifold (triangle mesh) in 2D space."""
    return Mesh(
        points=torch.randn(10, 2),
        cells=torch.tensor([[0, 1, 2]]),
    )


@pytest.fixture
def volume_3d():
    """A 3-manifold (tet mesh) in 3D space."""
    return Mesh(
        points=torch.randn(10, 3),
        cells=torch.tensor([[0, 1, 2, 3]]),
    )


### __class_getitem__ syntax


class TestClassGetitemSyntax:
    """Tests for Mesh[m, s] subscript syntax and validation."""

    def test_concrete_both(self):
        spec = Mesh[2, 3]
        assert repr(spec) == "Mesh[2, 3]"

    def test_concrete_zero_manifold(self):
        spec = Mesh[0, 3]
        assert repr(spec) == "Mesh[0, 3]"

    def test_concrete_equal_dims(self):
        spec = Mesh[3, 3]
        assert repr(spec) == "Mesh[3, 3]"

    def test_ellipsis_spatial(self):
        spec = Mesh[2, ...]
        assert repr(spec) == "Mesh[2, ...]"

    def test_ellipsis_manifold(self):
        spec = Mesh[..., 3]
        assert repr(spec) == "Mesh[..., 3]"

    def test_ellipsis_both(self):
        spec = Mesh[..., ...]
        assert repr(spec) == "Mesh[..., ...]"

    def test_symbolic(self):
        spec = Mesh["n-1", "n"]
        assert repr(spec) == "Mesh['n-1', 'n']"

    def test_symbolic_equal(self):
        spec = Mesh["n", "n"]
        assert repr(spec) == "Mesh['n', 'n']"

    def test_symbolic_plus(self):
        spec = Mesh["n", "n+1"]
        assert repr(spec) == "Mesh['n', 'n+1']"

    def test_single_param_raises(self):
        with pytest.raises(TypeError, match="requires exactly 2 parameters"):
            Mesh[2]

    def test_three_params_raises(self):
        with pytest.raises(TypeError, match="requires exactly 2 parameters"):
            Mesh[1, 2, 3]

    def test_manifold_exceeds_spatial_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            Mesh[4, 3]

    def test_negative_manifold_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            Mesh[-1, 3]

    def test_negative_spatial_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            Mesh[2, -1]

    def test_invalid_symbolic_raises(self):
        with pytest.raises(ValueError, match="Invalid symbolic"):
            Mesh["123bad", "n"]

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="int, str, or None"):
            Mesh[2.5, 3]

    def test_symbolic_manifold_with_ellipsis_raises(self):
        with pytest.raises(TypeError, match="requires a paired n_spatial_dims"):
            Mesh["n-1", ...]

    def test_symbolic_spatial_with_ellipsis_raises(self):
        with pytest.raises(TypeError, match="requires a paired n_manifold_dims"):
            Mesh[..., "n"]

    def test_symbolic_with_concrete_int_works(self):
        spec = Mesh["n", 3]
        assert repr(spec) == "Mesh['n', 3]"


### Caching and identity


class TestCaching:
    """Tests that parametrized types are cached and share identity."""

    def test_same_concrete_is_identical(self):
        assert Mesh[2, 3] is Mesh[2, 3]

    def test_same_partial_is_identical(self):
        assert Mesh[2, ...] is Mesh[2, ...]
        assert Mesh[..., 3] is Mesh[..., 3]

    def test_same_symbolic_is_identical(self):
        assert Mesh["n-1", "n"] is Mesh["n-1", "n"]

    def test_same_unconstrained_is_identical(self):
        assert Mesh[..., ...] is Mesh[..., ...]

    def test_different_specs_are_distinct(self):
        assert Mesh[2, 3] is not Mesh[1, 3]
        assert Mesh[2, 3] is not Mesh[2, ...]

    def test_equivalent_specs_are_identical(self):
        assert Mesh[0, ...] is Mesh[0, ...]
        assert Mesh[1, ...] is Mesh[1, ...]


### isinstance checks


class TestIsinstance:
    """Tests for isinstance(mesh, Mesh[m, s]) runtime dimension checks."""

    def test_concrete_match(self, surface_3d):
        assert isinstance(surface_3d, Mesh[2, 3])

    def test_concrete_mismatch_manifold(self, surface_3d):
        assert not isinstance(surface_3d, Mesh[1, 3])

    def test_concrete_mismatch_spatial(self, surface_3d):
        assert not isinstance(surface_3d, Mesh[2, 2])

    def test_partial_manifold_match(self, surface_3d):
        assert isinstance(surface_3d, Mesh[2, ...])

    def test_partial_manifold_mismatch(self, surface_3d):
        assert not isinstance(surface_3d, Mesh[1, ...])

    def test_partial_spatial_match(self, surface_3d):
        assert isinstance(surface_3d, Mesh[..., 3])

    def test_partial_spatial_mismatch(self, surface_3d):
        assert not isinstance(surface_3d, Mesh[..., 2])

    def test_unconstrained_always_matches(self, surface_3d, curve_3d, point_cloud_3d):
        assert isinstance(surface_3d, Mesh[..., ...])
        assert isinstance(curve_3d, Mesh[..., ...])
        assert isinstance(point_cloud_3d, Mesh[..., ...])

    def test_non_mesh_never_matches(self):
        assert not isinstance("not a mesh", Mesh[2, 3])
        assert not isinstance(42, Mesh[..., ...])

    def test_point_cloud_spec(self, point_cloud_3d, surface_3d):
        assert isinstance(point_cloud_3d, Mesh[0, ...])
        assert not isinstance(surface_3d, Mesh[0, ...])

    def test_graph_spec(self, curve_3d, surface_3d):
        assert isinstance(curve_3d, Mesh[1, ...])
        assert not isinstance(surface_3d, Mesh[1, ...])

    def test_same_manifold_different_spatial(self, surface_3d, surface_2d):
        assert isinstance(surface_3d, Mesh[2, 3])
        assert not isinstance(surface_2d, Mesh[2, 3])
        assert isinstance(surface_2d, Mesh[2, 2])
        assert isinstance(surface_3d, Mesh[2, ...])
        assert isinstance(surface_2d, Mesh[2, ...])

    def test_volume_mesh(self, volume_3d):
        assert isinstance(volume_3d, Mesh[3, 3])
        assert not isinstance(volume_3d, Mesh[2, 3])


### Symbolic constraint validation


class TestSymbolicConstraints:
    """Tests for symbolic codimension validation at isinstance time."""

    def test_codim_1_matches_surface_in_3d(self, surface_3d):
        """surface_3d has codimension 1 (2D manifold in 3D space)."""
        assert isinstance(surface_3d, Mesh["n-1", "n"])

    def test_codim_1_matches_curve_in_2d(self):
        """A 1D curve in 2D space also has codimension 1."""
        curve_2d = Mesh(
            points=torch.randn(5, 2),
            cells=torch.tensor([[0, 1], [1, 2]]),
        )
        assert isinstance(curve_2d, Mesh["n-1", "n"])

    def test_codim_0_matches_volume_in_3d(self, volume_3d):
        """volume_3d has codimension 0 (3D manifold in 3D space)."""
        assert isinstance(volume_3d, Mesh["n", "n"])

    def test_codim_0_rejects_surface_in_3d(self, surface_3d):
        """surface_3d has codimension 1, so doesn't match codim-0 spec."""
        assert not isinstance(surface_3d, Mesh["n", "n"])

    def test_codim_1_rejects_codim_2(self, curve_3d):
        """curve_3d has codimension 2 (1D in 3D), doesn't match codim-1."""
        assert not isinstance(curve_3d, Mesh["n-1", "n"])

    def test_different_variables_always_match(self, surface_3d):
        """Different variable names impose no constraint."""
        assert isinstance(surface_3d, Mesh["a", "b"])

    def test_codim_via_plus_syntax(self, surface_3d):
        """Mesh['n', 'n+1'] is equivalent to codimension 1."""
        assert isinstance(surface_3d, Mesh["n", "n+1"])

    def test_mixed_symbolic_concrete(self, surface_3d):
        """One symbolic, one concrete: only the concrete is checked."""
        assert isinstance(surface_3d, Mesh["n", 3])
        assert not isinstance(surface_3d, Mesh["n", 2])


### Boundary derived types


class TestBoundary:
    """Tests for Mesh[m, s].boundary derived type."""

    def test_concrete_boundary(self):
        assert Mesh[2, 3].boundary is Mesh[1, 3]

    def test_concrete_boundary_chain(self):
        assert Mesh[3, 3].boundary is Mesh[2, 3]
        assert Mesh[3, 3].boundary.boundary is Mesh[1, 3]

    def test_boundary_preserves_spatial(self):
        spec = Mesh[2, 3].boundary
        assert spec._mesh_dims.n_spatial_dims == 3

    def test_boundary_with_ellipsis_spatial(self):
        spec = Mesh[2, ...].boundary
        assert spec._mesh_dims == MeshDims(1, None)

    def test_boundary_of_zero_manifold_raises(self):
        with pytest.raises(ValueError, match="0-dimensional"):
            Mesh[0, 3].boundary

    def test_boundary_of_unconstrained_manifold_raises(self):
        with pytest.raises(TypeError, match="unconstrained"):
            Mesh[..., 3].boundary

    def test_symbolic_boundary(self):
        spec = Mesh["n", "n+1"].boundary
        assert spec._mesh_dims == MeshDims("n-1", "n+1")

    def test_symbolic_boundary_chain(self):
        spec = Mesh["n", "n"].boundary
        assert spec._mesh_dims == MeshDims("n-1", "n")


### MeshDims dataclass


class TestMeshDims:
    """Tests for the MeshDims frozen dataclass."""

    def test_frozen(self):
        dims = MeshDims(2, 3)
        with pytest.raises(AttributeError):
            dims.n_manifold_dims = 1  # type: ignore[misc]

    def test_hashable(self):
        d1 = MeshDims(2, 3)
        d2 = MeshDims(2, 3)
        assert hash(d1) == hash(d2)
        assert d1 == d2

    def test_different_not_equal(self):
        assert MeshDims(2, 3) != MeshDims(1, 3)
        assert MeshDims(2, 3) != MeshDims(2, None)

    def test_is_concrete(self):
        assert MeshDims(2, 3).is_concrete
        assert not MeshDims(2, None).is_concrete
        assert not MeshDims(None, 3).is_concrete
        assert not MeshDims("n", "n+1").is_concrete

    def test_str(self):
        assert str(MeshDims(2, 3)) == "2, 3"
        assert str(MeshDims(2, None)) == "2, ..."
        assert str(MeshDims(None, 3)) == "..., 3"
        assert str(MeshDims(None, None)) == "..., ..."
        assert str(MeshDims("n-1", "n")) == "'n-1', 'n'"


### _parse_dim_expr helper


class TestParseDimExpr:
    """Tests for the symbolic dimension expression parser."""

    def test_simple_variable(self):
        assert _parse_dim_expr("n") == ("n", 0)

    def test_variable_minus(self):
        assert _parse_dim_expr("n-1") == ("n", -1)

    def test_variable_plus(self):
        assert _parse_dim_expr("n+2") == ("n", 2)

    def test_underscore_variable(self):
        assert _parse_dim_expr("dim_x-3") == ("dim_x", -3)

    def test_multichar_variable(self):
        assert _parse_dim_expr("spatial") == ("spatial", 0)

    def test_whitespace_ignored(self):
        assert _parse_dim_expr("  n-1  ") == ("n", -1)
        assert _parse_dim_expr("n - 1") == ("n", -1)
        assert _parse_dim_expr("  n  -  1  ") == ("n", -1)
        assert _parse_dim_expr("n\t-\t1") == ("n", -1)
        assert _parse_dim_expr("dim - 3") == ("dim", -3)

    def test_invalid_starts_with_digit(self):
        with pytest.raises(ValueError, match="Invalid symbolic"):
            _parse_dim_expr("123")

    def test_invalid_operator(self):
        with pytest.raises(ValueError, match="Invalid symbolic"):
            _parse_dim_expr("n*2")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Invalid symbolic"):
            _parse_dim_expr("")
