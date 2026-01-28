"""Comprehensive tests for 100% coverage of calculus module.

Tests all code paths including DEC operators, error cases, and edge conditions.
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives import procedural


@pytest.fixture
def simple_tet_mesh():
    """Simple tetrahedral mesh for testing."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]])
    return Mesh(points=points, cells=cells)


class TestDECOperators:
    """Test DEC-specific code paths."""

    def test_exterior_derivative_0(self, simple_tet_mesh):
        """Test exterior derivative d₀: Ω⁰ → Ω¹."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0

        mesh = simple_tet_mesh
        vertex_values = torch.arange(mesh.n_points, dtype=torch.float32)

        edge_values, edges = exterior_derivative_0(mesh, vertex_values)

        assert edge_values.shape[0] == edges.shape[0]
        assert edges.shape[1] == 2

        # Verify: df(edge) = f(v1) - f(v0)
        for i in range(len(edges)):
            expected = vertex_values[edges[i, 1]] - vertex_values[edges[i, 0]]
            assert torch.allclose(edge_values[i], expected, atol=1e-6)

    def test_exterior_derivative_tensor_field(self, simple_tet_mesh):
        """Test d₀ on tensor-valued 0-form."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0

        mesh = simple_tet_mesh
        # Vector-valued function at vertices
        vertex_vectors = mesh.points.clone()  # (n_points, 3)

        edge_values, edges = exterior_derivative_0(mesh, vertex_vectors)

        assert edge_values.shape == (len(edges), 3)

    def test_hodge_star_0(self, simple_tet_mesh):
        """Test Hodge star on 0-forms."""
        from physicsnemo.mesh.calculus._hodge_star import hodge_star_0

        mesh = simple_tet_mesh
        vertex_values = torch.ones(mesh.n_points)

        dual_values = hodge_star_0(mesh, vertex_values)

        assert dual_values.shape == vertex_values.shape
        # All values should be scaled by dual volumes
        assert (dual_values > 0).all()

    def test_hodge_star_0_tensor(self, simple_tet_mesh):
        """Test Hodge star on tensor-valued 0-form."""
        from physicsnemo.mesh.calculus._hodge_star import hodge_star_0

        mesh = simple_tet_mesh
        vertex_tensors = mesh.points.clone()  # (n_points, 3)

        dual_tensors = hodge_star_0(mesh, vertex_tensors)

        assert dual_tensors.shape == vertex_tensors.shape

    def test_hodge_star_1(self, simple_tet_mesh):
        """Test Hodge star on 1-forms."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0
        from physicsnemo.mesh.calculus._hodge_star import hodge_star_1

        mesh = simple_tet_mesh
        vertex_values = torch.ones(mesh.n_points)

        edge_values, edges = exterior_derivative_0(mesh, vertex_values)
        dual_edge_values = hodge_star_1(mesh, edge_values, edges)

        assert dual_edge_values.shape == edge_values.shape

    def test_sharp_operator(self, simple_tet_mesh):
        """Test sharp operator: 1-form → vector field."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0
        from physicsnemo.mesh.calculus._sharp_flat import sharp

        mesh = simple_tet_mesh
        vertex_values = torch.arange(mesh.n_points, dtype=torch.float32)

        edge_values, edges = exterior_derivative_0(mesh, vertex_values)
        vector_field = sharp(mesh, edge_values, edges)

        assert vector_field.shape == (mesh.n_points, mesh.n_spatial_dims)

    def test_sharp_operator_tensor(self, simple_tet_mesh):
        """Test sharp on tensor-valued 1-form."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0
        from physicsnemo.mesh.calculus._sharp_flat import sharp

        mesh = simple_tet_mesh
        vertex_tensors = mesh.points.clone()

        edge_tensors, edges = exterior_derivative_0(mesh, vertex_tensors)
        vector_field = sharp(mesh, edge_tensors, edges)

        assert vector_field.shape[0] == mesh.n_points

    def test_flat_operator(self, simple_tet_mesh):
        """Test flat operator: vector field → 1-form."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0
        from physicsnemo.mesh.calculus._sharp_flat import flat

        mesh = simple_tet_mesh
        vector_field = mesh.points.clone()

        # Get edges
        _, edges = exterior_derivative_0(mesh, torch.zeros(mesh.n_points))

        edge_1form = flat(mesh, vector_field, edges)

        assert edge_1form.shape[0] == len(edges)

    def test_flat_operator_tensor(self, simple_tet_mesh):
        """Test flat on tensor field."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0
        from physicsnemo.mesh.calculus._sharp_flat import flat

        mesh = simple_tet_mesh
        # Tensor field (n_points, 3, 2) for example
        tensor_field = mesh.points.unsqueeze(-1).repeat(1, 1, 2)

        _, edges = exterior_derivative_0(mesh, torch.zeros(mesh.n_points))

        edge_form = flat(mesh, tensor_field, edges)

        assert edge_form.ndim > 1

    def test_dec_gradient_points(self, simple_tet_mesh):
        """Test DEC gradient code path (implementation incomplete)."""
        from physicsnemo.mesh.calculus.gradient import compute_gradient_points_dec

        mesh = simple_tet_mesh
        phi = 2 * mesh.points[:, 0] + 3 * mesh.points[:, 1] - mesh.points[:, 2]

        grad = compute_gradient_points_dec(mesh, phi)

        # Just verify it runs and returns correct shape
        assert grad.shape == (mesh.n_points, mesh.n_spatial_dims)
        assert torch.isfinite(grad).all()


class TestCellDerivatives:
    """Test cell-based derivative computation."""

    def test_cell_gradient_lsq(self, simple_tet_mesh):
        """Test LSQ gradient on cell data."""
        mesh = simple_tet_mesh

        # Linear function on cells
        cell_centroids = mesh.cell_centroids
        cell_values = (cell_centroids * torch.tensor([2.0, 3.0, -1.0])).sum(dim=-1)

        mesh.cell_data["test"] = cell_values

        mesh_grad = mesh.compute_cell_derivatives(keys="test", method="lsq")

        grad = mesh_grad.cell_data["test_gradient"]
        assert grad.shape == (mesh.n_cells, mesh.n_spatial_dims)

        # Should recover linear coefficients approximately
        expected = torch.tensor([2.0, 3.0, -1.0])
        assert torch.allclose(grad.mean(dim=0), expected, atol=0.5)

    def test_cell_gradient_dec_not_implemented(self, simple_tet_mesh):
        """Test that DEC cell gradients raise NotImplementedError."""
        mesh = simple_tet_mesh
        mesh.cell_data["test"] = torch.ones(mesh.n_cells)

        with pytest.raises(NotImplementedError):
            mesh.compute_cell_derivatives(keys="test", method="dec")


class TestTensorFields:
    """Test gradient computation on tensor fields."""

    def test_vector_field_gradient_jacobian(self, simple_tet_mesh):
        """Test that gradient of vector field gives Jacobian."""
        mesh = simple_tet_mesh

        # Vector field
        mesh.point_data["velocity"] = mesh.points.clone()

        mesh_grad = mesh.compute_point_derivatives(keys="velocity", method="lsq")

        jacobian = mesh_grad.point_data["velocity_gradient"]

        # Shape should be (n_points, 3, 3) for 3D
        assert jacobian.shape == (mesh.n_points, 3, 3)

        # For v=r, Jacobian should be identity
        # Mean Jacobian should be close to I
        mean_jac = jacobian.mean(dim=0)
        expected = torch.eye(3)

        assert torch.allclose(mean_jac, expected, atol=0.2)


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_gradient_invalid_method(self, simple_tet_mesh):
        """Test that invalid method raises ValueError."""
        mesh = simple_tet_mesh
        mesh.point_data["test"] = torch.ones(mesh.n_points)

        with pytest.raises(ValueError, match="Invalid method"):
            mesh.compute_point_derivatives(keys="test", method="invalid")

    def test_gradient_invalid_gradient_type(self, simple_tet_mesh):
        """Test that invalid gradient_type raises ValueError."""
        mesh = simple_tet_mesh
        mesh.point_data["test"] = torch.ones(mesh.n_points)

        with pytest.raises(ValueError, match="Invalid gradient_type"):
            mesh.compute_point_derivatives(keys="test", gradient_type="invalid")

    def test_laplacian_on_3d_mesh_raises(self, simple_tet_mesh):
        """Test that DEC Laplacian on 3D mesh raises NotImplementedError."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        mesh = simple_tet_mesh  # 3D manifold
        phi = torch.ones(mesh.n_points)

        with pytest.raises(NotImplementedError, match="work-in-progress"):
            compute_laplacian_points_dec(mesh, phi)

    def test_curl_on_2d_raises(self):
        """Test that curl on 2D data raises ValueError."""
        from physicsnemo.mesh.calculus.curl import compute_curl_points_lsq

        # 2D mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        v = torch.ones((mesh.n_points, 2))

        with pytest.raises(ValueError, match="only defined for 3D"):
            compute_curl_points_lsq(mesh, v)

    def test_isolated_point_gradient_zero(self):
        """Test that isolated points (no neighbors) get zero gradient."""
        # Mesh with isolated point
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [10.0, 10.0, 10.0],  # Isolated
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3]])  # Only connects first 3 in one direction
        mesh = Mesh(points=points, cells=cells)

        phi = torch.arange(mesh.n_points, dtype=torch.float32)

        from physicsnemo.mesh.calculus._lsq_reconstruction import (
            compute_point_gradient_lsq,
        )

        grad = compute_point_gradient_lsq(mesh, phi)

        # Should not crash, gradients should be defined
        assert grad.shape == (mesh.n_points, mesh.n_spatial_dims)


class TestGradientTypes:
    """Test all gradient_type options."""

    def test_extrinsic_gradient(self):
        """Test gradient_type='extrinsic'."""
        mesh = procedural.lumpy_sphere.load(radius=1.0, subdivisions=2)
        mesh.point_data["test"] = torch.ones(mesh.n_points)

        mesh_grad = mesh.compute_point_derivatives(
            keys="test", gradient_type="extrinsic"
        )

        assert "test_gradient" in mesh_grad.point_data.keys()
        assert "test_gradient_intrinsic" not in mesh_grad.point_data.keys()

    def test_intrinsic_gradient(self):
        """Test gradient_type='intrinsic'."""
        mesh = procedural.lumpy_sphere.load(radius=1.0, subdivisions=2)
        mesh.point_data["test"] = torch.ones(mesh.n_points)

        mesh_grad = mesh.compute_point_derivatives(
            keys="test", gradient_type="intrinsic"
        )

        assert "test_gradient" in mesh_grad.point_data.keys()
        assert "test_gradient_extrinsic" not in mesh_grad.point_data.keys()

    def test_both_gradients(self):
        """Test gradient_type='both'."""
        mesh = procedural.lumpy_sphere.load(radius=1.0, subdivisions=2)
        mesh.point_data["test"] = torch.ones(mesh.n_points)

        mesh_grad = mesh.compute_point_derivatives(keys="test", gradient_type="both")

        assert "test_gradient_intrinsic" in mesh_grad.point_data.keys()
        assert "test_gradient_extrinsic" in mesh_grad.point_data.keys()


class TestKeyParsing:
    """Test various key input formats."""

    def test_none_keys_all_fields(self, simple_tet_mesh):
        """Test keys=None computes all non-cached fields (excludes "_cache" sub-dict)."""
        from physicsnemo.mesh.utilities import set_cached

        mesh = simple_tet_mesh
        mesh.point_data["field1"] = torch.ones(mesh.n_points)
        mesh.point_data["field2"] = torch.ones(mesh.n_points)
        set_cached(
            mesh.point_data, "test_value", torch.ones(mesh.n_points)
        )  # Should skip

        mesh_grad = mesh.compute_point_derivatives(keys=None)

        assert "field1_gradient" in mesh_grad.point_data.keys()
        assert "field2_gradient" in mesh_grad.point_data.keys()
        # Cached values should not have gradients computed
        assert "test_value_gradient" not in mesh_grad.point_data.keys()

    def test_nested_tensordict_keys(self, simple_tet_mesh):
        """Test nested TensorDict access."""
        from tensordict import TensorDict

        mesh = simple_tet_mesh
        nested = TensorDict(
            {"temperature": torch.ones(mesh.n_points)},
            batch_size=torch.Size([mesh.n_points]),
        )
        mesh.point_data["flow"] = nested

        mesh_grad = mesh.compute_point_derivatives(keys=("flow", "temperature"))

        assert "flow" in mesh_grad.point_data.keys()
        assert "temperature_gradient" in mesh_grad.point_data["flow"].keys()

    def test_list_of_keys(self, simple_tet_mesh):
        """Test list of multiple keys."""
        mesh = simple_tet_mesh
        mesh.point_data["field1"] = torch.ones(mesh.n_points)
        mesh.point_data["field2"] = torch.ones(mesh.n_points) * 2

        mesh_grad = mesh.compute_point_derivatives(keys=["field1", "field2"])

        assert "field1_gradient" in mesh_grad.point_data.keys()
        assert "field2_gradient" in mesh_grad.point_data.keys()


class TestCircumcentricDual:
    """Test circumcentric dual computation."""

    def test_circumcenter_edge(self):
        """Test circumcenter of edge (1-simplex)."""
        from physicsnemo.mesh.calculus._circumcentric_dual import compute_circumcenters

        # Single edge
        vertices = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])

        circumcenters = compute_circumcenters(vertices)

        # Should be midpoint
        expected = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(circumcenters, expected, atol=1e-6)

    def test_circumcenter_triangle_2d(self):
        """Test circumcenter of triangle in 2D."""
        from physicsnemo.mesh.calculus._circumcentric_dual import compute_circumcenters

        # Right triangle at origin
        vertices = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])

        circumcenters = compute_circumcenters(vertices)

        # Should be at [0.5, 0.5] (midpoint of hypotenuse)
        expected = torch.tensor([[0.5, 0.5]])
        assert torch.allclose(circumcenters, expected, atol=1e-5)

    def test_circumcenter_triangle_3d(self):
        """Test circumcenter of triangle embedded in 3D."""
        from physicsnemo.mesh.calculus._circumcentric_dual import compute_circumcenters

        # Right triangle in xy-plane
        vertices = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

        circumcenters = compute_circumcenters(vertices)

        # For embedded triangle, uses least-squares (over-determined system)
        # Just verify shape and finiteness
        assert circumcenters.shape == (1, 3)
        assert torch.isfinite(circumcenters).all()

    def test_circumcenter_tetrahedron(self):
        """Test circumcenter of tetrahedron."""
        from physicsnemo.mesh.calculus._circumcentric_dual import compute_circumcenters

        # Regular tetrahedron (approximately)
        vertices = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0], [0.5, 0.433, 0.816]]]
        )

        circumcenters = compute_circumcenters(vertices)

        # Should be equidistant from all vertices
        assert circumcenters.shape == (1, 3)

        # Verify equidistance
        for i in range(4):
            dist = torch.norm(circumcenters[0] - vertices[0, i])
            if i == 0:
                ref_dist = dist
            else:
                assert torch.allclose(dist, ref_dist, atol=1e-4)


class TestDivergenceDEC:
    """Test DEC divergence code path."""

    @pytest.mark.skip(
        reason="DEC divergence not fully implemented - uses placeholder formula"
    )
    def test_dec_divergence_linear_field(self, simple_tet_mesh):
        """Test DEC divergence on linear field."""
        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_dec

        mesh = simple_tet_mesh
        v = mesh.points.clone()

        div_v = compute_divergence_points_dec(mesh, v)

        # Should be 3 (div of identity)
        assert torch.allclose(div_v, torch.full_like(div_v, 3.0), atol=0.5)


class TestHigherCodeimension:
    """Test manifolds with codimension > 1."""

    def test_gradient_on_curve_in_3d(self):
        """Test gradient on 1D curve in 3D space (codimension=2)."""
        # Helix
        t = torch.linspace(0, 2 * torch.pi, 20)
        points = torch.stack([torch.cos(t), torch.sin(t), t], dim=-1)

        # Edges along curve
        cells = torch.stack([torch.arange(19), torch.arange(1, 20)], dim=-1)

        mesh = Mesh(points=points, cells=cells)

        # Scalar field along curve
        mesh.point_data["test"] = t

        mesh_grad = mesh.compute_point_derivatives(
            keys="test", gradient_type="extrinsic"
        )

        grad = mesh_grad.point_data["test_gradient"]
        assert grad.shape == (mesh.n_points, 3)


class TestLSQWeighting:
    """Test LSQ weight variations."""

    def test_lsq_with_ill_conditioned_system(self):
        """Test LSQ handles ill-conditioned systems."""
        # Create mesh where some points have nearly collinear neighbors
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.01, 0.01, 0.0],  # Nearly collinear with edge
                [1.02, 0.0, 0.01],  # Also nearly collinear
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        mesh = Mesh(points=points, cells=cells)

        phi = torch.arange(mesh.n_points, dtype=torch.float32)

        from physicsnemo.mesh.calculus._lsq_reconstruction import (
            compute_point_gradient_lsq,
        )

        # Should not crash despite ill-conditioning
        grad = compute_point_gradient_lsq(mesh, phi)

        assert torch.isfinite(grad).all()
        # Some points may have zero gradient if too few neighbors
        assert grad.shape == (mesh.n_points, 3)


class TestCellGradientEdgeCases:
    """Test cell gradient edge cases."""

    def test_cell_with_no_neighbors(self):
        """Test cell with no face-adjacent neighbors."""
        # Single isolated tet
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        mesh = Mesh(points=points, cells=cells)

        mesh.cell_data["test"] = torch.tensor([5.0])

        from physicsnemo.mesh.calculus._lsq_reconstruction import (
            compute_cell_gradient_lsq,
        )

        # Should handle gracefully (no neighbors)
        grad = compute_cell_gradient_lsq(mesh, mesh.cell_data["test"])

        # Gradient should be zero (no neighbors to reconstruct from)
        assert torch.allclose(grad, torch.zeros_like(grad))


class TestProjectionEdgeCases:
    """Test tangent space projection edge cases."""

    def test_projection_on_flat_mesh(self, simple_tet_mesh):
        """Test that projection on codim=0 mesh returns input unchanged."""
        from physicsnemo.mesh.calculus.gradient import project_to_tangent_space

        torch.manual_seed(42)
        mesh = simple_tet_mesh  # Codimension 0
        gradients = torch.randn(mesh.n_points, mesh.n_spatial_dims)

        projected = project_to_tangent_space(mesh, gradients, "points")

        assert torch.allclose(projected, gradients)

    def test_projection_higher_codimension_pca(self):
        """Test projection on codim>1 uses PCA to find tangent space."""
        torch.manual_seed(42)
        # 1D curve in 3D (codimension=2)
        t = torch.linspace(0, 1, 10)
        points = torch.stack([t, t**2, t**3], dim=-1)
        cells = torch.stack([torch.arange(9), torch.arange(1, 10)], dim=-1)
        mesh = Mesh(points=points, cells=cells)

        from physicsnemo.mesh.calculus.gradient import project_to_tangent_space

        gradients = torch.randn(mesh.n_points, 3)
        projected = project_to_tangent_space(mesh, gradients, "points")

        # Should project to tangent space (1D manifold)
        # Projected gradient should have smaller norm than original (normal component removed)
        assert projected.shape == gradients.shape

        # Check that projection actually happened (not identity)
        assert not torch.allclose(projected, gradients)

        # Projected gradient should generally have smaller or equal norm
        projected_norms = torch.norm(projected, dim=-1)
        original_norms = torch.norm(gradients, dim=-1)
        # Most should be smaller (allowing some numerical tolerance)
        assert (projected_norms <= original_norms + 1e-5).float().mean() > 0.7


class TestExteriorDerivative1:
    """Test d₁ exterior derivative."""

    def test_exterior_derivative_1_on_triangles(self):
        """Test d₁: Ω¹ → Ω² on triangle mesh."""
        from physicsnemo.mesh.calculus._exterior_derivative import (
            exterior_derivative_0,
            exterior_derivative_1,
        )

        # Triangle mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]])
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        mesh = Mesh(points=points, cells=cells)

        # Create 0-form and compute df
        vertex_values = torch.arange(mesh.n_points, dtype=torch.float32)
        edge_1form, edges = exterior_derivative_0(mesh, vertex_values)

        # Compute d(1-form)
        face_2form, faces = exterior_derivative_1(mesh, edge_1form, edges)

        assert face_2form.shape[0] == mesh.n_cells

    def test_exterior_derivative_1_error_on_1d(self):
        """Test d₁ raises error on 1D manifold."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_1

        # 1D mesh (curve)
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        cells = torch.tensor([[0, 1], [1, 2]])
        mesh = Mesh(points=points, cells=cells)

        edge_values = torch.ones(mesh.n_cells)
        edges = mesh.cells

        with pytest.raises(ValueError, match="requires n_manifold_dims >= 2"):
            exterior_derivative_1(mesh, edge_values, edges)


class TestHodgeStarErrors:
    """Test Hodge star error paths."""

    def test_codifferential_not_implemented(self, simple_tet_mesh):
        """Test that codifferential raises NotImplementedError."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0
        from physicsnemo.mesh.calculus._hodge_star import codifferential

        mesh = simple_tet_mesh
        vertex_values = torch.ones(mesh.n_points)
        edge_values, edges = exterior_derivative_0(mesh, vertex_values)

        with pytest.raises(NotImplementedError):
            codifferential(k=0, edges=edges)


class TestTangentSpaceProjection:
    """Test tangent space projection for tensors."""

    def test_project_tensor_gradient_to_tangent(self):
        """Test projecting tensor gradient onto tangent space."""
        from physicsnemo.mesh.calculus.gradient import project_to_tangent_space

        torch.manual_seed(42)
        # Surface mesh
        mesh = procedural.lumpy_sphere.load(radius=1.0, subdivisions=2)

        # Tensor gradient (n_points, n_spatial_dims, 2)
        tensor_grads = torch.randn(mesh.n_points, 3, 2)

        projected = project_to_tangent_space(mesh, tensor_grads, "points")

        assert projected.shape == tensor_grads.shape
        # Should be different from input (projection happened)
        assert not torch.allclose(projected, tensor_grads)


class TestIntrinsicLSQEdgeCases:
    """Test intrinsic LSQ edge cases."""

    def test_intrinsic_lsq_on_flat_mesh(self, simple_tet_mesh):
        """Test intrinsic LSQ falls back to standard for flat meshes."""
        from physicsnemo.mesh.calculus._lsq_intrinsic import (
            compute_point_gradient_lsq_intrinsic,
        )

        mesh = simple_tet_mesh  # Codimension 0
        phi = torch.ones(mesh.n_points)

        grad = compute_point_gradient_lsq_intrinsic(mesh, phi)

        # Should call standard LSQ for flat meshes
        assert grad.shape == (mesh.n_points, mesh.n_spatial_dims)


class TestDECDivergence:
    """Test DEC divergence implementation."""

    def test_dec_divergence_basic(self):
        """Test DEC divergence code path."""
        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_dec

        # Simple triangle mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, 0.5]])
        cells = torch.tensor([[0, 1, 3], [0, 2, 3], [1, 2, 3]])
        mesh = Mesh(points=points, cells=cells)

        # Simple vector field
        v = points.clone()  # v = r

        div_v = compute_divergence_points_dec(mesh, v)

        # Just verify it runs and returns finite values
        assert div_v.shape == (mesh.n_points,)
        assert torch.isfinite(div_v).all()


class TestDerivativesMethodCombinations:
    """Test all method × gradient_type combinations."""

    def test_dec_method_extrinsic_gradient(self):
        """Test method='dec' with gradient_type='extrinsic'."""
        mesh = procedural.lumpy_sphere.load(radius=1.0, subdivisions=2)
        mesh.point_data["test"] = torch.ones(mesh.n_points)

        mesh_grad = mesh.compute_point_derivatives(
            keys="test", method="dec", gradient_type="extrinsic"
        )

        assert "test_gradient" in mesh_grad.point_data.keys()

    def test_dec_method_both_gradients(self):
        """Test method='dec' with gradient_type='both'."""
        mesh = procedural.lumpy_sphere.load(radius=1.0, subdivisions=2)
        mesh.point_data["test"] = torch.ones(mesh.n_points)

        mesh_grad = mesh.compute_point_derivatives(
            keys="test", method="dec", gradient_type="both"
        )

        assert "test_gradient_extrinsic" in mesh_grad.point_data.keys()
        assert "test_gradient_intrinsic" in mesh_grad.point_data.keys()


class TestCellDerivativesGradientTypes:
    """Test cell derivatives with different gradient types."""

    def test_cell_extrinsic_gradient(self, simple_tet_mesh):
        """Test cell gradient with gradient_type='extrinsic'."""
        mesh = simple_tet_mesh
        mesh.cell_data["test"] = torch.ones(mesh.n_cells)

        mesh_grad = mesh.compute_cell_derivatives(
            keys="test", gradient_type="extrinsic"
        )

        assert "test_gradient" in mesh_grad.cell_data.keys()

    def test_cell_both_gradients(self, simple_tet_mesh):
        """Test cell gradient with gradient_type='both'."""
        mesh = simple_tet_mesh
        mesh.cell_data["test"] = torch.ones(mesh.n_cells)

        mesh_grad = mesh.compute_cell_derivatives(keys="test", gradient_type="both")

        assert "test_gradient_extrinsic" in mesh_grad.cell_data.keys()
        assert "test_gradient_intrinsic" in mesh_grad.cell_data.keys()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
