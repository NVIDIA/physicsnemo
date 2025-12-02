from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from tensordict import TensorDict, tensorclass

from physicsnemo.mesh.utilities import get_cached, set_cached


@tensorclass  # Note: tensor_only=True provides minimal performance benefit (<5%) but reduces flexibility
class Mesh:
    points: torch.Tensor  # shape: (n_points, n_spatial_dimensions)
    cells: torch.Tensor  # shape: (n_cells, n_manifold_dimensions + 1)
    point_data: TensorDict = None  # accepts dict/None, converted to TensorDict in __post_init__  # ty: ignore
    cell_data: TensorDict = None  # accepts dict/None, converted to TensorDict in __post_init__  # ty: ignore
    global_data: TensorDict = None  # accepts dict/None, converted to TensorDict in __post_init__  # ty: ignore

    def __post_init__(self):
        ### Validate shapes
        if self.points.ndim != 2:
            raise ValueError(
                f"`points` must have shape (n_points, n_spatial_dimensions), but got {self.points.shape=}."
            )
        if self.cells.ndim != 2:
            raise ValueError(
                f"`cells` must have shape (n_cells, n_manifold_dimensions + 1), but got {self.cells.shape=}."
            )
        if self.n_manifold_dims > self.n_spatial_dims:
            raise ValueError(
                f"`n_manifold_dims` must be <= `n_spatial_dims`, but got {self.n_manifold_dims=} > {self.n_spatial_dims=}."
            )

        ### Validate dtypes
        if torch.is_floating_point(self.cells):
            raise TypeError(
                f"`cells` must have an int-like dtype, but got {self.cells.dtype=}."
            )

        ### Initialize data TensorDicts
        if self.point_data is None:
            self.point_data = {}
        if self.cell_data is None:
            self.cell_data = {}
        if self.global_data is None:
            self.global_data = {}

        if not isinstance(self.point_data, TensorDict):
            self.point_data = TensorDict(
                dict(self.point_data),
                batch_size=torch.Size([self.n_points]),
                device=self.points.device,
            )
        if not isinstance(self.cell_data, TensorDict):
            self.cell_data = TensorDict(
                dict(self.cell_data),
                batch_size=torch.Size([self.n_cells]),
                device=self.points.device,
            )
        if not isinstance(self.global_data, TensorDict):
            self.global_data = TensorDict(
                dict(self.global_data),
                batch_size=torch.Size([]),
                device=self.points.device,
            )

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def n_spatial_dims(self) -> int:
        return self.points.shape[-1]

    @property
    def n_cells(self) -> int:
        return self.cells.shape[0]

    @property
    def n_manifold_dims(self) -> int:
        return self.cells.shape[-1] - 1

    @property
    def codimension(self) -> int:
        """Compute the codimension of the mesh.

        The codimension is the difference between the spatial dimension and the
        manifold dimension: codimension = n_spatial_dims - n_manifold_dims.

        Examples:
            - Edges (1-simplices) in 2D: codimension = 2 - 1 = 1 (codimension-1)
            - Triangles (2-simplices) in 3D: codimension = 3 - 2 = 1 (codimension-1)
            - Edges in 3D: codimension = 3 - 1 = 2 (codimension-2)
            - Points in 2D: codimension = 2 - 0 = 2 (codimension-2)

        Returns:
            The codimension of the mesh (always non-negative).
        """
        return self.n_spatial_dims - self.n_manifold_dims

    @property
    def cell_centroids(self) -> torch.Tensor:
        """Compute the centroids (geometric centers) of all cells.

        The centroid of a cell is computed as the arithmetic mean of its vertex positions.
        For an n-simplex with vertices (v0, v1, ..., vn), the centroid is:
            centroid = (v0 + v1 + ... + vn) / (n + 1)

        The result is cached in cell_data["_cache"]["centroids"] for efficiency.

        Returns:
            Tensor of shape (n_cells, n_spatial_dims) containing the centroid of each cell.
        """
        cached = get_cached(self.cell_data, "centroids")
        if cached is None:
            cached = self.points[self.cells].mean(dim=1)
            set_cached(self.cell_data, "centroids", cached)
        return cached

    @property
    def cell_areas(self) -> torch.Tensor:
        """Compute volumes (areas) of n-simplices using the Gram determinant method.

        This works for simplices of any manifold dimension embedded in any spatial dimension.
        For example: edges in 2D/3D, triangles in 2D/3D/4D, tetrahedra in 3D/4D, etc.

        The volume of an n-simplex with vertices (v0, v1, ..., vn) is:
            Volume = (1/n!) * sqrt(det(E^T @ E))
        where E is the matrix with columns (v1-v0, v2-v0, ..., vn-v0).

        Returns:
            Tensor of shape (n_cells,) containing the volume of each cell.
        """
        cached = get_cached(self.cell_data, "areas")
        if cached is None:
            ### Compute relative vectors from first vertex to all others
            # Shape: (n_cells, n_manifold_dims, n_spatial_dims)
            relative_vectors = (
                self.points[self.cells[:, 1:]] - self.points[self.cells[:, [0]]]
            )

            ### Compute Gram matrix: G = E^T @ E
            # E conceptually has shape (n_spatial_dims, n_manifold_dims) per cell
            # Gram matrix has shape (n_manifold_dims, n_manifold_dims) per cell
            # In batch form: (n_cells, n_manifold_dims, n_spatial_dims) @ (n_cells, n_spatial_dims, n_manifold_dims)
            gram_matrix = torch.matmul(
                relative_vectors,  # (n_cells, n_manifold_dims, n_spatial_dims)
                relative_vectors.transpose(
                    -2, -1
                ),  # (n_cells, n_spatial_dims, n_manifold_dims)
            )  # Result: (n_cells, n_manifold_dims, n_manifold_dims)

            ### Compute volume: sqrt(|det(G)|) / n!
            # Compute factorial using torch for small integers
            factorial = torch.arange(1, self.n_manifold_dims + 1).prod()

            cached = gram_matrix.det().abs().sqrt() / factorial
            set_cached(self.cell_data, "areas", cached)

        return cached

    @property
    def cell_normals(self) -> torch.Tensor:
        """Compute unit normal vectors for codimension-1 cells.

        Normal vectors are uniquely defined (up to orientation) only for codimension-1
        manifolds, where n_manifold_dims = n_spatial_dims - 1. This is because the
        perpendicular subspace to an (n-1)-dimensional manifold in n-dimensional space
        is 1-dimensional, yielding a unique normal direction.

        Examples of valid codimension-1 manifolds:
        - Edges (1-simplices) in 2D space: normal is a 2D vector
        - Triangles (2-simplices) in 3D space: normal is a 3D vector
        - Tetrahedron cells (3-simplices) in 4D space: normal is a 4D vector

        Examples of invalid higher-codimension cases:
        - Edges in 3D space: perpendicular space is 2D (no unique normal)
        - Points in 2D/3D space: perpendicular space is 2D/3D (no unique normal)

        The implementation uses the generalized cross product (Hodge star operator),
        computed via signed minor determinants. This generalizes:
        - 2D: 90° counterclockwise rotation of edge vector
        - 3D: Standard cross product of two edge vectors
        - nD: Determinant-based formula for (n-1) edge vectors in n-space

        Returns:
            Tensor of shape (n_cells, n_spatial_dims) containing unit normal vectors.

        Raises:
            ValueError: If the mesh is not codimension-1 (n_manifold_dims ≠ n_spatial_dims - 1).
        """
        cached = get_cached(self.cell_data, "normals")
        if cached is None:
            ### Validate codimension-1 requirement
            if self.codimension != 1:
                raise ValueError(
                    f"cell normals are only defined for codimension-1 manifolds.\n"
                    f"Got {self.n_manifold_dims=} and {self.n_spatial_dims=}.\n"
                    f"Required: n_manifold_dims = n_spatial_dims - 1 (codimension-1).\n"
                    f"Current codimension: {self.codimension}"
                )

            ### Compute relative vectors from first vertex to all others
            # Shape: (n_cells, n_manifold_dims, n_spatial_dims)
            # These form the rows of matrix E for each cell
            relative_vectors = (
                self.points[self.cells[:, 1:]] - self.points[self.cells[:, [0]]]
            )

            ### Compute normal using generalized cross product (Hodge star)
            # For (n-1) vectors in R^n represented as rows of matrix E,
            # the perpendicular vector has components:
            #   n_i = (-1)^(n-1+i) * det(E with column i removed)
            # This generalizes 2D rotation and 3D cross product.
            normal_components = []

            for i in range(self.n_spatial_dims):
                ### Select all columns except the i-th to form (n-1)×(n-1) submatrix
                cols_mask = torch.ones(
                    self.n_spatial_dims,
                    dtype=torch.bool,
                    device=relative_vectors.device,
                )
                cols_mask[i] = False
                submatrix = relative_vectors[
                    :, :, cols_mask
                ]  # (n_cells, n_manifold_dims, n_manifold_dims)

                ### Compute signed minor: (-1)^(n_manifold_dims + i) * det(submatrix)
                det = submatrix.det()  # (n_cells,)
                sign = (-1) ** (self.n_manifold_dims + i)
                normal_components.append(sign * det)

            ### Stack components and normalize to unit length
            normals = torch.stack(
                normal_components, dim=-1
            )  # (n_cells, n_spatial_dims)
            cached = F.normalize(normals, dim=-1)
            set_cached(self.cell_data, "normals", cached)

        return cached

    @property
    def point_normals(self) -> torch.Tensor:
        """Compute area-weighted normal vectors at mesh vertices.

        For each point (vertex), computes a normal vector by taking an area-weighted
        average of the normals of all adjacent cells. This provides a smooth approximation
        of the surface normal at each vertex.

        The normal at vertex v is computed as:
            point_normal_v = normalize(sum_over_adjacent_cells(cell_normal * cell_area))

        Area weighting ensures that larger adjacent faces have more influence on the
        vertex normal, which is standard practice in computer graphics and produces
        better visual results than simple averaging.

        Normal vectors are only well-defined for codimension-1 manifolds, where each
        cell has a unique normal direction. For higher codimensions, normals are
        ambiguous and this property will raise an error.

        The result is cached in point_data["_cache"]["normals"] for efficiency.

        Returns:
            Tensor of shape (n_points, n_spatial_dims) containing unit normal vectors
            at each vertex. For isolated points (with no adjacent cells), the normal
            is a zero vector.

        Raises:
            ValueError: If the mesh is not codimension-1 (n_manifold_dims ≠ n_spatial_dims - 1).

        Example:
            >>> # Triangle mesh in 3D
            >>> mesh = create_triangle_mesh_3d()
            >>> normals = mesh.point_normals  # (n_points, 3)
            >>> # Normals are unit vectors (or zero for isolated points)
            >>> assert torch.allclose(normals.norm(dim=-1), torch.ones(mesh.n_points), atol=1e-6)
        """
        cached = get_cached(self.point_data, "normals")
        if cached is None:
            ### Validate codimension-1 requirement (same as cell_normals)
            if self.codimension != 1:
                raise ValueError(
                    f"Point normals are only defined for codimension-1 manifolds.\n"
                    f"Got {self.n_manifold_dims=} and {self.n_spatial_dims=}.\n"
                    f"Required: n_manifold_dims = n_spatial_dims - 1 (codimension-1).\n"
                    f"Current codimension: {self.codimension}"
                )

            ### Get cell normals and areas (triggers computation if not cached)
            cell_normals = self.cell_normals  # (n_cells, n_spatial_dims)
            cell_areas = self.cell_areas  # (n_cells,)

            ### Initialize accumulated weighted normals for each point
            # Shape: (n_points, n_spatial_dims)
            weighted_normals = torch.zeros(
                (self.n_points, self.n_spatial_dims),
                dtype=self.points.dtype,
                device=self.points.device,
            )

            ### Vectorized accumulation of area-weighted normals
            # For each cell, add (cell_normal * cell_area) to each of its vertices

            # Get all vertex indices from all cells
            # Shape: (n_cells, n_vertices_per_cell)
            n_vertices_per_cell = self.cells.shape[1]

            # Flatten point indices: (n_cells * n_vertices_per_cell,)
            point_indices = self.cells.flatten()

            # Repeat cell normals for each vertex in the cell
            # Shape: (n_cells, n_vertices_per_cell, n_spatial_dims)
            cell_normals_repeated = cell_normals.unsqueeze(1).expand(
                -1, n_vertices_per_cell, -1
            )
            # Flatten: (n_cells * n_vertices_per_cell, n_spatial_dims)
            cell_normals_flat = cell_normals_repeated.reshape(-1, self.n_spatial_dims)

            # Repeat cell areas for each vertex in the cell
            # Shape: (n_cells, n_vertices_per_cell)
            cell_areas_repeated = cell_areas.unsqueeze(1).expand(
                -1, n_vertices_per_cell
            )
            # Flatten: (n_cells * n_vertices_per_cell,)
            cell_areas_flat = cell_areas_repeated.flatten()

            # Weight normals by area
            # Shape: (n_cells * n_vertices_per_cell, n_spatial_dims)
            weighted_normals_flat = cell_normals_flat * cell_areas_flat.unsqueeze(-1)

            ### Scatter-add weighted normals to their corresponding points
            # Expand point_indices to match weighted_normals_flat shape
            point_indices_expanded = point_indices.unsqueeze(-1).expand(
                -1, self.n_spatial_dims
            )

            # Accumulate weighted normals at each point
            weighted_normals.scatter_add_(
                dim=0,
                index=point_indices_expanded,
                src=weighted_normals_flat,
            )

            ### Normalize to get unit normals
            # For isolated points (zero weighted sum), F.normalize returns zero vector
            cached = F.normalize(weighted_normals, dim=-1)
            set_cached(self.point_data, "normals", cached)

        return cached

    @property
    def gaussian_curvature_vertices(self) -> torch.Tensor:
        """Compute intrinsic Gaussian curvature at mesh vertices.

        Uses the angle defect method from discrete differential geometry:
            K = (full_angle - Σ angles) / voronoi_area

        This is an intrinsic measure of curvature (Theorema Egregium) that works
        for any codimension, as it depends only on distances within the manifold.

        Signed curvature:
        - Positive: Elliptic/convex (sphere-like)
        - Zero: Flat/parabolic (plane-like)
        - Negative: Hyperbolic/saddle (saddle-like)

        The result is cached in point_data["_cache"]["gaussian_curvature"] for efficiency.

        Returns:
            Tensor of shape (n_points,) containing signed Gaussian curvature.
            Isolated vertices have NaN curvature.

        Example:
            >>> # Sphere of radius r has K = 1/r²
            >>> sphere = create_sphere_mesh(radius=2.0)
            >>> K = sphere.gaussian_curvature_vertices
            >>> assert K.mean() ≈ 0.25

        Note:
            Satisfies discrete Gauss-Bonnet theorem:
                Σ_vertices (K_i * A_i) = 2π * χ(M)
        """
        cached = get_cached(self.point_data, "gaussian_curvature")
        if cached is None:
            from physicsnemo.mesh.curvature import gaussian_curvature_vertices

            cached = gaussian_curvature_vertices(self)
            set_cached(self.point_data, "gaussian_curvature", cached)

        return cached

    @property
    def gaussian_curvature_cells(self) -> torch.Tensor:
        """Compute Gaussian curvature at cell centers using dual mesh concept.

        Treats cell centroids as vertices of a dual mesh and computes curvature
        based on angles between connections to adjacent cell centroids.

        The result is cached in cell_data["_cache"]["gaussian_curvature"] for efficiency.

        Returns:
            Tensor of shape (n_cells,) containing Gaussian curvature at cells.

        Example:
            >>> K_cells = mesh.gaussian_curvature_cells
        """
        cached = get_cached(self.cell_data, "gaussian_curvature")
        if cached is None:
            from physicsnemo.mesh.curvature import gaussian_curvature_cells

            cached = gaussian_curvature_cells(self)
            set_cached(self.cell_data, "gaussian_curvature", cached)

        return cached

    @property
    def mean_curvature_vertices(self) -> torch.Tensor:
        """Compute extrinsic mean curvature at mesh vertices.

        Uses the cotangent Laplace-Beltrami operator:
            H = (1/2) * ||L @ points|| / voronoi_area

        Mean curvature is an extrinsic measure (depends on embedding) and is
        only defined for codimension-1 manifolds where normal vectors exist.

        For 2D surfaces: H = (k1 + k2) / 2 where k1, k2 are principal curvatures

        Signed curvature:
        - Positive: Convex (sphere exterior with outward normals)
        - Negative: Concave (sphere interior with outward normals)
        - Zero: Minimal surface (soap film)

        The result is cached in point_data["_cache"]["mean_curvature"] for efficiency.

        Returns:
            Tensor of shape (n_points,) containing signed mean curvature.
            Isolated vertices have NaN curvature.

        Raises:
            ValueError: If mesh is not codimension-1

        Example:
            >>> # Sphere of radius r has H = 1/r
            >>> sphere = create_sphere_mesh(radius=2.0)
            >>> H = sphere.mean_curvature_vertices
            >>> assert H.mean() ≈ 0.5
        """
        cached = get_cached(self.point_data, "mean_curvature")
        if cached is None:
            from physicsnemo.mesh.curvature import mean_curvature_vertices

            cached = mean_curvature_vertices(self)
            set_cached(self.point_data, "mean_curvature", cached)

        return cached

    @classmethod
    def merge(
        cls, meshes: Sequence["Mesh"], global_data_strategy: Literal["stack"] = "stack"
    ) -> "Mesh":
        ### Validate inputs
        if not torch.compiler.is_compiling():
            if len(meshes) == 0:
                raise ValueError("At least one Mesh must be provided to merge.")
            elif len(meshes) == 1:  # Short-circuit for speed in this case
                return meshes[0]
            if not all(isinstance(m, Mesh) for m in meshes):
                raise TypeError(
                    f"All objects must be Mesh types. Got:\n"
                    f"{[type(m) for m in meshes]=}"
                )
            # Check dimensional consistency across all meshes
            validations = {
                "spatial dimensions": [m.n_spatial_dims for m in meshes],
                "manifold dimensions": [m.n_manifold_dims for m in meshes],
            }
            for name, values in validations.items():
                if not all(v == values[0] for v in values):
                    raise ValueError(
                        f"All meshes must have the same {name}. Got:\n{values=}"
                    )
            # Check that all cell_data dicts have the same keys across all meshes
            if not all(
                m.cell_data.keys() == meshes[0].cell_data.keys() for m in meshes
            ):
                raise ValueError("All meshes must have the same cell_data keys.")

        ### Merge the meshes

        # Compute the number of points for each mesh, cumulatively, so that we can update
        # the point indices for the constituent cells arrays accordingly.
        n_points_for_meshes = torch.tensor(
            [m.n_points for m in meshes],
            device=meshes[0].points.device,
        )
        cumsum_n_points = torch.cumsum(n_points_for_meshes, dim=0)
        cell_index_offsets = cumsum_n_points.roll(1)
        cell_index_offsets[0] = 0

        if global_data_strategy == "stack":
            global_data = TensorDict.stack([m.global_data for m in meshes])
        else:
            raise ValueError(f"Invalid {global_data_strategy=}")

        return cls(
            points=torch.cat([m.points for m in meshes], dim=0),
            cells=torch.cat(
                [m.cells + offset for m, offset in zip(meshes, cell_index_offsets)],
                dim=0,
            ),
            point_data=TensorDict.cat([m.point_data for m in meshes], dim=0),
            cell_data=TensorDict.cat([m.cell_data for m in meshes], dim=0),
            global_data=global_data,
        )

    def slice_points(self, indices: int | slice | torch.Tensor) -> "Mesh":
        """Returns a new Mesh with a subset of the points.

        Args:
            indices: Indices or mask to select points.
        """
        new_point_data: TensorDict = self.point_data[indices]  # type: ignore
        return Mesh(
            points=self.points[indices],
            cells=self.cells,
            point_data=new_point_data,
            cell_data=self.cell_data,
            global_data=self.global_data,
        )

    def slice_cells(self, indices: int | slice | torch.Tensor) -> "Mesh":
        """Returns a new Mesh with a subset of the cells.

        Args:
            indices: Indices or mask to select cells.
        """
        new_cell_data: TensorDict = self.cell_data[indices]  # type: ignore
        return Mesh(
            points=self.points,
            cells=self.cells[indices],
            point_data=self.point_data,
            cell_data=new_cell_data,
            global_data=self.global_data,
        )

    def sample_random_points_on_cells(
        self,
        cell_indices: Sequence[int] | torch.Tensor | None = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Sample random points on specified cells of the mesh.

        Uses a Dirichlet distribution to generate barycentric coordinates, which are
        then used to compute random points as weighted combinations of cell vertices.
        The concentration parameter alpha controls the distribution of samples within
        each cell (simplex).

        This is a convenience method that delegates to physicsnemo.mesh.sampling.sample_random_points_on_cells.

        Args:
            cell_indices: Indices of cells to sample from. Can be a Sequence or tensor.
                Allows repeated indices to sample multiple points from the same cell.
                If None, samples one point from each cell (equivalent to arange(n_cells)).
                Shape: (n_samples,) where n_samples is the number of points to sample.
            alpha: Concentration parameter for the Dirichlet distribution. Controls how
                samples are distributed within each cell:
                - alpha = 1.0: Uniform distribution over the simplex (default)
                - alpha > 1.0: Concentrates samples toward the center of each cell
                - alpha < 1.0: Concentrates samples toward vertices and edges

        Returns:
            Random points on cells, shape (n_samples, n_spatial_dims). Each point lies
            within its corresponding cell. If cell_indices is None, n_samples = n_cells.

        Raises:
            NotImplementedError: If alpha != 1.0 and torch.compile is being used.
                This is due to a PyTorch limitation with Gamma distributions under torch.compile.
            IndexError: If any cell_indices are out of bounds.

        Example:
            >>> # Sample one point from each cell uniformly
            >>> points = mesh.sample_random_points_on_cells()
            >>>
            >>> # Sample points from specific cells (with repeats allowed)
            >>> cell_indices = torch.tensor([0, 0, 1, 5, 5, 5])
            >>> points = mesh.sample_random_points_on_cells(cell_indices=cell_indices)
            >>>
            >>> # Sample with concentration toward cell centers
            >>> points = mesh.sample_random_points_on_cells(alpha=3.0)
        """
        from physicsnemo.mesh.sampling import sample_random_points_on_cells

        return sample_random_points_on_cells(
            mesh=self,
            cell_indices=cell_indices,
            alpha=alpha,
        )

    def sample_data_at_points(
        self,
        query_points: torch.Tensor,
        data_source: Literal["cells", "points"] = "cells",
        multiple_cells_strategy: Literal["mean", "nan"] = "mean",
        project_onto_nearest_cell: bool = False,
        tolerance: float = 1e-6,
    ) -> "TensorDict":
        """Sample mesh data at query points in space.

        For each query point, finds the containing cell and returns interpolated data.

        This is a convenience method that delegates to physicsnemo.mesh.sampling.sample_data_at_points.

        Args:
            query_points: Query point locations, shape (n_queries, n_spatial_dims)
            data_source: How to sample data:
                - "cells": Use cell data directly (no interpolation)
                - "points": Interpolate point data using barycentric coordinates
            multiple_cells_strategy: How to handle query points in multiple cells:
                - "mean": Return arithmetic mean of values from all containing cells
                - "nan": Return NaN for ambiguous points
            project_onto_nearest_cell: If True, projects each query point onto the
                nearest cell before sampling. Useful for codimension != 0 manifolds.
            tolerance: Tolerance for considering a point inside a cell.

        Returns:
            TensorDict containing sampled data for each query point. Values are NaN
            for query points outside the mesh (unless project_onto_nearest_cell=True).

        Example:
            >>> # Sample cell data at specific points
            >>> query_pts = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
            >>> sampled_data = mesh.sample_data_at_points(query_pts, data_source="cells")
            >>>
            >>> # Interpolate point data
            >>> sampled_data = mesh.sample_data_at_points(query_pts, data_source="points")
        """
        from physicsnemo.mesh.sampling import sample_data_at_points

        return sample_data_at_points(
            mesh=self,
            query_points=query_points,
            data_source=data_source,
            multiple_cells_strategy=multiple_cells_strategy,
            project_onto_nearest_cell=project_onto_nearest_cell,
            tolerance=tolerance,
        )

    def cell_data_to_point_data(self, overwrite_keys: bool = False) -> "Mesh":
        """Convert cell data to point data by averaging.

        For each point, computes the average of the cell data values from all cells
        that contain that point. The resulting point data is added to the mesh's
        point_data dictionary. Original cell data is preserved.

        Args:
            overwrite_keys: If True, silently overwrite any existing point_data keys.
                If False (default), raise an error if a key already exists in point_data.

        Returns:
            New Mesh with converted data added to point_data. Original cell_data is preserved.

        Raises:
            ValueError: If a cell_data key already exists in point_data and overwrite_keys=False.

        Example:
            >>> mesh = Mesh(points, cells, cell_data={"pressure": cell_pressures})
            >>> mesh_with_point_data = mesh.cell_data_to_point_data()
            >>> # Now mesh has both cell_data["pressure"] and point_data["pressure"]
        """
        ### Check for key conflicts
        if not overwrite_keys:
            for key in self.cell_data.exclude("_cache").keys():
                if key in self.point_data.keys():
                    raise ValueError(
                        f"Key {key!r} already exists in point_data. "
                        f"Set overwrite_keys=True to overwrite."
                    )

        ### Convert each cell data field to point data
        from physicsnemo.mesh.utilities import scatter_aggregate

        new_point_data = self.point_data.clone()

        # Get flat list of point indices and corresponding cell indices
        # self.cells shape: (n_cells, n_vertices_per_cell)
        n_vertices_per_cell = self.cells.shape[1]

        # Flatten: all point indices that appear in cells
        # Shape: (n_cells * n_vertices_per_cell,)
        point_indices = self.cells.flatten()

        # Corresponding cell index for each point
        # Shape: (n_cells * n_vertices_per_cell,)
        cell_indices = torch.arange(
            self.n_cells, device=self.points.device
        ).repeat_interleave(n_vertices_per_cell)

        for key, cell_values in self.cell_data.exclude("_cache").items():
            ### Use scatter aggregation utility to average cell values to points
            # Expand cell values to one entry per vertex
            src_data = cell_values[cell_indices]

            # Aggregate to points using mean
            point_values = scatter_aggregate(
                src_data=src_data,
                src_to_dst_mapping=point_indices,
                n_dst=self.n_points,
                weights=None,
                aggregation="mean",
            )

            new_point_data[key] = point_values

        ### Return new mesh with updated point data
        return Mesh(
            points=self.points,
            cells=self.cells,
            point_data=new_point_data,
            cell_data=self.cell_data,
            global_data=self.global_data,
        )

    def point_data_to_cell_data(self, overwrite_keys: bool = False) -> "Mesh":
        """Convert point data to cell data by averaging.

        For each cell, computes the average of the point data values from all points
        (vertices) that define that cell. The resulting cell data is added to the mesh's
        cell_data dictionary. Original point data is preserved.

        Args:
            overwrite_keys: If True, silently overwrite any existing cell_data keys.
                If False (default), raise an error if a key already exists in cell_data.

        Returns:
            New Mesh with converted data added to cell_data. Original point_data is preserved.

        Raises:
            ValueError: If a point_data key already exists in cell_data and overwrite_keys=False.

        Example:
            >>> mesh = Mesh(points, cells, point_data={"temperature": point_temps})
            >>> mesh_with_cell_data = mesh.point_data_to_cell_data()
            >>> # Now mesh has both point_data["temperature"] and cell_data["temperature"]
        """
        ### Check for key conflicts
        if not overwrite_keys:
            for key in self.point_data.exclude("_cache").keys():
                if key in self.cell_data.keys():
                    raise ValueError(
                        f"Key {key!r} already exists in cell_data. "
                        f"Set overwrite_keys=True to overwrite."
                    )

        ### Convert each point data field to cell data
        new_cell_data = self.cell_data.clone()

        for key, point_values in self.point_data.exclude("_cache").items():
            # Get point values for each cell and average
            # cell_point_values shape: (n_cells, n_vertices_per_cell, ...)
            cell_point_values = point_values[self.cells]

            # Average over vertices dimension (dim=1)
            cell_values = cell_point_values.mean(dim=1)

            new_cell_data[key] = cell_values

        ### Return new mesh with updated cell data
        return Mesh(
            points=self.points,
            cells=self.cells,
            point_data=self.point_data,
            cell_data=new_cell_data,
            global_data=self.global_data,
        )

    def get_facet_mesh(
        self,
        manifold_codimension: int = 1,
        data_source: Literal["points", "cells"] = "cells",
        data_aggregation: Literal["mean", "area_weighted", "inverse_distance"] = "mean",
    ) -> "Mesh":
        """Extract k-codimension facet mesh from this n-dimensional mesh.

        Extracts all (n-k)-simplices from the current n-simplicial mesh. For example:
        - Triangle mesh (2-simplices) → edge mesh (1-simplices) [codimension=1, default]
        - Triangle mesh (2-simplices) → vertex mesh (0-simplices) [codimension=2]
        - Tetrahedral mesh (3-simplices) → triangular facet mesh (2-simplices) [codimension=1, default]
        - Tetrahedral mesh (3-simplices) → edge mesh (1-simplices) [codimension=2]

        The resulting mesh shares the same vertex positions but has connectivity
        representing the lower-dimensional simplices. Data can be inherited from
        either the parent cells or the boundary points.

        Args:
            manifold_codimension: Codimension of extracted mesh relative to parent.
                - 1: Extract (n-1)-facets (default, immediate boundaries of all cells)
                - 2: Extract (n-2)-facets (e.g., edges from tets, vertices from triangles)
                - k: Extract (n-k)-facets
            data_source: Source of data inheritance:
                - "cells": Facets inherit from parent cells they bound. When multiple
                  cells share a facet, data is aggregated according to data_aggregation.
                - "points": Facets inherit from their boundary vertices. Data from
                  multiple boundary points is averaged.
            data_aggregation: Strategy for aggregating data from multiple sources
                (only applies when data_source="cells"):
                - "mean": Simple arithmetic mean
                - "area_weighted": Weighted by parent cell areas
                - "inverse_distance": Weighted by inverse distance from facet centroid
                  to parent cell centroids

        Returns:
            New Mesh with n_manifold_dims = self.n_manifold_dims - manifold_codimension,
            embedded in the same spatial dimension. The mesh shares the same points array
            but has new cells connectivity and aggregated cell_data.

        Raises:
            ValueError: If manifold_codimension is too large for this mesh
                (would result in negative manifold dimension).

        Example:
            >>> # Extract edges from a triangle mesh (codimension 1)
            >>> triangle_mesh = Mesh(points, triangular_cells)
            >>> edge_mesh = triangle_mesh.get_facet_mesh(manifold_codimension=1)
            >>> edge_mesh.n_manifold_dims  # 1 (edges)
            >>>
            >>> # Extract vertices from a triangle mesh (codimension 2)
            >>> vertex_mesh = triangle_mesh.get_facet_mesh(manifold_codimension=2)
            >>> vertex_mesh.n_manifold_dims  # 0 (vertices)
            >>>
            >>> # Extract with area-weighted data aggregation
            >>> facet_mesh = triangle_mesh.get_facet_mesh(
            ...     data_source="cells",
            ...     data_aggregation="area_weighted"
            ... )
        """
        ### Validate that extraction is possible
        new_manifold_dims = self.n_manifold_dims - manifold_codimension
        if new_manifold_dims < 0:
            raise ValueError(
                f"Cannot extract facet mesh with {manifold_codimension=} from mesh with {self.n_manifold_dims=}.\n"
                f"Would result in negative manifold dimension ({new_manifold_dims=}).\n"
                f"Maximum allowed codimension is {self.n_manifold_dims}."
            )

        ### Call kernel to extract facet mesh data
        from physicsnemo.mesh.boundaries import extract_facet_mesh_data

        facet_cells, facet_cell_data = extract_facet_mesh_data(
            parent_mesh=self,
            manifold_codimension=manifold_codimension,
            data_source=data_source,
            data_aggregation=data_aggregation,
        )

        ### Create and return new Mesh
        # Filter out cached properties from point_data
        # Cached geometric properties depend on cell connectivity and would be invalid
        filtered_point_data = self.point_data.exclude("_cache")

        return Mesh(
            points=self.points,  # Share the same points
            cells=facet_cells,  # New connectivity for sub-simplices
            point_data=filtered_point_data,  # User data only, no cached properties
            cell_data=facet_cell_data,  # Aggregated cell data
            global_data=self.global_data,  # Share global data
        )

    def get_boundary_mesh(
        self,
        data_source: Literal["points", "cells"] = "cells",
        data_aggregation: Literal["mean", "area_weighted", "inverse_distance"] = "mean",
    ) -> "Mesh":
        """Extract the boundary surface of this mesh.

        Extracts only the codimension-1 facets that lie on the boundary (appear in
        exactly one cell). This produces the watertight boundary surface of a mesh.

        Key difference from get_facet_mesh():
        - get_facet_mesh(): Returns ALL facets (interior + boundary)
        - get_boundary_mesh(): Returns ONLY boundary facets (appear in 1 cell)

        For a closed watertight mesh, this returns an empty mesh. For an open mesh
        (e.g., a tetrahedral volume), this returns the triangulated surface boundary.

        Args:
            data_source: Source of data inheritance:
                - "cells": Boundary facets inherit from their single parent cell
                - "points": Boundary facets inherit from their boundary vertices
            data_aggregation: Strategy for aggregating data (only applies when
                data_source="cells"):
                - "mean": Simple arithmetic mean
                - "area_weighted": Weighted by parent cell areas
                - "inverse_distance": Weighted by inverse distance from facet centroid
                Note: For boundary facets, each has exactly one parent cell, so
                aggregation typically doesn't affect results.

        Returns:
            New Mesh with n_manifold_dims = self.n_manifold_dims - 1, containing
            only the boundary facets. The mesh shares the same points array but has
            new cells connectivity representing the boundary.

        Example:
            >>> # Extract triangular surface of a tetrahedral mesh
            >>> tet_mesh = Mesh(points, tetrahedra)
            >>> surface_mesh = tet_mesh.get_boundary_mesh()
            >>> surface_mesh.n_manifold_dims  # 2 (triangles)
            >>>
            >>> # For a closed watertight sphere
            >>> sphere = create_sphere_mesh(subdivisions=3)
            >>> boundary = sphere.get_boundary_mesh()
            >>> boundary.n_cells  # 0 (no boundary)
        """
        ### Call kernel to extract boundary mesh data
        from physicsnemo.mesh.boundaries import extract_boundary_mesh_data

        boundary_cells, boundary_cell_data = extract_boundary_mesh_data(
            parent_mesh=self,
            data_source=data_source,
            data_aggregation=data_aggregation,
        )

        ### Filter out cached properties from point_data
        filtered_point_data = self.point_data.exclude("_cache")

        return Mesh(
            points=self.points,  # Share the same points
            cells=boundary_cells,  # New connectivity for boundary facets only
            point_data=filtered_point_data,  # User data only, no cached properties
            cell_data=boundary_cell_data,  # Aggregated cell data
            global_data=self.global_data,  # Share global data
        )

    def is_watertight(self) -> bool:
        """Check if mesh is watertight (has no boundary).

        A mesh is watertight if every codimension-1 facet is shared by exactly 2 cells.
        This means the mesh forms a closed surface/volume with no holes or gaps.

        Returns:
            True if mesh is watertight (no boundary facets), False otherwise

        Example:
            >>> # Closed sphere is watertight
            >>> sphere = create_sphere_mesh(subdivisions=3)
            >>> sphere.is_watertight()  # True
            >>>
            >>> # Open cylinder with holes at ends
            >>> cylinder = create_cylinder_mesh(closed=False)
            >>> cylinder.is_watertight()  # False
            >>>
            >>> # Single tetrahedron has 4 boundary faces
            >>> tet = Mesh(points, cells=torch.tensor([[0, 1, 2, 3]]))
            >>> tet.is_watertight()  # False
        """
        from physicsnemo.mesh.boundaries import is_watertight

        return is_watertight(self)

    def is_manifold(
        self,
        check_level: Literal["facets", "edges", "full"] = "full",
    ) -> bool:
        """Check if mesh is a valid topological manifold.

        A mesh is a manifold if it locally looks like Euclidean space at every point.
        This function checks various topological constraints depending on the check level.

        Args:
            check_level: Level of checking to perform:
                - "facets": Only check codimension-1 facets (each appears 1-2 times)
                - "edges": Check facets + edge neighborhoods (for 2D/3D meshes)
                - "full": Complete manifold validation (default)

        Returns:
            True if mesh passes the specified manifold checks, False otherwise

        Example:
            >>> # Valid manifold (sphere)
            >>> sphere = create_sphere_mesh(subdivisions=3)
            >>> sphere.is_manifold()  # True
            >>>
            >>> # Non-manifold mesh with T-junction (edge shared by 3+ faces)
            >>> non_manifold = create_t_junction_mesh()
            >>> non_manifold.is_manifold()  # False
            >>>
            >>> # Manifold with boundary (open cylinder)
            >>> cylinder = create_cylinder_mesh(closed=False)
            >>> cylinder.is_manifold()  # True (manifold with boundary is OK)

        Note:
            This function checks topological constraints but does not check for
            geometric self-intersections (which would require expensive spatial queries).
        """
        from physicsnemo.mesh.boundaries import is_manifold

        return is_manifold(self, check_level=check_level)

    def get_point_to_cells_adjacency(self):
        """Compute the star of each vertex (all cells containing each point).

        For each point in the mesh, finds all cells that contain that point. This
        is the graph-theoretic "star" operation on vertices.

        Returns:
            Adjacency where adjacency.to_list()[i] contains all cell indices that
            contain point i. Isolated points (not in any cells) have empty lists.

        Example:
            >>> mesh = from_pyvista(pv.examples.load_airplane())
            >>> adj = mesh.get_point_to_cells_adjacency()
            >>> # Get cells containing point 0
            >>> cells_of_point_0 = adj.to_list()[0]
        """
        from physicsnemo.mesh.neighbors import get_point_to_cells_adjacency

        return get_point_to_cells_adjacency(self)

    def get_point_to_points_adjacency(self):
        """Compute point-to-point adjacency (graph edges of the mesh).

        For each point, finds all other points that share a cell with it. In simplicial
        meshes, this is equivalent to finding all points connected by an edge.

        Returns:
            Adjacency where adjacency.to_list()[i] contains all point indices that
            share a cell (edge) with point i. Isolated points have empty lists.

        Example:
            >>> mesh = from_pyvista(pv.examples.load_airplane())
            >>> adj = mesh.get_point_to_points_adjacency()
            >>> # Get neighbors of point 0
            >>> neighbors_of_point_0 = adj.to_list()[0]
        """
        from physicsnemo.mesh.neighbors import get_point_to_points_adjacency

        return get_point_to_points_adjacency(self)

    def get_cell_to_cells_adjacency(self, adjacency_codimension: int = 1):
        """Compute cell-to-cells adjacency based on shared facets.

        Two cells are considered adjacent if they share a k-codimension facet.

        Args:
            adjacency_codimension: Codimension of shared facets defining adjacency.
                - 1 (default): Cells must share a codimension-1 facet (e.g., triangles
                  sharing an edge, tetrahedra sharing a triangular face)
                - 2: Cells must share a codimension-2 facet (e.g., tetrahedra sharing
                  an edge)
                - k: Cells must share a codimension-k facet

        Returns:
            Adjacency where adjacency.to_list()[i] contains all cell indices that
            share a k-codimension facet with cell i.

        Example:
            >>> mesh = from_pyvista(pv.examples.load_tetbeam())
            >>> adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
            >>> # Get cells sharing a face with cell 0
            >>> neighbors_of_cell_0 = adj.to_list()[0]
        """
        from physicsnemo.mesh.neighbors import get_cell_to_cells_adjacency

        return get_cell_to_cells_adjacency(
            self, adjacency_codimension=adjacency_codimension
        )

    def get_cells_to_points_adjacency(self):
        """Get the vertices (points) that comprise each cell.

        This is a simple wrapper around the cells array that returns it in the
        standard Adjacency format for consistency with other neighbor queries.

        Returns:
            Adjacency where adjacency.to_list()[i] contains all point indices that
            are vertices of cell i. For simplicial meshes, all cells have the same
            number of vertices (n_manifold_dims + 1).

        Example:
            >>> mesh = from_pyvista(pv.examples.load_airplane())
            >>> adj = mesh.get_cells_to_points_adjacency()
            >>> # Get vertices of cell 0
            >>> vertices_of_cell_0 = adj.to_list()[0]
        """
        from physicsnemo.mesh.neighbors import get_cells_to_points_adjacency

        return get_cells_to_points_adjacency(self)

    def pad(
        self,
        target_n_points: int | None = None,
        target_n_cells: int | None = None,
        data_padding_value: float = 0.0,
    ) -> "Mesh":
        """Pad points and cells arrays to specified sizes.

        This is the low-level padding method that performs the actual padding operation.
        Padding uses null/degenerate elements that don't affect computations:
        - Points: Additional points at the last existing point (preserves bounding box)
        - cells: Degenerate cells with all vertices at the last existing point (zero area)
        - cell data: Zero-valued padding for all cell data fields

        Args:
            target_n_points: Target number of points. If None, no point padding is applied.
                Must be >= current n_points if specified.
            target_n_cells: Target number of cells. If None, no cell padding is applied.
                Must be >= current n_cells if specified.

        Returns:
            A new Mesh with padded arrays. If both targets are None or equal to
            current sizes, returns self unchanged.

        Raises:
            ValueError: If target sizes are less than current sizes.

        Example:
            >>> mesh = Mesh(points, cells, "no_slip")  # 100 points, 200 cells
            >>> padded = mesh.pad(target_n_points=128, target_n_cells=256)
            >>> padded.n_points  # 128
            >>> padded.n_cells   # 256
        """
        # Validate inputs
        if not torch.compiler.is_compiling():
            if target_n_points is not None and target_n_points < self.n_points:
                raise ValueError(f"{target_n_points=} must be >= {self.n_points=}")
            if target_n_cells is not None and target_n_cells < self.n_cells:
                raise ValueError(f"{target_n_cells=} must be >= {self.n_cells=}")

        # Short-circuit if no padding needed
        if target_n_points is None and target_n_cells is None:
            return self

        # Determine actual target sizes
        if target_n_points is None:
            target_n_points = self.n_points
        if target_n_cells is None:
            target_n_cells = self.n_cells

        from physicsnemo.mesh.utilities._padding import (
            _pad_by_tiling_last,
            _pad_with_value,
        )

        return self.__class__(
            points=_pad_by_tiling_last(self.points, target_n_points),
            cells=_pad_with_value(self.cells, target_n_cells, self.n_points - 1),
            point_data=self.point_data.apply(  # type: ignore
                lambda x: _pad_with_value(x, target_n_points, data_padding_value),
                batch_size=torch.Size([target_n_points]),
            ),
            cell_data=self.cell_data.apply(  # type: ignore
                lambda x: _pad_with_value(x, target_n_cells, data_padding_value),
                batch_size=torch.Size([target_n_cells]),
            ),
            global_data=self.global_data,
        )

    def pad_to_next_power(
        self, power: float = 1.5, data_padding_value: float = 0.0
    ) -> "Mesh":
        """Pads points and cells arrays to their next power of `power` (integer-floored).

        This is useful for torch.compile with dynamic=False, where fixed tensor shapes
        are required. By padding to powers of a base (default 1.5), we can reuse compiled
        kernels across a reasonable range of mesh sizes while minimizing memory overhead.

        This method computes the target sizes as floor(power^n) for the smallest n such that
        the result is >= the current size, then calls .pad() to perform the actual padding.

        Args:
            power: Base for computing the next power. Must be > 1. Default is 1.5,
                which provides a good balance between memory efficiency and compile
                cache hits.

        Returns:
            A new Mesh with padded points and cells arrays. The padding uses
            null elements that don't affect geometric computations.

        Raises:
            ValueError: If power <= 1.

        Example:
            >>> mesh = Mesh(points, cells, "no_slip")  # 100 points, 200 cells
            >>> padded = mesh.pad_to_next_power(power=1.5)
            >>> # Points padded to floor(1.5^n) >= 100, cells to floor(1.5^m) >= 200
            >>> # For power=1.5: 100 points -> 129 points, 200 cells -> 216 cells
            >>> # Padding cells have zero area and don't affect computations
        """
        if not torch.compiler.is_compiling():
            if power <= 1:
                raise ValueError(f"power must be > 1, got {power=}")

        def next_power_size(current_size: int, base: float) -> int:
            """Calculate the next power of base (integer-floored) that is >= current_size."""
            if not torch.compiler.is_compiling():
                if current_size <= 1:
                    return 1
            # Solve for n: floor(base^n) >= current_size
            # n >= log(current_size) / log(base)
            n = (torch.tensor(current_size).log() / torch.tensor(base).log()).ceil()
            return int(torch.tensor(base) ** n)

        target_n_points = next_power_size(self.n_points, power)
        target_n_cells = next_power_size(self.n_cells, power)

        return self.pad(
            target_n_points=target_n_points,
            target_n_cells=target_n_cells,
            data_padding_value=data_padding_value,
        )

    def draw(
        self,
        backend: Literal["matplotlib", "pyvista", "auto"] = "auto",
        show: bool = True,
        point_scalars: None | torch.Tensor | str | tuple[str, ...] = None,
        cell_scalars: None | torch.Tensor | str | tuple[str, ...] = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        alpha_points: float = 1.0,
        alpha_cells: float = 1.0,
        alpha_edges: float = 1.0,
        show_edges: bool = True,
        ax=None,
        **kwargs,
    ):
        """Draw the mesh using matplotlib or PyVista backend.

        Provides interactive 3D or 2D visualization with support for scalar data
        coloring, transparency control, and automatic backend selection.

        Args:
            backend: Visualization backend to use:
                - "auto": Automatically select based on n_spatial_dims
                  (matplotlib for 0D/1D/2D, PyVista for 3D)
                - "matplotlib": Force matplotlib backend (supports 3D via mplot3d)
                - "pyvista": Force PyVista backend (requires n_spatial_dims <= 3)
            show: Whether to display the plot immediately (calls plt.show() or
                plotter.show()). If False, returns the plotter/axes for further
                customization before display.
            point_scalars: Scalar data to color points. Mutually exclusive with
                cell_scalars. Can be:
                - None: Points use neutral color (black)
                - torch.Tensor: Direct scalar values, shape (n_points,) or
                  (n_points, ...) where trailing dimensions are L2-normed
                - str or tuple[str, ...]: Key to lookup in mesh.point_data
            cell_scalars: Scalar data to color cells. Mutually exclusive with
                point_scalars. Can be:
                - None: Cells use neutral color (lightblue if no scalars,
                  lightgray if point_scalars active)
                - torch.Tensor: Direct scalar values, shape (n_cells,) or
                  (n_cells, ...) where trailing dimensions are L2-normed
                - str or tuple[str, ...]: Key to lookup in mesh.cell_data
            cmap: Colormap name for scalar visualization (default: "viridis")
            vmin: Minimum value for colormap normalization. If None, uses data min.
            vmax: Maximum value for colormap normalization. If None, uses data max.
            alpha_points: Opacity for points, range [0, 1] (default: 1.0)
            alpha_cells: Opacity for cells/faces, range [0, 1] (default: 0.3)
            alpha_edges: Opacity for cell edges, range [0, 1] (default: 0.7)
            show_edges: Whether to draw cell edges (default: True)
            ax: (matplotlib only) Existing matplotlib axes to plot on. If None,
                creates new figure and axes.
            **kwargs: Additional backend-specific keyword arguments

        Returns:
            - matplotlib backend: matplotlib.axes.Axes object
            - PyVista backend: pyvista.Plotter object

        Raises:
            ValueError: If both point_scalars and cell_scalars are specified,
                or if n_spatial_dims is not supported by the chosen backend.

        Example:
            >>> # Draw mesh with automatic backend selection
            >>> mesh.draw()
            >>>
            >>> # Color cells by pressure data
            >>> mesh.draw(cell_scalars="pressure", cmap="coolwarm")
            >>>
            >>> # Color points by velocity magnitude (computing norm of vector field)
            >>> mesh.draw(point_scalars="velocity")  # velocity is (n_points, 3)
            >>>
            >>> # Use nested TensorDict key
            >>> mesh.draw(cell_scalars=("flow", "temperature"))
            >>>
            >>> # Customize and display later
            >>> ax = mesh.draw(show=False, backend="matplotlib")
            >>> ax.set_title("My Mesh")
            >>> import matplotlib.pyplot as plt
            >>> plt.show()
        """
        from physicsnemo.mesh.visualization import draw_mesh

        return draw_mesh(
            mesh=self,
            backend=backend,
            show=show,
            point_scalars=point_scalars,
            cell_scalars=cell_scalars,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha_points=alpha_points,
            alpha_cells=alpha_cells,
            alpha_edges=alpha_edges,
            show_edges=show_edges,
            ax=ax,
            **kwargs,
        )

    def translate(self, offset: torch.Tensor | list | tuple) -> "Mesh":
        """Apply a translation (affine transformation) to the mesh.

        Convenience wrapper for physicsnemo.mesh.transformations.translate().
        See that function for detailed documentation.

        Args:
            offset: Translation vector, shape (n_spatial_dims,) or broadcastable

        Returns:
            New Mesh with translated geometry

        Example:
            >>> translated = mesh.translate([1.0, 2.0, 3.0])
        """
        from physicsnemo.mesh.transformations import translate

        return translate(self, offset)

    def rotate(
        self,
        axis: torch.Tensor | list | tuple | None,
        angle: float,
        center: torch.Tensor | list | tuple | None = None,
        transform_data: bool = False,
    ) -> "Mesh":
        """Rotate the mesh about an axis by a specified angle.

        Convenience wrapper for physicsnemo.mesh.transformations.rotate().
        See that function for detailed documentation.

        Args:
            axis: Rotation axis vector (ignored for 2D, required for 3D)
            angle: Rotation angle in radians
            center: Center point for rotation (optional)
            transform_data: If True, also rotate vector/tensor fields

        Returns:
            New Mesh with rotated geometry

        Example:
            >>> # Rotate 90 degrees about z-axis
            >>> import numpy as np
            >>> rotated = mesh.rotate([0, 0, 1], np.pi/2)
        """
        from physicsnemo.mesh.transformations import rotate

        return rotate(self, axis, angle, center, transform_data)

    def scale(
        self,
        factor: float | torch.Tensor | list | tuple,
        center: torch.Tensor | list | tuple | None = None,
        transform_data: bool = False,
    ) -> "Mesh":
        """Scale the mesh by specified factor(s).

        Convenience wrapper for physicsnemo.mesh.transformations.scale().
        See that function for detailed documentation.

        Args:
            factor: Scale factor (scalar) or factors (per-dimension)
            center: Center point for scaling (optional)
            transform_data: If True, also scale vector/tensor fields

        Returns:
            New Mesh with scaled geometry

        Example:
            >>> # Uniform scaling
            >>> scaled = mesh.scale(2.0)
            >>>
            >>> # Non-uniform scaling
            >>> scaled = mesh.scale([2.0, 1.0, 0.5])
        """
        from physicsnemo.mesh.transformations import scale

        return scale(self, factor, center, transform_data)

    def transform(
        self,
        matrix: torch.Tensor,
        transform_data: bool = False,
    ) -> "Mesh":
        """Apply a linear transformation to the mesh.

        Convenience wrapper for physicsnemo.mesh.transformations.transform().
        See that function for detailed documentation.

        Args:
            matrix: Transformation matrix, shape (new_n_spatial_dims, n_spatial_dims)
            transform_data: If True, also transform vector/tensor fields

        Returns:
            New Mesh with transformed geometry

        Example:
            >>> # Shear transformation
            >>> shear = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
            >>> sheared = mesh.transform(shear)
        """
        from physicsnemo.mesh.transformations import transform

        return transform(self, matrix, transform_data)

    def compute_point_derivatives(
        self,
        keys: str | tuple[str, ...] | list[str | tuple[str, ...]] | None = None,
        method: Literal["lsq", "dec"] = "lsq",
        gradient_type: Literal["intrinsic", "extrinsic", "both"] = "intrinsic",
    ) -> "Mesh":
        """Compute gradients of point_data fields.

        This is a convenience method that delegates to physicsnemo.mesh.calculus.compute_point_derivatives.

        Args:
            keys: Fields to compute gradients of. Options:
                - None: All non-cached fields (excludes "_cache" subdictionary)
                - str: Single field name (e.g., "pressure")
                - tuple: Nested path (e.g., ("flow", "temperature"))
                - list: Multiple fields (e.g., ["pressure", "velocity"])
            method: Discretization method:
                - "lsq": Weighted least-squares reconstruction (default, CFD standard)
                - "dec": Discrete Exterior Calculus (differential geometry)
            gradient_type: Type of gradient:
                - "intrinsic": Project onto manifold tangent space (default)
                - "extrinsic": Full ambient space gradient
                - "both": Compute and store both

        Returns:
            Self (mesh) with gradient fields added to point_data (modified in place).
            Field naming: "{field}_gradient" or "{field}_gradient_intrinsic/extrinsic"

        Example:
            >>> # Compute gradient of pressure
            >>> mesh_grad = mesh.compute_point_derivatives(keys="pressure")
            >>> grad_p = mesh_grad.point_data["pressure_gradient"]
            >>>
            >>> # Multiple fields with DEC method
            >>> mesh_grad = mesh.compute_point_derivatives(
            ...     keys=["pressure", "temperature"],
            ...     method="dec"
            ... )
        """
        from physicsnemo.mesh.calculus import compute_point_derivatives

        return compute_point_derivatives(
            mesh=self,
            keys=keys,
            method=method,
            gradient_type=gradient_type,
        )

    def compute_cell_derivatives(
        self,
        keys: str | tuple[str, ...] | list[str | tuple[str, ...]] | None = None,
        method: Literal["lsq", "dec"] = "lsq",
        gradient_type: Literal["intrinsic", "extrinsic", "both"] = "intrinsic",
    ) -> "Mesh":
        """Compute gradients of cell_data fields.

        This is a convenience method that delegates to physicsnemo.mesh.calculus.compute_cell_derivatives.

        Args:
            keys: Fields to compute gradients of (same format as compute_point_derivatives)
            method: "lsq" or "dec" (currently only "lsq" is fully supported for cells)
            gradient_type: "intrinsic", "extrinsic", or "both"

        Returns:
            Self (mesh) with gradient fields added to cell_data (modified in place)

        Example:
            >>> # Compute gradient of cell-centered pressure
            >>> mesh_grad = mesh.compute_cell_derivatives(keys="pressure")
        """
        from physicsnemo.mesh.calculus import compute_cell_derivatives

        return compute_cell_derivatives(
            mesh=self,
            keys=keys,
            method=method,
            gradient_type=gradient_type,
        )

    def validate(
        self,
        check_degenerate_cells: bool = True,
        check_duplicate_vertices: bool = True,
        check_inverted_cells: bool = False,
        check_out_of_bounds: bool = True,
        check_manifoldness: bool = False,
        tolerance: float = 1e-10,
        raise_on_error: bool = False,
    ):
        """Validate mesh integrity and detect common errors.

        Convenience method that delegates to physicsnemo.mesh.validation.validate_mesh.

        Args:
            check_degenerate_cells: Check for zero/negative area cells
            check_duplicate_vertices: Check for coincident vertices
            check_inverted_cells: Check for negative orientation
            check_out_of_bounds: Check cell indices are valid
            check_manifoldness: Check manifold topology (2D only)
            tolerance: Tolerance for geometric checks
            raise_on_error: Raise ValueError on first error vs return report

        Returns:
            Dictionary with validation results

        Example:
            >>> report = mesh.validate()
            >>> if not report["valid"]:
            >>>     print(f"Validation failed: {report}")
        """
        from physicsnemo.mesh.validation import validate_mesh

        return validate_mesh(
            mesh=self,
            check_degenerate_cells=check_degenerate_cells,
            check_duplicate_vertices=check_duplicate_vertices,
            check_inverted_cells=check_inverted_cells,
            check_out_of_bounds=check_out_of_bounds,
            check_manifoldness=check_manifoldness,
            tolerance=tolerance,
            raise_on_error=raise_on_error,
        )

    @property
    def quality_metrics(self):
        """Compute geometric quality metrics for all cells.

        Returns TensorDict with per-cell quality metrics:
        - aspect_ratio: max_edge / characteristic_length
        - edge_length_ratio: max_edge / min_edge
        - min_angle, max_angle: Interior angles (triangles only)
        - quality_score: Combined metric in [0,1] (1.0 is perfect)

        Example:
            >>> metrics = mesh.quality_metrics
            >>> poor_cells = metrics["quality_score"] < 0.3
            >>> print(f"Found {poor_cells.sum()} poor quality cells")
        """
        from physicsnemo.mesh.validation import compute_quality_metrics

        return compute_quality_metrics(self)

    @property
    def statistics(self):
        """Compute summary statistics for mesh.

        Returns dictionary with mesh statistics including counts,
        edge length distributions, area distributions, and quality metrics.

        Example:
            >>> stats = mesh.statistics
            >>> print(f"Mesh: {stats['n_points']} points, {stats['n_cells']} cells")
            >>> print(f"Edge lengths: min={stats['edge_length_stats'][0]:.3f}")
        """
        from physicsnemo.mesh.validation import compute_mesh_statistics

        return compute_mesh_statistics(self)

    def subdivide(
        self,
        levels: int = 1,
        filter: Literal["linear", "butterfly", "loop"] = "linear",
    ) -> "Mesh":
        """Subdivide the mesh using iterative application of subdivision schemes.

        Subdivision refines the mesh by splitting each n-simplex into 2^n child
        simplices. Multiple subdivision schemes are supported, each with different
        geometric and smoothness properties.

        This method applies the chosen subdivision scheme iteratively for the
        specified number of levels. Each level independently subdivides the
        current mesh.

        Args:
            levels: Number of subdivision iterations to perform. Each level
                increases mesh resolution exponentially:
                - 0: No subdivision (returns original mesh)
                - 1: Each cell splits into 2^n children
                - 2: Each cell splits into 4^n children
                - k: Each cell splits into (2^k)^n children
            filter: Subdivision scheme to use:
                - "linear": Simple midpoint subdivision (interpolating).
                  New vertices at exact edge midpoints. Works for any dimension.
                  Preserves original vertices.
                - "butterfly": Weighted stencil subdivision (interpolating).
                  New vertices use weighted neighbor stencils for smoother results.
                  Currently only supports 2D manifolds (triangular meshes).
                  Preserves original vertices.
                - "loop": Valence-based subdivision (approximating).
                  Both old and new vertices are repositioned for C² smoothness.
                  Currently only supports 2D manifolds (triangular meshes).
                  Original vertices move to new positions.

        Returns:
            Subdivided mesh with refined geometry and connectivity.
            - Manifold and spatial dimensions are preserved
            - Point data is interpolated to new vertices
            - Cell data is propagated from parents to children
            - Global data is preserved unchanged

        Raises:
            ValueError: If levels < 0
            ValueError: If filter is not one of the supported schemes
            NotImplementedError: If butterfly/loop filter used with non-2D manifold

        Example:
            >>> # Linear subdivision of triangular mesh
            >>> mesh = create_triangle_mesh()
            >>> refined = mesh.subdivide(levels=2, filter="linear")
            >>> # Each triangle splits into 4, twice: 2 -> 8 -> 32 triangles
            >>>
            >>> # Smooth subdivision with Loop scheme
            >>> smooth = mesh.subdivide(levels=3, filter="loop")
            >>> # Produces smooth limit surface after 3 iterations
            >>>
            >>> # Butterfly for interpolating smooth subdivision
            >>> butterfly = mesh.subdivide(levels=1, filter="butterfly")
            >>> # Smoother than linear, preserves original vertices

        Note:
            Multi-level subdivision is achieved by iterative application.
            For levels=3, this is equivalent to:
            ```python
            mesh = mesh.subdivide(levels=1, filter=filter)
            mesh = mesh.subdivide(levels=1, filter=filter)
            mesh = mesh.subdivide(levels=1, filter=filter)
            ```
            This is the standard approach for all subdivision schemes.
        """
        from physicsnemo.mesh.subdivision import (
            subdivide_butterfly,
            subdivide_linear,
            subdivide_loop,
        )

        ### Validate inputs
        if levels < 0:
            raise ValueError(f"levels must be >= 0, got {levels=}")

        ### Apply subdivision iteratively
        mesh = self
        for _ in range(levels):
            if filter == "linear":
                mesh = subdivide_linear(mesh)
            elif filter == "butterfly":
                mesh = subdivide_butterfly(mesh)
            elif filter == "loop":
                mesh = subdivide_loop(mesh)
            else:
                raise ValueError(
                    f"Invalid {filter=}. Must be one of: 'linear', 'butterfly', 'loop'"
                )

        return mesh

    def clean(
        self,
        rtol: float = 1e-12,
        atol: float = 1e-12,
        merge_points: bool = True,
        remove_duplicate_cells: bool = True,
        remove_unused_points: bool = True,
    ) -> "Mesh":
        """Clean and repair this mesh.

        Performs various cleaning operations to fix common mesh issues:
        1. Merge duplicate points within tolerance
        2. Remove duplicate cells
        3. Remove unused points

        This is useful after mesh operations that may introduce duplicate geometry
        or after importing meshes from external sources that may have redundant data.

        Args:
            rtol: Relative tolerance for merging points (default 1e-12).
                Points p1 and p2 are merged if ||p1 - p2|| <= atol + rtol * ||p1||
            atol: Absolute tolerance for merging points (default 1e-12)
            merge_points: Whether to merge duplicate points (default True)
            remove_duplicate_cells: Whether to remove duplicate cells (default True)
            remove_unused_points: Whether to remove unused points (default True)

        Returns:
            Cleaned mesh with same structure but repaired topology

        Example:
            >>> # Mesh with duplicate points
            >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 0.], [1., 1.]])
            >>> cells = torch.tensor([[0, 1, 3], [2, 1, 3]])
            >>> mesh = Mesh(points=points, cells=cells)
            >>> cleaned = mesh.clean()
            >>> cleaned.n_points  # 3 (points 0 and 2 merged)
            >>>
            >>> # Adjust tolerance for coarser merging
            >>> mesh_loose = mesh.clean(rtol=1e-6, atol=1e-6)
            >>>
            >>> # Only merge points, keep duplicate cells
            >>> mesh_partial = mesh.clean(
            ...     merge_points=True,
            ...     remove_duplicate_cells=False
            ... )
        """
        from physicsnemo.mesh.boundaries import clean_mesh

        return clean_mesh(
            mesh=self,
            rtol=rtol,
            atol=atol,
            merge_points=merge_points,
            remove_duplicate_cells_flag=remove_duplicate_cells,
            remove_unused_points_flag=remove_unused_points,
        )


### Override the tensorclass __repr__ with custom formatting
# Note: Must be done after class definition because @tensorclass overrides __repr__
# even when defined inside the class body
def _mesh_repr(self) -> str:
    from physicsnemo.mesh.utilities.mesh_repr import format_mesh_repr

    return format_mesh_repr(self, exclude_cache=False)


Mesh.__repr__ = _mesh_repr  # type: ignore
