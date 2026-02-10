from dataclasses import dataclass, fields
from pathlib import Path
from textwrap import indent
from typing import Any, Literal, Sequence
import warnings


import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from physicsnemo.models.globe.utilities.input_validation import check_leaf_tensors


@dataclass
class BoundaryMesh:
    """A representation of a boundary mesh for computational fluid dynamics simulations.

    This class encapsulates the geometric and topological information of a boundary mesh,
    including points, connectivity (faces), and associated boundary condition types. It
    supports both 2D (line segments) and 3D (triangular faces) meshes and provides
    efficient computation of geometric properties like face centers, normals, and areas.

    The class is designed to work seamlessly with PyTorch tensors for GPU acceleration
    and integrates with PyVista for mesh I/O operations. Boundary meshes can be combined
    using the addition operator, provided they have compatible dimensions and boundary
    condition types.

    Attributes:
        points: Tensor of shape (n_points, n_spatial_dimensions) containing vertex coordinates.
            For 2D meshes, n_spatial_dimensions=2; for 3D meshes, n_spatial_dimensions=3.
        faces: Tensor of shape (n_faces, n_vertices_per_face) containing vertex indices
            that define the connectivity. For 2D meshes, each face is a line segment with
            2 vertices. For 3D meshes, each face is a triangle with 3 vertices.
        boundary_condition_type: String identifier for the boundary condition applied
            to this mesh (e.g., "no_slip", "slip", "inflow", "outflow").
        face_data: TensorDict of shape (n_faces, ...) containing per-face data.

    Examples:
        Creating a 2D circular boundary:

        >>> theta = torch.linspace(0, 2*torch.pi, 101)
        >>> points = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        >>> faces = torch.stack([torch.arange(100), torch.arange(1, 101)], dim=1)
        >>> boundary = BoundaryMesh(points, faces, "no_slip")
        >>> print(boundary.n_spatial_dimensions)  # 2
        >>> print(boundary.face_centers.shape)  # torch.Size([100, 2])

        Loading from PyVista:

        >>> import pyvista as pv
        >>> mesh_data = pv.examples.load_airplane()
        >>> boundary = BoundaryMesh.from_polydata(mesh_data, "no_slip")
        >>> normals = boundary.face_normals  # Automatically computed

        Combining boundaries:

        >>> wing_boundary = BoundaryMesh.from_polydata(wing_mesh, "no_slip")
        >>> fuselage_boundary = BoundaryMesh.from_polydata(fuselage_mesh, "no_slip")
        >>> aircraft_boundary = wing_boundary + fuselage_boundary

    Note:
        - The class is immutable (frozen dataclass) to ensure data integrity
        - Geometric properties are cached for performance using @cached_property
        - Face normals follow the right-hand rule convention
        - For 2D meshes, normals point to the left of the face direction vector
        - All tensors should be on the same device for proper operation
    """

    points: torch.Tensor  # shape: (n_points, n_spatial_dimensions)
    faces: torch.Tensor  # shape: (n_faces, n_spatial_dimensions); leads to lines in 2D and triangles in 3D
    boundary_condition_type: str
    face_data: TensorDict = None  # ty: ignore[invalid-assignment] # initialized in __post_init__
    face_centers: torch.Tensor = None  # ty: ignore[invalid-assignment]
    face_normals: torch.Tensor = None  # ty: ignore[invalid-assignment]
    face_areas: torch.Tensor = None  # ty: ignore[invalid-assignment]

    def __post_init__(self) -> None:
        """Post-initialization processing to compute geometric properties and validate inputs.

        This method is automatically called after dataclass initialization. It performs three
        main tasks:

        1. **Initialize face_data**: If not provided, creates an empty TensorDict with the
           correct batch size and device.

        2. **Compute geometric properties**: Calculates face centers, normals, and areas if
           not already provided:
           - **Face centers**: Computed as the mean of vertex positions for each face
           - **Face normals** (2D): Perpendicular to face direction, pointing to the left
             (90° counterclockwise rotation). Normalized to unit length.
           - **Face normals** (3D): Computed via cross product (p2-p1) × (p3-p1), following
             right-hand rule. Normalized to unit length.
           - **Face areas** (2D): Length of the line segment
           - **Face areas** (3D): Half the magnitude of the cross product (standard triangle area)

        3. **Validate inputs**: Calls validate() to check consistency of data types, devices,
           and dimensions (skipped during torch.compile for performance).

        The method is idempotent - if geometric properties are already provided, they are
        not recomputed. This allows efficient construction from pre-computed data.

        Raises:
            NotImplementedError: If n_spatial_dims is not 2 or 3.
            TypeError, ValueError: Various validation errors from validate() method.

        Note:
            For 2D meshes, normals point to the left of the face direction vector. This
            convention ensures consistent orientation for closed curves (e.g., airfoil
            boundaries) where the interior is to the left.
        """
        # Initialize face_data if it is not provided
        if self.face_data is None:
            self.face_data = TensorDict(
                {}, batch_size=torch.Size([self.n_faces]), device=self.points.device
            )

        # Computes face_centers, face_normals, and face_areas if they are not provided
        if self.face_centers is None:
            self.face_centers = self.points[self.faces].mean(dim=1)
        if (self.face_normals is None) or (self.face_areas is None):
            if self.n_spatial_dims == 2:
                p1 = self.points[self.faces[:, 0]]
                p2 = self.points[self.faces[:, 1]]
                face_vectors = p2 - p1
                if self.face_normals is None:
                    face_tangent_directions = F.normalize(
                        face_vectors, dim=-1, eps=1e-30
                    )
                    self.face_normals = torch.stack(
                        [-face_tangent_directions[:, 1], face_tangent_directions[:, 0]],
                        dim=-1,
                    )
                if self.face_areas is None:
                    self.face_areas = face_vectors.norm(dim=-1)
            elif self.n_spatial_dims == 3:
                p1 = self.points[self.faces[:, 0]]
                p2 = self.points[self.faces[:, 1]]
                p3 = self.points[self.faces[:, 2]]
                cross_products = torch.cross(p2 - p1, p3 - p1, dim=-1)
                if self.face_normals is None:
                    self.face_normals = F.normalize(cross_products, dim=-1, eps=1e-30)
                if self.face_areas is None:
                    self.face_areas = cross_products.norm(dim=-1) / 2.0
            else:
                raise NotImplementedError(f"{self.n_spatial_dims=} not supported!")

        if not torch.compiler.is_compiling():
            self.validate()

    def validate(self) -> None:
        """
        Performs a variety of validation checks on attributes of the BoundaryMesh instance.

        Raises (ValueError, TypeError, NotImplementedError): If the inputs are invalid.
        """
        ### Ensure that faces has an int-like dtype
        if torch.is_floating_point(self.faces):
            raise TypeError(
                f"`faces` must have an int-like dtype, but got {self.faces.dtype=}."
            )

        ### Ensure all tensors are on the same device
        check_leaf_tensors(
            value=self.__dict__,
            name=self.__class__.__name__,
            func=lambda x: x.device,
        )

    def __repr__(self) -> str:
        """Returns a concise string representation of the BoundaryMesh.

        The representation includes key mesh properties and face data summary. Format:
        ```
        BoundaryMesh(n_points=..., n_faces=..., n_spatial_dims=..., boundary_condition_type='...')
            face_data=TensorDict(...)
        ```

        Returns:
            String representation showing mesh dimensions, BC type, and face data summary.
        """
        instance_fields = [
            f"n_points={self.n_points}",
            f"n_faces={self.n_faces}",
            f"n_spatial_dims={self.n_spatial_dims}",
            f"boundary_condition_type='{self.boundary_condition_type}'",
        ]

        return f"{self.__class__.__name__}({', '.join(instance_fields)})\n" + indent(
            f"face_data={self.face_data}", prefix="\t"
        )

    def __eq__(self, other: Any) -> bool:
        """Check equality by comparing all dataclass fields."""
        if type(self) is not type(other):
            return False

        for field in fields(self):
            a, b = getattr(self, field.name), getattr(other, field.name)
            if type(a) is not type(b):
                return False
            if isinstance(a, torch.Tensor):
                if not torch.equal(a, b):
                    return False
            elif isinstance(a, TensorDict):
                if not (a == b).all():
                    return False
            elif a != b:
                return False
        return True

    @classmethod
    def from_polydata(
        cls,
        polydata: pv.PolyData,
        boundary_condition_type: str,
        device: torch.device | str | None = None,
        include_cell_data: bool = True,
        make_2D_along_axis: Literal["x", "y", "z"] | None = None,
    ) -> "BoundaryMesh":
        """
        Construct a BoundaryMesh from a PyVista PolyData object.

        This method extracts mesh points, faces, and any associated cell data
        from a PyVista PolyData object and constructs a BoundaryMesh instance.
        Cell data arrays with 1D shape are interpreted as scalar face data,
        while 2D arrays are interpreted as vector face data. All tensors are
        moved to the specified device.

        Notably, PyVista PolyData objects are always 3D, even if they represent
        a 2D boundary. In order to determine whether to construct a 2D or 3D
        BoundaryMesh, this method checks for the presence of faces and lines. If
        only faces are present, it assumes the mesh is 3D. If only lines are
        present, it assumes the mesh is 2D. If both are present, we follow the
        3D case and discard the lines, while emitting a warning.

        Notably, if your mesh is 2D, you should provide the `make_2D_along_axis`
        argument, which is used to project the mesh points onto an axis-aligned
        plane.

        Args:
            polydata: pv.PolyData
                The PyVista PolyData object representing the mesh. Must have
                `.points` and `.regular_faces` attributes, and may have cell
                data.
            boundary_condition_type: str
                The boundary condition type to assign to this mesh.
            device: torch.device | str | None, default=None
                The device on which to place the tensors. If None, defaults to
                "cpu".
            make_2D_along_axis: Literal["x", "y", "z"] | None, default=None
                Only relevant if the mesh is in 2D, not 3D. In this case, this
                argument is required, and the mesh points will be converted to
                2D by projecting along the specified axis.

        Returns:
            BoundaryMesh
                The constructed BoundaryMesh instance.

        Raises:
            ValueError: If cell data arrays are not 1D or 2D.

        Example:
            >>> import pyvista as pv
            >>> sphere = pv.Sphere()
            >>> sphere.cell_data["pressure"] = np.random.rand(sphere.n_cells)
            >>> sphere.cell_data["velocity"] = np.random.rand(sphere.n_cells, 3)
            >>> bm = BoundaryMesh.from_polydata(sphere, boundary_condition_type="no_slip")
        """
        if device is None:
            device = "cpu"

        # Determine whether the mesh is 2D or 3D. 3D meshes have only faces, and
        # 2D meshes have only lines.
        has_faces = len(polydata.faces) > 0
        has_lines = len(polydata.lines) > 0

        if has_faces:
            if has_lines:
                warnings.warn(
                    "Both faces and lines are present in the mesh, which makes it ambiguous whether the boundary is 2D or 3D.\n"
                    "Assuming 3D, and discarding lines."
                )
            else:
                n_spatial_dims = 3
        else:
            if has_lines:
                n_spatial_dims = 2
            else:
                raise ValueError(
                    "This mesh has no faces or lines, and therefore is not a valid boundary mesh."
                )

        if n_spatial_dims == 3:
            if not polydata.is_all_triangles:
                raise ValueError("In 3D, all faces must be triangles.")
            if make_2D_along_axis is not None:
                raise ValueError(
                    "In 3D, `make_2D_along_axis` is not allowed, and should be `None`."
                )
            faces = torch.tensor(polydata.regular_faces, device=device)

            def project_vector(vector: np.ndarray) -> np.ndarray:
                return vector

        elif n_spatial_dims == 2:
            if make_2D_along_axis is None:
                raise ValueError("In 2D, `make_2D_along_axis` is required.")

            def project_vector(vector: np.ndarray) -> np.ndarray:
                axis_indices = {
                    "x": [1, 2],
                    "y": [0, 2],
                    "z": [0, 1],
                }
                if make_2D_along_axis not in axis_indices:
                    raise ValueError(
                        f"Expected `make_2D_along_axis` to be one of {list(axis_indices.keys())}, got {make_2D_along_axis=!r}."
                    )
                return vector[:, axis_indices[make_2D_along_axis]]

            if not (np.all(polydata.lines[::3] == 2) and len(polydata.lines) % 3 == 0):
                raise ValueError("In 2D, all lines must have exactly 2 vertices.")
            regular_lines = polydata.lines.reshape(-1, 3)[:, [1, 2]]
            faces = torch.tensor(regular_lines, device=device)

        else:
            raise ValueError(f"Expected 2D or 3D mesh, got {n_spatial_dims=}.")

        face_data = TensorDict(
            {
                k: torch.tensor(v if v.ndim == 1 else project_vector(v), device=device)
                for k, v in polydata.cell_data.items()
            }
            if include_cell_data
            else {},
            batch_size=torch.Size([len(faces)]),
            device=device,
        )

        return cls(
            points=torch.tensor(
                project_vector(polydata.points), device=device
            ).contiguous(),
            faces=faces.contiguous(),
            boundary_condition_type=boundary_condition_type,
            face_data=face_data.contiguous(),
        )

    def to_polydata(self) -> pv.PolyData:
        """Converts the BoundaryMesh to a PyVista PolyData object.

        This method creates a PyVista PolyData representation of the mesh, which is useful
        for visualization, I/O operations, or interfacing with other tools in the PyVista
        ecosystem. Face data from the TensorDict is transferred to PyVista's cell_data.

        Returns:
            pv.PolyData: PyVista mesh object with points, faces, and cell data.

        Raises:
            ValueError: If the mesh is 2D (PyVista only supports 3D meshes).

        Note:
            - Points and faces are converted to NumPy arrays and copied to CPU if needed
            - All face_data entries are transferred as cell_data arrays
            - The boundary_condition_type is not stored in the PolyData object

        Example:
            >>> mesh_3d = BoundaryMesh.from_polydata(wing_surface, "no_slip")
            >>> polydata = mesh_3d.to_polydata()
            >>> polydata.plot(show_edges=True)  # Visualize with PyVista
        """
        if self.n_spatial_dims != 3:
            raise ValueError(
                f"Only 3D meshes can be converted to PyVista meshes. Got {self.n_spatial_dims=}."
            )
        mesh = pv.PolyData(
            np.asarray(self.points),
        )
        mesh.regular_faces = np.asarray(self.faces)
        for k, v in self.face_data.items():
            mesh.cell_data[k] = np.asarray(v)
        return mesh

    @classmethod
    def merge(cls, boundary_meshes: Sequence["BoundaryMesh"]) -> "BoundaryMesh":
        """Merges multiple BoundaryMesh objects into a single unified mesh.

        This class method combines several boundary meshes with the same boundary condition
        type and dimensionality into one mesh. Points and faces are concatenated, with face
        indices automatically adjusted to reference the correct points in the merged point array.
        All geometric properties (centers, normals, areas) are also concatenated.

        Args:
            boundary_meshes: Sequence of BoundaryMesh objects to merge. Must contain at
                least one mesh. All meshes must have:
                - The same n_spatial_dims (2 or 3)
                - The same boundary_condition_type
                - The same face_data keys (though values can differ)

        Returns:
            BoundaryMesh: A new merged mesh containing all points and faces from the
                input meshes. The boundary_condition_type is taken from the first mesh.

        Raises:
            ValueError: If boundary_meshes is empty, if spatial dimensions don't match
                across meshes, if boundary condition types don't match, or if face_data
                keys don't match.
            TypeError: If any element of boundary_meshes is not a BoundaryMesh instance.

        Note:
            - The merge is a lightweight operation - it doesn't copy geometric computations
            - Face indices are automatically offset to account for concatenated point arrays
            - If only one mesh is provided, it is returned directly without copying
            - The implementation uses torch.cat for efficient tensor concatenation

        Example:
            >>> # Merge multiple no-slip boundaries into one
            >>> wing = BoundaryMesh.from_polydata(wing_surface, "no_slip")
            >>> fuselage = BoundaryMesh.from_polydata(fuselage_surface, "no_slip")
            >>> tail = BoundaryMesh.from_polydata(tail_surface, "no_slip")
            >>> aircraft = BoundaryMesh.merge([wing, fuselage, tail])
            >>> print(aircraft.n_faces)  # Sum of individual face counts
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if len(boundary_meshes) == 0:
                raise ValueError("At least one BoundaryMesh must be provided to merge.")
            elif len(boundary_meshes) == 1:  # Short-circuit for speed in this case
                return boundary_meshes[0]
            if not all(isinstance(bm, BoundaryMesh) for bm in boundary_meshes):
                raise TypeError(
                    f"All objects must be BoundaryMesh types. Got:\n"
                    f"{[type(bm) for bm in boundary_meshes]=}"
                )
            # Check dimensional consistency across all meshes
            validations = {
                "spatial dimensions": [bm.n_spatial_dims for bm in boundary_meshes],
                "boundary condition types": [
                    bm.boundary_condition_type for bm in boundary_meshes
                ],
            }
            for name, values in validations.items():
                if not all(v == values[0] for v in values):
                    raise ValueError(
                        f"All meshes must have the same {name}. Got:\n{values=}"
                    )
            # Check that all face_data dicts have the same keys across all meshes
            if not all(
                bm.face_data.keys() == boundary_meshes[0].face_data.keys()
                for bm in boundary_meshes
            ):
                raise ValueError("All meshes must have the same face_data keys.")

        ### Merge the boundary meshes

        # Compute the number of points for each mesh, cumulatively, so that we can update
        # the point indices for the constituent faces arrays accordingly.
        n_points_for_meshes = torch.tensor(
            [bm.n_points for bm in boundary_meshes],
            device=boundary_meshes[0].points.device,
        )
        cumsum_n_points = torch.cumsum(n_points_for_meshes, dim=0)
        face_index_offsets = cumsum_n_points.roll(1)
        face_index_offsets[0] = 0

        return cls(
            points=torch.cat([bm.points for bm in boundary_meshes], dim=0),
            faces=torch.cat(
                [
                    bm.faces + offset
                    for bm, offset in zip(boundary_meshes, face_index_offsets)
                ],
                dim=0,
            ),
            boundary_condition_type=boundary_meshes[0].boundary_condition_type,
            face_data=TensorDict.cat([bm.face_data for bm in boundary_meshes], dim=0),
            face_centers=torch.cat([bm.face_centers for bm in boundary_meshes], dim=0),
            face_normals=torch.cat([bm.face_normals for bm in boundary_meshes], dim=0),
            face_areas=torch.cat([bm.face_areas for bm in boundary_meshes], dim=0),
        )

    @property
    def n_points(self) -> int:
        """Number of vertices in the mesh.

        Returns:
            int: The total count of points, equal to len(self.points).
        """
        return len(self.points)

    @property
    def n_faces(self) -> int:
        """Number of faces in the mesh.

        Returns:
            int: The total count of faces. In 2D, faces are line segments.
                In 3D, faces are triangles.
        """
        return len(self.faces)

    @property
    def n_spatial_dims(self) -> Literal[2, 3]:
        """Number of spatial dimensions of the mesh (2 or 3).

        This property is inferred from the last dimension of the points tensor and
        validated against the faces tensor shape. The validation ensures consistency:
        - 2D meshes have points of shape (n_points, 2) and faces of shape (n_faces, 2)
        - 3D meshes have points of shape (n_points, 3) and faces of shape (n_faces, 3)

        Returns:
            Literal[2, 3]: The number of spatial dimensions.

        Raises:
            ValueError: If points and faces tensors indicate different dimensionalities.
            NotImplementedError: If dimensionality is neither 2 nor 3.

        Note:
            In 2D, faces represent line segments (edges) with 2 vertices each.
            In 3D, faces represent triangles with 3 vertices each.
            The validation is skipped during torch.compile for performance.
        """
        dims_from_points = self.points.shape[-1]

        if (
            not torch.compiler.is_compiling()
        ):  # Skip validation when running under torch.compile for performance
            dims_from_faces = self.faces.shape[-1]
            if dims_from_points != dims_from_faces:
                raise ValueError(
                    "The `points` and `faces` tensors indicate differing numbers of spatial dimensions.\n"
                    f"Got {dims_from_points=} and {dims_from_faces=}.\n"
                    "Recall that `faces` are triangles in 3D and line segments in 2D."
                )
            if dims_from_points not in [2, 3]:
                raise NotImplementedError(
                    f"Got {dims_from_points=}; currently {type(self).__name__!r} only supports 2D and 3D meshes."
                )

        return dims_from_points

    def to(
        self, device: torch.device | str, dtype: torch.dtype | None = None
    ) -> "BoundaryMesh":
        """Moves all tensors of this BoundaryMesh to the specified device in-place.

        Despite BoundaryMesh being a frozen dataclass, this method modifies the instance
        in-place by directly updating the __dict__. This design preserves cached properties
        and avoids expensive recomputation of geometric quantities like face normals and
        areas when moving between devices.

        Args:
            device: Target device (e.g., 'cuda', 'cpu', torch.device('cuda:0'))
            dtype: Target dtype (e.g., torch.float32, torch.float64)

        Returns:
            self: The same BoundaryMesh instance with all tensors moved to the target device.

        Example:
            >>> boundary = BoundaryMesh(points, faces, "no_slip")
            >>> boundary.to('cuda')  # Modifies boundary in-place
            >>> boundary.points.device  # cuda:0
            >>> boundary.face_normals.device  # cuda:0 (cached property preserved)

        Note:
            This operation modifies the instance in-place, bypassing the frozen dataclass
            restriction to avoid recomputing expensive cached geometric properties.
        """

        def transfer(data: Any) -> Any:
            if isinstance(data, (torch.Tensor | TensorDict)):
                return data.to(device, dtype=dtype)
            elif isinstance(data, list):
                return [transfer(item) for item in data]
            elif isinstance(data, tuple):
                return tuple(transfer(item) for item in data)
            elif isinstance(data, set):
                return {transfer(item) for item in data}
            elif isinstance(data, dict):
                return {transfer(k): transfer(v) for k, v in data.items()}
            else:
                return data

        for k, v in self.__dict__.items():
            self.__dict__[k] = transfer(v)

        return self

    def slice_faces(self, indices: int | slice | torch.Tensor) -> "BoundaryMesh":
        """Returns a new BoundaryMesh with a subset of the faces.

        Creates a view or copy of the mesh containing only the specified faces. The points
        array is shared (not copied), but faces and all face-associated data (centers,
        normals, areas, face_data) are sliced. This is useful for:
        - Downsampling meshes for faster computation
        - Extracting regions of interest
        - Creating test subsets
        - Filtering faces based on some criterion

        Args:
            indices: Face selection specification, can be:
                - int: Single face index
                - slice: Python slice object (e.g., slice(0, 100, 2))
                - torch.Tensor: Boolean mask of shape (n_faces,) or integer indices

        Returns:
            BoundaryMesh: New mesh with selected faces. Points array is shared with the
                original mesh. Face indices reference the same point array.

        Note:
            - The returned mesh shares the points tensor with the original, so modifications
              to points will affect both meshes
            - Face indices are not remapped - they still reference the original point indices
            - This means some points in the shared array may be unreferenced in the sliced mesh
            - If you need an independent mesh, explicitly copy the returned mesh's tensors

        Example:
            >>> # Select every other face
            >>> downsampled = mesh.slice_faces(slice(None, None, 2))
            >>> # Select faces with high pressure
            >>> pressure = mesh.face_data["pressure"]
            >>> high_pressure_mask = pressure > threshold
            >>> high_p_mesh = mesh.slice_faces(high_pressure_mask)
            >>> # Select first 100 faces
            >>> subset = mesh.slice_faces(torch.arange(100))
        """
        return BoundaryMesh(
            points=self.points,
            faces=self.faces[indices],
            boundary_condition_type=self.boundary_condition_type,
            face_data=self.face_data[indices],  # type: ignore
            face_centers=self.face_centers[indices],
            face_normals=self.face_normals[indices],
            face_areas=self.face_areas[indices],
        )

    def sample_random_points_on_faces(self, alpha: float = 1.0) -> torch.Tensor:
        """Sample random points uniformly distributed on each face of the mesh.

        Uses a Dirichlet distribution to generate barycentric coordinates, which are
        then used to compute random points as weighted combinations of face vertices.
        The concentration parameter alpha controls the distribution of samples within
        each face (simplex).

        Args:
            alpha: Concentration parameter for the Dirichlet distribution. Controls how
                samples are distributed within each face:
                - alpha = 1.0: Uniform distribution over the simplex (default)
                - alpha > 1.0: Concentrates samples toward the center of each face
                - alpha < 1.0: Concentrates samples toward vertices and edges

        Returns:
            Random points on faces, shape (n_faces, n_spatial_dims). Each point lies
            within its corresponding face.

        Example:
            >>> # Generate random points uniformly distributed on faces
            >>> random_centers = mesh.sample_random_points_on_faces(alpha=1.0)
            >>> # Generate points concentrated toward face centers
            >>> centered_points = mesh.sample_random_points_on_faces(alpha=3.0)
        """
        if alpha != 1.0:
            raise NotImplementedError(
                (
                    "Only alpha=1.0 (exponential distribution) is currently supported.\n"
                    "PyTorch does not yet support sampling from a gamma distribution on CUDA 12 backends\n"
                    "when using torch.compile. See https://github.com/pytorch/pytorch/issues/165751."
                )
            )
        # Sample from Gamma(alpha, 1) distribution and normalize to get Dirichlet
        gamma_dist = torch.distributions.Exponential(
            # concentration=torch.tensor(alpha, device=self.points.device),  # TODO uncomment this when above NotImplementedError is resolved
            rate=torch.tensor(1.0, device=self.points.device),
        )
        gamma_samples = gamma_dist.sample((self.n_faces, self.n_spatial_dims))

        # Normalize to get barycentric coordinates
        barycentric_coords = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)

        # Compute weighted combination of face vertices
        return (barycentric_coords.unsqueeze(-1) * self.points[self.faces]).sum(dim=1)

    def pad(
        self,
        target_n_points: int | None = None,
        target_n_faces: int | None = None,
    ) -> "BoundaryMesh":
        """Pad points and faces arrays to specified sizes.

        This is the low-level padding method that performs the actual padding operation.
        Padding uses null/degenerate elements that don't affect computations:
        - Points: Additional points at the last existing point (preserves bounding box)
        - Faces: Degenerate faces with all vertices at the last existing point (zero area)
        - Face data: Zero-valued padding for all face data fields

        Args:
            target_n_points: Target number of points. If None, no point padding is applied.
                Must be >= current n_points if specified.
            target_n_faces: Target number of faces. If None, no face padding is applied.
                Must be >= current n_faces if specified.

        Returns:
            A new BoundaryMesh with padded arrays. If both targets are None or equal to
            current sizes, returns self unchanged.

        Raises:
            ValueError: If target sizes are less than current sizes.

        Example:
            >>> mesh = BoundaryMesh(points, faces, "no_slip")  # 100 points, 200 faces
            >>> padded = mesh.pad(target_n_points=128, target_n_faces=256)
            >>> padded.n_points  # 128
            >>> padded.n_faces   # 256
        """

        def _pad_by_tiling_last(tensor: torch.Tensor, size: int) -> torch.Tensor:
            return torch.cat(
                [tensor, torch.tile(tensor[-1:], (size - tensor.shape[0], 1))],
                dim=0,
            )

        def _pad_with_zeros(tensor: torch.Tensor, size: int) -> torch.Tensor:
            return torch.cat(
                [
                    tensor,
                    torch.zeros(
                        (size - tensor.shape[0], *tensor.shape[1:]),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    ),
                ],
                dim=0,
            )

        # Validate inputs
        if not torch.compiler.is_compiling():
            if target_n_points is not None and target_n_points < self.n_points:
                raise ValueError(f"{target_n_points=} must be >= {self.n_points=}")
            if target_n_faces is not None and target_n_faces < self.n_faces:
                raise ValueError(f"{target_n_faces=} must be >= {self.n_faces=}")

        # Short-circuit if no padding needed
        if target_n_points is None and target_n_faces is None:
            return self

        # Determine actual target sizes
        if target_n_points is None:
            target_n_points = self.n_points
        if target_n_faces is None:
            target_n_faces = self.n_faces

        return BoundaryMesh(
            points=_pad_by_tiling_last(self.points, target_n_points),
            faces=torch.cat(
                [
                    self.faces,
                    torch.full(
                        (target_n_faces - self.n_faces, self.n_spatial_dims),
                        fill_value=self.n_points - 1,
                        device=self.faces.device,
                    ),
                ],
                dim=0,
            ),
            boundary_condition_type=self.boundary_condition_type,
            face_data=TensorDict(
                {
                    k: _pad_with_zeros(v, target_n_faces)
                    for k, v in self.face_data.items()
                },
                batch_size=torch.Size([target_n_faces]),
                device=self.face_data.device,
            ),
            face_centers=_pad_by_tiling_last(self.face_centers, target_n_faces),
            face_normals=_pad_by_tiling_last(self.face_normals, target_n_faces),
            face_areas=_pad_with_zeros(self.face_areas, target_n_faces),
        )

    def pad_to_next_power(self, power: float = 1.5) -> "BoundaryMesh":
        """Pads points and faces arrays to their next power of `power` (integer-floored).

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
            A new BoundaryMesh with padded points and faces arrays. The padding uses
            null elements that don't affect geometric computations.

        Raises:
            ValueError: If power <= 1.

        Example:
            >>> mesh = BoundaryMesh(points, faces, "no_slip")  # 100 points, 200 faces
            >>> padded = mesh.pad_to_next_power(power=1.5)
            >>> # Points padded to floor(1.5^n) >= 100, faces to floor(1.5^m) >= 200
            >>> # For power=1.5: 100 points -> 129 points, 200 faces -> 216 faces
            >>> # Padding faces have zero area and don't affect computations
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
        target_n_faces = next_power_size(self.n_faces, power)

        return self.pad(target_n_points=target_n_points, target_n_faces=target_n_faces)


if __name__ == "__main__":
    ### 3D Mesh
    airplane_surface: pv.PolyData = pv.examples.load_airplane()
    # airplane_surface.plot(show_edges=True, show_bounds=True)
    b3 = BoundaryMesh.from_polydata(airplane_surface, boundary_condition_type="no_slip")
    print(b3.face_centers)
    print(b3.face_normals)
    print(b3.face_areas)

    ### 2D Mesh
    theta = torch.linspace(0, 2 * torch.pi, 361)
    points = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    start_indices = torch.arange(len(theta))
    end_indices = torch.roll(start_indices, shifts=-1)
    faces = torch.stack(
        [
            start_indices,
            end_indices,
        ],
        dim=1,
    )
    b2 = BoundaryMesh(
        points=points,
        faces=faces,
        boundary_condition_type="no_slip",
    )

    demo_path = Path("demo_airplane.boundarymesh")

    torch.save(b3, demo_path)

    b3_loaded = torch.load(demo_path, weights_only=False)

    print(
        "Loaded mesh matches originals points: ",
        torch.allclose(b3.points, b3_loaded.points),
    )
    print("Loaded mesh n_faces: ", b3_loaded.n_faces)
