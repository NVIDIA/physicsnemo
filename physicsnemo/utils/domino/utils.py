# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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
Utilities for data processing and training with the DoMINO model architecture.

This module provides essential utilities for computational fluid dynamics data processing,
mesh manipulation, field normalization, and geometric computations. It supports both
CPU (NumPy) and GPU (CuPy) operations with automatic fallbacks.
"""

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import vtk
from scipy.spatial import KDTree
from vtk import vtkDataSetTriangleFilter
from vtk.util import numpy_support

from physicsnemo.utils.profiling import profile

# Type alias for arrays that can be either NumPy or CuPy

try:
    import cupy as cp

    ArrayType = np.ndarray | cp.ndarray
except ImportError:
    ArrayType = np.ndarray


def array_type(array: ArrayType) -> "type[np] | type[cp]":
    """Determine the array module (NumPy or CuPy) for the given array.

    This function enables array-agnostic code by returning the appropriate
    array module that can be used for operations on the input array.

    Args:
        array: Input array that can be either NumPy or CuPy array.

    Returns:
        The array module (numpy or cupy) corresponding to the input array type.

    Examples:
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3])
        >>> xp = array_type(arr)
        >>> result = xp.sum(arr)  # Uses numpy.sum
    """
    try:
        import cupy as cp

        return cp.get_array_module(array)
    except ImportError:
        return np


def calculate_center_of_mass(centers: ArrayType, sizes: ArrayType) -> ArrayType:
    """Calculate the center of mass for a collection of elements.

    Computes the volume-weighted centroid of mesh elements, commonly used
    in computational fluid dynamics for mesh analysis and load balancing.

    Args:
        centers: Array of shape (n_elements, 3) containing the centroid
            coordinates of each element.
        sizes: Array of shape (n_elements,) containing the volume
            or area of each element used as weights.

    Returns:
        Array of shape (1, 3) containing the x, y, z coordinates of the center of mass.

    Raises:
        ValueError: If centers and sizes have incompatible shapes.

    Examples:
        >>> import numpy as np
        >>> centers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        >>> sizes = np.array([1.0, 2.0, 3.0])
        >>> com = calculate_center_of_mass(centers, sizes)
        >>> np.allclose(com, [[4.0/3.0, 4.0/3.0, 4.0/3.0]])
        True
    """
    xp = array_type(centers)

    total_weighted_position = xp.einsum("i,ij->j", sizes, centers)
    total_size = xp.sum(sizes)

    return total_weighted_position[None, ...] / total_size


def normalize(
    field: ArrayType, max_val: ArrayType | None = None, min_val: ArrayType | None = None
) -> ArrayType:
    """Normalize field values to the range [-1, 1].

    Applies min-max normalization to scale field values to a symmetric range
    around zero. This is commonly used in machine learning preprocessing to
    ensure numerical stability and faster convergence.

    Args:
        field: Input field array to be normalized.
        max_val: Maximum values for normalization, can be scalar or array.
            If None, computed from the field data.
        min_val: Minimum values for normalization, can be scalar or array.
            If None, computed from the field data.

    Returns:
        Normalized field with values in the range [-1, 1].

    Raises:
        ZeroDivisionError: If max_val equals min_val (zero range).

    Examples:
        >>> import numpy as np
        >>> field = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> normalized = normalize(field, 5.0, 1.0)
        >>> np.allclose(normalized, [-1.0, -0.5, 0.0, 0.5, 1.0])
        True
        >>> # Auto-compute min/max
        >>> normalized_auto = normalize(field)
        >>> np.allclose(normalized_auto, [-1.0, -0.5, 0.0, 0.5, 1.0])
        True
    """
    xp = array_type(field)

    if max_val is None:
        max_val = xp.max(field, axis=0, keepdims=True)
    if min_val is None:
        min_val = xp.min(field, axis=0, keepdims=True)

    field_range = max_val - min_val
    return 2.0 * (field - min_val) / field_range - 1.0


def unnormalize(
    normalized_field: ArrayType, max_val: ArrayType, min_val: ArrayType
) -> ArrayType:
    """Reverse the normalization process to recover original field values.

    Transforms normalized values from the range [-1, 1] back to their original
    physical range using the stored min/max values.

    Args:
        normalized_field: Field values in the normalized range [-1, 1].
        max_val: Maximum values used in the original normalization.
        min_val: Minimum values used in the original normalization.

    Returns:
        Field values restored to their original physical range.

    Examples:
        >>> import numpy as np
        >>> normalized = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        >>> original = unnormalize(normalized, 5.0, 1.0)
        >>> np.allclose(original, [1.0, 2.0, 3.0, 4.0, 5.0])
        True
    """
    field_range = max_val - min_val
    return (normalized_field + 1.0) * field_range * 0.5 + min_val


def standardize(
    field: ArrayType, mean: ArrayType | None = None, std: ArrayType | None = None
) -> ArrayType:
    """Standardize field values to have zero mean and unit variance.

    Applies z-score normalization to center the data around zero with
    standard deviation of one. This is preferred over min-max normalization
    when the data follows a normal distribution.

    Args:
        field: Input field array to be standardized.
        mean: Mean values for standardization. If None, computed from field data.
        std: Standard deviation values for standardization. If None, computed from field data.

    Returns:
        Standardized field with approximately zero mean and unit variance.

    Raises:
        ZeroDivisionError: If std contains zeros.

    Examples:
        >>> import numpy as np
        >>> field = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> standardized = standardize(field, 3.0, np.sqrt(2.5))
        >>> np.allclose(standardized, [-1.265, -0.632, 0.0, 0.632, 1.265], atol=1e-3)
        True
        >>> # Auto-compute mean/std
        >>> standardized_auto = standardize(field)
        >>> np.allclose(np.mean(standardized_auto), 0.0)
        True
        >>> np.allclose(np.std(standardized_auto, ddof=0), 1.0)
        True
    """
    xp = array_type(field)

    if mean is None:
        mean = xp.mean(field, axis=0, keepdims=True)
    if std is None:
        std = xp.std(field, axis=0, keepdims=True)

    return (field - mean) / std


def unstandardize(
    standardized_field: ArrayType, mean: ArrayType, std: ArrayType
) -> ArrayType:
    """Reverse the standardization process to recover original field values.

    Transforms standardized values (zero mean, unit variance) back to their
    original distribution using the stored mean and standard deviation.

    Args:
        standardized_field: Field values with zero mean and unit variance.
        mean: Mean values used in the original standardization.
        std: Standard deviation values used in the original standardization.

    Returns:
        Field values restored to their original distribution.

    Examples:
        >>> import numpy as np
        >>> standardized = np.array([-1.265, -0.632, 0.0, 0.632, 1.265])
        >>> original = unstandardize(standardized, 3.0, np.sqrt(2.5))
        >>> np.allclose(original, [1.0, 2.0, 3.0, 4.0, 5.0], atol=1e-3)
        True
    """
    return standardized_field * std + mean


def write_to_vtp(polydata: "vtk.vtkPolyData", filename: str) -> None:
    """Write VTK polydata to a VTP (VTK PolyData) file format.

    VTP files are XML-based and store polygonal data including points, polygons,
    and associated field data. This format is commonly used for surface meshes
    in computational fluid dynamics visualization.

    Args:
        polydata: VTK polydata object containing mesh geometry and fields.
        filename: Output filename with .vtp extension. Directory will be created
            if it doesn't exist.

    Raises:
        RuntimeError: If writing fails due to file permissions or disk space.

    """
    # Ensure output directory exists
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(polydata)

    if not writer.Write():
        raise RuntimeError(f"Failed to write polydata to {output_path}")


def write_to_vtu(unstructured_grid: "vtk.vtkUnstructuredGrid", filename: str) -> None:
    """Write VTK unstructured grid to a VTU (VTK Unstructured Grid) file format.

    VTU files store 3D volumetric meshes with arbitrary cell types including
    tetrahedra, hexahedra, and pyramids. This format is essential for storing
    finite element analysis results.

    Args:
        unstructured_grid: VTK unstructured grid object containing volumetric mesh
            geometry and field data.
        filename: Output filename with .vtu extension. Directory will be created
            if it doesn't exist.

    Raises:
        RuntimeError: If writing fails due to file permissions or disk space.

    """
    # Ensure output directory exists
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(unstructured_grid)

    if not writer.Write():
        raise RuntimeError(f"Failed to write unstructured grid to {output_path}")


def extract_surface_triangles(tetrahedral_mesh: "vtk.vtkUnstructuredGrid") -> list[int]:
    """Extract surface triangle indices from a tetrahedral mesh.

    This function identifies the boundary faces of a 3D tetrahedral mesh and
    returns the vertex indices that form triangular faces on the surface.
    This is essential for visualization and boundary condition application.

    Args:
        tetrahedral_mesh: VTK unstructured grid containing tetrahedral elements.

    Returns:
        List of vertex indices forming surface triangles. Every three consecutive
        indices define one triangle.

    Raises:
        NotImplementedError: If the surface contains non-triangular faces.

    """
    # Extract the surface using VTK filter
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(tetrahedral_mesh)
    surface_filter.Update()

    # Wrap with PyVista for easier manipulation
    import pyvista as pv

    surface_mesh = pv.wrap(surface_filter.GetOutput())
    triangle_indices = []

    # Process faces - PyVista stores faces as [n_vertices, v1, v2, ..., vn]
    faces = surface_mesh.faces.reshape((-1, 4))
    for face in faces:
        if face[0] == 3:  # Triangle (3 vertices)
            triangle_indices.extend([face[1], face[2], face[3]])
        else:
            raise NotImplementedError(
                f"Non-triangular face found with {face[0]} vertices"
            )

    return triangle_indices


def convert_to_tet_mesh(polydata: "vtk.vtkPolyData") -> "vtk.vtkUnstructuredGrid":
    """Convert surface polydata to a tetrahedral volumetric mesh.

    This function performs tetrahedralization of a surface mesh, creating
    a 3D volumetric mesh suitable for finite element analysis. The process
    fills the interior of the surface with tetrahedral elements.

    Args:
        polydata: VTK polydata representing a closed surface mesh.

    Returns:
        VTK unstructured grid containing tetrahedral elements filling the
        volume enclosed by the input surface.

    Raises:
        RuntimeError: If tetrahedralization fails (e.g., non-manifold surface).

    """
    tetrahedral_filter = vtkDataSetTriangleFilter()
    tetrahedral_filter.SetInputData(polydata)
    tetrahedral_filter.Update()

    tetrahedral_mesh = tetrahedral_filter.GetOutput()
    return tetrahedral_mesh


def convert_point_data_to_cell_data(input_data: "vtk.vtkDataSet") -> "vtk.vtkDataSet":
    """Convert point-based field data to cell-based field data.

    This function transforms field variables defined at mesh vertices (nodes)
    to values defined at cell centers. This conversion is often needed when
    switching between different numerical methods or visualization requirements.

    Args:
        input_data: VTK dataset with point data to be converted.

    Returns:
        VTK dataset with the same geometry but field data moved from points to cells.
        Values are typically averaged from the surrounding points.

    """
    point_to_cell_filter = vtk.vtkPointDataToCellData()
    point_to_cell_filter.SetInputData(input_data)
    point_to_cell_filter.Update()

    return point_to_cell_filter.GetOutput()


def get_node_to_elem(polydata: "vtk.vtkDataSet") -> "vtk.vtkDataSet":
    """Convert point data to cell data for VTK dataset.

    This function transforms field variables defined at mesh vertices to
    values defined at cell centers using VTK's built-in conversion filter.

    Args:
        polydata: VTK dataset with point data to be converted.

    Returns:
        VTK dataset with field data moved from points to cells.

    """
    point_to_cell_filter = vtk.vtkPointDataToCellData()
    point_to_cell_filter.SetInputData(polydata)
    point_to_cell_filter.Update()
    cell_data = point_to_cell_filter.GetOutput()
    return cell_data


def get_fields_from_cell(
    cell_data: "vtk.vtkCellData", variable_names: list[str]
) -> np.ndarray:
    """Extract field variables from VTK cell data.

    This function extracts multiple field variables from VTK cell data and
    organizes them into a structured NumPy array. Each variable becomes a
    column in the output array.

    Args:
        cell_data: VTK cell data object containing field variables.
        variable_names: List of variable names to extract from the cell data.

    Returns:
        NumPy array of shape (n_cells, n_variables) containing the extracted
        field data. Variables are ordered according to the input list.

    Raises:
        ValueError: If a requested variable name is not found in the cell data.

    """
    extracted_fields = []
    for variable_name in variable_names:
        variable_array = cell_data.GetArray(variable_name)
        if variable_array is None:
            raise ValueError(f"Variable '{variable_name}' not found in cell data")

        num_tuples = variable_array.GetNumberOfTuples()
        field_values = []
        for tuple_idx in range(num_tuples):
            variable_value = np.array(variable_array.GetTuple(tuple_idx))
            field_values.append(variable_value)
        field_values = np.asarray(field_values)
        extracted_fields.append(field_values)

    # Transpose to get shape (n_cells, n_variables)
    extracted_fields = np.transpose(np.asarray(extracted_fields), (1, 0))
    return extracted_fields


def get_fields(
    data_attributes: "vtk.vtkDataSetAttributes", variable_names: list[str]
) -> list[np.ndarray]:
    """Extract multiple field variables from VTK data attributes.

    This function extracts field variables from VTK data attributes (either
    point data or cell data) and returns them as a list of NumPy arrays.
    It handles both point and cell data seamlessly.

    Args:
        data_attributes: VTK data attributes object (point data or cell data).
        variable_names: List of variable names to extract.

    Returns:
        List of NumPy arrays, one for each requested variable. Each array
        has shape (n_points/n_cells, n_components) where n_components
        depends on the variable (1 for scalars, 3 for vectors, etc.).

    Raises:
        ValueError: If a requested variable is not found in the data attributes.

    """
    extracted_fields = []
    for variable_name in variable_names:
        try:
            vtk_array = data_attributes.GetArray(variable_name)
        except ValueError as e:
            raise ValueError(
                f"Failed to get array '{variable_name}' from the data attributes: {e}"
            )

        # Convert VTK array to NumPy array with proper shape
        numpy_array = numpy_support.vtk_to_numpy(vtk_array).reshape(
            vtk_array.GetNumberOfTuples(), vtk_array.GetNumberOfComponents()
        )
        extracted_fields.append(numpy_array)

    return extracted_fields


def get_vertices(polydata: "vtk.vtkPolyData") -> np.ndarray:
    """Extract vertex coordinates from VTK polydata object.

    This function converts VTK polydata to a NumPy array containing the 3D
    coordinates of all vertices in the mesh.

    Args:
        polydata: VTK polydata object containing mesh geometry.

    Returns:
        NumPy array of shape (n_points, 3) containing [x, y, z] coordinates
        for each vertex.

    """
    vtk_points = polydata.GetPoints()
    vertices = numpy_support.vtk_to_numpy(vtk_points.GetData())
    return vertices


def get_volume_data(
    polydata: "vtk.vtkPolyData", variable_names: list[str]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Extract vertices and field data from 3D volumetric mesh.

    This function extracts both geometric information (vertex coordinates)
    and field data from a 3D volumetric mesh. It's commonly used for
    processing finite element analysis results.

    Args:
        polydata: VTK polydata representing a 3D volumetric mesh.
        variable_names: List of field variable names to extract.

    Returns:
        Tuple containing:
        - Vertex coordinates as NumPy array of shape (n_vertices, 3)
        - List of field arrays, one per variable

    """
    vertices = get_vertices(polydata)
    point_data = polydata.GetPointData()
    fields = get_fields(point_data, variable_names)

    return vertices, fields


def get_surface_data(
    polydata: "vtk.vtkPolyData", variable_names: list[str]
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[int, int]]]:
    """Extract surface mesh data including vertices, fields, and edge connectivity.

    This function extracts comprehensive surface mesh information including
    vertex coordinates, field data at vertices, and edge connectivity information.
    It's commonly used for processing CFD surface results and boundary conditions.

    Args:
        polydata: VTK polydata representing a surface mesh.
        variable_names: List of field variable names to extract from the mesh.

    Returns:
        Tuple containing:
        - Vertex coordinates as NumPy array of shape (n_vertices, 3)
        - List of field arrays, one per variable
        - List of edge tuples representing mesh connectivity

    Raises:
        ValueError: If a requested variable is not found or polygon data is missing.

    """
    points = polydata.GetPoints()
    vertices = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])

    point_data = polydata.GetPointData()
    fields = []
    for array_name in variable_names:
        try:
            array = point_data.GetArray(array_name)
        except ValueError:
            raise ValueError(
                f"Failed to get array {array_name} from the unstructured grid."
            )
        array_data = np.zeros(
            (points.GetNumberOfPoints(), array.GetNumberOfComponents())
        )
        for j in range(points.GetNumberOfPoints()):
            array.GetTuple(j, array_data[j])
        fields.append(array_data)

    polys = polydata.GetPolys()
    if polys is None:
        raise ValueError("Failed to get polygons from the polydata.")
    polys.InitTraversal()
    edges = []
    id_list = vtk.vtkIdList()
    for _ in range(polys.GetNumberOfCells()):
        polys.GetNextCell(id_list)
        num_ids = id_list.GetNumberOfIds()
        edges = [
            (id_list.GetId(j), id_list.GetId((j + 1) % num_ids)) for j in range(num_ids)
        ]

    return vertices, fields, edges


def calculate_normal_positional_encoding(
    coordinates_a: ArrayType,
    coordinates_b: ArrayType | None = None,
    cell_dimensions: Sequence[float] = (1.0, 1.0, 1.0),
) -> ArrayType:
    """Calculate sinusoidal positional encoding for 3D coordinates.

    This function computes transformer-style positional encodings for 3D spatial
    coordinates, enabling neural networks to understand spatial relationships.
    The encoding uses sinusoidal functions at different frequencies to create
    unique representations for each spatial position.

    Args:
        coordinates_a: Primary coordinates array of shape (n_points, 3).
        coordinates_b: Optional secondary coordinates for computing relative positions.
            If provided, the encoding is computed for (coordinates_a - coordinates_b).
        cell_dimensions: Characteristic length scales for x, y, z dimensions used
            for normalization. Defaults to unit dimensions.

    Returns:
        Array of shape (n_points, 12) containing positional encodings with
        4 encoding dimensions per spatial axis (x, y, z).

    Examples:
        >>> import numpy as np
        >>> coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        >>> cell_size = [0.1, 0.1, 0.1]
        >>> encoding = calculate_normal_positional_encoding(coords, cell_dimensions=cell_size)
        >>> encoding.shape
        (2, 12)
        >>> # Relative positioning example
        >>> coords_b = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        >>> encoding_rel = calculate_normal_positional_encoding(coords, coords_b, cell_size)
        >>> encoding_rel.shape
        (2, 12)
    """
    dx, dy, dz = cell_dimensions[0], cell_dimensions[1], cell_dimensions[2]
    xp = array_type(coordinates_a)

    if coordinates_b is not None:
        normals = coordinates_a - coordinates_b
        pos_x = xp.asarray(calculate_pos_encoding(normals[:, 0] / dx, d=4))
        pos_y = xp.asarray(calculate_pos_encoding(normals[:, 1] / dy, d=4))
        pos_z = xp.asarray(calculate_pos_encoding(normals[:, 2] / dz, d=4))
        pos_normals = xp.concatenate((pos_x, pos_y, pos_z), axis=0).reshape(-1, 12)
    else:
        normals = coordinates_a
        pos_x = xp.asarray(calculate_pos_encoding(normals[:, 0] / dx, d=4))
        pos_y = xp.asarray(calculate_pos_encoding(normals[:, 1] / dy, d=4))
        pos_z = xp.asarray(calculate_pos_encoding(normals[:, 2] / dz, d=4))
        pos_normals = xp.concatenate((pos_x, pos_y, pos_z), axis=0).reshape(-1, 12)

    return pos_normals


def nd_interpolator(
    coordinates: ArrayType, field: ArrayType, grid: ArrayType, k: int = 2
) -> ArrayType:
    """Perform n-dimensional interpolation using k-nearest neighbors.

    This function interpolates field values from scattered points to a regular
    grid using k-nearest neighbor averaging. It's useful for reconstructing
    fields on regular grids from irregular measurement points.

    Args:
        coordinates: Array of shape (n_points, n_dims) containing source point coordinates.
        field: Array of shape (n_points, n_fields) containing field values at source points.
        grid: Array of shape (n_field_points, n_dims) containing target grid points for interpolation.
        k: Number of nearest neighbors to use for interpolation.

    Returns:
        Interpolated field values at grid points using k-nearest neighbor averaging.

    Note:
        This function currently uses SciPy's KDTree which only supports CPU arrays.
        A future enhancement could add CuML support for GPU acceleration.

    Examples:
        >>> import numpy as np
        >>> # Simple 2D interpolation example
        >>> coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        >>> field_vals = np.array([[1.0], [2.0], [3.0], [4.0]])
        >>> grid_points = np.array([[0.5, 0.5]])
        >>> result = nd_interpolator([coords], field_vals, grid_points)
        >>> result.shape[0] == 1  # One grid point
        True
    """
    # TODO - this function should get updated for cuml if using cupy.
    kdtree = KDTree(coordinates[0])
    distances, neighbor_indices = kdtree.query(grid, k=k)

    field_grid = field[neighbor_indices]
    field_grid = np.mean(field_grid, axis=1)
    return field_grid


def pad(arr: ArrayType, n_points: int, pad_value: float = 0.0) -> ArrayType:
    """Pad 2D array with constant values to reach target size.

    This function extends a 2D array by adding rows filled with a constant
    value. It's commonly used to standardize array sizes in batch processing
    for machine learning applications.

    Args:
        arr: Input array of shape (n_points, n_features) to be padded.
        n_points: Target number of points (rows) after padding.
        pad_value: Constant value used for padding. Defaults to 0.0.

    Returns:
        Padded array of shape (n_points, n_features). If n_points <= arr.shape[0],
        returns the original array unchanged.

    Examples:
        >>> import numpy as np
        >>> arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> padded = pad(arr, 4, -1.0)
        >>> padded.shape
        (4, 2)
        >>> np.array_equal(padded[:2], arr)
        True
        >>> bool(np.all(padded[2:] == -1.0))
        True
        >>> # No padding needed
        >>> same = pad(arr, 2)
        >>> np.array_equal(same, arr)
        True
    """
    xp = array_type(arr)
    if n_points <= arr.shape[0]:
        return arr

    arr_pad = pad_value * xp.ones(
        (n_points - arr.shape[0], arr.shape[1]), dtype=xp.float32
    )
    arr_padded = xp.concatenate((arr, arr_pad), axis=0)
    return arr_padded


def pad_inp(arr: ArrayType, n_points: int, pad_value: float = 0.0) -> ArrayType:
    """Pad 3D array with constant values to reach target size.

    This function extends a 3D array by adding entries along the first dimension
    filled with a constant value. Used for standardizing 3D tensor sizes in
    batch processing workflows.

    Args:
        arr: Input array of shape (n_points, height, width) to be padded.
        n_points: Target number of points along first dimension after padding.
        pad_value: Constant value used for padding. Defaults to 0.0.

    Returns:
        Padded array of shape (n_points, height, width). If n_points <= arr.shape[0],
        returns the original array unchanged.

    Examples:
        >>> import numpy as np
        >>> arr = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        >>> padded = pad_inp(arr, 4, 0.0)
        >>> padded.shape
        (4, 1, 2)
        >>> np.array_equal(padded[:2], arr)
        True
        >>> bool(np.all(padded[2:] == 0.0))
        True
    """
    xp = array_type(arr)
    if n_points <= arr.shape[0]:
        return arr

    arr_pad = pad_value * xp.ones(
        (n_points - arr.shape[0], arr.shape[1], arr.shape[2]), dtype=xp.float32
    )
    arr_padded = xp.concatenate((arr, arr_pad), axis=0)
    return arr_padded


@profile
def shuffle_array(
    arr: ArrayType,
    n_points: int,
) -> tuple[ArrayType, ArrayType]:
    """Randomly sample points from array without replacement.

    This function performs random sampling from the input array, selecting
    n_points points without replacement. It's commonly used for creating training
    subsets and data augmentation in machine learning workflows.

    Args:
        arr: Input array to sample from, shape (n_points, ...).
        n_points: Number of points to sample. If greater than arr.shape[0],
            all points are returned.

    Returns:
        Tuple containing:
        - Sampled array subset
        - Indices of the selected points

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> subset, indices = shuffle_array(data, 2)
        >>> subset.shape
        (2, 2)
        >>> indices.shape
        (2,)
        >>> len(np.unique(indices)) == 2  # No duplicates
        True
    """
    xp = array_type(arr)
    if n_points > arr.shape[0]:
        # If asking too many points, truncate the ask but still shuffle.
        n_points = arr.shape[0]
    idx = xp.random.choice(arr.shape[0], size=n_points, replace=False)
    return arr[idx], idx


def shuffle_array_without_sampling(arr: ArrayType) -> tuple[ArrayType, ArrayType]:
    """Shuffle array order without changing the number of elements.

    This function reorders all elements in the array randomly while preserving
    all data points. It's useful for randomizing data order before training
    while maintaining the complete dataset.

    Args:
        arr: Input array to shuffle, shape (n_points, ...).

    Returns:
        Tuple containing:
        - Shuffled array with same shape as input
        - Permutation indices used for shuffling

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> data = np.array([[1], [2], [3], [4]])
        >>> shuffled, indices = shuffle_array_without_sampling(data)
        >>> shuffled.shape
        (4, 1)
        >>> indices.shape
        (4,)
        >>> set(indices) == set(range(4))  # All original indices present
        True
    """
    xp = array_type(arr)
    idx = xp.arange(arr.shape[0])
    xp.random.shuffle(idx)
    return arr[idx], idx


def create_directory(filepath: str | Path) -> None:
    """Create directory and all necessary parent directories.

    This function creates a directory at the specified path, including any
    necessary parent directories. It's equivalent to `mkdir -p` in Unix systems.

    Args:
        filepath: Path to the directory to create. Can be string or Path object.

    """
    Path(filepath).mkdir(parents=True, exist_ok=True)


def get_filenames(filepath: str | Path, exclude_dirs: bool = False) -> list[str]:
    """Get list of filenames in a directory with optional directory filtering.

    This function returns all items in a directory, with options to exclude
    subdirectories. It handles special cases like .zarr directories which
    are treated as files even when exclude_dirs is True.

    Args:
        filepath: Path to the directory to list. Can be string or Path object.
        exclude_dirs: If True, exclude subdirectories from results.
            Exception: .zarr directories are always included as they represent
            data files in array storage format.

    Returns:
        List of filenames/directory names found in the specified directory.

    Raises:
        FileNotFoundError: If the specified directory does not exist.

    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Directory {filepath} does not exist")

    filenames = []
    for item in path.iterdir():
        if exclude_dirs and item.is_dir():
            # Include directories ending with .zarr even when exclude_dirs is True
            if item.name.endswith(".zarr"):
                filenames.append(item.name)
            continue
        filenames.append(item.name)
    return filenames


def calculate_pos_encoding(nx: ArrayType, d: int = 8) -> list[ArrayType]:
    """Calculate sinusoidal positional encoding for transformer architectures.

    This function computes positional encodings using alternating sine and cosine
    functions at different frequencies. These encodings help neural networks
    understand positional relationships in sequences or spatial data.

    Args:
        nx: Input positions/coordinates to encode.
        d: Encoding dimensionality. Must be even number. Defaults to 8.

    Returns:
        List of d arrays containing alternating sine and cosine encodings.
        Each pair (sin, cos) uses progressively lower frequencies.

    Examples:
        >>> import numpy as np
        >>> positions = np.array([0.0, 1.0, 2.0])
        >>> encodings = calculate_pos_encoding(positions, d=4)
        >>> len(encodings)
        4
        >>> all(enc.shape == (3,) for enc in encodings)
        True
    """
    vec = []
    xp = array_type(nx)
    for k in range(int(d / 2)):
        vec.append(xp.sin(nx / 10000 ** (2 * k / d)))
        vec.append(xp.cos(nx / 10000 ** (2 * k / d)))
    return vec


def combine_dict(old_dict: dict[Any, Any], new_dict: dict[Any, Any]) -> dict[Any, Any]:
    """Combine two dictionaries by adding values for matching keys.

    This function performs element-wise addition of dictionary values for
    keys that exist in both dictionaries. It's commonly used for accumulating
    statistics or metrics across multiple iterations.

    Args:
        old_dict: Base dictionary to update.
        new_dict: Dictionary with values to add to old_dict.

    Returns:
        Updated old_dict with combined values.

    Note:
        This function modifies old_dict in place and returns it.
        Values must support the + operator.

    Examples:
        >>> stats1 = {"loss": 0.5, "accuracy": 0.8}
        >>> stats2 = {"loss": 0.3, "accuracy": 0.1}
        >>> combined = combine_dict(stats1, stats2)
        >>> combined["loss"]
        0.8
        >>> combined["accuracy"]
        0.9
    """
    for key in old_dict.keys():
        old_dict[key] += new_dict[key]
    return old_dict


def create_grid(
    max_coords: ArrayType, min_coords: ArrayType, resolution: ArrayType
) -> ArrayType:
    """Create a 3D regular grid from coordinate bounds and resolution.

    This function generates a regular 3D grid spanning from min_coords to
    max_coords with the specified resolution in each dimension. The resulting
    grid is commonly used for interpolation, visualization, and regular sampling.

    Args:
        max_coords: Maximum coordinates [x_max, y_max, z_max] for the grid bounds.
        min_coords: Minimum coordinates [x_min, y_min, z_min] for the grid bounds.
        resolution: Number of grid points [nx, ny, nz] in each dimension.

    Returns:
        Grid array of shape (nx, ny, nz, 3) containing 3D coordinates for each
        grid point. The last dimension contains [x, y, z] coordinates.

    Examples:
        >>> import numpy as np
        >>> min_bounds = np.array([0.0, 0.0, 0.0])
        >>> max_bounds = np.array([1.0, 1.0, 1.0])
        >>> grid_res = np.array([2, 2, 2])
        >>> grid = create_grid(max_bounds, min_bounds, grid_res)
        >>> grid.shape
        (2, 2, 2, 3)
        >>> np.allclose(grid[0, 0, 0], [0.0, 0.0, 0.0])
        True
        >>> np.allclose(grid[1, 1, 1], [1.0, 1.0, 1.0])
        True
    """
    xp = array_type(max_coords)

    dx = xp.linspace(
        min_coords[0], max_coords[0], resolution[0], dtype=max_coords.dtype
    )
    dy = xp.linspace(
        min_coords[1], max_coords[1], resolution[1], dtype=max_coords.dtype
    )
    dz = xp.linspace(
        min_coords[2], max_coords[2], resolution[2], dtype=max_coords.dtype
    )

    xv, yv, zv = xp.meshgrid(dx, dy, dz)
    xv = xp.expand_dims(xv, -1)
    yv = xp.expand_dims(yv, -1)
    zv = xp.expand_dims(zv, -1)
    grid = xp.concatenate((xv, yv, zv), axis=-1)
    grid = xp.transpose(grid, (1, 0, 2, 3))

    return grid


def mean_std_sampling(
    field: ArrayType, mean: ArrayType, std: ArrayType, tolerance: float = 3.0
) -> list[int]:
    """Identify outlier points based on statistical distance from mean.

    This function identifies data points that are statistical outliers by
    checking if they fall outside mean ± tolerance*std for any field component.
    It's useful for data cleaning and identifying regions of interest in CFD data.

    Args:
        field: Input field array of shape (n_points, n_components).
        mean: Mean values for each field component, shape (n_components,).
        std: Standard deviation for each component, shape (n_components,).
        tolerance: Number of standard deviations to use as outlier threshold.
            Defaults to 3.0 (99.7% of normal distribution).

    Returns:
        List of indices identifying outlier points that exceed the statistical threshold.

    Examples:
        >>> import numpy as np
        >>> # Create test data with outliers
        >>> field = np.array([[1.0], [2.0], [3.0], [10.0]])  # 10.0 is outlier
        >>> field_mean = np.array([2.0])
        >>> field_std = np.array([1.0])
        >>> outliers = mean_std_sampling(field, field_mean, field_std, 2.0)
        >>> 3 in outliers  # Index 3 (value 10.0) should be detected as outlier
        True
    """
    xp = array_type(field)
    idx_all = []
    for v in range(field.shape[-1]):
        fv = field[:, v]
        idx = xp.where(
            (fv > mean[v] + tolerance * std[v]) | (fv < mean[v] - tolerance * std[v])
        )
        if len(idx[0]) != 0:
            idx_all += list(idx[0])

    return idx_all


def dict_to_device(
    state_dict: dict[str, Any], device: Any, exclude_keys: list[str] | None = None
) -> dict[str, Any]:
    """Move dictionary values to specified device (GPU/CPU).

    This function transfers PyTorch tensors in a dictionary to the specified
    compute device while preserving the dictionary structure. It's commonly
    used for moving model parameters and data between CPU and GPU.

    Args:
        state_dict: Dictionary containing tensors and other values.
        device: Target device (e.g., torch.device('cuda:0')).
        exclude_keys: List of keys to skip during device transfer.
            Defaults to ["filename"] if None.

    Returns:
        New dictionary with tensors moved to the specified device.
        Non-tensor values and excluded keys are preserved as-is.

    Examples:
        >>> import torch
        >>> data = {"weights": torch.randn(10, 10), "filename": "model.pt"}
        >>> gpu_data = dict_to_device(data, torch.device('cuda:0'))
    """
    if exclude_keys is None:
        exclude_keys = ["filename"]

    new_state_dict = {}
    for k, v in state_dict.items():
        if k not in exclude_keys:
            new_state_dict[k] = v.to(device)
    return new_state_dict


def area_weighted_shuffle_array(
    arr: ArrayType, n_points: int, area: ArrayType, area_factor: float = 1.0
) -> tuple[ArrayType, ArrayType]:
    """Perform area-weighted random sampling from array.

    This function samples points from an array with probability proportional to
    their associated area weights. This is particularly useful in CFD applications
    where larger cells or surface elements should have higher sampling probability.

    Args:
        arr: Input array to sample from, shape (n_points, ...).
        n_points: Number of points to sample. If greater than arr.shape[0],
            samples all available points.
        area: Area weights for each point, shape (n_points,). Larger values
            indicate higher sampling probability.
        area_factor: Exponent applied to area weights to control sampling bias.
            Values > 1.0 increase bias toward larger areas, values < 1.0 reduce bias.
            Defaults to 1.0 (linear weighting).

    Returns:
        Tuple containing:
        - Sampled array subset weighted by area
        - Indices of the selected points

    Note:
        For GPU arrays (CuPy), the sampling is performed on CPU due to memory
        efficiency considerations. The Alias method could be implemented for
        future GPU acceleration.

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> mesh_data = np.array([[1.0], [2.0], [3.0], [4.0]])
        >>> cell_areas = np.array([0.1, 0.1, 0.1, 10.0])  # Last point has much larger area
        >>> subset, indices = area_weighted_shuffle_array(mesh_data, 2, cell_areas)
        >>> subset.shape
        (2, 1)
        >>> indices.shape
        (2,)
        >>> # The point with large area (index 3) should likely be selected
        >>> len(set(indices)) <= 2  # At most 2 unique indices
        True
        >>> # Use higher area_factor for stronger bias toward large areas
        >>> subset_biased, _ = area_weighted_shuffle_array(mesh_data, 2, cell_areas, area_factor=2.0)
    """
    xp = array_type(arr)
    # Calculate area-weighted probabilities
    sampling_probabilities = area**area_factor
    sampling_probabilities /= xp.sum(sampling_probabilities)  # Normalize to sum to 1

    # Ensure we don't request more points than available
    n_points = min(n_points, arr.shape[0])

    # Create index array for all available points
    point_indices = xp.arange(arr.shape[0])

    selected_indices = xp.random.choice(
        xp.asarray(point_indices), size=n_points, p=xp.asarray(sampling_probabilities)
    )
    selected_indices = xp.asarray(selected_indices)

    return arr[selected_indices], selected_indices
