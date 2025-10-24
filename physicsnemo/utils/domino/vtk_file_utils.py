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

import numpy as np
import vtk
from vtk import vtkDataSetTriangleFilter
from vtk.util import numpy_support


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
