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

r"""
VTK File Utilities for DoMINO.

This module provides utilities for reading and writing VTK file formats,
mesh manipulation, and data extraction for computational fluid dynamics
applications. It supports both CPU (NumPy) operations with optional VTK
and PyVista dependencies.
"""

import importlib
from pathlib import Path

import numpy as np

from physicsnemo.core.version_check import check_version_spec

VTK_AVAILABLE = check_version_spec("vtk", "9.0.0", hard_fail=False)

if VTK_AVAILABLE:
    vtk = importlib.import_module("vtk")
    vtkDataSetTriangleFilter = vtk.vtkDataSetTriangleFilter
    numpy_support = importlib.import_module("vtk.util.numpy_support")

    def write_to_vtp(polydata: "vtk.vtkPolyData", filename: str) -> None:
        r"""
        Write VTK polydata to a VTP (VTK PolyData) file format.

        VTP files are XML-based and store polygonal data including points,
        polygons, and associated field data. This format is commonly used
        for surface meshes in computational fluid dynamics visualization.

        Parameters
        ----------
        polydata : vtk.vtkPolyData
            VTK polydata object containing mesh geometry and fields.
        filename : str
            Output filename with ``.vtp`` extension. Directory will be
            created if it doesn't exist.

        Raises
        ------
        RuntimeError
            If writing fails due to file permissions or disk space.
        """
        # Ensure output directory exists
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputData(polydata)

        if not writer.Write():
            raise RuntimeError(f"Failed to write polydata to {output_path}")

    def write_to_vtu(
        unstructured_grid: "vtk.vtkUnstructuredGrid", filename: str
    ) -> None:
        r"""
        Write VTK unstructured grid to a VTU file format.

        VTU files store 3D volumetric meshes with arbitrary cell types
        including tetrahedra, hexahedra, and pyramids. This format is
        essential for storing finite element analysis results.

        Parameters
        ----------
        unstructured_grid : vtk.vtkUnstructuredGrid
            VTK unstructured grid object containing volumetric mesh
            geometry and field data.
        filename : str
            Output filename with ``.vtu`` extension. Directory will be
            created if it doesn't exist.

        Raises
        ------
        RuntimeError
            If writing fails due to file permissions or disk space.
        """
        # Ensure output directory exists
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputData(unstructured_grid)

        if not writer.Write():
            raise RuntimeError(f"Failed to write unstructured grid to {output_path}")

    def convert_to_tet_mesh(polydata: "vtk.vtkPolyData") -> "vtk.vtkUnstructuredGrid":
        r"""
        Convert surface polydata to a tetrahedral volumetric mesh.

        This function performs tetrahedralization of a surface mesh, creating
        a 3D volumetric mesh suitable for finite element analysis. The process
        fills the interior of the surface with tetrahedral elements.

        Parameters
        ----------
        polydata : vtk.vtkPolyData
            VTK polydata representing a closed surface mesh.

        Returns
        -------
        vtk.vtkUnstructuredGrid
            VTK unstructured grid containing tetrahedral elements filling
            the volume enclosed by the input surface.

        Raises
        ------
        RuntimeError
            If tetrahedralization fails (e.g., non-manifold surface).
        """
        tetrahedral_filter = vtkDataSetTriangleFilter()
        tetrahedral_filter.SetInputData(polydata)
        tetrahedral_filter.Update()

        tetrahedral_mesh = tetrahedral_filter.GetOutput()
        return tetrahedral_mesh

    def convert_point_data_to_cell_data(
        input_data: "vtk.vtkDataSet",
    ) -> "vtk.vtkDataSet":
        r"""
        Convert point-based field data to cell-based field data.

        This function transforms field variables defined at mesh vertices
        (nodes) to values defined at cell centers. This conversion is often
        needed when switching between different numerical methods or
        visualization requirements.

        Parameters
        ----------
        input_data : vtk.vtkDataSet
            VTK dataset with point data to be converted.

        Returns
        -------
        vtk.vtkDataSet
            VTK dataset with the same geometry but field data moved from
            points to cells. Values are typically averaged from the
            surrounding points.
        """
        point_to_cell_filter = vtk.vtkPointDataToCellData()
        point_to_cell_filter.SetInputData(input_data)
        point_to_cell_filter.Update()

        return point_to_cell_filter.GetOutput()

    def get_node_to_elem(polydata: "vtk.vtkDataSet") -> "vtk.vtkDataSet":
        r"""
        Convert point data to cell data for VTK dataset.

        This function transforms field variables defined at mesh vertices
        to values defined at cell centers using VTK's built-in conversion
        filter.

        Parameters
        ----------
        polydata : vtk.vtkDataSet
            VTK dataset with point data to be converted.

        Returns
        -------
        vtk.vtkDataSet
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
        r"""
        Extract field variables from VTK cell data.

        This function extracts multiple field variables from VTK cell data
        and organizes them into a structured NumPy array. Each variable
        becomes a column in the output array.

        Parameters
        ----------
        cell_data : vtk.vtkCellData
            VTK cell data object containing field variables.
        variable_names : list[str]
            List of variable names to extract from the cell data.

        Returns
        -------
        numpy.ndarray
            NumPy array of shape :math:`(N_{cells}, N_{variables})`
            containing the extracted field data. Variables are ordered
            according to the input list.

        Raises
        ------
        ValueError
            If a requested variable name is not found in the cell data.
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
        r"""
        Extract multiple field variables from VTK data attributes.

        This function extracts field variables from VTK data attributes
        (either point data or cell data) and returns them as a list of
        NumPy arrays. It handles both point and cell data seamlessly.

        Parameters
        ----------
        data_attributes : vtk.vtkDataSetAttributes
            VTK data attributes object (point data or cell data).
        variable_names : list[str]
            List of variable names to extract.

        Returns
        -------
        list[numpy.ndarray]
            List of NumPy arrays, one for each requested variable. Each
            array has shape :math:`(N, C)` where :math:`N` is the number
            of points/cells and :math:`C` is the number of components
            (1 for scalars, 3 for vectors, etc.).

        Raises
        ------
        ValueError
            If a requested variable is not found in the data attributes.
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
        r"""
        Extract vertex coordinates from VTK polydata object.

        This function converts VTK polydata to a NumPy array containing
        the 3D coordinates of all vertices in the mesh.

        Parameters
        ----------
        polydata : vtk.vtkPolyData
            VTK polydata object containing mesh geometry.

        Returns
        -------
        numpy.ndarray
            NumPy array of shape :math:`(N_{points}, 3)` containing
            :math:`[x, y, z]` coordinates for each vertex.
        """
        vtk_points = polydata.GetPoints()
        vertices = numpy_support.vtk_to_numpy(vtk_points.GetData())
        return vertices

    def get_volume_data(
        polydata: "vtk.vtkPolyData", variable_names: list[str]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        r"""
        Extract vertices and field data from 3D volumetric mesh.

        This function extracts both geometric information (vertex
        coordinates) and field data from a 3D volumetric mesh. It's
        commonly used for processing finite element analysis results.

        Parameters
        ----------
        polydata : vtk.vtkPolyData
            VTK polydata representing a 3D volumetric mesh.
        variable_names : list[str]
            List of field variable names to extract.

        Returns
        -------
        tuple[numpy.ndarray, list[numpy.ndarray]]
            Tuple containing:

            - Vertex coordinates as NumPy array of shape
              :math:`(N_{vertices}, 3)`
            - List of field arrays, one per variable
        """
        vertices = get_vertices(polydata)
        point_data = polydata.GetPointData()
        fields = get_fields(point_data, variable_names)

        return vertices, fields

    def get_surface_data(
        polydata: "vtk.vtkPolyData", variable_names: list[str]
    ) -> tuple[np.ndarray, list[np.ndarray], list[tuple[int, int]]]:
        r"""
        Extract surface mesh data including vertices, fields, and edges.

        This function extracts comprehensive surface mesh information
        including vertex coordinates, field data at vertices, and edge
        connectivity information. It's commonly used for processing CFD
        surface results and boundary conditions.

        Parameters
        ----------
        polydata : vtk.vtkPolyData
            VTK polydata representing a surface mesh.
        variable_names : list[str]
            List of field variable names to extract from the mesh.

        Returns
        -------
        tuple[numpy.ndarray, list[numpy.ndarray], list[tuple[int, int]]]
            Tuple containing:

            - Vertex coordinates as NumPy array of shape
              :math:`(N_{vertices}, 3)`
            - List of field arrays, one per variable
            - List of edge tuples representing mesh connectivity

        Raises
        ------
        ValueError
            If a requested variable is not found or polygon data is
            missing.
        """
        points = polydata.GetPoints()
        vertices = np.array(
            [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
        )

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
                (id_list.GetId(j), id_list.GetId((j + 1) % num_ids))
                for j in range(num_ids)
            ]

        return vertices, fields, edges

    PYVISTA_AVAILABLE = check_version_spec("pyvista", "0.30.0", hard_fail=False)

    if PYVISTA_AVAILABLE:
        pv = importlib.import_module("pyvista")

        def extract_surface_triangles(
            tetrahedral_mesh: "vtk.vtkUnstructuredGrid",
        ) -> list[int]:
            r"""
            Extract surface triangle indices from a tetrahedral mesh.

            This function identifies the boundary faces of a 3D tetrahedral
            mesh and returns the vertex indices that form triangular faces
            on the surface. This is essential for visualization and boundary
            condition application.

            Parameters
            ----------
            tetrahedral_mesh : vtk.vtkUnstructuredGrid
                VTK unstructured grid containing tetrahedral elements.

            Returns
            -------
            list[int]
                List of vertex indices forming surface triangles. Every
                three consecutive indices define one triangle.

            Raises
            ------
            NotImplementedError
                If the surface contains non-triangular faces.
            """
            # Extract the surface using VTK filter
            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputData(tetrahedral_mesh)
            surface_filter.Update()

            # Wrap with PyVista for easier manipulation
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

    else:

        def _raise_pyvista_import_error():
            r"""Raise import error for when pyvista is not installed."""
            raise ImportError(
                "pyvista is not installed, can not be used from domino/utils/vtk_file_utils.py"
                "- To install pyvista, please see the installation guide at "
                "https://docs.pyvista.org/getting-started/installation.html"
            )

        def extract_surface_triangles(*args, **kwargs):
            r"""Dummy symbol for missing PyVista."""
            _raise_pyvista_import_error()

else:

    def _raise_vtk_import_error():
        r"""Raise import error for when vtk is not installed."""
        raise ImportError(
            "vtk is not installed, can not be used from domino/utils/vtk_file_utils.py"
            "- To install vtk, please see the installation guide at https://vtk.org/download/ \n"
            "- For `extract_surface_triangles`, you will also need to install pyvista."
            " See https://docs.pyvista.org/getting-started/installation.html for installation instructions."
        )

    def write_to_vtp(*args, **kwargs):
        r"""Dummy symbol for missing VTK."""
        _raise_vtk_import_error()

    def write_to_vtu(*args, **kwargs):
        r"""Dummy symbol for missing VTK."""
        _raise_vtk_import_error()

    def extract_surface_triangles(*args, **kwargs):
        r"""Dummy symbol for missing VTK."""
        _raise_vtk_import_error()

    def convert_to_tet_mesh(*args, **kwargs):
        r"""Dummy symbol for missing VTK."""
        _raise_vtk_import_error()

    def convert_point_data_to_cell_data(*args, **kwargs):
        r"""Dummy symbol for missing VTK."""
        _raise_vtk_import_error()

    def get_node_to_elem(*args, **kwargs):
        r"""Dummy symbol for missing VTK."""
        _raise_vtk_import_error()

    def get_fields_from_cell(*args, **kwargs):
        r"""Dummy symbol for missing VTK."""
        _raise_vtk_import_error()

    def get_fields(*args, **kwargs):
        r"""Dummy symbol for missing VTK."""
        _raise_vtk_import_error()

    def get_vertices(*args, **kwargs):
        r"""Dummy symbol for missing VTK."""
        _raise_vtk_import_error()

    def get_volume_data(*args, **kwargs):
        r"""Dummy symbol for missing VTK."""
        _raise_vtk_import_error()

    def get_surface_data(*args, **kwargs):
        r"""Dummy symbol for missing VTK."""
        _raise_vtk_import_error()
