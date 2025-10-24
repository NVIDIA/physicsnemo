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

import os
import random

import pytest
from pytest_utils import import_or_fail


@pytest.fixture
def cgns_data_dir(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/sample_formats")


@import_or_fail(["vtk", "warp"])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_mesh_datapipe(device, tmp_path, pytestconfig):
    """Tests the MeshDatapipe class with VTP and VTU files."""

    import vtk

    from physicsnemo.datapipes.cae import MeshDatapipe

    def _create_random_vtp_vtu_mesh(
        num_points: int, num_triangles: int, dir: str
    ) -> tuple:
        """
        Create a random VTP (VTK PolyData) mesh and a random VTU (VTK Unstructured Grid) mesh with triangles.

        Parameters:
            num_points (int): Number of random points.
            num_triangles (int): Number of triangles.
            dir (str): Directory to save the VTP and VTU files.

        Returns:
            tuple: A tuple containing the random VTP mesh (vtk.vtkPolyData) and the random VTU mesh (vtk.vtkUnstructuredGrid).
        """

        # make directory if it does not exist
        os.makedirs(dir, exist_ok=True)

        # Create random points
        points = vtk.vtkPoints()
        for _ in range(num_points):
            x, y, z = (
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(-10, 10),
            )
            points.InsertNextPoint(x, y, z)

        # Create triangles
        triangles = vtk.vtkCellArray()
        for _ in range(num_triangles):
            p1, p2, p3 = (
                random.randint(0, num_points - 1),
                random.randint(0, num_points - 1),
                random.randint(0, num_points - 1),
            )
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, p1)
            triangle.GetPointIds().SetId(1, p2)
            triangle.GetPointIds().SetId(2, p3)
            triangles.InsertNextCell(triangle)

        # Create a PolyData object (VTP mesh)
        vtp_mesh = vtk.vtkPolyData()
        vtp_mesh.SetPoints(points)
        vtp_mesh.SetPolys(triangles)

        # Assign random scalar values (features) to points in VTP mesh
        scalar_values = vtk.vtkDoubleArray()
        scalar_values.SetName("RandomFeatures")
        for _ in range(num_points):
            scalar_values.InsertNextValue(random.uniform(0, 1))
        vtp_mesh.GetPointData().SetScalars(scalar_values)

        # Write VTP mesh to file
        vtp_writer = vtk.vtkXMLPolyDataWriter()
        vtp_writer.SetFileName(os.path.join(dir, "random.vtp"))
        vtp_writer.SetInputData(vtp_mesh)
        vtp_writer.Write()

        # Create an Unstructured Grid (VTU mesh)
        vtu_mesh = vtk.vtkUnstructuredGrid()
        vtu_mesh.SetPoints(points)
        vtu_mesh.SetCells(vtk.VTK_TRIANGLE, triangles)

        # Assign random scalar values (features) to points in VTU mesh
        scalar_values = vtk.vtkDoubleArray()
        scalar_values.SetName("RandomFeatures")  # Set the name
        for _ in range(num_points):
            scalar_values.InsertNextValue(random.uniform(0, 1))
        vtu_mesh.GetPointData().SetScalars(scalar_values)

        # Write VTU mesh to file
        vtu_writer = vtk.vtkXMLUnstructuredGridWriter()
        vtu_writer.SetFileName(os.path.join(dir, "random.vtu"))
        vtu_writer.SetInputData(vtu_mesh)
        vtu_writer.Write()

    tmp_dir = tmp_path / "temp_data"
    tmp_dir.mkdir()
    _create_random_vtp_vtu_mesh(num_points=10, num_triangles=20, dir=tmp_dir)
    datapipe_vtp = MeshDatapipe(
        data_dir=tmp_dir,
        variables=["RandomFeatures"],
        num_variables=1,
        file_format="vtp",
        stats_dir=None,
        batch_size=1,
        num_samples=1,
        shuffle=True,
        num_workers=1,
        device=device,
    )

    assert len(datapipe_vtp) == 1
    for data in datapipe_vtp:
        assert data[0]["vertices"].shape == (1, 10, 3)
        assert data[0]["x"].shape == (1, 10, 1)

    datapipe_vtu = MeshDatapipe(
        data_dir=tmp_dir,
        variables=["RandomFeatures"],
        num_variables=1,
        file_format="vtu",
        stats_dir=None,
        batch_size=1,
        num_samples=1,
        shuffle=True,
        num_workers=1,
        device=device,
    )

    assert len(datapipe_vtu) == 1
    for data in datapipe_vtu:
        assert data[0]["vertices"].shape == (1, 10, 3)
        assert data[0]["x"].shape == (1, 10, 1)


# @import_or_fail(["vtk"])
# @pytest.mark.parametrize("device", ["cuda", "cpu"])
# def test_mesh_datapipe_cgns(device, cgns_data_dir, pytestconfig):
#     """Tests the mesh datapipe for CGNS file format."""
#     datapipe_cgns = MeshDatapipe(
#         data_dir=cgns_data_dir,
#         variables=[],
#         num_variables=0,
#         file_format="cgns",
#         stats_dir=None,
#         batch_size=1,
#         num_samples=1,
#         shuffle=True,
#         num_workers=1,
#         device=device,
#     )

#     assert len(datapipe_cgns) == 1
#     for data in datapipe_cgns:
#         assert data[0]["vertices"].shape == (1, 502, 3)
