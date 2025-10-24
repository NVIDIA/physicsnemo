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
This code processes mesh data from .vtu and .stl (or .vtp) files to create
voxel grids for large-scale simulations. The process involves converting
unstructured grids (from .vtu files) into voxel grids, extracting surface
triangles and vertices from the mesh files, and calculating the signed distance
field (SDF) and its derivatives (DSDF). The SDF is computed using the mesh surface
and the voxel grid. The resulting data, which includes voxel vertices, SDF, DSDF,
velocity (U), and pressure (p), is saved in an HDF5 format for training. The code
supports multiprocessing to process multiple files concurrently and can optionally
save the voxel grids as .vti files for debugging or visualization.
"""

import vtk
import pyvista as pv
import numpy as np
import h5py
import os
import hydra

from multiprocessing import Pool
from tqdm import tqdm
from pyvista.core import _vtk_core as _vtk
from vtk import vtkDataSetTriangleFilter
from physicsnemo.datapipes.cae.readers import read_vtp, read_vtu, read_stl
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from sdf import signed_distance_field


def unstructured2voxel(
    unstructured_grid, grid_size, bounds, write=False, output_filename="image.vti"
):
    """Converts an unstructured grid to a voxel grid (structured grid) using resampling."""
    resampler = vtk.vtkResampleToImage()
    resampler.AddInputDataObject(unstructured_grid)
    resampler.UseInputBoundsOff()
    resampler.SetSamplingDimensions(*grid_size)

    if not bounds:
        bounds = unstructured_grid.GetBounds()
    resampler.SetSamplingBounds(bounds)

    resampler.Update()
    voxel_grid = resampler.GetOutput()

    if write:
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(output_filename)
        writer.SetInputData(voxel_grid)
        writer.Write()

    return voxel_grid


def convert_to_triangular_mesh(
    polydata, write=False, output_filename="surface_mesh_triangular.vtu"
):
    """Converts a vtkPolyData object to a triangular mesh."""
    tet_filter = vtkDataSetTriangleFilter()
    tet_filter.SetInputData(polydata)
    tet_filter.Update()

    tet_mesh = pv.wrap(tet_filter.GetOutput())

    if write:
        tet_mesh.save(output_filename)

    return tet_mesh


def extract_surface_triangles(tet_mesh):
    """Extracts the surface triangles from a triangular mesh."""
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(tet_mesh)
    surface_filter.Update()

    surface_mesh = pv.wrap(surface_filter.GetOutput())
    triangle_indices = []
    faces = surface_mesh.faces.reshape((-1, 4))
    for face in faces:
        if face[0] == 3:
            triangle_indices.extend([face[1], face[2], face[3]])
        else:
            raise ValueError("Face is not a triangle")

    return triangle_indices


def fetch_mesh_vertices(mesh):
    """Fetches the vertices of a mesh."""
    points = mesh.GetPoints()
    num_points = points.GetNumberOfPoints()
    vertices = [points.GetPoint(i) for i in range(num_points)]
    return vertices


def get_cell_centers(voxel_grid):
    """Extracts the cell centers from a voxel grid."""
    cell_centers_filter = vtk.vtkCellCenters()
    cell_centers_filter.SetInputData(voxel_grid)
    cell_centers_filter.Update()

    cell_centers = cell_centers_filter.GetOutput()
    points = cell_centers.GetPoints()
    centers = [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
    return np.array(centers)


def process_file(task):
    """Process a single pair of VTU and VTP/STL files and save the output."""
    (
        vtu_path,
        surface_mesh_path,
        grid_size,
        bounds,
        output_dir,
        surface_mesh_file_format,
        save_vti,
    ) = task

    vtu_mesh = read_vtu(vtu_path)

    grid_size_expanded = tuple(
        s + 1 for s in grid_size
    )  # Add 1 to each dimension for the voxel grid
    voxel_grid = unstructured2voxel(vtu_mesh, grid_size_expanded, bounds)
    if surface_mesh_file_format == "vtp":
        surface_mesh = read_vtp(surface_mesh_path)
        surface_mesh = convert_to_triangular_mesh(surface_mesh)
    else:
        surface_mesh = read_stl(surface_mesh_path)
    triangle_indices = extract_surface_triangles(surface_mesh)
    surface_vertices = fetch_mesh_vertices(surface_mesh)
    volume_vertices = get_cell_centers(voxel_grid)

    sdf, dsdf = signed_distance_field(
        surface_vertices, triangle_indices, volume_vertices, include_hit_points=True
    )

    sdf = sdf.numpy()
    dsdf = dsdf.numpy()
    dsdf = -(dsdf - volume_vertices)
    dsdf = dsdf / np.linalg.norm(dsdf, axis=1, keepdims=True)

    voxel_grid = pv.wrap(voxel_grid).point_data_to_cell_data()
    data = voxel_grid.cell_data
    U = _vtk.vtk_to_numpy(data["UMeanTrim"])
    p = _vtk.vtk_to_numpy(data["pMeanTrim"])

    # Reshape the arrays according to the voxel grid dimensions
    volume_vertices = np.transpose(volume_vertices)
    sdf = np.expand_dims(sdf, axis=0)
    U = np.transpose(U)
    p = np.expand_dims(p, axis=0)
    volume_vertices = volume_vertices.reshape(3, *grid_size, order="F")
    sdf = sdf.reshape(1, *grid_size, order="F")
    dsdf = np.transpose(dsdf)
    dsdf = dsdf.reshape(3, *grid_size, order="F")
    U = U.reshape(3, *grid_size, order="F")
    p = p.reshape(1, *grid_size, order="F")

    # Create a merged array maintaining the voxel shape
    merged_array = np.concatenate([volume_vertices, sdf, dsdf, U, p], axis=0)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(
        output_dir, os.path.basename(vtu_path).replace(".vtu", ".h5")
    )

    with h5py.File(output_filename, "w") as hf:
        hf.create_dataset("data", data=merged_array)

    # Optionally save voxel grid as .vti for debugging
    if save_vti:
        voxel_grid.cell_data["SDF"] = sdf.flatten(order="F")
        voxel_grid.cell_data["DSDFx"] = dsdf[0].flatten(order="F")
        voxel_grid.cell_data["DSDFy"] = dsdf[1, :].flatten(order="F")
        voxel_grid.cell_data["DSDFz"] = dsdf[2, :].flatten(order="F")
        vti_filename = os.path.join(
            output_dir, os.path.basename(vtu_path).replace(".vtu", ".vti")
        )
        voxel_grid.save(vti_filename)


def process_directory(
    data_path,
    output_base_path,
    grid_size,
    bounds=None,
    surface_mesh_file_format="stl",
    num_workers=16,
    save_vti=False,
):
    """Process all VTU and VTP files in the given directory using multiprocessing with progress tracking."""
    tasks = []
    for root, _, files in os.walk(data_path):
        vtu_files = [f for f in files if f.endswith(".vtu")]
        for vtu_file in vtu_files:
            vtu_path = os.path.join(root, vtu_file)
            if surface_mesh_file_format == "vtp":
                surface_mesh_path = vtu_path.replace(".vtu", ".vtp")
            elif surface_mesh_file_format == "stl":
                vtu_id = vtu_file[len("volume_") : -len(".vtu")]  # Extract the ID part
                surface_mesh_file = f"drivaer_{vtu_id}.stl"
                surface_mesh_path = os.path.join(root, surface_mesh_file)
            else:
                raise ValueError(
                    f"Unsupported surface mesh file format: {surface_mesh_file_format}"
                )

            if os.path.exists(surface_mesh_path):
                relative_path = os.path.relpath(root, data_path)
                output_dir = os.path.join(output_base_path, relative_path)
                tasks.append(
                    (
                        vtu_path,
                        surface_mesh_path,
                        grid_size,
                        bounds,
                        output_dir,
                        surface_mesh_file_format,
                        save_vti,
                    )
                )
            else:
                print(
                    f"Warning: Corresponding surface mesh file not found for {vtu_path}"
                )

    # Use multiprocessing to process the tasks with progress tracking
    with Pool(num_workers) as pool:
        # Use imap_unordered to process tasks as they complete
        for _ in tqdm(
            pool.imap_unordered(process_file, tasks),
            total=len(tasks),
            desc="Processing Files",
            unit="file",
        ):
            pass


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    process_directory(
        to_absolute_path(cfg.data_path),
        to_absolute_path(cfg.h5_path),
        (cfg.num_voxels_x, cfg.num_voxels_y, cfg.num_voxels_z),
        (cfg.grid_origin_x, cfg.grid_origin_y, cfg.grid_origin_z),
        surface_mesh_file_format="stl",
        num_workers=cfg.num_preprocess_workers,
        save_vti=cfg.save_vti,
    )


if __name__ == "__main__":
    main()
