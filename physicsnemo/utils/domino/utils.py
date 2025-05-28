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
Important utilities for data processing and training, testing DoMINO.
"""

import os
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import KDTree

from physicsnemo.utils.profiling import profile

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


try:
    import pyvista as pv

    PV_AVAILABLE = True
except ImportError:
    PV_AVAILABLE = False
try:
    import vtk
    from vtk import vtkDataSetTriangleFilter
    from vtk.util import numpy_support

    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

# Define a typing that works for both numpy and cupy
if CUPY_AVAILABLE:
    ArrayType = Union[np.ndarray, cp.ndarray]
else:
    # Or just numpy, if cupy is not available.
    ArrayType = np.ndarray


def array_type(arr: ArrayType):
    """Function to return the array type.  It's just leveraging
    cupy to do this if available, fallback is numpy.
    """
    if CUPY_AVAILABLE:
        return cp.get_array_module(arr)
    else:
        return np


def calculate_center_of_mass(stl_centers: ArrayType, stl_sizes: ArrayType) -> ArrayType:
    """Function to calculate center of mass"""
    xp = array_type(stl_centers)
    stl_sizes = xp.expand_dims(stl_sizes, -1)
    center_of_mass = xp.sum(stl_centers * stl_sizes, axis=0) / xp.sum(stl_sizes, axis=0)
    return center_of_mass


def normalize(field: ArrayType, mx: ArrayType, mn: ArrayType) -> ArrayType:
    """Function to normalize fields"""
    return 2.0 * (field - mn) / (mx - mn) - 1.0


def unnormalize(field: ArrayType, mx: ArrayType, mn: ArrayType) -> ArrayType:
    """Function to unnormalize fields"""
    return (field + 1.0) * (mx - mn) * 0.5 + mn


def standardize(field: ArrayType, mean: ArrayType, std: ArrayType) -> ArrayType:
    """Function to standardize fields"""
    return (field - mean) / std


def unstandardize(field: ArrayType, mean: ArrayType, std: ArrayType) -> ArrayType:
    """Function to unstandardize fields"""
    return field * std + mean


def write_to_vtp(polydata: Any, filename: str):
    """Function to write polydata to vtp"""
    if not VTK_AVAILABLE:
        raise ImportError("VTK or is not installed. This function cannot be used.")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


def write_to_vtu(polydata: Any, filename: str):
    """Function to write polydata to vtu"""
    if not VTK_AVAILABLE:
        raise ImportError("VTK or is not installed. This function cannot be used.")
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


def extract_surface_triangles(tet_mesh: Any) -> List[int]:
    """Extracts the surface triangles from a triangular mesh."""
    if not VTK_AVAILABLE:
        raise ImportError("VTK or is not installed. This function cannot be used.")
    if not PV_AVAILABLE:
        raise ImportError("PyVista is not installed. This function cannot be used.")
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


def convert_to_tet_mesh(polydata: Any) -> Any:
    """Function to convert tet to stl"""
    if not VTK_AVAILABLE:
        raise ImportError("VTK or is not installed. This function cannot be used.")
    # Create a VTK DataSetTriangleFilter object
    tet_filter = vtkDataSetTriangleFilter()
    tet_filter.SetInputData(polydata)
    tet_filter.Update()  # Update to apply the filter

    # Get the output as an UnstructuredGrid
    # tet_mesh = pv.wrap(tet_filter.GetOutput())
    tet_mesh = tet_filter.GetOutput()
    return tet_mesh


def get_node_to_elem(polydata: Any) -> Any:
    """Function to convert node to elem"""
    if not VTK_AVAILABLE:
        raise ImportError("VTK or is not installed. This function cannot be used.")
    c2p = vtk.vtkPointDataToCellData()
    c2p.SetInputData(polydata)
    c2p.Update()
    cell_data = c2p.GetOutput()
    return cell_data


def get_fields_from_cell(ptdata, var_list):
    """Function to get fields from elem"""
    fields = []
    for var in var_list:
        variable = ptdata.GetArray(var)
        num_tuples = variable.GetNumberOfTuples()
        cell_fields = []
        for j in range(num_tuples):
            variable_value = np.array(variable.GetTuple(j))
            cell_fields.append(variable_value)
        cell_fields = np.asarray(cell_fields)
        fields.append(cell_fields)
    fields = np.transpose(np.asarray(fields), (1, 0))

    return fields


def get_fields(data, variables):
    """Function to get fields from VTP/VTU"""
    if not VTK_AVAILABLE:
        raise ImportError("VTK or is not installed. This function cannot be used.")
    fields = []
    for array_name in variables:
        try:
            array = data.GetArray(array_name)
        except ValueError:
            raise ValueError(
                f"Failed to get array {array_name} from the unstructured grid."
            )
        array_data = numpy_support.vtk_to_numpy(array).reshape(
            array.GetNumberOfTuples(), array.GetNumberOfComponents()
        )
        fields.append(array_data)
    return fields


def get_vertices(polydata):
    """Function to get vertices"""
    if not VTK_AVAILABLE:
        raise ImportError("VTK or is not installed. This function cannot be used.")
    points = polydata.GetPoints()
    vertices = numpy_support.vtk_to_numpy(points.GetData())
    return vertices


def get_volume_data(polydata, variables):
    """Function to get volume data"""
    vertices = get_vertices(polydata)
    point_data = polydata.GetPointData()

    fields = get_fields(point_data, variables)

    return vertices, fields


def get_surface_data(polydata, variables):
    """Function to get surface data"""
    if not VTK_AVAILABLE:
        raise ImportError("VTK or is not installed. This function cannot be used.")
    points = polydata.GetPoints()
    vertices = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])

    point_data = polydata.GetPointData()
    fields = []
    for array_name in variables:
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
    coordinates_b: Optional[ArrayType] = None,
    cell_length: Sequence[float] = [],
) -> ArrayType:
    """Function to get normal positional encoding"""
    dx = cell_length[0]
    dy = cell_length[1]
    dz = cell_length[2]
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


def nd_interpolator(coodinates, field, grid):
    """Function to for nd interpolation"""
    # TODO - this function should get updated for cuml if using cupy.
    interp_func = KDTree(coodinates[0])
    dd, ii = interp_func.query(grid, k=2)

    field_grid = field[ii]
    field_grid = np.float32(np.mean(field_grid, (3)))
    return field_grid


def pad(arr: ArrayType, npoin: int, pad_value: float = 0.0) -> ArrayType:
    """Function for padding"""
    xp = array_type(arr)
    arr_pad = pad_value * xp.ones(
        (npoin - arr.shape[0], arr.shape[1]), dtype=xp.float32
    )
    arr_padded = xp.concatenate((arr, arr_pad), axis=0)
    return arr_padded


def pad_inp(arr: ArrayType, npoin: int, pad_value: float = 0.0) -> ArrayType:
    """Function for padding arrays"""
    xp = array_type(arr)
    arr_pad = pad_value * xp.ones(
        (npoin - arr.shape[0], arr.shape[1], arr.shape[2]), dtype=xp.float32
    )
    arr_padded = xp.concatenate((arr, arr_pad), axis=0)
    return arr_padded


@profile
def shuffle_array(
    arr: ArrayType,
    npoin: int,
) -> Tuple[ArrayType, ArrayType]:
    """Function for shuffling arrays"""
    xp = array_type(arr)
    if npoin > arr.shape[0]:
        # If asking too many points, truncate the ask but still shuffle.
        npoin = arr.shape[0]
    idx = xp.random.choice(arr.shape[0], size=npoin, replace=False)
    return arr[idx], idx


def shuffle_array_without_sampling(arr: ArrayType) -> Tuple[ArrayType, ArrayType]:
    """Function for shuffline arrays without sampling."""
    xp = array_type(arr)
    idx = xp.arange(arr.shape[0])
    xp.random.shuffle(idx)
    return arr[idx], idx


def create_directory(filepath: str) -> None:
    """Function to create directories"""
    if not os.path.exists(filepath):
        os.makedirs(filepath)


def get_filenames(filepath: str, exclude_dirs: bool = False) -> List[str]:
    """Function to get filenames from a directory"""
    if os.path.exists(filepath):
        filenames = []
        for item in os.listdir(filepath):
            item_path = os.path.join(filepath, item)
            if exclude_dirs and os.path.isdir(item_path):
                # Include directories ending with .zarr even when exclude_dirs is True
                if item.endswith(".zarr"):
                    filenames.append(item)
                continue
            filenames.append(item)
        return filenames
    else:
        FileNotFoundError()


def calculate_pos_encoding(nx: ArrayType, d: int = 8) -> ArrayType:
    """Function for calculating positional encoding"""
    vec = []
    xp = array_type(nx)
    for k in range(int(d / 2)):
        vec.append(xp.sin(nx / 10000 ** (2 * (k) / d)))
        vec.append(xp.cos(nx / 10000 ** (2 * (k) / d)))
    return vec


def combine_dict(old_dict, new_dict):
    """Function to combine dictionaries"""
    for j in old_dict.keys():
        old_dict[j] += new_dict[j]
    return old_dict


def merge(*lists):
    """Function to merge lists"""
    newlist = lists[:]
    for x in lists:
        if x not in newlist:
            newlist.extend(x)
    return newlist


def create_grid(mx: ArrayType, mn: ArrayType, nres: ArrayType) -> ArrayType:
    """Function to create grid"""

    xp = array_type(mx)

    dx = xp.linspace(mn[0], mx[0], nres[0], dtype=mx.dtype)
    dy = xp.linspace(mn[1], mx[1], nres[1], dtype=mx.dtype)
    dz = xp.linspace(mn[2], mx[2], nres[2], dtype=mx.dtype)

    xv, yv, zv = xp.meshgrid(dx, dy, dz)
    xv = xp.expand_dims(xv, -1)
    yv = xp.expand_dims(yv, -1)
    zv = xp.expand_dims(zv, -1)
    grid = xp.concatenate((xv, yv, zv), axis=-1)
    grid = xp.transpose(grid, (1, 0, 2, 3))

    return grid


def mean_std_sampling(
    field: ArrayType, mean: ArrayType, std: ArrayType, tolerance: float = 3.0
) -> ArrayType:
    """Function for mean/std based sampling"""
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


def dict_to_device(state_dict, device, exclude_keys=["filename"]):
    """Function to load dictionary to device"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k not in exclude_keys:
            new_state_dict[k] = v.to(device)
    return new_state_dict


def area_weighted_shuffle_array(
    arr: ArrayType, npoin: int, area: ArrayType
) -> Tuple[ArrayType, ArrayType]:
    """Function for area weighted shuffling"""
    xp = array_type(arr)
    # Compute the total_area:
    factor = 1.0
    total_area = xp.sum(area**factor)
    probs = area**factor / total_area

    if npoin > arr.shape[0]:
        npoin = arr.shape[0]

    idx = xp.arange(arr.shape[0])

    # This is too memory intensive to run on the GPU.
    if xp == cp:
        idx = idx.get()
        probs = probs.get()
        # Under the hood, this has a search over the probabilities.
        # It's very expensive in memory, as far as I can tell.
        # In principle, we could use the Alias method to speed this up
        # on the GPU but it's not yet a bottleneck.

        ids = np.random.choice(idx, npoin, p=probs)
        ids = xp.asarray(ids)
    else:
        # Chug along on the CPU:
        ids = xp.random.choice(idx, npoin, p=probs)

    return arr[ids], ids
