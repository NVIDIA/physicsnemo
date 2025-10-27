# ignore_header_test
# Copyright 2023 Stanford University
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
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
import numpy as np


def read_geo(fname):
    """
    Read geometry from file.

    Arguments:
        fname: File name
    Returns:
        The vtk reader

    """
    _, ext = os.path.splitext(fname)
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError("File extension " + ext + " unknown.")
    reader.SetFileName(fname)
    reader.Update()
    return reader


def get_all_arrays(geo, components=None):
    """
    Get arrays from geometry file.

    Arguments:
        geo: Input geometry
        components (int): Number of array components to keep.
                          Default: None -> keep all
    Returns:
        Point data dictionary (key: array name, value: numpy array)
        Cell data dictionary (key: array name, value: numpy array)
        Points (numpy array)

    """
    # collect all arrays
    cell_data = collect_arrays(geo.GetCellData(), components)
    point_data = collect_arrays(geo.GetPointData(), components)
    points = collect_points(geo.GetPoints(), components)
    return point_data, cell_data, points


def get_edges(geo):
    """
    Get edges from geometry file.

    Arguments:
        geo: Input geometry

    Returns:
        List of nodes indices (first nodes in each edge)
        List of nodes indices (second nodes in each edge)

    """
    edges1 = []
    edges2 = []
    ncells = geo.GetNumberOfCells()
    for i in range(ncells):
        edges1.append(int(geo.GetCell(i).GetPointIds().GetId(0)))
        edges2.append(int(geo.GetCell(i).GetPointIds().GetId(1)))

    return np.array(edges1), np.array(edges2)


def collect_arrays(celldata, components=None):
    """
    Collect arrays from a cell data or point data object.

    Arguments:
        celldata: Input data
        components (int): Number of array components to keep.
                          Default: None -> keep all
    Returns:
        A dictionary of arrays (key: array name, value: numpy array)

    """
    res = {}
    for i in range(celldata.GetNumberOfArrays()):
        name = celldata.GetArrayName(i)
        data = celldata.GetArray(i)
        if components == None:
            res[name] = v2n(data).astype(np.float32)
        else:
            res[name] = v2n(data)[:components].astype(np.float32)
    return res


def collect_points(celldata, components=None):
    """
    Collect points from a cell data object.

    Arguments:
        celldata: Name of the directory
        components (int): Number of array components to keep.
                          Default: None -> keep allNone
    Returns:
        The array of points (numpy array)

    """
    if components == None:
        res = v2n(celldata.GetData()).astype(np.float32)
    else:
        res = v2n(celldata.GetData())[:components].astype(np.float32)
    return res


def gather_array(arrays, arrayname, mintime=1e-12):
    """
    Given a dictionary of numpy arrays, this method gathers all the arrays
    containing a certain substring in the array name.

    Arguments:
        arrays: Arrays look into.
        arrayname (string): Substring to look for.
        mintime (float): Minimum time to consider. Default value = 1e-12.
    Returns:
        Dictionary of arrays (key: time, value: numpy array)

    """
    out = {}
    for array in arrays:
        if arrayname in array:
            time = float(array.replace(arrayname + "_", ""))
            if time > mintime:
                out[time] = arrays[array]

    return out
