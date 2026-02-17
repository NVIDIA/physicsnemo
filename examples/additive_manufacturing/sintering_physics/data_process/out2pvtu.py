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

# © Copyright 2023 HP Development Company, L.P.

"""
This file takes the predicted output stored in /rollouts
convert into the .vtu format, that can be processed by simulation software
"""

import glob
import logging as log
import os
import pickle
import sys

import numpy as np
import pyvista as pv
import vtk
from rawdata2tfrecord_large_ts import get_solution_id

log.basicConfig(
    format="%(asctime)s - %(levelname)s\t%(message)s",
    datefmt="%I:%M:%S %p",
    level=log.INFO,
)


def write(name, obj):
    """Write output to vtk"""
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(name)
    writer.SetInputData(obj)
    writer.Update()
    writer.Write()
    log.info(f"Saved {name}")


def pv_get_data_position(data):
    """
    Read the sample solution.pvtu file, get the position indexes for predicted data updates
    """

    points = data.points
    n_points = points.shape[0]
    print(
        "complete points shape: ", points.shape
    )  # i.e. all points shape:  (134464, 3)
    uvw_values = data["displacement_U"]
    print("complete ori uvw_values shape: ", uvw_values.shape)

    pos_deformed_list = []  # [all unique deformed_positions]
    index_list = []  # [corresponding original first pid of the unique location]
    position_ids_dict = {}  # stores {xyz: [pid list of the same xyz location]}
    for p_id in range(n_points):
        # Read the point xyz-location
        point_xyz = data.GetPoint(p_id)

        if point_xyz not in position_ids_dict:
            # if the xyz-location point not stored before, read its uvw value, init dict
            position_ids_dict[point_xyz] = [p_id]
            uvw = uvw_values[p_id]

            # Compute the deformed physical location from original physical location
            # todo: can postentially skip the deformation compute here
            deformed_pos = point_xyz + uvw

            index_list.append(p_id)
            pos_deformed_list.append(deformed_pos)
        else:
            # if point of location xyz already appeared, update the id list
            position_ids_dict[point_xyz].append(p_id)

    print("index_list: ", len(index_list))
    return position_ids_dict, np.array(pos_deformed_list), index_list


def vtk_get_data_position(file_path):
    """
    Read the basemesh geometry, return points with xyz location recording
    Args:
        file_path:

    Returns:

    """
    # baseMeshReader = vtk.vtkGenericDataObjectReader()
    baseMeshReader = vtk.vtkXMLPUnstructuredGridReader()
    baseMeshReader.SetFileName(file_path)
    baseMeshReader.Update()

    basemesh = baseMeshReader.GetOutput()
    x0Points = basemesh.GetPoints()
    print("vtk_get_data_position X0points : ", x0Points)

    return x0Points


def update_points(basemesh_points, metadata, new_pos, position_ids_dict, index_list):
    """
    new_pos: example_rollout['predicted_rollout'], shape:(predicted_time_steps, num_particles, dim)
            i.e. (5, 21969, 3)
    :return
        location_deform_map:{(xyz, <class 'tuple'>): array([uvw_ list])}
            i.e. check map:  {(4.0, 48.0, 1.0): array([-0.0399062 , -0.01175825, -0.01040107])}
    """
    # initialize uvw of same node number as basemesh
    new_uvw_array = np.zeros((basemesh_points.GetNumberOfPoints(), 3))
    print("new_uvw_array shape: ", new_uvw_array.shape)
    pos_mean, pos_std = metadata["pos_mean"], metadata["pos_std"]
    # need to store last timestep prediction only
    # todo: may change per requirements
    new_pos = new_pos[-1, ...]
    denormed_new_pos = new_pos * pos_std + pos_mean
    print("predicted_rollout shape: ", new_pos.shape)

    location_deform_map = {}
    # update the deformed position for each point
    for p_id, point_new_pos in enumerate(denormed_new_pos):
        # iterate the predicted non-duplicate points set, i.e. 13500
        # get the xyz index (in mm) for each point id, from the match in basemesh
        xyz = basemesh_points.GetPoint(index_list[p_id])

        # get the point p_ids of the same xyz location
        xyz_dup_pids = position_ids_dict[xyz]

        # denormed_point_new_pos = point_new_pos * pos_std + pos_mean
        uvw_ = point_new_pos - xyz
        for id_ in xyz_dup_pids:
            # update the new pos value for all nodes of this same xyz location
            new_uvw_array[id_, :] = uvw_

        location_deform_map[xyz] = uvw_
    return new_uvw_array, location_deform_map


def write_output(basemesh_path, new_uvw_array, end_inference_index, outPath):
    """

    Args:
        basemesh_path: out/mesh.pvtu file that contains the xyz geometry information
        new_uvw_array:

    Returns:

    """
    # new uvw value= [ 0.25397945  0.55836469 -0.79864266]
    # prepare vtk array that will be added to our points
    uvw_vtk_array = vtk.vtkDoubleArray()
    uvw_vtk_array.SetNumberOfComponents(3)
    uvw_vtk_array.SetName("displacement_U")
    # add uvw-displacement values to the array
    for index in range(len(new_uvw_array)):
        uvw = new_uvw_array[index, :]  #  [ 0.25397945  0.55836469 -0.79864266]
        uvw_vtk_array.InsertNextTuple(uvw)
        # uvw_vtk_array.InsertTuple(point_index, uvw)

    ##### read mesh file (to copy its cells)
    mesh_vtu_file_reader = vtk.vtkXMLPUnstructuredGridReader()
    mesh_vtu_file_reader.SetFileName(basemesh_path)
    mesh_vtu_file_reader.Update()
    mesh_vtu_file = mesh_vtu_file_reader.GetOutput()

    ##### prepare new solution object
    predicted_vtu_solution = vtk.vtkUnstructuredGrid()
    # copy cells of mesh vtu file
    predicted_vtu_solution.DeepCopy(mesh_vtu_file)
    # add uvw-displacement array to our new solution vtk object
    predicted_vtu_solution.GetPointData().AddArray(uvw_vtk_array)

    ##### save vtu file
    predicted_vtu_solution_writer = vtk.vtkXMLUnstructuredGridWriter()
    predicted_vtu_solution_writer.SetInputData(predicted_vtu_solution)
    predicted_file_path = os.path.join(
        outPath, "predicted-displacement-{}.vtu".format(end_inference_index)
    )
    predicted_vtu_solution_writer.SetFileName(predicted_file_path)
    predicted_vtu_solution_writer.Write()
    print("wrote ", predicted_file_path)
    return predicted_file_path


def save_volume_deformation(
    basemesh_path, new_uvw_array, end_inference_index, outPath, core_id
):
    """Save the predicted deformation"""
    uvw_vtk_array = vtk.vtkDoubleArray()
    uvw_vtk_array.SetNumberOfComponents(3)
    uvw_vtk_array.SetName("displacement_U")
    # add uvw-displacement values to the array
    for index in range(len(new_uvw_array)):
        uvw = new_uvw_array[index, :]  #  [ 0.25397945  0.55836469 -0.79864266]
        uvw_vtk_array.InsertNextTuple(uvw)

    # read mesh file (to copy its cells)
    # todo: check the local_ugrid/ mesh_vtu_file consistent with C code
    mesh_vtu_file_reader = vtk.vtkXMLUnstructuredGridReader()
    mesh_vtu_file_reader.SetFileName(basemesh_path)
    mesh_vtu_file_reader.Update()
    mesh_vtu_file = mesh_vtu_file_reader.GetOutput()

    predicted_vtu_solution = vtk.vtkUnstructuredGrid()
    predicted_vtu_solution.SetPoints(mesh_vtu_file.GetPoints())
    predicted_vtu_solution.SetCells(
        mesh_vtu_file.GetCellTypesArray(), mesh_vtu_file.GetCells()
    )
    predicted_vtu_solution.GetPointData().SetVectors(uvw_vtk_array)

    ##### Write
    predicted_vtu_solution_writer = vtk.vtkXMLUnstructuredGridWriter()
    predicted_vtu_solution_writer.SetInputData(predicted_vtu_solution)
    # update names with leading-0
    predicted_file_path = os.path.join(
        outPath,
        "predicted-displacement-{}-{}.vtu".format(
            str(core_id).rjust(4, "0"), end_inference_index
        ),
    )
    predicted_vtu_solution_writer.SetFileName(predicted_file_path)
    predicted_vtu_solution_writer.Write()

    return predicted_file_path


def save_volume_deformation_master_record(outPath, vtu_list, end_inference_index):
    """Save the predicted deformation master file"""

    print("process solution: ", vtu_list[0])
    master_path = os.path.join(
        outPath, "predicted-displacement-{}.pvtu".format(end_inference_index)
    )
    master_file = open(master_path, "w")

    master_file.write('<?xml version="1.0"?>\n')
    master_file.write("<!-- #This file was generated by virtual foundry -->\n")
    master_file.write(
        '<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
    )
    master_file.write('<PUnstructuredGrid GhostLevel="0">\n')
    master_file.write('  <PPointData Scalars="scalars">\n')
    master_file.write(
        '    <PDataArray type="Float64" Name="displacement_U" NumberOfComponents="3" format="ascii"/>\n'
    )
    master_file.write("  </PPointData>\n")
    master_file.write("  <PPoints>\n")
    master_file.write('   <PDataArray type="Float64" NumberOfComponents="3"/>\n')
    master_file.write("  </PPoints>\n")

    for i, vtu_name in enumerate(vtu_list):
        vtu_name = os.path.basename(vtu_name)
        master_file.write(
            '  <Piece Source="' + vtu_name + '"/>\n'
        )  # displacement-0000-1505.vtu

    master_file.write("</PUnstructuredGrid>\n")
    master_file.write("</VTKFile>\n")
    print("complete writing to master file: ", master_path)
    return


def post_process(
    raw_data_path, metadata, example_rollout, end_inference_index, outPath
):
    """
    Args:
        raw_data_path: Virtual Foundry output solution file folder,
            i.e."/home/rachel_chen/dataset/ladder_fast"
        metadata: metadata path with corresponding VFGN trained model ckpt version
        example_rollout: predicted rollout map data structure, contains keys below
            {'initial_positions':, 'predicted_rollout':, 'particle_types':, 'global_context':,
            ''ground_truth_rollout':, ''metadata': }
        end_inference_index:
        outPath: VFGN predicted output store path,
            i.e."learning_to_simulate/rollouts/test"

    Returns:
        predicted_file_path:
    """
    print(example_rollout.keys())
    print(
        example_rollout["predicted_rollout"].shape
    )  # i.e. (47, 21969, 3): (predicted_time_steps, num_nodes, xyz-dim)

    # Read sample geometry with xyz node locations
    # solution_list, dict_sol_time, temp_list = read_configs(raw_data_path)

    ### Get the basemesh to build geometry from
    build_path = os.path.join(raw_data_path, "out")
    solution_list = glob.glob(build_path + "/displacement-*.pvtu")
    solution_list = sorted(solution_list, key=get_solution_id)

    print("\nread basemesh from ", solution_list[0])
    solution_data = pv.read(solution_list[0])
    position_ids_dict, _, index_list = pv_get_data_position(solution_data)
    # print(pos_deformed_array.shape)  # (21969, 3)

    # Compare the readings from VTK library function
    print(
        "\nCompare VTK read basemesh  ",
    )
    basemesh_path = os.path.join(raw_data_path, "mesh", "mesh.pvtu")
    assert os.path.exists(basemesh_path), print(
        f"basemesh does not exist: {basemesh_path}"
    )
    basemesh_points = vtk_get_data_position(basemesh_path)  # mesh_0.vtu

    # Match each xyz-location node with its uvw-displacement value
    new_uvw_array, _ = update_points(
        basemesh_points,
        metadata,
        example_rollout["predicted_rollout"],
        position_ids_dict,
        index_list,
    )

    # write to a new vtu file
    predicted_file_path = write_output(
        basemesh_path, new_uvw_array, end_inference_index, outPath
    )
    print("Complete writing to new vtu file, stored in ", outPath)
    return predicted_file_path


if __name__ == "__main__":
    argv = sys.argv[1:]
    raw_data_path, rollout_data_path, end_inference_index, outPath = argv

    if not rollout_data_path:
        raise ValueError("A `rollout_path` must be passed.")
    with open(rollout_data_path, "rb") as file:
        example_rollout = pickle.load(file)

    post_process(
        raw_data_path,
        example_rollout["metadata"],
        example_rollout,
        end_inference_index,
        outPath,
    )
