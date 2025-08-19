# ignore_header_test
# ruff: noqa: E402

# © Copyright 2023 HP Development Company, L.P.
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


import csv
import glob
import os
import re

import numpy as np
import tensorflow as tf
from natsort import natsorted


def read_raw_folder(data_dir):
    """
    data_dir: Physics Simulation Engine raw output folder path, contains folder /out/solution-*.pvtu
    Read the Physics Simulation Engine raw output folder
    sort all timestep deformation files in time-series
    """
    build_path = os.path.join(data_dir, "out")
    solution_list = glob.glob(build_path + "/volume-deformation-*.pvtu")
    # solution_list = sorted(solution_list, key=get_solution_id)
    solution_list = natsorted(solution_list)
    assert len(solution_list) >= 3, (
        "Need to have at least 3 solution files as input to start prediction or analysis!"
    )
    return solution_list


def get_solution_id(solution_name):
    """
    Read the Physics simulation file, current version file name: solution-xx.pvtu (2023-June)
    Previous used version solution_name: volume-deformation, displacement-xx.pvtu

    return: sorted keys by int index
    """
    m = re.search("solution-(\d+)", solution_name)
    if m:
        id = int(m.group(1))
        return id
    return -1


def time_diff(sequence_array):
    """
    sequence_array: Position/ velocity/ acceleration numpy array,
    return:  step-wise difference
    """
    return sequence_array[1:, :] - sequence_array[:-1, :]


def _bytes_feature(value):
    """
    Returns a bytes_list from a string / byte.
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_int_feature(values):
    """Create/ convert tf.Int64 feature"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))


def get_radius(data):
    """
    From data read from simulation path by pv.read, compute the radius by max(XYZ-distance) of a cell
    Args:
        data: data from pv.read
            format: UnstructuredGrid i.e. (0x7f1636fe5520)
                  N Cells:      21970
                  N Points:     175760
                  X Bounds:     -4.600e+01, 3.000e+00
                  Y Bounds:     -4.500e+00, 4.500e+00
                  Z Bounds:     0.000e+00, 1.300e+01
                  N Arrays:     11
    Returns:
        float, i.e. 0.6 for meshsize=500
    """
    # Cell: vtkHexahedron i.e. cell_0 (0x55f4e9928f20)
    #   Debug: Off
    #   Modified Time: 85931
    #   Reference Count: 2
    #   Registered Events: (none)
    #   Number Of Points: 8
    #   Bounds:
    #     Xmin,Xmax: (-2.5, -2)
    #     Ymin,Ymax: (4, 4.5)
    #     Zmin,Zmax: (4, 4.5)
    #     Point ids are: 0, 1, 3, 2, 4, 5, 7, 6
    #   Merge Tolerance: 0.01
    # ....
    cell_0 = data.GetCell(0)

    bounds = np.array((cell_0.GetBounds()))

    # compute the distance of each xyz-dimension
    len_s = bounds[1::2] - bounds[0:-1:2]
    radius = 1.2 * np.max(len_s)
    return radius


def get_data_position(data):
    """
    For the data read from one displacement-id.pvtu file,
    iterate each point data, filter out the points in existed physical xyz-location
    store the non-repeating point's uvw_values in
    Args:
        data: data read from displacement-id.pvtu file

    Returns: array of non-repeating nodes' current physical location (original location + displacement)

    """
    points = data.points
    n_points = points.shape[0]

    # uvw_values: the feature name storing voxel deformation,
    # depend on physics engine version, could also be data['u__v__w'], or other version i.e. data["displacement_U"]
    uvw_values = data["u__v__w"]

    pos_list = []
    index_list = []

    for point_index in range(n_points):
        uvw = uvw_values[point_index]

        # Compute the deformed physical location from original physical location
        pos = uvw

        index_list.append(point_index)
        pos_list.append(pos)

    return np.array(pos_list), index_list


def read_configs(raw_data_path):
    """
    Read the dataset information (without using solution.pvtu.series file)

    """
    # For each build, read the temperature profile at every time step
    params_prm_path = os.path.join(raw_data_path, "params.prm")
    assert os.path.exists(params_prm_path), (
        f"Temperature profile file params.prm not exists! {params_prm_path}"
    )

    # reading csv file
    with open(params_prm_path, "r") as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting each data row one by one
        for idx, row in enumerate(csvreader):
            if "sintering_temperature_curve" in row[0]:
                # i.e. ['set sintering_temperature_curve=0', '20', '16320', '1380', '23520', '1380']
                temp_row = row[1:]
            elif "initial_time" in row[0]:
                initial_time = int(float(row[0].split("=")[1].strip()))
            elif "final_time" in row[0]:
                final_time = int(float(row[0].split("=")[1].strip()))
            elif "time_step" in row[0]:
                time_step = int(float(row[0].split("=")[1].strip()))
            elif "save_every_n_steps" in row[0]:
                save_every_n_steps = int(float(row[0].split("=")[1].strip()))

    # temp_curve_list = read_temperature(params_prm_path)
    # Add temperature data to each solution file
    temp_row.insert(0, "0")
    temp_curve_list = [float(i) for i in temp_row]
    # Get the stage separation time-temperature pairs
    stage_t_list = []
    stage_temp_list = []
    for idx in range(len(temp_curve_list) // 2):
        # t_list.append(int(temp_curve_list[idx*2]) / 3600)
        stage_t_list.append(int(temp_curve_list[idx * 2]))
        stage_temp_list.append(int(temp_curve_list[idx * 2 + 1]))

    return (initial_time, final_time, time_step, save_every_n_steps), temp_curve_list
