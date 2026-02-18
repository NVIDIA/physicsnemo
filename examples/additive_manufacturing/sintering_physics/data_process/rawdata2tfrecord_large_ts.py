# ignore_header_test
# ruff: noqa: E402

# © Copyright 2023 HP Development Company, L.P.
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


"""
Include some basic functions for simulation data processings.
Convert simulation output displacement-*.pvtu files to tfRecord,
Store for model training

usage:
python rawdata2tfrecord.py data-root meta-file-root
i.e.
python rawdata2tfrecord.py "/home/rachel_chen/dataset/Virtual-Foundry" "./"
python rawdata2tfrecord.py "/home/lopezca/repos/sintervox-models/dl-models"
"""

import csv
import glob
import json
import logging
import os
import sys

import numpy as np
import pyvista as pv
import tensorflow as tf
from natsort import natsorted
from sklearn import neighbors

logging.basicConfig(filename="DS-retrain-2403.log", level=logging.DEBUG)

import hydra
import utils
from omegaconf import DictConfig
from utils import time_diff

# Create a description of the features.
_FEATURE_DESCRIPTION = {
    "position": tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT["step_context"] = tf.io.VarLenFeature(
    tf.string
)

_FEATURE_DTYPES = {
    "position": {"in": np.float64, "out": tf.float64},
    "step_context": {"in": np.float64, "out": tf.float64},
}

_CONTEXT_FEATURES = {
    "key": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "particle_type": tf.io.VarLenFeature(tf.string),
    "senders": tf.io.VarLenFeature(tf.string),
    "receivers": tf.io.VarLenFeature(tf.string),
}


def arrange_data(data):
    """
    Organizes data from a structured format into a dictionary keyed by
    point coordinates.
    """
    arranged_data = {}

    name2array = {}
    for array_name in data.array_names:
        name2array[array_name] = data[array_name]

    # Get all points in the solution file
    points = data.points

    # Number of points in one solution file
    n_points = points.shape[0]
    for point_index in range(n_points):
        point = data.GetPoint(point_index)
        if point not in arranged_data:
            data_item = {}
            for array_name in name2array:
                data_item[array_name] = name2array[array_name][point_index]
            arranged_data[point] = data_item
    return arranged_data


def get_data_position(data):
    """
    For the data read from one displacement-id.pvtu file,
    iterate each point data, filter out the points in existed physical xyz-location
    store the non-repeating point's uvw_values
    Args:
        data: data read from displacement-id.pvtu file with pv.read
            format: UnstructuredGrid i.e. (0x7f1636fe5520)
                  N Cells:      21970
                  N Points:     175760
                  X Bounds:     -4.600e+01, 3.000e+00
                  Y Bounds:     -4.500e+00, 4.500e+00
                  Z Bounds:     0.000e+00, 1.300e+01
                  N Arrays:     11

    Returns: array of non-repeating nodes' current physical location (original location + displacement)
        pos_list -> np.array
        index_list -> list of same size
    """
    # each points' coordinate, and displacement, shape [# point, dim], i.e. (175760, 3)
    points = data.points

    uvw_values = data["u__v__w"]

    try:
        points.shape == uvw_values.shape
    except:
        logging.error(
            f"pv.read solution file field failed {data['u__v__w']} dimension not matching "
        )
        raise

    # Construct a dictionary, store physical location {xyz: boolean}
    arranged_data = {}
    position_list = []
    index_list = []
    for point_index in range(points.shape[0]):
        # Read coordinates of each point (x_coor, y_coor, z_coor)
        # i.e.  point_index: 168395, point_coor = (-31.0, -1.5, 0.0)
        point_coor = data.GetPoint(point_index)

        # if there's not record of this point_coordinate, add; else skip to avoid duplicated points
        if point_coor not in arranged_data:
            # read displacement of this point
            uvw = uvw_values[point_index]
            # Compute the deformed physical location from original physical location
            pos = point_coor + uvw

            index_list.append(point_index)
            position_list.append(pos)
            arranged_data[point_coor] = True

    return np.array(position_list), index_list


def read_sol_time(series_file):
    """
    Read the solution.pvtu.series file
    Returns:
        Dictionary contains the sintering simulation file name, and corresponding timestamp for the simulation file
    """
    dict_sol_time = {}
    time_list = []
    with open(series_file, "r") as fobj:
        data = json.load(fobj)

    for idx, item in enumerate(data["files"]):
        time_list.append(item["time"])
        dict_sol_time[item["name"]] = [item["time"]]

    return dict_sol_time


def read_temperature(temp_file):
    """
    Open and read the temperature profile from params.prm file under each build data.
    Save Temp for each time solution file.
    Returns:
        list of temperature value, corresponding to each solution file (sorted with time)
    """
    # reading csv file
    logging.info(f"read temperature file from path: {temp_file}")
    with open(temp_file, "r") as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting each data row one by one
        for idx, row in enumerate(csvreader):
            if row[0].find("temperature_curve") != -1:
                temp_row = row[1:]
                break

    # Add temperature data to each solution file
    temp_row.insert(0, "0")
    return temp_row


def get_solution_temperature_customer(solution_path, dict_sol_time, temp_curve_list):
    """
    This function read each solution file, determine the time, Temperature of this file
    Args:
        solution_path: the solution.pvtu file to be processed
        dict_sol_time: Dictionary contains the {solution_fname: simulation_time}
        temp_curve_list: Read from the params file,
         i.e. ['0', '0', '600', '20', '6000', '200', '18000', '400', '32400', '400', '38400', '600', '43800', '1050', '51000', '1050', '55140', '1395', '62340', '1395', '70680', '700']

    Returns:
        Temperature at the simulation time -> float
    """
    t_list = []
    temp_list = []
    # Get the stage separation time-temperature pairs
    for idx in range(len(temp_curve_list) // 2):
        t_list.append(int(temp_curve_list[idx * 2]) / 3600)
        temp_list.append(int(temp_curve_list[idx * 2 + 1]))

    sol_name = os.path.basename(solution_path)
    sol_time = int(dict_sol_time[sol_name][0])

    # search range
    sol_time = sol_time / 3600

    def find_nearest(array, value):
        """Find the nearest value from the set of time lists"""
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    nearest_id, nearest_value = find_nearest(t_list, sol_time)

    # determine left / right / ==
    if nearest_value == sol_time:
        return temp_list[nearest_id]
    elif nearest_value < sol_time:
        start_temp, end_temp = temp_list[nearest_id], temp_list[nearest_id + 1]
        start_time, end_time = nearest_value, t_list[nearest_id + 1]
    else:
        start_temp, end_temp = temp_list[nearest_id - 1], temp_list[nearest_id]
        start_time, end_time = t_list[nearest_id - 1], nearest_value

    temp = ((end_temp - start_temp) / (end_time - start_time)) * (
        sol_time - start_time
    ) + start_temp

    logging.info(f"solname {sol_name} | soltime {sol_time} | temp: {temp}")
    return temp


def get_solution_temperature(solution_path, dict_sol_time, temp_curve_list):
    """
    This temperature point compute only works for 1st-version temp curve
    :param solution_path:
    :param dict_sol_time:
    :param temp_curve_list:
    :return:
    """
    sol_name = os.path.basename(solution_path)
    sol_time = int(dict_sol_time[sol_name][0])

    start_time, start_temp = int(temp_curve_list[0]), int(temp_curve_list[1])
    equil_time, equil_temp = int(temp_curve_list[2]), int(temp_curve_list[3])

    if sol_time >= equil_time:
        return equil_temp
    else:
        temp = (
            (equil_temp - start_temp) / (equil_time - start_time)
        ) * sol_time + start_temp
    return temp


def get_solution_time(solution_path, dict_sol_time):
    """
    Retrieves the time associated with a solution from a dictionary using the
    solution's file name.
    """
    sol_name = os.path.basename(solution_path)
    sol_time = int(dict_sol_time[sol_name][0])
    return sol_time


def _compute_connectivity(positions, radius):
    """Get the indices of connected edges with radius connectivity.

    Args:
      positions: Positions of nodes in the graph. Shape:
        [num_nodes_in_graph, num_dims]. i.e. (10000, 3)
      radius: Radius of connectivity. i.e. 1.2

    Returns:
      senders indices [num_edges_in_graph]
      receiver indices [num_edges_in_graph]

    """
    # Construct tree from the points' positions
    tree = neighbors.KDTree(positions)

    # For each point, get the list of nodes' indices within r
    # return -> list[array(all connecting node indices)]
    # i.e. receivers_list:  [array([ 300,    2,    0,    1,    4, 1371,  310])
    #  array([  3, 301,   0,   1,   5, 311,   8])
    #  array([1372,    6,    2,   24,  358,    3,    0]) ...
    receivers_list = tree.query_radius(positions, r=radius)

    # For each node with sorted index, repeat its len(receiver-nodes) times, to form the matching sender indices array
    senders = np.repeat(range(len(positions)), [len(a) for a in receivers_list])
    receivers = np.concatenate(receivers_list, axis=0)

    try:
        senders.shape == receivers.shape
    except:
        logging.error("Sender, receiver indices not match")
        raise

    return senders, receivers


def get_anchor_point(ds, index_list):
    """
    pvtu_file_name:
    with the given build files, return the index of point, that is the anchor point of the build
    Query criteria: point xyz-displacement == 0
    Ideally, should only exist 1 anchor point
    Args:
        data: data from pv.read
            format: UnstructuredGrid i.e. (0x7f1636fe5520)
                  N Cells:      21970
                  N Points:     175760
                  X Bounds:     -4.600e+01, 3.000e+00
                  Y Bounds:     -4.500e+00, 4.500e+00
                  Z Bounds:     0.000e+00, 1.300e+01
                  N Arrays:     11
        index_list: the non-duplicated point index

    Returns:
        anchor point index: int
    """

    # Read the Digital sintering software raw data, field version name: "u_v_w"
    ds1 = ds["u__v__w"]

    ## return index of the anchor point
    ds1_ax = np.where(ds1[:, 0] == 0)[0]
    ds1_ay = np.where(ds1[:, 1] == 0)[0]
    ds1_az = np.where(ds1[:, 2] == 0)[0]

    # Intersect 3 array to get the point with xyz-displacement == 0
    listx_as_set = set(ds1_ax)
    intersection = listx_as_set.intersection(ds1_ay)
    anchor_pset = intersection.intersection(ds1_az)

    # Intersect with the non-duplicated point index
    anchor_point = anchor_pset.intersection(index_list)

    try:
        len(anchor_point) == 1
    except:
        logging.error(
            f"Find non-unique anchor points {len(anchor_point)}, id list: {anchor_point}!"
        )
        raise

    # find the point id in the non-duplicated point list
    p_idx = list(anchor_point)[0]
    index_list_str = list(map(str, index_list))
    p_i = index_list_str.index(str(p_idx))

    return p_i


def get_anchor_zplane(ds, index_list):
    """
    with the given build files, return the index of point that are on the anchor z-plane of the build
    Query criteria: point z-position == 0, z-displacement == 0
    Args:
        data: data from pv.read
            format: UnstructuredGrid i.e. (0x7f1636fe5520)
                  N Cells:      21970
                  N Points:     175760
                  X Bounds:     -4.600e+01, 3.000e+00
                  Y Bounds:     -4.500e+00, 4.500e+00
                  Z Bounds:     0.000e+00, 1.300e+01
                  N Arrays:     11
        index_list: the non-duplicated point index

    Returns:
        anchor plane points index: List[int]
    """
    # each points' coordinate, shape [# point, dim], i.e. (175760, 3)
    n_points, _ = ds.points.shape
    uvw_values = ds["u__v__w"]

    z_plane = []
    for i, p_idx in enumerate(index_list):
        # for ip in range(n_points):
        point = ds.GetPoint(p_idx)

        # Filter the points on z==0 z-plane
        if point[2] == 0:
            # for all the points fall on the z-plane, collected the point id in the non-duplicated point list
            z_plane.append(i)
            uvw_ = uvw_values[p_idx]

            # set non-0 threshold for corner-cases
            threshold_z = 0.0002
            try:
                uvw_values[p_idx][2] < threshold_z
            except:
                logging.info(f"wrong anchor_zplane id: {p_idx} - z-displacement {uvw_}")
                raise

    return set(z_plane)


def read_solutions_data(raw_data_path=None, init_idx=0, metadata=None):
    """From the raw simulation files, read the deformation value at every time-step for pre-processing"""
    build_path = os.path.join(raw_data_path, "out")
    solution_list = glob.glob(build_path + "/volume-deformation-*.pvtu")
    # solution_list = sorted(solution_list, key=get_solution_id)
    solution_list = natsorted(solution_list)

    try:
        pv.read(solution_list[0])
    except:
        logging.error(f"solution_list not found from: {build_path}")
        raise

    # For each build, read the displacement.pvtu.series file
    # series_file = os.path.join(raw_data_path, 'out', "solution.pvtu.series")
    series_file = os.path.join(raw_data_path, "out", "volume-deformation.pvtu.series")
    try:
        dict_sol_time = read_sol_time(series_file)
    except:
        logging.error(f"solution.pvtu.series not exists!, {series_file}")
        raise

    # For each build, read the temperature profile at every time step
    temp_profile_path = os.path.join(raw_data_path, "params.prm")
    try:
        temp_curve_list = read_temperature(temp_profile_path)
    except:
        logging.error("Temperature profile file params.prm not exists!")
        raise
    logging.info(f"check temp_curve_list: {temp_curve_list}")

    # Record stage ids
    # For example, for an entire sintering duration of 3393 simulation steps, can choose the data process window
    # can either be the entire sintering duration, for process window for detailed accuracy
    # for the default sintering profile, there are 3393 simulation steps, with
    #   stage 1: temperature increase window, start_index, end_index =[0, 596],
    #   stage 2: temperature stable window, start_index, end_index = [596,1321]
    #   stage 3: T increase: [> 1325]
    #   if to consider the transition stage separately: [1200, 1700]

    particles_list = []
    temp_list = []

    # index for the step 100 model, 2 stages
    step = 5

    # solution_data_end = pv.read(solution_list[-1])
    solution_data_t0 = pv.read(solution_list[0])
    radius_begin = utils.get_radius(solution_data_t0)
    logging.info(f"get simulation radius: {radius_begin}")

    # Get the non-duplicated points' start position, and the matching point index
    pos_array_begin, index_list = get_data_position(solution_data_t0)
    # Get the connecting points' matching sending-receiving indices
    # return -> np.array of shape (#edges, )
    senders_graph_t0, receivers_graph_t0 = _compute_connectivity(
        pos_array_begin, radius_begin
    )
    logging.info(f"Computed the connected edges: {senders_graph_t0.shape}")

    # Proces each solution.pvtu file, with step / gap to skip
    solution_step_list = solution_list[init_idx::step]
    logging.info(f"process with start solution file idx: {init_idx}")
    logging.info(f"solution list len: {len(solution_step_list)}")

    for solution_path in solution_step_list:
        # For each displacement-*.pvtu file, get the displacement-id, read data points
        # filter out the repeated data
        time_ = int(dict_sol_time[os.path.basename(solution_path)][0])

        # if start_index <= solution_id <= end_index:
        solution_temp = get_solution_temperature_customer(
            solution_path, dict_sol_time, temp_curve_list
        )
        logging.info(f"process {solution_path}, time: {time_}, temp: {solution_temp}")

        solution_data_ = pv.read(solution_path)
        pos_array_, index_list_ = get_data_position(solution_data_)
        # todo: assert index_list_ matches with index_list

        particles_list.append(pos_array_)
        temp_list.append(solution_temp)

    # ensure the processed sequence window have same length, to confrom with other train data
    # todo: move this outside
    if init_idx != 0 and len(particles_list) != metadata["sequence_length"]:
        skip = True
    else:
        skip = False

    return (
        particles_list,
        temp_list,
        senders_graph_t0,
        receivers_graph_t0,
        radius_begin,
        skip,
    )


def compute_metadata_stats(
    metadata,
    particles_list_builds,
    velocity_list_builds,
    acceleration_list_builds,
    radius_list,
    temp_list_builds,
):
    """Compute stats of the train dataset, to update metadata for normalization in data processing"""
    # Compute position mean, std
    # todo: check why use different norm dimension
    # todo: change the pos stats to 3d as well
    position_stats_array = np.concatenate(particles_list_builds)
    position_std = position_stats_array.std()
    metadata["pos_mean"] = position_stats_array.mean()
    metadata["pos_std"] = position_stats_array.std()

    # Compute velocity mean, std
    velocity_stats_array = np.concatenate(velocity_list_builds)
    velocity_stats_array = velocity_stats_array / position_std
    metadata["vel_mean"] = [i for i in velocity_stats_array.mean(axis=0).tolist()]
    metadata["vel_std"] = [i for i in velocity_stats_array.std(axis=0).tolist()]

    # Compute acceleration mean, std
    acceleration_stats_array = np.concatenate(acceleration_list_builds)
    acceleration_stats_array = acceleration_stats_array / position_std
    metadata["acc_mean"] = [i for i in acceleration_stats_array.mean(axis=0).tolist()]
    metadata["acc_std"] = [i for i in acceleration_stats_array.std(axis=0).tolist()]

    # Compute radius mean, std
    radius_array = np.array(radius_list) / position_std
    metadata["default_connectivity_radius"] = radius_array.mean()

    # Compute temperature mean, std
    metadata["context_mean"] = [np.array(temp_list_builds).mean()]
    metadata["context_std"] = [np.array(temp_list_builds).std()]
    if np.array(temp_list_builds).ndim > 1:
        metadata["context_feat_len"] = np.array(temp_list_builds).shape[1]
    else:
        metadata["context_feat_len"] = 1

    return metadata


def write_tfrecord_entry(writer, features, particles_array, times_array):
    """
    Write data into entry
    Args:
        writer:
        features: contains context_features = {
                    'particle_type': _bytes_feature(particles_type), particles_type dim=[#nodes, ], i.e. (26487,)
                    'key': create_int_feature(key_i),
                    'senders': _bytes_feature(senders_graph_i.tobytes()),
                    'receivers': _bytes_feature(receivers_graph_i.tobytes()),
                }
        particles_array:
        times_array: <class 'numpy.ndarray'> of float64, dim=[sim_steps,] i.e. (56,)

    Returns:

    """
    tf_sequence_example = tf.train.SequenceExample(context=features)
    position_list = tf_sequence_example.feature_lists.feature_list["position"]
    timestep_list = tf_sequence_example.feature_lists.feature_list["step_context"]

    for i in range(len(particles_array)):
        position_list.feature.add().bytes_list.value.append(
            particles_array[i].tobytes()
        )
        timestep_list.feature.add().bytes_list.value.append(times_array[i].tobytes())

    writer.write(tf_sequence_example.SerializeToString())


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Args:
        raw_data_dir:  raw data directory
        metadata_json_path: path of metadata.json
        mode: there are three mode [train, test, stats]
        i.e. data path on server to test 69 parts generalization:
    """
    mode = cfg.data_options.mode
    raw_data_dir = cfg.data_options.raw_data_dir
    metadata_json_path = cfg.data_options.metadata_json_path

    with open(os.path.join(metadata_json_path, "metadata.json"), "r") as f:
        metadata = json.load(f)
        logging.info(f"meta: {metadata}")
    if mode != "stats":
        writer = tf.io.TFRecordWriter(
            os.path.join(metadata_json_path, mode + ".tfrecord")
        )

    # State the build names
    try:
        mode in ["train", "stats", "test"]
    except:
        logging.error("Mode not implemented, insert from [train|test|stats]")
        raise

    if mode in ["train", "stats"]:
        # for expanded version data
        build_list = cfg.data_options.builds_train
    elif mode == "test":
        build_list = cfg.data_options.builds_test

    key_i = 0
    n_steps = 0
    temp_list_builds = []
    particles_list_builds = []
    velocity_list_builds = []
    acceleration_list_builds = []
    radius_list = []
    # Read and process each build data set
    for build_name in build_list:
        logging.info(f"\n\nProcess build: {build_name}")
        # Get information for each build
        # todo: move the compute anchor information outside
        build_path = os.path.join(os.path.join(raw_data_dir, build_name), "out")
        solution_list = glob.glob(build_path + "/solution-*.pvtu")
        # solution_list = sorted(solution_list, key=get_solution_id)
        solution_list = natsorted(solution_list)
        logging.info(f"computing points from : {solution_list[-1]}")
        solution_data_end = pv.read(solution_list[-1])
        _, index_list = get_data_position(solution_data_end)

        if cfg.data_options.add_anchor:
            # Compute the anchor points from the sinter-end data
            zplane_anchors = get_anchor_zplane(solution_data_end, index_list)
            logging.info(f"Found points on the z-plane, cnt= {len(zplane_anchors)}")
            zplane_anchors = list(zplane_anchors)

            anchor_point = get_anchor_point(solution_data_end, index_list)
            logging.info(
                f"Found anchor point with 0-displacement, p_id: {anchor_point}"
            )
        else:
            anchor_point = None

        for init_idx in range(
            0, 4, 1
        ):  # for testing purpose, need to cover start point (92, 93)
            logging.info(f"\n\n process sequence with init_idx: {init_idx}")
            (
                particles_list,
                temp_list,
                senders_graph_i,
                receivers_graph_i,
                radius,
                skip,
            ) = read_solutions_data(
                os.path.join(raw_data_dir, build_name),
                init_idx=init_idx,
                metadata=metadata,
            )

            if skip:
                print("skip length different sequence ")
                continue

            # Gather information across builds, prep for builds stats calculation
            particles_list_builds += particles_list

            velocity_array = time_diff(np.array(particles_list))
            velocity_list = [velocity_array[i] for i in range(velocity_array.shape[0])]
            velocity_list_builds += velocity_list

            acceleration_array = time_diff(velocity_array)
            acceleration_list = [
                acceleration_array[i] for i in range(acceleration_array.shape[0])
            ]
            acceleration_list_builds += acceleration_list

            radius_list.append(radius)

            temp_list_builds += temp_list

            # particles_array.shape(num_time_steps, nodes_per_build, xyz-dim) i.e. (12, 1107, 3)
            particles_array = np.array(particles_list)
            n_steps, n_particles, _ = particles_array.shape
            logging.info(f"particles_array shape: {particles_array.shape}")
            if init_idx == 0:
                metadata["sequence_length"] = n_steps

            key_i += 1

            ##### Write to TFRECORD #####
            if mode != "stats":
                # TODO: reshape reason
                particles_array = particles_array.reshape((n_steps, -1)).astype(
                    np.float64
                )

                # # normalize data
                particles_mean, particles_std = (
                    metadata["pos_mean"],
                    metadata["pos_std"],
                )
                particles_array = (particles_array - particles_mean) / particles_std

                # TODO: Compute and add the boundary condition here
                # for normal particles, assign value = 2
                particles_type = np.repeat(2, n_particles)  # [5 5 5 ... 5 5 5] (1107,)
                if cfg.data_options.add_anchor:
                    particles_type[zplane_anchors] = 1
                    particles_type[anchor_point] = 0
                particles_type = particles_type.tobytes()

                # Add global features
                context_features = {
                    "particle_type": utils._bytes_feature(particles_type),
                    "key": utils.create_int_feature(key_i),
                    "senders": utils._bytes_feature(senders_graph_i.tobytes()),
                    "receivers": utils._bytes_feature(receivers_graph_i.tobytes()),
                }

                features = tf.train.Features(feature=context_features)

                # Write to tfrecord
                start_idx, end_idx = 0, particles_array.shape[0]
                logging.info(f"write range: {start_idx}-{end_idx}")
                write_tfrecord_entry(
                    writer,
                    features,
                    particles_array[start_idx:end_idx],
                    np.array(temp_list)[start_idx:end_idx],
                )
                logging.info(
                    f"Finished writing to tfrecord, finale feature shape: {particles_array.shape}"
                )

    # Write metadata file
    if mode == "stats":
        metadata = compute_metadata_stats(
            metadata,
            particles_list_builds,
            velocity_list_builds,
            acceleration_list_builds,
            radius_list,
            temp_list_builds,
        )

        metadata["sequence_length"] = n_steps - 1
        metadata["dim"] = 3

        with open(os.path.join(metadata_json_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    elif mode == "test" or mode == "generalization":
        logging.info(f"Finale feature shape: {particles_array.shape}")
        metadata["sequence_length"] = particles_array.shape[0] - 1
        with open(os.path.join(metadata_json_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)


"""
    Perform data processing over all builds defined in the raw_dir_path.

    Arguments:
        cfg: Dictionary of parameters.

    """
if __name__ == "__main__":
    argv = sys.argv[1:]
    main()
