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
import numpy as np
import dgl
from tqdm import tqdm
import json
import shutil
import copy
import vtk_tools as vtkt
import graph_tools as grpt
import scipy
import torch as th


def add_field(graph, field, field_name, offset=0, pad=10):
    """
    Add time-dependent fields to a DGL graph.

    Add time-dependent scalar fields as graph node features. The time-dependent
    fields are stored as n x 1 x m Pytorch tensors, where n is the number of
    graph nodes and m the number of timesteps.

    Arguments:
        graph: DGL graph
        field: dictionary with (key: timestep, value: field value)
        field_name (string): name of the field
        offset (int): number of timesteps to skip.
                      Default: 0 -> keep all timesteps
        pad (int): number of timesteps to add for interpolation from zero
                   zero initial conditions. Default: 0 -> start from actual
                   initial condition
    """
    timesteps = [float(t) for t in field]
    timesteps.sort()
    dt = timesteps[1] - timesteps[0]
    T = timesteps[-1]
    # we use the third dimension for time
    field_t = th.zeros(
        (list(field.values())[0].shape[0], 1, len(timesteps) - offset + pad)
    )

    times = [t for t in field]
    times.sort()
    times = times[offset:]

    loading_t = th.zeros(
        (list(field.values())[0].shape[0], 1, len(timesteps) - offset + pad),
        dtype=th.bool,
    )

    if pad > 0:
        inc = th.tensor(field[times[0]], dtype=th.float32)
        deft = inc * 0
        if field_name == "pressure":
            minp = np.infty
            for t in field:
                minp = np.min((minp, np.min(field[t])))
            deft = deft + minp
        for i in range(pad):
            field_t[:, 0, i] = deft * (pad - i) / pad + inc * (i / pad)
            loading_t[:, 0, i] = True

    for i, t in enumerate(times):
        f = th.tensor(field[t], dtype=th.float32)
        field_t[:, 0, i + pad] = f
        loading_t[:, 0, i + pad] = False

    graph.ndata[field_name] = field_t
    graph.ndata["loading"] = loading_t
    graph.ndata["dt"] = th.reshape(
        th.ones(graph.num_nodes(), dtype=th.float32) * dt, (-1, 1, 1)
    )
    graph.ndata["T"] = th.reshape(
        th.ones(graph.num_nodes(), dtype=th.float32) * T, (-1, 1, 1)
    )


def load_vtp(file, input_dir):
    """
    Load vtp file.

    Arguments:
        file (string): file name
        input_dir (string): path to input_dir

    Returns:
        dictionary containing point data (key: name, value: data)
        n x 3 numpy array of point coordinates
        numpy array containing indices of source nodes for every edge
        numpy array containing indices of dest nodes for every edge

    """
    soln = vtkt.read_geo(input_dir + "/" + file)
    point_data, _, points = vtkt.get_all_arrays(soln.GetOutput())
    edges1, edges2 = vtkt.get_edges(soln.GetOutput())

    # lets check for nans and delete points if they appear
    ni = np.argwhere(np.isnan(point_data["area"]))
    if ni.size > 0:
        for i in ni[0]:
            indices = np.where(edges1 >= i)[0]
            edges1[indices] = edges1[indices] - 1

            indices = np.where(edges2 >= i)[0]
            edges2[indices] = edges2[indices] - 1

            indices = np.where(edges1 == edges2)[0]
            edges1 = np.delete(edges1, indices)
            edges2 = np.delete(edges2, indices)

            points = np.delete(points, i, axis=0)
            for ndata in point_data:
                point_data[ndata] = np.delete(point_data[ndata], i)

    return point_data, points, edges1, edges2


def resample_time(field, timestep, period, shift=0):
    """
    Resample timesteps.

    Given a time-dependent field distributed over graph nodes, this function
    resamples the field in time using B-spline interpolation at every node.

    Arguments:
        field: dictionary containing the field for all timesteps
               (key: timestep, value: n-dimensional numpy array)
        timestep (float): the new timestep
        period (float): period of the simulation. We restrict to one cardiac
                        cycle

        shift (float): apply shift (s) to start at the beginning of the systole.
                       Default value -> 0

    Returns:
        dictionary containing the field for all resampled timesteps
            (key: timestep, value: n-dimensional numpy array)
    """
    original_timesteps = [t for t in field]
    original_timesteps.sort()

    t0 = original_timesteps[0]
    T = original_timesteps[-1]
    t = [t0 + shift]
    nnodes = field[t0].size
    resampled_field = {t0 + shift: np.zeros(nnodes)}
    while t[-1] < T and t[-1] <= t[0] + period:
        t.append(t[-1] + timestep)
        resampled_field[t[-1]] = np.zeros(nnodes)

    for inode in range(nnodes):
        values = []
        for time in original_timesteps:
            values.append(field[time][inode])

        tck, _ = scipy.interpolate.splprep([values], u=original_timesteps, s=0)
        values_interpolated = scipy.interpolate.splev(t, tck)[0]

        for i, time in enumerate(t):
            resampled_field[time][inode] = values_interpolated[i]

    return resampled_field


def generate_datastructures(vtp_data, resample_perc):
    """
    Generate data structures for graph generation from vtp data.

    Arguments:
        vtp_data: tuple containing data extracted from the vtp using load_vtp
        resample_perc: percentage of points in the original vtp file we keep
                       (between 0 and 1)
    Returns:
        dictionary containing graph data (key: field name, value: data)
    """
    point_data, points, edges1, edges2 = vtp_data
    point_data["tangent"] = grpt.generate_tangents(points, point_data["BranchIdTmp"])
    # first node is the inlet by convention
    inlet = [0]
    outlets = grpt.find_outlets(edges1, edges2)

    indices = {"inlet": inlet, "outlets": outlets}

    success = False

    while not success:
        try:
            sampled_indices, points, edges1, edges2, _ = grpt.resample_points(
                points.copy(),
                edges1.copy(),
                edges2.copy(),
                indices,
                resample_perc,
                remove_caps=3,
            )
            success = True
        except Exception as e:
            print(e)
            resample_perc = np.min([resample_perc * 2, 1])

    for ndata in point_data:
        point_data[ndata] = point_data[ndata][sampled_indices]

    inlet = [0]
    outlets = grpt.find_outlets(edges1, edges2)

    indices = {"inlet": inlet, "outlets": outlets}

    pressure = vtkt.gather_array(point_data, "pressure")
    flowrate = vtkt.gather_array(point_data, "flow")
    if len(flowrate) == 0:
        flowrate = vtkt.gather_array(point_data, "velocity")

    times = [t for t in pressure]
    timestep = float(dataset_info[file.replace(".vtp", "")]["dt"])
    for t in times:
        pressure[t * timestep] = pressure[t]
        flowrate[t * timestep] = flowrate[t]
        del pressure[t]
        del flowrate[t]

    # scale pressure to be mmHg
    for t in pressure:
        pressure[t] = pressure[t] / 1333.2

    times = [t for t in pressure]

    sampling_indices = np.arange(points.shape[0])
    graph_data = {
        "point_data": point_data,
        "points": points,
        "edges1": edges1,
        "edges2": edges2,
        "sampling_indices": sampling_indices,
        "pressure": pressure,
        "flowrate": flowrate,
        "timestep": timestep,
        "times": times,
    }

    return graph_data


def add_time_dependent_fields(
    graph, graph_data, do_resample_time=False, dt=0.01, copies=1
):
    """
    Add time-dependent data to a graph containing static data. This function
    can be used to create multiple graphs from a single trajectory by
    specifying do_resample_time and providing a number of copies > 1. In this
    case, every graph trajectories starts at a different offset from the
    starting time.

    Arguments:
        graph: a DGL graph.
        graph_data: dictionary containing graph_data (created using
                    generate_datastructures)
        do_resample_time (bool): specify whether we should resample the
                                 the timesteps. Default -> False
        dt (double): timestep size used for resampling. Default -> 0.01
        copies: number of copies to generate from a single trajectory (for
                data augmentation). Default -> 1

    Returns:
        list of 'copies' graphs.
    """

    ncopies = 1
    if do_resample_time:
        ncopies = copies
        dt = 0.01
        offset = int(np.floor((dt / graph_data["timestep"]) / ncopies))

    graphs = []
    intime = 0
    for icopy in range(ncopies):
        c_pressure = {}
        c_flowrate = {}

        si = graph_data["sampling_indices"]
        for t in graph_data["times"][intime:]:
            c_pressure[t] = graph_data["pressure"][t][si]
            c_flowrate[t] = graph_data["flowrate"][t][si]

        if do_resample_time:
            period = dataset_info[fname]["T"]
            shift = dataset_info[fname]["time_shift"]
            c_pressure = resample_time(
                c_pressure, timestep=dt, period=period, shift=shift
            )
            c_flowrate = resample_time(
                c_flowrate, timestep=dt, period=period, shift=shift
            )
            intime = intime + offset

        padt = 0.1
        new_graph = copy.deepcopy(graph)
        add_field(new_graph, c_pressure, "pressure", pad=int(padt / dt))
        add_field(new_graph, c_flowrate, "flowrate", pad=int(padt / dt))
        graphs.append(new_graph)

    return graphs


"""
The main function reads all vtps files from the folder specified in input_dir
and generates DGL graphs. The graphs are saved in output_dir.
"""
if __name__ == "__main__":
    input_dir = "raw_dataset/vtps"
    output_dir = "raw_dataset/graphs/"

    dataset_info = json.load(open(input_dir + "/dataset_info.json"))

    files = os.listdir(input_dir)

    print("Processing all files in {}".format(input_dir))
    print("File list:")
    print(files)
    for file in tqdm(files, desc="Generating graphs", colour="green"):
        if ".vtp" in file and "s" in file:
            vtp_data = load_vtp(file, input_dir)
            graph_data = generate_datastructures(vtp_data, resample_perc=0.06)

            fname = file.replace(".vtp", "")
            static_graph = grpt.generate_graph(
                graph_data["point_data"],
                graph_data["points"],
                graph_data["edges1"],
                graph_data["edges2"],
                add_boundary_edges=True,
                rcr_values=dataset_info[fname],
            )

            graphs = add_time_dependent_fields(
                static_graph, graph_data, do_resample_time=True, dt=0.1, copies=4
            )

            for i, graph in enumerate(graphs):
                filename = file.replace(".vtp", "." + str(i) + ".grph")
                dgl.save_graphs(output_dir + filename, graph)

    shutil.copy(input_dir + "/dataset_info.json", output_dir + "/dataset_info.json")
