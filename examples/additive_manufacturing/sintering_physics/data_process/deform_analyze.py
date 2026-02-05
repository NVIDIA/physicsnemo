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


import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import rawdata2tfrecord_large_ts as rawdata2tfrecord
from utils import get_solution_id, read_configs


def plot_temperature_curve(temp_curve_list):
    """Read from the list of sintering time-temp, visualize the sintering profile"""
    t_list = []
    temp_list = []
    for idx in range(len(temp_curve_list) // 2):
        t_list.append(int(temp_curve_list[idx * 2]))
        temp_list.append(int(temp_curve_list[idx * 2 + 1]))

    print("time list: ", t_list)
    print("temp_list : ", temp_list)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(t_list, temp_list, color="blue", marker=".")
    fig.savefig("temperature_profile.png", format="png", dpi=100, bbox_inches="tight")

    return t_list, temp_list


def plot_temperature_curve2(key_list, temp_list):
    """Read from the list of sintering step_idx-temp, visualize the sintering profile"""
    fig = plt.figure(figsize=(10, 6))
    plt.plot(key_list, temp_list, color="blue", marker=".")
    fig.savefig(
        "temperature_profile_complete.png", format="png", dpi=100, bbox_inches="tight"
    )


def read_sol_time(series_file):
    """
    Read the solution.pvtu.series file
    Returns:

    """
    dict_sol_time = {}
    time_list = []
    with open(series_file, "r") as fobj:
        data = json.load(fobj)

    for idx, item in enumerate(data["files"]):
        time_list.append(item["time"])
        dict_sol_time[item["name"]] = [item["time"]]

    return dict_sol_time


# plot
def plot_p_deform(
    temp_list, key_list, stage_keys, del_u, del_v, del_w, pid=0, split_stages=False
):
    """Read from selected point-id deformation, visualize this point's deformation over the sintering profile"""
    # sol_list = [i for i in range(len(temp_list))]
    sol_list = key_list

    fig, ax = plt.subplots()
    # ax.plot(sol_list, del_u, color="blue", marker=".")
    ax.plot(sol_list, del_u, "b-", linewidth=2)
    # ax.plot(sol_list, del_v, color="g", marker=".")
    ax.plot(sol_list, del_v, "g-", linewidth=2)
    ax.plot(sol_list, del_w, "y-", linewidth=2)
    ax.set_xlabel("Solution index ", fontsize=14)
    ax.set_ylabel("Sample point deformation (mm)", color="blue", fontsize=14)

    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(sol_list, temp_list, "r-", linewidth=2)
    ax2.set_ylabel("Temperature", color="red", fontsize=14)

    # plot cut-off regions
    x_lb = min(np.asarray(del_u + del_v + del_w))
    x_ub = max(np.asarray(del_u + del_v + del_w))
    x_lim = np.arange(x_lb, x_ub + 0.0005, 0.0005).tolist()

    if split_stages:
        ax.fill_betweenx(
            x_lim,
            x1=stage_keys[5],
            x2=stage_keys[6],
            where=None,
            step=None,
            color="gainsboro",
            interpolate=True,
            label="stage-separation-1",
        )
        ax.fill_betweenx(
            x_lim,
            x1=stage_keys[7],
            x2=stage_keys[8],
            where=None,
            step=None,
            color="gainsboro",
            interpolate=True,
            label="stage-separation-2",
        )
        ax.fill_betweenx(
            x_lim,
            x1=stage_keys[9],
            x2=max(sol_list),
            where=None,
            step=None,
            color="gainsboro",
            interpolate=True,
            label="stage-separation-3",
        )

    fig_name = "point_deform_curve_p" + str(pid)
    ax.set_title("p" + str(pid), fontsize=14)
    ax.legend(loc="lower right")
    fig.savefig(fig_name + ".jpg", format="png", dpi=100, bbox_inches="tight")


# Read a point id-x, from each file in solution list, plot its uvw
def read_point(file_name, p_id):
    """Read the sintering point id value, with pv library, from path: file_name"""
    data = pv.read(file_name)
    uvw_values = data["displacement_U"]

    p_xyz = data.GetPoint(p_id)
    p_uvw = uvw_values[p_id, ...]

    return p_xyz, p_uvw


def read_solutions_data_temp_anchor(
    raw_data_path, build_name, start_temp=500, end_temp=2000
):
    """1st version"""
    build_path = os.path.join(raw_data_path, "out")
    solution_list = glob.glob(build_path + "/displacement-*.pvtu")
    solution_list = sorted(solution_list, key=get_solution_id)

    # read configs from params.prm only, bypass the solution.pvtu.series file
    time_params, temp_curve_list = read_configs(raw_data_path)

    # For each build, read the displacement.pvtu.series file
    series_file = os.path.join(raw_data_path, "out", "displacement.pvtu.series")
    print("Find and read series file: ", series_file)
    assert os.path.exists(series_file), "displacement.pvtu.series not exists!"
    dict_sol_time = read_sol_time(series_file)
    print("dict_sol_time: ", dict_sol_time)

    key_list = []
    temp_list = []

    del_u, del_v, del_w = [], [], []

    # todo: sample point id, move to input variable
    read_point_id = 8000
    for solution_idx, solution_path in enumerate(solution_list):
        # For each solution-*.pvtu file, get the solution-id, read data points
        # filter out the repeated data
        solution_id = get_solution_id(solution_path)
        solution_temp = rawdata2tfrecord.get_solution_temperature_customer(
            solution_path, dict_sol_time, temp_curve_list
        )

        if solution_temp >= start_temp and solution_temp < end_temp:
            key_list.append(solution_id)
            temp_list.append(solution_temp)

            p_xyz, p_uvw = read_point(file_name=solution_path, p_id=read_point_id)
            del_u.append(p_uvw[0])
            del_v.append(p_uvw[1])
            del_w.append(p_uvw[2])

    plot_p_deform(
        build_name,
        temp_list,
        key_list=key_list,
        del_u=del_u,
        del_v=del_v,
        del_w=del_w,
        pid=read_point_id,
    )

    return key_list, temp_list


def main(argv):
    """
    Args:
        raw_data_dir:  raw data directory
        metadata_json_path: path of metadata.json
        mode: there are three mode [train, test, stats]
        i.e. data path on server to test 69 parts generalization:
    """
    raw_data_dir = argv[0]

    # sample builds to perform analysis, include the build names
    # i.e. build_list = ['10007564-slide-53710', 'Bearing_Support-slide-53710', 'GRF_SH20-slide-53710', '2289x1D_N_acc-slide-53710', 'S8001-slide-53710']
    build_list = ["MK-M-1045-rA-slide-53710"]

    # input the start and end point for ploting
    start_temp, end_temp = 600, 1310

    n_steps = 0
    for build_name in build_list:
        key_list, temp_list = read_solutions_data_temp_anchor(
            os.path.join(raw_data_dir, build_name),
            build_name,
            start_temp=start_temp,
            end_temp=end_temp,
        )

        if n_steps != 0 and len(key_list) != n_steps:
            print(build_name, " read failed")
            continue


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
