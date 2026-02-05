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
File to visualize and check the acceleration profile
"""

import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from omegaconf import DictConfig
from utils import get_data_position, read_raw_folder, time_diff

logging.basicConfig(filename="data_analysis.log", level=logging.DEBUG)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """The function to visualize and check the acceleration profile, builds to be analyzed are read from config"""
    raw_data_dir = cfg.data_options.raw_data_dir
    # build_name = "2024-02-27-Overhangs_t3_theta0_V3-deformation"

    for build_name in cfg.data_options.builds_train:
        solution_list = read_raw_folder(os.path.join(raw_data_dir, build_name))
        logging.info(
            f"\n\nRead solution files from {build_name}, cnt= {len(solution_list)}"
        )

        pos_list, pos_max_list, pos_mean_list = [], [], []
        step = cfg.data_options.step_size
        logging.info(f"Process for every {step} files ...... ")
        for i in range(0, len(solution_list), step):
            # Read the raw solution file with step size
            solution_data = pv.read(solution_list[i])
            # pos_array: np array stores displacement, dim: (num_nodes, 3)
            pos_array, _ = get_data_position(solution_data)
            pos_list.append(pos_array)

            pos_mean_list.append(np.mean(pos_array, axis=0))
            pos_max_list.append(np.max(pos_array))

        logging.info(f"Computed pos_list, shape {np.asarray(pos_list).shape}")

        # Compute the velovity, acceleration for each node
        velocity_array = time_diff(np.array(pos_list))
        acceleration_array = time_diff(velocity_array)
        logging.info(f"Computed velocity_array, shape {velocity_array.shape}")
        logging.info(f"Computed acceleration_array, shape {acceleration_array.shape}")

        acc_3d_mean = np.mean(acceleration_array, axis=1)
        logging.info(f"Computed Mean(acce), shape {acc_3d_mean.shape}")

        # Visualize
        fig, ax = plt.subplots()
        logging.info("Plot Mean(acce) ......... ")
        sol_index = [i for i in range(acc_3d_mean.shape[0])]
        ax.plot(sol_index, acc_3d_mean[:, 0], "b-", linewidth=1, label="x-dim velocity")
        ax.plot(sol_index, acc_3d_mean[:, 1], "y-", linewidth=1, label="y-dim velocity")
        ax.plot(sol_index, acc_3d_mean[:, 2], "g-", linewidth=1, label="z-dim velocity")
        ax.set_xlabel("time steps", color="blue", fontsize=14)
        ax.set_ylabel("acce", color="blue", fontsize=14)
        # ax.set_ylim(0, 3e-6)
        ax.set_title(build_name)

        ax.legend(loc="lower right")
        fig_name = "acc_3d_" + build_name + "_step" + str(step)
        fig.savefig(fig_name + ".jpg", format="png", dpi=100, bbox_inches="tight")
        logging.info(f"Saved figure at {fig_name}. ")

        logging.info(f"Mean(acce) {np.mean(np.abs(acc_3d_mean), axis=0)}")


"""
Perform data analyis on voxel moving speed, acceleration
"""
if __name__ == "__main__":
    main()
