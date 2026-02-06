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


import numpy as np

cross_plot_map = {
    0: (1, 0),
    1: (1, 1),
    2: (1, 2),
    3: (1, 3),
    4: (0, 0),
    5: (2, 0),
}

rotations = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
}


def rearrange_to_cross(data, cross_plot_map=cross_plot_map, rotations=rotations):
    cross_data = {}
    data_min, data_max = np.min(data), np.max(data)
    for tile in range(6):
        row, col = cross_plot_map[tile]
        rotated_data = np.rot90(data[tile], k=rotations[tile])
        cross_data[(row, col)] = rotated_data
    return cross_data, data_min, data_max


def plot_cross_subplot(data, data_min, data_max):
    data_total = np.empty((64 * 3, 64 * 4))
    data_total[:] = np.nan

    for (row, col), face_data in data.items():
        data_total[row * 64 : (row + 1) * 64, col * 64 : (col + 1) * 64] = face_data

    return data_total


def cube_sphere_plotter(data):
    cross_data, data_min, data_max = rearrange_to_cross(data)
    return plot_cross_subplot(cross_data, data_min, data_max)
