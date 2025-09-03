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

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree


def compute_mean_var(dir_path):
    """
    Compute mean and variance for forces.
    """
    all_forces = []

    # Process each file in the directory
    for filename in os.listdir(dir_path):
        if filename.endswith(".npz"):  # Check that we're only opening .npz files
            filepath = os.path.join(dir_path, filename)

            # Load the .npz file
            with np.load(filepath, "rb") as data:
                forces = data["forces"].astype(np.float32)
                all_forces.extend(forces.reshape(1, -1))

    force_mean = np.mean(np.array(all_forces))
    force_sd = np.std(np.array(all_forces))

    return force_mean, force_sd


class LJData(Dataset):
    """
    Dataset to load the Lennard Jones data.

    Reference: https://github.com/BaratiLab/GAMD
    """

    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        pos = data["pos"].astype(np.float32)
        forces = data["forces"].astype(np.float32)
        return pos, forces


def train_test_split(file_paths, test_size=0.2):
    """
    Split data into training and test data
    """
    total_size = len(file_paths)
    test_size = int(total_size * test_size)

    test_files = file_paths[:test_size]
    train_files = file_paths[test_size:]
    return train_files, test_files


def create_datasets(directory, test_size=0.2):
    """
    Create datasets given the path for data files
    """
    file_paths = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".npz")
    ]
    train_files, test_files = train_test_split(file_paths, test_size)

    train_dataset = LJData(train_files)
    test_dataset = LJData(test_files)

    return train_dataset, test_dataset


def _custom_collate(batch):
    collated_batch = [list(field) for field in zip(*batch)]
    return collated_batch


def get_rotation_matrix():
    """
    Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction

    Reference: https://github.com/BaratiLab/GAMD/blob/main/code/LJ/train_network_lj.py#L38
    """
    if np.random.uniform() < 0.3:
        angles = np.random.randint(-2, 2, size=(3,)) * np.pi
    else:
        angles = [0.0, 0.0, 0.0]
    Rx = np.array(
        [
            [1.0, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])],
        ],
        dtype=np.float32,
    )
    Ry = np.array(
        [
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])],
        ],
        dtype=np.float32,
    )
    Rz = np.array(
        [
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    rotation_matrix = np.matmul(Rz, np.matmul(Ry, Rx))

    return rotation_matrix


def create_edges(node_positions, threshold, box_size):
    """
    Create edges between nodes based on a distance threshold.
    """

    tree = cKDTree(
        node_positions,
        boxsize=np.ptp(node_positions, axis=0) + np.array([0.001, 0.001, 0.001]),
    )

    edges = []
    edge_features = []

    for idx, results in enumerate(tree.query_ball_point(node_positions, threshold)):
        nearby_points = node_positions[results]
        relative_pos = nearby_points - node_positions[idx]

        # handle periodicity
        relative_pos_periodic = (
            np.mod(relative_pos + 0.5 * box_size, box_size) - 0.5 * box_size
        )
        relative_pos_norm = np.linalg.norm(relative_pos_periodic, axis=1).reshape(-1, 1)
        relative_pos_periodic = relative_pos_periodic / relative_pos_norm
        relative_pos_periodic = np.nan_to_num(
            relative_pos_periodic, nan=0.0, posinf=0.0, neginf=0.0
        )

        for i in range(len(nearby_points)):
            edges.append((idx, results[i]))
            edge_features.append(
                np.append(relative_pos_periodic[i], relative_pos_norm[i] / threshold)
            )

    # Convert the edges to a format that DGL can use
    src, dst = tuple(zip(*edges))

    return src, dst, edge_features
