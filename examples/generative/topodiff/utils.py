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

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os


class DiffusionDataset_topodiff(Dataset):
    def __init__(self, topologies, vfs_stress_strain, load_im):

        image_size = topologies.shape[1]

        self.topologies = topologies
        self.vfs_stress_strain = vfs_stress_strain
        self.image_size = image_size
        self.load_im = load_im

    def __len__(self):
        return self.topologies.shape[0]

    def __getitem__(self, idx):

        cons = np.zeros((5, self.image_size, self.image_size))

        cons[0] = self.vfs_stress_strain[idx][:, :, 0]
        cons[1] = self.vfs_stress_strain[idx][:, :, 1]
        cons[2] = self.vfs_stress_strain[idx][:, :, 2]
        cons[3] = self.load_im[idx][:, :, 0]
        cons[4] = self.load_im[idx][:, :, 1]

        return np.expand_dims(self.topologies[idx], 0) * 2 - 1, cons


def load_data_topodiff(
    topologies, vfs_stress_strain, load_im, batch_size, deterministic=False
):

    dataset = DiffusionDataset_topodiff(topologies, vfs_stress_strain, load_im)

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def load_data(root, prefix, file_format, num_file_start=0, num_file_end=30000):
    """
    root: path to the folder of training data
    prefix: file prefix to the ground truth topology, boundary condition and stress/strain
    file_format: .npy for the conditions; .png for the ground truth topologies
    """
    data_array = []

    for i in range(num_file_start, num_file_end):
        file = f"{root}{prefix}{i}{file_format}"
        if file_format == ".npy":
            data_array.append(np.load(file))
        elif file_format == ".png":
            data_array.append(np.array(Image.open(file)) / 255)
        else:
            raise NotImplementedError

    return np.array(data_array).astype(np.float64)


def load_data_regressor(root):

    file_list = os.listdir(root)
    idx_list = []
    for file in file_list:
        if file.startswith("gt_topo_"):
            idx = int(file.split(".")[0][8:])
            idx_list.append(idx)
    idx_list.sort()

    topology_array, load_array, pf_array = [], [], []
    for i in idx_list:

        topology_array.append(
            np.array(Image.open(root + "gt_topo_" + str(i) + ".png")) / 255
        )
        load_array.append(np.load(root + "cons_load_array_" + str(i) + ".npy"))
        pf_array.append(np.load(root + "cons_pf_array_" + str(i) + ".npy"))

    labels = np.load(root + "deflections_scaled_diff.npy")
    return (
        np.array(topology_array).astype(np.float64),
        np.array(load_array).astype(np.float64),
        np.array(pf_array).astype(np.float64),
        labels[idx_list],
    )


def load_data_classifier(root):
    """
    root: path to the folder of training data
    prefix: file prefix to the ground truth topology, boundary condition and stress/strain
    file_format: .npy for the conditions; .png for the ground truth topologies
    """
    file_list = os.listdir(root)
    labels = np.load(root + "labels.npy")
    image_list = []
    label_list = []
    for file in file_list:
        if file.startswith("img_"):
            idx = int(file.split(".")[0][4:])
            image = Image.open(root + file)
            image_list.append(np.array(image) / 255)
            label_list.append(labels[idx])

    return np.array(image_list).astype(np.float64), np.array(label_list).astype(
        np.float64
    )
