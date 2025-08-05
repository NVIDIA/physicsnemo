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


"""
This code defines a custom dataset class GraphDataset for loading and normalizing
graph partition data stored in .bin files. The dataset is initialized with a list
of file paths and global mean and standard deviation for node and edge attributes.
It normalizes node data (like coordinates, normals, pressure) and edge data based
on these statistics before returning the processed graph partitions and a corresponding
label (extracted from the file name). The code also provides a function create_dataloader
to create a data loader for efficient batch loading with configurable parameters such as
batch size, shuffle, and prefetching options. 
"""

import json
import torch
from torch.utils.data import Dataset
import os
import sys
import dgl
from dgl.dataloading import GraphDataLoader

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils import find_bin_files


class GraphDataset(Dataset):
    """
    Custom dataset class for loading

    Parameters:
    ----------
        file_list (list of str): List of paths to .bin files containing partitions.
        mean (np.ndarray): Global mean for normalization.
        std (np.ndarray): Global standard deviation for normalization.
    """

    def __init__(self, file_list, mean, std):
        self.file_list = file_list
        self.mean = mean
        self.std = std

        # Store normalization stats as tensors
        self.coordinates_mean = torch.tensor(mean["coordinates"])
        self.coordinates_std = torch.tensor(std["coordinates"])
        self.normals_mean = torch.tensor(mean["normals"])
        self.normals_std = torch.tensor(std["normals"])
        self.area_mean = torch.tensor(mean["area"])
        self.area_std = torch.tensor(std["area"])
        self.pressure_mean = torch.tensor(mean["pressure"])
        self.pressure_std = torch.tensor(std["pressure"])
        self.shear_stress_mean = torch.tensor(mean["shear_stress"])
        self.shear_stress_std = torch.tensor(std["shear_stress"])
        self.edge_x_mean = torch.tensor(mean["x"])
        self.edge_x_std = torch.tensor(std["x"])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # Extract the ID from the file name
        file_name = os.path.basename(file_path)
        # Assuming file format is "graph_partitions_<run_id>.bin"
        run_id = file_name.split("_")[-1].split(".")[0]  # Extract the run ID

        # Load the partitioned graphs from the .bin file
        graphs, _ = dgl.load_graphs(file_path)

        # Process each partition (graph)
        normalized_partitions = []
        for graph in graphs:
            # Normalize node data
            graph.ndata["coordinates"] = (
                graph.ndata["coordinates"] - self.coordinates_mean
            ) / self.coordinates_std
            graph.ndata["normals"] = (
                graph.ndata["normals"] - self.normals_mean
            ) / self.normals_std
            graph.ndata["area"] = (graph.ndata["area"] - self.area_mean) / self.area_std
            graph.ndata["pressure"] = (
                graph.ndata["pressure"] - self.pressure_mean
            ) / self.pressure_std
            graph.ndata["shear_stress"] = (
                graph.ndata["shear_stress"] - self.shear_stress_mean
            ) / self.shear_stress_std

            # Normalize edge data
            if "x" in graph.edata:
                graph.edata["x"] = (
                    graph.edata["x"] - self.edge_x_mean
                ) / self.edge_x_std

            normalized_partitions.append(graph)

        return normalized_partitions, run_id


def create_dataloader(
    file_list,
    mean,
    std,
    batch_size=1,
    shuffle=False,
    use_ddp=True,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
):
    """
    Creates a DataLoader for the GraphDataset with prefetching.

    Args:
        file_list (list of str): List of paths to .bin files.
        mean (np.ndarray): Global mean for normalization.
        std (np.ndarray): Global standard deviation for normalization.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker processes for data loading.
        pin_memory (bool): If True, the data loader will copy tensors into CUDA pinned memory.

    Returns:
        DataLoader: Configured DataLoader for the dataset.
    """
    dataset = GraphDataset(file_list, mean, std)
    dataloader = GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        use_ddp=use_ddp,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    return dataloader


if __name__ == "__main__":
    data_path = "partitions"
    stats_file = "global_stats.json"

    # Load global statistics
    with open(stats_file, "r") as f:
        stats = json.load(f)
    mean = stats["mean"]
    std = stats["std_dev"]

    # Find all .bin files in the directory
    file_list = find_bin_files(data_path)

    # Create DataLoader
    dataloader = create_dataloader(
        file_list,
        mean,
        std,
        batch_size=1,
        prefetch_factor=None,
        use_ddp=False,
        num_workers=1,
    )

    # Example usage
    for batch_partitions, label in dataloader:
        for graph in batch_partitions:
            print(graph)
        print(label)
