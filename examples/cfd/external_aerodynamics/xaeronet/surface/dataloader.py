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
import torch_geometric as pyg
from torch_geometric.loader import ClusterData
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils import find_bin_files


class PartitionedGraph:
    """
    A class for partitioning a graph into multiple parts with halo regions.

    Parameters:
    ----------
        graph (pyg.data.Data): The graph data.
        num_parts (int): The number of partitions.
        halo_size (int): The size of the halo region.
    """

    def __init__(self, graph: pyg.data.Data, num_parts: int, halo_size: int):
        self.num_nodes = graph.num_nodes
        self.num_parts = num_parts
        self.halo_size = halo_size

        # Partition the graph using PyG METIS.
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.ClusterData
        cluster_data = pyg.loader.ClusterData(graph, num_parts=self.num_parts)
        part_meta = cluster_data.partition

        # Create partitions with halo regions using PyG `k_hop_subgraph`.
        self.partitions = []
        for i in range(self.num_parts):
            # Get inner nodes of the partition.
            part_inner_node = part_meta.node_perm[
                part_meta.partptr[i] : part_meta.partptr[i + 1]
            ]
            # Partition the graph with halo regions.
            # https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html?#torch_geometric.utils.k_hop_subgraph
            part_node, part_edge_index, inner_node_mapping, edge_mask = (
                pyg.utils.k_hop_subgraph(
                    part_inner_node,
                    num_hops=self.halo_size,
                    edge_index=graph.edge_index,
                    num_nodes=self.num_nodes,
                    relabel_nodes=True,
                )
            )

            partition = pyg.data.Data(
                edge_index=part_edge_index,
                edge_attr=graph.edge_attr[edge_mask],
                num_nodes=part_node.size(0),
                part_node=part_node,
                inner_node=inner_node_mapping,
            )
            # Set partition node attributes.
            for k, v in graph.items():
                if graph.is_node_attr(k):
                    setattr(partition, k, v[part_node])

            self.partitions.append(partition)

    def __len__(self):
        return len(self.partitions)

    def __getitem__(self, idx: int) -> pyg.data.Data:
        return self.partitions[idx]


class GraphDataset(Dataset):
    """
    Custom dataset class for loading partitioned graphs from .bin files.

    Parameters:
    ----------
        file_list (list of str): List of paths to .bin files containing partitions.
        mean (np.ndarray): Global mean for normalization.
        std (np.ndarray): Global standard deviation for normalization.
    """

    NODE_ATTRS: list[str] = [
        "coordinates",
        "normals",
        "area",
        "pressure",
        "shear_stress",
    ]

    def __init__(self, file_list, mean, std):
        self.file_list = file_list
        self.mean = mean
        self.std = std

        # Store normalization stats as tensors
        # Set normalization statistics as attributes using a loop.
        for key in self.NODE_ATTRS:
            setattr(self, f"{key}_mean", torch.tensor(mean[key]))
            setattr(self, f"{key}_std", torch.tensor(std[key]))
        self.edge_x_mean = torch.tensor(mean["x"])
        self.edge_x_std = torch.tensor(std["x"])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple[PartitionedGraph, str]:
        file_path = self.file_list[idx]

        # Extract the ID from the file name
        file_name = os.path.basename(file_path)
        # Assuming file format is "graph_partitions_<run_id>.bin"
        run_id = file_name.split("_")[-1].split(".")[0]  # Extract the run ID

        # Load the partitioned graphs from the .bin file
        graphs = torch.load(file_path, weights_only=False)

        # Normalize node and edge data
        for part in graphs:
            for key in self.NODE_ATTRS:
                part[key] = (part[key] - getattr(self, f"{key}_mean")) / getattr(
                    self, f"{key}_std"
                )

            part.edge_attr = (part.edge_attr - self.edge_x_mean) / self.edge_x_std

        return graphs, run_id

    @staticmethod
    def collate_fn(
        batch: list[tuple[ClusterData, str]],
    ) -> tuple[list[ClusterData], list[str]]:
        graphs, run_ids = zip(*batch)
        return list(graphs), list(run_ids)


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

    Parameters:
    ----------
        file_list (list of str): List of paths to .bin files.
        mean (np.ndarray): Global mean for normalization.
        std (np.ndarray): Global standard deviation for normalization.
        batch_size (int): Number of samples per batch.
        shuffle (bool): If True, the data will be reshuffled at every epoch.
        use_ddp (bool): If True, the data loader will use distributed sampling.
        drop_last (bool): If True, the last batch will be dropped if it is not complete.
        num_workers (int): Number of worker processes for data loading.
        pin_memory (bool): If True, the data loader will copy tensors into CUDA pinned memory.
        prefetch_factor (int): Number of batches loaded in advance by each worker.

    Returns
    -------
        DataLoader: Configured DataLoader for the dataset.
    """
    if batch_size != 1:
        raise ValueError(f"Batch size must be 1 for now, but got {batch_size}")

    dataset = GraphDataset(file_list, mean, std)
    if use_ddp:
        from physicsnemo.distributed import DistributedManager

        dist = DistributedManager()
        world_size = dist.world_size
        rank = dist.rank
    else:
        world_size = 1
        rank = 0

    sampler = DistributedSampler(
        dataset,
        shuffle=shuffle,
        drop_last=drop_last,
        num_replicas=world_size,
        rank=rank,
    )

    # instantiate dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=GraphDataset.collate_fn,
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
