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
Custom dataset and dataloader utilities for reservoir simulation graph data.
Provides GraphDataset for loading and normalizing partitioned graphs from .pt files,
along with utilities for computing global statistics and creating efficient dataloaders.
"""

import os
import sys
import json
import logging

import torch
import torch_geometric as pyg
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Module-level logger
logger = logging.getLogger(__name__)


def find_pt_files(directory):
    """
    Find all .pt files in a directory.

    Parameters
    ----------
    directory : str
        Directory to search for .pt files

    Returns
    -------
    file_paths : list
        List of file paths to .pt files
    """
    import glob

    if not os.path.exists(directory):
        return []

    pattern = os.path.join(directory, "**", "*.pt")
    file_paths = glob.glob(pattern, recursive=True)
    return sorted(file_paths)


def save_stats(stats, output_file):
    """
    Save statistics to a JSON file.

    Parameters
    ----------
    stats : dict
        Statistics dictionary
    output_file : str
        Output file path
    """
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Statistics saved to {output_file}")


def load_stats(stats_file):
    """
    Load statistics from a JSON file.

    Parameters
    ----------
    stats_file : str
        Path to the statistics file

    Returns
    -------
    stats : dict
        Statistics dictionary
    """
    with open(stats_file, "r") as f:
        stats = json.load(f)

    return stats


def compute_global_statistics(graph_files, stats_file=None):
    """
    Compute global statistics (mean and std) across all graphs for normalization.

    Parameters
    ----------
    graph_files : list
        List of paths to graph files (.pt files)
    stats_file : str, optional
        Path to save statistics JSON file. If None, statistics are not saved.

    Returns
    -------
    dict : Dictionary containing node, edge, and target statistics
    """
    logger.info(f"Computing global statistics across {len(graph_files)} graphs...")

    # Collect all node, edge, and target features
    all_node_features = []
    all_edge_features = []
    all_target_features = []

    # Process all graphs to compute statistics
    logger.info(f"Computing statistics from {len(graph_files)} graphs...")
    for i, file_path in enumerate(graph_files, 1):
        try:
            graph = torch.load(file_path, weights_only=False)

            # Collect node features
            if hasattr(graph, "x") and graph.x is not None:
                all_node_features.append(graph.x)

            # Collect edge features
            if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
                all_edge_features.append(graph.edge_attr)

            # Collect target features
            if hasattr(graph, "y") and graph.y is not None:
                all_target_features.append(graph.y)

            if i % 100 == 0:
                logger.info(f"  Processed {i}/{len(graph_files)} graphs...")

        except Exception as e:
            logger.warning(f"Failed to load graph {file_path}: {e}")
            continue

    # Compute statistics for node features
    if all_node_features:
        # Filter out graphs with inconsistent feature dimensions
        if len(all_node_features) > 1:
            # Find the most common feature dimension
            feature_dims = [
                feat.shape[1] for feat in all_node_features if feat.numel() > 0
            ]
            if feature_dims:
                from collections import Counter

                most_common_dim = Counter(feature_dims).most_common(1)[0][0]
                # Keep only graphs with the most common feature dimension
                all_node_features = [
                    feat
                    for feat in all_node_features
                    if feat.shape[1] == most_common_dim
                ]
                logger.info(
                    f"  Filtered to {len(all_node_features)} graphs with consistent {most_common_dim} node features"
                )

        if all_node_features:
            # Concatenate all node features: [total_nodes, num_node_features]
            all_nodes = torch.cat(all_node_features, dim=0)
            node_mean = torch.mean(all_nodes, dim=0)  # [num_node_features]
            node_std = torch.std(all_nodes, dim=0)  # [num_node_features]
            logger.info(
                f"  Node features: {all_nodes.shape[1]} features, {all_nodes.shape[0]} total nodes"
            )
        else:
            node_mean = torch.tensor([])
            node_std = torch.tensor([])
            logger.warning("  No consistent node features found")
    else:
        node_mean = torch.tensor([])
        node_std = torch.tensor([])
        logger.warning("  No node features found")

    # Compute statistics for edge features
    if all_edge_features:
        # Filter out graphs with inconsistent feature dimensions
        if len(all_edge_features) > 1:
            # Find the most common feature dimension
            feature_dims = [
                feat.shape[1] for feat in all_edge_features if feat.numel() > 0
            ]
            if feature_dims:
                from collections import Counter

                most_common_dim = Counter(feature_dims).most_common(1)[0][0]
                # Keep only graphs with the most common feature dimension
                all_edge_features = [
                    feat
                    for feat in all_edge_features
                    if feat.shape[1] == most_common_dim
                ]
                logger.info(
                    f"  Filtered to {len(all_edge_features)} graphs with consistent {most_common_dim} edge features"
                )

        if all_edge_features:
            # Concatenate all edge features: [total_edges, num_edge_features]
            all_edges = torch.cat(all_edge_features, dim=0)
            edge_mean = torch.mean(all_edges, dim=0)  # [num_edge_features]
            edge_std = torch.std(all_edges, dim=0)  # [num_edge_features]
            logger.info(
                f"  Edge features: {all_edges.shape[1]} features, {all_edges.shape[0]} total edges"
            )
        else:
            edge_mean = torch.tensor([])
            edge_std = torch.tensor([])
            logger.warning("  No consistent edge features found")
    else:
        edge_mean = torch.tensor([])
        edge_std = torch.tensor([])
        logger.warning("  No edge features found")

    # Compute statistics for target features
    if all_target_features:
        # Filter out graphs with inconsistent feature dimensions
        if len(all_target_features) > 1:
            # Find the most common feature dimension
            feature_dims = [
                feat.shape[1] for feat in all_target_features if feat.numel() > 0
            ]
            if feature_dims:
                from collections import Counter

                most_common_dim = Counter(feature_dims).most_common(1)[0][0]
                # Keep only graphs with the most common feature dimension
                all_target_features = [
                    feat
                    for feat in all_target_features
                    if feat.shape[1] == most_common_dim
                ]
                logger.info(
                    f"  Filtered to {len(all_target_features)} graphs with consistent {most_common_dim} target features"
                )

        if all_target_features:
            # Concatenate all target features: [total_nodes, num_target_features]
            all_targets = torch.cat(all_target_features, dim=0)
            target_mean = torch.mean(all_targets, dim=0)  # [num_target_features]
            target_std = torch.std(all_targets, dim=0)  # [num_target_features]
            logger.info(
                f"  Target features: {all_targets.shape[1]} features, {all_targets.shape[0]} total nodes"
            )
        else:
            target_mean = torch.tensor([])
            target_std = torch.tensor([])
            logger.warning("  No consistent target features found")
    else:
        target_mean = torch.tensor([])
        target_std = torch.tensor([])
        logger.warning("  No target features found")

    # Create statistics dictionary
    stats = {
        "node_features": {"mean": node_mean.tolist(), "std": node_std.tolist()},
        "edge_features": {"mean": edge_mean.tolist(), "std": edge_std.tolist()},
        "target_features": {"mean": target_mean.tolist(), "std": target_std.tolist()},
    }

    # Save statistics if requested
    if stats_file:
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"  Statistics saved to {stats_file}")

    logger.info(f"  Node features - Mean: {node_mean.tolist()}")
    logger.info(f"  Node features - Std:  {node_std.tolist()}")
    logger.info(f"  Edge features - Mean: {edge_mean.tolist()}")
    logger.info(f"  Edge features - Std:  {edge_std.tolist()}")
    logger.info(f"  Target features - Mean: {target_mean.tolist()}")
    logger.info(f"  Target features - Std:  {target_std.tolist()}")

    return stats


class PartitionedGraph:
    """
    A class for partitioning a graph into multiple parts with halo regions.

    Parameters
    ----------
    graph : pyg.data.Data
        The graph data.
    num_parts : int
        The number of partitions.
    halo_size : int
        The size of the halo region.
    """

    def __init__(self, graph: pyg.data.Data, num_parts: int, halo_size: int):
        self.num_nodes = graph.num_nodes
        self.num_parts = num_parts
        self.halo_size = halo_size

        # Try to partition the graph using PyG METIS, with fallback to simple partitioning
        try:
            # Partition the graph using PyG METIS.
            # https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.ClusterData
            cluster_data = pyg.loader.ClusterData(graph, num_parts=self.num_parts)
            part_meta = cluster_data.partition
        except Exception as e:
            logger.warning(
                f"     METIS partitioning failed ({e}), using simple partitioning..."
            )
            # Fallback: simple sequential partitioning
            part_meta = self._create_simple_partition(graph.num_nodes, num_parts)

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
        return self.num_parts

    def __getitem__(self, idx):
        return self.partitions[idx]

    def _create_simple_partition(self, num_nodes, num_parts):
        """Create a simple sequential partition as fallback when METIS is not available."""
        import torch

        # Create a simple partition object that mimics the METIS partition structure
        class SimplePartition:
            def __init__(self, num_nodes, num_parts):
                self.node_perm = torch.arange(num_nodes)

                # Calculate partition boundaries
                part_size = num_nodes // num_parts
                remainder = num_nodes % num_parts

                self.partptr = [0]
                for i in range(num_parts):
                    current_size = part_size + (1 if i < remainder else 0)
                    self.partptr.append(self.partptr[-1] + current_size)

        return SimplePartition(num_nodes, num_parts)


class GraphDataset(Dataset):
    """
    A custom dataset class for loading and normalizing graph partition data.

    Parameters
    ----------
    file_paths : list
        List of file paths to the graph partition files.
    node_mean : torch.Tensor
        Global mean for node attributes (shape: [num_node_features]).
    node_std : torch.Tensor
        Global standard deviation for node attributes (shape: [num_node_features]).
    edge_mean : torch.Tensor
        Global mean for edge attributes (shape: [num_edge_features]).
    edge_std : torch.Tensor
        Global standard deviation for edge attributes (shape: [num_edge_features]).
    target_mean : torch.Tensor
        Global mean for target attributes (shape: [num_target_features]).
    target_std : torch.Tensor
        Global standard deviation for target attributes (shape: [num_target_features]).
    """

    def __init__(
        self,
        file_paths,
        node_mean,
        node_std,
        edge_mean,
        edge_std,
        target_mean=None,
        target_std=None,
    ):
        self.file_paths = file_paths
        self.node_mean = node_mean
        self.node_std = node_std
        self.edge_mean = edge_mean
        self.edge_std = edge_std
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the list of graph partitions (following xaeronet pattern)
        partitions = torch.load(self.file_paths[idx], weights_only=False)

        # Extract label from filename (sample index)
        filename = os.path.basename(self.file_paths[idx])
        # Handle different filename formats:
        # - Raw graphs: CASE_2D_1_000.pt or CASE_2D_0001_000.pt -> extract sample index
        # - Partitions: partitions_CASE_2D_630_009.pt or partitions_CASE_2D_0630_009.pt -> extract sample index
        # - Norne format: partitions_NORNE_ATW2013_DOE_0004_002.pt -> extract 0004 (sample index)
        parts = filename.replace(".pt", "").split("_")
        label = 0  # Default label

        # Find all numeric parts in the filename
        numeric_parts = []
        for i, part in enumerate(parts):
            try:
                numeric_value = int(part)
                numeric_parts.append((i, numeric_value))
            except ValueError:
                continue

        # The sample index is typically the second-to-last numeric part
        # (last numeric part is usually the timestep)
        if len(numeric_parts) >= 2:
            # Get the second-to-last numeric part as the sample index
            label = numeric_parts[-2][1]
        elif len(numeric_parts) == 1:
            # If only one numeric part, use it as the label
            label = numeric_parts[0][1]

        # Normalize each partition in the list
        for partition in partitions:
            # Normalize node attributes (per-feature normalization)
            if hasattr(partition, "x") and partition.x is not None:
                # Ensure dimensions match: partition.x shape should be [num_nodes, num_features]
                # node_mean and node_std should be [num_features]
                if partition.x.dim() == 2 and self.node_mean.dim() == 1:
                    # Broadcasting: [num_nodes, num_features] - [num_features] -> [num_nodes, num_features]
                    partition.x = (partition.x - self.node_mean) / (
                        self.node_std + 1e-8
                    )
                else:
                    # Fallback for mismatched dimensions
                    logger.warning(
                        f"Dimension mismatch in node features. Partition shape: {partition.x.shape}, Stats shape: {self.node_mean.shape}"
                    )
                    partition.x = (partition.x - self.node_mean.unsqueeze(0)) / (
                        self.node_std.unsqueeze(0) + 1e-8
                    )

            # Normalize edge attributes (per-feature normalization)
            if hasattr(partition, "edge_attr") and partition.edge_attr is not None:
                # Ensure dimensions match: partition.edge_attr shape should be [num_edges, num_edge_features]
                # edge_mean and edge_std should be [num_edge_features]
                if partition.edge_attr.dim() == 2 and self.edge_mean.dim() == 1:
                    # Broadcasting: [num_edges, num_edge_features] - [num_edge_features] -> [num_edges, num_edge_features]
                    partition.edge_attr = (partition.edge_attr - self.edge_mean) / (
                        self.edge_std + 1e-8
                    )
                else:
                    # Fallback for mismatched dimensions
                    logger.warning(
                        f"Dimension mismatch in edge features. Partition shape: {partition.edge_attr.shape}, Stats shape: {self.edge_mean.shape}"
                    )
                    partition.edge_attr = (
                        partition.edge_attr - self.edge_mean.unsqueeze(0)
                    ) / (self.edge_std.unsqueeze(0) + 1e-8)

            # Normalize target attributes (per-feature normalization)
            if (
                hasattr(partition, "y")
                and partition.y is not None
                and self.target_mean is not None
                and self.target_std is not None
            ):
                # Ensure dimensions match: partition.y shape should be [num_nodes, num_target_features]
                # target_mean and target_std should be [num_target_features]
                if partition.y.dim() == 2 and self.target_mean.dim() == 1:
                    # Broadcasting: [num_nodes, num_target_features] - [num_target_features] -> [num_nodes, num_target_features]
                    partition.y = (partition.y - self.target_mean) / (
                        self.target_std + 1e-8
                    )
                else:
                    # Fallback for mismatched dimensions
                    logger.warning(
                        f"Dimension mismatch in target features. Partition shape: {partition.y.shape}, Stats shape: {self.target_mean.shape}"
                    )
                    partition.y = (partition.y - self.target_mean.unsqueeze(0)) / (
                        self.target_std.unsqueeze(0) + 1e-8
                    )

        return partitions, label


def custom_collate_fn(batch):
    """
    Custom collate function for lists of PartitionedGraph objects (following xaeronet pattern).

    Parameters
    ----------
    batch : list
        List of (partitions, label) tuples from the dataset
        where partitions is a list of PartitionedGraph objects

    Returns
    -------
    tuple
        (partitions_list, labels) where partitions_list is a list of lists of PartitionedGraph objects
        and labels is a tensor of labels
    """
    partitions_list, labels = zip(*batch)
    return list(partitions_list), torch.tensor(labels, dtype=torch.long)


def create_dataloader(
    partitions_path,
    validation_partitions_path,
    stats_file,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    prefetch_factor=2,
    pin_memory=True,
    is_validation=False,
):
    """
    Create a data loader for graph partition data.

    Parameters
    ----------
    partitions_path : str
        Path to the partitions directory.
    validation_partitions_path : str
        Path to the validation partitions directory.
    stats_file : str
        Path to the global statistics file.
    batch_size : int
        Batch size for the data loader.
    shuffle : bool
        Whether to shuffle the data.
    num_workers : int
        Number of worker processes for data loading.
    prefetch_factor : int
        Number of batches to prefetch.
    pin_memory : bool
        Whether to pin memory for faster GPU transfer.
    is_validation : bool
        Whether this is for validation data.

    Returns
    -------
    DataLoader
        The data loader.
    """
    # Load global statistics
    with open(stats_file, "r") as f:
        stats = json.load(f)

    # Load per-feature statistics
    # node_features should be a list of means/stds for each feature
    node_mean = torch.tensor(
        stats["node_features"]["mean"]
    )  # Shape: [num_node_features]
    node_std = torch.tensor(stats["node_features"]["std"])  # Shape: [num_node_features]
    edge_mean = torch.tensor(
        stats["edge_features"]["mean"]
    )  # Shape: [num_edge_features]
    edge_std = torch.tensor(stats["edge_features"]["std"])  # Shape: [num_edge_features]

    # Load target feature statistics (if available)
    target_mean = None
    target_std = None
    if "target_features" in stats:
        target_mean = torch.tensor(
            stats["target_features"]["mean"]
        )  # Shape: [num_target_features]
        target_std = torch.tensor(
            stats["target_features"]["std"]
        )  # Shape: [num_target_features]

    # Find partition files
    if is_validation:
        file_paths = find_pt_files(validation_partitions_path)
    else:
        file_paths = find_pt_files(partitions_path)

    # Create dataset
    dataset = GraphDataset(
        file_paths, node_mean, node_std, edge_mean, edge_std, target_mean, target_std
    )

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,  # Use custom collate function for lists of PartitionedGraph objects
    )

    return dataloader
