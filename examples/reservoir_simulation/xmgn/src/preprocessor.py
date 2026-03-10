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
Preprocessing pipeline for reservoir simulation data.
Converts simulation output to partitioned graphs for XMeshGraphNet training.
Extracts grid properties, connections, and well data, computes global statistics,
and partitions graphs for efficient distributed training.
"""

import os
import sys
import json
import random
import re
import shutil
import contextlib
import io
import warnings
import logging

# Add src directory to Python path for flexible imports
current_dir = os.path.dirname(os.path.abspath(__file__))  # This is src/
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add repository root to Python path for sim_utils import
repo_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels from src/
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch_geometric as pyg
from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data.graph_builder import ReservoirGraphBuilder
from data.dataloader import PartitionedGraph, compute_global_statistics
from utils import get_dataset_dir

logger = logging.getLogger(__name__)


class SimplePartition:
    """Simple sequential partition as fallback when METIS is not available.

    Mimics the METIS partition structure for compatibility.
    """

    def __init__(self, num_nodes, num_parts):
        """
        Initialize a simple sequential partition.

        Parameters
        -----------
        num_nodes : int
            Total number of nodes to partition
        num_parts : int
            Number of partitions to create
        """
        self.node_perm = torch.arange(num_nodes)

        # Calculate partition boundaries
        part_size = num_nodes // num_parts
        remainder = num_nodes % num_parts

        # Create partition pointers
        self.partptr = [0]
        for i in range(num_parts):
            # Add extra node to first 'remainder' partitions
            current_size = part_size + (1 if i < remainder else 0)
            self.partptr.append(self.partptr[-1] + current_size)


class ReservoirPreprocessor:
    """
    A class to handle the complete preprocessing pipeline for reservoir simulation data.

    This class manages the creation of raw graphs from simulation data, partitioning them
    for efficient training, computing global statistics, and organizing data splits.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the ReservoirPreprocessor with configuration.

        Parameters
        -----------
        cfg : DictConfig
            Hydra configuration object containing all preprocessing parameters
        """
        self.cfg = cfg

        # Get dataset directory using path_utils utility for consistent job name handling
        self.dataset_dir = get_dataset_dir(cfg)

        self.graphs_dir = os.path.join(self.dataset_dir, "graphs")
        self.partitions_dir = os.path.join(self.dataset_dir, "partitions")
        self.stats_file = os.path.join(self.dataset_dir, "global_stats.json")

        # Set default values for preprocessing
        self.cfg.preprocessing.num_preprocess_workers = getattr(
            cfg.preprocessing, "num_preprocess_workers", 4
        )
        self.cfg.preprocessing.num_partitions = getattr(
            cfg.preprocessing, "num_partitions", 3
        )
        self.cfg.preprocessing.halo_size = getattr(cfg.preprocessing, "halo_size", 1)

        self.graph_file_list = None
        self.generated_files = None

        # Extract job name from dataset directory for display
        job_name = os.path.basename(self.dataset_dir)
        logger.info(f"Dataset directory: {self.dataset_dir}")
        logger.info(f"Job name: {job_name}")

    def _extract_case_name_from_filename(self, filename):
        """
        Extract case name from a graph filename by removing the timestep suffix.

        Expected format: {case_name}_{timestep:03d}.pt
        where timestep is typically 3 digits (e.g., 000, 001, 123).

        Examples:
            CASE_2D_1_000.pt -> CASE_2D_1
            NORNE_ATW2013_DOE_0004_002.pt -> NORNE_ATW2013_DOE_0004
            sample_005_123.pt -> sample_005

        Parameters
        -----------
        filename : str
            Graph filename (with or without .pt extension)

        Returns
        --------
        str: Case name without timestep suffix
        """
        # Remove .pt extension if present
        name = filename.replace(".pt", "")

        # Pattern: match case_name followed by underscore and 3-digit timestep at end
        # The timestep is formatted as {timestep_id:03d} in graph_builder.py
        match = re.match(r"^(.+)_(\d{3})$", name)

        if match:
            return match.group(1)  # Return everything before the last _XXX
        else:
            # Fallback: if pattern doesn't match, assume entire name is the case
            # (this handles edge cases or future format changes)
            return name

    def save_graph_file_list(self, graph_files, list_file="generated_graphs.json"):
        """
        Save list of generated graph files for tracking.

        Parameters
        -----------
        graph_files : list
            List of generated graph file paths
        list_file : str
            Path to save graph file list
        """
        # Save in the graphs directory
        list_path = os.path.join(self.graphs_dir, list_file)

        graph_list = {
            "generated_files": [os.path.basename(f) for f in graph_files],
            "graphs_dir": self.graphs_dir,
            "count": len(graph_files),
            "timestamp": torch.tensor(0).item(),  # Simple timestamp placeholder
        }

        with open(list_path, "w") as f:
            json.dump(graph_list, f, indent=2)

        logger.info(f"Saved graph file list to: {list_path}")

    def load_graph_file_list(self, list_file="generated_graphs.json"):
        """
        Load list of generated graph files.

        Parameters
        -----------
        list_file : str
            Path to graph file list

        Returns
        --------
        list or None: List of graph file names, or None if not found
        """
        list_path = os.path.join(self.graphs_dir, list_file)

        if not os.path.exists(list_path):
            return None

        try:
            with open(list_path, "r") as f:
                data = json.load(f)
            return data.get("generated_files", [])
        except (json.JSONDecodeError, KeyError):
            return None

    def save_preprocessing_metadata(self, metadata_file="preprocessing_metadata.json"):
        """
        Save preprocessing paths to a metadata file for later retrieval.

        Parameters
        -----------
        metadata_file : str
            Path to save metadata file
        """
        metadata = {
            "graphs_dir": self.graphs_dir,
            "partitions_dir": self.partitions_dir,
            "preprocessing_completed": True,
            "partition_config": {
                "num_partitions": self.cfg.preprocessing.num_partitions,
                "halo_size": self.cfg.preprocessing.halo_size,
            },
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved preprocessing metadata to: {metadata_file}")

    def save_dataset_metadata(self, metadata_file="dataset_metadata.json"):
        """
        Save dataset metadata for inference use.

        Parameters
        -----------
        metadata_file : str
            Path to save dataset metadata file
        """
        # Get absolute path to sim_dir
        sim_dir_abs = to_absolute_path(self.cfg.dataset.sim_dir)

        metadata = {
            "sim_dir": sim_dir_abs,  # Absolute path to simulator data directory
            "dataset_dir": self.dataset_dir,
            "graphs_dir": self.graphs_dir,
            "partitions_dir": self.partitions_dir,
            "stats_file": self.stats_file,
            "preprocessing_completed": True,
            "job_name": os.path.basename(self.dataset_dir),
            "config": {
                "simulator": self.cfg.dataset.simulator,
                "num_samples": getattr(self.cfg.dataset, "num_samples", None),
            },
            "partition_config": {
                "num_partitions": self.cfg.preprocessing.num_partitions,
                "halo_size": self.cfg.preprocessing.halo_size,
            },
        }

        metadata_path = os.path.join(self.dataset_dir, metadata_file)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved dataset metadata to: {metadata_path}")

    def validate_partition_topology(self):
        """
        Validate that existing partitions match the current configuration.

        Returns
        --------
        bool
            True if partitions are valid and match current config, False otherwise
        """
        # Check if dataset metadata exists
        metadata_path = os.path.join(self.dataset_dir, "dataset_metadata.json")
        if not os.path.exists(metadata_path):
            logger.warning(
                "No dataset metadata found, cannot validate partition topology"
            )
            return False

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Check if partition config exists in metadata
            if "partition_config" not in metadata:
                logger.warning(
                    "Partition configuration not found in metadata, "
                    "partitions may have been created with older version"
                )
                return False

            saved_config = metadata["partition_config"]
            current_num_partitions = self.cfg.preprocessing.num_partitions
            current_halo_size = self.cfg.preprocessing.halo_size

            saved_num_partitions = saved_config.get("num_partitions")
            saved_halo_size = saved_config.get("halo_size")

            # Validate num_partitions
            if saved_num_partitions != current_num_partitions:
                logger.warning(
                    f"Partition topology mismatch: "
                    f"existing partitions have num_partitions={saved_num_partitions}, "
                    f"but current config has num_partitions={current_num_partitions}"
                )
                return False

            # Validate halo_size
            if saved_halo_size != current_halo_size:
                logger.warning(
                    f"Partition topology mismatch: "
                    f"existing partitions have halo_size={saved_halo_size}, "
                    f"but current config has halo_size={current_halo_size}"
                )
                return False

            logger.info(
                f"Partition topology validated: "
                f"num_partitions={current_num_partitions}, halo_size={current_halo_size}"
            )
            return True

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to validate partition topology: {e}")
            return False

    def split_samples_by_case(self, train_ratio, val_ratio, test_ratio, random_seed=42):
        """
        Split graph files by case (sample) to ensure all timesteps of a sample stay together.

        Parameters
        -----------
        train_ratio : float
            Ratio of samples for training
        val_ratio : float
            Ratio of samples for validation
        test_ratio : float
            Ratio of samples for testing
        random_seed : int
            Random seed for reproducible splits

        Returns
        --------
        dict: Dictionary with 'train', 'val', 'test' keys containing lists of file names
        """
        # Extract unique case names from file names
        case_names = set()
        for filename in self.graph_file_list:
            # Extract case name using robust regex-based parsing
            case_name = self._extract_case_name_from_filename(filename)
            case_names.add(case_name)

        case_names = sorted(list(case_names))
        total_cases = len(case_names)

        # Validate that we have enough samples for the split
        min_samples_needed = 3  # Need at least 3 samples for train/val/test split
        if total_cases < min_samples_needed:
            raise ValueError(
                f"Insufficient samples for train/val/test split! "
                f"Found {total_cases} samples, but need at least {min_samples_needed}. "
                f"Please increase num_samples in config or adjust split ratios."
            )

        # Validate split ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, but got {total_ratio}")

        # Set random seed for reproducible splits
        random.seed(random_seed)
        random.shuffle(case_names)

        # Calculate split indices
        train_end = int(total_cases * train_ratio)
        val_end = train_end + int(total_cases * val_ratio)

        train_cases = case_names[:train_end]
        val_cases = case_names[train_end:val_end]
        test_cases = case_names[val_end:]

        # Ensure at least one sample in each split
        if len(train_cases) == 0:
            train_cases = [case_names[0]]
            if len(val_cases) > 0:
                val_cases = val_cases[1:]
            elif len(test_cases) > 0:
                test_cases = test_cases[1:]

        if len(val_cases) == 0 and len(test_cases) > 0:
            val_cases = [test_cases[0]]
            test_cases = test_cases[1:]

        logger.info(f"Sample split:")
        logger.info(
            f"   Training: {len(train_cases)} cases ({len(train_cases) / total_cases * 100:.1f}%)"
        )
        logger.info(
            f"   Validation: {len(val_cases)} cases ({len(val_cases) / total_cases * 100:.1f}%)"
        )
        logger.info(
            f"   Test: {len(test_cases)} cases ({len(test_cases) / total_cases * 100:.1f}%)"
        )

        # Group files by split
        splits = {"train": [], "val": [], "test": []}

        for filename in self.graph_file_list:
            case_name = self._extract_case_name_from_filename(filename)
            if case_name in train_cases:
                splits["train"].append(filename)
            elif case_name in val_cases:
                splits["val"].append(filename)
            elif case_name in test_cases:
                splits["test"].append(filename)

        logger.info(f"File split:")
        logger.info(f"   Training: {len(splits['train'])} files")
        logger.info(f"   Validation: {len(splits['val'])} files")
        logger.info(f"   Test: {len(splits['test'])} files")

        return splits

    def organize_partitions_by_split(self, splits):
        """
        Create partitions and organize them into train/val/test subdirectories.

        Parameters
        -----------
        splits : dict
            Dictionary with 'train', 'val', 'test' keys containing file lists
        """
        logger.info(f"\nOrganizing partitions by split...")

        # Create subdirectories
        train_dir = os.path.join(self.partitions_dir, "train")
        val_dir = os.path.join(self.partitions_dir, "val")
        test_dir = os.path.join(self.partitions_dir, "test")

        for split_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(split_dir, exist_ok=True)

        # Process each split
        total_moved = 0
        for split_name, file_list in splits.items():
            if not file_list:
                logger.info(f"   → {split_name.capitalize()}: No files to process")
                continue

            split_dir = os.path.join(self.partitions_dir, split_name)
            logger.info(f"   → Processing {split_name} split: {len(file_list)} files")

            moved_count = 0
            logger.info(f"Organizing {split_name} split ({len(file_list)} files)...")
            for filename in file_list:
                # Load the graph
                try:
                    graph_path = os.path.join(
                        self.partitions_dir, f"partitions_{filename}"
                    )
                    if not os.path.exists(graph_path):
                        continue

                    # Move the partition file to the appropriate subdirectory
                    dest_path = os.path.join(split_dir, f"partitions_{filename}")
                    shutil.move(graph_path, dest_path)
                    moved_count += 1

                except Exception as e:
                    continue

            logger.info(
                f"     Moved {moved_count}/{len(file_list)} files to {split_name}/"
            )
            total_moved += moved_count

        logger.info(f"Partition organization complete!")

    @contextlib.contextmanager
    def suppress_all_output(self):
        """Context manager to suppress all output including stdout, stderr, warnings, and logging."""
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Temporarily disable logging
                logging.disable(logging.CRITICAL)
                try:
                    yield
                finally:
                    logging.disable(logging.NOTSET)

    def create_simple_partition(self, num_nodes, num_parts):
        """Create a simple sequential partition as fallback when METIS is not available.

        Parameters
        -----------
        num_nodes : int
            Total number of nodes to partition
        num_parts : int
            Number of partitions to create

        Returns
        --------
        SimplePartition
            Partition object with node_perm and partptr attributes
        """
        return SimplePartition(num_nodes, num_parts)

    def create_partitions_from_graphs(self, graph_file_list=None):
        """
        Create partitions from raw graphs for efficient training.

        Parameters
        -----------
        graph_file_list : list or None
            List of specific graph files to process (if None, process all .pt files)
        """
        logger.info(f"\nCreating partitions from graphs...")

        # Create partitions directory
        os.makedirs(self.partitions_dir, exist_ok=True)

        # Determine which graph files to process
        if graph_file_list is not None:
            # Use specific list of files
            graph_files = [
                os.path.join(self.graphs_dir, f)
                for f in graph_file_list
                if f.endswith(".pt")
            ]
        else:
            # Find all graph files
            graph_files = []
            for file in os.listdir(self.graphs_dir):
                if file.endswith(".pt"):
                    graph_files.append(os.path.join(self.graphs_dir, file))

        logger.info(
            f"   → Processing {len(graph_files)} graphs with {self.cfg.preprocessing.num_partitions} partitions each..."
        )

        # Track partition assignments by case (we'll save once per case, not per timestep)
        partition_assignments_by_case = {}

        # Process each graph file
        successful_partitions = 0
        for i, graph_file in tqdm(
            enumerate(graph_files, 1),
            total=len(graph_files),
            desc="Creating partitions",
            unit="graph",
        ):
            # Load the graph
            try:
                graph = torch.load(graph_file, weights_only=False)

                # Create partitions directly without using PartitionedGraph class
                # to avoid module path issues
                # Try to partition the graph using PyG METIS, with fallback to simple partitioning
                try:
                    with self.suppress_all_output():
                        # Partition the graph using PyG METIS
                        cluster_data = pyg.loader.ClusterData(
                            graph, num_parts=self.cfg.preprocessing.num_partitions
                        )
                        part_meta = cluster_data.partition
                except Exception as e:
                    logger.warning(
                        f"     WARNING: METIS partitioning failed ({e}), using simple partitioning..."
                    )
                    # Fallback: simple sequential partitioning
                    part_meta = self.create_simple_partition(
                        graph.num_nodes, self.cfg.preprocessing.num_partitions
                    )

                # Extract partition assignments (which node belongs to which partition)
                # Create an array: partition_id[node_idx] = partition_number (1-indexed)
                partition_assignment = torch.zeros(graph.num_nodes, dtype=torch.int32)
                for part_idx in range(self.cfg.preprocessing.num_partitions):
                    # Get inner nodes of this partition
                    part_inner_nodes = part_meta.node_perm[
                        part_meta.partptr[part_idx] : part_meta.partptr[part_idx + 1]
                    ]
                    # Assign partition ID (1-indexed for visualization)
                    partition_assignment[part_inner_nodes] = part_idx + 1

                # Save partition assignments per case (only once per case)
                filename = os.path.basename(graph_file)
                case_name = self._extract_case_name_from_filename(filename)

                # Only save if we haven't saved for this case yet
                if case_name not in partition_assignments_by_case:
                    partition_assignments_by_case[case_name] = (
                        partition_assignment.numpy().tolist()
                    )

                # Create partitions with halo regions using PyG `k_hop_subgraph`
                partitions = []
                for part_idx in range(self.cfg.preprocessing.num_partitions):
                    # Get inner nodes of the partition
                    part_inner_node = part_meta.node_perm[
                        part_meta.partptr[part_idx] : part_meta.partptr[part_idx + 1]
                    ]
                    # Partition the graph with halo regions
                    part_node, part_edge_index, inner_node_mapping, edge_mask = (
                        pyg.utils.k_hop_subgraph(
                            part_inner_node,
                            num_hops=self.cfg.preprocessing.halo_size,
                            edge_index=graph.edge_index,
                            num_nodes=graph.num_nodes,
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
                    # Set partition node attributes
                    for k, v in graph.items():
                        if graph.is_node_attr(k):
                            setattr(partition, k, v[part_node])

                    partitions.append(partition)

                # Save partitions as a list (following xaeronet pattern)
                partition_file = os.path.join(
                    self.partitions_dir, f"partitions_{os.path.basename(graph_file)}"
                )
                torch.save(partitions, partition_file)

                successful_partitions += 1

            except Exception as e:
                logger.error(f"ERROR: processing {os.path.basename(graph_file)}: {e}")
                continue

        # Save partition assignments to JSON files (one per case)
        if partition_assignments_by_case:
            logger.info(
                f"\nSaving partition assignments for {len(partition_assignments_by_case)} cases..."
            )
            for case_name, partition_array in partition_assignments_by_case.items():
                partition_json_file = os.path.join(
                    self.dataset_dir, f"{case_name}_partitions.json"
                )
                partition_data = {
                    "case_name": case_name,
                    "num_partitions": self.cfg.preprocessing.num_partitions,
                    "num_nodes": len(partition_array),
                    "partition_assignment": partition_array,  # 1-indexed partition IDs for each active cell
                }
                with open(partition_json_file, "w") as f:
                    json.dump(partition_data, f, indent=2)
            logger.info(
                f"  → Saved partition assignments to {self.dataset_dir}/*_partitions.json"
            )

        logger.info(
            f"Partitioning complete! {successful_partitions}/{len(graph_files)} graphs processed successfully"
        )

    def check_existing_data(self):
        """
        Check if preprocessing data already exists and ask user for overwrite decision.

        Returns
        --------
        bool: Whether to overwrite existing data
        """
        graphs_exist = (
            os.path.exists(self.graphs_dir)
            and len([f for f in os.listdir(self.graphs_dir) if f.endswith(".pt")]) > 0
        )
        stats_exist = os.path.exists(self.stats_file)

        if not graphs_exist and not stats_exist:
            return True  # No existing data, proceed normally

        logger.warning("\nWARNING: Existing preprocessing data detected:")
        if graphs_exist:
            graph_count = len(
                [f for f in os.listdir(self.graphs_dir) if f.endswith(".pt")]
            )
            logger.warning(
                f"   → Graphs directory exists with {graph_count} graph files"
            )
        if stats_exist:
            logger.warning(f"   → Global statistics file exists")

        # Check if we're in a non-interactive environment
        if not sys.stdin.isatty():
            logger.info(
                "\nNon-interactive environment detected. Auto-selecting 'y' (overwrite)"
            )
            logger.info("Will overwrite all existing data")
            return True

        logger.info("\nOptions:")
        logger.info("y. Overwrite all existing data and start fresh")
        logger.info("n. Exit")

        while True:
            try:
                choice = input("\nOverwrite existing data? (y/n): ").strip().lower()
                if choice in ["y", "yes"]:
                    logger.info("Will overwrite all existing data")
                    return True
                elif choice in ["n", "no"]:
                    logger.info("Exiting preprocessing")
                    sys.exit(0)
                else:
                    logger.info("Invalid choice. Please enter y or n.")
            except KeyboardInterrupt:
                logger.info("\nExiting preprocessing")
                sys.exit(0)

    def validate_config(self) -> None:
        """
        Validate configuration parameters relevant to preprocessing.
        """
        logger.info("Validating configuration...")

        # Validate dataset parameters
        if not hasattr(self.cfg, "dataset") or not hasattr(self.cfg.dataset, "sim_dir"):
            raise ValueError("Missing required config: dataset.sim_dir")

        sim_dir_abs = to_absolute_path(self.cfg.dataset.sim_dir)
        if not os.path.exists(sim_dir_abs):
            raise ValueError(f"Simulation directory not found: {sim_dir_abs}")

        # Validate sample count
        num_samples = self.cfg.dataset.get("num_samples", None)
        if num_samples is not None and num_samples < 3:
            raise ValueError(
                f"Insufficient samples: {num_samples} for train/val/test split. Need at least 3."
            )

        # Validate data split ratios
        if hasattr(self.cfg, "preprocessing") and hasattr(
            self.cfg.preprocessing, "data_split"
        ):
            data_split = self.cfg.preprocessing.data_split
            train_ratio = data_split.get("train_ratio", 0.7)
            val_ratio = data_split.get("val_ratio", 0.2)
            test_ratio = data_split.get("test_ratio", 0.1)

            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 1e-6:
                raise ValueError(
                    f"Data split ratios must sum to 1.0, but got {total_ratio:.6f} (train={train_ratio}, val={val_ratio}, test={test_ratio})"
                )

            if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
                raise ValueError(
                    f"All split ratios must be positive. Got train={train_ratio}, val={val_ratio}, test={test_ratio}"
                )

        # Validate preprocessing parameters
        if hasattr(self.cfg, "preprocessing"):
            num_partitions = getattr(self.cfg.preprocessing, "num_partitions", 3)
            halo_size = getattr(self.cfg.preprocessing, "halo_size", 1)

            if num_partitions < 1:
                raise ValueError(f"num_partitions must be >= 1, got {num_partitions}")

            if halo_size < 0:
                raise ValueError(f"halo_size must be >= 0, got {halo_size}")

        logger.info("Configuration validation passed!")

    def execute(self):
        """
        Execute the complete preprocessing pipeline.

        This method orchestrates the entire preprocessing workflow:
        1. Create raw graphs from simulation data
        2. Create partitions from raw graphs
        3. Split samples and organize partitions
        4. Compute global statistics
        5. Save preprocessing metadata
        """
        logger.info("Reservoir Simulation XMeshGraphNet Preprocessor")
        logger.info("=" * 50)

        # Validate configuration first
        self.validate_config()

        # Check for existing data and get user input
        overwrite_data = self.check_existing_data()

        # Get skip options
        skip_graphs = (
            getattr(self.cfg.preprocessing, "skip_graphs", False) or not overwrite_data
        )

        # Step 1: Create raw graphs (unless skipped)
        if not skip_graphs:
            logger.info("\nStep 1: Creating graphs from simulation data...")
            processor = ReservoirGraphBuilder(self.cfg)

            # Override the output path to use our job-specific dataset directory
            processor._output_path_graph = self.graphs_dir
            os.makedirs(self.graphs_dir, exist_ok=True)

            self.generated_files = processor.execute()

            # Save list of generated graph files
            self.save_graph_file_list(
                [os.path.join(self.graphs_dir, f) for f in self.generated_files]
            )
            self.graph_file_list = self.generated_files
        else:
            logger.info(
                "\nStep 1: Skipping graph generation (using existing graphs)..."
            )
            if not os.path.exists(self.graphs_dir):
                raise FileNotFoundError(
                    f"Graphs directory not found: {self.graphs_dir}"
                )

            # Load existing graph file list
            self.graph_file_list = self.load_graph_file_list()
            if self.graph_file_list is None:
                logger.info(
                    "   → No tracked graph files found, will process all .pt files"
                )
                self.graph_file_list = None

        # Step 2: Create partitions from the raw graphs
        partitions_exist = (
            os.path.exists(self.partitions_dir)
            and len([f for f in os.listdir(self.partitions_dir) if f.endswith(".pt")])
            > 0
        )

        # Validate partition topology if partitions exist
        topology_valid = False
        if partitions_exist and not overwrite_data:
            topology_valid = self.validate_partition_topology()
            if not topology_valid:
                logger.warning(
                    "Existing partitions do not match current configuration. "
                    "Partitions will be recreated."
                )

        if overwrite_data or not partitions_exist or not topology_valid:
            logger.info("\nStep 2: Creating partitions from graphs...")
            self.create_partitions_from_graphs(graph_file_list=self.graph_file_list)
        else:
            logger.info(
                "\nStep 2: Skipping partition creation (using existing partitions)"
            )
            logger.info(f"   → Using existing partitions from {self.partitions_dir}")

        # Step 2b: Split samples and organize partitions
        # Check if all split directories exist (train, val, test)
        train_dir = os.path.join(self.partitions_dir, "train")
        val_dir = os.path.join(self.partitions_dir, "val")
        test_dir = os.path.join(self.partitions_dir, "test")
        splits_exist = all(os.path.exists(d) for d in [train_dir, val_dir, test_dir])

        if overwrite_data or not splits_exist:
            if not splits_exist:
                logger.info("\nStep 2b: Splitting samples and organizing partitions...")
                logger.info(
                    "   → One or more split directories (train/val/test) are missing"
                )
            else:
                logger.info("\nStep 2b: Splitting samples and organizing partitions...")

            # Get split configuration
            data_split = getattr(self.cfg.preprocessing, "data_split", {})
            train_ratio = data_split.get("train_ratio", 0.7)
            val_ratio = data_split.get("val_ratio", 0.2)
            test_ratio = data_split.get("test_ratio", 0.1)
            random_seed = data_split.get("random_seed", 42)

            # Split samples by case
            splits = self.split_samples_by_case(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                random_seed=random_seed,
            )

            # Organize partitions into subdirectories
            self.organize_partitions_by_split(splits)
        else:
            logger.info(
                "\nStep 2b: Skipping partition organization (using existing splits)"
            )
            logger.info(
                f"   → Using existing train/val/test splits in {self.partitions_dir}"
            )

        # Step 3: Compute and save global statistics
        if overwrite_data or not os.path.exists(self.stats_file):
            logger.info("\nStep 3: Computing global statistics...")

            # Get all graph files
            graph_files = [
                os.path.join(self.graphs_dir, f)
                for f in os.listdir(self.graphs_dir)
                if f.endswith(".pt")
            ]

            logger.info(
                f"   → Computing statistics from {len(graph_files)} graph files..."
            )
            logger.info(
                f"   → This includes node features, edge features, and target features"
            )

            # Suppress METIS logging during statistics computation
            with self.suppress_all_output():
                stats = compute_global_statistics(graph_files, self.stats_file)

            if stats is not None:
                logger.info(
                    f"Global statistics computed and saved to {self.stats_file}"
                )
                logger.info(
                    f"   → Node features: {len(stats['node_features']['mean'])} features"
                )
                logger.info(
                    f"   → Edge features: {len(stats['edge_features']['mean'])} features"
                )
                if "target_features" in stats:
                    logger.info(
                        f"   → Target features: {len(stats['target_features']['mean'])} features"
                    )
                else:
                    logger.info(
                        f"   → Target features: Not found (graphs may not have target data)"
                    )
            else:
                logger.error("Failed to compute global statistics")
        else:
            logger.info(
                "\nStep 3: Skipping statistics computation (using existing file)"
            )
            logger.info(f"   → Using existing statistics from {self.stats_file}")

        # Step 4: Save preprocessing metadata
        logger.info("\nStep 4: Saving preprocessing metadata...")
        # Always save metadata in the outputs directory
        # Since hydra.run.dir is not available when running preprocessor directly,
        # we'll use the current directory (which should be the outputs directory when run through Hydra)
        outputs_dir = os.getcwd()
        metadata_file = os.path.join(outputs_dir, "preprocessing_metadata.json")
        self.save_preprocessing_metadata(metadata_file)

        # Step 5: Save dataset metadata for inference
        logger.info("\nStep 5: Saving dataset metadata...")
        self.save_dataset_metadata()

        logger.info("\nPreprocessing complete!")
        logger.info(f"   → Raw graphs: {self.graphs_dir}")
        logger.info(f"   → Partitions: {self.partitions_dir}")


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to preprocess reservoir simulation data.
    """

    preprocessor = ReservoirPreprocessor(cfg)

    preprocessor.execute()


if __name__ == "__main__":
    main()
