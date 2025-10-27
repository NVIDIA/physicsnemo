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
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data.graph_builder import ReservoirGraphBuilder
from data.dataloader import PartitionedGraph, compute_global_statistics
from utils.path_utils import get_dataset_dir


class ReservoirPreprocessor:
    """
    A class to handle the complete preprocessing pipeline for reservoir simulation data.

    This class manages the creation of raw graphs from simulation data, partitioning them
    for efficient training, computing global statistics, and organizing data splits.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the ReservoirPreprocessor with configuration.

        Parameters:
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
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Job name: {job_name}")

    def _extract_case_name_from_filename(self, filename):
        """
        Extract case name from a graph filename by removing the timestep suffix.

        Expected format: {case_name}_{timestep:03d}.pt
        where timestep is typically 3 digits (e.g., 000, 001, 123).

        Examples:
            CASE_2D_1_000.pt -> CASE_2D_1
            NORNE_ATW2013_DOE_0004_002.pt -> NORNE_ATW2013_DOE_0004
            sample_005_123.pt -> sample_005

        Parameters:
        -----------
        filename : str
            Graph filename (with or without .pt extension)

        Returns:
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

        Parameters:
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

        print(f"Saved graph file list to: {list_path}")

    def load_graph_file_list(self, list_file="generated_graphs.json"):
        """
        Load list of generated graph files.

        Parameters:
        -----------
        list_file : str
            Path to graph file list

        Returns:
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

        Parameters:
        -----------
        metadata_file : str
            Path to save metadata file
        """
        metadata = {
            "graphs_dir": self.graphs_dir,
            "partitions_dir": self.partitions_dir,
            "preprocessing_completed": True,
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved preprocessing metadata to: {metadata_file}")

    def save_dataset_metadata(self, metadata_file="dataset_metadata.json"):
        """
        Save dataset metadata for inference use.

        Parameters:
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
                "uniform_geometry": getattr(self.cfg.dataset, "uniform_geometry", True),
                "num_samples": getattr(self.cfg.dataset, "num_samples", None),
            },
        }

        metadata_path = os.path.join(self.dataset_dir, metadata_file)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved dataset metadata to: {metadata_path}")

    def split_samples_by_case(self, train_ratio, val_ratio, test_ratio, random_seed=42):
        """
        Split graph files by case (sample) to ensure all timesteps of a sample stay together.

        Parameters:
        -----------
        train_ratio : float
            Ratio of samples for training
        val_ratio : float
            Ratio of samples for validation
        test_ratio : float
            Ratio of samples for testing
        random_seed : int
            Random seed for reproducible splits

        Returns:
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
        print(f"Found {total_cases} unique cases: {case_names}")

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

        print(f"Sample split:")
        print(
            f"   Training: {len(train_cases)} cases ({len(train_cases) / total_cases * 100:.1f}%)"
        )
        print(
            f"   Validation: {len(val_cases)} cases ({len(val_cases) / total_cases * 100:.1f}%)"
        )
        print(
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

        print(f"File split:")
        print(f"   Training: {len(splits['train'])} files")
        print(f"   Validation: {len(splits['val'])} files")
        print(f"   Test: {len(splits['test'])} files")

        return splits

    def organize_partitions_by_split(self, splits):
        """
        Create partitions and organize them into train/val/test subdirectories.

        Parameters:
        -----------
        splits : dict
            Dictionary with 'train', 'val', 'test' keys containing file lists
        """
        print(f"\nOrganizing partitions by split...")

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
                print(f"   → {split_name.capitalize()}: No files to process")
                continue

            split_dir = os.path.join(self.partitions_dir, split_name)
            print(f"   → Processing {split_name} split: {len(file_list)} files")

            moved_count = 0
            print(f"Organizing {split_name} split ({len(file_list)} files)...")
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

            print(f"     Moved {moved_count}/{len(file_list)} files to {split_name}/")
            total_moved += moved_count

        print(f"Partition organization complete!")

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
        """Create a simple sequential partition as fallback when METIS is not available."""

        # Create a simple partition object that mimics the METIS partition structure
        class SimplePartition:
            def __init__(self, num_nodes, num_parts):
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

        return SimplePartition(num_nodes, num_parts)

    def create_partitions_from_graphs(self, graph_file_list=None):
        """
        Create partitions from raw graphs for efficient training.

        Parameters:
        -----------
        graph_file_list : list or None
            List of specific graph files to process (if None, process all .pt files)
        """
        print(f"\nCreating partitions from graphs...")

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

        print(
            f"   → Processing {len(graph_files)} graphs with {self.cfg.preprocessing.num_partitions} partitions each..."
        )

        # Process each graph file
        successful_partitions = 0
        print(f"Creating partitions for {len(graph_files)} graphs...")
        for i, graph_file in enumerate(graph_files, 1):
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
                    print(
                        f"     WARNING: METIS partitioning failed ({e}), using simple partitioning..."
                    )
                    # Fallback: simple sequential partitioning
                    part_meta = self.create_simple_partition(
                        graph.num_nodes, self.cfg.preprocessing.num_partitions
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
                print(
                    f"  [{i}/{len(graph_files)}] Partitioned {os.path.basename(graph_file)}"
                )

            except Exception as e:
                print(f"ERROR: processing {os.path.basename(graph_file)}: {e}")
                continue

        print(
            f"Partitioning complete! {successful_partitions}/{len(graph_files)} graphs processed successfully"
        )

    def check_existing_data(self):
        """
        Check if preprocessing data already exists and ask user for overwrite decision.

        Returns:
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

        print("\nWARNING: Existing preprocessing data detected:")
        if graphs_exist:
            graph_count = len(
                [f for f in os.listdir(self.graphs_dir) if f.endswith(".pt")]
            )
            print(f"   → Graphs directory exists with {graph_count} graph files")
        if stats_exist:
            print(f"   → Global statistics file exists")

        # Check if we're in a non-interactive environment
        if not sys.stdin.isatty():
            print(
                "\nNon-interactive environment detected. Auto-selecting 'y' (overwrite)"
            )
            print("Will overwrite all existing data")
            return True

        print("\nOptions:")
        print("y. Overwrite all existing data and start fresh")
        print("n. Exit and handle manually")

        while True:
            try:
                choice = input("\nOverwrite existing data? (y/n): ").strip().lower()
                if choice in ["y", "yes"]:
                    print("Will overwrite all existing data")
                    return True
                elif choice in ["n", "no"]:
                    print("Exiting preprocessing")
                    sys.exit(0)
                else:
                    print("Invalid choice. Please enter y or n.")
            except KeyboardInterrupt:
                print("\nExiting preprocessing")
                sys.exit(0)

    def validate_config(self) -> None:
        """
        Validate configuration parameters relevant to preprocessing.
        """
        print("🔍 Validating configuration...")

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

        print("Configuration validation passed!")

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
        print("Reservoir Simulation XMeshGraphNet Preprocessor")
        print("=" * 50)

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
            print("\nStep 1: Creating raw graphs from simulation data...")
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
            print("\nStep 1: Skipping graph generation (using existing graphs)...")
            if not os.path.exists(self.graphs_dir):
                raise FileNotFoundError(
                    f"Graphs directory not found: {self.graphs_dir}"
                )

            # Load existing graph file list
            self.graph_file_list = self.load_graph_file_list()
            if self.graph_file_list is None:
                print("   → No tracked graph files found, will process all .pt files")
                self.graph_file_list = None

        # Step 2: Create partitions from the raw graphs
        if (
            overwrite_data
            or not os.path.exists(self.partitions_dir)
            or len([f for f in os.listdir(self.partitions_dir) if f.endswith(".pt")])
            == 0
        ):
            print("\nStep 2: Creating partitions from raw graphs...")
            self.create_partitions_from_graphs(graph_file_list=self.graph_file_list)
        else:
            print("\nStep 2: Skipping partition creation (using existing partitions)")
            print(f"   → Using existing partitions from {self.partitions_dir}")

        # Step 2b: Split samples and organize partitions
        if overwrite_data or not os.path.exists(
            os.path.join(self.partitions_dir, "train")
        ):
            print("\nStep 2b: Splitting samples and organizing partitions...")

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
            print("\nStep 2b: Skipping partition organization (using existing splits)")
            print(f"   → Using existing train/val/test splits in {self.partitions_dir}")

        # Step 3: Compute and save global statistics
        if overwrite_data or not os.path.exists(self.stats_file):
            print("\nStep 3: Computing global statistics...")

            # Get all graph files
            graph_files = [
                os.path.join(self.graphs_dir, f)
                for f in os.listdir(self.graphs_dir)
                if f.endswith(".pt")
            ]

            print(f"   → Computing statistics from {len(graph_files)} graph files...")
            print(
                f"   → This includes node features, edge features, and target features"
            )

            # Suppress METIS logging during statistics computation
            with self.suppress_all_output():
                stats = compute_global_statistics(graph_files, self.stats_file)

            if stats is not None:
                print(f"Global statistics computed and saved to {self.stats_file}")
                print(
                    f"   → Node features: {len(stats['node_features']['mean'])} features"
                )
                print(
                    f"   → Edge features: {len(stats['edge_features']['mean'])} features"
                )
                if "target_features" in stats:
                    print(
                        f"   → Target features: {len(stats['target_features']['mean'])} features"
                    )
                else:
                    print(
                        f"   → Target features: Not found (graphs may not have target data)"
                    )
            else:
                print("Failed to compute global statistics")
        else:
            print("\nStep 3: Skipping statistics computation (using existing file)")
            print(f"   → Using existing statistics from {self.stats_file}")

        # Step 4: Save preprocessing metadata
        print("\nStep 4: Saving preprocessing metadata...")
        # Always save metadata in the outputs directory
        # Since hydra.run.dir is not available when running preprocessor directly,
        # we'll use the current directory (which should be the outputs directory when run through Hydra)
        outputs_dir = os.getcwd()
        metadata_file = os.path.join(outputs_dir, "preprocessing_metadata.json")
        self.save_preprocessing_metadata(metadata_file)

        # Step 5: Save dataset metadata for inference
        print("\nStep 5: Saving dataset metadata...")
        self.save_dataset_metadata()

        print("\nPreprocessing complete!")
        print(f"   → Raw graphs: {self.graphs_dir}")
        print(f"   → Partitions: {self.partitions_dir}")


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to preprocess reservoir simulation data.
    """

    preprocessor = ReservoirPreprocessor(cfg)

    preprocessor.execute()


if __name__ == "__main__":
    main()
