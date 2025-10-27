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
Inference script for XMeshGraphNet on reservoir simulation data.
Loads the best checkpoint and performs autoregressive inference on test samples.
Generates GRDECL files with predictions for post-processing.
"""

import os
import sys
import json
from datetime import datetime

# Add repository root to Python path for sim_utils import
current_dir = os.path.dirname(os.path.abspath(__file__))  # This is src/
repo_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels from src/
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
import numpy as np
import h5py
import hydra
from omegaconf import DictConfig

# Fix LayerNorm compatibility issue
if not hasattr(nn.LayerNorm, "register_load_state_dict_pre_hook"):
    nn.LayerNorm.register_load_state_dict_pre_hook = lambda self, hook: None

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.launch.utils import load_checkpoint
from data.dataloader import GraphDataset, load_stats, find_pt_files
from sim_utils import EclReader, Grid
from utils.path_utils import get_dataset_paths


def InitializeLoggers(cfg: DictConfig):
    """Initialize distributed manager and loggers for inference."""
    DistributedManager.initialize()
    dist = DistributedManager()
    logger = PythonLogger(name="xmgn_inference")

    logger.info("XMeshGraphNet - Autoregressive Inference for Reservoir Simulation")

    return dist, RankZeroLoggingWrapper(logger, dist)


class InferenceRunner:
    """Inference runner for XMeshGraphNet."""

    def __init__(self, cfg: DictConfig, dist, logger):
        """Initialize the inference runner."""
        self.cfg = cfg
        self.dist = dist
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up paths with job name
        paths = get_dataset_paths(cfg)
        self.dataset_dir = paths["dataset_dir"]
        self.stats_file = paths["stats_file"]
        self.test_partitions_path = paths["test_partitions_path"]

        # Set up inference output directory
        self.inference_output_dir = "inference"
        self.inference_metadata_file = os.path.join(
            self.inference_output_dir, "inference_metadata.json"
        )

        # Load global statistics
        self.stats = load_stats(self.stats_file)

        # Get model dimensions
        input_dim_nodes = len(self.stats["node_features"]["mean"])
        input_dim_edges = len(self.stats["edge_features"]["mean"])
        output_dim = len(cfg.dataset.graph.target_vars.node_features)

        # Initialize model
        self.model = MeshGraphNet(
            input_dim_nodes=input_dim_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=output_dim,
            processor_size=cfg.model.num_message_passing_layers,
            aggregation="sum",
            hidden_dim_node_encoder=cfg.model.hidden_dim,
            hidden_dim_edge_encoder=cfg.model.hidden_dim,
            hidden_dim_node_decoder=cfg.model.hidden_dim,
            mlp_activation_fn=cfg.model.activation,
            do_concat_trick=cfg.performance.use_concat_trick,
        ).to(self.device)

        # Load best checkpoint using PhysicsNeMo's load_checkpoint (same as training)
        # Set up checkpoint arguments (same as training)
        base_output_dir = os.getcwd()
        best_checkpoint_dir = os.path.join(base_output_dir, "best_checkpoints")

        # Set up checkpoint arguments (following training pattern - same as bst_ckpt_args)
        ckpt_args = {
            "path": best_checkpoint_dir,
            "models": self.model,
        }

        # Check for explicit checkpoint paths in config
        explicit_checkpoint = getattr(cfg.inference, "checkpoint_path", None)
        explicit_model = getattr(cfg.inference, "model_path", None)

        if explicit_checkpoint or explicit_model:
            # Use explicit checkpoint/model paths
            if explicit_checkpoint:
                self.logger.info(f"Using explicit checkpoint: {explicit_checkpoint}")
                checkpoint = torch.load(explicit_checkpoint, map_location=self.device)

                # Load model state
                if "models" in checkpoint:
                    self.model.load_state_dict(checkpoint["models"])
                else:
                    self.model.load_state_dict(checkpoint)

                # Extract epoch from filename if possible
                filename = os.path.basename(explicit_checkpoint)
                try:
                    parts = filename.split(".")
                    if len(parts) >= 3:
                        loaded_epoch = int(parts[2])
                    else:
                        loaded_epoch = 0
                except Exception:
                    loaded_epoch = 0

                self.logger.info(
                    f"Loaded explicit checkpoint from epoch {loaded_epoch}"
                )

            elif explicit_model:
                self.logger.info(f"Using explicit model: {explicit_model}")
                # For .mdlus files, we need to use PhysicsNeMo's load_checkpoint
                model_ckpt_args = {
                    "path": os.path.dirname(explicit_model),
                    "models": self.model,
                }
                loaded_epoch = load_checkpoint(**model_ckpt_args, device=self.device)
                self.logger.info(f"Loaded explicit model from epoch {loaded_epoch}")
        else:
            # Use automatic best checkpoint selection
            self.logger.info("Using automatic best checkpoint selection")

            # Check for multiple checkpoint files and log them
            if os.path.exists(best_checkpoint_dir):
                checkpoint_files = [
                    f for f in os.listdir(best_checkpoint_dir) if f.endswith(".pt")
                ]
                if len(checkpoint_files) > 1:
                    self.logger.info(
                        f"Found {len(checkpoint_files)} checkpoint files in best_checkpoints:"
                    )
                    for file in sorted(checkpoint_files):
                        self.logger.info(f"   - {file}")
                    self.logger.info(
                        "PhysicsNeMo will automatically select the best performing checkpoint"
                    )

            # Load checkpoint using PhysicsNeMo's system
            loaded_epoch = load_checkpoint(**ckpt_args, device=self.device)
            self.logger.info(f"Loaded BEST checkpoint from epoch {loaded_epoch}")

        self.model.eval()
        self.logger.info(f"Checkpoint directory: {best_checkpoint_dir}")

        # Create test dataset (following training pattern)
        # Find partition files
        file_paths = find_pt_files(self.test_partitions_path)

        # Load per-feature statistics
        node_mean = torch.tensor(self.stats["node_features"]["mean"])
        node_std = torch.tensor(self.stats["node_features"]["std"])
        edge_mean = torch.tensor(self.stats["edge_features"]["mean"])
        edge_std = torch.tensor(self.stats["edge_features"]["std"])

        # Load target feature statistics (if available)
        target_mean = None
        target_std = None
        if "target_features" in self.stats:
            target_mean = torch.tensor(self.stats["target_features"]["mean"])
            target_std = torch.tensor(self.stats["target_features"]["std"])

        # Create dataset
        self.test_dataset = GraphDataset(
            file_paths,
            node_mean,
            node_std,
            edge_mean,
            edge_std,
            target_mean,
            target_std,
        )

        self.logger.info(f"Test dataset loaded with {len(self.test_dataset)} samples")

    def denormalize_predictions(self, pred):
        """Denormalize predictions using global statistics."""
        target_mean = torch.tensor(
            self.stats["target_features"]["mean"], device=self.device
        )
        target_std = torch.tensor(
            self.stats["target_features"]["std"], device=self.device
        )
        return pred * target_std + target_mean

    def denormalize_targets(self, target):
        """Denormalize targets using global statistics."""
        target_mean = torch.tensor(
            self.stats["target_features"]["mean"], device=self.device
        )
        target_std = torch.tensor(
            self.stats["target_features"]["std"], device=self.device
        )
        return target * target_std + target_mean

    def _get_target_feature_indices(self):
        """
        Get the indices in node features that correspond to target variables.
        These are the features we need to replace with predictions during autoregressive inference.
        """
        # Get target variable names
        target_vars = self.cfg.dataset.graph.target_vars.node_features

        # Get dynamic variable names from config
        dynamic_vars = self.cfg.dataset.graph.node_features.dynamic.variables

        # Find indices of target variables in dynamic variables
        target_indices = []
        for target_var in target_vars:
            if target_var in dynamic_vars:
                idx = dynamic_vars.index(target_var)
                target_indices.append(idx)

        return target_indices, target_vars

    def _update_node_features_with_predictions(
        self, partitions_list, predictions_normalized
    ):
        """
        Update node features in partitions with predictions from previous timestep.
        Replace only the features that correspond to target variables.

        Args:
            partitions_list: List of graph partitions
            predictions_normalized: Normalized predictions from previous timestep (list of arrays per partition)

        Returns:
            Updated partitions_list with predictions in node features
        """
        target_indices, _ = self._get_target_feature_indices()

        # Get the number of dynamic variables to know the offset in node features
        num_static_features = len(self.cfg.dataset.graph.node_features.static.variables)
        num_dynamic_features = len(
            self.cfg.dataset.graph.node_features.dynamic.variables
        )
        prev_timesteps = self.cfg.dataset.graph.node_features.dynamic.prev_timesteps

        # Dynamic features start after static features
        # For prev_timesteps=0: dynamic features are at indices [num_static: num_static+num_dynamic]
        # For prev_timesteps>0: current timestep is at the end of dynamic features

        if prev_timesteps == 0:
            # Current timestep dynamic features start at num_static_features
            dynamic_offset = num_static_features
        else:
            # Current timestep is at the last block of dynamic features
            dynamic_offset = num_static_features + prev_timesteps * num_dynamic_features

        # Update each partition
        updated_partitions = []
        for partition, pred_array in zip(partitions_list, predictions_normalized):
            # Clone the partition to avoid modifying the original
            # PyTorch Geometric Data objects need special handling
            if hasattr(partition, "clone"):
                partition = partition.clone()

            # Clone the node features tensor
            partition.x = partition.x.clone()

            # Convert prediction array to tensor if needed
            if isinstance(pred_array, np.ndarray):
                pred_tensor = torch.tensor(
                    pred_array, dtype=torch.float32, device=partition.x.device
                )
            else:
                pred_tensor = (
                    pred_array.clone() if hasattr(pred_array, "clone") else pred_array
                )

            # Replace target features in node features with predictions
            for i, target_idx in enumerate(target_indices):
                feature_idx = dynamic_offset + target_idx
                partition.x[:, feature_idx] = pred_tensor[:, i]

            updated_partitions.append(partition)

        return updated_partitions

    def evaluate_sample(
        self,
        partitions_list,
        use_predictions_as_input=False,
        prev_predictions_normalized=None,
    ):
        """
        Evaluate a single sample (list of partitions).

        Args:
            partitions_list: List of graph partitions for this timestep
            use_predictions_as_input: If True, replace target features with predictions from previous timestep
            prev_predictions_normalized: Normalized predictions from previous timestep (for autoregressive inference)

        Returns:
            avg_loss, avg_denorm_loss, predictions, targets, predictions_normalized
        """
        total_loss = 0.0
        total_denorm_loss = 0.0
        num_partitions = 0

        predictions = []
        targets = []
        predictions_normalized = []  # Store normalized predictions for next timestep

        with torch.no_grad():
            # If using autoregressive mode, update node features with previous predictions
            if use_predictions_as_input and prev_predictions_normalized is not None:
                partitions_list = self._update_node_features_with_predictions(
                    partitions_list, prev_predictions_normalized
                )

            for partition in partitions_list:
                partition = partition.to(self.device)

                # Ensure data is in float32
                if hasattr(partition, "x") and partition.x is not None:
                    partition.x = partition.x.float()
                if hasattr(partition, "edge_attr") and partition.edge_attr is not None:
                    partition.edge_attr = partition.edge_attr.float()
                if hasattr(partition, "y") and partition.y is not None:
                    partition.y = partition.y.float()

                # Forward pass
                pred = self.model(partition.x, partition.edge_attr, partition)

                # Get inner nodes if available
                if hasattr(partition, "inner_node"):
                    pred_inner = pred[partition.inner_node]
                    target_inner = partition.y[partition.inner_node]
                else:
                    pred_inner = pred
                    target_inner = partition.y

                # Calculate losses
                loss = torch.nn.functional.mse_loss(pred_inner, target_inner)

                # Denormalize for physical units
                pred_denorm = self.denormalize_predictions(pred_inner)
                target_denorm = self.denormalize_targets(target_inner)
                denorm_loss = torch.nn.functional.mse_loss(pred_denorm, target_denorm)

                total_loss += loss.item()
                total_denorm_loss += denorm_loss.item()
                num_partitions += 1

                # Store predictions and targets
                predictions.append(pred_denorm.cpu().numpy())
                targets.append(target_denorm.cpu().numpy())

                # Store normalized predictions for next timestep's input
                predictions_normalized.append(pred_inner.cpu().numpy())

        avg_loss = total_loss / num_partitions if num_partitions > 0 else 0.0
        avg_denorm_loss = (
            total_denorm_loss / num_partitions if num_partitions > 0 else 0.0
        )

        return avg_loss, avg_denorm_loss, predictions, targets, predictions_normalized

    def _extract_case_and_timestep(self, filename):
        """Extract case name and time step from filename."""

        if filename.startswith("partitions_"):
            # Remove 'partitions_' prefix
            filename = filename[11:]  # Remove 'partitions_'

        # Remove .pt extension
        filename = filename.replace(".pt", "")

        # Split by underscore and extract case name and time step
        parts = filename.split("_")
        if len(parts) >= 4:
            # Format: CASE_2D_1_000
            case_name = "_".join(parts[:-1])  # CASE_2D_1
            timestep = parts[-1]  # 000
        else:
            # Fallback
            case_name = filename
            timestep = "000"

        return case_name, timestep

    def run_inference(self):
        """Run autoregressive inference on test dataset."""
        self.logger.info("=" * 70)
        self.logger.info("STARTING INFERENCE")
        self.logger.info("=" * 70)

        # Get prev_timesteps config for determining initial conditions
        prev_timesteps = self.cfg.dataset.graph.node_features.dynamic.prev_timesteps
        num_initial_true_timesteps = (
            prev_timesteps + 1
        )  # Initial + prev_timesteps as true inputs

        self.logger.info(
            f"Initial timesteps with true features: {num_initial_true_timesteps}"
        )
        self.logger.info(
            f"Subsequent timesteps: predictions feed into next timestep (autoregressive)"
        )

        # First, organize all samples by case and timestep
        case_timestep_data = {}
        for idx in range(len(self.test_dataset)):
            file_path = self.test_dataset.file_paths[idx]
            filename = os.path.basename(file_path)
            case_name, timestep = self._extract_case_and_timestep(filename)

            if case_name not in case_timestep_data:
                case_timestep_data[case_name] = {}

            case_timestep_data[case_name][timestep] = idx

        # Now process each case autoregressively
        total_loss = 0.0
        total_denorm_loss = 0.0
        num_samples = 0
        case_results = {}

        all_cases = sorted(case_timestep_data.keys())
        total_cases = len(all_cases)
        self.logger.info(f"Processing {total_cases} cases...")

        for case_idx, case_name in enumerate(all_cases, 1):
            case_results[case_name] = {
                "predictions": {},
                "targets": {},
                "losses": [],
                "denorm_losses": [],
            }

            # Get sorted timesteps for this case
            timesteps = sorted(case_timestep_data[case_name].keys())

            self.logger.info(
                f"[{case_idx}/{total_cases}] Processing case: {case_name} ({len(timesteps)} timesteps)"
            )

            # Track predictions from previous timestep (normalized)
            prev_predictions_normalized = None

            # Process each timestep in order
            for timestep_idx, timestep in enumerate(timesteps):
                idx = case_timestep_data[case_name][timestep]
                partitions_list, label = self.test_dataset[idx]

                # Determine if we should use predictions as input
                # Use true features for first num_initial_true_timesteps, then use predictions
                use_predictions_as_input = timestep_idx >= num_initial_true_timesteps

                # Evaluate this timestep
                loss, denorm_loss, predictions, targets, predictions_normalized = (
                    self.evaluate_sample(
                        partitions_list,
                        use_predictions_as_input=use_predictions_as_input,
                        prev_predictions_normalized=prev_predictions_normalized,
                    )
                )

                total_loss += loss
                total_denorm_loss += denorm_loss
                num_samples += 1

                # Store results
                case_results[case_name]["predictions"][timestep] = predictions
                case_results[case_name]["targets"][timestep] = targets
                case_results[case_name]["losses"].append(loss)
                case_results[case_name]["denorm_losses"].append(denorm_loss)

                # Store predictions for next timestep
                prev_predictions_normalized = predictions_normalized

        # Calculate final metrics
        avg_loss = total_loss / num_samples
        avg_denorm_loss = total_denorm_loss / num_samples

        # Save results per simulation case as HDF5 files
        self._save_case_results_hdf5(case_results)

        # Calculate overall metrics for logging
        all_predictions = []
        all_targets = []
        for case_data in case_results.values():
            for timestep_preds in case_data["predictions"].values():
                all_predictions.extend(timestep_preds)
            for timestep_targets in case_data["targets"].values():
                all_targets.extend(timestep_targets)

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate additional metrics
        mae = np.mean(np.abs(all_predictions - all_targets))
        mse = np.mean((all_predictions - all_targets) ** 2)
        rmse = np.sqrt(mse)

        # Log final results
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("AUTOREGRESSIVE INFERENCE RESULTS")
        self.logger.info("=" * 70)
        self.logger.info(f"Test samples processed: {num_samples}")
        self.logger.info(f"Simulation cases: {len(case_results)}")
        self.logger.info(f"Average normalized MSE: {avg_loss:.6e}")
        self.logger.info(f"Average denormalized MSE: {avg_denorm_loss:.6e}")
        self.logger.info(f"Overall MAE: {mae:.6e}")
        self.logger.info(f"Overall RMSE: {rmse:.6e}")
        self.logger.info("")
        self.logger.info("Per-Variable Metrics:")
        self.logger.info("-" * 70)

        # Per-variable metrics
        target_names = self.cfg.dataset.graph.target_vars.node_features
        for i, var_name in enumerate(target_names):
            var_mae = np.mean(np.abs(all_predictions[:, i] - all_targets[:, i]))
            var_rmse = np.sqrt(
                np.mean((all_predictions[:, i] - all_targets[:, i]) ** 2)
            )
            self.logger.info(
                f"  {var_name:>12s}  |  MAE: {var_mae:>12.6e}  |  RMSE: {var_rmse:>12.6e}"
            )

        self.logger.info("=" * 70)

        return {
            "avg_loss": avg_loss,
            "avg_denorm_loss": avg_denorm_loss,
            "mae": mae,
            "rmse": rmse,
            "predictions": all_predictions,
            "targets": all_targets,
            "num_samples": num_samples,
            "case_results": case_results,
        }

    def _save_case_results_hdf5(self, case_results):
        """Save inference results per simulation case as HDF5 files."""
        os.makedirs(self.inference_output_dir, exist_ok=True)

        self.logger.info("")
        self.logger.info("Saving inference results to HDF5 files...")

        target_names = self.cfg.dataset.graph.target_vars.node_features

        for case_name, case_data in case_results.items():
            hdf5_file = os.path.join(self.inference_output_dir, f"{case_name}.hdf5")

            with h5py.File(hdf5_file, "w") as f:
                # Create groups for predictions and targets
                pred_group = f.create_group("predictions")
                target_group = f.create_group("targets")

                # Save metadata
                f.attrs["case_name"] = case_name
                f.attrs["num_timesteps"] = len(case_data["predictions"])
                f.attrs["target_variables"] = [
                    str(name) for name in target_names
                ]  # Convert to list of strings
                f.attrs["avg_loss"] = np.mean(case_data["losses"])
                f.attrs["avg_denorm_loss"] = np.mean(case_data["denorm_losses"])

                # Organize data by variable (PRESSURE, SWAT) with lists of vectors per timestep
                for i, var_name in enumerate(target_names):
                    var_name_clean = var_name.upper()  # Use capital case

                    # Collect all timestep data for this variable
                    pred_vectors = []
                    target_vectors = []

                    for timestep in sorted(case_data["predictions"].keys()):
                        predictions = case_data["predictions"][timestep]
                        targets = case_data["targets"][timestep]

                        if predictions:
                            pred_array = np.concatenate(predictions, axis=0)
                            target_array = np.concatenate(targets, axis=0)

                            # Extract this variable's data (column i)
                            pred_vectors.append(pred_array[:, i])
                            target_vectors.append(target_array[:, i])

                    # Save as variable groups with lists of vectors
                    if pred_vectors:
                        # Create variable groups
                        var_pred_group = pred_group.create_group(var_name_clean)
                        var_target_group = target_group.create_group(var_name_clean)

                        # Save each timestep as a separate dataset within the variable group
                        for timestep_idx, (pred_vec, target_vec) in enumerate(
                            zip(pred_vectors, target_vectors)
                        ):
                            next_timestep = (
                                timestep_idx + 1
                            )  # tstep + 1 because we're predicting next timestep
                            var_pred_group.create_dataset(
                                f"timestep_{next_timestep:03d}", data=pred_vec
                            )
                            var_target_group.create_dataset(
                                f"timestep_{next_timestep:03d}", data=target_vec
                            )

                        # Save metadata for this variable
                        var_pred_group.attrs["num_timesteps"] = len(pred_vectors)
                        var_pred_group.attrs["num_nodes"] = (
                            len(pred_vectors[0]) if pred_vectors else 0
                        )
                        var_target_group.attrs["num_timesteps"] = len(target_vectors)
                        var_target_group.attrs["num_nodes"] = (
                            len(target_vectors[0]) if target_vectors else 0
                        )

        # Save metadata file with list of HDF5 files
        hdf5_files = [f"{case_name}.hdf5" for case_name in case_results.keys()]
        metadata = {
            "hdf5_files": hdf5_files,
            "num_cases": len(case_results),
            "target_variables": [str(name) for name in target_names],
            "created_at": datetime.now().isoformat(),
        }

        with open(self.inference_metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(
            f"Saved {len(case_results)} HDF5 files to: {self.inference_output_dir}"
        )
        self.logger.info(f"Saved metadata: {self.inference_metadata_file}")

    def _extract_coordinates_from_grid(self, sample_idx):
        """Extract coordinates from grid files using the general Grid approach."""
        # Load dataset metadata from preprocessing
        dataset_metadata_file = os.path.join(self.dataset_dir, "dataset_metadata.json")
        if not os.path.exists(dataset_metadata_file):
            raise FileNotFoundError(
                f"Dataset metadata not found at {dataset_metadata_file}. Please run preprocessing first."
            )

        with open(dataset_metadata_file, "r") as f:
            dataset_metadata = json.load(f)

        # Get the case name from the HDF5 metadata
        if not os.path.exists(self.inference_metadata_file):
            raise FileNotFoundError(
                f"No inference metadata found at {self.inference_metadata_file}"
            )

        with open(self.inference_metadata_file, "r") as f:
            inference_metadata = json.load(f)

        hdf5_files = inference_metadata.get("hdf5_files", [])
        if sample_idx >= len(hdf5_files):
            raise IndexError(
                f"Sample index {sample_idx} exceeds available cases ({len(hdf5_files)})"
            )

        # Extract case name from HDF5 filename (remove .hdf5 extension)
        case_name = hdf5_files[sample_idx].replace(".hdf5", "")

        # Get the original sim_dir from dataset metadata
        original_sim_dir = dataset_metadata.get("sim_dir")
        if not original_sim_dir:
            raise KeyError("sim_dir not found in dataset metadata")

        # Construct the path to the simulator data directory using the original path
        data_file = os.path.join(original_sim_dir, f"{case_name}.DATA")

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Simulator data file not found: {data_file}")

        # Create reader and read grid data
        reader = EclReader(data_file)

        # Read grid data (COORD, ZCORN for coordinates)
        egrid_keys = ["COORD", "ZCORN", "FILEHEAD", "NNC1", "NNC2"]
        egrid_data = reader.read_egrid(egrid_keys)

        # Read init data for grid dimensions and porosity
        init_keys = ["INTEHEAD", "PORV"]
        init_data = reader.read_init(init_keys)

        # Create grid object to get coordinates (same as in reservoir_graph_builder.py)
        grid = Grid(init_data, egrid_data)
        X, Y, Z = grid.X, grid.Y, grid.Z

        # Get grid dimensions from the grid object
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        nact = grid.nact  # number of active cells

        self.logger.info(f"Extracted coordinates from grid for {case_name}:")
        self.logger.info(
            f"   Grid dimensions: {nx} × {ny} × {nz} = {nx * ny * nz} total cells"
        )
        self.logger.info(
            f"   Active cells: {nact} ({nact / (nx * ny * nz) * 100:.1f}% of total)"
        )
        self.logger.info(f"   Coordinates: {len(X)} active nodes")

        return X, Y, Z, case_name, (nx, ny, nz), nact, grid

    def run_post(self):
        """
        Generate Eclipse-style GRDECL ASCII files from HDF5 inference results.
        Each HDF5 file is converted to a GRDECL file with format:
        KEY_<timestep>
        <values>
        /
        """
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("POST-PROCESSING: GENERATING GRDECL FILES")
        self.logger.info("=" * 70)

        # Load metadata to get HDF5 files
        if not os.path.exists(self.inference_metadata_file):
            self.logger.warning(
                f"No inference metadata found at {self.inference_metadata_file}"
            )
            return

        with open(self.inference_metadata_file, "r") as f:
            metadata = json.load(f)

        hdf5_files = metadata.get("hdf5_files", [])
        if not hdf5_files:
            self.logger.warning("No HDF5 files found in metadata")
            return

        # Output directory (directly under inference/)
        grdecl_output_dir = self.inference_output_dir
        os.makedirs(grdecl_output_dir, exist_ok=True)

        self.logger.info(f"Output directory: {grdecl_output_dir}")
        self.logger.info(f"Processing {len(hdf5_files)} HDF5 file(s)...")

        # Process each HDF5 file
        for sample_idx, hdf5_filename in enumerate(hdf5_files, 1):
            hdf5_file = os.path.join(self.inference_output_dir, hdf5_filename)
            case_name = os.path.basename(hdf5_filename).replace(".hdf5", "")

            self.logger.info(
                f"[{sample_idx}/{len(hdf5_files)}] Generating GRDECL for {case_name}..."
            )

            # Output GRDECL filename
            grdecl_filename = f"{case_name}.GRDECL"
            grdecl_filepath = os.path.join(grdecl_output_dir, grdecl_filename)

            try:
                # Get grid information and actnum for this case
                X, Y, Z, _, grid_dims, nact, grid = self._extract_coordinates_from_grid(
                    sample_idx
                )
                nx, ny, nz = grid_dims
                total_cells = nx * ny * nz
                actnum = grid.actnum_bool

                with h5py.File(hdf5_file, "r") as f:
                    with open(grdecl_filepath, "w") as grdecl_file:
                        # Get target variables
                        target_variables = f.attrs.get("target_variables", [])

                        # Process each target variable
                        for var_name in target_variables:
                            var_name_clean = var_name.upper()

                            if var_name_clean not in f["predictions"]:
                                self.logger.warning(
                                    f"Variable {var_name_clean} not found in {hdf5_filename}"
                                )
                                continue

                            # Get all timesteps for this variable
                            timesteps = sorted(f["predictions"][var_name_clean].keys())

                            for timestep_key in timesteps:
                                # Extract timestep number from key (e.g., "timestep_001" -> 1)
                                timestep_num = int(timestep_key.split("_")[-1])

                                # Read prediction data (only active cells)
                                pred_data_active = f["predictions"][var_name_clean][
                                    timestep_key
                                ][:]

                                # Initialize full array with zeros for all cells
                                pred_data_full = np.zeros((total_cells,))

                                # Populate only active cells with predicted values
                                pred_data_full[actnum] = pred_data_active

                                # Write in Eclipse GRDECL format
                                # KEY_<timestep> with 3-digit formatting (e.g., 001, 010, 120)
                                grdecl_file.write(
                                    f"{var_name_clean}_{timestep_num:03d}\n"
                                )

                                # Values (write 5 values per line for readability)
                                for i, value in enumerate(pred_data_full):
                                    grdecl_file.write(f"{value:.6e} ")
                                    if (i + 1) % 5 == 0:
                                        grdecl_file.write("\n")

                                # Ensure newline before '/'
                                if len(pred_data_full) % 5 != 0:
                                    grdecl_file.write("\n")

                                # Terminator
                                grdecl_file.write("/\n")

            except Exception as e:
                self.logger.error(f"Failed to process {hdf5_filename}: {e}")
                continue

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"GRDECL files saved to: {grdecl_output_dir}")
        self.logger.info("=" * 70)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main inference entry point.
    Performs autoregressive inference and generates GRDECL files.
    """

    dist, logger = InitializeLoggers(cfg)

    runner = InferenceRunner(cfg, dist, logger)

    runner.run_inference()

    runner.run_post()

    logger.success("Inference and post-processing completed successfully!")


if __name__ == "__main__":
    main()
