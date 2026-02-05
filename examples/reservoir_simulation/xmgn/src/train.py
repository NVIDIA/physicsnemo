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
Training pipeline for XMeshGraphNet on reservoir simulation data.
Loads partitioned graphs, normalizes features using precomputed statistics,
trains the model with early stopping, and saves checkpoints using PhysicsNeMo utilities.
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
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import numpy as np
import hydra
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.logging.mlflow import initialize_mlflow
from physicsnemo.utils.logging import LaunchLogger
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.meshgraphnet import MeshGraphNet

from utils import get_dataset_paths, fix_layernorm_compatibility, EarlyStopping
from data.dataloader import load_stats, find_pt_files, GraphDataset, custom_collate_fn

# Fix LayerNorm compatibility issue
fix_layernorm_compatibility()


def InitializeLoggers(cfg: DictConfig):
    """Initialize distributed manager and loggers following PhysicsNeMo pattern.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object

    Returns
    -------
    tuple
        (DistributedManager, PythonLogger)
    """
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()
    logger = PythonLogger(name="xmgn_reservoir")

    logger.info("XMeshGraphNet - Training for Reservoir Simulation")

    # Initialize MLflow (only on rank 0, following PhysicsNeMo pattern)
    if dist.rank == 0:
        # Clean up only .trash directory to avoid "deleted experiment" conflicts
        # while preserving historical results
        import shutil

        for mlflow_dir in ["mlruns", ".mlflow"]:
            trash_dir = os.path.join(mlflow_dir, ".trash")
            if os.path.exists(trash_dir):
                shutil.rmtree(trash_dir)
                logger.info(
                    f"Cleaned {trash_dir} directory to avoid deleted experiment conflicts"
                )

        # Get system username from environment variables
        user_name = (
            os.getenv("USER")
            or os.getenv("USERNAME")
            or os.getenv("LOGNAME")
            or "unknown"
        )

        # Initialize PhysicsNeMo's MLflow integration
        initialize_mlflow(
            experiment_name=cfg.runspec.job_name,
            experiment_desc=cfg.runspec.description,
            run_name=f"{cfg.runspec.job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            run_desc=f"Training: {cfg.runspec.description}",
            user_name=user_name,
            mode="offline",
        )
        LaunchLogger.initialize(use_mlflow=True)

    return dist, RankZeroLoggingWrapper(logger, dist)


class Trainer:
    """
    Unified trainer class that handles both partitioned and raw graphs.
    Eliminates code duplication between training and validation.
    """

    def __init__(self, cfg, dist, logger):
        """
        Initialize trainer with complete setup.

        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration object
        dist : DistributedManager
            Distributed manager instance
        logger : PythonLogger
            Logger instance
        """
        self.dist = dist
        self.device = self.dist.device
        self.logger = logger
        self.cfg = cfg

        # Enable cuDNN auto-tuner (only for GPU)
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.backends.cudnn.benchmark = cfg.performance.enable_cudnn_benchmark

        # Auto-generate checkpoint filename with best practices
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.basename(cfg.dataset.sim_dir)
        self.checkpoint_filename = f"checkpoint_{dataset_name}_{timestamp}.pth"

        # Set up dataset paths with job name
        paths = get_dataset_paths(cfg)
        self.dataset_dir = paths["dataset_dir"]
        self.stats_file = paths["stats_file"]
        self.train_partitions_path = paths["train_partitions_path"]
        self.val_partitions_path = paths["val_partitions_path"]

        # Load statistics (automatically generated in dataset directory)
        self.stats = load_stats(self.stats_file)

        # Initialize components
        self._initialize_dataloaders(cfg)
        self._initialize_model(cfg)
        self._initialize_optimizer(cfg)
        self._initialize_training_config(cfg)
        self._initialize_early_stopping(cfg)
        self._initialize_checkpoints(cfg)

    def _initialize_dataloaders(self, cfg):
        """Initialize training and validation dataloaders."""
        # Create unified data loaders (automatically handles partitions vs raw graphs)
        self.train_dataloader = self._create_dataloader(cfg, is_validation=False)

        # Create validation dataloader on all ranks for proper DDP validation
        self.val_dataloader = self._create_dataloader(cfg, is_validation=True)

        # Log dataset information
        self.logger.info(
            f"Dataset: {len(self.train_dataloader)} training samples, {len(self.val_dataloader)} validation samples"
        )

    def _initialize_model(self, cfg):
        """Initialize the MeshGraphNet model."""
        # Get dimensions from stats and data
        input_dim_nodes = len(self.stats["node_features"]["mean"])
        input_dim_edges = len(self.stats["edge_features"]["mean"])
        output_dim = len(cfg.dataset.graph.target_vars.node_features)

        # Create model
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
            num_processor_checkpoint_segments=cfg.performance.checkpoint_segments,
        ).to(self.device)

        # Wrap model for multi-GPU training if available
        if self.dist.world_size > 1:
            # Use DistributedDataParallel for multi-node/multi-GPU training
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
                gradient_as_bucket_view=True,
                static_graph=True,
            )

    def _initialize_optimizer(self, cfg):
        """Initialize optimizer, scheduler, and gradient scaler."""

        weight_decay = getattr(cfg.training, "weight_decay", 1.0e-3)

        # Create optimizer (AdamW with decoupled weight decay)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.start_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.99),
            eps=1e-8,
        )

        # Create cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.training.num_epochs, eta_min=cfg.training.end_lr
        )

        # Create gradient scaler for mixed precision (GPU only)
        self.scaler = GradScaler() if self.device.type == "cuda" else None

        self.logger.info(
            f"Optimizer: AdamW (lr={cfg.training.start_lr}, weight_decay={weight_decay})"
        )

    def _initialize_training_config(self, cfg):
        """Initialize training configuration and loss functions."""
        # Store training config
        self.num_epochs = cfg.training.num_epochs
        self.validation_freq = cfg.training.validation_freq

        # Load target variable weights
        self.target_weights = torch.tensor(
            cfg.dataset.graph.target_vars.weights, device=self.device
        )
        self.logger.info(
            f"Target variables: {cfg.dataset.graph.target_vars.node_features}"
        )
        self.logger.info(f"Target variable weights: {self.target_weights.tolist()}")

        # Initialize loss functions (handles defaults and validation)
        self._initialize_loss_functions(cfg)

    def _initialize_early_stopping(self, cfg):
        """Initialize early stopping if configured."""
        if hasattr(cfg.training, "early_stopping") and hasattr(
            cfg.training.early_stopping, "patience"
        ):
            self.early_stopping = EarlyStopping(
                patience=cfg.training.early_stopping.patience,
                min_delta=cfg.training.early_stopping.min_delta,
            )
            self.logger.info(
                f"Early stopping enabled: patience={cfg.training.early_stopping.patience}, "
                f"min_delta={cfg.training.early_stopping.min_delta}"
            )
        else:
            self.early_stopping = None
            self.logger.info("Early stopping disabled")

    def _initialize_checkpoints(self, cfg):
        """Initialize checkpoint directories and arguments."""
        # Set up checkpoint arguments (following PhysicsNeMo pattern)
        # Use current working directory (Hydra changes to output directory)
        base_output_dir = os.getcwd()

        checkpoint_dir = os.path.join(base_output_dir, "checkpoints")
        best_checkpoint_dir = os.path.join(base_output_dir, "best_checkpoints")

        # Create checkpoint directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(best_checkpoint_dir, exist_ok=True)

        # Log checkpoint paths
        self.logger.info(f"Checkpoint directory: {checkpoint_dir}")
        self.logger.info(f"Best checkpoint directory: {best_checkpoint_dir}")

        self.ckpt_args = {
            "path": checkpoint_dir,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "models": self.model,
        }
        self.bst_ckpt_args = {
            "path": best_checkpoint_dir,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "models": self.model,
        }

    def _create_dataloader(self, cfg, is_validation=False):
        """Create dataloader using instance variables with DistributedSampler support."""
        if is_validation:
            partitions_path = self.val_partitions_path
        else:
            partitions_path = self.train_partitions_path

        # Load global statistics
        with open(self.stats_file, "r") as f:
            stats = json.load(f)

        # Load per-feature statistics
        node_mean = torch.tensor(stats["node_features"]["mean"])
        node_std = torch.tensor(stats["node_features"]["std"])
        edge_mean = torch.tensor(stats["edge_features"]["mean"])
        edge_std = torch.tensor(stats["edge_features"]["std"])

        # Load target feature statistics (if available)
        target_mean = None
        target_std = None
        if "target_features" in stats:
            target_mean = torch.tensor(stats["target_features"]["mean"])
            target_std = torch.tensor(stats["target_features"]["std"])

        # Find partition files
        file_paths = find_pt_files(partitions_path)

        # Create dataset
        dataset = GraphDataset(
            file_paths,
            node_mean,
            node_std,
            edge_mean,
            edge_std,
            target_mean,
            target_std,
        )

        # Create DistributedSampler for proper distributed training
        if self.dist.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.dist.world_size,
                rank=self.dist.rank,
                shuffle=not is_validation,
                drop_last=False,
            )
            shuffle = False  # DistributedSampler handles shuffling
        else:
            sampler = None
            shuffle = not is_validation

        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.training.get("batch_size", 1),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=custom_collate_fn,  # Use custom collate function for lists of PartitionedGraph objects
        )

        # Store sampler for set_epoch calls
        if not is_validation:
            self.train_sampler = sampler
        else:
            self.val_sampler = sampler

        self.logger.info(
            f"Using partitioned data loader with {len(dataloader)} batches"
        )
        return dataloader

    def denormalize_predictions(self, predictions):
        """Denormalize predictions using global statistics."""
        if "target_features" not in self.stats:
            self.logger.warning(
                "No target feature statistics found for denormalization"
            )
            return predictions

        target_mean = torch.tensor(
            self.stats["target_features"]["mean"], device=predictions.device
        )
        target_std = torch.tensor(
            self.stats["target_features"]["std"], device=predictions.device
        )

        # Denormalize: pred_denorm = pred_norm * std + mean
        denormalized = predictions * target_std + target_mean
        return denormalized

    def denormalize_targets(self, targets):
        """Denormalize targets using global statistics."""
        if "target_features" not in self.stats:
            self.logger.warning(
                "No target feature statistics found for denormalization"
            )
            return targets

        target_mean = torch.tensor(
            self.stats["target_features"]["mean"], device=targets.device
        )
        target_std = torch.tensor(
            self.stats["target_features"]["std"], device=targets.device
        )

        # Denormalize: target_denorm = target_norm * std + mean
        denormalized = targets * target_std + target_mean
        return denormalized

    def _initialize_loss_functions(self, cfg):
        """Initialize PyTorch loss functions for each target variable with defaults and validation."""
        # Load loss function configuration with defaults
        self.loss_functions = getattr(
            cfg.dataset.graph.target_vars, "loss_functions", None
        )
        self.huber_delta = getattr(cfg.dataset.graph.target_vars, "huber_delta", None)

        # Set defaults if not provided
        if self.loss_functions is None:
            self.loss_functions = ["L2"] * len(
                cfg.dataset.graph.target_vars.node_features
            )
            self.logger.warning(
                f"Loss functions not specified in config. Using default: {self.loss_functions}"
            )
        elif len(self.loss_functions) != len(
            cfg.dataset.graph.target_vars.node_features
        ):
            self.logger.warning(
                f"Number of loss functions ({len(self.loss_functions)}) doesn't match number of target variables ({len(cfg.dataset.graph.target_vars.node_features)}). Using L2 for all."
            )
            self.loss_functions = ["L2"] * len(
                cfg.dataset.graph.target_vars.node_features
            )

        # Validate loss function names and set defaults for invalid ones (case-insensitive)
        valid_losses = ["L1", "L2", "Huber"]
        for i, loss_func in enumerate(self.loss_functions):
            # Convert to proper case for consistency
            loss_func_upper = loss_func.upper()
            if loss_func_upper == "L1":
                self.loss_functions[i] = "L1"
            elif loss_func_upper == "L2":
                self.loss_functions[i] = "L2"
            elif loss_func_upper == "HUBER":
                self.loss_functions[i] = "Huber"
            else:
                self.logger.warning(
                    f"Invalid loss function '{loss_func}' for variable {i}. Using L2 instead."
                )
                self.loss_functions[i] = "L2"

        self.logger.info(f"Loss functions: {self.loss_functions}")
        if "Huber" in self.loss_functions:
            if self.huber_delta is None:  # Set Huber delta default
                self.huber_delta = 0.5
                self.logger.info(
                    f"Huber delta not specified. Using default value: {self.huber_delta}"
                )
            self.logger.info(f"Huber delta: {self.huber_delta}")

        # Initialize PyTorch loss functions
        self.loss_fn_objects = []

        for loss_func in self.loss_functions:
            if loss_func == "L1":
                self.loss_fn_objects.append(torch.nn.L1Loss())
            elif loss_func == "L2":
                self.loss_fn_objects.append(torch.nn.MSELoss())
            elif loss_func == "Huber":
                self.loss_fn_objects.append(torch.nn.HuberLoss(delta=self.huber_delta))
            else:
                raise ValueError(f"Unknown loss function: {loss_func}")

        # Move loss functions to device
        for loss_fn in self.loss_fn_objects:
            loss_fn.to(self.device)

    def compute_weighted_loss(self, predictions, targets):
        """
        Compute weighted loss for each target variable using configurable loss functions.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions [N, num_target_vars]
        targets : torch.Tensor
            Target values [N, num_target_vars]

        Returns
        -------
        torch.Tensor
            Weighted loss
        """
        losses_per_var = []

        for i, loss_fn in enumerate(self.loss_fn_objects):
            pred_var = predictions[:, i]
            target_var = targets[:, i]

            # Use the initialized PyTorch loss function
            loss = loss_fn(pred_var, target_var)
            losses_per_var.append(loss)

        # Convert to tensor and apply weights
        losses_tensor = torch.stack(losses_per_var)
        weighted_loss = torch.sum(self.target_weights * losses_tensor)

        return weighted_loss

    def compute_per_variable_losses(self, predictions, targets):
        """
        Compute per-variable losses for logging purposes.

        Parameters
        ----------
        predictions : torch.Tensor or np.ndarray
            Model predictions [N, num_target_vars]
        targets : torch.Tensor or np.ndarray
            Target values [N, num_target_vars]

        Returns
        -------
        list
            List of per-variable losses
        """
        losses_per_var = []

        for i, loss_fn in enumerate(self.loss_fn_objects):
            pred_var = predictions[:, i]
            target_var = targets[:, i]

            # Convert to torch tensors if needed
            if not isinstance(pred_var, torch.Tensor):
                pred_var = torch.tensor(pred_var, device=self.device)
            if not isinstance(target_var, torch.Tensor):
                target_var = torch.tensor(target_var, device=self.device)

            # Use the initialized PyTorch loss function
            loss = loss_fn(pred_var, target_var)
            losses_per_var.append(loss.item())

        return losses_per_var

    def _process_partition(self, part, is_training=True):
        """
        Process a single partition (for both training and validation).

        Parameters
        ----------
        part : torch_geometric.data.Data
            The partition to process
        is_training : bool
            Whether this is training (affects gradient computation)

        Returns
        -------
        tuple
            (loss, denorm_loss, pred, target)
        """
        part = part.to(self.device)

        # Ensure data is in float32 for mixed precision training
        if hasattr(part, "x") and part.x is not None:
            part.x = part.x.float()
        if hasattr(part, "edge_attr") and part.edge_attr is not None:
            part.edge_attr = part.edge_attr.float()
        if hasattr(part, "y") and part.y is not None:
            part.y = part.y.float()

        # Forward pass (disable mixed precision for now to avoid dtype issues)
        pred = self.model(part.x, part.edge_attr, part)

        # Get inner nodes if available (for partitioned graphs)
        if hasattr(part, "inner_node"):
            pred_inner = pred[part.inner_node]
            target_inner = (
                part.y[part.inner_node]
                if hasattr(part, "y")
                else part.y[part.inner_node]
            )
        else:
            pred_inner = pred
            target_inner = part.y

        # Compute weighted normalized loss
        loss = self.compute_weighted_loss(pred_inner, target_inner)

        # Denormalize for evaluation
        pred_denorm = self.denormalize_predictions(pred_inner)
        target_denorm = self.denormalize_targets(target_inner)
        denorm_loss = self.compute_weighted_loss(pred_denorm, target_denorm)

        return loss, denorm_loss, pred_inner, target_inner

    def _process_graph(self, graph, is_training=True):
        """
        Process a single graph (for both training and validation).

        Parameters
        ----------
        graph : torch_geometric.data.Data
            The graph to process
        is_training : bool
            Whether this is training (affects gradient computation)

        Returns
        -------
        tuple
            (loss, denorm_loss, pred, target)
        """
        graph = graph.to(self.device)

        # Forward pass
        if is_training and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = self.model(graph.x, graph.edge_attr, graph)
        else:
            pred = self.model(graph.x, graph.edge_attr, graph)

        # Compute weighted normalized loss
        loss = self.compute_weighted_loss(pred, graph.y)

        # Denormalize for evaluation
        pred_denorm = self.denormalize_predictions(pred)
        target_denorm = self.denormalize_targets(graph.y)
        denorm_loss = self.compute_weighted_loss(pred_denorm, target_denorm)

        return loss, denorm_loss, pred, graph.y

    def train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_dataloader):
            # Handle the new format: batch is (partitions_list, labels)
            partitions_list, labels = batch

            self.optimizer.zero_grad()

            # Process each sample's partitions in the batch
            total_batch_loss = 0.0
            num_samples = len(partitions_list)

            for sample_idx, partitions in enumerate(partitions_list):
                # Process each partition in this sample
                sample_loss = 0.0
                num_partitions = len(partitions)

                for partition in partitions:
                    loss, _, _, _ = self._process_partition(partition, is_training=True)

                    # For logging: accumulate loss scaled only by num_partitions (consistent with validation)
                    sample_loss += loss.item() / num_partitions

                    # For gradient computation: scale by total number of forward passes in the batch
                    loss = loss / (num_partitions * num_samples)

                    # Backward pass
                    if self.device.type == "cuda":
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # Accumulate loss from this sample
                total_batch_loss += sample_loss

            # Update optimizer after processing all samples and partitions
            if self.device.type == "cuda":
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            total_loss += total_batch_loss
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Synchronize loss across all GPUs for accurate reporting
        if self.dist.world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        return avg_loss

    def validate_epoch(self):
        """Validate the model for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_denorm_loss = 0.0
        num_batches = 0

        # Collect all predictions and targets for per-variable metrics
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                # Handle the new format: batch is (partitions_list, labels)
                partitions_list, labels = batch

                # Process each sample's partitions in the batch
                batch_loss = 0.0
                batch_denorm_loss = 0.0
                num_samples = len(partitions_list)

                for sample_idx, partitions in enumerate(partitions_list):
                    # Process each partition in this sample
                    sample_loss = 0.0
                    sample_denorm_loss = 0.0
                    num_partitions = len(partitions)

                    for partition in partitions:
                        loss, denorm_loss, pred, target = self._process_partition(
                            partition, is_training=False
                        )
                        loss = loss / num_partitions
                        denorm_loss = denorm_loss / num_partitions
                        sample_loss += loss.item()
                        sample_denorm_loss += denorm_loss.item()

                        # Collect predictions and targets for per-variable metrics
                        all_predictions.append(pred.cpu().numpy())
                        all_targets.append(target.cpu().numpy())

                    batch_loss += sample_loss
                    batch_denorm_loss += sample_denorm_loss

                total_loss += batch_loss
                total_denorm_loss += batch_denorm_loss

                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_denorm_loss = total_denorm_loss / num_batches

        # Synchronize validation losses across all GPUs
        if self.dist.world_size > 1:
            loss_tensor = torch.tensor([avg_loss, avg_denorm_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss, avg_denorm_loss = loss_tensor[0].item(), loss_tensor[1].item()

        # Calculate per-variable metrics (simplified logging without tabulate)
        if all_predictions and all_targets:
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

        # Prepare metrics dictionary for MLflow logging
        metrics = {}

        # Calculate overall MAE and RMSE
        if len(all_predictions) > 0 and len(all_targets) > 0:
            overall_mae = np.mean(np.abs(all_predictions - all_targets))
            overall_mse = np.mean((all_predictions - all_targets) ** 2)
            overall_rmse = np.sqrt(overall_mse)

            metrics["mae"] = overall_mae
            metrics["mse"] = overall_mse
            metrics["rmse"] = overall_rmse

            # Calculate per-variable metrics (normalized)
            target_names = self.cfg.dataset.graph.target_vars.node_features
            for i, var_name in enumerate(target_names):
                var_mae = np.mean(np.abs(all_predictions[:, i] - all_targets[:, i]))
                var_rmse = np.sqrt(
                    np.mean((all_predictions[:, i] - all_targets[:, i]) ** 2)
                )

                metrics[f"mae_{var_name.lower()}"] = var_mae
                metrics[f"rmse_{var_name.lower()}"] = var_rmse

            # Calculate denormalized per-variable metrics if available
            if "target_features" in self.stats:
                target_mean = np.array(self.stats["target_features"]["mean"])
                target_std = np.array(self.stats["target_features"]["std"])

                # Denormalize predictions and targets
                all_predictions_denorm = all_predictions * target_std + target_mean
                all_targets_denorm = all_targets * target_std + target_mean

                # Overall denormalized metrics
                overall_mae_denorm = np.mean(
                    np.abs(all_predictions_denorm - all_targets_denorm)
                )
                overall_mse_denorm = np.mean(
                    (all_predictions_denorm - all_targets_denorm) ** 2
                )
                overall_rmse_denorm = np.sqrt(overall_mse_denorm)

                metrics["mae_denorm"] = overall_mae_denorm
                metrics["mse_denorm"] = overall_mse_denorm
                metrics["rmse_denorm"] = overall_rmse_denorm

                # Per-variable denormalized metrics
                for i, var_name in enumerate(target_names):
                    var_mae_denorm = np.mean(
                        np.abs(all_predictions_denorm[:, i] - all_targets_denorm[:, i])
                    )
                    var_rmse_denorm = np.sqrt(
                        np.mean(
                            (all_predictions_denorm[:, i] - all_targets_denorm[:, i])
                            ** 2
                        )
                    )

                    metrics[f"mae_{var_name.lower()}_denorm"] = var_mae_denorm
                    metrics[f"rmse_{var_name.lower()}_denorm"] = var_rmse_denorm

        # Synchronize all metrics across GPUs
        if self.dist.world_size > 1 and metrics:
            # Convert metrics dict to tensor for reduction
            metric_keys = sorted(metrics.keys())
            metric_values = torch.tensor(
                [metrics[k] for k in metric_keys], device=self.device
            )
            dist.all_reduce(metric_values, op=dist.ReduceOp.AVG)

            # Update metrics dict with synchronized values
            for i, key in enumerate(metric_keys):
                metrics[key] = metric_values[i].item()

        return avg_loss, avg_denorm_loss, metrics

    def train(self):
        """
        Complete training loop with validation and checkpointing.
        Handles resume logic internally based on configuration.

        Returns
        --------
        float: Best validation loss
        """
        # Handle training resume based on config
        loaded_epoch = self._handle_training_resume()

        # Initialize best validation loss
        best_val_loss = float("inf")

        # If resuming from a checkpoint, run validation to get current best validation loss
        if loaded_epoch > 0:
            self.logger.info(
                f"Resuming training from epoch {loaded_epoch + 1}. Running validation to get current best validation loss..."
            )
            val_loss, _, _ = self.validate_epoch()
            best_val_loss = val_loss
            self.logger.info(f"Current best validation loss: {best_val_loss:.6f}")

        for epoch in range(max(1, loaded_epoch + 1), self.num_epochs + 1):
            # Set epoch for proper distributed sampling
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            if self.val_sampler is not None:
                self.val_sampler.set_epoch(epoch)

            # Log progress
            self.logger.info(f"Starting Epoch {epoch}/{self.num_epochs}")

            # Increment early stopping epoch counter
            if self.early_stopping is not None:
                self.early_stopping.step()

            # Train with LaunchLogger (handles MLflow automatically)
            with LaunchLogger(
                name_space="train",
                num_mini_batch=len(self.train_dataloader),
                epoch=epoch,
                epoch_alert_freq=1,
            ) as log:
                train_loss = self.train_epoch()
                log.log_epoch(
                    {
                        "train_loss": train_loss,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "best_val_loss": best_val_loss
                        if best_val_loss != float("inf")
                        else 0.0,
                    }
                )

            # Validation step
            val_loss, val_denorm_loss, val_metrics = self._validation_step(epoch)

            # Save best model and check early stopping
            should_stop = self._check_early_stopping(
                epoch, val_loss, val_metrics, best_val_loss
            )

            if val_loss != float("inf") and val_loss < best_val_loss:
                best_val_loss = val_loss

            # Save regular checkpoint (only on rank 0)
            if self.dist.rank == 0 and (
                epoch % self.validation_freq == 0 or epoch == self.num_epochs
            ):
                save_checkpoint(**self.ckpt_args, epoch=epoch)

            # Update learning rate
            self.scheduler.step()

            # Log training progress (ZeroRankLogger handles rank 0 automatically)
            self.logger.info(
                f"Epoch {epoch}/{self.num_epochs}, Train loss: {train_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # Break if early stopping triggered
            if should_stop:
                self.logger.info(
                    f"Training stopped early at epoch {epoch} due to early stopping"
                )
                break

        self.logger.info(
            f"Training completed! Best validation loss: {best_val_loss:.6f}"
        )

    def _validation_step(self, epoch):
        """
        Perform validation step with comprehensive metrics logging.
        Validation runs on all ranks for proper DDP, but only rank 0 logs to MLflow.

        Parameters
        ----------
        epoch : int
            Current epoch number

        Returns
        -------
        tuple
            (val_loss, val_denorm_loss, val_metrics)
        """
        val_loss = float("inf")
        val_denorm_loss = float("inf")
        val_metrics = None

        # Run validation on all ranks at validation frequency
        if epoch % self.validation_freq == 0 or epoch == self.num_epochs:
            val_loss, val_denorm_loss, val_metrics = self.validate_epoch()

            # Only log to MLflow on rank 0
            if self.dist.rank == 0:
                with LaunchLogger("valid", epoch=epoch) as log:
                    # Prepare comprehensive metrics for logging
                    metrics_to_log = self._prepare_validation_metrics(
                        val_loss, val_denorm_loss, val_metrics
                    )
                    log.log_epoch(metrics_to_log)

        return val_loss, val_denorm_loss, val_metrics

    def _prepare_validation_metrics(self, val_loss, val_denorm_loss, val_metrics):
        """
        Prepare comprehensive validation metrics for logging.

        Parameters
        ----------
        val_loss : float
            Validation loss
        val_denorm_loss : float
            Denormalized validation loss
        val_metrics : dict
            Additional validation metrics

        Returns
        -------
        dict
            Metrics to log
        """
        metrics_to_log = {
            "val_loss": val_loss,
            "val_denorm_loss": val_denorm_loss,
        }

        if val_metrics:
            # Add overall MAE and MSE
            if "mae" in val_metrics:
                metrics_to_log["val_mae"] = val_metrics["mae"]
            if "mse" in val_metrics:
                metrics_to_log["val_mse"] = val_metrics["mse"]
            if "rmse" in val_metrics:
                metrics_to_log["val_rmse"] = val_metrics["rmse"]

            # Add per-variable metrics (normalized)
            target_names = self.cfg.dataset.graph.target_vars.node_features
            for i, var_name in enumerate(target_names):
                if f"mae_{var_name.lower()}" in val_metrics:
                    metrics_to_log[f"val_mae_{var_name.lower()}"] = val_metrics[
                        f"mae_{var_name.lower()}"
                    ]
                if f"rmse_{var_name.lower()}" in val_metrics:
                    metrics_to_log[f"val_rmse_{var_name.lower()}"] = val_metrics[
                        f"rmse_{var_name.lower()}"
                    ]

            # Add denormalized per-variable metrics if available
            for i, var_name in enumerate(target_names):
                if f"mae_{var_name.lower()}_denorm" in val_metrics:
                    metrics_to_log[f"val_mae_{var_name.lower()}_denorm"] = val_metrics[
                        f"mae_{var_name.lower()}_denorm"
                    ]
                if f"rmse_{var_name.lower()}_denorm" in val_metrics:
                    metrics_to_log[f"val_rmse_{var_name.lower()}_denorm"] = val_metrics[
                        f"rmse_{var_name.lower()}_denorm"
                    ]

        return metrics_to_log

    def _check_early_stopping(self, epoch, val_loss, val_metrics, best_val_loss):
        """
        Check early stopping and save best model.

        Parameters
        ----------
        epoch : int
            Current epoch number
        val_loss : float
            Validation loss
        val_metrics : dict
            Validation metrics
        best_val_loss : float
            Current best validation loss

        Returns
        -------
        bool
            True if early stopping should trigger, False otherwise
        """
        should_stop = False

        # Save best model (only if validation was performed and only on rank 0)
        if (
            self.dist.rank == 0
            and val_loss != float("inf")
            and val_loss < best_val_loss
        ):
            save_checkpoint(**self.bst_ckpt_args, epoch=epoch)

        # Check early stopping (only on rank 0 and if validation was performed)
        if (
            self.dist.rank == 0
            and self.early_stopping is not None
            and val_loss != float("inf")
        ):
            # Check if validation improved
            self.early_stopping.check_improvement(val_loss)

            # Check if we should stop based on epochs without improvement
            should_stop = self.early_stopping.should_stop()

            if should_stop:
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best val_loss: {self.early_stopping.best_score:.6f}, "
                    f"Current: {val_loss:.6f}, "
                    f"Epochs without improvement: {self.early_stopping.epochs_since_improvement}/{self.early_stopping.patience}"
                )

        # Broadcast early stopping decision to all ranks
        if self.dist.world_size > 1:
            should_stop_tensor = torch.tensor(int(should_stop), device=self.device)
            dist.broadcast(should_stop_tensor, src=0)
            should_stop = bool(should_stop_tensor.item())

        return should_stop

    def _handle_training_resume(self):
        """Handle training resume based on configuration."""
        import os
        import shutil

        checkpoint_dir = self.ckpt_args["path"]
        best_checkpoint_dir = self.bst_ckpt_args["path"]

        # Check if any checkpoint files exist
        has_checkpoints = False
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [
                f
                for f in os.listdir(checkpoint_dir)
                if f.endswith(".pt") or f.endswith(".mdlus")
            ]
            if checkpoint_files:
                has_checkpoints = True

        if os.path.exists(best_checkpoint_dir):
            best_checkpoint_files = [
                f
                for f in os.listdir(best_checkpoint_dir)
                if f.endswith(".pt") or f.endswith(".mdlus")
            ]
            if best_checkpoint_files:
                has_checkpoints = True

        if self.cfg.training.resume and has_checkpoints:
            self.logger.info("Resuming training from existing checkpoints...")
            # Load checkpoint and return the epoch
            return load_checkpoint(**self.ckpt_args, device=self.dist.device)
        elif self.cfg.training.resume and not has_checkpoints:
            self.logger.warning(
                "Resume enabled but no checkpoints found. Starting fresh training..."
            )
            return 0
        elif not self.cfg.training.resume and has_checkpoints:
            self.logger.info("Resume disabled: Deleting existing checkpoint files...")
            try:
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                if os.path.exists(best_checkpoint_dir):
                    shutil.rmtree(best_checkpoint_dir)
                    os.makedirs(best_checkpoint_dir, exist_ok=True)
                self.logger.success(
                    "Checkpoint files deleted. Starting fresh training..."
                )
            except (OSError, PermissionError) as e:
                self.logger.warning(f"Could not delete some checkpoint files: {e}")
                self.logger.info("Starting fresh training anyway...")
            return 0
        else:
            self.logger.info("Starting fresh training...")
            return 0


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training entry point.
    Trains XMeshGraphNet on reservoir simulation data.
    """

    dist, logger = InitializeLoggers(cfg)

    trainer = Trainer(cfg, dist, logger)

    trainer.train()

    logger.success("Training completed successfully!")


if __name__ == "__main__":
    main()
