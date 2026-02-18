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
Utility functions and classes for XMeshGraphNet training and inference.
"""

import os
import logging
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def fix_layernorm_compatibility():
    """
    Fix LayerNorm compatibility issue for PyTorch.

    This addresses a compatibility issue where LayerNorm may not have
    the register_load_state_dict_pre_hook method in some PyTorch versions.
    Should be called early in script initialization.
    """
    import torch.nn as nn

    if not hasattr(nn.LayerNorm, "register_load_state_dict_pre_hook"):

        def _register_load_state_dict_pre_hook(self, hook):
            """Dummy implementation for compatibility."""
            return None

        nn.LayerNorm.register_load_state_dict_pre_hook = (
            _register_load_state_dict_pre_hook
        )


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    Counts actual epochs, not validation checks.
    """

    def __init__(self, patience=20, min_delta=1e-6):
        """
        Initialize early stopping.

        Parameters
        ----------
        patience : int
            Number of epochs to wait for improvement
        min_delta : float
            Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.epochs_since_improvement = 0
        self.early_stop = False

    def step(self):
        """Increment epoch counter. Call this every epoch."""
        self.epochs_since_improvement += 1

    def check_improvement(self, current_score):
        """
        Check if validation score has improved.

        Parameters
        ----------
        current_score : float
            Current validation loss

        Returns
        -------
        bool
            True if there was improvement, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            self.epochs_since_improvement = 0
            return True

        # Always use "min" mode (lower is better for loss)
        improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.epochs_since_improvement = 0
            return True

        return False

    def should_stop(self):
        """
        Check if training should be stopped.

        Returns
        -------
        bool
            True if training should stop, False otherwise
        """
        if self.epochs_since_improvement >= self.patience:
            self.early_stop = True

        return self.early_stop


def get_dataset_dir(cfg: DictConfig) -> str:
    """
    Get the job-specific dataset directory path.

    Parameters
    -----------
    cfg : DictConfig
        Hydra configuration object

    Returns
    --------
    str: Path to the job-specific dataset directory
    """
    # Get job name from runspec (required)
    if not hasattr(cfg, "runspec") or not hasattr(cfg.runspec, "job_name"):
        raise ValueError("runspec.job_name is required in configuration")

    job_name = cfg.runspec.job_name

    # Get simulation directory from dataset (required)
    if not hasattr(cfg, "dataset") or not hasattr(cfg.dataset, "sim_dir"):
        raise ValueError("dataset.sim_dir is required in configuration")

    # Create base dataset directory path
    base_dataset_dir = to_absolute_path(cfg.dataset.sim_dir + ".dataset")

    # Return job-specific dataset directory
    return os.path.join(base_dataset_dir, job_name)


def get_dataset_paths(cfg: DictConfig) -> dict:
    """
    Get all dataset-related paths for a given configuration.

    Parameters
    -----------
    cfg : DictConfig
        Hydra configuration object

    Returns
    --------
    dict: Dictionary containing all dataset paths
    """
    dataset_dir = get_dataset_dir(cfg)

    return {
        "dataset_dir": dataset_dir,
        "graphs_dir": os.path.join(dataset_dir, "graphs"),
        "partitions_dir": os.path.join(dataset_dir, "partitions"),
        "stats_file": os.path.join(dataset_dir, "global_stats.json"),
        "train_partitions_path": os.path.join(dataset_dir, "partitions", "train"),
        "val_partitions_path": os.path.join(dataset_dir, "partitions", "val"),
        "test_partitions_path": os.path.join(dataset_dir, "partitions", "test"),
    }


def print_dataset_info(cfg: DictConfig) -> None:
    """
    Print dataset directory information for debugging.

    Parameters
    -----------
    cfg : DictConfig
        Hydra configuration object
    """
    job_name = cfg.runspec.job_name
    dataset_dir = get_dataset_dir(cfg)

    logger.info(f"Job name: {job_name}")
    logger.info(f"Dataset directory: {dataset_dir}")
