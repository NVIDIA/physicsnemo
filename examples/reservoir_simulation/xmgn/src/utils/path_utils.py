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
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def get_dataset_dir(cfg: DictConfig) -> str:
    """
    Get the job-specific dataset directory path.

    Parameters:
    -----------
    cfg : DictConfig
        Hydra configuration object

    Returns:
    --------
    str: Path to the job-specific dataset directory
    """
    # Get job name from runspec (required)
    if not hasattr(cfg, "runspec") or not hasattr(cfg.runspec, "job_name"):
        raise ValueError("runspec.job_name is required in configuration")

    job_name = cfg.runspec.job_name

    # Create base dataset directory path
    base_dataset_dir = to_absolute_path(cfg.dataset.sim_dir + ".dataset")

    # Return job-specific dataset directory
    return os.path.join(base_dataset_dir, job_name)


def get_dataset_paths(cfg: DictConfig) -> dict:
    """
    Get all dataset-related paths for a given configuration.

    Parameters:
    -----------
    cfg : DictConfig
        Hydra configuration object

    Returns:
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

    Parameters:
    -----------
    cfg : DictConfig
        Hydra configuration object
    """
    job_name = cfg.runspec.job_name
    dataset_dir = get_dataset_dir(cfg)

    print(f"📁 Job name: {job_name}")
    print(f"📁 Dataset directory: {dataset_dir}")
