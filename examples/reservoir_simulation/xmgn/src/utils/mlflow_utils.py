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
Utility functions for MLflow logging in XMeshGraphNet.
"""

import os
import subprocess
from datetime import datetime
from omegaconf import DictConfig


def get_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()[:8]  # Return first 8 characters
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def flatten_config_section(config_section, prefix: str = "") -> dict:
    """
    Flatten a specific configuration section for MLflow parameters.

    Parameters:
    -----------
    config_section : DictConfig or dict
        Configuration section to flatten
    prefix : str
        Prefix for parameter names

    Returns:
    --------
    dict: Flattened configuration section
    """
    flattened = {}

    if isinstance(config_section, (dict, DictConfig)):
        for key, value in config_section.items():
            param_name = f"{prefix}_{key}" if prefix else key

            try:
                if isinstance(value, (dict, DictConfig)):
                    # Recursively flatten nested dictionaries
                    nested = flatten_config_section(value, param_name)
                    flattened.update(nested)
                elif isinstance(value, list):
                    # Handle lists by joining with commas
                    if all(isinstance(item, (str, int, float, bool)) for item in value):
                        flattened[param_name] = ",".join([str(item) for item in value])
                    else:
                        flattened[param_name] = str(value)
                elif isinstance(value, (str, int, float, bool)):
                    # Handle primitive types directly
                    flattened[param_name] = value
                else:
                    # Convert other types to string
                    flattened[param_name] = str(value)
            except Exception:
                # Handle Hydra interpolations and other complex values
                try:
                    flattened[param_name] = str(value)
                except Exception:
                    flattened[param_name] = f"<unresolved:{key}>"

    return flattened


def log_mlflow_tags_and_params(config: DictConfig, logger, mode: str = "training"):
    """
    Log MLflow tags and parameters in an organized way.

    Parameters:
    -----------
    config : DictConfig
        Hydra configuration object
    logger : PythonLogger
        Logger instance
    mode : str
        Mode identifier ("training", "inference", or "preprocessing")
    """
    import mlflow

    try:
        # Get job name and dataset info
        job_name = config.runspec.job_name if hasattr(config, "runspec") else "unknown"
        dataset_name = (
            os.path.basename(config.dataset.sim_dir)
            if hasattr(config, "dataset")
            else "unknown"
        )
        description = (
            getattr(config.runspec, "description", "")
            if hasattr(config, "runspec")
            else ""
        )

        # Set high-level tags for easy filtering
        tags = {
            "job_name": job_name,
            "dataset": dataset_name,
            "simulator": getattr(config.dataset, "simulator", "unknown"),
            "model_type": "MeshGraphNet",
            "git_commit": get_git_commit(),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Add description as a tag if available
        if description:
            tags["description"] = description

        mlflow.set_tags(tags)

        # Log configuration sections as parameters
        sections_to_log = []

        # Always log runspec if available
        if hasattr(config, "runspec"):
            sections_to_log.append(("runspec", config.runspec))

        if hasattr(config, "dataset"):
            sections_to_log.append(("dataset", config.dataset))

        if hasattr(config, "model"):
            sections_to_log.append(("model", config.model))

        if hasattr(config, "training") and mode in ["training"]:
            sections_to_log.append(("training", config.training))

        if hasattr(config, "preprocessing") and mode in ["preprocessing", "training"]:
            sections_to_log.append(("preprocessing", config.preprocessing))

        if hasattr(config, "graph"):
            sections_to_log.append(("graph", config.graph))

        if hasattr(config, "performance"):
            sections_to_log.append(("performance", config.performance))

        if hasattr(config, "inference") and mode in ["inference"]:
            sections_to_log.append(("inference", config.inference))

        # Log each section as parameters
        total_params = 0
        for section_name, section_config in sections_to_log:
            try:
                section_params = flatten_config_section(section_config, section_name)

                # Clean up file paths for better readability
                cleaned_params = {}
                for key, value in section_params.items():
                    if isinstance(value, str) and ("sim_dir" in key or "path" in key):
                        # Use basename for file paths
                        cleaned_params[key] = os.path.basename(str(value))
                    else:
                        cleaned_params[key] = value

                mlflow.log_params(cleaned_params)
                total_params += len(cleaned_params)

                if logger:
                    logger.info(
                        f"Logged {len(cleaned_params)} {section_name} parameters"
                    )

            except Exception as e:
                if logger:
                    logger.warning(f"Failed to log {section_name} parameters: {e}")

        if logger:
            logger.info(
                f"Successfully logged {total_params} total parameters and tags to MLflow"
            )

        # Log key parameters for verification
        key_params = {}
        if hasattr(config, "model"):
            key_params.update(
                {
                    "model_hidden_dim": getattr(config.model, "hidden_dim", "unknown"),
                    "model_layers": getattr(
                        config.model, "num_message_passing_layers", "unknown"
                    ),
                    "model_activation": getattr(config.model, "activation", "unknown"),
                }
            )

        if hasattr(config, "training") and mode == "training":
            key_params.update(
                {
                    "training_epochs": getattr(
                        config.training, "num_epochs", "unknown"
                    ),
                    "training_batch_size": getattr(
                        config.training, "batch_size", "unknown"
                    ),
                    "training_lr": getattr(config.training, "start_lr", "unknown"),
                }
            )

        if hasattr(config, "preprocessing"):
            key_params.update(
                {
                    "preprocessing_partitions": getattr(
                        config.preprocessing, "num_partitions", "unknown"
                    ),
                    "preprocessing_halo": getattr(
                        config.preprocessing, "halo_size", "unknown"
                    ),
                }
            )

        if key_params and logger:
            logger.info(f"Key parameters: {key_params}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to log MLflow tags and parameters: {e}")

        # Fallback: log basic info
        try:
            basic_params = {
                "config_file": "config.yaml",
                "dataset": dataset_name,
                "model_type": "MeshGraphNet",
                "mode": mode,
                "job_name": job_name,
            }
            mlflow.log_params(basic_params)
            mlflow.set_tags(
                {
                    "job_name": job_name,
                    "mode": mode,
                    "git_commit": get_git_commit(),
                }
            )
            if logger:
                logger.info(f"Logged basic parameters as fallback")
        except Exception as fallback_error:
            if logger:
                logger.error(f"Failed to log even basic parameters: {fallback_error}")


# Legacy function for backward compatibility
def log_config_parameters(config: DictConfig, logger, mode: str = "training"):
    """
    Legacy function - redirects to the new organized logging function.
    """
    log_mlflow_tags_and_params(config, logger, mode)
