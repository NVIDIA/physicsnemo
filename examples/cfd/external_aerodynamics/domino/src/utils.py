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

import os

from dataclasses import dataclass
from typing import Dict, Optional, Any
import numpy as np
import torch
import torch.distributed as dist
import pickle
from pathlib import Path
from typing import Literal
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager

from torch.distributed.tensor.placement_types import (
    Shard,
    Replicate,
)


def get_num_vars(cfg: dict, model_type: Literal["volume", "surface", "combined"]):
    """Calculate the number of variables for volume, surface, and global features.

    This function analyzes the configuration to determine how many variables are needed
    for different mesh data types based on the model type. Vector variables contribute
    3 components (x, y, z) while scalar variables contribute 1 component each.

    Args:
        cfg: Configuration object containing variable definitions for volume, surface,
             and global parameters with their types (scalar/vector).
        model_type (str): Type of model - can be "volume", "surface", or "combined".
                         Determines which variable types are included in the count.

    Returns:
        tuple: A 3-tuple containing:
            - num_vol_vars (int or None): Number of volume variables. None if model_type
              is not "volume" or "combined".
            - num_surf_vars (int or None): Number of surface variables. None if model_type
              is not "surface" or "combined".
            - num_global_features (int): Number of global parameter features.
    """
    num_vol_vars = 0
    volume_variable_names = []
    if model_type == "volume" or model_type == "combined":
        volume_variable_names = list(cfg.variables.volume.solution.keys())
        for j in volume_variable_names:
            if cfg.variables.volume.solution[j] == "vector":
                num_vol_vars += 3
            else:
                num_vol_vars += 1
    else:
        num_vol_vars = None

    num_surf_vars = 0
    surface_variable_names = []
    if model_type == "surface" or model_type == "combined":
        surface_variable_names = list(cfg.variables.surface.solution.keys())
        num_surf_vars = 0
        for j in surface_variable_names:
            if cfg.variables.surface.solution[j] == "vector":
                num_surf_vars += 3
            else:
                num_surf_vars += 1
    else:
        num_surf_vars = None

    num_global_features = 0
    global_params_names = list(cfg.variables.global_parameters.keys())
    for param in global_params_names:
        if cfg.variables.global_parameters[param].type == "vector":
            num_global_features += len(cfg.variables.global_parameters[param].reference)
        elif cfg.variables.global_parameters[param].type == "scalar":
            num_global_features += 1
        else:
            raise ValueError(f"Unknown global parameter type")

    return num_vol_vars, num_surf_vars, num_global_features


def get_keys_to_read(
    cfg: dict,
    model_type: Literal["volume", "surface", "combined"],
    get_ground_truth: bool = True,
):
    """
    This function helps configure the keys to read from the dataset.

    And, if some global parameter values are provided in the config,
    they are also read here and passed to the dataset.

    """

    # Always read these keys:
    keys_to_read = ["stl_coordinates", "stl_centers", "stl_faces", "stl_areas"]

    # If these keys are in the config, use them, else provide defaults in
    # case they aren't in the dataset:
    cfg_params_vec = []
    for key in cfg.variables.global_parameters:
        if cfg.variables.global_parameters[key].type == "vector":
            cfg_params_vec.extend(cfg.variables.global_parameters[key].reference)
        else:
            cfg_params_vec.append(cfg.variables.global_parameters[key].reference)
    keys_to_read_if_available = {
        "global_params_values": torch.tensor(cfg_params_vec).reshape(-1, 1),
        "global_params_reference": torch.tensor(cfg_params_vec).reshape(-1, 1),
    }

    # Volume keys:
    volume_keys = [
        "volume_mesh_centers",
    ]
    if get_ground_truth:
        volume_keys.append("volume_fields")

    # Surface keys:
    surface_keys = [
        "surface_mesh_centers",
        "surface_normals",
        "surface_areas",
    ]
    if get_ground_truth:
        surface_keys.append("surface_fields")

    if model_type == "volume" or model_type == "combined":
        keys_to_read.extend(volume_keys)
    if model_type == "surface" or model_type == "combined":
        keys_to_read.extend(surface_keys)

    return keys_to_read, keys_to_read_if_available


def coordinate_distributed_environment(cfg: DictConfig):
    """
    Initialize the distributed env for DoMINO.  This is actually always a 2D Mesh:
    one dimension is the data-parallel dimension (DDP), and the other is the
    domain dimension.

    For the training scripts, we need to know the rank, size of each dimension,
    and return the domain_mesh and placements for the loader.

    Args:
        cfg: Configuration object containing the domain parallelism configuration.

    Returns:
        domain_mesh: torch.distributed.DeviceMesh: The domain mesh for the domain parallel dimension.
        data_mesh: torch.distributed.DeviceMesh: The data mesh for the data parallel dimension.
        placements: dict[str, torch.distributed.tensor.Placement]: The placements for the data set
    """

    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    dist = DistributedManager()

    # Default to no domain parallelism:
    domain_size = cfg.get("domain_parallelism", {}).get("domain_size", 1)

    if dist.world_size == 1:
        domain_mesh = None
        data_mesh = None
        placements = None
    else:
        # Initialize the device mesh:
        mesh = dist.initialize_mesh(
            mesh_shape=(-1, domain_size), mesh_dim_names=("ddp", "domain")
        )
        domain_mesh = mesh["domain"]
        data_mesh = mesh["ddp"]

        if domain_size > 1:
            # Define the default placements for each tensor that might show up in
            # the data.  Note that we'll define placements for all keys, even if
            # they aren't actually used.

            # Note that placements are defined for pre-batched data, no batch index!

            shard_grid = cfg.get("domain_parallelism", {}).get("shard_grid", False)
            shard_points = cfg.get("domain_parallelism", {}).get("shard_points", False)

            if not shard_grid and not shard_points:
                raise ValueError(
                    "Either shard_grid or shard_points must be True if domain_size > 1"
                )

            # Not supported with physics loss:
            if cfg.train.add_physics_loss:
                raise ValueError(
                    "Domain parallelism is not supported with physics loss"
                )

            if shard_grid:
                grid_like_placement = [
                    Shard(0),
                ]
            else:
                grid_like_placement = [
                    Replicate(),
                ]

            if shard_points:
                point_like_placement = [
                    Shard(0),
                ]
            else:
                point_like_placement = [
                    Replicate(),
                ]

            placements = {
                "stl_coordinates": point_like_placement,
                "stl_centers": point_like_placement,
                "stl_faces": point_like_placement,
                "stl_areas": point_like_placement,
                "surface_fields": point_like_placement,
                "volume_mesh_centers": point_like_placement,
                "volume_fields": point_like_placement,
                "surface_mesh_centers": point_like_placement,
                "surface_normals": point_like_placement,
                "surface_areas": point_like_placement,
            }
        else:
            domain_mesh = None
            placements = None

    return domain_mesh, data_mesh, placements


@dataclass
class ScalingFactors:
    """
    Data structure for storing scaling factors computed for DoMINO datasets.

    This class provides a clean, easily serializable format for storing
    mean, std, min, and max values for different array keys in the dataset.
    Uses numpy arrays for easy serialization and cross-platform compatibility.

    Attributes:
        mean: Dictionary mapping keys to mean numpy arrays
        std: Dictionary mapping keys to standard deviation numpy arrays
        min_val: Dictionary mapping keys to minimum value numpy arrays
        max_val: Dictionary mapping keys to maximum value numpy arrays
        field_keys: List of field keys for which statistics were computed
    """

    mean: Dict[str, np.ndarray]
    std: Dict[str, np.ndarray]
    min_val: Dict[str, np.ndarray]
    max_val: Dict[str, np.ndarray]
    field_keys: list[str]

    def to_torch(
        self, device: Optional[torch.device] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Convert numpy arrays to torch tensors for use in training/inference."""
        device = device or torch.device("cpu")

        return {
            "mean": {k: torch.from_numpy(v).to(device) for k, v in self.mean.items()},
            "std": {k: torch.from_numpy(v).to(device) for k, v in self.std.items()},
            "min_val": {
                k: torch.from_numpy(v).to(device) for k, v in self.min_val.items()
            },
            "max_val": {
                k: torch.from_numpy(v).to(device) for k, v in self.max_val.items()
            },
        }

    def save(self, filepath: str | Path) -> None:
        """Save scaling factors to pickle file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str | Path) -> "ScalingFactors":
        """Load scaling factors from pickle file."""
        with open(filepath, "rb") as f:
            factors = pickle.load(f)
        return factors

    def get_field_shapes(self) -> Dict[str, tuple]:
        """Get the shape of each field's statistics."""
        return {key: self.mean[key].shape for key in self.field_keys}

    def summary(self) -> str:
        """Generate a human-readable summary of the scaling factors."""
        summary = ["Scaling Factors Summary:"]
        summary.append(f"Field Keys: {self.field_keys}")

        for key in self.field_keys:
            mean_val = self.mean[key]
            std_val = self.std[key]
            min_val = self.min_val[key]
            max_val = self.max_val[key]

            summary.append(f"\n{key}:")
            summary.append(f"  Shape: {mean_val.shape}")
            summary.append(f"  Mean: {mean_val}")
            summary.append(f"  Std: {std_val}")
            summary.append(f"  Min: {min_val}")
            summary.append(f"  Max: {max_val}")

        return "\n".join(summary)


def load_scaling_factors(
    cfg: DictConfig, logger=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load scaling factors from the configuration."""
    pickle_path = os.path.join(cfg.data.scaling_factors)

    try:
        scaling_factors = ScalingFactors.load(pickle_path)
        if logger is not None:
            logger.info(f"Scaling factors loaded from: {pickle_path}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Scaling factors not found at: {pickle_path}; please run compute_statistics.py to compute them."
        )

    if cfg.model.normalization == "min_max_scaling":
        vol_factors = np.asarray(
            [
                scaling_factors.max_val["volume_fields"],
                scaling_factors.min_val["volume_fields"],
            ]
        )
        surf_factors = np.asarray(
            [
                scaling_factors.max_val["surface_fields"],
                scaling_factors.min_val["surface_fields"],
            ]
        )
    elif cfg.model.normalization == "mean_std_scaling":
        vol_factors = np.asarray(
            [
                scaling_factors.mean["volume_fields"],
                scaling_factors.std["volume_fields"],
            ]
        )
        surf_factors = np.asarray(
            [
                scaling_factors.mean["surface_fields"],
                scaling_factors.std["surface_fields"],
            ]
        )
    else:
        raise ValueError(f"Invalid normalization mode: {cfg.model.normalization}")

    vol_factors_tensor = torch.from_numpy(vol_factors)
    surf_factors_tensor = torch.from_numpy(surf_factors)

    dm = DistributedManager()
    vol_factors_tensor = vol_factors_tensor.to(dm.device, dtype=torch.float32)
    surf_factors_tensor = surf_factors_tensor.to(dm.device, dtype=torch.float32)

    return vol_factors_tensor, surf_factors_tensor


def compute_l2(
    pred_surface: torch.Tensor | None,
    pred_volume: torch.Tensor | None,
    batch,
    dataloader,
) -> dict[str, torch.Tensor]:
    """
    Compute the L2 norm between prediction and target.

    Requires the dataloader to unscale back to original values
    """

    l2_dict = {}

    if pred_surface is not None:
        _, target_surface = dataloader.unscale_model_outputs(
            surface_fields=batch["surface_fields"]
        )
        _, pred_surface = dataloader.unscale_model_outputs(surface_fields=pred_surface)
        l2_surface = metrics_fn_surface(pred_surface, target_surface)
        l2_dict.update(l2_surface)
    if pred_volume is not None:
        target_volume, _ = dataloader.unscale_model_outputs(
            volume_fields=batch["volume_fields"]
        )
        pred_volume, _ = dataloader.unscale_model_outputs(volume_fields=pred_volume)
        l2_volume = metrics_fn_volume(pred_volume, target_volume)
        l2_dict.update(l2_volume)

    return l2_dict


def metrics_fn_surface(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Computes L2 surface metrics between prediction and target.

    Args:
        pred: Predicted values (normalized).
        target: Target values (normalized).

    Returns:
        Dictionary of L2 surface metrics for pressure and shear components.
    """

    l2_num = (pred - target) ** 2
    l2_num = torch.sum(l2_num, dim=1)
    l2_num = torch.sqrt(l2_num)

    l2_denom = target**2
    l2_denom = torch.sum(l2_denom, dim=1)
    l2_denom = torch.sqrt(l2_denom)

    l2 = l2_num / l2_denom

    metrics = {
        "l2_surf_pressure": torch.mean(l2[:, 0]),
        "l2_shear_x": torch.mean(l2[:, 1]),
        "l2_shear_y": torch.mean(l2[:, 2]),
        "l2_shear_z": torch.mean(l2[:, 3]),
    }

    return metrics


def metrics_fn_volume(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Computes L2 volume metrics between prediction and target.
    """
    l2_num = (pred - target) ** 2
    l2_num = torch.sum(l2_num, dim=1)
    l2_num = torch.sqrt(l2_num)

    l2_denom = target**2
    l2_denom = torch.sum(l2_denom, dim=1)
    l2_denom = torch.sqrt(l2_denom)

    l2 = l2_num / l2_denom

    metrics = {
        "l2_vol_pressure": torch.mean(l2[:, 3]),
        "l2_velocity_x": torch.mean(l2[:, 0]),
        "l2_velocity_y": torch.mean(l2[:, 1]),
        "l2_velocity_z": torch.mean(l2[:, 2]),
        "l2_nut": torch.mean(l2[:, 4]),
    }

    return metrics


def all_reduce_dict(
    metrics: dict[str, torch.Tensor], dm: DistributedManager
) -> dict[str, torch.Tensor]:
    """
    Reduces a dictionary of metrics across all distributed processes.

    Args:
        metrics: Dictionary of metric names to torch.Tensor values.
        dm: DistributedManager instance for distributed context.

    Returns:
        Dictionary of reduced metrics.
    """
    # TODO - update this to use domains and not the full world

    if dm.world_size == 1:
        return metrics

    for key, value in metrics.items():
        dist.all_reduce(value)
        value = value / dm.world_size
        metrics[key] = value

    return metrics
