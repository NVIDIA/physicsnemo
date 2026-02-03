# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
Normalize - Standardize tensor values by mean and standard deviation or min-max scaling.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from tensordict import TensorDict

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.base import Transform


@register()
class Normalize(Transform):
    """
    Normalize specified fields using mean-std or min-max scaling.

    Supports two normalization methods:
    - ``mean_std``: Applies (x - mean) / std for each specified field
    - ``min_max``: Applies (x - center) / half_range, normalizing to [-1, 1]
      where center = (max + min) / 2 and half_range = (max - min) / 2

    Parameters can be provided directly or loaded from a ``.npz`` file.

    Examples
    --------
    Mean-std scaling:

    >>> import torch
    >>> from tensordict import TensorDict
    >>> sample = TensorDict({
    ...     "pressure": torch.tensor([101325.0, 102325.0, 100325.0]),
    ...     "velocity": torch.tensor([10.0, -10.0, 0.0]),
    ... })
    >>> norm = Normalize(
    ...     input_keys=["pressure", "velocity"],
    ...     method="mean_std",
    ...     means={"pressure": 101325.0, "velocity": 0.0},
    ...     stds={"pressure": 1000.0, "velocity": 10.0},
    ... )
    >>> normalized = norm(sample)
    >>> normalized["pressure"]
    tensor([ 0.,  1., -1.])
    >>> normalized["velocity"]
    tensor([ 1., -1.,  0.])

    Min-max scaling (normalizes to [-1, 1]):

    >>> sample = TensorDict({
    ...     "pressure": torch.tensor([100000.0, 105000.0, 110000.0]),
    ... })
    >>> norm = Normalize(
    ...     input_keys=["pressure"],
    ...     method="min_max",
    ...     mins={"pressure": 100000.0},
    ...     maxs={"pressure": 110000.0},
    ... )
    >>> normalized = norm(sample)
    >>> normalized["pressure"]
    tensor([-1.,  0.,  1.])
    """

    def __init__(
        self,
        input_keys: list[str],
        means: Optional[dict[str, float | torch.Tensor] | float | torch.Tensor] = None,
        stds: Optional[dict[str, float | torch.Tensor] | float | torch.Tensor] = None,
        *,
        method: Optional[Literal["mean_std", "min_max"]] = None,
        mins: Optional[dict[str, float | torch.Tensor] | float | torch.Tensor] = None,
        maxs: Optional[dict[str, float | torch.Tensor] | float | torch.Tensor] = None,
        stats_file: Optional[str | Path] = None,
        eps: float = 1e-8,
    ) -> None:
        """
        Initialize the normalizer.

        Parameters
        ----------
        input_keys : list[str]
            List of field names to normalize.
        means : dict[str, float | torch.Tensor] or float or torch.Tensor, optional
            Mean values for mean_std method. Either a dict mapping field names
            to values, or a single value applied to all fields. Deprecated if
            ``method`` is not specified.
        stds : dict[str, float | torch.Tensor] or float or torch.Tensor, optional
            Standard deviation values for mean_std method. Same format as means.
        method : {"mean_std", "min_max"}, optional
            Normalization method - either ``"mean_std"`` or ``"min_max"``.
        mins : dict[str, float | torch.Tensor] or float or torch.Tensor, optional
            Minimum values for min_max method. Same format as means.
        maxs : dict[str, float | torch.Tensor] or float or torch.Tensor, optional
            Maximum values for min_max method. Same format as means.
        stats_file : str or Path, optional
            Path to ``.npz`` file containing normalization statistics.
            File should contain per-field dicts with keys like 'mean',
            'std', 'min', 'max'.
        eps : float, default=1e-8
            Small value added to prevent division by zero.

        Raises
        ------
        ValueError
            If input_keys is empty, method is invalid, or required
            parameters are missing.
        """
        super().__init__()

        if not input_keys:
            raise ValueError("input_keys cannot be empty")

        self.input_keys = list(input_keys)
        self.eps = eps

        # Handle backward compatibility: if means/stds provided without method
        if means is not None and stds is not None and method is None:
            warnings.warn(
                "Providing 'means' and 'stds' without 'method' parameter is deprecated. "
                "Please specify method='mean_std' explicitly. "
                "This will become an error in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            method = "mean_std"

        # Validate method
        if method not in ["mean_std", "min_max"]:
            raise ValueError(f"method must be 'mean_std' or 'min_max', got: {method}")

        self.method = method

        # Load stats from file if provided
        if stats_file is not None:
            stats = self._load_stats_from_npz(stats_file)
            if method == "mean_std":
                if means is None:
                    means = stats.get("means", {})
                if stds is None:
                    stds = stats.get("stds", {})
            else:  # min_max
                if mins is None:
                    mins = stats.get("mins", {})
                if maxs is None:
                    maxs = stats.get("maxs", {})

        # Initialize storage based on method
        if method == "mean_std":
            if means is None or stds is None:
                raise ValueError(
                    "For method='mean_std', both 'means' and 'stds' must be provided "
                    "either directly or via stats_file"
                )
            self._means = self._process_stats_dict(means, "mean")
            self._stds = self._process_stats_dict(stds, "std")
            self._mins: Optional[dict[str, torch.Tensor]] = None
            self._maxs: Optional[dict[str, torch.Tensor]] = None

        else:  # min_max
            if mins is None or maxs is None:
                raise ValueError(
                    "For method='min_max', both 'mins' and 'maxs' must be provided "
                    "either directly or via stats_file"
                )
            self._mins = self._process_stats_dict(mins, "min")
            self._maxs = self._process_stats_dict(maxs, "max")
            self._means: Optional[dict[str, torch.Tensor]] = None
            self._stds: Optional[dict[str, torch.Tensor]] = None

    def _process_stats_dict(
        self,
        stats: dict[str, float | torch.Tensor] | float | torch.Tensor,
        stat_name: str,
    ) -> dict[str, torch.Tensor]:
        """Process statistics into dict of tensors for each field."""
        result: dict[str, torch.Tensor] = {}

        if isinstance(stats, dict):
            for key in self.input_keys:
                if key not in stats:
                    raise ValueError(
                        f"{stat_name.capitalize()} not provided for field '{key}'"
                    )
                val = stats[key]
                result[key] = (
                    torch.as_tensor(val) if not isinstance(val, torch.Tensor) else val
                )
        else:
            # Single value for all fields
            stat_tensor = (
                torch.as_tensor(stats) if not isinstance(stats, torch.Tensor) else stats
            )
            for key in self.input_keys:
                result[key] = stat_tensor.clone()

        return result

    def _load_stats_from_npz(self, stats_file: str | Path) -> dict[str, dict]:
        """
        Load normalization statistics from .npz file.

        Expected file structure: Dictionary mapping field names to dicts with
        keys 'mean', 'std', 'min', 'max' (numpy arrays).

        Parameters
        ----------
        stats_file : str or Path
            Path to .npz file.

        Returns
        -------
        dict[str, dict]
            Dictionary with keys 'means', 'stds', 'mins', 'maxs', each mapping
            field names to torch tensors.

        Raises
        ------
        FileNotFoundError
            If file doesn't exist.
        ValueError
            If required statistics are missing.
        """
        file_path = Path(stats_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {stats_file}")

        # Load npz file
        data = np.load(str(file_path), allow_pickle=True)

        # Initialize output dicts
        means_dict = {}
        stds_dict = {}
        mins_dict = {}
        maxs_dict = {}

        # Process each field
        for key in self.input_keys:
            if key not in data:
                raise ValueError(
                    f"Field '{key}' not found in stats file. "
                    f"Available fields: {list(data.keys())}"
                )

            field_stats = data[key]
            if isinstance(field_stats, np.ndarray) and field_stats.dtype == object:
                # It's a dict stored as numpy object
                field_stats = field_stats.item()

            # Extract stats if available
            if "mean" in field_stats:
                means_dict[key] = torch.as_tensor(field_stats["mean"])
            if "std" in field_stats:
                stds_dict[key] = torch.as_tensor(field_stats["std"])
            if "min" in field_stats:
                mins_dict[key] = torch.as_tensor(field_stats["min"])
            if "max" in field_stats:
                maxs_dict[key] = torch.as_tensor(field_stats["max"])

        return {
            "means": means_dict,
            "stds": stds_dict,
            "mins": mins_dict,
            "maxs": maxs_dict,
        }

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Normalize the specified fields in the TensorDict.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict.

        Returns
        -------
        TensorDict
            TensorDict with normalized fields.

        Raises
        ------
        KeyError
            If a specified field is not in the TensorDict.
        """
        updates = {}

        for key in self.input_keys:
            if key not in data.keys():
                raise KeyError(
                    f"Field '{key}' not found in data. Available: {list(data.keys())}"
                )

            tensor = data[key]

            if self.method == "mean_std":
                mean = self._means[key]
                std = self._stds[key]

                # Normalize: (x - mean) / std
                updates[key] = (tensor - mean) / (std + self.eps)

            elif self.method == "min_max":
                min_val = self._mins[key]
                max_val = self._maxs[key]

                # Normalize to [-1, 1]: (x - center) / half_range
                center = (max_val + min_val) / 2.0
                half_range = (max_val - min_val) / 2.0
                updates[key] = (tensor - center) / (half_range + self.eps)
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")

        # Update TensorDict with normalized values
        return data.update(updates)

    def to(self, device: torch.device | str) -> Normalize:
        """Move normalization parameters to the specified device."""
        super().to(device)
        device = torch.device(device) if isinstance(device, str) else device

        if self.method == "mean_std":
            for key in self.input_keys:
                self._means[key] = self._means[key].to(device, non_blocking=True)
                self._stds[key] = self._stds[key].to(device, non_blocking=True)
        else:  # min_max
            for key in self.input_keys:
                self._mins[key] = self._mins[key].to(device, non_blocking=True)
                self._maxs[key] = self._maxs[key].to(device, non_blocking=True)

        return self

    def inverse(self, data: TensorDict) -> TensorDict:
        """
        Apply inverse normalization (denormalize).

        For mean_std method: x * std + mean
        For min_max method: x * half_range + center

        Parameters
        ----------
        data : TensorDict
            Normalized TensorDict.

        Returns
        -------
        TensorDict
            Denormalized TensorDict.
        """
        updates = {}

        for key in self.input_keys:
            if key not in data.keys():
                raise KeyError(f"Field '{key}' not found in data")

            tensor = data[key]

            if self.method == "mean_std":
                mean = self._means[key]
                std = self._stds[key]

                updates[key] = tensor * (std + self.eps) + mean

            else:  # min_max
                min_val = self._mins[key]
                max_val = self._maxs[key]

                center = (max_val + min_val) / 2.0
                half_range = (max_val - min_val) / 2.0
                updates[key] = tensor * (half_range + self.eps) + center

        return data.update(updates)

    def state_dict(self) -> dict[str, Any]:
        """Return normalization parameters."""
        state = {
            "input_keys": self.input_keys,
            "method": self.method,
            "eps": self.eps,
        }

        if self.method == "mean_std":
            state["means"] = {k: v.cpu() for k, v in self._means.items()}
            state["stds"] = {k: v.cpu() for k, v in self._stds.items()}
        else:  # min_max
            state["mins"] = {k: v.cpu() for k, v in self._mins.items()}
            state["maxs"] = {k: v.cpu() for k, v in self._maxs.items()}

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load normalization parameters."""
        self.input_keys = state_dict["input_keys"]
        self.method = state_dict.get(
            "method", "mean_std"
        )  # Default for backward compat
        self.eps = state_dict["eps"]

        if self.method == "mean_std":
            self._means = {k: v.clone() for k, v in state_dict["means"].items()}
            self._stds = {k: v.clone() for k, v in state_dict["stds"].items()}
            self._mins = None
            self._maxs = None
        else:  # min_max
            self._mins = {k: v.clone() for k, v in state_dict["mins"].items()}
            self._maxs = {k: v.clone() for k, v in state_dict["maxs"].items()}
            self._means = None
            self._stds = None

    def extra_repr(self) -> str:
        return f"method={self.method}, input_keys={self.input_keys}, eps={self.eps}"
