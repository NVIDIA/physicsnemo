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

"""Flexible metric calculator for configurable target fields."""

from __future__ import annotations

import torch
import torch.distributed as dist

from utils import FieldSpec, parse_target_config


# ---------------------------------------------------------------------------
# Core metric functions operating on [batch, points] tensors
# ---------------------------------------------------------------------------


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error (absolute)."""
    return torch.mean(torch.abs(pred - target))


def compute_relative_l1(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Relative L1: sum|diff| / sum|target| per sample, then mean."""
    abs_diff = torch.abs(pred - target)
    l1_num = torch.sum(abs_diff, dim=1)
    l1_denom = torch.sum(torch.abs(target), dim=1)
    return torch.mean(l1_num / (l1_denom + eps))


def compute_relative_l2(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Relative L2: sqrt(sum(diff^2)) / sqrt(sum(target^2)) per sample, then mean."""
    diff = pred - target
    l2_num = torch.sqrt(torch.sum(diff**2, dim=1))
    l2_denom = torch.sqrt(torch.sum(target**2, dim=1))
    return torch.mean(l2_num / (l2_denom + eps))


METRIC_FUNCTIONS = {
    "mae": compute_mae,
    "l1": compute_relative_l1,
    "l2": compute_relative_l2,
}


# ---------------------------------------------------------------------------
# MetricCalculator class
# ---------------------------------------------------------------------------


class MetricCalculator:
    """Configurable metric calculator for scalar and vector target fields.

    Computes L1, L2, and MAE metrics for each configured target field.
    For vector fields, computes both elementwise metrics (per component)
    and aggregate metrics (magnitude-based).

    Parameters
    ----------
    target_config : dict[str, str]
        Mapping of field names to types. Order determines channel indices.
        Example: {"pressure": "scalar", "velocity": "vector", "turbulence": "scalar"}
    process_group : dist.ProcessGroup | None, optional
        Process group for distributed all-reduce. If None, no reduction is performed.
    n_spatial_dims : int, optional
        Dimensionality of vector fields. Default is 3.
    metrics : list[str] | None, optional
        Which metrics to compute. Options: "l1", "l2", "mae".
        Default is all three: ["l1", "l2", "mae"].
    prefix : str, optional
        Prefix for all metric names (e.g., "surface" -> "surface/pressure_l1").
        Default is empty string.

    Examples
    --------
    >>> calc = MetricCalculator(
    ...     target_config={"pressure": "scalar", "velocity": "vector"},
    ...     prefix="surface",
    ... )
    >>> pred = torch.randn(2, 100, 4)  # [batch, points, channels]
    >>> target = torch.randn(2, 100, 4)
    >>> metrics = calc(pred, target)
    """

    VECTOR_COMPONENTS = ("x", "y", "z")

    def __init__(
        self,
        target_config: dict[str, str],
        process_group: dist.ProcessGroup | None = None,
        n_spatial_dims: int = 3,
        metrics: list[str] | None = None,
        prefix: str = "",
    ):
        self.process_group = process_group
        self.n_spatial_dims = n_spatial_dims
        self.metric_names = metrics if metrics is not None else ["l1", "l2", "mae"]
        self.prefix = prefix

        # Validate metric names
        for m in self.metric_names:
            if m not in METRIC_FUNCTIONS:
                raise ValueError(
                    f"Unknown metric '{m}'. Available: {list(METRIC_FUNCTIONS.keys())}"
                )

        # Parse target config to build field specifications using shared utility
        self.field_specs = parse_target_config(target_config, n_spatial_dims)
        self.total_channels = sum(spec.dim for spec in self.field_specs)

    def _make_key(self, *parts: str) -> str:
        """Construct a metric key with optional prefix."""
        key = "_".join(parts)
        return f"{self.prefix}/{key}" if self.prefix else key

    def _compute_metrics_for_field(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        name: str,
    ) -> dict[str, torch.Tensor]:
        """Compute all configured metrics for a [batch, points] field."""
        return {
            self._make_key(name, m): METRIC_FUNCTIONS[m](pred, target)
            for m in self.metric_names
        }

    def _compute_scalar_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        name: str,
    ) -> dict[str, torch.Tensor]:
        """Compute metrics for a scalar field [batch, points]."""
        return self._compute_metrics_for_field(pred, target, name)

    def _compute_vector_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        name: str,
    ) -> dict[str, torch.Tensor]:
        """Compute metrics for a vector field [batch, points, dim].

        Computes elementwise (per component) and aggregate (magnitude) metrics.
        """
        metrics = {}

        # Elementwise metrics (per component)
        for i, comp in enumerate(self.VECTOR_COMPONENTS[: pred.shape[-1]]):
            comp_metrics = self._compute_metrics_for_field(
                pred[:, :, i], target[:, :, i], f"{name}_{comp}"
            )
            metrics.update(comp_metrics)

        # Aggregate metrics (magnitude)
        pred_mag = torch.linalg.vector_norm(pred, dim=-1)
        target_mag = torch.linalg.vector_norm(target, dim=-1)
        metrics.update(self._compute_metrics_for_field(pred_mag, target_mag, name))

        return metrics

    def _all_reduce(self, metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """All-reduce metrics across the process group."""
        if self.process_group is None:
            return metrics

        world_size = dist.get_world_size(self.process_group)
        if world_size == 1:
            return metrics

        keys = list(metrics.keys())
        stacked = torch.stack([metrics[k] for k in keys])

        dist.all_reduce(stacked, group=self.process_group)
        stacked = stacked / world_size

        return {k: stacked[i] for i, k in enumerate(keys)}

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all configured metrics.

        Args:
            pred: Predicted values, shape [batch, points, channels].
            target: Target values, shape [batch, points, channels].

        Returns:
            Dictionary of metric name -> scalar tensor value.
        """
        if pred.shape[-1] != self.total_channels:
            raise ValueError(
                f"Expected {self.total_channels} channels based on target config, "
                f"but got {pred.shape[-1]}."
            )

        metrics = {}

        with torch.no_grad():
            for spec in self.field_specs:
                pred_field = pred[:, :, spec.start_index : spec.end_index]
                target_field = target[:, :, spec.start_index : spec.end_index]

                if spec.field_type == "scalar":
                    field_metrics = self._compute_scalar_metrics(
                        pred_field.squeeze(-1), target_field.squeeze(-1), spec.name
                    )
                else:
                    field_metrics = self._compute_vector_metrics(
                        pred_field, target_field, spec.name
                    )

                metrics.update(field_metrics)

            metrics = self._all_reduce(metrics)

        return metrics

    def __repr__(self) -> str:
        fields_str = ", ".join(
            f"{s.name}:{s.field_type}[{s.start_index}:{s.end_index}]"
            for s in self.field_specs
        )
        return (
            f"MetricCalculator(fields=[{fields_str}], "
            f"metrics={self.metric_names}, prefix='{self.prefix}')"
        )
