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

"""Configurable metric calculator on TensorDict inputs.

Mirrors the TensorDict-based interface of :class:`LossCalculator`. Each
named target field declared in ``target_config`` produces:

- For ``"scalar"`` types: per-metric values (``l1``, ``l2``, ``mae`` by
  default), keyed ``"<prefix>/<name>_<metric>"``.
- For ``"vector"`` types: per-component values
  (``"<prefix>/<name>_x_<metric>"`` etc.) plus aggregate magnitude values
  (``"<prefix>/<name>_<metric>"``).

Metrics are reported unweighted -- per-field weighting belongs in the
loss, not in diagnostic summaries.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import torch
import torch.distributed as dist
from jaxtyping import Float
from tensordict import TensorDict

from utils import FieldType, align_scalar_shapes, field_dim, validate_field_coverage

### Recipe-wide alias for the metric-name enum that the dataset YAMLs use.
MetricName: TypeAlias = Literal["mae", "l1", "l2"]


### ---------------------------------------------------------------------------
### Per-tensor metric kernels
### ---------------------------------------------------------------------------


def _mean_absolute_error(
    pred: torch.Tensor, target: torch.Tensor
) -> Float[torch.Tensor, ""]:
    """Mean absolute error over all elements."""
    return torch.mean(torch.abs(pred - target))


def _relative_l1(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> Float[torch.Tensor, ""]:
    """``sum|pred - target| / sum|target|``, computed over the spatial axis."""
    abs_diff = torch.abs(pred - target)
    if pred.ndim == 0:
        return abs_diff / (torch.abs(target) + eps)
    ### Sum over the spatial axis (treat all leading dims as batch).
    spatial_axis = -1
    num = torch.sum(abs_diff, dim=spatial_axis)
    denom = torch.sum(torch.abs(target), dim=spatial_axis)
    return torch.mean(num / (denom + eps))


def _relative_l2(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> Float[torch.Tensor, ""]:
    """``sqrt(sum diff^2) / sqrt(sum target^2)``, over the spatial axis."""
    diff_sq = (pred - target) ** 2
    if pred.ndim == 0:
        return torch.sqrt(diff_sq) / (torch.sqrt(target**2) + eps)
    spatial_axis = -1
    num = torch.sqrt(torch.sum(diff_sq, dim=spatial_axis))
    denom = torch.sqrt(torch.sum(target**2, dim=spatial_axis))
    return torch.mean(num / (denom + eps))


METRIC_FUNCTIONS: dict[MetricName, Callable[..., torch.Tensor]] = {
    "mae": _mean_absolute_error,
    "l1": _relative_l1,
    "l2": _relative_l2,
}

### Default metrics computed when the user doesn't override `metrics:` in
### the dataset YAML. Exposed as a constant so train.py can fall back to
### the same list when the dataset YAML omits the block.
DEFAULT_METRICS: tuple[MetricName, ...] = ("l1", "l2", "mae")


### ---------------------------------------------------------------------------
### MetricCalculator
### ---------------------------------------------------------------------------


VECTOR_COMPONENTS = ("x", "y", "z", "w")


class MetricCalculator:
    """Per-field metric aggregator over `TensorDict` predictions.

    Args:
        target_config: ``{name: scalar|vector}`` mapping.
        process_group: Optional distributed process group for all-reduce.
            When ``None`` (default), no reduction is performed.
        n_spatial_dims: Vector field dimensionality (used to label
            components when ``> len(VECTOR_COMPONENTS)`` falls back to
            integer indices).
        metrics: Names of metrics to compute. Subset of
            ``METRIC_FUNCTIONS``. Defaults to :data:`DEFAULT_METRICS`.
        prefix: Optional prefix on the returned metric keys.
    """

    def __init__(
        self,
        target_config: dict[str, FieldType],
        process_group: dist.ProcessGroup | None = None,
        n_spatial_dims: int = 3,
        metrics: Sequence[MetricName] | None = None,
        prefix: str = "",
    ) -> None:
        ### `target_config` values are required to be lowercase per the
        ### `FieldType` contract; copy verbatim so callers can mutate their
        ### original without affecting us.
        self.target_config: dict[str, FieldType] = dict(target_config)
        self.process_group = process_group
        self.n_spatial_dims = n_spatial_dims
        self.metric_names = (
            list(metrics) if metrics is not None else list(DEFAULT_METRICS)
        )
        self.prefix = prefix

        for m in self.metric_names:
            if m not in METRIC_FUNCTIONS:
                raise ValueError(
                    f"Unknown metric {m!r}; available {list(METRIC_FUNCTIONS)!r}"
                )

        ### `field_dim` raises on unknown field types, validating the config.
        self.total_channels = sum(
            field_dim(t, n_spatial_dims) for t in self.target_config.values()
        )

    def _make_key(self, *parts: str) -> str:
        """Build a flat metric key, ``"<prefix>/<part1>_<part2>_..."``.

        ``"_"`` joins the parts (e.g. ``"pressure_x_l2"``) because metric
        names are leaf-level dashboard entries; the optional ``prefix/``
        carries the namespacing. Compare with
        :class:`LossCalculator._make_key`, which uses ``"/"`` everywhere
        because loss keys feed into TensorBoard's nested-tag hierarchy
        (``"loss/surface/pressure"``).
        """
        key = "_".join(parts)
        return f"{self.prefix}/{key}" if self.prefix else key

    def _metrics_for_tensor(
        self, pred: torch.Tensor, target: torch.Tensor, name_parts: tuple[str, ...]
    ) -> dict[str, torch.Tensor]:
        return {
            self._make_key(*name_parts, m): METRIC_FUNCTIONS[m](pred, target)
            for m in self.metric_names
        }

    def _all_reduce(self, metrics: TensorDict) -> TensorDict:
        if self.process_group is None:
            return metrics
        world_size = dist.get_world_size(self.process_group)
        if world_size == 1:
            return metrics
        ### Single all_reduce over a stacked 1-D tensor (vs. one comm
        ### per leaf) -- one collective beats N regardless of the
        ### container type. Rebuild a TensorDict from the reduced
        ### stack so callers see the same per-key access pattern.
        keys = list(metrics.keys())
        stacked = torch.stack([metrics[k] for k in keys])
        dist.all_reduce(stacked, group=self.process_group)
        stacked = stacked / world_size
        return TensorDict({k: stacked[i] for i, k in enumerate(keys)}, batch_size=[])

    def __call__(
        self,
        pred: TensorDict,
        target: TensorDict,
    ) -> TensorDict:
        """Compute per-field metrics over a TensorDict pred / target pair.

        Args:
            pred: TensorDict of predictions, one leaf per target field.
            target: TensorDict of the same structure as ``pred``.

        Returns:
            0-D ``TensorDict`` (``batch_size=[]``) keyed by
            ``"<prefix>/<name>_<metric>"`` for scalar fields and by
            ``"<prefix>/<name>_<comp>_<metric>"`` plus
            ``"<prefix>/<name>_<metric>"`` (aggregate magnitude) for
            vector fields. Slash-containing keys are stored verbatim;
            TensorDict only treats ``/`` as nested when the caller
            explicitly invokes ``flatten_keys("/")``.
        """
        validate_field_coverage(self.target_config, pred, target)

        ### Build the per-field bag as a plain dict during the loop so
        ### the inner ``out.update(...)`` calls stay simple, then wrap
        ### into a 0-D TensorDict at the boundary so callers get
        ### TensorDict's batched ops (``.detach()``, ``.add_()``, ...).
        out: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, field_type in self.target_config.items():
                p, t = pred[name], target[name]
                if field_type == "scalar":
                    p, t = align_scalar_shapes(p, t)
                    out.update(self._metrics_for_tensor(p, t, (name,)))
                else:  # vector
                    n_components = p.shape[-1]
                    ### Per-component metrics.
                    for i in range(n_components):
                        comp = (
                            VECTOR_COMPONENTS[i]
                            if i < len(VECTOR_COMPONENTS)
                            else str(i)
                        )
                        out.update(
                            self._metrics_for_tensor(p[..., i], t[..., i], (name, comp))
                        )
                    ### Aggregate magnitude metric.
                    p_mag = torch.linalg.vector_norm(p, dim=-1)
                    t_mag = torch.linalg.vector_norm(t, dim=-1)
                    out.update(self._metrics_for_tensor(p_mag, t_mag, (name,)))

        return self._all_reduce(TensorDict(out, batch_size=[]))

    def __repr__(self) -> str:
        fields_str = ", ".join(f"{n}:{t}" for n, t in self.target_config.items())
        return (
            f"MetricCalculator(fields=[{fields_str}], "
            f"metrics={self.metric_names}, prefix='{self.prefix}')"
        )
