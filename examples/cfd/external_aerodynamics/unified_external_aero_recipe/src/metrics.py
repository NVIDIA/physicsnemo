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
  (``"<prefix>/<name>_x_<metric>"`` etc.) plus aggregate values over all
  components jointly (``"<prefix>/<name>_<metric>"``).

Per-field loss weights (``field_weights``) do not apply to metrics. With
``measure_weighted=True`` (default) and a ``target_measure`` passed to the
call, each sum or mean over target points is weighted by the point's
measure (its cell's area on a surface, its volume in a volume), so the
metrics do not depend on how finely the mesh resolves each region.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias, cast

import torch
from jaxtyping import Float
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from utils import (
    FieldType,
    align_scalar_shapes,
    align_target_measure,
    mask_nonfinite,
    normalize_target_measure,
    validate_field_coverage,
)

from physicsnemo.datapipes.keys import NestedKey, as_nested_key

### Recipe-wide alias for the metric-name enum that the dataset YAMLs use.
MetricName: TypeAlias = Literal["mae", "l1", "l2"]


### ---------------------------------------------------------------------------
### Per-tensor metric kernels
### ---------------------------------------------------------------------------


### Each kernel takes an optional ``weights``: mean-one per-point weights
### (``measure / mean(measure)``) of shape ``(..., N)``. Equal measures give
### weights of one, so the weighted result equals the unweighted one.


def _mean_absolute_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> Float[torch.Tensor, ""]:
    """Mean absolute error over all elements."""
    abs_diff = torch.abs(pred - target)
    if weights is None or pred.ndim == 0:
        return torch.mean(abs_diff)
    return torch.mean(torch.mean(abs_diff * weights, dim=-1))


def _relative_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> Float[torch.Tensor, ""]:
    """``sum|pred - target| / sum|target|``, computed over the spatial axis."""
    abs_diff = torch.abs(pred - target)
    if pred.ndim == 0:
        return abs_diff / (torch.abs(target) + eps)
    abs_target = torch.abs(target)
    if weights is not None:
        abs_diff, abs_target = abs_diff * weights, abs_target * weights
    ### Sum over the spatial axis (treat all leading dims as batch).
    spatial_axis = -1
    num = torch.sum(abs_diff, dim=spatial_axis)
    denom = torch.sum(abs_target, dim=spatial_axis)
    return torch.mean(num / (denom + eps))


def _relative_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> Float[torch.Tensor, ""]:
    """``sqrt(sum diff^2) / sqrt(sum target^2)``, over the spatial axis."""
    diff_sq = (pred - target) ** 2
    if pred.ndim == 0:
        return torch.sqrt(diff_sq) / (torch.sqrt(target**2) + eps)
    target_sq = target**2
    if weights is not None:
        diff_sq, target_sq = diff_sq * weights, target_sq * weights
    spatial_axis = -1
    num = torch.sqrt(torch.sum(diff_sq, dim=spatial_axis))
    denom = torch.sqrt(torch.sum(target_sq, dim=spatial_axis))
    return torch.mean(num / (denom + eps))


def _point_weights(
    target_measure: torch.Tensor | TensorDict | None,
    key: NestedKey,
    reference: torch.Tensor,
    field_type: FieldType,
) -> torch.Tensor | None:
    """Mean-one per-point weights for one field, or ``None`` without a measure."""
    measure = align_target_measure(target_measure, key, reference, field_type)
    return None if measure is None else normalize_target_measure(measure)


METRIC_FUNCTIONS: dict[MetricName, Callable[..., torch.Tensor]] = {
    "mae": _mean_absolute_error,
    "l1": _relative_l1,
    "l2": _relative_l2,
}

### Default metrics computed when the user doesn't override `metrics:` in
### the dataset YAML. Exposed as a constant so train.py can fall back to
### the same list when the dataset YAML omits the block.
DEFAULT_METRICS: tuple[MetricName, ...] = ("l1", "l2", "mae")


def resolve_metrics(cfg: DictConfig) -> list[MetricName]:
    """Resolve the recipe-side ``cfg.metrics`` list, or the default set.

    ``metrics:`` is a recipe-level (not per-dataset) choice; both the
    trainer and the inference companion read it the same way. Falls back
    to :data:`DEFAULT_METRICS` when the key is unset.

    The resolved list is validated against :data:`METRIC_FUNCTIONS` here so
    a typo (e.g. ``rmse``) fails fast with a clear message rather than
    surfacing later as a :class:`MetricCalculator` lookup error (which
    still guards as a backstop).

    Raises:
        TypeError: If ``metrics:`` is set but is not a list.
        ValueError: If any entry is not a known metric name.
    """
    metrics_cfg = OmegaConf.select(cfg, "metrics", default=None)
    if metrics_cfg is None:
        return list(DEFAULT_METRICS)
    resolved = OmegaConf.to_container(metrics_cfg, resolve=True)
    if not isinstance(resolved, list):
        raise TypeError(
            f"`metrics:` must be a list of metric names, got "
            f"{type(resolved).__name__} ({resolved!r})."
        )
    unknown = [m for m in resolved if m not in METRIC_FUNCTIONS]
    if unknown:
        raise ValueError(
            f"Unknown metric(s) {unknown!r} in `metrics:`; "
            f"available: {list(METRIC_FUNCTIONS)!r}."
        )
    return cast("list[MetricName]", resolved)


### ---------------------------------------------------------------------------
### MetricCalculator
### ---------------------------------------------------------------------------


VECTOR_COMPONENTS = ("x", "y", "z", "w")


class MetricCalculator:
    """Per-field metric aggregator over `TensorDict` predictions.

    Args:
        target_config: ``{name: scalar|vector}`` mapping.
        n_spatial_dims: Vector field dimensionality (used to label
            components when ``> len(VECTOR_COMPONENTS)`` falls back to
            integer indices).
        metrics: Names of metrics to compute. Subset of
            ``METRIC_FUNCTIONS``. Defaults to :data:`DEFAULT_METRICS`.
        prefix: Optional prefix on the returned metric keys.
        measure_weighted: When ``True`` (default), weight each target point
            by the ``target_measure`` passed to the call, if any. When
            ``False``, ignore it and count every point alike.
        mask_nonfinite_targets: When ``True``, target elements that are not
            finite are left out, as in :class:`loss.LossCalculator`; a field
            with no finite target gets NaN metrics.
    """

    def __init__(
        self,
        target_config: dict[str, FieldType],
        n_spatial_dims: int = 3,
        metrics: Sequence[MetricName] | None = None,
        prefix: str = "",
        measure_weighted: bool = True,
        mask_nonfinite_targets: bool = False,
    ) -> None:
        ### `target_config` values are required to be lowercase per the
        ### `FieldType` contract; copy verbatim so callers can mutate their
        ### original without affecting us.
        self.target_config: dict[str, FieldType] = dict(target_config)
        self.n_spatial_dims = n_spatial_dims
        self.metric_names = (
            list(metrics) if metrics is not None else list(DEFAULT_METRICS)
        )
        self.prefix = prefix
        self.measure_weighted = measure_weighted
        self.mask_nonfinite_targets = mask_nonfinite_targets

        for m in self.metric_names:
            if m not in METRIC_FUNCTIONS:
                raise ValueError(
                    f"Unknown metric {m!r}; available {list(METRIC_FUNCTIONS)!r}"
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

    def expected_keys(self) -> list[str]:
        """The exact key set :meth:`__call__` produces, derivable without data.

        Vector fields contribute keys per spatial component plus the bare-name
        aggregate keys; scalars contribute one key per metric.
        Lets callers pre-size accumulators identically on every rank --
        e.g. ``infer.py`` zero-fills its running sums with these keys so
        the cross-rank all-reduce packs the same tensor length even on a
        rank whose sampler shard was empty.
        """
        keys: list[str] = []
        for name, field_type in self.target_config.items():
            if field_type == "vector":
                for i in range(self.n_spatial_dims):
                    comp = (
                        VECTOR_COMPONENTS[i] if i < len(VECTOR_COMPONENTS) else str(i)
                    )
                    keys.extend(
                        self._make_key(name, comp, m) for m in self.metric_names
                    )
            keys.extend(self._make_key(name, m) for m in self.metric_names)
        return keys

    def _metrics_for_tensor(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        name_parts: tuple[str, ...],
        weights: torch.Tensor | None = None,
        fraction: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        values = {
            m: METRIC_FUNCTIONS[m](pred, target, weights) for m in self.metric_names
        }
        ### With masked targets (``fraction``), the mean (``mae``) is rescaled
        ### to the valid elements; the relative norms are sum ratios already.
        ### A field with no valid element has no error to report: NaN.
        if fraction is not None:
            if "mae" in values:
                values["mae"] = values["mae"] / fraction
            values = {
                m: torch.where(fraction > 0, v, torch.nan) for m, v in values.items()
            }
        return {self._make_key(*name_parts, m): v for m, v in values.items()}

    def __call__(
        self,
        pred: TensorDict,
        target: TensorDict,
        target_measure: torch.Tensor | TensorDict | None = None,
    ) -> TensorDict:
        """Compute per-field metrics over a TensorDict pred / target pair.

        Args:
            pred: TensorDict of predictions, one leaf per target field.
            target: TensorDict of the same structure as ``pred``.
            target_measure: Optional per-point measure, shape ``(..., N)``,
                aligned with the target points, or a TensorDict keyed like
                ``target`` with one per field (a field it lacks counts every
                point alike). Ignored when the calculator was built with
                ``measure_weighted=False``.

        Returns:
            0-D ``TensorDict`` (``batch_size=[]``) keyed by
            ``"<prefix>/<name>_<metric>"`` for scalar fields and by
            ``"<prefix>/<name>_<comp>_<metric>"`` plus
            ``"<prefix>/<name>_<metric>"`` (aggregate over all components
            jointly) for vector fields.
            Slash-containing keys are stored verbatim;
            TensorDict only treats ``/`` as nested when the caller
            explicitly invokes ``flatten_keys("/")``.
        """
        validate_field_coverage(self.target_config, pred, target)
        if not self.measure_weighted:
            target_measure = None

        ### Build the per-field bag as a plain dict during the loop so
        ### the inner ``out.update(...)`` calls stay simple, then wrap
        ### into a 0-D TensorDict at the boundary so callers get
        ### TensorDict's batched ops (``.detach()``, ``.add_()``, ...).
        out: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, field_type in self.target_config.items():
                ### Nested lookup; metric keys keep the config spelling.
                key = as_nested_key(name)
                p, t = pred[key], target[key]
                if field_type == "scalar":
                    p, t = align_scalar_shapes(p, t)
                weights = _point_weights(target_measure, key, p, field_type)
                fraction = None
                if self.mask_nonfinite_targets:
                    p, t, fraction = mask_nonfinite(p, t, field_type, weights)
                if field_type == "scalar":
                    out.update(
                        self._metrics_for_tensor(p, t, (name,), weights, fraction)
                    )
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
                            self._metrics_for_tensor(
                                p[..., i], t[..., i], (name, comp), weights, fraction
                            )
                        )
                    ### Aggregate vector metric under the bare field name:
                    ### all components jointly, flattened so the relative
                    ### norms reduce over the whole field (Frobenius for
                    ### l2) and direction errors count.
                    ### The weights repeat per component, staying mean-one.
                    if weights is not None:
                        weights = weights.unsqueeze(-1).expand_as(p).flatten(-2)
                    out.update(
                        self._metrics_for_tensor(
                            p.flatten(-2), t.flatten(-2), (name,), weights, fraction
                        )
                    )

        return TensorDict(out)

    def __repr__(self) -> str:
        fields_str = ", ".join(f"{n}:{t}" for n, t in self.target_config.items())
        return (
            f"MetricCalculator(fields=[{fields_str}], "
            f"metrics={self.metric_names}, prefix='{self.prefix}', "
            f"measure_weighted={self.measure_weighted})"
        )
