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

"""Configurable loss calculator on TensorDict inputs.

The loss accepts `TensorDict` predictions and targets keyed by field name,
matching the recipe's DomainMesh-native flow. For each named target field
declared in ``target_config``, the loss type is applied according to the
field type:

- ``"scalar"``: single mean over all elements.
- ``"vector"``: per-component mean, summed across components, so the
  contribution of a ``D``-dimensional vector field scales as ``D`` rather
  than ``1``.

Per-field weights (``field_weights``) multiply each per-field loss
(default ``1.0``) before the per-field losses are summed into a total.
When ``normalize_by_channels=True`` the total is divided by the total
channel count (``sum(per_field_dims)``) so the total scale is invariant
to how many channels each field contributes.

With ``measure_weighted=True`` (default) and a ``target_measure`` passed to
the call, each mean over target points is weighted by the point's measure
(its cell's area on a surface, its volume in a volume), so the loss
approximates an integral over the domain rather than a per-point average.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from tensordict import TensorDict
from utils import (
    FieldType,
    align_scalar_shapes,
    align_target_measure,
    field_dim,
    mask_nonfinite,
    normalize_target_measure,
    validate_field_coverage,
)

from physicsnemo.datapipes.keys import as_nested_key
from physicsnemo.metrics.general.relative_error import relative_mse

_LOGGER = logging.getLogger("training.loss")

DEFAULT_HUBER_DELTA = 1.0

### ``"rmse"`` is a deprecated alias for ``"relative_mse"``: the quantity it
### always computed is the target-normalized relative MSE (no square root
### anywhere), so the old name misled anyone comparing against reported RMSE
### numbers. ``LossCalculator`` rewrites it on construction with a warning.
LossType = Literal["huber", "mse", "relative_mse", "rmse"]
_VALID_LOSS_TYPES = ("huber", "mse", "relative_mse", "rmse")


### ---------------------------------------------------------------------------
### Per-field loss kernels
### ---------------------------------------------------------------------------


def _scalar_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: LossType,
    delta: float,
    target_measure: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> Float[torch.Tensor, ""]:
    """Element-wise loss reduced to a scalar.

    A defensive shape check guards against config bugs where a ``"scalar"``
    target is fed a vector tensor (or vice versa): after
    :func:`align_scalar_shapes`, ``pred`` and ``target`` are required to
    share the same shape, since broadcasting a mismatched pair would
    silently inflate the loss instead of raising.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shapes must match for scalar loss, got "
            f"{tuple(pred.shape)} vs {tuple(target.shape)}; check that the "
            f"target_config field type matches the actual tensor rank."
        )
    if target_measure is not None:
        ### Mean-one weights, measure / mean(measure), as in the metrics: a
        ### weighted mean over the points is then a plain mean of weights * values.
        weights = normalize_target_measure(target_measure)
        if loss_type == "huber":
            values = F.huber_loss(pred, target, reduction="none", delta=delta)
            return torch.mean(weights * values)
        if loss_type == "mse":
            return torch.mean(weights * (pred - target) ** 2)
        if loss_type == "relative_mse":
            dim = -1 if pred.ndim > 0 else None
            return relative_mse(pred, target, dim=dim, weights=weights, eps=eps).mean()
    if loss_type == "huber":
        return F.huber_loss(pred, target, reduction="mean", delta=delta)
    if loss_type == "mse":
        return torch.mean((pred - target) ** 2)
    if loss_type == "relative_mse":
        return relative_mse(pred, target, eps=eps)
    raise ValueError(f"Unknown loss_type {loss_type!r}")


def _vector_loss(
    pred: Float[torch.Tensor, "*batch d"],
    target: Float[torch.Tensor, "*batch d"],
    loss_type: LossType,
    delta: float,
    target_measure: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> Float[torch.Tensor, ""]:
    """Per-component scalar loss summed across components.

    For a vector field of dimension ``D``, the result is
    ``D * mean_huber_over_all_elements`` (or the MSE / relative-error
    analogue), not a single mean over the flattened tensor.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shapes must match, got {tuple(pred.shape)} vs "
            f"{tuple(target.shape)}"
        )
    n_components = pred.shape[-1]

    ### Relative MSE: each component normalized by its own target energy
    ### (reduce over every axis but the last), then summed across components.
    if loss_type == "relative_mse":
        if target_measure is not None:
            weights = normalize_target_measure(target_measure).unsqueeze(-1)
            per_component = relative_mse(pred, target, dim=-2, weights=weights, eps=eps)
            return per_component.sum(dim=-1).mean()
        return torch.sum(
            relative_mse(pred, target, dim=tuple(range(pred.ndim - 1)), eps=eps)
        )

    total = torch.zeros((), device=pred.device, dtype=pred.dtype)
    for i in range(n_components):
        p, t = pred[..., i], target[..., i]
        if target_measure is not None:
            total = total + _scalar_loss(p, t, loss_type, delta, target_measure)
        elif loss_type == "huber":
            total = total + F.huber_loss(p, t, reduction="mean", delta=delta)
        elif loss_type == "mse":
            total = total + torch.mean((p - t) ** 2)
        else:
            raise ValueError(f"Unknown loss_type {loss_type!r}")
    return total


### ---------------------------------------------------------------------------
### LossCalculator
### ---------------------------------------------------------------------------


class LossCalculator:
    """Per-field loss aggregator over `TensorDict` predictions.

    Args:
        target_config: ``{name: scalar|vector}`` mapping. Iteration order
            determines the order in the loss dict and the channel weighting
            in the total.
        loss_type: One of ``"huber"``, ``"mse"``, or ``"relative_mse"``
            (``sum((pred - target)^2) / sum(target^2)``). ``"rmse"`` is
            accepted only as a deprecated alias for ``"relative_mse"`` and
            warns.
        n_spatial_dims: Vector field dimensionality. Used to compute
            channel counts for the normalization denominator.
        field_weights: Optional per-field multiplicative weights. Each
            per-field loss is multiplied by ``field_weights[name]`` before
            summation. Default 1.0 for any unspecified name.
        prefix: Optional prefix for the keys in the returned loss dict
            (e.g. ``"surface"`` produces ``"loss/surface/pressure"``).
        normalize_by_channels: When ``True`` (default), divide the
            (weighted) total loss by ``sum(per_field_dims)`` so the total
            is invariant to how many channels each field contributes.
        measure_weighted: When ``True`` (default), weight each target point
            by the ``target_measure`` passed to the call, if any. When
            ``False``, ignore it and count every point alike.
        mask_nonfinite_targets: When ``True``, target elements that are not
            finite (any component, for vectors) are left out of each field's
            loss, for point sets where some points lack a field (stored as
            NaN there). A field with no finite target contributes 0.

    The returned loss dict contains one entry per field
    (``"loss/[prefix/]<name>"``) plus ``"loss/total"`` (or
    ``"loss/<prefix>"`` when ``prefix`` is set).
    """

    def __init__(
        self,
        target_config: dict[str, FieldType],
        loss_type: LossType = "huber",
        n_spatial_dims: int = 3,
        field_weights: dict[str, float] | None = None,
        prefix: str = "",
        normalize_by_channels: bool = True,
        delta: float = DEFAULT_HUBER_DELTA,
        measure_weighted: bool = True,
        mask_nonfinite_targets: bool = False,
    ) -> None:
        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"Unknown loss_type {loss_type!r}; expected one of "
                f"{_VALID_LOSS_TYPES!r}."
            )
        if loss_type == "rmse":
            ### FutureWarning rather than DeprecationWarning: this is a
            ### user-facing config value, and DeprecationWarning is hidden
            ### by default outside ``__main__``.
            warnings.warn(
                'loss_type="rmse" is a deprecated misnomer: the quantity it '
                "computes is the target-normalized relative MSE (no square "
                'root). Use "relative_mse", which computes the same quantity '
                "for any target with nonzero energy (only the denominator "
                "floor for an all-zero target differs). The alias will be "
                "removed.",
                FutureWarning,
                stacklevel=2,
            )
            loss_type = "relative_mse"
        ### `target_config` values are required to be lowercase per the
        ### `FieldType` contract; we copy the dict verbatim so callers can
        ### mutate their original without affecting us.
        self.target_config: dict[str, FieldType] = dict(target_config)
        self.loss_type = loss_type
        self.n_spatial_dims = n_spatial_dims
        self.prefix = prefix
        self.normalize_by_channels = normalize_by_channels
        self.delta = delta
        self.measure_weighted = measure_weighted
        self.mask_nonfinite_targets = mask_nonfinite_targets

        ### Per-field tensors are looked up by name in the input TensorDict,
        ### so we just need a per-field dim count for total_channels.
        ### `field_dim` raises on unknown field types, validating the config.
        self.total_channels = sum(
            field_dim(t, n_spatial_dims) for t in self.target_config.values()
        )

        ### Per-field weights default to 1.0 for any field not in the dict.
        weights = dict(field_weights or {})
        unknown = set(weights) - set(self.target_config)
        if unknown:
            raise ValueError(
                f"field_weights references unknown fields {sorted(unknown)!r}; "
                f"target_config has {sorted(self.target_config)!r}."
            )
        self.field_weights: dict[str, float] = {
            name: float(weights.get(name, 1.0)) for name in self.target_config
        }

        ### Surface partial-coverage so missing entries are auditable in
        ### the run logs. Only fires when the user supplies *some* but
        ### not *all* names: an empty / None dict is the documented
        ### "all 1.0" path and doesn't deserve a per-construction line.
        if weights and len(weights) < len(self.target_config):
            inferred = sorted(set(self.target_config) - set(weights))
            _LOGGER.info(
                f"LossCalculator field_weights: {self.field_weights} "
                f"(fields {inferred!r} defaulted to 1.0)."
            )

    def _make_key(self, *parts: str) -> str:
        """Build a TensorBoard-tag-shaped key, ``"loss/<prefix>/<part>/.../<part>"``.

        Slash-separated so the result feeds directly into TB scalar tags
        (which use ``/`` as the dashboard hierarchy separator). Compare
        with :class:`MetricCalculator._make_key`, which uses ``"_"`` to
        join the metric-name suffix because metric names like
        ``pressure_l2`` are flat dashboard names rather than nested tags.
        """
        segments = ["loss"]
        if self.prefix:
            segments.append(self.prefix)
        segments.extend(parts)
        return "/".join(segments)

    def __call__(
        self,
        pred: TensorDict,
        target: TensorDict,
        target_measure: torch.Tensor | TensorDict | None = None,
    ) -> tuple[torch.Tensor, TensorDict]:
        """Compute per-field losses and a (weighted) total.

        Args:
            pred: TensorDict of predictions, one leaf per target field.
                Per-element scalars are shape ``(..., N)``, per-element vectors
                are ``(..., N, D)``. Leading batch dims are arbitrary; the loss
                kernels reduce over them.
            target: TensorDict of the same structure as ``pred``.
            target_measure: Optional per-point measure, shape ``(..., N)``,
                aligned with the target points, or a TensorDict keyed like
                ``target`` with one per field (a field it lacks counts every
                point alike). Ignored when the calculator was built with
                ``measure_weighted=False``.

        Returns:
            ``(total_loss, loss_td)``. ``loss_td`` is a 0-D ``TensorDict`` keyed
            by ``"loss/[prefix/]<name>"`` (one entry per field) plus the total.
            Slash-containing keys are stored verbatim; TensorDict only treats
            ``/`` as nested when the caller explicitly invokes
            ``flatten_keys("/")``.
        """
        validate_field_coverage(self.target_config, pred, target)
        if not self.measure_weighted:
            target_measure = None

        ### Find a leaf tensor (descending into nested groups) to seed the
        ### accumulator's dtype/device.
        any_pred = next(iter(pred.values(include_nested=True, leaves_only=True)))
        total_loss = torch.zeros((), device=any_pred.device, dtype=any_pred.dtype)
        ### Build the per-field bag as a plain dict during the loop so the
        ### inner ``loss_dict[key] = ...`` assignment stays simple, then
        ### wrap into a 0-D TensorDict at the boundary so callers get
        ### TensorDict's batched ops (``.detach()``, ``.add_()``, ...).
        loss_dict: dict[str, torch.Tensor] = {}

        for name, field_type in self.target_config.items():
            ### ``name`` may spell a nested leaf ("solution.p"); the loss
            ### key keeps the config spelling while the lookup is nested.
            key = as_nested_key(name)
            p, t = pred[key], target[key]
            if field_type == "scalar":
                ### Caller may pass scalar fields as (..., 1) or (...);
                ### normalize to a single shape so the loss is shape-agnostic.
                p, t = align_scalar_shapes(p, t)
            measure = align_target_measure(target_measure, key, p, field_type)
            if self.mask_nonfinite_targets:
                p, t, fraction = mask_nonfinite(p, t, field_type, measure)
            if field_type == "scalar":
                field_loss = _scalar_loss(p, t, self.loss_type, self.delta, measure)
            else:  # vector
                field_loss = _vector_loss(p, t, self.loss_type, self.delta, measure)
            ### Masked elements add 0 to every sum; a mean also needs the
            ### valid count (the valid share of the measure, when weighted),
            ### while a relative MSE (a sum ratio) does not. A field with no
            ### valid element has loss 0, kept by the clamp.
            if self.mask_nonfinite_targets and self.loss_type != "relative_mse":
                field_loss = field_loss / fraction.clamp_min(
                    torch.finfo(fraction.dtype).tiny
                )

            weighted = field_loss * self.field_weights[name]
            loss_dict[self._make_key(name)] = weighted
            total_loss = total_loss + weighted

        if self.normalize_by_channels and self.total_channels > 0:
            total_loss = total_loss / self.total_channels

        total_key = f"loss/{self.prefix}" if self.prefix else "loss/total"
        loss_dict[total_key] = total_loss
        return total_loss, TensorDict(loss_dict)

    def __repr__(self) -> str:
        fields_str = ", ".join(f"{n}:{t}" for n, t in self.target_config.items())
        weights_str = ", ".join(
            f"{name}={w}" for name, w in self.field_weights.items() if w != 1.0
        )
        parts = [
            f"fields=[{fields_str}]",
            f"loss_type='{self.loss_type}'",
            f"measure_weighted={self.measure_weighted}",
        ]
        if weights_str:
            parts.append(f"field_weights={{{weights_str}}}")
        if self.prefix:
            parts.append(f"prefix='{self.prefix}'")
        return f"LossCalculator({', '.join(parts)})"
