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

- ``"scalar"`` : single mean over all elements (matches per-element loss).
- ``"vector"`` : per-component mean, summed across components (matches
                  the legacy per-component summing convention).

Per-field weights (``field_weights``) replace the legacy implicit equal
weighting. Each per-field loss is multiplied by ``field_weights[name]``
(default 1.0) before summation. The total is normalized by the total
channel count (sum of per-field dims) when ``normalize_by_channels=True``,
preserving the previous total-loss scale when all weights are 1.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from utils import align_scalar_shapes, field_dim

DEFAULT_HUBER_DELTA = 1.0

LossType = Literal["huber", "mse", "rmse"]


### ---------------------------------------------------------------------------
### Per-field loss kernels
### ---------------------------------------------------------------------------


def _scalar_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: LossType,
    delta: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Element-wise loss reduced to a scalar (matches legacy scalar behavior)."""
    if loss_type == "huber":
        return F.huber_loss(pred, target, reduction="mean", delta=delta)
    if loss_type == "mse":
        return torch.mean((pred - target) ** 2)
    if loss_type == "rmse":
        num = torch.mean((pred - target) ** 2)
        denom = torch.mean(target**2)
        return num / (denom + eps)
    raise ValueError(f"Unknown loss_type {loss_type!r}")


def _vector_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: LossType,
    delta: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-component scalar loss summed across components.

    Matches the legacy ``compute_huber_vector`` / ``compute_mse_vector`` /
    ``compute_relative_mse`` semantics: for a vector field of dimension
    ``D``, the result is ``D * mean_huber_over_all_elements`` (or the MSE /
    RMSE analogue), not a single mean over the flattened tensor.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shapes must match, got {tuple(pred.shape)} vs "
            f"{tuple(target.shape)}"
        )
    n_components = pred.shape[-1]

    if loss_type == "rmse":
        ### Per-component relative MSE, summed.
        diff_sq = torch.mean((pred - target) ** 2, dim=tuple(range(pred.ndim - 1)))
        target_sq = torch.mean(target**2, dim=tuple(range(pred.ndim - 1)))
        return torch.sum(diff_sq / (target_sq + eps))

    total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    for i in range(n_components):
        p, t = pred[..., i], target[..., i]
        if loss_type == "huber":
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
        loss_type: One of ``"huber"``, ``"mse"``, ``"rmse"``.
        n_spatial_dims: Vector field dimensionality. Used to compute
            channel counts for the normalization denominator.
        field_weights: Optional per-field multiplicative weights. Each
            per-field loss is multiplied by ``field_weights[name]`` before
            summation. Default 1.0 for any unspecified name.
        prefix: Optional prefix for the keys in the returned loss dict
            (e.g. ``"surface"`` produces ``"loss/surface/pressure"``).
        normalize_by_channels: When ``True`` (default), divide the
            (weighted) total loss by ``sum(per_field_dims)``. Matches the
            legacy normalization semantics when all weights are 1.

    The returned loss dict contains one entry per field
    (``"loss/[prefix/]<name>"``) plus ``"loss/total"`` (or
    ``"loss/<prefix>"`` when ``prefix`` is set).
    """

    def __init__(
        self,
        target_config: dict[str, str],
        loss_type: LossType = "huber",
        n_spatial_dims: int = 3,
        field_weights: dict[str, float] | None = None,
        prefix: str = "",
        normalize_by_channels: bool = True,
        delta: float = DEFAULT_HUBER_DELTA,
    ) -> None:
        if loss_type not in ("huber", "mse", "rmse"):
            raise ValueError(
                f"Unknown loss_type {loss_type!r}; expected one of "
                f"'huber', 'mse', 'rmse'."
            )
        ### Normalize types to lowercase up front so per-call branches can
        ### compare with literal "scalar" / "vector" without re-lowering.
        self.target_config = {k: v.lower() for k, v in target_config.items()}
        self.loss_type = loss_type
        self.n_spatial_dims = n_spatial_dims
        self.prefix = prefix
        self.normalize_by_channels = normalize_by_channels
        self.delta = delta

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

    def _make_key(self, *parts: str) -> str:
        segments = ["loss"]
        if self.prefix:
            segments.append(self.prefix)
        segments.extend(parts)
        return "/".join(segments)

    def __call__(
        self,
        pred: TensorDict,
        target: TensorDict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute per-field losses and a (weighted) total.

        Args:
            pred: TensorDict of predictions, one leaf per target field.
                Per-element scalars are shape ``(..., N)``, per-element
                vectors are ``(..., N, D)``. Leading batch dims are
                arbitrary; the loss kernels reduce over them.
            target: TensorDict of the same structure as ``pred``.

        Returns:
            ``(total_loss, loss_dict)``. ``loss_dict`` has one entry per
            field (``"loss/[prefix/]<name>"``) plus a total entry.
        """
        pred_keys = set(pred.keys())
        target_keys = set(target.keys())
        missing_pred = set(self.target_config) - pred_keys
        missing_target = set(self.target_config) - target_keys
        if missing_pred:
            raise KeyError(f"pred is missing target fields {sorted(missing_pred)!r}")
        if missing_target:
            raise KeyError(
                f"target is missing target fields {sorted(missing_target)!r}"
            )

        ### Find a tensor we can use to seed the accumulator's dtype/device.
        any_pred = next(iter(pred.values()))
        total_loss = torch.zeros((), device=any_pred.device, dtype=any_pred.dtype)
        loss_dict: dict[str, torch.Tensor] = {}

        for name, field_type in self.target_config.items():
            p, t = pred[name], target[name]
            if field_type == "scalar":
                ### Caller may pass scalar fields as (..., 1) or (...);
                ### normalize to a single shape so the loss is shape-agnostic.
                p, t = align_scalar_shapes(p, t)
                field_loss = _scalar_loss(p, t, self.loss_type, self.delta)
            else:  # vector
                field_loss = _vector_loss(p, t, self.loss_type, self.delta)

            weighted = field_loss * self.field_weights[name]
            loss_dict[self._make_key(name)] = weighted
            total_loss = total_loss + weighted

        if self.normalize_by_channels and self.total_channels > 0:
            total_loss = total_loss / self.total_channels

        total_key = f"loss/{self.prefix}" if self.prefix else "loss/total"
        loss_dict[total_key] = total_loss
        return total_loss, loss_dict

    def __repr__(self) -> str:
        fields_str = ", ".join(f"{n}:{t}" for n, t in self.target_config.items())
        weights_str = ", ".join(
            f"{name}={w}" for name, w in self.field_weights.items() if w != 1.0
        )
        parts = [f"fields=[{fields_str}]", f"loss_type='{self.loss_type}'"]
        if weights_str:
            parts.append(f"field_weights={{{weights_str}}}")
        if self.prefix:
            parts.append(f"prefix='{self.prefix}'")
        return f"LossCalculator({', '.join(parts)})"
