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
Unified loss functions for reservoir simulation modeling.

Data-fitting losses: mse, l1, relative_l2, huber
Regularisation:      spatial derivative constraints (dimension-agnostic)
Physics losses:      mass conservation (and future additions)

All losses work with 2D spatial (B, H, W, T) and 3D spatial (B, X, Y, Z, T)
predictions, both full-mapping and autoregressive regimes.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.physics_losses import (
    build_physics_losses,
    cell_centre_distance,
    central_difference,
    extract_grid_widths_for_axis,
    get_deriv_map,
)

# ---------------------------------------------------------------------------
# Standalone convenience loss
# ---------------------------------------------------------------------------


class SimpleRelativeL2Loss(nn.Module):
    """Relative L2 loss without bells and whistles.

    loss = mean_b( ||pred_b - target_b||_2 / (||target_b||_2 + eps) )
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, predictions, targets, inputs=None, **kwargs):
        batch_size = predictions.shape[0]
        pred_flat = predictions.reshape(batch_size, -1)
        target_flat = targets.reshape(batch_size, -1)
        diff_norm = torch.norm(pred_flat - target_flat, p=2, dim=1)
        target_norm = torch.norm(target_flat, p=2, dim=1)
        return (diff_norm / (target_norm + self.eps)).mean()


# ---------------------------------------------------------------------------
# Unified loss
# ---------------------------------------------------------------------------


class UnifiedLoss(nn.Module):
    """Configurable loss with data-fitting, derivative, and physics terms.

    total = sum(w_i * data_loss_i)
          + derivative.weight * derivative_loss
          + sum(alpha_j * physics_loss_j)

    Data-fitting losses
    -------------------
    mse : Mean Squared Error, mean((pred - target)^2).
        Standard regression loss. Penalises large errors quadratically.
    l1 : Mean Absolute Error, mean(|pred - target|).
        More robust to outliers than MSE. Linear penalty.
    relative_l2 : ||pred - target||_2 / (||target||_2 + eps), per sample.
        Scale-invariant; recommended when output magnitude varies across
        samples (e.g. pressure with large dynamic range).
    huber : Smooth L1 / Huber loss (controlled by ``huber_delta``).
        Behaves like MSE for errors < delta and L1 for errors > delta.
        Combines MSE precision near zero with L1 robustness for outliers.

    Spatial derivative regularization
    ---------------------------------
    Penalises differences in spatial gradients between pred and target
    using central finite differences on the NOF grid-width channels.
    Configured via ``derivative_config`` with keys:

    - ``enabled`` (bool): toggle on/off.
    - ``weight`` (float): multiplier for the derivative term.
    - ``dims`` (list of str): which spatial directions to differentiate.
      2D data: ``dx`` (W/horizontal), ``dy`` (H/vertical).
      3D data: ``dx`` (X), ``dy`` (Y), ``dz`` (Z).
    - ``metric`` (str or None): loss metric for comparing derivatives.
      ``None`` inherits the first entry in ``types``.

    Masking
    -------
    When ``spatial_mask`` is provided (boolean tensor over spatial dims),
    MSE, L1, and Huber average only over active cells. relative_l2
    zeros out inactive cells before computing norms.

    Parameters
    ----------
    types : list of str
        Data loss types.
    weights : list of float
        Weights for each data loss type.
    huber_delta : float
        Transition threshold for Huber loss.
    derivative_config : dict or None
        Derivative regularization settings (see above).
    eps : float
        Epsilon for numerical stability (relative_l2 denominator).
    reduction : str
        'mean', 'sum', or 'none'.
    physics_losses : dict or None
        Pre-built physics losses: ``{name: (module, weight)}``.
        Built by :func:`training.physics_losses.build_physics_losses`.
    """

    VALID_TYPES = {"mse", "l1", "relative_l2", "huber"}

    def __init__(
        self,
        types=None,
        weights=None,
        huber_delta: float = 1.0,
        derivative_config=None,
        eps: float = 1e-6,
        reduction: str = "mean",
        physics_losses=None,
    ):
        super().__init__()

        if types is None:
            types = ["relative_l2"]
        if isinstance(types, str):
            types = [types]
        types = [t.lower() for t in types]

        if weights is None:
            weights = [1.0] * len(types)
        if isinstance(weights, (int, float)):
            weights = [float(weights)]
        weights = [float(w) for w in weights]

        if len(types) != len(weights):
            raise ValueError(
                f"types and weights must have same length, got {len(types)} vs {len(weights)}"
            )
        for t in types:
            if t not in self.VALID_TYPES:
                raise ValueError(
                    f"Loss type must be one of {self.VALID_TYPES}, got '{t}'"
                )
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )

        self.loss_types = types
        self.loss_weights = weights
        self.huber_delta = huber_delta
        self.eps = eps
        self.reduction = reduction

        # Derivative config
        self._deriv_cfg = derivative_config or {}
        self._deriv_enabled = self._deriv_cfg.get("enabled", False)
        self._deriv_weight = float(self._deriv_cfg.get("weight", 0.5))
        self._deriv_dims: List[str] = list(self._deriv_cfg.get("dims", ["dx"]))
        self._deriv_metric: Optional[str] = self._deriv_cfg.get("metric", None)

        # Physics losses
        self._physics_losses: Dict[str, tuple] = physics_losses or {}
        for name, (mod, _w) in self._physics_losses.items():
            self.add_module(f"physics_{name}", mod)

    # -----------------------------------------------------------------------
    # Data losses
    # -----------------------------------------------------------------------

    @staticmethod
    def _is_per_sample_mask(spatial_mask, pred):
        """True when mask has a leading batch dimension matching pred."""
        return (
            spatial_mask.dim() == pred.dim() - 1
            and spatial_mask.shape[0] == pred.shape[0]
        )

    @staticmethod
    def _expand_mask(spatial_mask, pred):
        """Expand spatial mask to match pred shape, per-sample aware.

        Accepts ``(*spatial)`` (one mask for all samples) or
        ``(B, *spatial)`` (per-sample masks).  Returns a boolean tensor
        broadcastable to ``pred`` shape ``(B, *spatial, T)``.
        """
        if UnifiedLoss._is_per_sample_mask(spatial_mask, pred):
            return spatial_mask.unsqueeze(-1).expand_as(pred)
        else:
            return spatial_mask.unsqueeze(0).unsqueeze(-1).expand_as(pred)

    def _compute_single_loss(self, pred, target, loss_type, spatial_mask=None):
        """Compute a single loss term with proper per-sample masking.

        When ``spatial_mask`` is provided, only active cells contribute
        to the loss.  For ``relative_l2``, active cells are selected
        per-sample so that norms are computed exclusively on active
        values (not diluted by zeros).  Supports both ``(*spatial)``
        and ``(B, *spatial)`` masks.
        """
        if loss_type in ("mse", "l1", "huber"):
            if loss_type == "mse":
                diff = (pred - target) ** 2
            elif loss_type == "l1":
                diff = torch.abs(pred - target)
            else:
                diff = F.smooth_l1_loss(
                    pred, target, reduction="none", beta=self.huber_delta
                )
            if spatial_mask is not None:
                mask_exp = self._expand_mask(spatial_mask, diff)
                diff = diff[mask_exp]
            return diff.mean() if self.reduction == "mean" else diff.sum()

        elif loss_type == "relative_l2":
            batch_size = pred.shape[0]
            losses = []
            for i in range(batch_size):
                if spatial_mask is not None:
                    if self._is_per_sample_mask(spatial_mask, pred):
                        m = spatial_mask[i]
                    else:
                        m = spatial_mask
                    m_exp = m.unsqueeze(-1).expand_as(pred[i])
                    p = pred[i][m_exp]
                    t = target[i][m_exp]
                else:
                    p = pred[i].reshape(-1)
                    t = target[i].reshape(-1)
                diff_norm = torch.norm(p - t, p=2)
                target_norm = torch.norm(t, p=2)
                losses.append(diff_norm / (target_norm + self.eps))
            loss = torch.stack(losses)
            return loss.mean() if self.reduction == "mean" else loss.sum()

        raise ValueError(f"Unknown loss type: {loss_type}")

    def _compute_base_loss(self, pred, target, spatial_mask=None):
        total = torch.tensor(0.0, device=pred.device)
        for loss_type, weight in zip(self.loss_types, self.loss_weights):
            total = total + weight * self._compute_single_loss(
                pred, target, loss_type, spatial_mask
            )
        return total

    # -----------------------------------------------------------------------
    # Derivative regularisation (dimension-agnostic)
    # -----------------------------------------------------------------------

    @staticmethod
    def _to_union_mask(spatial_mask, pred):
        """Reduce a per-sample mask ``(B, *spatial)`` to ``(*spatial)``.

        Takes the union (logical OR) across the batch so every cell
        active in any sample is included.  Static masks ``(*spatial)``
        are returned unchanged.
        """
        if spatial_mask is None:
            return None
        if UnifiedLoss._is_per_sample_mask(spatial_mask, pred):
            return spatial_mask.any(dim=0)
        return spatial_mask

    def _compute_derivative_loss(self, pred, target, inputs, spatial_mask=None):
        """Compute derivative loss along each configured direction.

        Uses the **union** mask for zeroing out inactive cells before
        differentiation (safe for the stencil) and for the stencil-safe
        derivative mask.  The derivative metric then receives this
        ``(*spatial)``-shaped mask for comparison.
        """
        ndim = pred.dim()
        spatial_ndim = ndim - 2
        if spatial_ndim not in (2, 3):
            raise ValueError(
                f"Derivative loss requires 2 or 3 spatial dims, got {spatial_ndim}"
            )

        union_mask = self._to_union_mask(spatial_mask, pred)

        deriv_metric = self._deriv_metric or self.loss_types[0]
        total = torch.tensor(0.0, device=pred.device)

        if union_mask is not None:
            mask_exp = union_mask.unsqueeze(0).unsqueeze(-1).expand_as(pred)
            pred = pred * mask_exp
            target = target * mask_exp

        for dim_name in self._deriv_dims:
            dmap = get_deriv_map(spatial_ndim)
            if dim_name not in dmap:
                valid = list(dmap.keys())
                raise ValueError(
                    f"Derivative dim '{dim_name}' not valid for {spatial_ndim}D data. Valid: {valid}"
                )
            tensor_axis, _ch_offset = dmap[dim_name]

            widths = extract_grid_widths_for_axis(inputs, spatial_ndim, dim_name)
            spacing = cell_centre_distance(widths)

            dy_pred = central_difference(pred, tensor_axis, spacing)
            dy_target = central_difference(target, tensor_axis, spacing)

            deriv_mask = None
            if union_mask is not None:
                mask_axis = tensor_axis - 1
                n = union_mask.shape[mask_axis]
                m_left = union_mask.narrow(mask_axis, 0, n - 2)
                m_centre = union_mask.narrow(mask_axis, 1, n - 2)
                m_right = union_mask.narrow(mask_axis, 2, n - 2)
                deriv_mask = m_left & m_centre & m_right

            total = total + self._compute_single_loss(
                dy_pred, dy_target, deriv_metric, spatial_mask=deriv_mask
            )

        return total / max(len(self._deriv_dims), 1)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, pred, target, inputs=None, spatial_mask=None):
        """Compute unified loss.

        Parameters
        ----------
        pred : Tensor
            (B, H, W, T) or (B, X, Y, Z, T)
        target : Tensor
            Same shape as pred.
        inputs : Tensor or None
            (B, *spatial, T, C). Required for derivative and physics losses.
        spatial_mask : Tensor or None
            (*spatial) boolean mask; active cells = True/1.
        """
        if self._deriv_enabled and inputs is None:
            raise ValueError("inputs required when derivative loss is enabled")

        # 1. Data loss
        data_loss = self._compute_base_loss(pred, target, spatial_mask)

        # 2. Derivative loss
        if self._deriv_enabled:
            deriv_loss = self._compute_derivative_loss(
                pred, target, inputs, spatial_mask
            )
            data_loss = data_loss + self._deriv_weight * deriv_loss

        # 3. Physics losses
        for mod, weight in self._physics_losses.values():
            phys = mod(pred, target, inputs, spatial_mask=spatial_mask)
            data_loss = data_loss + weight * phys

        return data_loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_loss_function(loss_config, variable=None):
    """Create a UnifiedLoss from a Hydra config."""
    types = loss_config.get("types", ["relative_l2"])
    weights = loss_config.get("weights", None)

    if hasattr(types, "__iter__") and not isinstance(types, str):
        types = list(types)
    if weights is not None and hasattr(weights, "__iter__"):
        weights = list(weights)

    # Derivative config
    deriv_cfg = loss_config.get("derivative", None)
    derivative_config = dict(deriv_cfg) if deriv_cfg is not None else {"enabled": False}

    # Physics losses
    physics_cfg = loss_config.get("physics", None)
    default_metric = types[0] if types else "relative_l2"
    physics_losses = build_physics_losses(
        physics_cfg, variable=variable, default_metric=default_metric
    )

    return UnifiedLoss(
        types=types,
        weights=weights,
        huber_delta=float(loss_config.get("huber_delta", 1.0)),
        derivative_config=derivative_config,
        eps=float(loss_config.get("eps", 1e-6)),
        reduction=loss_config.get("reduction", "mean"),
        physics_losses=physics_losses,
    )


__all__ = [
    "SimpleRelativeL2Loss",
    "UnifiedLoss",
    "get_loss_function",
]
