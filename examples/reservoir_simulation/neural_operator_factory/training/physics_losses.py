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
Physics-informed loss functions for reservoir simulation modeling.

All losses are dimension-agnostic and work with both
3D (B, H, W, T) and 4D (B, X, Y, Z, T) predictions.

Available physics losses:
- mass_conservation: penalises discrepancies in spatially-integrated
  quantities between prediction and ground truth at each timestep
  (Chandra et al. 2025, arXiv:2503.11031, Eq. 4).

Grid convention
---------------
NOF datasets store cell widths (block sizes) as the last input channels:
- 2D: [..., grid_x, grid_y, grid_t] at channels [-3, -2, -1]
- 3D: [..., grid_x, grid_y, grid_z, grid_t] at channels [-4, -3, -2, -1]

Cell volumes are computed as the outer product of per-axis widths.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid / volume utilities
# ---------------------------------------------------------------------------


def _extract_grid_widths(inputs: Tensor, spatial_ndim: int) -> List[Tensor]:
    """Extract per-axis cell widths from the NOF last-channel convention.

    Returns one 1-D tensor of cell widths per spatial axis, in axis order:
    - 2D: [widths_H (grid_y), widths_W (grid_x)]
    - 3D: [widths_X (grid_x), widths_Y (grid_y), widths_Z (grid_z)]
    """
    if spatial_ndim == 2:
        widths_w = inputs[0, 0, :, 0, -3]  # (W,) x-direction cell widths
        widths_h = inputs[0, :, 0, 0, -2]  # (H,) y-direction cell widths
        return [widths_h, widths_w]  # axis-1, axis-2 order
    else:
        widths_x = inputs[0, :, 0, 0, 0, -4]  # (X,)
        widths_y = inputs[0, 0, :, 0, 0, -3]  # (Y,)
        widths_z = inputs[0, 0, 0, :, 0, -2]  # (Z,)
        return [widths_x, widths_y, widths_z]  # axis-1, axis-2, axis-3 order


def compute_cell_volumes_from_widths(inputs: Tensor, spatial_ndim: int) -> Tensor:
    """Compute per-cell volumes from the NOF grid-width channels.

    Returns (H, W) for 2D or (X, Y, Z) for 3D.
    """
    widths = _extract_grid_widths(inputs, spatial_ndim)
    if spatial_ndim == 2:
        return widths[0].unsqueeze(1) * widths[1].unsqueeze(0)
    else:
        return (
            widths[0].unsqueeze(1).unsqueeze(2)
            * widths[1].unsqueeze(0).unsqueeze(2)
            * widths[2].unsqueeze(0).unsqueeze(1)
        )


# ---------------------------------------------------------------------------
# Derivative utilities (used by losses.py for derivative regularization)
# ---------------------------------------------------------------------------

# {dim_name: (tensor_axis, grid_channel_offset_from_end)}
_DERIV_MAP_2D: Dict[str, Tuple[int, int]] = {
    "dx": (2, -3),  # W axis, grid_x channel
    "dy": (1, -2),  # H axis, grid_y channel
}
_DERIV_MAP_3D: Dict[str, Tuple[int, int]] = {
    "dx": (1, -4),  # X axis, grid_x channel
    "dy": (2, -3),  # Y axis, grid_y channel
    "dz": (3, -2),  # Z axis, grid_z channel
}


def get_deriv_map(spatial_ndim: int) -> Dict[str, Tuple[int, int]]:
    """Return {dim_name: (tensor_axis, channel_offset)} for the given dimensionality."""
    return _DERIV_MAP_2D if spatial_ndim == 2 else _DERIV_MAP_3D


def cell_centre_distance(cell_widths: Tensor, min_spacing: float = 1e-6) -> Tensor:
    """Distance between centres of cell i and cell i+2.

    d[i] = cell_widths[i]/2 + cell_widths[i+1] + cell_widths[i+2]/2

    A minimum floor is applied to prevent division-by-zero when grid
    widths are zero (e.g. inactive cells) or very small (normalized data).

    Parameters
    ----------
    cell_widths : Tensor  shape (N,)
    min_spacing : float   floor value for the output

    Returns
    -------
    Tensor  shape (N-2,)
    """
    d = cell_widths[:-2] / 2.0 + cell_widths[1:-1] + cell_widths[2:] / 2.0
    return d.clamp(min=min_spacing)


def central_difference(field: Tensor, axis: int, spacing: Tensor) -> Tensor:
    """Central-difference derivative along *axis*.

    Computes (f[i+2] - f[i]) / spacing[i] for each interior point.

    Parameters
    ----------
    field : Tensor   arbitrary-rank tensor
    axis : int       dimension along which to differentiate
    spacing : Tensor (N-2,) cell-centre distances

    Returns
    -------
    Tensor  with field.shape[axis] reduced by 2
    """
    n = field.shape[axis]
    f_right = field.narrow(axis, 2, n - 2)
    f_left = field.narrow(axis, 0, n - 2)

    shape = [1] * field.dim()
    shape[axis] = -1
    sp = spacing.reshape(shape)

    return (f_right - f_left) / sp


def extract_grid_widths_for_axis(
    inputs: Tensor,
    spatial_ndim: int,
    dim_name: str,
) -> Tensor:
    """Extract 1-D cell widths for a single derivative direction.

    Parameters
    ----------
    inputs : Tensor       full input tensor
    spatial_ndim : int    2 or 3
    dim_name : str        'dx', 'dy', or 'dz'

    Returns
    -------
    Tensor  1-D cell widths along the requested axis
    """
    dmap = get_deriv_map(spatial_ndim)
    if dim_name not in dmap:
        valid = list(dmap.keys())
        raise ValueError(
            f"Unknown derivative dim '{dim_name}' for {spatial_ndim}D. Valid: {valid}"
        )
    _axis, ch_offset = dmap[dim_name]
    # Extract widths along the correct spatial axis
    if spatial_ndim == 2:
        if dim_name == "dx":
            return inputs[0, 0, :, 0, ch_offset]
        else:
            return inputs[0, :, 0, 0, ch_offset]
    else:
        if dim_name == "dx":
            return inputs[0, :, 0, 0, 0, ch_offset]
        elif dim_name == "dy":
            return inputs[0, 0, :, 0, 0, ch_offset]
        else:
            return inputs[0, 0, 0, :, 0, ch_offset]


# ---------------------------------------------------------------------------
# Mass conservation loss
# ---------------------------------------------------------------------------


class MassConservationLoss(nn.Module):
    """Weak mass-conservation constraint via spatial integration.

    Parameters
    ----------
    use_cell_volumes : bool
        If True, compute cell volumes from the grid-width input channels
        (NOF convention).  If False, uniform weighting (volume = 1).
    eps : float
        Numerical stability constant.
    """

    VALID_METRICS = {"relative_l2", "mse", "l1", "huber"}

    def __init__(
        self,
        use_cell_volumes: bool = False,
        metric: str = "relative_l2",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.use_cell_volumes = use_cell_volumes
        if metric not in self.VALID_METRICS:
            raise ValueError(
                f"metric must be one of {self.VALID_METRICS}, got '{metric}'"
            )
        self.metric = metric
        self.eps = eps
        self._cell_volumes: Optional[Tensor] = None
        self._volumes_device: Optional[torch.device] = None

    def _get_cell_volumes(self, inputs: Tensor, spatial_ndim: int) -> Tensor:
        if self._cell_volumes is not None and self._volumes_device == inputs.device:
            return self._cell_volumes
        if self.use_cell_volumes:
            vol = compute_cell_volumes_from_widths(inputs, spatial_ndim)
        else:
            spatial_shape = inputs.shape[1 : 1 + spatial_ndim]
            vol = torch.ones(spatial_shape, device=inputs.device, dtype=inputs.dtype)
        self._cell_volumes = vol.detach()
        self._volumes_device = inputs.device
        return self._cell_volumes

    def forward(self, pred, target, inputs, spatial_mask=None):
        ndim = pred.dim()
        if ndim == 4:
            spatial_ndim, spatial_dims = 2, (1, 2)
        elif ndim == 5:
            spatial_ndim, spatial_dims = 3, (1, 2, 3)
        else:
            raise ValueError(
                f"Expected 4D (B,H,W,T) or 5D (B,X,Y,Z,T), got {ndim}D shape {tuple(pred.shape)}"
            )

        vol = self._get_cell_volumes(inputs, spatial_ndim)
        if spatial_mask is not None:
            # Reduce per-sample (B, *spatial) to union (*spatial)
            m = spatial_mask
            if m.dim() == pred.dim() - 1:
                m = m.any(dim=0)
            w = (vol * m.float()).unsqueeze(0).unsqueeze(-1)
        else:
            w = vol.unsqueeze(0).unsqueeze(-1)

        m_pred = (pred * w).sum(dim=spatial_dims)  # (B, T)
        m_true = (target * w).sum(dim=spatial_dims)  # (B, T)

        if self.metric == "relative_l2":
            diff_norm = torch.norm(m_true - m_pred, p=2, dim=-1)
            true_norm = torch.norm(m_true, p=2, dim=-1)
            per_sample = diff_norm / (true_norm + self.eps)
        elif self.metric == "mse":
            per_sample = ((m_true - m_pred) ** 2).mean(dim=-1)
        elif self.metric == "l1":
            per_sample = (m_true - m_pred).abs().mean(dim=-1)
        elif self.metric == "huber":
            per_sample = torch.nn.functional.smooth_l1_loss(
                m_pred, m_true, reduction="none"
            ).mean(dim=-1)

        return per_sample.mean()


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

_PHYSICS_LOSS_REGISTRY: Dict[str, type] = {
    "mass_conservation": MassConservationLoss,
}


def build_physics_losses(physics_config, variable=None, default_metric="relative_l2"):
    """Instantiate physics losses from the loss.physics config block."""
    if physics_config is None:
        return {}

    active: Dict[str, tuple] = {}
    for name in _PHYSICS_LOSS_REGISTRY:
        sub = physics_config.get(name, None)
        if sub is None or not sub.get("enabled", False):
            continue
        weight = float(sub.get("weight", 1.0))
        if weight <= 0:
            continue

        cls = _PHYSICS_LOSS_REGISTRY[name]
        kwargs: dict = {}

        if name == "mass_conservation":
            kwargs["use_cell_volumes"] = bool(sub.get("use_cell_volumes", False))
            kwargs["eps"] = float(sub.get("eps", 1e-8))
            metric = sub.get("metric", None)
            kwargs["metric"] = metric if metric is not None else default_metric
            if variable is not None and variable.lower() == "pressure":
                warnings.warn(
                    "mass_conservation loss is enabled for variable='pressure'. "
                    "Spatially-integrated pressure is not a conserved quantity; "
                    "this loss is physically meaningful only for mass-like "
                    "variables (e.g. saturation, CO2 mass).",
                    stacklevel=2,
                )

        loss_module = cls(**kwargs)
        active[name] = (loss_module, weight)
        logger.info("Physics loss '%s' enabled (weight=%.4g)", name, weight)

    return active


__all__ = [
    "MassConservationLoss",
    "compute_cell_volumes_from_widths",
    "cell_centre_distance",
    "central_difference",
    "get_deriv_map",
    "extract_grid_widths_for_axis",
    "build_physics_losses",
]
