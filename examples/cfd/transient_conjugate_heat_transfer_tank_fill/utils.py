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
"""Shared helpers for transient conjugate heat-transfer training/inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import DictConfig


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------


def resolve_path(path_like: str | os.PathLike[str], base: Path | None = None) -> Path:
    """Resolve a path relative to a base directory (default: caller's dir)."""
    base_dir = base or Path(__file__).resolve().parent
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


# -----------------------------------------------------------------------------
# Specs / shapes
# -----------------------------------------------------------------------------


def count_variable_components(spec_map: Dict[str, str]) -> int:
    """Count the number of components in a variable specification."""
    total = 0
    for kind in spec_map.values():
        total += 3 if kind == "vector" else 1
    return total


def count_global_features(cfg: DictConfig, future_steps: int) -> int:
    """Count the number of global features in a configuration."""
    base = 0
    for param_cfg in cfg.variables.global_parameters.values():
        if param_cfg.type == "vector":
            ref = param_cfg.reference
            base += len(ref) if isinstance(ref, (list, tuple)) else 1
        else:
            base += 1
    if not getattr(cfg.model, "encode_parameters", False):
        return base
    return base


def component_layout(spec_map: Dict[str, str]) -> List[Tuple[str, int]]:
    """Return [(name, width), ...] for scalar/vector specs."""
    return [(name, 3 if kind == "vector" else 1) for name, kind in spec_map.items()]


def infer_shapes(
    sample_path: Path, cfg: DictConfig, include_surface: bool, include_volume: bool
) -> Tuple[int, int, int]:
    """Infer output channels and future steps from a sample npz."""
    surface_channels = volume_channels = future_steps = 0
    with np.load(sample_path) as data:
        if include_surface and "surface_fields" in data:
            surface_channels = int(data["surface_fields"].shape[1])
            comp_surface = max(
                1, count_variable_components(cfg.variables.surface.solution)
            )
            future_steps = surface_channels // comp_surface
        if include_volume and "volume_fields" in data:
            volume_channels = int(data["volume_fields"].shape[1])
            comp_volume = max(
                1, count_variable_components(cfg.variables.volume.solution)
            )
            if future_steps == 0:
                future_steps = volume_channels // comp_volume
    return surface_channels, volume_channels, future_steps


# -----------------------------------------------------------------------------
# Scaling helpers
# -----------------------------------------------------------------------------


def load_scaling(
    stats_dir: Path,
    include_surface: bool,
    include_volume: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load scaling factors for volume/surface fields (precomputed stats required)."""
    surf_path = stats_dir / "surface_scaling_factors.npy"
    vol_path = stats_dir / "volume_scaling_factors.npy"
    surf = np.load(surf_path) if include_surface else None
    vol = np.load(vol_path) if include_volume else None
    return vol, surf


# -----------------------------------------------------------------------------
# Torch helpers
# -----------------------------------------------------------------------------


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """Compute the masked mean squared error between predicted and target tensors."""
    mask_t = (
        torch.ones_like(target, dtype=pred.dtype)
        if mask is None
        else mask.to(dtype=pred.dtype)
    )
    denom = torch.clamp(mask_t.sum(), min=1.0)
    return ((pred - target) * mask_t).pow(2).sum() / denom


def to_torch_sample(
    sample: Dict[str, np.ndarray], device: torch.device
) -> Dict[str, torch.Tensor]:
    """Convert a sample dictionary to a dictionary of PyTorch tensors."""
    torch_sample: Dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if value is None:
            continue
        tensor = torch.as_tensor(value, device=device)
        if key in {"stl_faces"}:
            tensor = tensor.to(torch.int32)
        torch_sample[key] = tensor
    return torch_sample


# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------


def split_fields_by_step(
    fields: np.ndarray,
    layout: List[Tuple[str, int]],
    num_steps: int,
) -> List[Dict[str, np.ndarray]]:
    """Split concatenated channels into per-step dicts following layout."""
    if fields.ndim != 2:
        raise ValueError(f"Expected 2D array for fields, got shape {fields.shape}")
    comps_per_step = sum(width for _, width in layout)
    available_steps = fields.shape[1] // max(1, comps_per_step)
    steps = min(num_steps, available_steps)
    trimmed = fields[:, : comps_per_step * steps]
    reshaped = trimmed.reshape(fields.shape[0], steps, comps_per_step)
    outputs: List[Dict[str, np.ndarray]] = []
    for step_idx in range(steps):
        step_slice = reshaped[:, step_idx, :]
        offset = 0
        step_fields: Dict[str, np.ndarray] = {}
        for name, width in layout:
            chunk = step_slice[:, offset : offset + width]
            offset += width
            step_fields[name] = chunk if width > 1 else chunk.squeeze(-1)
        outputs.append(step_fields)
    return outputs
