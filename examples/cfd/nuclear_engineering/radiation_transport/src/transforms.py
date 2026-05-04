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

"""Transform framework + flux / coordinates / sampling transforms.

This module consolidates the RTE transform framework with the concrete
preprocessing transforms used by the Transolver pipeline. It is intentionally
flat (no submodules) so the standalone example can be read top-to-bottom:

* ``Transform`` and ``Compose`` are re-exports from PhysicsNeMo
  (``physicsnemo.datapipes.transforms``); RTE transforms subclass ``Transform``
  and operate on ``tensordict.TensorDict`` instances.
* The ``@register(...)`` decorator from
  ``physicsnemo.datapipes.registry`` populates the global PhysicsNeMo registry;
  we apply it for config-driven instantiation if needed.

Material transforms live in the sibling ``material.py``.
"""

from __future__ import annotations

# =========================================================================
# Imports
# =========================================================================

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms import Transform
from tensordict import TensorDict


# =========================================================================
# Framework: Transform base + TensorDict utilities
# =========================================================================
#
# ``Transform`` is imported above. RTE transforms subclass it and operate on
# ``TensorDict``. The TensorDict helpers below
# bridge numpy / torch / non-tensor (str, dict, None) values that flow through
# the pipeline.


def td_get(data: TensorDict, key: str, default: Any = None) -> Any:
    """``td[key]``-equivalent lookup with a default for missing keys.

    ``TensorDict.get`` returns the raw ``NonTensorData`` wrapper; bracket access
    unwraps it but raises ``KeyError`` for missing keys. This helper combines
    both semantics.
    """
    if key in data:
        return data[key]
    return default


def to_numpy(value: Any) -> np.ndarray:
    """Coerce a torch tensor / numpy array / array-like into a numpy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# =========================================================================
# Flux
# =========================================================================
#
# ``RTEFluxLogClip`` is the canonical pre-step that clamps flux and applies
# log10 before z-score normalization (the latter performed by
# ``physicsnemo.datapipes.transforms.Normalize``). ``denormalize_flux`` inverts
# the full ``RTEFluxLogClip + Normalize`` chain for evaluation.


def denormalize_flux(
    normalized_flux: torch.Tensor,
    stats: Dict[str, float],
) -> torch.Tensor:
    """Invert the ``RTEFluxLogClip + Normalize`` chain for evaluation/inference.

    ``normalized_flux`` is the model output in z-score-of-log space;
    ``stats`` is the ``flux_normalization_stats`` dict that ``RTEFluxLogClip``
    recorded on the sample.
    """
    mean = stats["log_flux_mean"]
    std = stats["log_flux_std"]
    clip = stats["clip_threshold"]
    log_flux = normalized_flux * std + mean
    log_flux = torch.clamp(log_flux, min=-300, max=300)
    flux = torch.pow(10.0, log_flux) - clip
    return torch.clamp(flux, min=0.0)


@register("RTEFluxLogClip")
class RTEFluxLogClip(Transform):
    """Clip flux to a threshold, apply ``log10``, and record denorm stats.

    Input:
        ``scalar_flux`` -- shape ``(T, N)`` or ``(N,)``, float tensor.

    Output:
        ``scalar_flux`` -- same shape, ``log10(clamp(x, clip) + clip)``.
        ``flux_normalization_stats`` -- non-tensor dict with ``log_flux_mean``,
        ``log_flux_std``, ``clip_threshold`` for downstream denormalization.

    Args:
        clip_threshold: minimum flux value before log.
        log_flux_mean / log_flux_std: stats to record for denorm; if a
            ``normalization_stats_file`` is provided these are read from it.
        normalization_stats_file: optional path to the RTE flux stats YAML
            (``load_flux_stats``). When provided, overrides the inline args
            and validates ``clip_threshold`` against the file's value.
    """

    def __init__(
        self,
        clip_threshold: float = 1e-8,
        log_flux_mean: Optional[float] = None,
        log_flux_std: Optional[float] = None,
        normalization_stats_file: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()

        if normalization_stats_file is not None:
            # Imported lazily to avoid a circular import: ``dataset.py`` itself
            # may pull symbols from this module via the flat-import shim.
            from dataset import load_flux_stats

            stats = load_flux_stats(normalization_stats_file)
            if abs(stats["clip_threshold"] - clip_threshold) > 1e-10:
                raise ValueError(
                    f"Clip threshold mismatch: got {clip_threshold}, "
                    f"stats computed with {stats['clip_threshold']}"
                )
            self.clip_threshold = float(stats["clip_threshold"])
            self.log_flux_mean = float(stats["log_flux_mean"])
            self.log_flux_std = float(stats["log_flux_std"])
        elif log_flux_mean is not None and log_flux_std is not None:
            self.clip_threshold = float(clip_threshold)
            self.log_flux_mean = float(log_flux_mean)
            self.log_flux_std = float(log_flux_std)
        else:
            raise ValueError(
                "Either normalization_stats_file or (log_flux_mean, log_flux_std) "
                "must be provided."
            )

    def __call__(self, data: TensorDict) -> TensorDict:
        flux = data["scalar_flux"]
        clip = torch.tensor(self.clip_threshold, dtype=flux.dtype, device=flux.device)
        flux = torch.clamp(flux, min=clip)
        data["scalar_flux"] = torch.log10(flux + clip)
        data.set_non_tensor(
            "flux_normalization_stats",
            {
                "log_flux_mean": self.log_flux_mean,
                "log_flux_std": self.log_flux_std,
                "clip_threshold": self.clip_threshold,
            },
        )
        return data

    def extra_repr(self) -> str:
        return (
            f"clip_threshold={self.clip_threshold}, "
            f"log_flux_mean={self.log_flux_mean:.4f}, "
            f"log_flux_std={self.log_flux_std:.4f}"
        )


# =========================================================================
# Coordinates (Fourier features)
# =========================================================================
#
# The default coordinate-normalization chain is
# ``[RTEBackupCoords, Translate, Scale]`` (the latter two are stock PhysicsNeMo
# transforms). ``GLOBAL_DOMAIN_BOUNDS`` is the canonical per-case bbox table
# referenced from the config-build site and direct consumers.
# ``FourierFeatures`` has no PhysicsNeMo equivalent and stays custom.


GLOBAL_DOMAIN_BOUNDS = {
    "lattice": {
        "min": np.array([-3.5, -3.5, -0.01], dtype=np.float32),
        "max": np.array([3.5, 3.5, 0.01], dtype=np.float32),
    },
    "hohlraum": {
        "min": np.array([-0.65, -0.65, -0.01], dtype=np.float32),
        "max": np.array([0.65, 0.65, 0.01], dtype=np.float32),
    },
}


@register("RTEBackupCoords")
class RTEBackupCoords(Transform):
    """Clone ``coordinates`` into ``coordinates_unnormalized`` before Translate/Scale.

    Downstream consumers (e.g. graph construction or rasterization) read
    ``coordinates_unnormalized`` for physical-space operations. Place this
    transform immediately before
    ``physicsnemo.datapipes.transforms.Translate`` + ``Scale`` in the
    pipeline so the raw coords survive the normalization.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: TensorDict) -> TensorDict:
        data["coordinates_unnormalized"] = data["coordinates"].clone()
        return data

    def extra_repr(self) -> str:
        return "preserve raw coordinates"


@register("RTEFourierFeatures")
class FourierFeatures(Transform):
    """Sin/cos positional encoding features at multiple frequency scales."""

    def __init__(
        self,
        num_frequencies: int = 3,
        coord_dims: int = 2,
        base_frequency: float = 1.0,
        append_to_coordinates: bool = True,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.coord_dims = coord_dims
        self.base_frequency = base_frequency
        self.append_to_coordinates = append_to_coordinates
        self.frequency_multipliers = [
            2**i * base_frequency for i in range(num_frequencies)
        ]

    def get_output_dim(self) -> int:
        return 2 * self.num_frequencies * self.coord_dims

    def __call__(self, data: TensorDict) -> TensorDict:
        coords = data["coordinates"]
        coords_subset = coords[:, : self.coord_dims].to(dtype=torch.float32)

        two_pi = 2.0 * np.pi
        parts = []
        for freq_mult in self.frequency_multipliers:
            angle = two_pi * float(freq_mult) * coords_subset
            parts.append(torch.sin(angle))
            parts.append(torch.cos(angle))

        fourier_features = torch.cat(parts, dim=-1).to(dtype=torch.float32)
        data["fourier_features"] = fourier_features

        if self.append_to_coordinates:
            data["coordinates"] = torch.cat(
                [coords.to(dtype=torch.float32), fourier_features], dim=-1
            )
        return data

    def extra_repr(self) -> str:
        return (
            f"num_frequencies={self.num_frequencies}, coord_dims={self.coord_dims}, "
            f"base_frequency={self.base_frequency}, "
            f"append_to_coordinates={self.append_to_coordinates}"
        )


# =========================================================================
# Sampling (spatial + steady-state)
# =========================================================================
#
# ``SpatialSampler`` randomly subsamples point clouds to a target size.
# ``SteadyStateSampler`` extracts the fixed initial->final flux mapping.


@register("RTESpatialSampler")
class SpatialSampler(Transform):
    """Randomly subsample spatial points to ``num_points``.

    ``num_points = -1`` is a passthrough. Otherwise ``num_available`` must be
    ``>= num_points`` (the shipped lattice / hohlraum meshes have tens of
    thousands of cells, far above any practical ``num_points``).
    """

    def __init__(self, num_points: int, seed: Optional[int] = None):
        super().__init__()
        self.num_points = num_points
        self.seed = seed
        self.rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )

    def __call__(self, data: TensorDict) -> TensorDict:
        if self.num_points == -1:
            return data

        num_available = data["coordinates"].shape[0]
        if num_available == self.num_points:
            return data
        if num_available < self.num_points:
            raise ValueError(
                f"SpatialSampler: num_available={num_available} < "
                f"num_points={self.num_points}; the shipped meshes are larger "
                "than any configured num_points, so this should never happen."
            )

        indices_np = self.rng.choice(num_available, self.num_points, replace=False)
        indices = torch.from_numpy(indices_np.astype(np.int64))

        spatial_keys = [
            "coordinates",
            "cell_areas",
            "material_properties",
            "physical_properties",
            "geometric_features",
            "sigma_t",
            "sigma_s",
            "sigma_a",
            "Q",
        ]
        for key in spatial_keys:
            if key in data and data[key] is not None:
                data[key] = data[key][indices]

        if "scalar_flux" in data:
            data["scalar_flux"] = data["scalar_flux"][:, indices]

        for flux_key in ("flux_input", "flux_target"):
            if flux_key in data:
                data[flux_key] = data[flux_key][indices]

        return data

    def extra_repr(self) -> str:
        return f"num_points={self.num_points}"


@register("RTESteadyStateSampler")
class SteadyStateSampler(Transform):
    """Extract the fixed steady-state mapping: first flux -> final flux."""

    def __init__(self):
        super().__init__()

    def __call__(self, data: TensorDict) -> TensorDict:
        flux_all = data["scalar_flux"]
        if flux_all.shape[0] == 0:
            raise ValueError("scalar_flux must contain at least one snapshot")

        input_idx = 0
        target_idx = flux_all.shape[0] - 1
        metadata = td_get(data, "metadata", default={}) or {}
        max_timestep = (
            metadata.get("max_timestep") if isinstance(metadata, dict) else None
        )

        data["flux_input"] = flux_all[input_idx].clone()
        data["flux_target"] = flux_all[target_idx].clone()
        data.set_non_tensor("timestep_input", 0)
        data.set_non_tensor(
            "timestep_target",
            int(max_timestep) if max_timestep is not None else int(target_idx),
        )
        return data

# =========================================================================
# Public API
# =========================================================================

__all__ = [
    # Framework
    "Transform",
    "td_get",
    "to_numpy",
    # Flux
    "RTEFluxLogClip",
    "denormalize_flux",
    # Coordinates
    "GLOBAL_DOMAIN_BOUNDS",
    "RTEBackupCoords",
    "FourierFeatures",
    # Sampling
    "SpatialSampler",
    "SteadyStateSampler",
]
