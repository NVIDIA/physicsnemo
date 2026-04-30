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

"""Transform framework + flux / coordinates / sampling / qoi transforms.

This module consolidates the RTE transform framework with the concrete
preprocessing transforms used by the Transolver pipeline. It is intentionally
flat (no submodules) so the standalone example can be read top-to-bottom:

* ``Transform`` and ``Compose`` are re-exports from PhysicsNeMo
  (``physicsnemo.datapipes.transforms``); RTE transforms subclass ``Transform``
  and operate on ``tensordict.TensorDict`` instances.
* ``TRANSFORM_REGISTRY`` is a module-level dict mapping the string names used
  by Hydra configs (``"RTEFluxLogClip"``, etc.) to the transform classes
  defined here. Sibling modules (e.g. ``material.py``) register their own
  transforms into the same dict.
* The ``@register(...)`` decorator from
  ``physicsnemo.datapipes.registry`` populates the global PhysicsNeMo registry;
  we apply it alongside the local registry for instantiation by either path.

Material transforms live in the sibling ``material.py``.
"""

from __future__ import annotations

# =========================================================================
# Imports
# =========================================================================

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Type, Union
import warnings

import numpy as np
import torch
import zarr
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms import Compose, Transform
from tensordict import NonTensorData, TensorDict


# =========================================================================
# Framework: Transform base, Compose, TensorDict utilities, registry
# =========================================================================
#
# ``Transform`` and ``Compose`` are imported above. RTE transforms subclass
# ``Transform`` and operate on ``TensorDict``. The TensorDict helpers below
# bridge numpy / torch / non-tensor (str, dict, None) values that flow through
# the pipeline. ``TRANSFORM_REGISTRY`` is the local string->class map; the
# ``@register(name)`` decorator additionally populates the PhysicsNeMo global
# registry so existing config-driven instantiation paths continue to work.


def td_from_dict(sample: Mapping[str, Any]) -> TensorDict:
    """Wrap a heterogeneous sample dict into a zero-batch-size ``TensorDict``.

    ``numpy`` arrays and ``torch`` tensors become tensor entries. Any other
    value (``None``, dict, str, Python scalar) is stored as ``NonTensorData``.
    Use bracket access (``td["key"]``) on the result to retrieve the original
    value; ``NonTensorData`` entries are transparently unwrapped.
    """
    out = TensorDict({}, batch_size=[])
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            out[key] = value
        elif isinstance(value, np.ndarray):
            out[key] = torch.from_numpy(np.ascontiguousarray(value))
        else:
            out.set_non_tensor(key, value)
    return out


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


# Local registry: maps the string name used in configs to the transform class.
# Sibling modules (e.g. ``material.py``) extend this dict at import time.
TRANSFORM_REGISTRY: Dict[str, Type[Transform]] = {}


def _register_local(name: str):
    """Combined decorator: register with PhysicsNeMo's global registry and
    record the class in ``TRANSFORM_REGISTRY`` under the same name."""

    pnm_register = register(name)

    def _decorator(cls):
        TRANSFORM_REGISTRY[name] = cls
        return pnm_register(cls)

    return _decorator


# =========================================================================
# Flux
# =========================================================================
#
# ``RTEFluxLogClip`` is the canonical pre-step that clamps flux and applies
# log10 before z-score normalization (the latter performed by
# ``physicsnemo.datapipes.transforms.Normalize``). ``FluxClipper`` and
# ``LogTransform`` are kept as small standalone utilities for notebooks.
# ``denormalize_flux`` inverts the full ``RTEFluxLogClip + Normalize`` chain
# for evaluation.


@_register_local("RTEFluxClipper")
class FluxClipper(Transform):
    """Clip flux below a threshold."""

    def __init__(self, threshold: float = 1e-8):
        super().__init__()
        self.threshold = threshold

    def __call__(self, data: TensorDict) -> TensorDict:
        data["scalar_flux"] = torch.clamp(data["scalar_flux"], min=self.threshold)
        return data

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"


@_register_local("RTELogTransform")
class LogTransform(Transform):
    """Log10 transformation for flux."""

    def __init__(self, offset: float = 1e-8):
        super().__init__()
        self.offset = offset

    def __call__(self, data: TensorDict) -> TensorDict:
        data["scalar_flux"] = torch.log10(data["scalar_flux"] + self.offset)
        return data

    def extra_repr(self) -> str:
        return f"offset={self.offset}"


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


@_register_local("RTEFluxLogClip")
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


@_register_local("RTEBackupCoords")
class RTEBackupCoords(Transform):
    """Clone ``coordinates`` into ``coordinates_unnormalized`` before Translate/Scale.

    Downstream consumers (e.g. graph construction or rasterization) read
    ``coordinates_unnormalized`` for physical-space operations. Place this
    transform immediately before
    ``physicsnemo.datapipes.transforms.Translate`` + ``Scale`` in the
    pipeline so the raw coords survive the normalization.

    Optionally also writes ``bbox_min`` / ``bbox_max`` tensors to the
    TensorDict so downstream code that previously read them from the legacy
    ``CoordinateNormalizer`` output still finds them.
    """

    def __init__(
        self,
        bbox_min: Optional[torch.Tensor] = None,
        bbox_max: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.bbox_min = (
            None if bbox_min is None else torch.as_tensor(bbox_min, dtype=torch.float32)
        )
        self.bbox_max = (
            None if bbox_max is None else torch.as_tensor(bbox_max, dtype=torch.float32)
        )

    def __call__(self, data: TensorDict) -> TensorDict:
        data["coordinates_unnormalized"] = data["coordinates"].clone()
        if self.bbox_min is not None:
            data["bbox_min"] = self.bbox_min.clone()
        if self.bbox_max is not None:
            data["bbox_max"] = self.bbox_max.clone()
        return data

    def extra_repr(self) -> str:
        if self.bbox_min is None:
            return "no bbox recorded"
        return f"bbox_min={self.bbox_min.tolist()}, bbox_max={self.bbox_max.tolist()}"


@_register_local("RTEFourierFeatures")
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


@_register_local("RTEStandardScaler")
class StandardScaler(Transform):
    """Z-score normalization for coordinates (utility, not used by default chain)."""

    def __init__(
        self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, data: TensorDict) -> TensorDict:
        coords = data["coordinates"]
        if self.mean is not None:
            mean_t = torch.as_tensor(
                self.mean, dtype=coords.dtype, device=coords.device
            )
        else:
            mean_t = coords.mean(dim=0)
        if self.std is not None:
            std_t = torch.as_tensor(self.std, dtype=coords.dtype, device=coords.device)
        else:
            std_t = coords.std(dim=0)
        std_t = torch.where(std_t < 1e-10, torch.ones_like(std_t), std_t)

        data["coordinates"] = (coords - mean_t) / std_t
        data["coord_mean"] = mean_t
        data["coord_std"] = std_t
        return data


# =========================================================================
# Sampling (spatial + temporal)
# =========================================================================
#
# ``SpatialSampler`` randomly subsamples / pads point clouds to a target size.
# ``TemporalSampler`` and its subclasses define the prediction task structure
# (next-step vs. steady-state).


@_register_local("RTESpatialSampler")
class SpatialSampler(Transform):
    """Sample spatial points from mesh.

    Supports random sampling, fixed N, and padding for variable mesh sizes.
    """

    def __init__(
        self,
        num_points: int,
        pad_value: float = -100.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.num_points = num_points
        self.pad_value = pad_value
        self.seed = seed
        self.rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )

    def __call__(self, data: TensorDict) -> TensorDict:
        num_available = data["coordinates"].shape[0]

        if self.num_points == -1:
            data.set_non_tensor("spatial_indices", None)
            data.set_non_tensor("spatial_num_original", num_available)
            return data

        needs_sampling = num_available > self.num_points

        if needs_sampling:
            indices_np = self.rng.choice(num_available, self.num_points, replace=False)
        else:
            if num_available == self.num_points:
                data.set_non_tensor("spatial_indices", None)
                data.set_non_tensor("spatial_num_original", num_available)
                return data
            indices_np = np.arange(num_available)

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
            if key in data:
                arr = data[key]
                if arr is None:
                    continue
                sampled = arr[indices]
                if sampled.shape[0] < self.num_points:
                    sampled = self._pad_tensor(sampled, self.num_points)
                data[key] = sampled

        if "scalar_flux" in data:
            flux = data["scalar_flux"][:, indices]  # (T, N_sampled)
            if flux.shape[1] < self.num_points:
                flux = self._pad_flux(flux, self.num_points)
            data["scalar_flux"] = flux

        for flux_key in ("flux_input", "flux_target"):
            if flux_key in data:
                flux_1d = data[flux_key][indices]
                if flux_1d.shape[0] < self.num_points:
                    flux_1d = self._pad_tensor(flux_1d, self.num_points)
                data[flux_key] = flux_1d

        data["spatial_indices"] = indices
        data.set_non_tensor("spatial_num_original", int(num_available))
        return data

    def _pad_tensor(self, tensor: torch.Tensor, target_size: int) -> torch.Tensor:
        if tensor.shape[0] >= target_size:
            return tensor[:target_size]
        pad_shape = list(tensor.shape)
        pad_shape[0] = target_size - tensor.shape[0]
        padding = torch.full(
            pad_shape, float(self.pad_value), dtype=tensor.dtype, device=tensor.device
        )
        return torch.cat([tensor, padding], dim=0)

    def _pad_flux(self, flux: torch.Tensor, target_size: int) -> torch.Tensor:
        if flux.shape[1] >= target_size:
            return flux[:, :target_size]
        pad_shape = (flux.shape[0], target_size - flux.shape[1])
        padding = torch.full(pad_shape, -10.0, dtype=flux.dtype, device=flux.device)
        return torch.cat([flux, padding], dim=1)

    def extra_repr(self) -> str:
        return f"num_points={self.num_points}"


class TemporalSampler(Transform):
    """Base class for temporal sampling strategies.

    Subclasses implement different prediction tasks:
    - NextStepSampler: Predict t+stride from t
    - SteadyStateSampler: Predict t=T from t=0
    """

    def select_time_indices(
        self, num_timesteps: int, rng: np.random.Generator
    ) -> Tuple[int, int]:
        raise NotImplementedError

    def __call__(self, data: TensorDict) -> TensorDict:
        """Apply temporal sampling; extracts input/target flux slices."""
        num_timesteps = data["scalar_flux"].shape[0]

        if "_timestep_idx" in data:
            original_idx = int(data["_timestep_idx"])

            metadata = td_get(data, "metadata", default={}) or {}
            max_timestep = (
                metadata.get("max_timestep") if isinstance(metadata, dict) else None
            )
            if max_timestep is not None:
                selective_loading_used = (max_timestep + 1) != num_timesteps
            else:
                selective_loading_used = original_idx >= num_timesteps

            if selective_loading_used:
                input_idx = 0
                target_idx = min(self.stride, num_timesteps - 1)
                data.set_non_tensor("timestep_input", original_idx)
                data.set_non_tensor("timestep_target", original_idx + self.stride)
            else:
                input_idx = original_idx
                target_idx = min(original_idx + self.stride, num_timesteps - 1)
                data.set_non_tensor("timestep_input", input_idx)
                data.set_non_tensor("timestep_target", target_idx)

            del data["_timestep_idx"]
        else:
            rng = np.random.default_rng()
            input_idx, target_idx = self.select_time_indices(num_timesteps, rng)
            data.set_non_tensor("timestep_input", int(input_idx))
            data.set_non_tensor("timestep_target", int(target_idx))

        flux_all = data["scalar_flux"]
        data["flux_input"] = flux_all[input_idx].clone()
        data["flux_target"] = flux_all[target_idx].clone()
        return data


@_register_local("RTENextStepSampler")
class NextStepSampler(TemporalSampler):
    """Sample for next-step prediction task.

    Selects random timestep t and predicts t+stride.
    """

    def __init__(self, stride: int = 1, seed: Optional[int] = None):
        super().__init__()
        self.stride = stride
        self.seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def select_time_indices(
        self, num_timesteps: int, rng: np.random.Generator
    ) -> Tuple[int, int]:
        if self.rng is not None:
            rng = self.rng

        max_start = num_timesteps - self.stride
        if max_start <= 0:
            warnings.warn(
                "Not enough timesteps to sample for next-step prediction. "
                "Using first and last timestep."
            )
            return 0, num_timesteps - 1

        input_idx = rng.integers(0, max_start)
        target_idx = input_idx + self.stride
        return input_idx, target_idx

    def extra_repr(self) -> str:
        return f"stride={self.stride}"


@_register_local("RTESteadyStateSampler")
class SteadyStateSampler(TemporalSampler):
    """Sample for steady state prediction task (t=0 input, t=T target)."""

    def __init__(self):
        super().__init__()

    def select_time_indices(
        self, num_timesteps: int, rng: np.random.Generator
    ) -> Tuple[int, int]:
        return 0, num_timesteps - 1


# =========================================================================
# QoI loader
# =========================================================================
#
# Loads ground-truth QoI values for the target timestep from each sample's
# zarr ``global_metrics`` array. Lattice writes a single scalar; hohlraum
# writes the three regional cumulated fluxes plus a ``ground_truth_qoi``
# alias pointing at the center value (used by the loss).


@_register_local("RTELoadGroundTruthQoI")
class LoadGroundTruthQoI(Transform):
    """Load ground truth QoI for the target timestep from zarr global_metrics."""

    def __init__(self, data_path: Union[str, Path]):
        super().__init__()
        self.data_path = Path(data_path)

    def __call__(self, data: TensorDict) -> TensorDict:
        filename = td_get(data, "filename")
        if filename is None:
            raise ValueError(
                "Sample is missing 'filename' field required for QoI loading"
            )

        timestep_target_idx = td_get(data, "timestep_target")
        if timestep_target_idx is None:
            raise ValueError(f"Sample from {filename} missing timestep_target field")

        zarr_path = self.data_path / filename
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr file not found: {zarr_path}")

        z = zarr.open(str(zarr_path), mode="r")
        if "global_metrics" not in z:
            raise KeyError(f"'global_metrics' not found in zarr file: {zarr_path}")

        global_metrics = np.array(z["global_metrics"], dtype=np.float32)
        timesteps_full = np.array(z["timesteps"])

        metadata = td_get(data, "metadata", default={}) or {}
        case_type = metadata.get("case_type") if isinstance(metadata, dict) else None
        if case_type is None:
            raise ValueError("metadata.case_type is required for QoI computation")

        timestep_target_idx = int(timestep_target_idx)
        if timestep_target_idx >= len(timesteps_full):
            raise ValueError(
                f"timestep_target index {timestep_target_idx} out of bounds for "
                f"full timesteps array of length {len(timesteps_full)} in {filename}"
            )

        global_metrics_idx = timestep_target_idx
        if global_metrics_idx >= global_metrics.shape[0]:
            raise IndexError(
                f"QoI index {global_metrics_idx} out of bounds for global_metrics "
                f"shape {global_metrics.shape} in {filename}"
            )

        if case_type == "lattice":
            data["ground_truth_qoi"] = torch.tensor(
                float(global_metrics[global_metrics_idx, 1]), dtype=torch.float32
            )
        elif case_type == "hohlraum":
            center = float(global_metrics[global_metrics_idx, 1])
            vertical = float(global_metrics[global_metrics_idx, 2])
            horizontal = float(global_metrics[global_metrics_idx, 3])
            data["ground_truth_qoi_cumulated_center"] = torch.tensor(
                center, dtype=torch.float32
            )
            data["ground_truth_qoi_cumulated_vertical"] = torch.tensor(
                vertical, dtype=torch.float32
            )
            data["ground_truth_qoi_cumulated_horizontal"] = torch.tensor(
                horizontal, dtype=torch.float32
            )
            data["ground_truth_qoi"] = torch.tensor(center, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown case type: {case_type}")

        return data

    def extra_repr(self) -> str:
        return f"data_path={self.data_path}"


# =========================================================================
# Public API
# =========================================================================

__all__ = [
    # Framework
    "Transform",
    "Compose",
    "TRANSFORM_REGISTRY",
    "td_from_dict",
    "td_get",
    "to_numpy",
    "NonTensorData",
    # Flux
    "RTEFluxLogClip",
    "FluxClipper",
    "LogTransform",
    "denormalize_flux",
    # Coordinates
    "GLOBAL_DOMAIN_BOUNDS",
    "RTEBackupCoords",
    "FourierFeatures",
    "StandardScaler",
    # Sampling
    "SpatialSampler",
    "TemporalSampler",
    "NextStepSampler",
    "SteadyStateSampler",
    # QoI
    "LoadGroundTruthQoI",
]
