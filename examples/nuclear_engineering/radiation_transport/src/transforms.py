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

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms import Transform
from tensordict import TensorDict

__all__ = [
    "Transform",
    "RTEFluxLogClip",
    "denormalize_flux",
    "GLOBAL_DOMAIN_BOUNDS",
    "RTEBackupCoords",
    "FourierFeatures",
    "coord_bounds_for_case",
    "coord_translate_scale_params",
    "MaterialPropertyExtractor",
    "SpatialSampler",
    "FinalTimeSampler",
]


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
    log_flux = torch.clamp(log_flux, min=-38, max=38)
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
    """

    def __init__(
        self,
        clip_threshold: float,
        log_flux_mean: float,
        log_flux_std: float,
    ) -> None:
        super().__init__()
        self.clip_threshold = float(clip_threshold)
        self.log_flux_mean = float(log_flux_mean)
        self.log_flux_std = float(log_flux_std)

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


GLOBAL_DOMAIN_BOUNDS = {
    "lattice": {
        "min": torch.tensor([-3.5, -3.5], dtype=torch.float32),
        "max": torch.tensor([3.5, 3.5], dtype=torch.float32),
    },
    "hohlraum": {
        "min": torch.tensor([-0.65, -0.65], dtype=torch.float32),
        "max": torch.tensor([0.65, 0.65], dtype=torch.float32),
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
        """Number of Fourier-feature channels emitted (``2 * num_frequencies * coord_dims``)."""
        return 2 * self.num_frequencies * self.coord_dims

    def __call__(self, data: TensorDict) -> TensorDict:
        coords = data["coordinates"]
        coords_subset = coords[:, : self.coord_dims].to(dtype=torch.float32)

        two_pi = 2.0 * math.pi
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


@register("RTESpatialSampler")
class SpatialSampler(Transform):
    """Randomly subsample spatial points to ``num_points``.

    ``num_points = -1`` is a passthrough. Otherwise ``num_available`` must be
    ``>= num_points`` (the shipped lattice / hohlraum meshes have tens of
    thousands of cells, far above any practical ``num_points``).
    """

    # Stride used when re-seeding per epoch; large prime keeps streams disjoint.
    _EPOCH_PRIME: int = 1_000_003

    def __init__(self, num_points: int, seed: Optional[int] = None):
        super().__init__()
        self.num_points = num_points
        self.seed = seed
        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(int(seed))

    def set_epoch(self, epoch: int) -> None:
        """Re-seed the generator for a new epoch (deterministic reshuffle).

        No-op when ``self.seed`` is ``None`` (caller opted into a non-deterministic
        run; preserve current generator state).
        """
        if self.seed is None:
            return
        self.gen.manual_seed(int(self.seed) + int(epoch) * self._EPOCH_PRIME)

    def to(self, device):
        """No-op device move. ``self.gen`` stays pinned to CPU because
        ``torch.randperm`` requires its generator and output to share a
        device; selected indices are moved inside ``__call__``.
        """
        return self

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

        indices = torch.randperm(num_available, generator=self.gen)[: self.num_points]
        indices = indices.to(torch.int64).to(data["coordinates"].device)

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


@register("RTEFinalTimeSampler")
class FinalTimeSampler(Transform):
    """Extract the fixed final-time mapping: first flux -> final flux."""

    def __init__(self):
        super().__init__()

    def __call__(self, data: TensorDict) -> TensorDict:
        flux_all = data["scalar_flux"]
        if flux_all.shape[0] == 0:
            raise ValueError("scalar_flux must contain at least one snapshot")

        input_idx = 0
        target_idx = flux_all.shape[0] - 1

        data["flux_input"] = flux_all[input_idx].clone()
        data["flux_target"] = flux_all[target_idx].clone()
        data.set_non_tensor("timestep_input", 0)
        data.set_non_tensor("timestep_target", int(target_idx))
        return data


@register("RTEMaterialPropertyExtractor")
class MaterialPropertyExtractor(Transform):
    """Stack precomputed sigma fields into a per-cell ``(N, 4)`` tensor.

    Q must be present in the source data; it may be all-zero for source-free
    regimes (e.g., hohlraum).
    """

    def __call__(self, data: TensorDict) -> TensorDict:
        for key in ("sigma_a", "sigma_s", "sigma_t", "Q"):
            if key not in data:
                raise KeyError(
                    f"Mesh store is missing required field {key!r}. "
                    "All four fields (sigma_a, sigma_s, sigma_t, Q) must be precomputed."
                )

        data["physical_properties"] = torch.stack(
            [data["sigma_a"], data["sigma_s"], data["sigma_t"], data["Q"]],
            dim=-1,
        ).to(dtype=torch.float32)
        return data


def coord_bounds_for_case(case_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(bbox_min, bbox_max)`` as float32 tensors for a known case."""
    if case_type not in GLOBAL_DOMAIN_BOUNDS:
        raise ValueError(
            f"Unknown case_type '{case_type}'. "
            f"Expected one of: {list(GLOBAL_DOMAIN_BOUNDS.keys())}"
        )
    bounds = GLOBAL_DOMAIN_BOUNDS[case_type]
    return (
        torch.as_tensor(bounds["min"], dtype=torch.float32),
        torch.as_tensor(bounds["max"], dtype=torch.float32),
    )


def coord_translate_scale_params(
    case_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(center, half_extent)`` for ``Translate`` + ``Scale``.

    Returns the tensors so the caller can wire them straight into
    ``Translate(center_key_or_value=center, subtract=True)`` followed by
    ``Scale(scale=half_extent, divide=True)`` — i.e. the standard
    ``(x - center) / half_extent`` normalization into ``[-1, 1]``.
    """
    bbox_min, bbox_max = coord_bounds_for_case(case_type)
    center = 0.5 * (bbox_min + bbox_max)
    half_extent = 0.5 * (bbox_max - bbox_min)
    return center, half_extent
