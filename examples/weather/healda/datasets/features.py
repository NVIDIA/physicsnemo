# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Literal
import torch
import math
import healda.profiling


@healda.profiling.nvtx
def compute_unified_metadata(
    target_time_sec: torch.Tensor,  # int64 seconds
    lat: torch.Tensor,
    lon: torch.Tensor,
    time: torch.Tensor,  # int64 nanoseconds
    # Raw metadata fields
    height: torch.Tensor | None = None,
    pressure: torch.Tensor | None = None,
    scan_angle: torch.Tensor | None = None,
    sat_zenith_angle: torch.Tensor | None = None,
    sol_zenith_angle: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute unified metadata from raw fields.

    Features are concatenated in the following order:
    - Local solar time (4 features): Fourier encoding with 2 frequencies
    - Relative time features (2 features): normalized time difference and its square
    - Height features (8 features, NaN for satellite): Fourier encoding with 4 frequencies
    - Pressure features (8 features, NaN for satellite): Fourier encoding with 4 frequencies
    - Scan angle features (2 features, NaN for conventional): normalized scan angle and its square
    - Satellite zenith features (2 features, NaN for conventional): cos(θ_sat) and cos(θ_sat)²
    - Solar zenith features (2 features, NaN for conventional): cos(θ_sun) and sin(θ_sun)

    Note: time inputs use int64 to preserve precision. Float conversion happens only
    after magnitude reduction to avoid precision loss with large Unix timestamps.
    """
    device = lat.device
    n_obs = lat.shape[0]

    lst = local_solar_time(lon, time)

    # Build metadata features as a list
    metadata_features = []

    # Local solar time features (4 features)
    local_solar_time_feats = fourier_features(
        lst / 24.0, 2
    )  # 2 frequencies = 4 features
    metadata_features.append(local_solar_time_feats)

    # Relative time features (2 features)
    target_time_ns = target_time_sec * 1_000_000_000
    dt_sec = (time - target_time_ns).float() * 1e-9
    relative_time_hours = dt_sec / 3600.0
    dt_norm = relative_time_hours / 24.0  # Normalize
    time_norm_feats = torch.stack([dt_norm, dt_norm**2], dim=-1)
    metadata_features.append(time_norm_feats)

    # Height features (16 features, NaN for satellite)
    if height is not None:
        height_norm = normalize(
            height,
            "linear",
            100.0,  # height_min
            60000.0,  # height_max
            0.5,  # height_power
        )
        height_feats = fourier_features(height_norm, 4)  # 4 frequencies = 8 features
        metadata_features.append(height_feats)
    else:
        # Add NaN tensor for height features
        metadata_features.append(
            torch.full((n_obs, 8), float("nan"), device=device, dtype=torch.float32)
        )

    # Pressure features (16 features, NaN for satellite)
    if pressure is not None:
        pressure_norm = normalize(
            pressure,
            "linear",
            10.0,  # pressure_min
            1100.0,  # pressure_max
            3.0,  # pressure_power
        )
        pressure_feats = fourier_features(
            pressure_norm, 4
        )  # 4 frequencies = 8 features
        metadata_features.append(pressure_feats)
    else:
        # Add NaN tensor for pressure features
        metadata_features.append(
            torch.full((n_obs, 8), float("nan"), device=device, dtype=torch.float32)
        )

    # Scan angle features (2 features, NaN for conventional)
    if scan_angle is not None:
        xi_norm = scan_angle / 50.0  # ~[-1,1] as in existing code
        scan_angle_feats = torch.stack([xi_norm, xi_norm**2], dim=-1)
        metadata_features.append(scan_angle_feats)
    else:
        # Add NaN tensor for scan angle features
        metadata_features.append(
            torch.full((n_obs, 2), float("nan"), device=device, dtype=torch.float32)
        )

    # Satellite zenith features (2 features, NaN for conventional)
    if sat_zenith_angle is not None:
        cos_theta_sat = torch.cos(torch.deg2rad(sat_zenith_angle))
        sat_zenith_feats = torch.stack([cos_theta_sat, cos_theta_sat**2], dim=-1)
        metadata_features.append(sat_zenith_feats)
    else:
        # Add NaN tensor for satellite zenith features
        metadata_features.append(
            torch.full((n_obs, 2), float("nan"), device=device, dtype=torch.float32)
        )

    # Solar zenith features (2 features, NaN for conventional)
    if sol_zenith_angle is not None:
        cos_theta_sun = torch.cos(torch.deg2rad(sol_zenith_angle))
        sin_theta_sun = torch.sin(torch.deg2rad(sol_zenith_angle))
        sol_zenith_feats = torch.stack([cos_theta_sun, sin_theta_sun], dim=-1)
        metadata_features.append(sol_zenith_feats)
    else:
        # Add NaN tensor for solar zenith features
        metadata_features.append(
            torch.full((n_obs, 2), float("nan"), device=device, dtype=torch.float32)
        )

    # Concatenate all features
    metadata = torch.cat(metadata_features, dim=-1)
    metadata = metadata.nan_to_num(0.0)

    return metadata


def normalize(
    x: torch.Tensor,
    scale: Literal["linear", "log", "power"],
    x_min: float,
    x_max: float,
    power: float,
) -> torch.Tensor:
    # map x onto [0,1] using chosen scale
    if scale == "linear":
        return torch.clamp(x / x_max, 0.0, 1.0)
    elif scale == "log":
        # ensure positive
        return (torch.log(x + x_min) - math.log(x_min)) / (
            math.log(x_max + x_min) - math.log(x_min)
        )
    elif scale == "power":
        x_lin = torch.clamp(x / x_max, 0.0, 1.0)
        return x_lin.pow(power)
    else:
        raise ValueError(f"Unknown scale '{scale}'")


def fourier_features(x_norm: torch.Tensor, num_freqs: int) -> torch.Tensor:
    # x_norm: (N,) in [0,1]
    # produce (N, 2*num_freqs) of sin/cos features
    device = x_norm.device
    freqs = torch.arange(1, num_freqs + 1, device=device, dtype=x_norm.dtype) * (
        2 * math.pi
    )
    x_expanded = x_norm.unsqueeze(-1) * freqs  # (N, num_freqs)
    sin_feats = torch.sin(x_expanded)
    cos_feats = torch.cos(x_expanded)
    return torch.cat([sin_feats, cos_feats], dim=-1)


def local_solar_time(
    lon_deg: torch.Tensor,
    abs_time_ns: torch.Tensor,
) -> torch.Tensor:
    # Approximate without equation of time correction
    sec_of_day = (abs_time_ns // 1_000_000_000) % 86400
    utc_hours = sec_of_day.float() / 3600.0
    lst = (utc_hours + lon_deg / 15.0) % 24.0
    return lst
