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
from typing import Literal
import math
import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import fast_sinf as fsin, fast_cosf as fcos, isnan

N_FEATURES = 28


def _compute_unified_metadata_reference(
    target_time_sec: torch.Tensor,  # int64 seconds
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
    device = lon.device
    n_obs = lon.shape[0]

    lst = local_solar_time(lon, time)

    # Build metadata features as a list
    metadata_features = []

    # Local solar time features (4 features)
    local_solar_time_features = fourier_features(
        lst / 24.0, 2
    )  # 2 frequencies = 4 features
    metadata_features.append(local_solar_time_features)

    # Relative time features (2 features)
    target_time_ns = target_time_sec * 1_000_000_000
    dt_sec = (time - target_time_ns).float() * 1e-9
    relative_time_hours = dt_sec / 3600.0
    dt_norm = relative_time_hours / 24.0  # Normalize
    time_norm_features = torch.stack([dt_norm, dt_norm**2], dim=-1)
    metadata_features.append(time_norm_features)

    # Height features (8 features, NaN for satellite)
    if height is not None:
        height_norm = normalize(
            height,
            "linear",
            100.0,  # height_min
            60000.0,  # height_max
            0.5,  # height_power
        )
        height_features = fourier_features(height_norm, 4)  # 4 frequencies = 8 features
        metadata_features.append(height_features)
    else:
        # Add NaN tensor for height features
        metadata_features.append(
            torch.full((n_obs, 8), float("nan"), device=device, dtype=torch.float32)
        )

    # Pressure features (8 features, NaN for satellite)
    if pressure is not None:
        pressure_norm = normalize(
            pressure,
            "linear",
            10.0,  # pressure_min
            1100.0,  # pressure_max
            3.0,  # pressure_power
        )
        pressure_features = fourier_features(
            pressure_norm, 4
        )  # 4 frequencies = 8 features
        metadata_features.append(pressure_features)
    else:
        # Add NaN tensor for pressure features
        metadata_features.append(
            torch.full((n_obs, 8), float("nan"), device=device, dtype=torch.float32)
        )

    # Scan angle features (2 features, NaN for conventional)
    if scan_angle is not None:
        xi_norm = scan_angle / 50.0  # ~[-1,1] as in existing code
        scan_angle_features = torch.stack([xi_norm, xi_norm**2], dim=-1)
        metadata_features.append(scan_angle_features)
    else:
        # Add NaN tensor for scan angle features
        metadata_features.append(
            torch.full((n_obs, 2), float("nan"), device=device, dtype=torch.float32)
        )

    # Satellite zenith features (2 features, NaN for conventional)
    if sat_zenith_angle is not None:
        cos_theta_sat = torch.cos(torch.deg2rad(sat_zenith_angle))
        sat_zenith_features = torch.stack([cos_theta_sat, cos_theta_sat**2], dim=-1)
        metadata_features.append(sat_zenith_features)
    else:
        metadata_features.append(
            torch.full((n_obs, 2), float("nan"), device=device, dtype=torch.float32)
        )

    # Solar zenith features (2 features, NaN for conventional)
    if sol_zenith_angle is not None:
        cos_theta_sun = torch.cos(torch.deg2rad(sol_zenith_angle))
        sin_theta_sun = torch.sin(torch.deg2rad(sol_zenith_angle))
        sol_zenith_features = torch.stack([cos_theta_sun, sin_theta_sun], dim=-1)
        metadata_features.append(sol_zenith_features)
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
    sin_features = torch.sin(x_expanded)
    cos_features = torch.cos(x_expanded)
    return torch.cat([sin_features, cos_features], dim=-1)


def local_solar_time(
    lon_deg: torch.Tensor,
    abs_time_ns: torch.Tensor,
) -> torch.Tensor:
    # Approximate without equation of time correction
    sec_of_day = (abs_time_ns // 1_000_000_000) % 86400
    utc_hours = sec_of_day.float() / 3600.0
    lst = (utc_hours + lon_deg / 15.0) % 24.0
    return lst


#########################################################
# Triton implementations
#########################################################


@triton.jit
def _fourier_store(out_ptr, base, offset, angle, valid, m, NUM_FREQS: tl.constexpr):
    """Store fourier features [sin(k*angle), cos(k*angle)] for k=1..NUM_FREQS, zeroed when !valid."""
    for k in tl.static_range(1, NUM_FREQS + 1):
        tl.store(
            out_ptr + base + offset + k - 1,
            tl.where(valid, fsin(angle * k), 0.0),
            mask=m,
        )
        tl.store(
            out_ptr + base + offset + NUM_FREQS + k - 1,
            tl.where(valid, fcos(angle * k), 0.0),
            mask=m,
        )


@triton.jit
def _metadata_kernel(
    lon_ptr,
    time_ptr,
    target_ptr,
    height_ptr,
    press_ptr,
    scan_ptr,
    sat_ptr,
    sol_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    """Compute unified metadata using a single Triton kernel.
    Using torch.compile on compute_unified_metadata runs into issues with torch dynamo when using DistributedDataParallel.
    Seems compiling a function that isn't in main network/main thread does not work.
    """

    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < N

    lon = tl.load(lon_ptr + off, mask=m, other=0.0).to(tl.float32)
    time_ns = tl.load(time_ptr + off, mask=m, other=0)
    target_s = tl.load(target_ptr + off, mask=m, other=0)
    height = tl.load(height_ptr + off, mask=m, other=0.0).to(tl.float32)
    pressure = tl.load(press_ptr + off, mask=m, other=0.0).to(tl.float32)
    scan = tl.load(scan_ptr + off, mask=m, other=0.0).to(tl.float32)
    sat_zen = tl.load(sat_ptr + off, mask=m, other=0.0).to(tl.float32)
    sol_zen = tl.load(sol_ptr + off, mask=m, other=0.0).to(tl.float32)

    # Fields are NaN when the observation type doesn't carry that metadata
    # (e.g. satellite obs lack height/pressure, conventional obs lack zenith angles).
    height_valid = ~isnan(height)
    pressure_valid = ~isnan(pressure)
    scan_valid = ~isnan(scan)
    sat_zen_valid = ~isnan(sat_zen)
    sol_zen_valid = ~isnan(sol_zen)

    TWO_PI: tl.constexpr = 6.283185307179586
    DEG2RAD: tl.constexpr = 0.017453292519943295
    base = off * 28
    idx = 0

    # ======== Local Solar Time — fourier(2) -> 4 features ========
    sod = (time_ns // 1000000000) % 86400
    utc_hr = sod.to(tl.float32) / 3600.0
    lst = (utc_hr + lon / 15.0) % 24.0
    lst_angle = lst / 24.0 * TWO_PI
    _fourier_store(out_ptr, base, idx, lst_angle, True, m, 2)
    idx += 4  # 2 freqs * 2 (sin+cos)

    # ======== Relative Time -> 2 features ========
    dt_days = (time_ns - target_s * 1000000000).to(tl.float32) * 1e-9 / 86400.0
    tl.store(out_ptr + base + idx, dt_days, mask=m)
    tl.store(out_ptr + base + idx + 1, dt_days * dt_days, mask=m)
    idx += 2

    # ======== Height — fourier(4) -> 8 features ========
    height_angle = tl.where(
        height_valid, tl.minimum(tl.maximum(height / 60000.0, 0.0), 1.0) * TWO_PI, 0.0
    )
    _fourier_store(out_ptr, base, idx, height_angle, height_valid, m, 4)
    idx += 8  # 4 freqs * 2

    # ======== Pressure — fourier(4) -> 8 features ========
    press_angle = tl.where(
        pressure_valid,
        tl.minimum(tl.maximum(pressure / 1100.0, 0.0), 1.0) * TWO_PI,
        0.0,
    )
    _fourier_store(out_ptr, base, idx, press_angle, pressure_valid, m, 4)
    idx += 8  # 4 freqs * 2

    # ======== Scan Angle -> 2 features ========
    scan_norm = tl.where(scan_valid, scan / 50.0, 0.0)
    tl.store(out_ptr + base + idx, tl.where(scan_valid, scan_norm, 0.0), mask=m)
    tl.store(
        out_ptr + base + idx + 1,
        tl.where(scan_valid, scan_norm * scan_norm, 0.0),
        mask=m,
    )
    idx += 2

    # ======== Satellite Zenith -> 2 features ========
    cos_sat = fcos(tl.where(sat_zen_valid, sat_zen * DEG2RAD, 0.0))
    tl.store(out_ptr + base + idx, tl.where(sat_zen_valid, cos_sat, 0.0), mask=m)
    tl.store(
        out_ptr + base + idx + 1,
        tl.where(sat_zen_valid, cos_sat * cos_sat, 0.0),
        mask=m,
    )
    idx += 2

    # ======== Solar Zenith -> 2 features ========
    sol_rad = tl.where(sol_zen_valid, sol_zen * DEG2RAD, 0.0)
    tl.store(out_ptr + base + idx, tl.where(sol_zen_valid, fcos(sol_rad), 0.0), mask=m)
    tl.store(
        out_ptr + base + idx + 1, tl.where(sol_zen_valid, fsin(sol_rad), 0.0), mask=m
    )
    idx += 2


def compute_unified_metadata(
    target_time_sec: torch.Tensor,
    time: torch.Tensor,
    lon: torch.Tensor,
    height: torch.Tensor,
    pressure: torch.Tensor,
    scan_angle: torch.Tensor,
    sat_zenith_angle: torch.Tensor,
    sol_zenith_angle: torch.Tensor,
) -> torch.Tensor:
    if not lon.is_cuda:
        return _compute_unified_metadata_reference(
            target_time_sec,
            lon=lon,
            time=time,
            height=height,
            pressure=pressure,
            scan_angle=scan_angle,
            sat_zenith_angle=sat_zenith_angle,
            sol_zenith_angle=sol_zenith_angle,
        )
    N = lon.shape[0]
    out = torch.empty(N, N_FEATURES, dtype=torch.float32, device=lon.device)
    if N == 0:
        return out
    BLOCK = 256
    grid = ((N + BLOCK - 1) // BLOCK,)
    _metadata_kernel[grid](
        lon,
        time,
        target_time_sec,
        height,
        pressure,
        scan_angle,
        sat_zenith_angle,
        sol_zenith_angle,
        out,
        N,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out
