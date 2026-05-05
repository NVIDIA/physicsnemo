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
import math
import torch

from physicsnemo.core.version_check import OptionalImport

triton = OptionalImport("triton")
_libdevice = OptionalImport("triton.language.extra.libdevice")

N_FEATURES = 30
# Layout:
#   [0:4)   LST fourier(2)                          — shared
#   [4:6)   relative-time  [dt_days, dt_days²]      — shared (polynomial)
#   [6:8)   relative-time  fourier(1)               — shared (cyclic)
#   [8:10)  latitude  [sin(lat), cos(lat)]           — shared
#   [10:30) BRANCH (conv OR sat):
#     conv: height fourier(5) [10:20) + pressure fourier(5) [20:30)
#     sat:  scan fourier(3) [10:16) + sat_zen fourier(4) [16:24) + sol_zen fourier(3) [24:30)
#
# Normalization conventions (normalize to [0,1] by physical max):
#   height:  h / 60000  -> [0, 1]
#   pressure: p / 1100  -> [0, 1]
#   scan:    ξ / 50     -> ~[-1, 1]   (data range: ~[-50, +50] deg)
#   sat_zen: θ / 90     -> ~[-0.67, +0.67]  (data range: ~[-60, +60] deg, signed)
#   sol_zen: θ / 180    -> ~[0.06, 0.94]    (data range: ~[10, 170] deg)


def _compute_unified_metadata_reference(
    target_time_sec: torch.Tensor,
    lon: torch.Tensor,
    lat: torch.Tensor,
    time: torch.Tensor,
    height: torch.Tensor,
    pressure: torch.Tensor,
    scan_angle: torch.Tensor,
    sat_zenith_angle: torch.Tensor,
    sol_zenith_angle: torch.Tensor,
) -> torch.Tensor:
    """Reference (CPU-friendly) implementation of unified metadata v2.

    Conv/sat specialization: height validity determines which branch fills
    slots 10-29.  Every slot carries signal — no zero padding.
    """
    device = lon.device
    n_obs = lon.shape[0]
    out = torch.zeros(n_obs, N_FEATURES, dtype=torch.float32, device=device)

    if n_obs == 0:
        return out

    is_conv = ~torch.isnan(height)

    TWO_PI = 2 * math.pi

    # --- shared: LST fourier(2) -> 4 ---
    lst = local_solar_time(lon, time)
    out[:, 0:4] = fourier_features(lst / 24.0 * TWO_PI, 2)

    # --- shared: relative time polynomial -> 2 ---
    target_time_ns = target_time_sec * 1_000_000_000
    dt_days = (time - target_time_ns).float() * 1e-9 / 86400.0
    out[:, 4] = dt_days
    out[:, 5] = dt_days**2

    # --- shared: relative time fourier(1) -> 2 ---
    out[:, 6:8] = fourier_features(dt_days, 1)

    # --- shared: latitude -> 2 ---
    lat_rad = torch.deg2rad(lat)
    out[:, 8] = torch.sin(lat_rad)
    out[:, 9] = torch.cos(lat_rad)

    # --- conv branch: height fourier(5) + pressure fourier(5) ---
    if is_conv.any():
        c = is_conv
        h_norm = torch.clamp(height[c] / 60000.0, 0.0, 1.0)
        out[c, 10:20] = fourier_features(h_norm * TWO_PI, 5)
        p_norm = torch.clamp(pressure[c] / 1100.0, 0.0, 1.0)
        out[c, 20:30] = fourier_features(p_norm * TWO_PI, 5)

    # --- sat branch: scan fourier(3) + sat_zen fourier(4) + sol_zen fourier(3) ---
    is_sat = ~is_conv
    if is_sat.any():
        s = is_sat
        out[s, 10:16] = fourier_features(scan_angle[s] / 50.0 * TWO_PI, 3)
        out[s, 16:24] = fourier_features(sat_zenith_angle[s] / 90.0 * TWO_PI, 4)
        out[s, 24:30] = fourier_features(sol_zenith_angle[s] / 180.0 * TWO_PI, 3)

    return out


def fourier_features(x_norm: torch.Tensor, num_freqs: int) -> torch.Tensor:
    device = x_norm.device
    freqs = torch.arange(1, num_freqs + 1, device=device, dtype=x_norm.dtype)
    x_expanded = x_norm.unsqueeze(-1) * freqs
    sin_features = torch.sin(x_expanded)
    cos_features = torch.cos(x_expanded)
    return torch.cat([sin_features, cos_features], dim=-1)


def local_solar_time(
    lon_deg: torch.Tensor,
    abs_time_ns: torch.Tensor,
) -> torch.Tensor:
    sec_of_day = (abs_time_ns // 1_000_000_000) % 86400
    utc_hours = sec_of_day.float() / 3600.0
    lst = (utc_hours + lon_deg / 15.0) % 24.0
    return lst


#########################################################
# Triton implementations
#########################################################

if triton.available:
    tl = triton.language
    fsin = _libdevice.fast_sinf
    fcos = _libdevice.fast_cosf
    isnan = _libdevice.isnan

    @triton.jit
    def _fourier_store(out_ptr, base, offset, x_norm, valid, m, NUM_FREQS: tl.constexpr):
        """Store sin(kx), cos(kx) for k=1..NUM_FREQS.  Matches fourier_features()."""
        for k in tl.static_range(1, NUM_FREQS + 1):
            angle = x_norm * k
            tl.store(
                out_ptr + base + offset + k - 1,
                tl.where(valid, fsin(angle), 0.0),
                mask=m,
            )
            tl.store(
                out_ptr + base + offset + NUM_FREQS + k - 1,
                tl.where(valid, fcos(angle), 0.0),
                mask=m,
            )

    @triton.jit
    def _metadata_kernel(
        lat_ptr,
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
        N_FEAT: tl.constexpr,
    ):
        """Extended observation metadata — 30 features per observation.

        Compared to the standard 28-feature encoding (obs_features.py):
        - Adds latitude encoding (sin/cos) as shared features [8:10).
        - Uses Fourier encoding for relative time instead of raw polynomial only.
        - Replaces NaN-padded conv/sat columns with mutually exclusive branches:
          slots 10-29 are written by exactly one branch per row (conv or sat,
          selected by NaN in height), so every feature carries signal.

        Triton implementation because torch.compile on the equivalent
        `_compute_unified_metadata_reference()` hits dynamo errors under
        multi-gpu DDP training (compiling a function in a non-main thread).
        """
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        m = off < N

        lat = tl.load(lat_ptr + off, mask=m, other=0.0).to(tl.float32)
        lon = tl.load(lon_ptr + off, mask=m, other=0.0).to(tl.float32)
        time_ns = tl.load(time_ptr + off, mask=m, other=0)
        target_s = tl.load(target_ptr + off, mask=m, other=0)
        height = tl.load(height_ptr + off, mask=m, other=0.0).to(tl.float32)
        pressure = tl.load(press_ptr + off, mask=m, other=0.0).to(tl.float32)
        scan = tl.load(scan_ptr + off, mask=m, other=0.0).to(tl.float32)
        sat_zen = tl.load(sat_ptr + off, mask=m, other=0.0).to(tl.float32)
        sol_zen = tl.load(sol_ptr + off, mask=m, other=0.0).to(tl.float32)

        is_conv = ~isnan(height)
        m_conv = m & is_conv
        m_sat = m & ~is_conv

        DEG2RAD: tl.constexpr = 0.017453292519943295
        base = off * N_FEAT
        idx = 0
        TWO_PI: tl.constexpr = 6.283185307179586
        # ======== Shared: LST fourier(2) -> 4 features ========
        sod = (time_ns // 1000000000) % 86400
        utc_hr = sod.to(tl.float32) / 3600.0
        lst = (utc_hr + lon / 15.0) % 24.0
        lst_norm = lst / 24.0
        _fourier_store(out_ptr, base, idx, lst_norm * TWO_PI, True, m, 2)
        idx += 4

        # ======== Shared: Relative time polynomial -> 2 features ========
        dt_days = (time_ns - target_s * 1000000000).to(tl.float32) * 1e-9 / 86400.0
        tl.store(out_ptr + base + idx, dt_days, mask=m)
        tl.store(out_ptr + base + idx + 1, dt_days * dt_days, mask=m)
        idx += 2

        # ======== Shared: Relative time fourier(1) -> 2 features ========
        _fourier_store(out_ptr, base, idx, dt_days, True, m, 1)
        idx += 2

        # ======== Shared: Latitude -> 2 features ========
        lat_rad = lat * DEG2RAD
        tl.store(out_ptr + base + idx, fsin(lat_rad), mask=m)
        tl.store(out_ptr + base + idx + 1, fcos(lat_rad), mask=m)
        idx += 2

        # ======== Conv branch [idx:idx+20) — guarded by m_conv ========
        branch = idx
        h_norm = tl.minimum(tl.maximum(height / 60000.0, 0.0), 1.0) * TWO_PI
        _fourier_store(out_ptr, base, branch, h_norm, True, m_conv, 5)
        branch += 10

        p_norm = tl.minimum(tl.maximum(pressure / 1100.0, 0.0), 1.0) * TWO_PI
        _fourier_store(out_ptr, base, branch, p_norm, True, m_conv, 5)

        # ======== Sat branch [idx:idx+20) — guarded by m_sat ========
        branch = idx
        scan_norm = scan / 50.0 * TWO_PI
        _fourier_store(out_ptr, base, branch, scan_norm, True, m_sat, 3)
        branch += 6

        sat_norm = sat_zen / 90.0 * TWO_PI
        _fourier_store(out_ptr, base, branch, sat_norm, True, m_sat, 4)
        branch += 8

        sol_norm = sol_zen / 180.0 * TWO_PI
        _fourier_store(out_ptr, base, branch, sol_norm, True, m_sat, 3)


def compute_unified_metadata(
    target_time_sec: torch.Tensor,
    time: torch.Tensor,
    lon: torch.Tensor,
    lat: torch.Tensor,
    height: torch.Tensor,
    pressure: torch.Tensor,
    scan_angle: torch.Tensor,
    sat_zenith_angle: torch.Tensor,
    sol_zenith_angle: torch.Tensor,
) -> torch.Tensor:
    """Compute unified metadata features (v2) for observations.

    Args:
        target_time_sec: Target time in seconds since epoch, shape (N,)
        time: Observation time in nanoseconds, shape (N,)
        lon: Longitude in degrees, shape (N,)
        lat: Latitude in degrees, shape (N,)
        height: Height in meters (NaN for satellite obs), shape (N,)
        pressure: Pressure in hPa (NaN for satellite obs), shape (N,)
        scan_angle: Scan angle in degrees (NaN for conv obs), shape (N,)
        sat_zenith_angle: Satellite zenith angle in degrees (NaN for conv obs), shape (N,)
        sol_zenith_angle: Solar zenith angle in degrees (NaN for conv obs), shape (N,)

    Returns:
        Tensor of shape (N, 30) with unified metadata features.
    """
    # Validate input shapes
    N = lon.shape[0]
    for name, tensor in [
        ("target_time_sec", target_time_sec),
        ("time", time),
        ("lat", lat),
        ("height", height),
        ("pressure", pressure),
        ("scan_angle", scan_angle),
        ("sat_zenith_angle", sat_zenith_angle),
        ("sol_zenith_angle", sol_zenith_angle),
    ]:
        if tensor.shape[0] != N:
            raise ValueError(f"{name} has length {tensor.shape[0]}, expected {N}")

    if not lon.is_cuda or not triton.available:
        return _compute_unified_metadata_reference(
            target_time_sec,
            lon=lon,
            lat=lat,
            time=time,
            height=height,
            pressure=pressure,
            scan_angle=scan_angle,
            sat_zenith_angle=sat_zenith_angle,
            sol_zenith_angle=sol_zenith_angle,
        )

    out = torch.empty(N, N_FEATURES, dtype=torch.float32, device=lon.device)
    if N == 0:
        return out
    BLOCK = 256
    grid = ((N + BLOCK - 1) // BLOCK,)
    _metadata_kernel[grid](
        lat,
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
        N_FEAT=N_FEATURES,
        num_warps=4,
    )
    return out
