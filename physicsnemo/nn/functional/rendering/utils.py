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
from collections.abc import Sequence

import torch
import warp as wp


def _normalize_torch(vector: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


def _as_vec3(
    value: torch.Tensor | Sequence[float], *, name: str, device
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        value = value.to(device=device, dtype=torch.float32, non_blocking=True)
    else:
        value = torch.tensor(value, device=device, dtype=torch.float32)
    if value.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {tuple(value.shape)}")
    return value


def _optional_tensor_arg(value: torch.Tensor | Sequence[float] | None, *, device):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    return torch.as_tensor(value, device=device)


def _camera_basis(
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    *,
    device,
) -> torch.Tensor:
    eye = _as_vec3(eye, name="eye", device=device)
    center = _as_vec3(center, name="center", device=device)
    up = _as_vec3(up, name="up", device=device)
    forward_raw = center - eye
    if bool((forward_raw.norm() <= 1.0e-12).item()):
        raise ValueError("eye and center must not be equal")
    forward = _normalize_torch(forward_raw)
    up_hint = _normalize_torch(up)
    right_raw = torch.linalg.cross(up_hint, forward, dim=0)
    if bool((right_raw.norm() <= 1.0e-12).item()):
        raise ValueError("up must not be parallel to the camera direction")
    right = _normalize_torch(right_raw)
    camera_up = _normalize_torch(torch.linalg.cross(forward, right, dim=0))
    return torch.stack([eye, forward, right, camera_up]).contiguous()


def _bounds_tensor(
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    *,
    device,
) -> torch.Tensor:
    bounds_min = _as_vec3(bounds_min, name="bounds_min", device=device)
    bounds_max = _as_vec3(bounds_max, name="bounds_max", device=device)
    if bool(torch.any(bounds_max <= bounds_min).item()):
        raise ValueError("bounds_max must be greater than bounds_min in all dimensions")
    return torch.stack([bounds_min, bounds_max]).contiguous()


def _color_tensor(
    color: torch.Tensor | None,
    *,
    device,
    shape_name: str,
    expected_rank: int,
) -> torch.Tensor:
    if color is None:
        return torch.zeros((1,) * (expected_rank - 1) + (4,), device=device)
    if color.ndim != expected_rank or color.shape[-1] not in (3, 4):
        raise ValueError(
            f"{shape_name} must have shape (..., 3) or (..., 4), got {tuple(color.shape)}"
        )
    color = color.to(device=device)
    if color.dtype == torch.uint8:
        color = color.to(torch.float32) / 255.0
    else:
        color = color.to(torch.float32)
    if color.shape[-1] == 3:
        alpha = torch.ones(*color.shape[:-1], 1, device=device, dtype=torch.float32)
        color = torch.cat([color, alpha], dim=-1)
    return color.contiguous().clamp(0.0, 1.0)


def _uniform_color_tensor(
    surface_color: torch.Tensor | None,
    *,
    device,
) -> torch.Tensor:
    if surface_color is None:
        color = torch.tensor([[1.0, 1.0, 1.0, 1.0]], device=device)
    else:
        color = torch.as_tensor(surface_color, device=device)
        if color.shape not in ((3,), (4,)):
            raise ValueError(
                f"surface_color must have shape (3,) or (4,), got {tuple(color.shape)}"
            )
        if color.dtype == torch.uint8:
            color = color.to(torch.float32) / 255.0
        else:
            color = color.to(torch.float32)
        if color.shape == (3,):
            color = torch.cat([color, torch.ones(1, device=device)])
        color = color.reshape(1, 4)
    return color.contiguous().clamp(0.0, 1.0)


def _light_tensor(light_direction: torch.Tensor | None, *, device) -> torch.Tensor:
    if light_direction is None:
        light_direction = torch.tensor([-0.45, 0.75, -1.0], device=device)
    light_direction = _as_vec3(
        light_direction, name="light_direction", device=device
    ).reshape(1, 3)
    return _normalize_torch(light_direction).contiguous()


@wp.func
def _normalize_vec3(vector: wp.vec3) -> wp.vec3:
    length = wp.length(vector)
    if length <= 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)
    return vector / length


@wp.func
def _clamp_int(value: int, lo: int, hi: int) -> int:
    return wp.min(wp.max(value, lo), hi)


@wp.func
def _make_ray_direction(
    tid: int,
    width: int,
    height: int,
    camera: wp.array(dtype=wp.vec3),
    tan_half_fov: wp.float32,
    aspect: wp.float32,
) -> wp.vec3:
    y = tid / width
    x = tid - y * width
    px = ((wp.float32(x) + 0.5) / wp.float32(width)) * 2.0 - 1.0
    py = 1.0 - (((wp.float32(y) + 0.5) / wp.float32(height)) * 2.0)
    px = px * tan_half_fov * aspect
    py = py * tan_half_fov
    return _normalize_vec3(camera[1] + px * camera[2] + py * camera[3])


@wp.func
def _sample_field_trilinear(
    field: wp.array3d(dtype=wp.float32),
    point: wp.vec3,
    bounds_min: wp.vec3,
    bounds_max: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
) -> wp.float32:
    sx = (
        (point[0] - bounds_min[0])
        / (bounds_max[0] - bounds_min[0])
        * wp.float32(nx - 1)
    )
    sy = (
        (point[1] - bounds_min[1])
        / (bounds_max[1] - bounds_min[1])
        * wp.float32(ny - 1)
    )
    sz = (
        (point[2] - bounds_min[2])
        / (bounds_max[2] - bounds_min[2])
        * wp.float32(nz - 1)
    )

    i0 = _clamp_int(int(wp.floor(sx)), 0, nx - 2)
    j0 = _clamp_int(int(wp.floor(sy)), 0, ny - 2)
    k0 = _clamp_int(int(wp.floor(sz)), 0, nz - 2)
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    fx = wp.min(wp.max(sx - wp.float32(i0), 0.0), 1.0)
    fy = wp.min(wp.max(sy - wp.float32(j0), 0.0), 1.0)
    fz = wp.min(wp.max(sz - wp.float32(k0), 0.0), 1.0)

    c000 = field[i0, j0, k0]
    c100 = field[i1, j0, k0]
    c010 = field[i0, j1, k0]
    c110 = field[i1, j1, k0]
    c001 = field[i0, j0, k1]
    c101 = field[i1, j0, k1]
    c011 = field[i0, j1, k1]
    c111 = field[i1, j1, k1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


@wp.func
def _sample_color_trilinear(
    colors: wp.array4d(dtype=wp.float32),
    point: wp.vec3,
    bounds_min: wp.vec3,
    bounds_max: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
) -> wp.vec4:
    sx = (
        (point[0] - bounds_min[0])
        / (bounds_max[0] - bounds_min[0])
        * wp.float32(nx - 1)
    )
    sy = (
        (point[1] - bounds_min[1])
        / (bounds_max[1] - bounds_min[1])
        * wp.float32(ny - 1)
    )
    sz = (
        (point[2] - bounds_min[2])
        / (bounds_max[2] - bounds_min[2])
        * wp.float32(nz - 1)
    )

    i0 = _clamp_int(int(wp.floor(sx)), 0, nx - 2)
    j0 = _clamp_int(int(wp.floor(sy)), 0, ny - 2)
    k0 = _clamp_int(int(wp.floor(sz)), 0, nz - 2)
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    fx = wp.min(wp.max(sx - wp.float32(i0), 0.0), 1.0)
    fy = wp.min(wp.max(sy - wp.float32(j0), 0.0), 1.0)
    fz = wp.min(wp.max(sz - wp.float32(k0), 0.0), 1.0)

    out = wp.vec4(0.0, 0.0, 0.0, 0.0)
    for channel in range(4):
        c000 = colors[i0, j0, k0, channel]
        c100 = colors[i1, j0, k0, channel]
        c010 = colors[i0, j1, k0, channel]
        c110 = colors[i1, j1, k0, channel]
        c001 = colors[i0, j0, k1, channel]
        c101 = colors[i1, j0, k1, channel]
        c011 = colors[i0, j1, k1, channel]
        c111 = colors[i1, j1, k1, channel]

        c00 = c000 * (1.0 - fx) + c100 * fx
        c10 = c010 * (1.0 - fx) + c110 * fx
        c01 = c001 * (1.0 - fx) + c101 * fx
        c11 = c011 * (1.0 - fx) + c111 * fx
        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy
        out[channel] = c0 * (1.0 - fz) + c1 * fz
    return out


@wp.func
def _field_gradient(
    field: wp.array3d(dtype=wp.float32),
    point: wp.vec3,
    bounds_min: wp.vec3,
    bounds_max: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
) -> wp.vec3:
    dx = (bounds_max[0] - bounds_min[0]) / wp.float32(nx - 1)
    dy = (bounds_max[1] - bounds_min[1]) / wp.float32(ny - 1)
    dz = (bounds_max[2] - bounds_min[2]) / wp.float32(nz - 1)
    gx = (
        _sample_field_trilinear(
            field,
            point + wp.vec3(0.5 * dx, 0.0, 0.0),
            bounds_min,
            bounds_max,
            nx,
            ny,
            nz,
        )
        - _sample_field_trilinear(
            field,
            point - wp.vec3(0.5 * dx, 0.0, 0.0),
            bounds_min,
            bounds_max,
            nx,
            ny,
            nz,
        )
    ) / dx
    gy = (
        _sample_field_trilinear(
            field,
            point + wp.vec3(0.0, 0.5 * dy, 0.0),
            bounds_min,
            bounds_max,
            nx,
            ny,
            nz,
        )
        - _sample_field_trilinear(
            field,
            point - wp.vec3(0.0, 0.5 * dy, 0.0),
            bounds_min,
            bounds_max,
            nx,
            ny,
            nz,
        )
    ) / dy
    gz = (
        _sample_field_trilinear(
            field,
            point + wp.vec3(0.0, 0.0, 0.5 * dz),
            bounds_min,
            bounds_max,
            nx,
            ny,
            nz,
        )
        - _sample_field_trilinear(
            field,
            point - wp.vec3(0.0, 0.0, 0.5 * dz),
            bounds_min,
            bounds_max,
            nx,
            ny,
            nz,
        )
    ) / dz
    return wp.vec3(gx, gy, gz)


@wp.func
def _axis_intersection(
    origin: wp.float32,
    direction: wp.float32,
    lo: wp.float32,
    hi: wp.float32,
) -> wp.vec3:
    if wp.abs(direction) < 1.0e-12:
        if origin < lo or origin > hi:
            return wp.vec3(1.0, 0.0, 0.0)
        return wp.vec3(0.0, -3.402823e38, 3.402823e38)

    inv_d = 1.0 / direction
    t0 = (lo - origin) * inv_d
    t1 = (hi - origin) * inv_d
    return wp.vec3(0.0, wp.min(t0, t1), wp.max(t0, t1))


@wp.func
def _ray_box_intersection(
    origin: wp.vec3,
    direction: wp.vec3,
    bounds_min: wp.vec3,
    bounds_max: wp.vec3,
) -> wp.vec3:
    x = _axis_intersection(origin[0], direction[0], bounds_min[0], bounds_max[0])
    y = _axis_intersection(origin[1], direction[1], bounds_min[1], bounds_max[1])
    z = _axis_intersection(origin[2], direction[2], bounds_min[2], bounds_max[2])

    miss = x[0] + y[0] + z[0]
    t_near = wp.max(0.0, wp.max(x[1], wp.max(y[1], z[1])))
    t_far = wp.min(x[2], wp.min(y[2], z[2]))
    if miss > 0.0 or t_far < t_near:
        return wp.vec3(0.0, 0.0, -1.0)
    return wp.vec3(1.0, t_near, t_far)


@wp.func
def _shade(
    color: wp.vec4,
    normal: wp.vec3,
    light_direction: wp.vec3,
    ambient: wp.float32,
) -> wp.vec4:
    diffuse = wp.max(wp.dot(normal, light_direction), 0.0)
    intensity = ambient + (1.0 - ambient) * diffuse
    return wp.vec4(
        color[0] * intensity,
        color[1] * intensity,
        color[2] * intensity,
        color[3],
    )


@wp.func
def _jet_colormap(value: wp.float32) -> wp.vec3:
    r = wp.min(4.0 * value - 1.5, -4.0 * value + 4.5)
    g = wp.min(4.0 * value - 0.5, -4.0 * value + 3.5)
    b = wp.min(4.0 * value + 0.5, -4.0 * value + 2.5)
    return wp.vec3(
        wp.min(wp.max(r, 0.0), 1.0),
        wp.min(wp.max(g, 0.0), 1.0),
        wp.min(wp.max(b, 0.0), 1.0),
    )


@wp.func
def _sample_seed_trilinear(
    seed: wp.array3d(dtype=wp.float32),
    pos: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
) -> wp.float32:
    i0 = _clamp_int(int(wp.floor(pos[0])), 0, nx - 1)
    j0 = _clamp_int(int(wp.floor(pos[1])), 0, ny - 1)
    k0 = _clamp_int(int(wp.floor(pos[2])), 0, nz - 1)
    i1 = _clamp_int(i0 + 1, 0, nx - 1)
    j1 = _clamp_int(j0 + 1, 0, ny - 1)
    k1 = _clamp_int(k0 + 1, 0, nz - 1)

    fx = wp.min(wp.max(pos[0] - wp.float32(i0), 0.0), 1.0)
    fy = wp.min(wp.max(pos[1] - wp.float32(j0), 0.0), 1.0)
    fz = wp.min(wp.max(pos[2] - wp.float32(k0), 0.0), 1.0)

    c000 = seed[i0, j0, k0]
    c100 = seed[i1, j0, k0]
    c010 = seed[i0, j1, k0]
    c110 = seed[i1, j1, k0]
    c001 = seed[i0, j0, k1]
    c101 = seed[i1, j0, k1]
    c011 = seed[i0, j1, k1]
    c111 = seed[i1, j1, k1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


@wp.func
def _sample_vector_trilinear(
    vector_field: wp.array4d(dtype=wp.float32),
    pos: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
) -> wp.vec3:
    i0 = _clamp_int(int(wp.floor(pos[0])), 0, nx - 1)
    j0 = _clamp_int(int(wp.floor(pos[1])), 0, ny - 1)
    k0 = _clamp_int(int(wp.floor(pos[2])), 0, nz - 1)
    i1 = _clamp_int(i0 + 1, 0, nx - 1)
    j1 = _clamp_int(j0 + 1, 0, ny - 1)
    k1 = _clamp_int(k0 + 1, 0, nz - 1)

    fx = wp.min(wp.max(pos[0] - wp.float32(i0), 0.0), 1.0)
    fy = wp.min(wp.max(pos[1] - wp.float32(j0), 0.0), 1.0)
    fz = wp.min(wp.max(pos[2] - wp.float32(k0), 0.0), 1.0)

    result = wp.vec3(0.0, 0.0, 0.0)
    for channel in range(3):
        c000 = vector_field[i0, j0, k0, channel]
        c100 = vector_field[i1, j0, k0, channel]
        c010 = vector_field[i0, j1, k0, channel]
        c110 = vector_field[i1, j1, k0, channel]
        c001 = vector_field[i0, j0, k1, channel]
        c101 = vector_field[i1, j0, k1, channel]
        c011 = vector_field[i0, j1, k1, channel]
        c111 = vector_field[i1, j1, k1, channel]

        c00 = c000 * (1.0 - fx) + c100 * fx
        c10 = c010 * (1.0 - fx) + c110 * fx
        c01 = c001 * (1.0 - fx) + c101 * fx
        c11 = c011 * (1.0 - fx) + c111 * fx
        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy
        result[channel] = c0 * (1.0 - fz) + c1 * fz
    return result


@wp.func
def _project_point(
    point: wp.vec3,
    camera: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    tan_half_fov: wp.float32,
    aspect: wp.float32,
) -> wp.vec4:
    rel = point - camera[0]
    z = wp.dot(rel, camera[1])
    x = wp.dot(rel, camera[2])
    y = wp.dot(rel, camera[3])
    if z <= 1.0e-12:
        return wp.vec4(0.0, 0.0, z, 0.0)
    screen_x = (x / (z * tan_half_fov * aspect) + 1.0) * 0.5 * wp.float32(width)
    screen_y = (1.0 - (y / (z * tan_half_fov) + 1.0) * 0.5) * wp.float32(height)
    return wp.vec4(screen_x, screen_y, z, 1.0)


def _validate_image_shape(image_height: int, image_width: int) -> None:
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be strictly positive")


def _validate_fov(fov_y_degrees: float) -> None:
    if fov_y_degrees <= 0.0 or fov_y_degrees >= 180.0:
        raise ValueError("fov_y_degrees must lie in the open interval (0, 180)")


def _validate_ambient(ambient: float) -> None:
    if ambient < 0.0 or ambient > 1.0:
        raise ValueError("ambient must lie in the closed interval [0, 1]")


def _empty_render_outputs(
    image_height: int,
    image_width: int,
    *,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgba = torch.empty(
        (image_height, image_width, 4), device=device, dtype=torch.float32
    )
    depth = torch.empty((image_height, image_width), device=device, dtype=torch.float32)
    normal = torch.empty(
        (image_height, image_width, 3), device=device, dtype=torch.float32
    )
    return rgba, depth, normal


def _empty_image_outputs(
    image_height: int,
    image_width: int,
    *,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rgba = torch.zeros(
        (image_height, image_width, 4), device=device, dtype=torch.float32
    )
    depth = torch.full(
        (image_height, image_width), 3.402823e38, device=device, dtype=torch.float32
    )
    return rgba, depth


def _validate_transfer_range(vmin: float, vmax: float) -> None:
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        raise ValueError("vmax must be greater than vmin")


def _validate_opacity(value: float, *, name: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must lie in the closed interval [0, 1]")


def _validate_clip_range(near: float, far: float) -> None:
    if not math.isfinite(near) or not math.isfinite(far) or near <= 0.0 or far <= near:
        raise ValueError("near and far must satisfy 0 < near < far")


def _validate_vector_field(vector_field: torch.Tensor) -> None:
    if vector_field.ndim != 4 or vector_field.shape[-1] != 3:
        raise ValueError(
            "vector_field must have shape (nx, ny, nz, 3), got "
            f"{tuple(vector_field.shape)}"
        )
    if any(size < 2 for size in vector_field.shape[:3]):
        raise ValueError("vector_field must have at least two samples per dimension")


def _normalize_rgba_volume(rgba_volume: torch.Tensor) -> torch.Tensor:
    if rgba_volume.ndim != 4 or rgba_volume.shape[-1] != 4:
        raise ValueError(
            "rgba_volume must have shape (nx, ny, nz, 4), got "
            f"{tuple(rgba_volume.shape)}"
        )
    if any(size < 2 for size in rgba_volume.shape[:3]):
        raise ValueError("rgba_volume must have at least two samples per dimension")
    if rgba_volume.dtype == torch.uint8:
        rgba_volume = rgba_volume.to(torch.float32) / 255.0
    else:
        rgba_volume = rgba_volume.to(torch.float32)
    return rgba_volume.contiguous().clamp(0.0, 1.0)
