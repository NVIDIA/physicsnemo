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

"""Experimental compact-cell-points CUDA radius-search backend.

This variant targets the static training path used by GeoTransolver:
``max_points`` is set, ``return_points=True`` is common, and first-found early
exit is acceptable.
It does the following:
    1) map points to radius sized spatial cells
    2) inserts occupied cells into a hash table
    3) prefix sums cell count scatters points in each cell into a compact contiguous array
    4) performs neighbor search using a warp per query point

"""

from __future__ import annotations

import importlib
from functools import lru_cache
from importlib.resources import files

import torch

from physicsnemo.core.version_check import check_version_spec

from .utils import format_returns, validate_inputs

CUPY_AVAILABLE = check_version_spec("cupy", "13.6.0", hard_fail=False)

if CUPY_AVAILABLE:
    cp = importlib.import_module("cupy")
else:
    cp = None


@lru_cache(maxsize=1)
def _load_compact_cell_points_kernel_source() -> str:
    """Load the package-shipped CUDA source for the compact-cell backend."""
    return (
        files(__package__)
        .joinpath("kernels")
        .joinpath("compact_cell_points.cu")
        .read_text(encoding="utf-8")
    )


@lru_cache(maxsize=1)
def _get_compact_cell_points_kernels():
    if cp is None:
        raise ImportError(
            "physicsnemo radius_search compact_cell_points implementation requires "
            "cupy>=13.6.0"
        )
    module = cp.RawModule(
        code=_load_compact_cell_points_kernel_source(),
        options=("--std=c++17",),
        name_expressions=(
            "count_point_cells",
            "scatter_point_bins",
            "radius_search",
        ),
    )
    return (
        module.get_function("count_point_cells"),
        module.get_function("scatter_point_bins"),
        module.get_function("radius_search"),
    )


def _require_cupy() -> None:
    if cp is None:
        raise ImportError(
            "physicsnemo radius_search compact_cell_points implementation requires "
            "cupy>=13.6.0"
        )


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _as_cupy(tensor: torch.Tensor):
    """Create a zero-copy CuPy view of a CUDA torch tensor."""
    if cp is None:
        raise RuntimeError("CuPy >= 13.6.0 is required for this backend")
    return cp.from_dlpack(tensor.detach().contiguous())


def _gather_points(
    points: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    was_unbatched: bool,
) -> torch.Tensor:
    batch_size, _, point_dim = points.shape
    _, num_queries, max_points = indices.shape
    if points.shape[1] == 0:
        output = torch.zeros(
            batch_size,
            num_queries,
            max_points,
            point_dim,
            device=points.device,
            dtype=points.dtype,
        )
        return output.squeeze(0) if was_unbatched else output

    batch = torch.arange(batch_size, device=points.device)[:, None, None]
    gathered = points[batch, indices.to(torch.long)]
    valid = (
        torch.arange(max_points, device=points.device)[None, None, :]
        < counts[:, :, None]
    )
    gathered = torch.where(valid[..., None], gathered, torch.zeros_like(gathered))
    return gathered.squeeze(0) if was_unbatched else gathered


def radius_search_impl(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Static first-found radius search using contiguous GPU cell bins."""

    _require_cupy()

    if points.device != queries.device:
        raise ValueError("points and queries must be on the same device")
    if points.device.type != "cuda":
        raise ValueError(
            "compact_cell_points radius_search implementation requires CUDA input"
        )
    if max_points is None:
        raise ValueError(
            "compact_cell_points radius_search implementation requires max_points"
        )
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    input_points = points
    input_queries = queries
    use_compact_query_order = (
        input_points.shape == input_queries.shape
        and input_points.stride() == input_queries.stride()
        and input_points.data_ptr() == input_queries.data_ptr()
    )
    points, queries, was_unbatched = validate_inputs(points, queries)
    if points.shape[-1] != 3 or queries.shape[-1] != 3:
        raise ValueError(
            "compact_cell_points radius_search implementation requires 3D points"
        )

    points_kernel = (
        points.to(torch.float32) if points.dtype != torch.float32 else points
    )
    queries_kernel = (
        queries.to(torch.float32) if queries.dtype != torch.float32 else queries
    )
    points_kernel = points_kernel.contiguous()
    queries_kernel = queries_kernel.contiguous()

    batch_size, num_points, _ = points_kernel.shape
    _, num_queries, _ = queries_kernel.shape
    total_points = batch_size * num_points
    total_queries = batch_size * num_queries

    indices = torch.zeros(
        (batch_size, num_queries, max_points),
        dtype=torch.int32,
        device=points.device,
    )
    counts = torch.zeros(
        (batch_size, num_queries), dtype=torch.int32, device=points.device
    )
    distance_workspace = torch.zeros(
        (batch_size, num_queries, max_points),
        dtype=torch.float32,
        device=points.device,
    )
    distances = (
        distance_workspace
        if return_dists
        else torch.empty(0, dtype=torch.float32, device=points.device)
    )

    use_gather_for_points = return_points and input_points.requires_grad
    write_points_in_kernel = return_points and not use_gather_for_points
    if write_points_in_kernel:
        points_out = torch.zeros(
            (batch_size, num_queries, max_points, 3),
            dtype=torch.float32,
            device=points.device,
        )
    else:
        points_out = torch.empty(
            0,
            max_points,
            3,
            dtype=torch.float32,
            device=points.device,
        )

    if total_queries > 0 and total_points > 0:
        flat_points = points_kernel.reshape(total_points, 3)
        flat_queries = queries_kernel.reshape(total_queries, 3)

        table_capacity = _next_power_of_two(max(2, total_points * 2))
        table_states = torch.zeros(
            table_capacity, dtype=torch.int32, device=points.device
        )
        table_cell_coords = torch.empty(
            table_capacity, 3, dtype=torch.int64, device=points.device
        )
        table_batches = torch.full(
            (table_capacity,),
            torch.iinfo(torch.int64).min,
            dtype=torch.int64,
            device=points.device,
        )
        cell_counts = torch.zeros(
            table_capacity, dtype=torch.int32, device=points.device
        )
        point_cell_slots = torch.empty(
            total_points, dtype=torch.int32, device=points.device
        )
        cell_write_offsets = torch.zeros_like(cell_counts)
        compact_points = torch.empty(
            total_points, 4, dtype=torch.float32, device=points.device
        )

        device_index = points.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        count_kernel, scatter_kernel, search_kernel = _get_compact_cell_points_kernels()
        current_stream = torch.cuda.current_stream(device_index).cuda_stream
        cupy_stream = cp.cuda.ExternalStream(current_stream)

        with cp.cuda.Device(device_index), cupy_stream:
            threads = 256
            point_blocks = (total_points + threads - 1) // threads
            count_kernel(
                (point_blocks,),
                (threads,),
                (
                    _as_cupy(flat_points),
                    _as_cupy(table_states),
                    _as_cupy(table_cell_coords),
                    _as_cupy(table_batches),
                    _as_cupy(cell_counts),
                    _as_cupy(point_cell_slots),
                    cp.int32(total_points),
                    cp.int32(num_points),
                    cp.int32(table_capacity),
                    cp.float32(radius),
                ),
            )

        cell_prefix = torch.cumsum(cell_counts, dim=0, dtype=torch.int32)
        cell_offsets = torch.empty_like(cell_counts)
        cell_offsets[0] = 0
        if table_capacity > 1:
            cell_offsets[1:] = cell_prefix[:-1]

        with cp.cuda.Device(device_index), cupy_stream:
            scatter_kernel(
                (point_blocks,),
                (threads,),
                (
                    _as_cupy(flat_points),
                    _as_cupy(point_cell_slots),
                    _as_cupy(cell_offsets),
                    _as_cupy(cell_write_offsets),
                    _as_cupy(compact_points),
                    cp.int32(total_points),
                ),
            )

            warps_per_block = 8
            query_blocks = (total_queries + warps_per_block - 1) // warps_per_block
            search_kernel(
                (query_blocks,),
                (threads,),
                (
                    _as_cupy(flat_points),
                    _as_cupy(compact_points),
                    _as_cupy(flat_queries),
                    _as_cupy(table_states),
                    _as_cupy(table_cell_coords),
                    _as_cupy(table_batches),
                    _as_cupy(cell_offsets),
                    _as_cupy(cell_counts),
                    _as_cupy(indices.reshape(total_queries, max_points)),
                    _as_cupy(points_out.reshape(-1, 3)),
                    _as_cupy(distance_workspace.reshape(total_queries, max_points)),
                    _as_cupy(counts.reshape(total_queries)),
                    cp.int32(total_queries),
                    cp.int32(num_queries),
                    cp.int32(num_points),
                    cp.int32(max_points),
                    cp.int32(table_capacity),
                    cp.int32(use_compact_query_order),
                    cp.float32(radius),
                    cp.float32(radius * radius),
                    cp.int32(write_points_in_kernel),
                    cp.int32(return_dists),
                ),
            )

    if return_points:
        if use_gather_for_points:
            original_points, _, _ = validate_inputs(input_points, input_queries)
            points_out = _gather_points(original_points, indices, counts, was_unbatched)
        elif points_out.dtype != points.dtype:
            points_out = points_out.to(points.dtype)
    else:
        points_out = torch.empty(
            0, max_points, 3, device=points.device, dtype=points.dtype
        )

    if was_unbatched:
        indices = indices.squeeze(0)
        if return_points and points_out.ndim == 4:
            points_out = points_out.squeeze(0)
        if return_dists:
            distances = distances.squeeze(0)

    return indices, points_out, distances


@torch.compiler.disable
def radius_search(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
):
    """compact_cell_points backend entry point for radius search with formatted returns."""
    indices, points_out, distances = radius_search_impl(
        points,
        queries,
        radius,
        max_points,
        return_dists,
        return_points,
    )
    return format_returns(indices, points_out, distances, return_dists, return_points)
