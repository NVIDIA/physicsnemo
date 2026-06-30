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

import torch
from torch.profiler import record_function

from physicsnemo.core.version_check import check_version_spec

from .utils import (
    format_returns,
    maybe_capture_radius_search,
    time_radius_search,
    validate_inputs,
)

CUPY_AVAILABLE = check_version_spec("cupy", "13.6.0", hard_fail=False)

if CUPY_AVAILABLE:
    cp = importlib.import_module("cupy")
else:
    cp = None


_RADIUS_SEARCH_V2_KERNEL = r"""
namespace {

__device__ __forceinline__ unsigned long long cell_hash(
    const long long cx,
    const long long cy,
    const long long cz,
    const long long batch
) {
    return static_cast<unsigned long long>(cx) * 73856093ULL
        ^ static_cast<unsigned long long>(cy) * 19349663ULL
        ^ static_cast<unsigned long long>(cz) * 83492791ULL
        ^ static_cast<unsigned long long>(batch) * 2654435761ULL;
}

__device__ __forceinline__ unsigned long long mix_hash(unsigned long long x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

__device__ __forceinline__ int cell_equal(
    const long long* __restrict__ table_cell_coords,
    const long long* __restrict__ table_batches,
    const int slot,
    const long long cx,
    const long long cy,
    const long long cz,
    const long long batch
) {
    const int coord_offset = slot * 3;
    return table_batches[slot] == batch
        && table_cell_coords[coord_offset + 0] == cx
        && table_cell_coords[coord_offset + 1] == cy
        && table_cell_coords[coord_offset + 2] == cz;
}

__device__ __forceinline__ int table_lookup(
    const long long cx,
    const long long cy,
    const long long cz,
    const long long batch,
    const int* __restrict__ table_states,
    const long long* __restrict__ table_cell_coords,
    const long long* __restrict__ table_batches,
    const int table_capacity
) {
    const int mask = table_capacity - 1;
    int slot = static_cast<int>(mix_hash(cell_hash(cx, cy, cz, batch)) & mask);

    for (int probe = 0; probe < table_capacity; ++probe) {
        int state = table_states[slot];
        if (state == 0) {
            return -1;
        }
        while (state == 1) {
            state = table_states[slot];
        }
        if (cell_equal(table_cell_coords, table_batches, slot,
                       cx, cy, cz, batch)) {
            return slot;
        }
        slot = (slot + 1) & mask;
    }

    return -1;
}

__device__ __forceinline__ int table_lookup_or_insert(
    const long long cx,
    const long long cy,
    const long long cz,
    const long long batch,
    int* __restrict__ table_states,
    long long* __restrict__ table_cell_coords,
    long long* __restrict__ table_batches,
    const int table_capacity
) {
    const int mask = table_capacity - 1;
    int slot = static_cast<int>(mix_hash(cell_hash(cx, cy, cz, batch)) & mask);

    for (int probe = 0; probe < table_capacity; ++probe) {
        int state = atomicCAS(&table_states[slot], 0, 1);
        if (state == 0) {
            const int coord_offset = slot * 3;
            table_cell_coords[coord_offset + 0] = cx;
            table_cell_coords[coord_offset + 1] = cy;
            table_cell_coords[coord_offset + 2] = cz;
            table_batches[slot] = batch;
            __threadfence();
            atomicExch(&table_states[slot], 2);
            return slot;
        }

        while (state == 1) {
            state = atomicAdd(&table_states[slot], 0);
        }
        __threadfence();
        if (cell_equal(table_cell_coords, table_batches, slot,
                       cx, cy, cz, batch)) {
            return slot;
        }

        slot = (slot + 1) & mask;
    }

    return -1;
}

}  // namespace

extern "C" __global__
void count_point_cells_v2(
    const float* __restrict__ points,
    int* __restrict__ table_states,
    long long* __restrict__ table_cell_coords,
    long long* __restrict__ table_batches,
    int* __restrict__ cell_counts,
    int* __restrict__ point_cell_slots,
    const int total_points,
    const int num_points,
    const int table_capacity,
    const float radius
) {
    const int point_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_id >= total_points) {
        return;
    }

    const long long batch = point_id / num_points;
    const float px = points[point_id * 3 + 0];
    const float py = points[point_id * 3 + 1];
    const float pz = points[point_id * 3 + 2];
    const double inv_radius = 1.0 / static_cast<double>(radius);
    const long long cx = static_cast<long long>(
        floor(static_cast<double>(px) * inv_radius));
    const long long cy = static_cast<long long>(
        floor(static_cast<double>(py) * inv_radius));
    const long long cz = static_cast<long long>(
        floor(static_cast<double>(pz) * inv_radius));

    const int slot = table_lookup_or_insert(
        cx, cy, cz, batch,
        table_states,
        table_cell_coords,
        table_batches,
        table_capacity);
    point_cell_slots[point_id] = slot;
    if (slot >= 0) {
        atomicAdd(&cell_counts[slot], 1);
    }
}

extern "C" __global__
void scatter_point_bins_v2(
    const int* __restrict__ point_cell_slots,
    const int* __restrict__ cell_offsets,
    int* __restrict__ cell_write_offsets,
    int* __restrict__ compact_point_ids,
    const int total_points
) {
    const int point_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_id >= total_points) {
        return;
    }

    const int slot = point_cell_slots[point_id];
    if (slot < 0) {
        return;
    }

    const int local_offset = atomicAdd(&cell_write_offsets[slot], 1);
    compact_point_ids[cell_offsets[slot] + local_offset] = point_id;
}

extern "C" __global__
void radius_search_v2(
    const float* __restrict__ points,
    const float* __restrict__ queries,
    const int* __restrict__ table_states,
    const long long* __restrict__ table_cell_coords,
    const long long* __restrict__ table_batches,
    const int* __restrict__ cell_offsets,
    const int* __restrict__ cell_counts,
    const int* __restrict__ compact_point_ids,
    int* __restrict__ indices,
    float* __restrict__ output_points,
    float* __restrict__ distances,
    int* __restrict__ counts,
    const int total_queries,
    const int queries_per_batch,
    const int num_points,
    const int max_points,
    const int table_capacity,
    const float radius,
    const float radius_sq,
    const int write_points,
    const int write_dists
) {
    constexpr int warp_size = 32;
    constexpr int warps_per_block = 8;
    constexpr unsigned int full_mask = 0xffffffffU;

    const int lane = threadIdx.x & (warp_size - 1);
    const int warp_id = threadIdx.x / warp_size;
    const int query_id = blockIdx.x * warps_per_block + warp_id;
    if (query_id >= total_queries) {
        return;
    }

    const float qx = queries[query_id * 3 + 0];
    const float qy = queries[query_id * 3 + 1];
    const float qz = queries[query_id * 3 + 2];
    const long long query_batch = query_id / queries_per_batch;
    const double inv_radius = 1.0 / static_cast<double>(radius);
    const long long qcx = static_cast<long long>(
        floor(static_cast<double>(qx) * inv_radius));
    const long long qcy = static_cast<long long>(
        floor(static_cast<double>(qy) * inv_radius));
    const long long qcz = static_cast<long long>(
        floor(static_cast<double>(qz) * inv_radius));

    int found = 0;
    const int out_base = query_id * max_points;

    for (int dx_cell = -1; dx_cell <= 1 && found < max_points; ++dx_cell) {
        for (int dy_cell = -1; dy_cell <= 1 && found < max_points; ++dy_cell) {
            for (int dz_cell = -1; dz_cell <= 1 && found < max_points; ++dz_cell) {
                int table_slot = -1;
                if (lane == 0) {
                    table_slot = table_lookup(
                        qcx + dx_cell,
                        qcy + dy_cell,
                        qcz + dz_cell,
                        query_batch,
                        table_states,
                        table_cell_coords,
                        table_batches,
                        table_capacity);
                }
                table_slot = __shfl_sync(full_mask, table_slot, 0);
                if (table_slot < 0) {
                    continue;
                }

                const int cell_start = cell_offsets[table_slot];
                const int cell_count = cell_counts[table_slot];
                const int cell_end = cell_start + cell_count;
                for (int candidate_base = cell_start;
                     candidate_base < cell_end && found < max_points;
                     candidate_base += warp_size) {
                    const int candidate_offset = candidate_base + lane;
                    int point_id = -1;
                    float dist_sq = 0.0f;
                    bool hit = false;
                    if (candidate_offset < cell_end) {
                        point_id = compact_point_ids[candidate_offset];
                        const float px = points[point_id * 3 + 0];
                        const float py = points[point_id * 3 + 1];
                        const float pz = points[point_id * 3 + 2];
                        const float xdiff = px - qx;
                        const float ydiff = py - qy;
                        const float zdiff = pz - qz;
                        dist_sq = xdiff * xdiff + ydiff * ydiff + zdiff * zdiff;
                        hit = dist_sq <= radius_sq;
                    }

                    const unsigned int hit_mask = __ballot_sync(full_mask, hit);
                    const int num_hits = __popc(hit_mask);
                    if (num_hits == 0) {
                        continue;
                    }

                    const int space = max_points - found;
                    const int write_hits = num_hits < space ? num_hits : space;
                    const unsigned int lower_lane_mask = (1U << lane) - 1U;
                    const int hit_rank = __popc(hit_mask & lower_lane_mask);
                    if (hit && hit_rank < write_hits) {
                        const int out_offset = out_base + found + hit_rank;
                        const int local_point = point_id % num_points;
                        indices[out_offset] = local_point;
                        if (write_points) {
                            output_points[out_offset * 3 + 0] = points[point_id * 3 + 0];
                            output_points[out_offset * 3 + 1] = points[point_id * 3 + 1];
                            output_points[out_offset * 3 + 2] = points[point_id * 3 + 2];
                        }
                        if (write_dists) {
                            distances[out_offset] = sqrtf(dist_sq);
                        }
                    }

                    found += write_hits;
                }
            }
        }
    }

    if (lane == 0) {
        counts[query_id] = found;
    }
}
"""


@lru_cache(maxsize=1)
def _get_radius_search_v2_kernels():
    if cp is None:
        raise ImportError(
            "physicsnemo radius_search compact_cell_points implementation requires "
            "cupy>=13.6.0"
        )
    module = cp.RawModule(
        code=_RADIUS_SEARCH_V2_KERNEL,
        options=("--std=c++17",),
        name_expressions=(
            "count_point_cells_v2",
            "scatter_point_bins_v2",
            "radius_search_v2",
        ),
    )
    return (
        module.get_function("count_point_cells_v2"),
        module.get_function("scatter_point_bins_v2"),
        module.get_function("radius_search_v2"),
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
    assert cp is not None  # noqa: S101
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
        raise ValueError("compact_cell_points radius_search implementation requires CUDA input")
    if max_points is None:
        raise ValueError("compact_cell_points radius_search implementation requires max_points")
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    input_points = points
    input_queries = queries
    points, queries, was_unbatched = validate_inputs(points, queries)
    if points.shape[-1] != 3 or queries.shape[-1] != 3:
        raise ValueError("compact_cell_points radius_search implementation requires 3D points")

    points_kernel = points.to(torch.float32) if points.dtype != torch.float32 else points
    queries_kernel = queries.to(torch.float32) if queries.dtype != torch.float32 else queries
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
        compact_point_ids = torch.empty(
            total_points, dtype=torch.int32, device=points.device
        )

        device_index = points.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        count_kernel, scatter_kernel, search_kernel = _get_radius_search_v2_kernels()
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
                    _as_cupy(point_cell_slots),
                    _as_cupy(cell_offsets),
                    _as_cupy(cell_write_offsets),
                    _as_cupy(compact_point_ids),
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
                    _as_cupy(flat_queries),
                    _as_cupy(table_states),
                    _as_cupy(table_cell_coords),
                    _as_cupy(table_batches),
                    _as_cupy(cell_offsets),
                    _as_cupy(cell_counts),
                    _as_cupy(compact_point_ids),
                    _as_cupy(indices.reshape(total_queries, max_points)),
                    _as_cupy(points_out.reshape(-1, 3)),
                    _as_cupy(distance_workspace.reshape(total_queries, max_points)),
                    _as_cupy(counts.reshape(total_queries)),
                    cp.int32(total_queries),
                    cp.int32(num_queries),
                    cp.int32(num_points),
                    cp.int32(max_points),
                    cp.int32(table_capacity),
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

    capture_counts = counts.squeeze(0) if was_unbatched else counts
    if was_unbatched:
        indices = indices.squeeze(0)
        if return_points and points_out.ndim == 4:
            points_out = points_out.squeeze(0)
        if return_dists:
            distances = distances.squeeze(0)

    maybe_capture_radius_search(
        "compact_cell_points",
        input_points,
        input_queries,
        radius,
        max_points,
        return_dists,
        return_points,
        num_neighbors=capture_counts,
    )
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
    with (
        record_function("physicsnemo::radius_search_compact_cell_points"),
        time_radius_search("compact_cell_points"),
    ):
        indices, points_out, distances = radius_search_impl(
            points,
            queries,
            radius,
            max_points,
            return_dists,
            return_points,
        )
        return format_returns(indices, points_out, distances, return_dists, return_points)
