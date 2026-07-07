// SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

namespace {

struct __align__(16) CompactPoint {
    float x;
    float y;
    float z;
    int point_id;
};

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
void count_point_cells(
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
void scatter_point_bins(
    const float* __restrict__ points,
    const int* __restrict__ point_cell_slots,
    const int* __restrict__ cell_offsets,
    int* __restrict__ cell_write_offsets,
    CompactPoint* __restrict__ compact_points,
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
    const int compact_offset = cell_offsets[slot] + local_offset;
    CompactPoint compact_point;
    compact_point.x = points[point_id * 3 + 0];
    compact_point.y = points[point_id * 3 + 1];
    compact_point.z = points[point_id * 3 + 2];
    compact_point.point_id = point_id;
    compact_points[compact_offset] = compact_point;
}

extern "C" __global__
void radius_search(
    const float* __restrict__ points,
    const CompactPoint* __restrict__ compact_points,
    const float* __restrict__ queries,
    const int* __restrict__ table_states,
    const long long* __restrict__ table_cell_coords,
    const long long* __restrict__ table_batches,
    const int* __restrict__ cell_offsets,
    const int* __restrict__ cell_counts,
    int* __restrict__ indices,
    float* __restrict__ output_points,
    float* __restrict__ distances,
    int* __restrict__ counts,
    const int total_queries,
    const int queries_per_batch,
    const int num_points,
    const int max_points,
    const int table_capacity,
    const int use_compact_query_order,
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
    const int linear_query_id = blockIdx.x * warps_per_block + warp_id;
    if (linear_query_id >= total_queries) {
        return;
    }
    const int query_id = use_compact_query_order
        ? compact_points[linear_query_id].point_id
        : linear_query_id;

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

    // Resolve all 27 neighboring cells concurrently. Processing below remains
    // in the original dx/dy/dz order so first-found truncation is unchanged.
    int lane_table_slot = -1;
    if (lane < 27) {
        const int dx_cell = lane / 9 - 1;
        const int dy_cell = (lane / 3) % 3 - 1;
        const int dz_cell = lane % 3 - 1;
        lane_table_slot = table_lookup(
            qcx + dx_cell,
            qcy + dy_cell,
            qcz + dz_cell,
            query_batch,
            table_states,
            table_cell_coords,
            table_batches,
            table_capacity);
    }

    for (int cell_index = 0;
         cell_index < 27 && found < max_points;
         ++cell_index) {
        const int table_slot = __shfl_sync(
            full_mask, lane_table_slot, cell_index);
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
                const CompactPoint compact_point = compact_points[candidate_offset];
                point_id = compact_point.point_id;
                const float px = compact_point.x;
                const float py = compact_point.y;
                const float pz = compact_point.z;
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

    if (lane == 0) {
        counts[query_id] = found;
    }
}
