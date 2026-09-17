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

"""
This file contains the warp kernel for farthest-point sampling.

It should be pure warp code, no pytorch here.
"""

import warp as wp


@wp.kernel
def fps_fused(
    points: wp.array3d(dtype=wp.float32),  # (B, N, D)
    start: wp.array(dtype=wp.int32),  # (B,)
    selected: wp.array2d(dtype=wp.int32),  # (B, K) output
    min_dist: wp.array2d(dtype=wp.float32),  # (B, N) scratch, init large
    num_points: wp.int32,
    num_samples: wp.int32,
    dim: wp.int32,
    block_size: wp.int32,
):
    """Single-launch farthest-point sampling, one block per point cloud.

    Launched with ``dim=(B, block_size)`` and ``block_dim=block_size``: the
    ``block_size`` lanes of block ``b`` cooperate on cloud ``b``. Each lane
    owns a strided slice of the ``num_points`` points (``t, t+block_size,
    ...``) so the running ``min_dist`` is partitioned across lanes with no
    cross-lane aliasing. The ``num_samples`` selection steps run inside this
    one kernel; each step reduces the per-lane best to the global farthest
    point via a block ``tile_max`` (value) + ``tile_min`` (tie-broken index),
    whose result broadcasts to every lane.
    """
    b, t = wp.tid()

    cur = start[b]
    if t == 0:
        selected[b, 0] = cur

    for s in range(1, num_samples):
        # Each lane updates its strided points against the latest centroid and
        # tracks its local farthest (max running min-distance).
        local_best = float(-1.0e30)
        local_idx = num_points  # sentinel: loses the argmin tie-break
        p = t
        while p < num_points:
            d = float(0.0)
            for k in range(dim):
                diff = points[b, p, k] - points[b, cur, k]
                d += diff * diff
            md = wp.min(min_dist[b, p], d)
            min_dist[b, p] = md
            if md > local_best:
                local_best = md
                local_idx = p
            p += block_size

        # Block-wide argmax: the max running min-distance, then the smallest
        # index achieving it (matches torch.argmax's first-occurrence rule).
        max_tile = wp.tile_max(wp.tile(local_best))
        max_dist = max_tile[0]
        cand = num_points
        if local_best >= max_dist:
            cand = local_idx
        idx_tile = wp.tile_min(wp.tile(cand))
        cur = idx_tile[0]

        if t == 0:
            selected[b, s] = cur
