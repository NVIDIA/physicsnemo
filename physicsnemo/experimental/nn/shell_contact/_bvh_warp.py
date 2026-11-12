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

"""Discrete, two-pass BVH candidate enumeration; no floating-point atomics."""

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec


@wp.kernel
def _count(
    tree: wp.uint64,
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
    faces: wp.array2d(dtype=wp.int64),
    batch: wp.array(dtype=wp.int64),
    counts: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    query = wp.bvh_query_aabb(tree, lower[i], upper[i])
    face = int(0)
    count = wp.int64(0)
    while wp.bvh_query_next(query, face):
        incident = bool(False)
        for k in range(faces.shape[1]):
            if wp.int64(i) == faces[face, k]:
                incident = True
        if not incident and batch[i] == batch[faces[face, 0]]:
            count += wp.int64(1)
    counts[i] = count


@wp.kernel
def _fill(
    tree: wp.uint64,
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
    faces: wp.array2d(dtype=wp.int64),
    batch: wp.array(dtype=wp.int64),
    offsets: wp.array(dtype=wp.int64),
    pairs: wp.array2d(dtype=wp.int64),
    first_query: int,
    pair_start: wp.int64,
    pair_end: wp.int64,
):
    i = wp.tid() + first_query
    query = wp.bvh_query_aabb(tree, lower[i], upper[i])
    face = int(0)
    slot = offsets[i]
    while wp.bvh_query_next(query, face):
        incident = bool(False)
        for k in range(faces.shape[1]):
            if wp.int64(i) == faces[face, k]:
                incident = True
        if not incident and batch[i] == batch[faces[face, 0]]:
            if slot >= pair_start and slot < pair_end:
                pairs[0, slot - pair_start] = wp.int64(i)
                pairs[1, slot - pair_start] = wp.int64(face)
            slot += wp.int64(1)


def bvh_pair_chunks(
    lower,
    upper,
    query_lower,
    query_upper,
    faces,
    batch,
    pair_chunk_size,
    max_pairs=None,
):
    """Enumerate every BVH pair with bounded output storage, including dense nodes.

    Count once, then page the exact enumeration by global pair offset. A single
    query may span pages; no per-node top-k or pair truncation is applied. Only
    queries intersecting a page are revisited. Only O(number of pages) query
    ranges cross to the CPU, not the O(N) prefix vector or O(NF) candidate list.
    GPU searchsorted uses the same left/right boundary rules as the original
    host bisect implementation, including empty queries and dense single nodes.
    """
    wp.init()
    # Warp work runs on Torch's current stream of the geometry's device; the
    # scope orders Warp's stream-ordered cleanup after every launch.
    device, stream = FunctionSpec.warp_launch_context(lower)
    with FunctionSpec.warp_stream_scope(stream):
        wl = wp.from_torch(lower.contiguous(), dtype=wp.vec3)
        wu = wp.from_torch(upper.contiguous(), dtype=wp.vec3)
        tree = wp.Bvh(wl, wu)
        ql = wp.from_torch(query_lower.contiguous(), dtype=wp.vec3)
        qu = wp.from_torch(query_upper.contiguous(), dtype=wp.vec3)
        wf, wb = wp.from_torch(faces.contiguous()), wp.from_torch(batch.contiguous())
        counts = torch.empty(len(query_lower), dtype=torch.long, device=lower.device)
        args = [tree.id, ql, qu, wf, wb]
        wp.launch(
            _count,
            dim=len(counts),
            inputs=args + [wp.from_torch(counts)],
            device=device,
            stream=stream,
        )
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        total = int(offsets[-1].item())
        if max_pairs is not None and total > max_pairs:
            raise RuntimeError(
                f"surface candidate budget exceeded ({total}>{max_pairs}); no pairs truncated"
            )
        starts = torch.arange(0, total, pair_chunk_size, device=offsets.device)
        stops = (starts + pair_chunk_size).clamp_max(total)
        first_queries = torch.searchsorted(offsets, starts, right=True) - 1
        last_queries = torch.searchsorted(offsets, stops, right=False)
        ranges = torch.stack((first_queries, last_queries), -1).cpu().tolist()
    for page, (first, last) in enumerate(ranges):
        start = page * pair_chunk_size
        stop = min(start + pair_chunk_size, total)
        # Never keep a Warp stream scope active while yielding to the caller
        # (which may launch other work or abandon the iterator).
        with FunctionSpec.warp_stream_scope(stream):
            pairs = torch.empty(
                (2, stop - start), device=lower.device, dtype=torch.long
            )
            wp.launch(
                _fill,
                dim=last - first,
                inputs=args
                + [wp.from_torch(offsets), wp.from_torch(pairs), first, start, stop],
                device=device,
                stream=stream,
            )
        # CPU calls are synchronous; CUDA consumers use the same Torch stream.
        yield pairs
